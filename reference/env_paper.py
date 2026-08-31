"""Canonical rendezvous environment for the paper revision (rev2).

This is the single source of truth for the environment described in the
manuscript (Sections III-B, Tables I-III).  It replaces the legacy v5.py
environment for all new training, evaluation, baseline, and sensitivity
experiments, fixing the following defects of the legacy code:

  1. Collision penalties were dead code (reward assigned, then overwritten
     by the area-shaping branch).  Here collision penalties accumulate
     with the area term every step.  (The goal reward replaces the +20
     shaping bonus on the terminal step, exactly as in Algorithm 1.)
  2. The `valid_move` latch disabled robot-robot collision detection for
     all robots after the first conflict, and swap/pass-through conflicts
     were never detected.  Here conflicts are resolved with an iterative
     two-pass scheme (same-target, swap, occupied-cell).
  3. Unknown cells were indistinguishable from free cells (known_map
     initialised to 0).  Here known_map uses -1 = unknown, 0 = free,
     1 = obstacle, matching Table II.
  4. Episodes had no step limit.  Here episodes truncate at max_steps
     (reported via gymnasium's `truncated`, so SB3 bootstraps correctly).
  5. No seeding existed and the map generator mixed two RNG streams.
     Here a single numpy Generator drives map generation, spawning, and
     everything else; reset(seed=...) is fully reproducible.
  6. pygame rendering ran inside step() during training.  Here rendering
     only happens when render_mode="human" is requested.
  7. Training started at import time (no __main__ guard).  This module
     has no side effects on import.

Reward table (paper Table III):
    area decrease (below previous best)   +20
    area increase (above previous step)   -0.5
    obstacle collision (per robot)         -5
    robot-robot collision (per robot)      -5
    goal (bounding square <= threshold,
          square region obstacle-free)   +100  -> terminated

Bounding-area semantics: the paper's Table III and Section III text refer
to the *bounding square*; legacy v5.py also used the square.  We keep the
square: side = max(x-extent, y-extent), area = side**2, anchored at
(min_x, min_y).  NOTE: Eq. (5) in the current manuscript states the
rectangle product instead -- the equation should be corrected in the
revision (see README).

Goal check: evaluated every step (area <= threshold AND the anchored
square is inside the map and obstacle-free), independently of whether the
area is a new best.  Algorithm 1 in the manuscript nests the goal check
under the improvement branch; the revision should update that line.
"""

import copy
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

UNKNOWN, FREE, OBSTACLE = -1, 0, 1


@dataclass
class RewardConfig:
    """Table III values. Sensitivity analysis perturbs these one at a time.

    step_cost: added every step (0 disables). A small negative value makes
    dawdling costly so the learned policy is decisive rather than a
    high-entropy drift (the original Table III had no urgency term, which
    empirically produced near-uniform policies whose argmax fails).

    potential_coef: if nonzero, the area shaping term becomes true
    potential-based shaping  r_shape = potential_coef * (prev_area -
    new_area)  applied every step with both signs (Ng et al.), replacing
    the {area_decrease on new best / area_increase on growth} scheme.
    """
    area_decrease: float = 20.0
    area_increase: float = -0.5
    collide_obstacle: float = -5.0
    collide_robot: float = -5.0
    goal: float = 100.0
    step_cost: float = 0.0
    potential_coef: float = 0.0
    move_cost: float = 0.0        # per robot per realized move (energy term)


@dataclass
class EnvConfig:
    num_robots: int = 4
    rows: int = 20
    cols: int = 20
    lidar_radius: int = 1              # Chebyshev radius -> (2r+1)^2 window
    threshold_area: int = 16           # bounding-square area for success (4x4)
    max_steps: int = 300
    # map generator (ported from map_gen_v4, now seeded and bounded)
    num_clusters: int = 5
    cluster_size_range: Tuple[int, int] = (2, 10)
    min_cluster_distance: float = 3.0
    # dynamic obstacles (evaluation-time extension; 0 = fully static env).
    # Each dynamic obstacle spawns on a random static-free cell and performs
    # an unbiased 4-neighbour random walk over static-free cells, moving
    # AFTER the robots commit each step, never onto a robot or another
    # obstacle. Sensed only through the normal LiDAR reveal.
    num_dynamic_obstacles: int = 0
    dyn_move_prob: float = 1.0
    rewards: RewardConfig = field(default_factory=RewardConfig)


class MapGen:
    """Seeded port of map_gen_v4.generate_connected_clusters_map.

    Semantics preserved: cluster centres rejection-sampled in the interior
    with a minimum Euclidean distance between centres; clusters grown
    8-directionally, interior-only (the 1-cell border ring stays free);
    free space made a single 4-connected component by converting
    unreachable free pockets into obstacles.

    Added: every rejection/growth loop is bounded, and all randomness
    comes from the numpy Generator passed in.
    """

    @staticmethod
    def generate(rows: int, cols: int, num_clusters: int,
                 cluster_size_range: Tuple[int, int], min_distance: float,
                 rng: np.random.Generator) -> np.ndarray:
        grid = np.zeros((rows, cols), dtype=np.int8)

        # --- cluster centres ---
        centres: List[Tuple[int, int]] = []
        tries = 0
        while len(centres) < num_clusters and tries < 5000:
            tries += 1
            c = (int(rng.integers(1, rows - 1)), int(rng.integers(1, cols - 1)))
            if all(np.hypot(c[0] - o[0], c[1] - o[1]) >= min_distance for o in centres):
                centres.append(c)

        # --- grow clusters (8-directional, interior only) ---
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        for cx, cy in centres:
            size = int(rng.integers(cluster_size_range[0], cluster_size_range[1] + 1))
            cells = [(cx, cy)]
            grid[cx, cy] = OBSTACLE
            stuck = 0
            while len(cells) < size and stuck < 200:
                base = cells[int(rng.integers(0, len(cells)))]
                order = rng.permutation(len(directions))
                grew = False
                for k in order:
                    dx, dy = directions[k]
                    nx, ny = base[0] + dx, base[1] + dy
                    if 1 <= nx < rows - 1 and 1 <= ny < cols - 1 and grid[nx, ny] == FREE:
                        grid[nx, ny] = OBSTACLE
                        cells.append((nx, ny))
                        grew = True
                        break
                stuck = 0 if grew else stuck + 1

        # --- single 4-connected free component (fill unreachable pockets) ---
        free = np.argwhere(grid == FREE)
        if len(free) == 0:
            return grid
        start = tuple(free[0])          # (0,0) is always free (border ring)
        seen = np.zeros_like(grid, dtype=bool)
        stack = [start]
        seen[start] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not seen[nx, ny] \
                        and grid[nx, ny] == FREE:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
        grid[(grid == FREE) & (~seen)] = OBSTACLE
        return grid


class RendezvousEnv(gym.Env):
    """Centralised multi-robot rendezvous on a partially observed grid."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # action encoding kept from v5.py: 0=up(x-1) 1=down(x+1) 2=left(y-1) 3=right(y+1) 4=stay
    _MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    def __init__(self, config: Optional[EnvConfig] = None,
                 fixed_map: Optional[np.ndarray] = None,
                 fixed_starts: Optional[np.ndarray] = None,
                 fixed_dyn_starts: Optional[np.ndarray] = None,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        # deep-copy: the env must never mutate a caller-owned (possibly
        # shared) config object, e.g. when fixed_map overrides rows/cols
        self.cfg = copy.deepcopy(config) if config is not None else EnvConfig()
        self.render_mode = render_mode
        self._fixed_map = None if fixed_map is None else np.array(fixed_map, dtype=np.int8)
        self._fixed_starts = None if fixed_starts is None else np.array(fixed_starts, dtype=np.int32)
        if self._fixed_map is not None:
            r, c = self._fixed_map.shape
            self.cfg.rows, self.cfg.cols = r, c
        if self._fixed_starts is not None:
            fs = self._fixed_starts
            if len(fs) != self.cfg.num_robots:
                raise ValueError("fixed_starts must have one row per robot")
            if len({tuple(p) for p in fs}) != len(fs):
                raise ValueError("fixed_starts contains duplicate cells")
            if (fs[:, 0].min() < 0 or fs[:, 0].max() >= self.cfg.rows
                    or fs[:, 1].min() < 0 or fs[:, 1].max() >= self.cfg.cols):
                raise ValueError("fixed_starts out of bounds")
            if self._fixed_map is not None and \
                    any(self._fixed_map[p[0], p[1]] == OBSTACLE for p in fs):
                raise ValueError("fixed_starts placed on an obstacle cell")

        self._fixed_dyn_starts = None if fixed_dyn_starts is None \
            else np.array(fixed_dyn_starts, dtype=np.int32)
        if self._fixed_dyn_starts is not None:
            fd = self._fixed_dyn_starts
            self.cfg.num_dynamic_obstacles = len(fd)
            if len({tuple(p) for p in fd}) != len(fd):
                raise ValueError("fixed_dyn_starts contains duplicate cells")
            if (fd[:, 0].min() < 0 or fd[:, 0].max() >= self.cfg.rows
                    or fd[:, 1].min() < 0 or fd[:, 1].max() >= self.cfg.cols):
                raise ValueError("fixed_dyn_starts out of bounds")
            if self._fixed_map is not None and \
                    any(self._fixed_map[p[0], p[1]] == OBSTACLE for p in fd):
                raise ValueError("fixed_dyn_starts placed on a static obstacle")
            if self._fixed_starts is not None and \
                    {tuple(p) for p in fd} & {tuple(p) for p in self._fixed_starts}:
                raise ValueError("fixed_dyn_starts overlap fixed_starts")

        n, R, C = self.cfg.num_robots, self.cfg.rows, self.cfg.cols
        self.action_space = spaces.MultiDiscrete([5] * n)
        self.observation_space = spaces.Dict({
            "known_map": spaces.Box(low=-1, high=1, shape=(R, C), dtype=np.int8),
            "robot_positions": spaces.Box(low=0, high=max(R, C) - 1,
                                          shape=(n, 2), dtype=np.int32),
        })

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # NOTE: assign after the np_random property setter, which clobbers
        # gymnasium's internal seed record with -1
        self._ctor_seed = seed

        self.grid_map: np.ndarray = np.zeros((R, C), dtype=np.int8)
        self._static_map: np.ndarray = np.zeros((R, C), dtype=np.int8)
        self._dyn_pos: np.ndarray = np.zeros((0, 2), dtype=np.int32)
        self.known_map: np.ndarray = np.full((R, C), UNKNOWN, dtype=np.int8)
        self.positions: np.ndarray = np.zeros((n, 2), dtype=np.int32)
        self.best_area: int = 0
        self.prev_area: int = 0
        self.step_count: int = 0
        self.distances: np.ndarray = np.zeros(n, dtype=np.int64)
        self._screen = None

    # ------------------------------------------------------------------ #
    # geometry                                                           #
    # ------------------------------------------------------------------ #
    def _bounding_square(self) -> Tuple[int, Tuple[int, int, int, int]]:
        """Return (area, (min_x, min_y, side)) of the fleet bounding square.

        side = max extent over both axes; square anchored at (min_x, min_y).
        """
        xs, ys = self.positions[:, 0], self.positions[:, 1]
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        side = max(max_x - min_x, max_y - min_y) + 1
        return side * side, (min_x, min_y, side, side)

    def _square_free(self, min_x: int, min_y: int, side: int) -> bool:
        if min_x + side > self.cfg.rows or min_y + side > self.cfg.cols:
            return False
        region = self.grid_map[min_x:min_x + side, min_y:min_y + side]
        return bool((region == FREE).all())

    def _reveal(self, pos: np.ndarray) -> None:
        r = self.cfg.lidar_radius
        x0, x1 = max(0, pos[0] - r), min(self.cfg.rows, pos[0] + r + 1)
        y0, y1 = max(0, pos[1] - r), min(self.cfg.cols, pos[1] + r + 1)
        self.known_map[x0:x1, y0:y1] = self.grid_map[x0:x1, y0:y1]

    # ------------------------------------------------------------------ #
    # gym API                                                            #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg

        placed = False
        for _attempt in range(64):
            if self._fixed_map is not None:
                self.grid_map = self._fixed_map.copy()
            else:
                self.grid_map = MapGen.generate(
                    cfg.rows, cfg.cols, cfg.num_clusters,
                    cfg.cluster_size_range, cfg.min_cluster_distance,
                    self.np_random)

            if self._fixed_starts is not None:
                self.positions = self._fixed_starts.copy()
            else:
                free_cells = np.argwhere(self.grid_map == FREE)
                if len(free_cells) < cfg.num_robots:
                    continue
                idx = self.np_random.choice(len(free_cells),
                                            size=cfg.num_robots, replace=False)
                self.positions = free_cells[idx].astype(np.int32)

            area, _ = self._bounding_square()
            # don't start the episode already solved (legacy envs did, which
            # produced trivial 1-step successes in the stored revision data)
            if area > cfg.threshold_area or self._fixed_starts is not None:
                placed = True
                break
        if not placed:
            raise RuntimeError(
                "reset(): could not place robots with bounding area > "
                f"threshold ({cfg.threshold_area}) in 64 attempts -- "
                "map too small/crowded for this configuration")
        if self._fixed_starts is not None and self._fixed_map is None:
            if any(self.grid_map[p[0], p[1]] == OBSTACLE for p in self.positions):
                raise ValueError("fixed_starts collide with generated map "
                                 "obstacles; supply fixed_map as well")

        # dynamic obstacles: spawn AFTER robot placement so the static map
        # and robot starts are bit-identical across num_dynamic_obstacles
        # levels for a given reset seed (paired evaluation design).  With
        # k == 0 this block consumes no RNG draws.
        self._static_map = self.grid_map.copy()
        k = cfg.num_dynamic_obstacles
        if self._fixed_dyn_starts is not None:
            fd = self._fixed_dyn_starts
            robot_cells = {tuple(p) for p in self.positions}
            for p in fd:
                if self.grid_map[p[0], p[1]] == OBSTACLE:
                    raise ValueError("fixed_dyn_starts on a static obstacle")
                if tuple(p) in robot_cells:
                    raise ValueError("fixed_dyn_starts on a robot cell")
            self._dyn_pos = fd.copy()
        elif k > 0:
            robot_cells = {tuple(p) for p in self.positions}
            pool = np.array([c for c in np.argwhere(self.grid_map == FREE)
                             if tuple(c) not in robot_cells])
            if len(pool) < k:
                raise RuntimeError(
                    f"reset(): only {len(pool)} free non-robot cells for "
                    f"{k} dynamic obstacles")
            idx = self.np_random.choice(len(pool), size=k, replace=False)
            self._dyn_pos = pool[idx].astype(np.int32)
        else:
            self._dyn_pos = np.zeros((0, 2), dtype=np.int32)
        for p in self._dyn_pos:
            self.grid_map[p[0], p[1]] = OBSTACLE

        self.known_map = np.full((cfg.rows, cfg.cols), UNKNOWN, dtype=np.int8)
        for p in self.positions:
            self._reveal(p)

        self.best_area, _ = self._bounding_square()
        self.prev_area = self.best_area
        self.step_count = 0
        self.distances = np.zeros(cfg.num_robots, dtype=np.int64)

        return self._obs(), {"bounding_area": self.best_area}

    def step(self, actions):
        cfg = self.cfg
        self.step_count += 1
        n = cfg.num_robots
        actions = np.asarray(actions).astype(int)

        cur = [tuple(p) for p in self.positions]
        targets = []
        obstacle_hit = [False] * n
        for i in range(n):
            dx, dy = self._MOVES[int(actions[i]) if int(actions[i]) in self._MOVES else 4]
            nx = int(np.clip(cur[i][0] + dx, 0, cfg.rows - 1))
            ny = int(np.clip(cur[i][1] + dy, 0, cfg.cols - 1))
            if self.grid_map[nx, ny] == OBSTACLE:
                obstacle_hit[i] = True
                targets.append(cur[i])           # reverted
            else:
                targets.append((nx, ny))

        robot_collide = self._resolve_conflicts(cur, targets)

        # commit moves, distances, map reveal
        for i in range(n):
            if targets[i] != cur[i]:
                self.distances[i] += 1
            self.positions[i] = targets[i]
            self._reveal(self.positions[i])

        # dynamic obstacles move after the robots commit (robots acted on
        # the pre-move obstacle field); re-reveal so known_map reflects the
        # post-move truth inside every LiDAR window before the goal check
        if self._dyn_pos.shape[0] > 0:
            self._move_dynamic_obstacles()
            for p in self.positions:
                self._reveal(p)

        # ---------------- reward (accumulative, Table III) -------------- #
        rw = cfg.rewards
        reward = 0.0
        n_obs = int(np.sum(obstacle_hit))
        n_rob = int(np.sum(robot_collide))
        reward += n_obs * rw.collide_obstacle
        reward += n_rob * rw.collide_robot

        area, (min_x, min_y, side, _) = self._bounding_square()
        terminated = False
        reward += rw.step_cost
        if rw.move_cost != 0.0:
            n_moves = sum(1 for i in range(n) if targets[i] != cur[i])
            reward += rw.move_cost * n_moves
        if area <= cfg.threshold_area and self._square_free(min_x, min_y, side):
            reward += rw.goal
            terminated = True
        if rw.potential_coef != 0.0:
            reward += rw.potential_coef * (self.prev_area - area)
        elif not terminated:
            if area < self.best_area:
                reward += rw.area_decrease
            elif area > self.prev_area:
                reward += rw.area_increase
        self.best_area = min(self.best_area, area)
        self.prev_area = area

        truncated = (not terminated) and self.step_count >= cfg.max_steps

        info = {
            "bounding_area": area,
            "best_area": self.best_area,
            "obstacle_collisions": n_obs,
            "robot_collisions": n_rob,
            "distances": self.distances.copy(),
            "total_distance": int(self.distances.sum()),
            "max_distance": int(self.distances.max()),
            "is_success": terminated,
            "dyn_positions": self._dyn_pos.copy(),
        }

        if self.render_mode == "human":
            self._render_frame()

        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # dynamic obstacles                                                  #
    # ------------------------------------------------------------------ #
    def _move_dynamic_obstacles(self) -> None:
        """Unbiased 4-neighbour random walk over static-free cells.

        Runs after the robots commit.  Candidate cells exclude static
        obstacles, committed robot cells, cells already taken this tick,
        and the current cells of obstacles that have not moved yet, so
        obstacles never overlap robots or each other (an obstacle may
        chain-follow into a cell vacated earlier this tick, mirroring the
        robot semantics).  An obstacle with no candidate, or one gated out
        by dyn_move_prob, stays put.  All randomness comes from
        self.np_random, so episodes are reproducible given the reset seed.
        """
        k = self._dyn_pos.shape[0]
        robot_cells = {tuple(p) for p in self.positions}
        cur_cells = [tuple(p) for p in self._dyn_pos]
        new_cells = list(cur_cells)
        taken = set()
        for i in range(k):
            gated = (self.cfg.dyn_move_prob < 1.0
                     and self.np_random.random() > self.cfg.dyn_move_prob)
            if not gated:
                cands = []
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = cur_cells[i][0] + dx, cur_cells[i][1] + dy
                    if not (0 <= nx < self.cfg.rows and 0 <= ny < self.cfg.cols):
                        continue
                    cell = (nx, ny)
                    if (self._static_map[nx, ny] != FREE
                            or cell in robot_cells or cell in taken
                            or any(cell == cur_cells[j] for j in range(i + 1, k))):
                        continue
                    cands.append(cell)
                if cands:
                    new_cells[i] = cands[int(self.np_random.integers(len(cands)))]
            taken.add(new_cells[i])
        # two-phase grid rewrite so stayers are handled correctly
        for cell in cur_cells:
            self.grid_map[cell] = self._static_map[cell]
        for cell in new_cells:
            self.grid_map[cell] = OBSTACLE
        self._dyn_pos = np.array(new_cells, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # collision resolution                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_conflicts(cur: List[tuple], targets: List[tuple]) -> List[bool]:
        """Iteratively revert conflicting moves. Mutates `targets` in place.

        Detects: (a) two or more robots targeting the same cell,
                 (b) swaps (i -> j's cell while j -> i's cell),
                 (c) moving into a cell whose occupant stays (or was reverted).
        Following a robot into the cell it vacates is allowed.
        Reverts cascade until a fixpoint (reverting one robot can invalidate
        another's move). Returns a per-robot collision flag.
        """
        n = len(cur)
        collide = [False] * n
        changed = True
        while changed:
            changed = False
            # (a) same-target conflicts among moving robots
            seen = {}
            for i in range(n):
                if targets[i] == cur[i]:
                    continue
                if targets[i] in seen:
                    j = seen[targets[i]]
                    for k in (i, j):
                        if targets[k] != cur[k]:
                            targets[k] = cur[k]
                            collide[k] = True
                            changed = True
                else:
                    seen[targets[i]] = i
            # (b) swaps
            for i in range(n):
                if targets[i] == cur[i]:
                    continue
                for j in range(i + 1, n):
                    if targets[j] == cur[j]:
                        continue
                    if targets[i] == cur[j] and targets[j] == cur[i]:
                        targets[i], targets[j] = cur[i], cur[j]
                        collide[i] = collide[j] = True
                        changed = True
            # (c) moving into a cell whose occupant is not leaving
            stay_cells = {cur[i] for i in range(n) if targets[i] == cur[i]}
            for i in range(n):
                if targets[i] != cur[i] and targets[i] in stay_cells:
                    targets[i] = cur[i]
                    collide[i] = True
                    changed = True
        return collide

    # ------------------------------------------------------------------ #
    def _obs(self):
        return {
            "known_map": self.known_map.copy(),
            "robot_positions": self.positions.copy(),
        }

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        import pygame
        cell = 24
        R, C = self.cfg.rows, self.cfg.cols
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((C * cell, R * cell))
        s = self._screen
        s.fill((255, 255, 255))
        for x in range(R):
            for y in range(C):
                rect = (y * cell, x * cell, cell, cell)
                if self.known_map[x, y] == OBSTACLE:
                    pygame.draw.rect(s, (0, 0, 0), rect)
                elif self.known_map[x, y] == UNKNOWN:
                    pygame.draw.rect(s, (220, 220, 220), rect)
                pygame.draw.rect(s, (180, 180, 180), rect, 1)
        for p in self.positions:
            pygame.draw.rect(s, (200, 30, 30),
                             (p[1] * cell, p[0] * cell, cell, cell))
        _, (mx, my, side, _) = self._bounding_square()
        pygame.draw.rect(s, (30, 160, 30),
                         (my * cell, mx * cell, side * cell, side * cell), 2)
        pygame.display.update()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # convenience for provenance sidecars
    def config_dict(self):
        return asdict(self.cfg)
