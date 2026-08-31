"""Isaac-independent core of the continuous-space zero-shot validation.

The trained final2m policies act on the same dict observation they saw in
training (known occupancy grid + robot grid cells).  This module supplies
everything between a *continuous* world backend and that policy:

  OccupancyMapper   continuous poses <-> grid cells, known-map maintenance
                    with UNKNOWN=-1 semantics and two reveal modes:
                      'grid'  oracle blanket reveal of the Chebyshev
                              radius-1 window (bit-parity with
                              env_paper._reveal, used for parity tests)
                      'lidar' classification from raycasts with realistic
                              occlusion (used for the actual validation)
  resolve_conflicts deployment-side mirror of env_paper's per-robot
                    conflict semantics (obstacle reject, same-target,
                    swap, occupied cell, cascading reverts)
  LockstepRunner    the deployment loop: observe -> sample action ->
                    conflict-resolve -> drive waypoints -> rescan ->
                    score, with the same 300-step cap, torch seeding, and
                    success predicate as the paper protocol

Backends implement: reset(grid, starts_cells), get_poses(),
drive_to(targets_m, tol_m, max_sim_s) -> sim_s, raycast(origin_m) ->
(n_beams, 3) array of [hit, x, y], close().  See mock_env.MockBackend
and isaac_env.IsaacBackend.

Scoring uses the TRUE grid (the evaluator's privilege, never the
policy's).  All randomness is seeded so runs are exactly reproducible.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_paper import UNKNOWN, FREE, OBSTACLE, RendezvousEnv  # noqa: E402

MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}


# --------------------------------------------------------------------- #
# geometry helpers                                                      #
# --------------------------------------------------------------------- #
def cell_center(cell: Tuple[int, int], cell_size: float) -> np.ndarray:
    """World (x, y) of a cell centre; world x maps to grid row axis."""
    return np.array([(cell[0] + 0.5) * cell_size,
                     (cell[1] + 0.5) * cell_size], dtype=np.float64)


def pose_to_cell(pose: np.ndarray, cell_size: float,
                 rows: int, cols: int) -> Tuple[int, int]:
    r = int(np.clip(math.floor(pose[0] / cell_size), 0, rows - 1))
    c = int(np.clip(math.floor(pose[1] / cell_size), 0, cols - 1))
    return (r, c)


# --------------------------------------------------------------------- #
# occupancy mapping                                                     #
# --------------------------------------------------------------------- #
class OccupancyMapper:
    def __init__(self, rows: int, cols: int, cell_size: float,
                 reveal: str = "lidar", sense_radius: int = 1):
        assert reveal in ("grid", "lidar")
        self.rows, self.cols = rows, cols
        self.cell_size = cell_size
        self.reveal = reveal
        self.sense_radius = sense_radius
        self.known = np.full((rows, cols), UNKNOWN, dtype=np.int8)

    def _in_window(self, cell, center_cell) -> bool:
        return (abs(cell[0] - center_cell[0]) <= self.sense_radius
                and abs(cell[1] - center_cell[1]) <= self.sense_radius)

    def update_grid_mode(self, true_grid: np.ndarray, robot_cell) -> None:
        r = self.sense_radius
        x0, x1 = max(0, robot_cell[0] - r), min(self.rows, robot_cell[0] + r + 1)
        y0, y1 = max(0, robot_cell[1] - r), min(self.cols, robot_cell[1] + r + 1)
        self.known[x0:x1, y0:y1] = true_grid[x0:x1, y0:y1]

    def update_lidar_mode(self, rays: np.ndarray, origin: np.ndarray,
                          robot_cell) -> None:
        """Classify cells in the sensing window from raycast returns.

        A ray segment travelling through a cell marks it free; a hit
        endpoint marks its cell an obstacle.  Cells occluded behind
        obstacles keep their previous value (realism gap vs 'grid').
        """
        step = self.cell_size * 0.25
        for hit, ex, ey in rays:
            end = np.array([ex, ey])
            vec = end - origin
            dist = float(np.linalg.norm(vec))
            if dist < 1e-9:
                continue
            direction = vec / dist
            travelled = 0.0
            # free space along the ray (stop short of the hit cell)
            limit = dist - (step * 0.5 if hit else 0.0)
            while travelled < limit:
                p = origin + direction * travelled
                cell = pose_to_cell(p, self.cell_size, self.rows, self.cols)
                if self._in_window(cell, robot_cell):
                    self.known[cell] = FREE
                travelled += step
            if hit:
                cell = pose_to_cell(end + direction * (step * 0.5),
                                    self.cell_size, self.rows, self.cols)
                if self._in_window(cell, robot_cell):
                    self.known[cell] = OBSTACLE
        # the robot's own cell is trivially free
        self.known[robot_cell] = FREE


# --------------------------------------------------------------------- #
# conflict resolution: delegate to the training env's own static method #
# so deployment semantics are guaranteed identical, not re-implemented  #
# --------------------------------------------------------------------- #
def resolve_conflicts(cur: List[Tuple[int, int]],
                      targets: List[Tuple[int, int]]) -> List[bool]:
    return RendezvousEnv._resolve_conflicts(cur, targets)


# --------------------------------------------------------------------- #
# the deployment loop                                                   #
# --------------------------------------------------------------------- #
@dataclass
class RunnerConfig:
    cell_size: float = 0.25
    reveal: str = "lidar"          # 'lidar' | 'grid'
    sense_radius: int = 1
    threshold_area: int = 16
    max_steps: int = 300
    arrive_tol: float = 0.02       # m
    max_sim_s_per_step: float = 5.0
    noise_sigma: float = 0.0       # m, gaussian on reported poses
    n_beams: int = 72


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    sim_time_s: float
    dist_m: np.ndarray
    dist_cells: np.ndarray
    jain: float
    obstacle_rejections: int
    robot_conflicts: int
    final_cells: List[Tuple[int, int]] = field(default_factory=list)


def bounding_square(cells: List[Tuple[int, int]]):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 1
    return side * side, (min(xs), min(ys), side)


def square_free(grid: np.ndarray, min_x: int, min_y: int, side: int) -> bool:
    if min_x + side > grid.shape[0] or min_y + side > grid.shape[1]:
        return False
    return bool((grid[min_x:min_x + side, min_y:min_y + side] == FREE).all())


class LockstepRunner:
    """Run one episode of the trained policy through a continuous backend."""

    def __init__(self, model, backend, true_grid: np.ndarray,
                 starts_cells: List[Tuple[int, int]],
                 cfg: RunnerConfig, noise_rng: Optional[np.random.Generator] = None):
        self.model = model
        self.backend = backend
        self.grid = np.asarray(true_grid, dtype=np.int8)
        self.rows, self.cols = self.grid.shape
        self.cfg = cfg
        self.n = len(starts_cells)
        self.noise_rng = noise_rng or np.random.default_rng(0)
        backend.reset(self.grid, starts_cells, cfg.cell_size)
        self.mapper = OccupancyMapper(self.rows, self.cols, cfg.cell_size,
                                      cfg.reveal, cfg.sense_radius)
        self.cells = list(starts_cells)
        self._scan_all()
        self.dist_m = np.zeros(self.n)
        self.prev_poses = backend.get_poses().copy()

    # ---------------- sensing ---------------- #
    def _reported_cells(self) -> List[Tuple[int, int]]:
        poses = self.backend.get_poses()
        cells = []
        for i in range(self.n):
            p = poses[i].astype(np.float64)
            if self.cfg.noise_sigma > 0:
                p = p + self.noise_rng.normal(0.0, self.cfg.noise_sigma, 2)
            cells.append(pose_to_cell(p, self.cfg.cell_size,
                                      self.rows, self.cols))
        return cells

    def _scan_all(self) -> None:
        poses = self.backend.get_poses()
        for i in range(self.n):
            cell = self.cells[i]
            if self.cfg.reveal == "grid":
                self.mapper.update_grid_mode(self.grid, cell)
            else:
                rays = self.backend.raycast(poses[i])
                self.mapper.update_lidar_mode(rays, poses[i], cell)

    # ---------------- one episode ---------------- #
    def run(self) -> EpisodeResult:
        cfg = self.cfg
        obstacle_rejections = 0
        robot_conflicts = 0
        sim_time = 0.0
        steps = 0
        terminated = False
        while steps < cfg.max_steps and not terminated:
            steps += 1
            obs = {
                "known_map": self.mapper.known.copy(),
                "robot_positions": np.array(self._reported_cells(),
                                            dtype=np.int32),
            }
            action, _ = self.model.predict(obs, deterministic=False)
            cur = list(self.cells)
            targets = []
            for i in range(self.n):
                a = int(action[i])
                dx, dy = MOVES.get(a, (0, 0))
                nx = int(np.clip(cur[i][0] + dx, 0, self.rows - 1))
                ny = int(np.clip(cur[i][1] + dy, 0, self.cols - 1))
                # reject a waypoint into a KNOWN obstacle (the host knows
                # the shared map; with sense_radius>=1 every adjacent cell
                # was observed, so this mirrors the env's truth check)
                if self.mapper.known[nx, ny] == OBSTACLE:
                    obstacle_rejections += 1
                    targets.append(cur[i])
                else:
                    targets.append((nx, ny))
            flags = resolve_conflicts(cur, targets)
            robot_conflicts += sum(flags)
            goal_pts = np.array([cell_center(t, cfg.cell_size)
                                 for t in targets])
            sim_time += self.backend.drive_to(goal_pts, cfg.arrive_tol,
                                              cfg.max_sim_s_per_step)
            poses = self.backend.get_poses()
            self.dist_m += np.linalg.norm(poses - self.prev_poses, axis=1)
            self.prev_poses = poses.copy()
            self.cells = [pose_to_cell(poses[i], cfg.cell_size,
                                       self.rows, self.cols)
                          for i in range(self.n)]
            self._scan_all()
            area, (mx, my, side) = bounding_square(self.cells)
            if area <= cfg.threshold_area and square_free(self.grid, mx,
                                                          my, side):
                terminated = True
        d = self.dist_m
        jain = float((d.sum() ** 2) / (self.n * (d ** 2).sum())) \
            if d.sum() > 0 else 1.0
        return EpisodeResult(
            success=terminated, steps=steps, sim_time_s=sim_time,
            dist_m=d.copy(), dist_cells=d / cfg.cell_size, jain=jain,
            obstacle_rejections=obstacle_rejections,
            robot_conflicts=robot_conflicts, final_cells=list(self.cells))
