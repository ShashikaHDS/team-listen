"""Isaac Sim backend for the continuous-space zero-shot validation.

Boots a headless (or windowed) Isaac Sim, builds the grid world as fixed
obstacle cuboids on a ground plane, represents robots as kinematically
driven prims (holonomic point kinematics, velocity-limited, matching
MockBackend semantics), and implements the LiDAR through PhysX
closest-hit raycasts against the obstacle colliders.

Import strategy is defensive across Isaac Sim 4.x/5.x API layouts; if
your installation still fails to import, run with --backend mock to
validate the pipeline and report the traceback so the imports can be
extended.  Tested design targets: SimulationApp from `isaacsim`
(4.0+) or `omni.isaac.kit` (2023.x), World/objects from
`isaacsim.core.api` or `omni.isaac.core`.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

OBSTACLE_HEIGHT = 0.30
RAY_Z = 0.15


class IsaacBackend:
    def __init__(self, vmax: float = 0.5, dt: float = 1.0 / 60.0,
                 n_beams: int = 72, max_range_factor: float = 2.5,
                 headless: bool = True):
        self.vmax = vmax
        self.dt = dt
        self.n_beams = n_beams
        self.max_range_factor = max_range_factor

        try:
            from isaacsim import SimulationApp          # Isaac Sim >= 4.0
        except ImportError:                             # 2022/2023.x
            from omni.isaac.kit import SimulationApp
        self._app = SimulationApp({"headless": headless})

        try:
            from isaacsim.core.api import World         # >= 4.5 layout
            from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        except ImportError:
            from omni.isaac.core import World
            from omni.isaac.core.objects import FixedCuboid, VisualCuboid
        self._World = World
        self._FixedCuboid = FixedCuboid
        self._VisualCuboid = VisualCuboid

        from omni.physx import get_physx_scene_query_interface
        self._query = get_physx_scene_query_interface()

        self.world = self._World(stage_units_in_meters=1.0,
                                 physics_dt=self.dt, rendering_dt=self.dt)
        self.world.scene.add_default_ground_plane()
        self._grid_key = None
        self._robot_prims = []
        self._obstacle_prims = []
        self.poses = np.zeros((0, 2))

    # ------------------------------------------------------------- #
    def _build_obstacles(self, grid: np.ndarray, cell_size: float) -> None:
        for prim in self._obstacle_prims:
            self.world.scene.remove_object(prim.name)
        self._obstacle_prims = []
        rows, cols = grid.shape
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 1:
                    cube = self._FixedCuboid(
                        prim_path=f"/World/obst_{idx}",
                        name=f"obst_{idx}",
                        position=np.array([(r + 0.5) * cell_size,
                                           (c + 0.5) * cell_size,
                                           OBSTACLE_HEIGHT / 2]),
                        scale=np.array([cell_size, cell_size,
                                        OBSTACLE_HEIGHT]),
                    )
                    self.world.scene.add(cube)
                    self._obstacle_prims.append(cube)
                    idx += 1

    def _build_robots(self, n: int, cell_size: float) -> None:
        for prim in self._robot_prims:
            self.world.scene.remove_object(prim.name)
        self._robot_prims = []
        size = cell_size * 0.6
        for i in range(n):
            # visual-only cuboid: rays must not hit robots (the shared map
            # holds obstacles only, matching the training environment)
            cube = self._VisualCuboid(
                prim_path=f"/World/robot_{i}",
                name=f"robot_{i}",
                position=np.array([0.0, 0.0, size / 2]),
                scale=np.array([size, size, size]),
                color=np.array([0.9, 0.2, 0.1]),
            )
            self.world.scene.add(cube)
            self._robot_prims.append(cube)

    # ------------------------------------------------------------- #
    def reset(self, grid: np.ndarray, starts_cells: List[Tuple[int, int]],
              cell_size: float) -> None:
        self.grid = np.asarray(grid, dtype=np.int8)
        self.cell_size = cell_size
        self.max_range = cell_size * self.max_range_factor
        key = (self.grid.tobytes(), cell_size)
        if key != self._grid_key:
            self._build_obstacles(self.grid, cell_size)
            self._grid_key = key
        if len(self._robot_prims) != len(starts_cells):
            self._build_robots(len(starts_cells), cell_size)
        self.poses = np.array(
            [[(c[0] + 0.5) * cell_size, (c[1] + 0.5) * cell_size]
             for c in starts_cells], dtype=np.float64)
        self._apply_poses()
        self.world.reset()
        for _ in range(3):                    # settle physics/queries
            self.world.step(render=False)

    def _apply_poses(self) -> None:
        z = self.cell_size * 0.3
        for i, prim in enumerate(self._robot_prims):
            prim.set_world_pose(
                position=np.array([self.poses[i][0], self.poses[i][1], z]))

    def get_poses(self) -> np.ndarray:
        return self.poses.copy()

    # ------------------------------------------------------------- #
    def drive_to(self, targets_m: np.ndarray, tol_m: float,
                 max_sim_s: float) -> float:
        t = 0.0
        targets_m = np.asarray(targets_m, dtype=np.float64)
        while t < max_sim_s:
            delta = targets_m - self.poses
            dist = np.linalg.norm(delta, axis=1)
            if (dist <= tol_m).all():
                break
            step = np.zeros_like(delta)
            moving = dist > tol_m
            step[moving] = (delta[moving].T
                            * np.minimum(self.vmax * self.dt / dist[moving],
                                         1.0)).T
            self.poses = self.poses + step
            self._apply_poses()
            self.world.step(render=False)
            t += self.dt
        delta = targets_m - self.poses
        dist = np.linalg.norm(delta, axis=1)
        self.poses[dist <= tol_m] = targets_m[dist <= tol_m]
        self._apply_poses()
        return t

    # ------------------------------------------------------------- #
    def raycast(self, origin_m: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_beams, 3))
        origin = (float(origin_m[0]), float(origin_m[1]), RAY_Z)
        for b in range(self.n_beams):
            ang = 2.0 * math.pi * b / self.n_beams
            d = (math.cos(ang), math.sin(ang), 0.0)
            res = self._query.raycast_closest(origin, d, self.max_range)
            if res.get("hit", False):
                p = res["position"]
                out[b] = [1.0, p[0], p[1]]
            else:
                out[b] = [0.0,
                          origin[0] + d[0] * self.max_range,
                          origin[1] + d[1] * self.max_range]
        return out

    def close(self) -> None:
        self._app.close()
