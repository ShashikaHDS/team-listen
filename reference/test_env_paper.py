"""Unit tests for the canonical rendezvous env (run: python test_env_paper.py).

Each test constructs small fixed maps so every collision/reward branch is
exercised deterministically. Written as plain asserts so no pytest install
is required (pytest will also pick them up if available).
"""

import numpy as np

from env_paper import RendezvousEnv, EnvConfig, RewardConfig, UNKNOWN, FREE, OBSTACLE, MapGen

FREE5 = np.zeros((7, 7), dtype=np.int8)          # 7x7 all-free arena


def make_env(starts, grid=None, n=None, max_steps=300, threshold=4):
    starts = np.array(starts, dtype=np.int32)
    n = n or len(starts)
    grid = FREE5 if grid is None else np.array(grid, dtype=np.int8)
    cfg = EnvConfig(num_robots=n, rows=grid.shape[0], cols=grid.shape[1],
                    threshold_area=threshold, max_steps=max_steps)
    env = RendezvousEnv(cfg, fixed_map=grid, fixed_starts=starts)
    env.reset(seed=0)
    return env


def test_determinism():
    cfg = EnvConfig(num_robots=4)
    a = RendezvousEnv(cfg)
    b = RendezvousEnv(cfg)
    oa, _ = a.reset(seed=123)
    ob, _ = b.reset(seed=123)
    assert np.array_equal(a.grid_map, b.grid_map), "same seed -> same map"
    assert np.array_equal(oa["robot_positions"], ob["robot_positions"])
    rng = np.random.default_rng(7)
    for _ in range(50):
        act = rng.integers(0, 5, size=4)
        ra = a.step(act)
        rb = b.step(act)
        assert ra[1] == rb[1] and np.array_equal(
            ra[0]["robot_positions"], rb[0]["robot_positions"])
        if ra[2] or ra[3]:
            break
    print("PASS determinism")


def test_unknown_init():
    env = RendezvousEnv(EnvConfig(num_robots=3))
    obs, _ = env.reset(seed=1)
    km = obs["known_map"]
    assert (km == UNKNOWN).any(), "most of the map should start unknown"
    # every revealed cell must match ground truth; each robot reveals 3x3
    revealed = km != UNKNOWN
    assert (km[revealed] == env.grid_map[revealed]).all()
    for p in obs["robot_positions"]:
        assert km[p[0], p[1]] != UNKNOWN
    print("PASS unknown_init")


def test_obstacle_collision():
    grid = FREE5.copy()
    grid[2, 3] = OBSTACLE
    # robot 0 at (2,2) moves right into the obstacle; robot 1 idles far away
    env = make_env([[2, 2], [6, 6]], grid=grid)
    obs, r, term, trunc, info = env.step([3, 4])          # right, stay
    assert tuple(obs["robot_positions"][0]) == (2, 2), "move reverted"
    assert info["obstacle_collisions"] == 1
    assert r <= RewardConfig().collide_obstacle, f"collision must reach reward, got {r}"
    print("PASS obstacle_collision")


def test_same_target_conflict():
    # robots at (3,2) and (3,4) both move toward (3,3)
    env = make_env([[3, 2], [3, 4], [6, 6]], n=3)
    obs, r, *_ , info = env.step([3, 2, 4])               # right, left, stay
    assert tuple(obs["robot_positions"][0]) == (3, 2)
    assert tuple(obs["robot_positions"][1]) == (3, 4)
    assert info["robot_collisions"] == 2
    print("PASS same_target_conflict")


def test_swap_conflict():
    env = make_env([[3, 2], [3, 3], [6, 6]], n=3)
    obs, r, *_, info = env.step([3, 2, 4])                # 0 right, 1 left = swap
    assert tuple(obs["robot_positions"][0]) == (3, 2)
    assert tuple(obs["robot_positions"][1]) == (3, 3)
    assert info["robot_collisions"] == 2
    print("PASS swap_conflict")


def test_move_into_stationary():
    env = make_env([[3, 2], [3, 3], [6, 6]], n=3)
    obs, r, *_, info = env.step([3, 4, 4])                # 0 right into staying 1
    assert tuple(obs["robot_positions"][0]) == (3, 2)
    assert info["robot_collisions"] == 1, "only the mover collides"
    print("PASS move_into_stationary")


def test_follow_vacated_cell_allowed():
    env = make_env([[3, 2], [3, 3], [6, 6]], n=3)
    obs, r, *_, info = env.step([3, 3, 4])                # both move right in a chain
    assert tuple(obs["robot_positions"][0]) == (3, 3)
    assert tuple(obs["robot_positions"][1]) == (3, 4)
    assert info["robot_collisions"] == 0
    print("PASS follow_vacated_cell_allowed")


def test_revert_cascade():
    # 0 and 1 collide head-on (swap); 2 was moving into the cell 1 vacates.
    # After 1 is reverted, 2's move must also be reverted (cascade).
    env = make_env([[3, 2], [3, 3], [2, 3], [6, 6]], n=4)
    obs, r, *_, info = env.step([3, 2, 1, 4])   # 0 right, 1 left (swap), 2 down into (3,3)
    assert tuple(obs["robot_positions"][2]) == (2, 3), "cascade revert"
    assert info["robot_collisions"] == 3
    print("PASS revert_cascade")


def test_reward_accounting():
    rw = RewardConfig()
    # shrink: 2 robots far apart, one steps closer -> area decreases below best
    env = make_env([[1, 1], [1, 5], [5, 1]], n=3, threshold=1)
    _, r, *_ = env.step([4, 2, 4])            # robot 1 left: extent 4->3, area 25->25? x-extent 5
    # area before: side max(4,4)+1=5 ->25 ; after: max(4,3)+1=5 -> still 25 (x extent governs)
    assert r == 0.0, f"no improvement, no growth -> 0, got {r}"
    _, r, *_ = env.step([4, 4, 0])            # robot 2 up: x-extent 4->3, side=max(3,3)+1=4 area16<25
    assert r == rw.area_decrease, f"shrink -> +20, got {r}"
    _, r, *_ = env.step([4, 4, 1])            # robot 2 back down: area grows 16->25
    assert r == rw.area_increase, f"growth -> -0.5, got {r}"
    # collision + shrink must ACCUMULATE in the same step (the legacy env
    # overwrote the -5 with the area term -- the exact bug we fixed)
    grid = FREE5.copy()
    grid[1, 2] = OBSTACLE
    env2 = make_env([[1, 1], [1, 5], [5, 1]], grid=grid, n=3, threshold=1)
    # robot0 hits the obstacle (-5) while robots 1+2 shrink both extents:
    # side 5->4, area 25->16 < best (+20) => net +15
    _, r, *_, info = env2.step([3, 2, 0])
    assert info["obstacle_collisions"] == 1
    expected = rw.collide_obstacle + rw.area_decrease
    assert r == expected, f"collision+shrink must accumulate: want {expected}, got {r}"
    # collision alone with no area change -> exactly -5
    _, r, *_, info = env2.step([3, 4, 4])
    assert info["obstacle_collisions"] == 1
    assert r == rw.collide_obstacle, f"collision penalty must survive, got {r}"
    print("PASS reward_accounting")


def test_goal_and_termination():
    rw = RewardConfig()
    # 3 robots nearly together; start area 9 (>4), one move reaches area 4
    env = make_env([[2, 2], [2, 3], [4, 3]], n=3, threshold=4)
    obs, r, term, trunc, info = env.step([4, 4, 4])       # all stay: area 9, no goal
    assert not term and r == 0.0
    obs, r, term, trunc, info = env.step([4, 4, 0])       # robot2 (4,3)->(3,3): side 2, area 4
    assert term and info["is_success"]
    assert r == rw.goal, f"goal reward exactly +100, got {r}"
    print("PASS goal_and_termination")


def test_goal_blocked_by_obstacle():
    grid = FREE5.copy()
    grid[2, 2] = OBSTACLE                                  # inside the bounding square
    env = make_env([[1, 1], [1, 3], [3, 1]], grid=grid, n=3, threshold=9)
    obs, r, term, trunc, info = env.step([4, 4, 4])        # area 9 <= 9 but square not free
    assert not term, "square containing an obstacle is not a valid goal"
    print("PASS goal_blocked_by_obstacle")


def test_truncation():
    env = make_env([[1, 1], [5, 5]], max_steps=5, threshold=1)
    for i in range(5):
        obs, r, term, trunc, info = env.step([4, 4])
        assert not term
    assert trunc, "episode must truncate at max_steps"
    print("PASS truncation")


def test_spawn_not_solved():
    cfg = EnvConfig(num_robots=4, threshold_area=16)
    env = RendezvousEnv(cfg)
    for s in range(30):
        _, info = env.reset(seed=s)
        assert info["bounding_area"] > 16, "episodes must not start solved"
    print("PASS spawn_not_solved")


def test_mapgen_connectivity():
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = MapGen.generate(20, 20, 5, (2, 10), 3.0, rng)
        free = np.argwhere(g == FREE)
        assert len(free) > 0
        # BFS from first free cell must reach all free cells
        seen = np.zeros_like(g, dtype=bool)
        stack = [tuple(free[0])]
        seen[tuple(free[0])] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 20 and 0 <= ny < 20 and not seen[nx, ny] and g[nx, ny] == FREE:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
        assert seen.sum() == len(free), "free space must be one component"
        assert g[0, :].max() == FREE and g[-1, :].max() == FREE, "border ring free"
    print("PASS mapgen_connectivity")


def test_config_not_mutated():
    # regression: constructing an env with a fixed_map of a different size
    # must not write into the caller's (possibly shared) EnvConfig
    shared = EnvConfig(num_robots=2, rows=20, cols=20)
    small = np.zeros((6, 6), dtype=np.int8)
    env_small = RendezvousEnv(shared, fixed_map=small,
                              fixed_starts=np.array([[0, 0], [5, 5]]))
    assert shared.rows == 20 and shared.cols == 20, "caller config mutated!"
    env_big = RendezvousEnv(shared)
    env_big.reset(seed=0)
    assert env_big.grid_map.shape == (20, 20)
    print("PASS config_not_mutated")


def test_fixed_starts_validation():
    grid = FREE5.copy()
    grid[2, 2] = OBSTACLE
    for bad in ([[1, 1], [1, 1]],          # duplicate
                [[1, 1], [9, 9]],          # out of bounds (7x7 grid)
                [[1, 1], [2, 2]]):         # on obstacle
        try:
            RendezvousEnv(EnvConfig(num_robots=2, rows=7, cols=7),
                          fixed_map=grid, fixed_starts=np.array(bad))
            raise AssertionError(f"fixed_starts {bad} accepted")
        except ValueError:
            pass
    print("PASS fixed_starts_validation")


def test_astar_baseline_sanity():
    # exercises the stall-recovery + goal-validity fixes: across heuristics
    # and 10 held-out maps the baseline must mostly succeed, and any failure
    # must not be a frozen-fleet truncation (zero distance)
    from astar_paper import run_astar_episode, HEURISTICS
    env = RendezvousEnv(EnvConfig(num_robots=4))
    results = []
    for h in HEURISTICS:
        for ep in range(10):
            m = run_astar_episode(env, h, seed=10_000 + ep)
            results.append((h, ep, m))
            assert m["success"] or m["total_distance"] > 0, \
                f"frozen-fleet stall: {h} ep{ep} -> {m}"
    rate = np.mean([m["success"] for _, _, m in results])
    per_h = {h: np.mean([m["success"] for hh, _, m in results if hh == h])
             for h in HEURISTICS}
    print(f"     A* success rates: {per_h} (overall {rate:.2f})")
    assert rate >= 0.8, f"A* baseline too weak after fixes: {per_h}"
    env.close()
    print("PASS astar_baseline_sanity")


def test_sb3_env_checker():
    try:
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("SKIP sb3_env_checker (SB3 not installed)")
        return
    env = RendezvousEnv(EnvConfig(num_robots=4), seed=0)
    check_env(env, warn=True)
    print("PASS sb3_env_checker")


# ---------------------------------------------------------------------- #
# dynamic obstacles (evaluation-time extension)                          #
# ---------------------------------------------------------------------- #

def test_dyn_static_equivalence():
    """num_dynamic_obstacles=0 must be bit-for-bit the static env."""
    a = RendezvousEnv(EnvConfig(num_robots=4))
    b = RendezvousEnv(EnvConfig(num_robots=4, num_dynamic_obstacles=0))
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.array_equal(a.grid_map, b.grid_map)
    assert np.array_equal(oa["known_map"], ob["known_map"])
    rng = np.random.default_rng(3)
    for _ in range(50):
        act = rng.integers(0, 5, size=4)
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(ra[0]["known_map"], rb[0]["known_map"])
        assert np.array_equal(ra[0]["robot_positions"], rb[0]["robot_positions"])
        assert ra[1] == rb[1] and ra[2] == rb[2] and ra[3] == rb[3]
        if ra[2] or ra[3]:
            break
    print("PASS dyn_static_equivalence")


def test_dyn_spawn_validity():
    for seed in (0, 1, 2, 42):
        env = RendezvousEnv(EnvConfig(num_robots=4, num_dynamic_obstacles=8))
        env.reset(seed=seed)
        dyn = {tuple(p) for p in env._dyn_pos}
        assert len(dyn) == 8, "spawn count/duplicates"
        robots = {tuple(p) for p in env.positions}
        assert not (dyn & robots), "dyn spawned on a robot"
        for p in env._dyn_pos:
            assert env._static_map[p[0], p[1]] == FREE, "dyn on static obstacle"
            assert env.grid_map[p[0], p[1]] == OBSTACLE, "dyn not in composite"
        g = env.grid_map.copy()
        for p in env._dyn_pos:
            g[p[0], p[1]] = env._static_map[p[0], p[1]]
        assert np.array_equal(g, env._static_map), "composite minus dyn != static"
    print("PASS dyn_spawn_validity")


def test_dyn_paired_maps():
    """Same reset seed => identical static map and robot starts across k."""
    a = RendezvousEnv(EnvConfig(num_robots=4, num_dynamic_obstacles=0))
    b = RendezvousEnv(EnvConfig(num_robots=4, num_dynamic_obstacles=8))
    a.reset(seed=10005)
    b.reset(seed=10005)
    assert np.array_equal(a.grid_map, b._static_map), "static map differs"
    assert np.array_equal(a.positions, b.positions), "robot starts differ"
    print("PASS dyn_paired_maps")


def test_dyn_forced_move_and_stay():
    # corridor: obstacle at (3,2) has exactly one free neighbour (3,3);
    # pocketed obstacle at (5,5) is fully walled -> must stay
    grid = np.ones((7, 7), dtype=np.int8)
    grid[3, 2] = FREE
    grid[3, 3] = FREE
    grid[1, 1] = FREE          # robot parked far away
    grid[5, 5] = FREE          # pocket
    env = RendezvousEnv(EnvConfig(num_robots=1, rows=7, cols=7),
                        fixed_map=grid, fixed_starts=[(1, 1)],
                        fixed_dyn_starts=[(3, 2), (5, 5)])
    env.reset(seed=0)
    env.step([4])
    assert tuple(env._dyn_pos[0]) == (3, 3), "forced move not taken"
    assert tuple(env._dyn_pos[1]) == (5, 5), "walled obstacle moved"
    assert env.grid_map[3, 2] == FREE, "vacated cell not restored to static"
    assert env.grid_map[3, 3] == OBSTACLE, "new cell not marked"
    print("PASS dyn_forced_move_and_stay")


def test_dyn_no_move_onto_robot():
    # obstacle at (3,3); its only static-free neighbour (3,4) holds a robot
    grid = np.ones((7, 7), dtype=np.int8)
    grid[3, 3] = FREE
    grid[3, 4] = FREE
    grid[1, 1] = FREE
    env = RendezvousEnv(EnvConfig(num_robots=2, rows=7, cols=7),
                        fixed_map=grid, fixed_starts=[(1, 1), (3, 4)],
                        fixed_dyn_starts=[(3, 3)])
    env.reset(seed=0)
    env.step([4, 4])
    assert tuple(env._dyn_pos[0]) == (3, 3), "obstacle moved onto robot"
    print("PASS dyn_no_move_onto_robot")


def test_dyn_blocks_goal():
    # fleet inside the threshold square; a dynamic obstacle boxed in by the
    # robots and two static walls sits inside the square and cannot leave,
    # so the goal must never fire while it is there
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[2, 1] = OBSTACLE
    grid[1, 2] = OBSTACLE
    starts = [(0, 0), (0, 1), (1, 0)]
    env = RendezvousEnv(EnvConfig(num_robots=3, rows=7, cols=7,
                                  threshold_area=4),
                        fixed_map=grid, fixed_starts=starts,
                        fixed_dyn_starts=[(1, 1)])
    env.reset(seed=0)
    for _ in range(5):
        obs, r, term, trunc, info = env.step([4, 4, 4])
        assert tuple(env._dyn_pos[0]) == (1, 1), "boxed obstacle moved"
        assert not term, "goal fired with dynamic obstacle in the square"
    # without the dynamic obstacle the same configuration terminates at once
    env2 = RendezvousEnv(EnvConfig(num_robots=3, rows=7, cols=7,
                                   threshold_area=4),
                         fixed_map=grid, fixed_starts=starts)
    env2.reset(seed=0)
    _, _, term2, _, _ = env2.step([4, 4, 4])
    assert term2, "static control did not terminate"
    print("PASS dyn_blocks_goal")


def test_dyn_robot_blocked():
    # robot tries to step into a walled (immobile) dynamic obstacle
    grid = np.ones((7, 7), dtype=np.int8)
    grid[3, 3] = FREE          # dyn obstacle pocket (walled)
    grid[3, 2] = FREE          # robot cell adjacent
    grid[1, 1] = FREE
    env = RendezvousEnv(EnvConfig(num_robots=2, rows=7, cols=7),
                        fixed_map=grid, fixed_starts=[(3, 2), (1, 1)],
                        fixed_dyn_starts=[(3, 3)])
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step([3, 4])   # robot 0 right into dyn
    assert tuple(env.positions[0]) == (3, 2), "move not reverted"
    assert info["obstacle_collisions"] == 1, "contact not counted"
    rw = env.cfg.rewards
    assert abs(r - (rw.collide_obstacle + rw.step_cost)) < 1e-9, \
        "reward != obstacle penalty + step cost"
    print("PASS dyn_robot_blocked")


def test_dyn_ghost_memory():
    # robot sees the obstacle, walks away, obstacle moves: the vacated cell
    # keeps its stale OBSTACLE value in known_map (a ghost) once it lies
    # outside every LiDAR window, while the true grid there is FREE
    grid = np.zeros((9, 9), dtype=np.int8)
    env = RendezvousEnv(EnvConfig(num_robots=2, rows=9, cols=9),
                        fixed_map=grid, fixed_starts=[(1, 1), (8, 8)],
                        fixed_dyn_starts=[(2, 2)])
    env.reset(seed=0)
    assert env.known_map[2, 2] == OBSTACLE, "dyn not revealed at reset"
    ghost_seen = False
    for _ in range(80):
        env.step([3, 4])                   # robot 0 walks right, robot 1 stays
        rx, ry = env.positions[0]
        outside = abs(2 - rx) > 1 or abs(2 - ry) > 1
        if outside and tuple(env._dyn_pos[0]) != (2, 2) \
                and env.grid_map[2, 2] == FREE:
            assert env.known_map[2, 2] == OBSTACLE, "ghost was cleared"
            ghost_seen = True
            break
    assert ghost_seen, "ghost condition never arose within 80 steps"
    print("PASS dyn_ghost_memory")


def test_dyn_determinism():
    a = RendezvousEnv(EnvConfig(num_robots=4, num_dynamic_obstacles=6))
    b = RendezvousEnv(EnvConfig(num_robots=4, num_dynamic_obstacles=6))
    a.reset(seed=99)
    b.reset(seed=99)
    assert np.array_equal(a._dyn_pos, b._dyn_pos), "spawn differs"
    rng = np.random.default_rng(5)
    for _ in range(50):
        act = rng.integers(0, 5, size=4)
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(a._dyn_pos, b._dyn_pos), "walk diverged"
        assert ra[1] == rb[1]
        assert np.array_equal(ra[0]["known_map"], rb[0]["known_map"])
        if ra[2] or ra[3]:
            break
    print("PASS dyn_determinism")


def test_fixed_dyn_starts_validation():
    grid = np.zeros((7, 7), dtype=np.int8)
    grid[2, 2] = OBSTACLE
    for bad in ([(1, 1), (1, 1)],            # duplicate
                [(9, 9)],                    # out of bounds
                [(2, 2)],                    # on static obstacle
                [(0, 0)]):                   # on a robot (fixed_starts)
        try:
            RendezvousEnv(EnvConfig(num_robots=2, rows=7, cols=7),
                          fixed_map=grid, fixed_starts=[(0, 0), (6, 6)],
                          fixed_dyn_starts=bad)
            raise AssertionError(f"validation accepted {bad}")
        except ValueError:
            pass
    print("PASS fixed_dyn_starts_validation")


if __name__ == "__main__":
    test_determinism()
    test_unknown_init()
    test_obstacle_collision()
    test_same_target_conflict()
    test_swap_conflict()
    test_move_into_stationary()
    test_follow_vacated_cell_allowed()
    test_revert_cascade()
    test_reward_accounting()
    test_goal_and_termination()
    test_goal_blocked_by_obstacle()
    test_truncation()
    test_spawn_not_solved()
    test_mapgen_connectivity()
    test_config_not_mutated()
    test_fixed_starts_validation()
    test_astar_baseline_sanity()
    test_sb3_env_checker()
    test_dyn_static_equivalence()
    test_dyn_spawn_validity()
    test_dyn_paired_maps()
    test_dyn_forced_move_and_stay()
    test_dyn_no_move_onto_robot()
    test_dyn_blocks_goal()
    test_dyn_robot_blocked()
    test_dyn_ghost_memory()
    test_dyn_determinism()
    test_fixed_dyn_starts_validation()
    print("\nAll tests passed.")
