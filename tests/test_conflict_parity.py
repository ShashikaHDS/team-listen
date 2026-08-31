"""Differential parity: grid_core.step_positions vs the reference env.

M1_SPEC section 1.8 (closed-form N=2 conflict resolution) and section 7
(test plan): the conflict-resolution port is the single most likely silent
divergence from ``reference/env_paper.py`` and is invisible in reward
curves, so it is tested differentially against the REAL reference code:

* the proposal phase (np.clip + obstacle revert) is mirrored verbatim from
  ``RendezvousEnv.step`` in ``ref_step`` below, and the conflict pass is the
  actual ``RendezvousEnv._resolve_conflicts`` static method -- never a
  re-implementation;
* the five verified reference cases (obstacle collision, same-target, swap,
  move-into-stationary, follow-vacated-cell-allowed) are reduced to N=2 and
  checked BOTH against ``ref_step`` and end-to-end against an instantiated
  ``RendezvousEnv`` (smallest valid env: num_robots=2, fixed map + starts);
* randomized fuzzing over fixed seed lists (>= 200 configurations total,
  no unseeded randomness): multi-step lockstep rollouts, one big batched
  call (exercising the (E,...) indexing), a latched stratum (a latched
  robot is a stayer, spec 1.11), and an instantiated-env lockstep stratum.

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_conflict_parity.py`` discovers and runs its
own tests with a pass/fail summary.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "reference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env_paper import RendezvousEnv, EnvConfig, OBSTACLE  # reference truth
from tasks.team_listen import grid_core
from tasks.team_listen.grid_core import UP, DOWN, LEFT, RIGHT, STAY  # noqa: F401


# ---------------------------------------------------------------------------
# Fixed seed lists (M1_SPEC section 7: fixed seeds, no unseeded randomness)
# ---------------------------------------------------------------------------

MULTISTEP_SEEDS = tuple(range(0, 200))          # 200 configs x 8 steps each
BATCHED_SEEDS = tuple(range(1000, 1256))        # 256 configs, one (E,...) call
LATCHED_SEEDS = tuple(range(5000, 5100))        # 100 configs with latched robots
ENV_SEEDS = tuple(range(9000, 9040))            # 40 configs vs instantiated env
STEPS_PER_CONFIG = 8


# ---------------------------------------------------------------------------
# Reference side
# ---------------------------------------------------------------------------

def ref_step(grid, cur, actions):
    """One reference transition: proposal phase + real _resolve_conflicts.

    The proposal loop below is copied line-for-line from
    ``RendezvousEnv.step`` (delta lookup via ``_MOVES``, ``np.clip`` to the
    grid, obstacle revert with a per-robot ``obstacle_hit`` flag).  The
    conflict pass is the reference's own static method, so the semantics
    under test are the reference's, not a transcription.

    Returns (targets, obstacle_hit, robot_collide) as python lists.
    """
    rows, cols = grid.shape
    cur_t = [tuple(int(x) for x in p) for p in cur]
    n = len(cur_t)
    targets = []
    obstacle_hit = [False] * n
    for i in range(n):
        a = int(actions[i])
        dx, dy = RendezvousEnv._MOVES[a if a in RendezvousEnv._MOVES else 4]
        nx = int(np.clip(cur_t[i][0] + dx, 0, rows - 1))
        ny = int(np.clip(cur_t[i][1] + dy, 0, cols - 1))
        if grid[nx, ny] == OBSTACLE:
            obstacle_hit[i] = True
            targets.append(cur_t[i])
        else:
            targets.append((nx, ny))
    robot_collide = RendezvousEnv._resolve_conflicts(cur_t, targets)
    return targets, obstacle_hit, robot_collide


def make_ref_env(grid, starts):
    """Smallest valid reference env: N=2, fixed map + starts.

    threshold_area=1 can never be met by two robots on distinct cells
    (bounding side >= 2 -> area >= 4), so the goal branch never terminates
    an episode mid-fuzz.
    """
    cfg = EnvConfig(num_robots=2, rows=grid.shape[0], cols=grid.shape[1],
                    threshold_area=1, max_steps=10_000)
    env = RendezvousEnv(cfg, fixed_map=grid,
                        fixed_starts=np.array(starts, dtype=np.int32))
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# Port side
# ---------------------------------------------------------------------------

def core_step(grid, cur, actions, latched=(False, False)):
    """Single-env call into grid_core.step_positions, python-ified output."""
    occ = torch.from_numpy(np.ascontiguousarray(grid)).to(torch.int8)
    nxt, ho, hr = grid_core.step_positions(
        torch.tensor([list(map(list, cur))], dtype=torch.int16),
        torch.tensor([list(actions)], dtype=torch.long),
        occ.unsqueeze(0),
        torch.tensor([list(latched)], dtype=torch.bool),
        (grid.shape[0], grid.shape[1]),
    )
    return ([tuple(int(x) for x in p) for p in nxt[0]],
            [bool(x) for x in ho[0]],
            [bool(x) for x in hr[0]])


def assert_parity(grid, cur, actions, ctx):
    """Compare one transition on both sides; return the reference outputs."""
    ref_t, ref_o, ref_r = ref_step(grid, cur, actions)
    core_t, core_o, core_r = core_step(grid, cur, actions)
    assert core_t == ref_t, (
        "position divergence at %s: actions=%s cur=%s ref=%s core=%s"
        % (ctx, list(actions), list(cur), ref_t, core_t))
    assert core_o == ref_o, (
        "hit_obstacle divergence at %s: actions=%s cur=%s ref=%s core=%s"
        % (ctx, list(actions), list(cur), ref_o, core_o))
    assert core_r == ref_r, (
        "hit_robot divergence at %s: actions=%s cur=%s ref=%s core=%s"
        % (ctx, list(actions), list(cur), ref_r, core_r))
    return ref_t, ref_o, ref_r


# ---------------------------------------------------------------------------
# The five verified reference cases (reference/test_env_paper.py), at N=2,
# each ALSO checked end-to-end against an instantiated RendezvousEnv.
# ---------------------------------------------------------------------------

FREE7 = np.zeros((7, 7), dtype=np.int8)


def _named_case(name, grid, starts, actions, exp_pos, exp_obs, exp_rob):
    # (1) explicit expectations from the verified reference tests
    core_t, core_o, core_r = core_step(grid, starts, actions)
    assert core_t == exp_pos, "%s: positions %s != expected %s" % (name, core_t, exp_pos)
    assert core_o == exp_obs, "%s: hit_obstacle %s != expected %s" % (name, core_o, exp_obs)
    assert core_r == exp_rob, "%s: hit_robot %s != expected %s" % (name, core_r, exp_rob)
    # (2) differential vs the reference proposal + _resolve_conflicts
    assert_parity(grid, starts, actions, name)
    # (3) end-to-end vs the instantiated reference env
    env = make_ref_env(grid, starts)
    obs, _r, _term, _trunc, info = env.step(list(actions))
    env_pos = [tuple(int(x) for x in p) for p in obs["robot_positions"]]
    assert env_pos == exp_pos, "%s: env positions %s != %s" % (name, env_pos, exp_pos)
    assert info["obstacle_collisions"] == sum(exp_obs), name
    assert info["robot_collisions"] == sum(exp_rob), name


def test_case_obstacle_collision():
    # reference test_obstacle_collision: revert + obstacle flag, NO robot flag
    grid = FREE7.copy()
    grid[2, 3] = OBSTACLE
    _named_case("obstacle_collision", grid, [(2, 2), (6, 6)], (RIGHT, STAY),
                exp_pos=[(2, 2), (6, 6)],
                exp_obs=[True, False], exp_rob=[False, False])


def test_case_same_target_conflict():
    # reference test_same_target_conflict: both reverted, both flagged
    _named_case("same_target", FREE7, [(3, 2), (3, 4)], (RIGHT, LEFT),
                exp_pos=[(3, 2), (3, 4)],
                exp_obs=[False, False], exp_rob=[True, True])


def test_case_swap_conflict():
    # reference test_swap_conflict: both reverted, both flagged
    _named_case("swap", FREE7, [(3, 2), (3, 3)], (RIGHT, LEFT),
                exp_pos=[(3, 2), (3, 3)],
                exp_obs=[False, False], exp_rob=[True, True])


def test_case_move_into_stationary():
    # reference test_move_into_stationary: ONLY the mover collides
    _named_case("move_into_stationary", FREE7, [(3, 2), (3, 3)], (RIGHT, STAY),
                exp_pos=[(3, 2), (3, 3)],
                exp_obs=[False, False], exp_rob=[True, False])


def test_case_follow_vacated_cell_allowed():
    # reference test_follow_vacated_cell_allowed: convoy motion is LEGAL
    _named_case("follow_vacated", FREE7, [(3, 2), (3, 3)], (RIGHT, RIGHT),
                exp_pos=[(3, 3), (3, 4)],
                exp_obs=[False, False], exp_rob=[False, False])


# Two extra deterministic corner cases, still differential + end-to-end.

def test_case_edge_clamp_no_flags():
    # off-grid move -> clamped to stay, no flags (== np.clip semantics)
    _named_case("edge_clamp", FREE7, [(0, 0), (6, 6)], (UP, DOWN),
                exp_pos=[(0, 0), (6, 6)],
                exp_obs=[False, False], exp_rob=[False, False])


def test_case_move_into_obstacle_reverted_robot():
    # r1 reverted by an obstacle counts as stationary for the robot masks:
    # r0 moving into r1's cell is reverted + robot-flagged (mover only)
    grid = FREE7.copy()
    grid[3, 4] = OBSTACLE
    _named_case("into_obstacle_reverted", grid, [(3, 2), (3, 3)],
                (RIGHT, RIGHT),
                exp_pos=[(3, 2), (3, 3)],
                exp_obs=[False, True], exp_rob=[True, False])


# ---------------------------------------------------------------------------
# Randomized fuzzing
# ---------------------------------------------------------------------------

def make_config(rng, rows=None, cols=None):
    """One random (grid, two distinct free spawn cells) configuration.

    Half of the configs force the spawns within Chebyshev distance 2 so the
    conflict masks (same-target / swap / into-stationary / convoy) actually
    fire; obstacle density up to 0.35 exercises the obstacle pre-pass.
    """
    rows = int(rng.integers(4, 13)) if rows is None else rows
    cols = int(rng.integers(4, 13)) if cols is None else cols
    density = float(rng.uniform(0.0, 0.35))
    while True:
        grid = (rng.random((rows, cols)) < density).astype(np.int8)
        free = np.argwhere(grid == 0)
        if len(free) >= 2:
            break
    order = rng.permutation(len(free))
    p0 = tuple(int(x) for x in free[order[0]])
    p1 = tuple(int(x) for x in free[order[1]])
    if rng.random() < 0.5:  # adjacency-biased stratum
        near = [tuple(int(x) for x in c) for c in free
                if max(abs(int(c[0]) - p0[0]), abs(int(c[1]) - p0[1])) <= 2
                and tuple(int(x) for x in c) != p0]
        if near:
            p1 = near[int(rng.integers(len(near)))]
    return grid, [p0, p1]


def test_fuzz_multistep_parity():
    """200 fixed-seed configs x 8 lockstep steps; coverage counters guard
    against a vacuous fuzz (all counts are deterministic given the seeds)."""
    n_rob = n_obs = n_convoy = 0
    for seed in MULTISTEP_SEEDS:
        rng = np.random.default_rng(seed)
        grid, cur = make_config(rng)
        for t in range(STEPS_PER_CONFIG):
            actions = [int(a) for a in rng.integers(0, 5, size=2)]
            ctx = "seed=%d t=%d" % (seed, t)
            ref_t, ref_o, ref_r = assert_parity(grid, cur, actions, ctx)
            n_rob += sum(ref_r)
            n_obs += sum(ref_o)
            # convoy: a robot successfully entered the cell the other vacated
            for i, j in ((0, 1), (1, 0)):
                if (ref_t[i] == tuple(cur[j]) and ref_t[j] != tuple(cur[j])
                        and not ref_r[i] and not ref_r[j]):
                    n_convoy += 1
            cur = ref_t
            assert cur[0] != cur[1], "distinct-cell invariant broken at " + ctx
    assert n_rob >= 30, "fuzz never exercised robot conflicts (n=%d)" % n_rob
    assert n_obs >= 100, "fuzz never exercised obstacle reverts (n=%d)" % n_obs
    assert n_convoy >= 3, "fuzz never exercised convoy motion (n=%d)" % n_convoy


def test_fuzz_batched_parity():
    """256 fixed-seed configs stepped in ONE batched (E, ...) call: verifies
    the vectorised env indexing never cross-contaminates envs."""
    grids, curs, acts = [], [], []
    for seed in BATCHED_SEEDS:
        rng = np.random.default_rng(seed)
        grid, cur = make_config(rng, rows=12, cols=12)
        grids.append(grid)
        curs.append([list(p) for p in cur])
        acts.append([int(a) for a in rng.integers(0, 5, size=2)])
    occ = torch.from_numpy(np.stack(grids)).to(torch.int8)
    cur_t = torch.tensor(curs, dtype=torch.int16)          # bank pos dtype
    act_t = torch.tensor(acts, dtype=torch.long)
    lat = torch.zeros((len(grids), 2), dtype=torch.bool)
    nxt, ho, hr = grid_core.step_positions(cur_t, act_t, occ, lat, (12, 12))
    assert nxt.dtype == torch.int16, "nxt must keep cur's dtype"
    assert ho.dtype == torch.bool and hr.dtype == torch.bool
    for e, seed in enumerate(BATCHED_SEEDS):
        ref_t, ref_o, ref_r = ref_step(grids[e], curs[e], acts[e])
        got_t = [tuple(int(x) for x in p) for p in nxt[e]]
        got_o = [bool(x) for x in ho[e]]
        got_r = [bool(x) for x in hr[e]]
        ctx = "batched seed=%d (env %d)" % (seed, e)
        assert got_t == ref_t, "%s: %s != %s" % (ctx, got_t, ref_t)
        assert got_o == ref_o, "%s: obstacle %s != %s" % (ctx, got_o, ref_o)
        assert got_r == ref_r, "%s: robot %s != %s" % (ctx, got_r, ref_r)


def test_fuzz_latched_parity():
    """100 fixed-seed configs with random latch masks.

    Spec 1.11: the env overwrites a latched robot's action to STAY, and a
    latched robot is an immovable blocker.  In reference terms a latched
    robot is therefore exactly a stayer, so parity must hold when both
    sides receive the STAY-forced action stream.
    """
    n_lat_conflicts = 0
    for seed in LATCHED_SEEDS:
        rng = np.random.default_rng(seed)
        grid, cur = make_config(rng)
        latched = [bool(rng.random() < 0.5), bool(rng.random() < 0.5)]
        for t in range(4):
            actions = [int(a) for a in rng.integers(0, 5, size=2)]
            # env contract (spec 1.11): latched robots' actions -> STAY
            actions = [STAY if latched[i] else actions[i] for i in range(2)]
            ref_t, ref_o, ref_r = ref_step(grid, cur, actions)
            core_t, core_o, core_r = core_step(grid, cur, actions,
                                               latched=latched)
            ctx = "latched seed=%d t=%d latched=%s" % (seed, t, latched)
            assert core_t == ref_t, "%s: %s != %s" % (ctx, core_t, ref_t)
            assert core_o == ref_o, "%s: obstacle %s != %s" % (ctx, core_o, ref_o)
            assert core_r == ref_r, "%s: robot %s != %s" % (ctx, core_r, ref_r)
            for i in range(2):
                assert not (latched[i] and core_t[i] != tuple(cur[i])), \
                    "%s: latched robot moved" % ctx
            n_lat_conflicts += sum(
                1 for i in range(2) if core_r[i] and latched[1 - i])
            cur = ref_t
    assert n_lat_conflicts >= 5, (
        "latched fuzz never exercised move-into-latched-robot (n=%d)"
        % n_lat_conflicts)


def test_fuzz_instantiated_env_parity():
    """40 fixed-seed configs rolled lockstep against a REAL instantiated
    RendezvousEnv (not the mirrored proposal helper): positions elementwise
    and per-step collision counts must match grid_core exactly."""
    for seed in ENV_SEEDS:
        rng = np.random.default_rng(seed)
        grid, starts = make_config(rng)
        env = make_ref_env(grid, starts)
        cur = [tuple(p) for p in starts]
        for t in range(STEPS_PER_CONFIG):
            actions = [int(a) for a in rng.integers(0, 5, size=2)]
            obs, _r, term, trunc, info = env.step(actions)
            core_t, core_o, core_r = core_step(grid, cur, actions)
            env_pos = [tuple(int(x) for x in p) for p in obs["robot_positions"]]
            ctx = "env seed=%d t=%d" % (seed, t)
            assert core_t == env_pos, "%s: %s != env %s" % (ctx, core_t, env_pos)
            assert sum(core_o) == info["obstacle_collisions"], ctx
            assert sum(core_r) == info["robot_collisions"], ctx
            assert not term and not trunc, ctx  # threshold=1 forbids the goal
            cur = env_pos


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  " + name)
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("{}/{} tests passed".format(len(tests) - n_fail, len(tests)))
    sys.exit(1 if n_fail else 0)
