"""Unit tests for tasks/team_listen/grid_core.py: reveal, latch, slip.

M1_SPEC coverage:
* 1.9  ``reveal``       -- Chebyshev window-slice fog-of-war (clamping ==
                           the reference's max/min window truncation), split
                           into the two boolean planes of spec 1.3,
                           monotone accumulation, batched isolation.
* 1.11 ``latch_update`` -- absorbing first-latch semantics (slot/time keep
                           their first values), target_valid masking, int
                           and per-env-tensor ``t``; plus the in-kernel
                           guarantee in ``step_positions`` that a latched
                           robot is immovable and a blocking obstacle.
* 1.10 ``apply_slip``   -- NO_SLIP(=5) keeps the policy action, 0..4 force
                           the replacement, padded MAX_AGENTS columns are
                           ignored, inputs are never mutated (pure).

Conflict-resolution parity itself lives in tests/test_conflict_parity.py.

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_grid_core.py`` discovers and runs its own
tests with a pass/fail summary.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import grid_core
from tasks.team_listen.grid_core import (
    NO_SLIP, STAY, LEFT, RIGHT, UP, REVEAL_OFFSETS, apply_slip,
    chebyshev_offsets, latch_update, reveal, step_positions,
)

R = C = 12  # spec 1.3 grid


def _planes(E=1):
    """Fresh all-unknown fog planes (both False == unknown, spec 1.3)."""
    return (torch.zeros((E, R, C), dtype=torch.bool),
            torch.zeros((E, R, C), dtype=torch.bool))


def _window_mask(positions, radius=1, rows=R, cols=C):
    """Reference-style window slice union (env_paper._reveal semantics)."""
    m = torch.zeros((rows, cols), dtype=torch.bool)
    for (x, y) in positions:
        x0, x1 = max(0, x - radius), min(rows, x + radius + 1)
        y0, y1 = max(0, y - radius), min(cols, y + radius + 1)
        m[x0:x1, y0:y1] = True
    return m


# ---------------------------------------------------------------------------
# Constants / offsets
# ---------------------------------------------------------------------------

def test_deltas_match_reference_encoding():
    # bit-identical to RendezvousEnv._MOVES (spec 1.3):
    # 0=up(row-1) 1=down(row+1) 2=left(col-1) 3=right(col+1) 4=stay
    expect = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]]
    assert grid_core.DELTAS.tolist() == expect
    assert grid_core.N_ACTIONS == 5 and NO_SLIP == 5


def test_chebyshev_offsets():
    off1 = chebyshev_offsets(1)
    assert off1.shape == (9, 2)
    # row-major, matching the spec 1.3 (2r+1)^2 window
    assert off1.tolist() == [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0],
                             [0, 1], [1, -1], [1, 0], [1, 1]]
    assert torch.equal(REVEAL_OFFSETS, off1)
    assert chebyshev_offsets(0).tolist() == [[0, 0]]
    assert chebyshev_offsets(2).shape == (25, 2)


# ---------------------------------------------------------------------------
# reveal (spec 1.9)
# ---------------------------------------------------------------------------

def test_reveal_center_window():
    occ = torch.zeros((1, R, C), dtype=torch.int8)
    occ[0, 4, 4] = 1   # obstacle inside the window
    occ[0, 0, 0] = 1   # obstacle far outside the window
    kf, ko = _planes()
    ret = reveal(kf, ko, occ, torch.tensor([[[5, 5]]]), REVEAL_OFFSETS)
    assert ret is None, "reveal is in-place and returns None"
    win = _window_mask([(5, 5)])
    # inside the window: exactly the ground truth, split across the planes
    assert torch.equal(kf[0], win & (occ[0] == 0))
    assert torch.equal(ko[0], win & (occ[0] == 1))
    # outside: still unknown (both planes False) -- (0,0) not leaked
    assert not kf[0][~win].any() and not ko[0][~win].any()


def test_reveal_corner_clamp():
    # clamping duplicates edge cells; revealed set must equal the reference's
    # max/min-truncated window slice, at all four corners
    occ = torch.zeros((1, R, C), dtype=torch.int8)
    occ[0, 0, 1] = 1
    for corner in [(0, 0), (0, C - 1), (R - 1, 0), (R - 1, C - 1)]:
        kf, ko = _planes()
        reveal(kf, ko, occ, torch.tensor([[list(corner)]]), REVEAL_OFFSETS)
        win = _window_mask([corner])
        assert int(win.sum()) == 4, "corner window is 2x2"
        assert torch.equal(kf[0] | ko[0], win), "corner %s" % (corner,)
        assert torch.equal(ko[0], win & (occ[0] == 1))


def test_reveal_matches_window_slice():
    # randomized cross-check vs the reference slicing semantics, fixed seed
    g = torch.Generator().manual_seed(1234)
    for trial in range(50):
        occ = (torch.rand((1, R, C), generator=g) < 0.3).to(torch.int8)
        pos = torch.randint(0, R, (1, 2, 2), generator=g)
        kf, ko = _planes()
        reveal(kf, ko, occ, pos, REVEAL_OFFSETS)
        win = _window_mask([tuple(p.tolist()) for p in pos[0]])
        assert torch.equal(kf[0] | ko[0], win), "trial %d" % trial
        assert torch.equal(kf[0], win & (occ[0] == 0)), "trial %d" % trial
        assert torch.equal(ko[0], win & (occ[0] == 1)), "trial %d" % trial
        assert not (kf[0] & ko[0]).any(), "planes must be disjoint"


def test_reveal_accumulates():
    # fog is monotone: cells revealed at A stay revealed after moving to B
    occ = torch.zeros((1, R, C), dtype=torch.int8)
    kf, ko = _planes()
    reveal(kf, ko, occ, torch.tensor([[[1, 1]]]), REVEAL_OFFSETS)
    snap = kf.clone()
    reveal(kf, ko, occ, torch.tensor([[[9, 9]]]), REVEAL_OFFSETS)
    assert torch.equal(kf & snap, snap), "earlier reveal was lost"
    assert torch.equal(kf[0] | ko[0], _window_mask([(1, 1), (9, 9)]))


def test_reveal_two_agents_one_call():
    # both live agents reveal their windows in a single batched scatter,
    # including overlapping windows (duplicate indices scatter the same value)
    occ = torch.zeros((1, R, C), dtype=torch.int8)
    occ[0, 5, 6] = 1
    kf, ko = _planes()
    reveal(kf, ko, occ, torch.tensor([[[5, 5], [5, 7]]]), REVEAL_OFFSETS)
    win = _window_mask([(5, 5), (5, 7)])
    assert torch.equal(kf[0] | ko[0], win)
    assert bool(ko[0, 5, 6]) and not bool(kf[0, 5, 6])


def test_reveal_batched_isolation():
    # env 0's reveal must not leak into env 1's planes and vice versa
    occ = torch.zeros((2, R, C), dtype=torch.int8)
    occ[1, 2, 2] = 1
    kf, ko = _planes(E=2)
    reveal(kf, ko, occ, torch.tensor([[[1, 1], [1, 3]], [[2, 3], [10, 10]]]),
           REVEAL_OFFSETS)
    assert torch.equal(kf[0] | ko[0], _window_mask([(1, 1), (1, 3)]))
    assert torch.equal(kf[1] | ko[1], _window_mask([(2, 3), (10, 10)]))
    # the obstacle exists only in env 1's occ: env 0 sees free at (2,2)
    assert bool(kf[0, 2, 2]) and not bool(ko[0, 2, 2])
    assert bool(ko[1, 2, 2]) and not bool(kf[1, 2, 2])


def test_reveal_radius_2_window():
    # spec 1.9: r = cfg.lidar_radius; the window scales as (2r+1)^2
    occ = torch.zeros((1, R, C), dtype=torch.int8)
    kf, ko = _planes()
    reveal(kf, ko, occ, torch.tensor([[[6, 6]]]), chebyshev_offsets(2))
    assert torch.equal(kf[0] | ko[0], _window_mask([(6, 6)], radius=2))
    assert int(kf[0].sum()) == 25


# ---------------------------------------------------------------------------
# latch_update (spec 1.11)
# ---------------------------------------------------------------------------

def _latch_state(E=1, N=2, T=3):
    latched = torch.zeros((E, N), dtype=torch.bool)
    latch_slot = torch.full((E, N), -1, dtype=torch.int8)
    latch_time = torch.full((E, N), -1, dtype=torch.int16)
    return latched, latch_slot, latch_time


def test_latch_on_target():
    pos = torch.tensor([[[2, 3], [8, 8]]], dtype=torch.int16)
    targets = torch.tensor([[[0, 0], [2, 3], [11, 11]]], dtype=torch.int16)
    valid = torch.tensor([[True, True, True]])
    latched, slot, time = _latch_state()
    ret = latch_update(pos, targets, valid, latched, slot, time, 7)
    assert ret is None, "latch_update is in-place and returns None"
    assert latched.tolist() == [[True, False]]
    assert slot.tolist() == [[1, -1]], "latch_slot = matching target index"
    assert time.tolist() == [[7, -1]]
    assert slot.dtype == torch.int8 and time.dtype == torch.int16


def test_latch_respects_target_valid():
    # standing on a PADDED (invalid) target slot must not latch (spec 1.3
    # presence masks; padded entity slots are inert)
    pos = torch.tensor([[[2, 3], [8, 8]]])
    targets = torch.tensor([[[2, 3], [5, 5], [0, 0]]])
    valid = torch.tensor([[False, True, True]])
    latched, slot, time = _latch_state()
    latch_update(pos, targets, valid, latched, slot, time, 4)
    assert latched.tolist() == [[False, False]]
    assert slot.tolist() == [[-1, -1]] and time.tolist() == [[-1, -1]]


def test_latch_absorbing():
    # OPEN(2) default: the latch is absorbing -- a later call (even with the
    # robot's cell aliasing another target) never rewrites slot/time
    pos = torch.tensor([[[2, 3], [8, 8]]])
    targets = torch.tensor([[[2, 3], [5, 5], [0, 0]]])
    valid = torch.tensor([[True, True, True]])
    latched, slot, time = _latch_state()
    latch_update(pos, targets, valid, latched, slot, time, 3)
    assert latched.tolist() == [[True, False]] and slot.tolist() == [[0, -1]]
    # same robot now "on" target 1 (hypothetically): nothing may change
    pos2 = torch.tensor([[[5, 5], [8, 8]]])
    latch_update(pos2, targets, valid, latched, slot, time, 9)
    assert latched.tolist() == [[True, False]]
    assert slot.tolist() == [[0, -1]], "first-latch slot overwritten"
    assert time.tolist() == [[3, -1]], "first-latch time overwritten"


def test_latch_no_target_noop():
    pos = torch.tensor([[[6, 6], [7, 7]]])
    targets = torch.tensor([[[2, 3], [5, 5], [0, 0]]])
    valid = torch.tensor([[True, True, False]])
    latched, slot, time = _latch_state()
    latch_update(pos, targets, valid, latched, slot, time, 5)
    assert not latched.any()
    assert slot.tolist() == [[-1, -1]] and time.tolist() == [[-1, -1]]


def test_latch_both_agents_same_step():
    pos = torch.tensor([[[2, 3], [5, 5]]])
    targets = torch.tensor([[[2, 3], [5, 5], [0, 0]]])
    valid = torch.tensor([[True, True, False]])
    latched, slot, time = _latch_state()
    latch_update(pos, targets, valid, latched, slot, time, 11)
    assert latched.tolist() == [[True, True]]
    assert slot.tolist() == [[0, 1]]
    assert time.tolist() == [[11, 11]]


def test_latch_tensor_time_per_env():
    # t may be the per-env episode_length_buf (E,), unsqueezed over agents
    pos = torch.tensor([[[2, 3], [8, 8]], [[9, 9], [5, 5]]])
    targets = torch.tensor([[[2, 3], [5, 5], [0, 0]],
                            [[2, 3], [5, 5], [0, 0]]])
    valid = torch.tensor([[True, True, True], [True, True, True]])
    latched, slot, time = _latch_state(E=2)
    latch_update(pos, targets, valid, latched, slot, time,
                 torch.tensor([13, 42], dtype=torch.int64))
    assert latched.tolist() == [[True, False], [False, True]]
    assert slot.tolist() == [[0, -1], [-1, 1]]
    assert time.tolist() == [[13, -1], [-1, 42]]
    assert time.dtype == torch.int16, "t cast to latch_time's dtype"


def test_latched_robot_immovable_in_step():
    # spec 1.11 in-kernel guarantee (step_positions latch line): even with a
    # raw MOVE action, a latched robot stays put, and it blocks like a
    # stationary robot -- the mover into it is reverted + flagged, the
    # latched robot itself gets no robot flag.
    occ = torch.zeros((1, 5, 5), dtype=torch.int8)
    cur = torch.tensor([[[2, 2], [2, 3]]], dtype=torch.int16)
    latched = torch.tensor([[False, True]])
    # r1 latched with a raw RIGHT action; r0 moves RIGHT into r1's cell
    nxt, ho, hr = step_positions(cur, torch.tensor([[RIGHT, RIGHT]]), occ,
                                 latched, (5, 5))
    assert nxt.tolist() == [[[2, 2], [2, 3]]]
    assert hr.tolist() == [[True, False]], "only the mover is flagged"
    assert ho.tolist() == [[False, False]]
    # attempted swap with a latched robot degenerates to into-stationary
    nxt, ho, hr = step_positions(cur, torch.tensor([[RIGHT, LEFT]]), occ,
                                 latched, (5, 5))
    assert nxt.tolist() == [[[2, 2], [2, 3]]]
    assert hr.tolist() == [[True, False]]
    # a latched robot moving nowhere near anyone: complete no-op, no flags
    far = torch.tensor([[[0, 0], [4, 4]]], dtype=torch.int16)
    nxt, ho, hr = step_positions(far, torch.tensor([[STAY, UP]]), occ,
                                 torch.tensor([[False, True]]), (5, 5))
    assert nxt.tolist() == [[[0, 0], [4, 4]]]
    assert not ho.any() and not hr.any()


# ---------------------------------------------------------------------------
# apply_slip (spec 1.10)
# ---------------------------------------------------------------------------

def test_slip_no_slip_keeps_action():
    actions = torch.tensor([[0, 4], [3, 1]], dtype=torch.int64)
    slip = torch.full((2, 2), NO_SLIP, dtype=torch.uint8)  # bank dtype
    out = apply_slip(actions, slip)
    assert torch.equal(out, actions)
    assert out.dtype == torch.int64


def test_slip_forces_action():
    actions = torch.tensor([[0, 1], [2, 3]])
    slip = torch.tensor([[4, NO_SLIP], [NO_SLIP, 0]], dtype=torch.uint8)
    out = apply_slip(actions, slip)
    assert out.tolist() == [[4, 1], [2, 0]]


def test_slip_ignores_padded_agent_columns():
    # bank rows are (K, ..., MAX_AGENTS=4); trailing padded columns must be
    # ignored at N=2 (spec 1.3 fixed-capacity entity slots)
    actions = torch.tensor([[1, 2]])
    slip = torch.tensor([[NO_SLIP, 3, 0, 0]], dtype=torch.uint8)
    out = apply_slip(actions, slip)
    assert out.shape == (1, 2)
    assert out.tolist() == [[1, 3]], "padded columns leaked into actions"


def test_slip_pure_no_mutation():
    actions = torch.tensor([[0, 1]], dtype=torch.int32)  # skrl int32 path
    slip = torch.tensor([[2, NO_SLIP]], dtype=torch.uint8)
    a_snap, s_snap = actions.clone(), slip.clone()
    out = apply_slip(actions, slip)
    assert out.tolist() == [[2, 1]] and out.dtype == torch.int64
    assert torch.equal(actions, a_snap) and torch.equal(slip, s_snap), \
        "apply_slip must not mutate its inputs"


def test_slip_identical_across_paired_lanes():
    # spec 1.10: slips are keyed by (scenario, stream, t) -- two lanes given
    # the SAME bank row get identical executed actions even when their
    # policies disagree at slipped steps, and differ only through the policy
    # at NO_SLIP steps.  This is the frozen-physics guarantee CSI relies on.
    slip_row = torch.tensor([[0, NO_SLIP], [NO_SLIP, NO_SLIP]],
                            dtype=torch.uint8)
    lane_a = torch.tensor([[1, 2], [3, 4]])
    lane_b = torch.tensor([[4, 2], [3, 4]])          # differs at a slipped step
    out_a, out_b = apply_slip(lane_a, slip_row), apply_slip(lane_b, slip_row)
    assert torch.equal(out_a, out_b), "slipped step must override the policy"
    assert out_a.tolist() == [[0, 2], [3, 4]]


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
