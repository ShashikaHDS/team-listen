"""Unit tests for harness/metrics.py on hand-built records (M1_SPEC 6.1/6.4).

Every divergence and coordination metric is exercised on small hand-built
two-episode tensors whose expected values are computed by hand in the
test body, so a regression in any primitive shows up as a wrong NUMBER,
not just a changed shape:

* truncation to min(T_A, T_B) via the prefix-mask AND (spec 6.4);
* trajectory divergence under all three norms (CSI_traj's object);
* TV distance, factored joint, decision-point masking (CSI_step's
  object, spec 6.1 [FIXED: decision points]);
* Hamming action mismatch (CSI_intent's object);
* realised-outcome codes and the paired flip indicator (CSI_flip's
  object), defined only where BOTH lanes completed (spec 2.3);
* ITT outcome aggregates (spec 2.3 estimands);
* the RESEARCH_PLAN.md coordination metrics: idle time, redundant-action
  rate, deadlock, role stability, cross-agent interference -- including
  every NaN (empty denominator) edge, which must be NaN, never 0.

No env, no bank, no Isaac: pure tensors.

pytest-compatible; standalone: ``python tests/test_metrics.py``.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import metrics as M                     # noqa: E402
from tasks.team_listen import grid_core              # noqa: E402

UP, DOWN, LEFT, RIGHT, STAY = (grid_core.UP, grid_core.DOWN, grid_core.LEFT,
                               grid_core.RIGHT, grid_core.STAY)


def _prefix_active(T, lengths):
    """(T, E) bool prefix mask from per-episode lengths."""
    t = torch.arange(T).unsqueeze(1)
    return t < torch.as_tensor(lengths).unsqueeze(0)


def _isnan(x):
    return math.isnan(float(x))


# ---------------------------------------------------------------------------
# Pairing / truncation
# ---------------------------------------------------------------------------

def test_pair_mask_is_min_truncation():
    active_a = _prefix_active(5, [5, 4])
    active_b = _prefix_active(5, [3, 5])
    m = M.pair_mask(active_a, active_b)
    # common prefix = min(T_A, T_B) per episode (spec 6.4 length control)
    assert torch.equal(m.long().sum(0), torch.tensor([3, 4]))
    assert torch.equal(M.episode_lengths(active_a), torch.tensor([5, 4]))
    # prefix property preserved: mask is contiguous from t = 0
    assert torch.equal(m, _prefix_active(5, [3, 4]))


def test_masked_step_mean_empty_is_nan():
    vals = torch.ones((4, 2))
    mask = _prefix_active(4, [4, 0])
    out = M.masked_step_mean(vals, mask)
    assert float(out[0]) == 1.0
    assert _isnan(out[1])


# ---------------------------------------------------------------------------
# Trajectory-level divergence (hand-built two-episode record)
# ---------------------------------------------------------------------------

def _traj_pair():
    """Two episodes, T = 4, N = 2; hand-computed per-step distances.

    Agent 0 offsets (rows) between lanes: t0 -> (1,0), t1 -> (2,0),
    t2 -> (0,0), t3 -> (1,1).  Agent 1 identical across lanes.
    Per-step agent-mean L1: [0.5, 1.0, 0.0, 1.0].
    """
    T, E, N = 4, 2, 2
    pos_a = torch.zeros((T, E, N, 2), dtype=torch.int16)
    pos_b = pos_a.clone()
    off = torch.tensor([[1, 0], [2, 0], [0, 0], [1, 1]], dtype=torch.int16)
    pos_b[:, :, 0, :] += off.unsqueeze(1)
    active_a = _prefix_active(T, [4, 4])
    active_b = _prefix_active(T, [4, 3])       # ep1 truncates to 3 steps
    return pos_a, pos_b, active_a, active_b


def test_traj_divergence_l1():
    pos_a, pos_b, aa, ab = _traj_pair()
    d = M.traj_divergence(pos_a, pos_b, aa, ab, norm="l1")
    # ep0: (0.5 + 1.0 + 0.0 + 1.0) / 4;  ep1: (0.5 + 1.0 + 0.0) / 3
    assert torch.allclose(d, torch.tensor([0.625, 0.5]))
    # identical lanes -> exactly zero
    z = M.traj_divergence(pos_a, pos_a, aa, ab, norm="l1")
    assert torch.equal(z, torch.zeros(2))


def test_traj_divergence_norms():
    pos_a, pos_b, aa, ab = _traj_pair()
    d_h = M.traj_divergence(pos_a, pos_b, aa, ab, norm="hamming")
    # per-step agent-mean mismatch: [0.5, 0.5, 0.0, 0.5]
    assert torch.allclose(d_h, torch.tensor([1.5 / 4, 1.0 / 3]))
    d_2 = M.traj_divergence(pos_a, pos_b, aa, ab, norm="l2")
    # per-step agent-mean L2: [0.5, 1.0, 0.0, sqrt(2)/2]
    s2 = math.sqrt(2.0) / 2.0
    assert torch.allclose(
        d_2, torch.tensor([(0.5 + 1.0 + 0.0 + s2) / 4, 0.5]))
    try:
        M.traj_divergence(pos_a, pos_b, aa, ab, norm="linf")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown norm accepted")


def test_traj_divergence_empty_prefix_nan():
    pos_a, pos_b, aa, _ = _traj_pair()
    ab = _prefix_active(4, [4, 0])             # ep1 has no common step
    d = M.traj_divergence(pos_a, pos_b, aa, ab, norm="l1")
    assert not _isnan(d[0]) and _isnan(d[1])


# ---------------------------------------------------------------------------
# TV distance / joint / decision-point masking
# ---------------------------------------------------------------------------

def test_tv_distance_known_values():
    p = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0])
    q = torch.tensor([0.0, 0.5, 0.5, 0.0, 0.0])
    assert abs(float(M.tv_distance(p, q)) - 0.5) < 1e-7
    assert float(M.tv_distance(p, p)) == 0.0
    disjoint = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5])
    assert abs(float(M.tv_distance(p, disjoint)) - 1.0) < 1e-7


def test_joint_probs_product():
    # two agents, A = 2: joint of [1,0] x [0.5,0.5] = [0.5, 0.5, 0, 0]
    probs = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])       # (1, N=2, A=2)
    j = M.joint_probs(probs)
    assert torch.allclose(j, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
    assert abs(float(j.sum()) - 1.0) < 1e-7


def test_action_tv_marginal_joint_and_mask():
    T, E, N, A = 2, 2, 2, 5
    big = 50.0
    la = torch.zeros((T, E, N, A))
    lb = torch.zeros((T, E, N, A))
    # step 0: agent 0 flips e0 -> e1 (TV ~= 1), agent 1 unchanged;
    # step 1: identical
    la[0, :, 0, 0] = big
    lb[0, :, 0, 1] = big
    la[0, :, 1, 2] = big
    lb[0, :, 1, 2] = big
    active = _prefix_active(T, [2, 2])
    d_marg = M.action_tv(la, lb, active, active, joint=False)
    # per-step marginal-mean TV: [ (1+0)/2, 0 ] -> mean 0.25
    assert torch.allclose(d_marg, torch.tensor([0.25, 0.25]), atol=1e-4)
    d_joint = M.action_tv(la, lb, active, active, joint=True)
    # joint at step 0: e0 x e2 vs e1 x e2 are disjoint -> TV ~= 1
    assert torch.allclose(d_joint, torch.tensor([0.5, 0.5]), atol=1e-4)
    # decision-point mask: restrict to step 0 only (spec 6.1 [FIXED])
    mask = torch.zeros((T, E), dtype=torch.bool)
    mask[0] = True
    d_dp = M.action_tv(la, lb, active, active, decision_mask=mask)
    assert torch.allclose(d_dp, torch.tensor([0.5, 0.5]), atol=1e-4)
    # empty decision mask -> NaN, never 0
    d_empty = M.action_tv(la, lb, active, active,
                          decision_mask=torch.zeros_like(mask))
    assert _isnan(d_empty[0]) and _isnan(d_empty[1])


def test_action_mismatch():
    T, E, N = 2, 2, 2
    a = torch.tensor([[[UP, STAY], [UP, STAY]],
                      [[DOWN, DOWN], [DOWN, DOWN]]], dtype=torch.int8)
    b = torch.tensor([[[DOWN, STAY], [UP, STAY]],
                      [[DOWN, UP], [DOWN, DOWN]]], dtype=torch.int8)
    assert a.shape == (T, E, N)
    active = _prefix_active(T, [2, 1])
    d = M.action_mismatch(a, b, active, active)
    # ep0: per-step agent-mean [0.5, 0.5] -> 0.5; ep1 (1 step): [0.0] -> 0.0
    assert torch.allclose(d, torch.tensor([0.5, 0.0]))


# ---------------------------------------------------------------------------
# Outcome-level divergence (spec 2.3 / 3.3 codes)
# ---------------------------------------------------------------------------

def test_outcome_codes_and_flip():
    slot_a = torch.tensor([[0, 1], [1, 0]], dtype=torch.int8)
    slot_b = torch.tensor([[1, 0], [1, 0]], dtype=torch.int8)
    ca = M.realised_assignment(slot_a)
    cb = M.realised_assignment(slot_b)
    assert ca.tolist() != cb.tolist()
    assert int(ca[1]) == int(cb[1])            # ep1 assignment unchanged
    # order codes: ep0 robot_0 first (+1), ep1 robot_1 first (-1)
    t = torch.tensor([[3, 7], [5, 2]], dtype=torch.int16)
    assert M.realised_order(t).tolist() == [1, -1]
    # dispatch
    assert torch.equal(M.outcome_codes(slot_a, t, "RoleBinding"), ca)
    assert torch.equal(M.outcome_codes(slot_a, t, "Precedence"),
                       M.realised_order(t))
    try:
        M.outcome_codes(slot_a, t, "Rendezvous")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown variant accepted")
    # flip: defined only where BOTH lanes completed (spec 2.3)
    valid_a = torch.tensor([True, True])
    valid_b = torch.tensor([True, False])
    flip, valid = M.outcome_flip(ca, cb, valid_a, valid_b)
    assert float(flip[0]) == 1.0               # ep0 flipped
    assert _isnan(flip[1]) and valid.tolist() == [True, False]
    # unlatched (-1) encodes distinctly from slot 0
    part = torch.tensor([[0, -1], [0, 1]], dtype=torch.int8)
    codes = M.realised_assignment(part)
    assert int(codes[0]) != int(codes[1])


def test_outcome_rates():
    rec = SimpleNamespace(
        completed=torch.tensor([True, True, False]),
        correct=torch.tensor([True, False, False]),
        outcome=torch.tensor([0, 1, 2], dtype=torch.int8))
    r = M.outcome_rates(rec)
    assert abs(r["completion_rate"] - 2 / 3) < 1e-6
    assert abs(r["p_correct_itt"] - 1 / 3) < 1e-6
    assert abs(r["p_wrong_itt"] - 1 / 3) < 1e-6
    assert abs(r["p_incomplete_itt"] - 1 / 3) < 1e-6
    assert abs(r["y_given_c"] - 0.5) < 1e-6
    # no completed episode -> conditional is NaN, not 0
    rec2 = SimpleNamespace(completed=torch.tensor([False]),
                           correct=torch.tensor([False]),
                           outcome=torch.tensor([2], dtype=torch.int8))
    assert _isnan(M.outcome_rates(rec2)["y_given_c"])


# ---------------------------------------------------------------------------
# Coordination metrics (RESEARCH_PLAN.md definitions fixed in metrics.py)
# ---------------------------------------------------------------------------

def test_prelatch_mask_frames():
    # latch step t == latch_time is the ARRIVAL move: excluded; -1 = never
    active = _prefix_active(4, [4, 2])
    lt = torch.tensor([[2, -1], [0, 1]], dtype=torch.int16)
    pre = M.prelatch_mask(active, lt)
    assert pre[:, 0, 0].tolist() == [True, True, False, False]
    assert pre[:, 0, 1].tolist() == [True, True, True, True]
    assert pre[:, 1, 0].tolist() == [False, False, False, False]
    assert pre[:, 1, 1].tolist() == [True, False, False, False]


def test_idle_fraction():
    T, E, N = 6, 2, 2
    executed = torch.zeros((T, E, N), dtype=torch.int8)
    # ep0 agent0: latch at 3, pre-latch executed [STAY, UP, STAY] -> 2/3
    executed[:, 0, 0] = torch.tensor([STAY, UP, STAY, RIGHT, STAY, STAY])
    # ep0 agent1: never latched, executed [STAY]*4 + [UP]*2 -> 4/6
    executed[:, 0, 1] = torch.tensor([STAY, STAY, STAY, STAY, UP, UP])
    # ep1 agent0: latched at t=0 -> no pre-latch step -> NaN
    # ep1 agent1: latch at 2, pre-latch [UP, UP] -> 0
    executed[:, 1, 1] = torch.tensor([UP, UP, STAY, STAY, STAY, STAY])
    active = _prefix_active(T, [6, 3])
    lt = torch.tensor([[3, -1], [0, 2]], dtype=torch.int16)
    idle = M.idle_fraction(executed, active, lt)
    assert abs(float(idle[0, 0]) - 2 / 3) < 1e-6
    assert abs(float(idle[0, 1]) - 4 / 6) < 1e-6
    assert _isnan(idle[1, 0])
    assert float(idle[1, 1]) == 0.0


def test_redundant_action_fraction():
    T, E, N = 4, 2, 2
    spawn = torch.zeros((E, N, 2), dtype=torch.int16)
    spawn[0, 0] = torch.tensor([5, 5], dtype=torch.int16)
    spawn[0, 1] = torch.tensor([2, 2], dtype=torch.int16)
    spawn[1, 0] = torch.tensor([8, 8], dtype=torch.int16)
    spawn[1, 1] = torch.tensor([9, 9], dtype=torch.int16)
    pos = torch.zeros((T, E, N, 2), dtype=torch.int16)
    executed = torch.full((T, E, N), STAY, dtype=torch.int8)
    # ep0 agent0: blocked UP (t0), ok RIGHT (t1), backtrack LEFT (t2),
    # STAY (t3) -> redundant {t0, t2} of 4 considered = 0.5
    pos[:, 0, 0] = torch.tensor([[5, 5], [5, 6], [5, 5], [5, 5]],
                                dtype=torch.int16)
    executed[:, 0, 0] = torch.tensor([UP, RIGHT, LEFT, STAY])
    # ep0 agent1: clean run of successful moves -> 0.0
    pos[:, 0, 1] = torch.tensor([[3, 2], [4, 2], [5, 2], [6, 2]],
                                dtype=torch.int16)
    executed[:, 0, 1] = torch.tensor([DOWN, DOWN, DOWN, DOWN])
    # ep1 agent0: latch at t=1 -> only t0 considered, a successful move
    pos[:, 1, 0] = torch.tensor([[8, 9], [8, 10], [8, 10], [8, 10]],
                                dtype=torch.int16)
    executed[:, 1, 0] = torch.tensor([RIGHT, RIGHT, STAY, STAY])
    # ep1 agent1: all STAY -> nothing redundant
    pos[:, 1, 1] = spawn[1, 1].unsqueeze(0).expand(T, 2).clone()
    active = _prefix_active(T, [4, 4])
    lt = torch.tensor([[-1, -1], [1, -1]], dtype=torch.int16)

    frac = M.redundant_action_fraction(executed, pos, active, lt,
                                       spawn=spawn)
    assert abs(float(frac[0, 0]) - 0.5) < 1e-6
    assert float(frac[0, 1]) == 0.0
    assert float(frac[1, 0]) == 0.0
    assert float(frac[1, 1]) == 0.0

    # without spawn: t0 not evaluable -> ep0 agent0 counts only the
    # backtrack among {t1, t2, t3} -> 1/3
    frac_ns = M.redundant_action_fraction(executed, pos, active, lt)
    assert abs(float(frac_ns[0, 0]) - 1 / 3) < 1e-6
    # ep1 agent0 without spawn: latch at 1 leaves zero considered steps
    assert _isnan(frac_ns[1, 0])


def test_deadlocked():
    T, E, N = 6, 3, 2
    pos = torch.zeros((T, E, N, 2), dtype=torch.int16)
    # ep0: frozen forever, incomplete -> deadlocked
    # ep1: keeps moving, incomplete -> not deadlocked
    pos[:, 1, 0, 1] = torch.arange(T, dtype=torch.int16)
    # ep2: frozen but completed -> not deadlocked
    active = _prefix_active(T, [6, 6, 4])
    completed = torch.tensor([False, False, True])
    d = M.deadlocked(pos, active, completed, window=8)
    assert d.tolist() == [True, False, False]
    # moving early but frozen in the window -> deadlocked
    pos2 = pos.clone()
    pos2[:, 1, 0, 1] = torch.tensor([0, 1, 2, 3, 3, 3], dtype=torch.int16)
    d2 = M.deadlocked(pos2, active, completed, window=2)
    assert d2.tolist() == [True, True, False]


def test_role_stability():
    T, E, N = 5, 2, 1
    targets = torch.zeros((E, 3, 2), dtype=torch.int16)
    targets[:, 0] = torch.tensor([0, 0], dtype=torch.int16)
    targets[:, 1] = torch.tensor([0, 10], dtype=torch.int16)
    targets[:, 2] = torch.tensor([11, 11], dtype=torch.int16)  # invalid slot
    tv = torch.tensor([[True, True, False], [True, True, False]])
    pos = torch.zeros((T, E, N, 2), dtype=torch.int16)
    # ep0: nearest target sequence 0,0,1,1,0 -> 2 switches / 4 transitions
    pos[:, 0, 0, 1] = torch.tensor([1, 2, 8, 9, 3], dtype=torch.int16)
    # ep1: constant nearest -> stability 1
    pos[:, 1, 0, 1] = torch.tensor([1, 1, 2, 1, 2], dtype=torch.int16)
    active = _prefix_active(T, [5, 5])
    s = M.role_stability(pos, targets, tv, active)
    assert abs(float(s[0]) - 0.5) < 1e-6
    assert float(s[1]) == 1.0
    # single-step episode: no transition -> NaN
    s1 = M.role_stability(pos, targets, tv, _prefix_active(T, [1, 5]))
    assert _isnan(s1[0])


def test_interference_rate():
    counts = torch.tensor([[2, 1], [0, 0]], dtype=torch.int16)
    active = _prefix_active(6, [6, 4])
    r = M.interference_rate(counts, active)
    assert abs(float(r[0]) - 3 / 12) < 1e-6
    assert float(r[1]) == 0.0


def test_coordination_summary_duck_record():
    T, E, N = 4, 2, 2
    rec = SimpleNamespace(
        positions=torch.zeros((T, E, N, 2), dtype=torch.int16),
        executed=torch.full((T, E, N), STAY, dtype=torch.int8),
        active=_prefix_active(T, [4, 4]),
        latch_time=torch.full((E, N), -1, dtype=torch.int16),
        completed=torch.tensor([False, False]),
        correct=torch.tensor([False, False]),
        outcome=torch.tensor([2, 2], dtype=torch.int8),
        n_robot_collisions=torch.zeros((E, N), dtype=torch.int16),
    )
    targets = torch.zeros((E, 3, 2), dtype=torch.int16)
    tvalid = torch.tensor([[True, True, False]] * E)
    out = M.coordination_summary(rec, targets=targets, target_valid=tvalid)
    # frozen, incomplete, all-stay episodes
    assert out["deadlock_rate"] == 1.0
    assert out["idle_time"] == 1.0
    assert out["redundant_action_rate"] == 0.0
    assert out["interference_rate"] == 0.0
    assert out["role_stability"] == 1.0
    assert out["completion_rate"] == 0.0
    assert out["p_incomplete_itt"] == 1.0
    assert _isnan(out["assignment_accuracy"])
    # role_stability omitted (not zero-filled) without targets
    out2 = M.coordination_summary(rec)
    assert "role_stability" not in out2


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    import unittest

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  " + name)
        except unittest.SkipTest as exc:
            n_skip += 1
            print("SKIP  {}  ({})".format(name, exc))
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("{} passed, {} failed, {} skipped".format(
        len(tests) - n_fail - n_skip, n_fail, n_skip))
    sys.exit(1 if n_fail else 0)
