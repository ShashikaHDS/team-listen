"""Tests for harness/reward_audit.py on a tiny hand-built scenario bank.

M1_SPEC coverage (1.12 [FIXED: reward-fairness bound], 3.4, section 7):

* the exact bank-computed compliance bound -- on two hand-built rows (one
  RoleBinding, one Precedence) whose latch-aware BFS distances are small
  integers counted by hand, the audit's per-class ``G_comply - G_defect``
  and the ``min`` bound must equal plain-Python arithmetic done here with
  explicit loop sums (independent of the module's closed forms);
* exactness against the TRUE reward -- the audit's G_comply for the
  RoleBinding row equals the discounted return of the compliant plan
  rolled step by step through ``grid_core`` + ``rewards`` (which also
  validates the telescoped-shaping identity ``sum == lambda * M0`` the
  hand formula relies on);
* end-to-end from a bank FILE through the production loader
  (``compliance_bound_file`` -> ``scenario_bank.load_bank``);
* the ``assert_compliance_bound`` margin gate raises exactly when
  ``bound <= margin``;
* lambda cancels in the bound (shaping telescopes identically on both
  sides) while gamma is respected;
* model-precondition refusals: unreachable spawn, != 2 valid stations,
  precedence rows whose two station distances disagree (mouth-transit
  violation), and ``delta_gap`` drift against the recomputed fields.

Hand-derived geometry (all distances counted on paper):

RoleBinding row -- free cells are rows 5-6 x cols 1..10; stations
t0=(5,1) (LEFT), t1=(5,10); spawns r0=(6,3), r1=(6,8):
    d(r0,t0)=3  d(r0,t1)=8  d(r1,t0)=8  d(r1,t1)=3
    T(r0->t0, r1->t1) = 3;  T(swapped) = 8;  M0 = 6.

Precedence row -- unique-mouth airlock: mouth m=(5,5); leaf stations
t0=(4,5), t1=(6,5); corridor (5,2),(5,3),(5,4); open block rows 4-6 x
cols 0-1; spawns r0=(5,1), r1=(4,0):
    d(spawn_i, either station) = dm_i + 1;  dm_0 = 4, dm_1 = 6
    T(r0 first) = max(6, 4+1)+1 = 7;  T(r1 first) = max(4, 6+1)+1 = 8
    M0 = 5 + 7 = 12;  delta_gap = dm_0 - dm_1 = -2.

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_reward_audit.py`` discovers and runs its
own tests with a pass/fail/skip summary.
"""

import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import reward_audit as ra
from tasks.team_listen import grid_core
from tasks.team_listen import obs_layout as L
from tasks.team_listen import rewards
from tasks.team_listen import scenario_bank

GAMMA = 0.99
LAM = 0.1


# ---------------------------------------------------------------------------
# Independent reference pieces: tiny BFS + loop-sum return arithmetic
# ---------------------------------------------------------------------------

def _bfs(occ, src, blocked):
    """(R, C) int16 4-neighbour BFS distances over free cells, -1 where
    unreachable; ``blocked`` cells are treated as obstacles (latch-aware)."""
    dist = torch.full((L.R, L.C), -1, dtype=torch.int16)
    if int(occ[src]) != 0 or src in blocked:
        return dist
    dist[src] = 0
    queue = deque([src])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < L.R and 0 <= nc < L.C and int(dist[nr, nc]) < 0
                    and int(occ[nr, nc]) == 0 and (nr, nc) not in blocked):
                dist[nr, nc] = int(dist[r, c]) + 1
                queue.append((nr, nc))
    return dist


def _g_hand(T, B, M0, gamma=GAMMA, lam=LAM, latch=None):
    """Hand return: explicit loop sum of step costs + telescoped shaping
    (== lam * M0, verified empirically by the rolled-trajectory test) +
    discounted terminal bonus + (DECISIONS.md first-latch amendment) the
    per-agent MEAN of the discounted +2.0 first-latch bonuses at the
    1-based latch steps ``latch`` -- G stays in per-agent units.
    Independent of reward_audit's closed form."""
    steps = sum(-0.01 * gamma ** t for t in range(T))
    latch_term = (sum(2.0 * gamma ** (tau - 1) for tau in latch) / len(latch)
                  if latch else 0.0)
    return steps + lam * M0 + gamma ** (T - 1) * B + latch_term


def _delta_hand(gc, gd, latch_c, latch_d, gamma=GAMMA):
    """Per-agent-tightest delta (amendment convention): the broadcast
    difference plus the WORST single agent's latch-bonus difference."""
    mean_c = sum(2.0 * gamma ** (t - 1) for t in latch_c) / len(latch_c)
    mean_d = sum(2.0 * gamma ** (t - 1) for t in latch_d) / len(latch_d)
    per_agent = [2.0 * (gamma ** (tc - 1) - gamma ** (td - 1))
                 for tc, td in zip(latch_c, latch_d)]
    return (gc - mean_c) - (gd - mean_d) + min(per_agent)


# The four hand deltas (class 0 = RB0/PR0 "robot_0 -> left / first").
# Latch pairs by hand from the rows below: RB assignment A0 latches (3, 3),
# A1 latches (8, 8); PR class 0 latches (5, 7) (r0 through the mouth first),
# class 1 latches (8, 7).  The latch term does NOT cancel in the deltas:
# the compliant plan latches later, so the amendment lowers every bound.
RB_DELTA = (_delta_hand(_g_hand(3, 12.0, 6, latch=(3, 3)),   # RB0 comply T=3
                        _g_hand(8, 2.0, 6, latch=(8, 8)),
                        (3, 3), (8, 8)),
            _delta_hand(_g_hand(8, 12.0, 6, latch=(8, 8)),   # RB1 comply T=8
                        _g_hand(3, 2.0, 6, latch=(3, 3)),
                        (8, 8), (3, 3)))
PR_DELTA = (_delta_hand(_g_hand(7, 12.0, 12, latch=(5, 7)),  # PR0 comply T=7
                        _g_hand(8, 2.0, 12, latch=(8, 7)),
                        (5, 7), (8, 7)),
            _delta_hand(_g_hand(8, 12.0, 12, latch=(8, 7)),  # PR1 comply T=8
                        _g_hand(7, 2.0, 12, latch=(5, 7)),
                        (8, 7), (5, 7)))


# ---------------------------------------------------------------------------
# Tiny hand-built bank rows (schema of spec 1.10, verified by the loader)
# ---------------------------------------------------------------------------

def _rb_row():
    occ = torch.ones((L.R, L.C), dtype=torch.uint8)
    occ[5, 1:11] = 0
    occ[6, 1:11] = 0
    return {"occ": occ, "spawns": [(6, 3), (6, 8)],
            "targets": [(5, 1), (5, 10)], "mouth": (-1, -1), "delta_gap": 0}


def _pr_row():
    occ = torch.ones((L.R, L.C), dtype=torch.uint8)
    free = [(4, 5), (6, 5), (5, 5),                    # stations + mouth
            (5, 4), (5, 3), (5, 2),                    # width-1 corridor
            (4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1)]   # spawn block
    for cell in free:
        occ[cell] = 0
    return {"occ": occ, "spawns": [(5, 1), (4, 0)],
            "targets": [(4, 5), (6, 5)], "mouth": (5, 5), "delta_gap": -2}


def _dist_field(row):
    """(MAX_TARGETS, R, C) int16 latch-aware BFS field for one row."""
    df = torch.full((L.MAX_TARGETS, L.R, L.C), -1, dtype=torch.int16)
    targets = row["targets"]
    for j, tgt in enumerate(targets):
        df[j] = _bfs(row["occ"], tgt, {targets[1 - j]})
    return df


def _payload(rows):
    """Full spec 1.10 payload dict for ``torch.save`` (loader-checkable)."""
    k = len(rows)
    spawn = torch.zeros((k, L.MAX_AGENTS, 2), dtype=torch.int16)
    target = torch.zeros((k, L.MAX_TARGETS, 2), dtype=torch.int16)
    tv = torch.zeros((k, L.MAX_TARGETS), dtype=torch.bool)
    df = torch.stack([_dist_field(r) for r in rows])
    mouth = torch.tensor([r["mouth"] for r in rows], dtype=torch.int16)
    for i, r in enumerate(rows):
        spawn[i, :2] = torch.tensor(r["spawns"], dtype=torch.int16)
        target[i, :2] = torch.tensor(r["targets"], dtype=torch.int16)
        tv[i, :2] = True
    return {
        "occ": torch.stack([r["occ"] for r in rows]),
        "spawn": spawn,
        "spawn_alt": spawn.clone(),
        "target": target,
        "target_valid": tv,
        "dist_field": df,
        "mouth": mouth,
        "delta_gap": torch.tensor([r["delta_gap"] for r in rows],
                                  dtype=torch.int8),
        "leak_bit": torch.zeros((k,), dtype=torch.uint8),
        "instr_switch_time": torch.full((k,), -1, dtype=torch.int16),
        "slip": torch.full((k, scenario_bank.N_STREAMS, L.T_DECISION,
                            L.MAX_AGENTS), grid_core.NO_SLIP,
                           dtype=torch.uint8),
        "split": torch.zeros((k,), dtype=torch.uint8),
    }


def _bank(rows):
    """In-memory ScenarioBank (bypasses the file loader; the file path is
    exercised by ``test_bound_from_bank_file``)."""
    payload = _payload(rows)
    return scenario_bank.ScenarioBank(
        meta={}, sha256="", path="<in-memory>", **payload)


# ---------------------------------------------------------------------------
# The hand-counted BFS distances hold (grounds "derivable by hand")
# ---------------------------------------------------------------------------

def test_bfs_distances_match_hand_counts():
    df_rb = _dist_field(_rb_row())
    assert int(df_rb[0][6, 3]) == 3 and int(df_rb[1][6, 3]) == 8
    assert int(df_rb[0][6, 8]) == 8 and int(df_rb[1][6, 8]) == 3
    # stations are 0 to themselves; the OTHER station is blocked out (-1)
    assert int(df_rb[0][5, 1]) == 0 and int(df_rb[0][5, 10]) == -1

    df_pr = _dist_field(_pr_row())
    # both stations sit behind the unique mouth: equal distances everywhere
    for spawn, dist in (((5, 1), 5), ((4, 0), 7)):
        assert int(df_pr[0][spawn]) == dist
        assert int(df_pr[1][spawn]) == dist
    assert int(df_pr[0][5, 5]) == 1                    # mouth adjacency


# ---------------------------------------------------------------------------
# RoleBinding row: bound == hand arithmetic
# ---------------------------------------------------------------------------

def test_rb_row_bound_matches_hand_arithmetic():
    audit = ra.compliance_bound(_bank([_rb_row()]))
    assert audit.k == 1 and not bool(audit.is_precedence[0])
    assert audit.t_comply[0].tolist() == [3, 8]        # RB0 fast, RB1 slow
    assert audit.t_defect[0].tolist() == [8, 3]
    assert float(audit.matching0[0]) == 6.0
    for cls in (0, 1):
        assert abs(float(audit.delta[0, cls]) - RB_DELTA[cls]) < 1e-9
        t_c, t_d = (3, 8) if cls == 0 else (8, 3)
        assert abs(float(audit.g_comply[0, cls])
                   - _g_hand(t_c, 12.0, 6, latch=(t_c, t_c))) < 1e-9
        assert abs(float(audit.g_defect[0, cls])
                   - _g_hand(t_d, 2.0, 6, latch=(t_d, t_d))) < 1e-9
    assert abs(audit.bound - min(RB_DELTA)) < 1e-9
    assert audit.argmin_class == 1                     # comply-slow is tighter
    assert audit.class_label(0, 1) == "RB1"
    # compliance is strictly optimal on both classes (spec 3.4)
    assert min(RB_DELTA) > 0


# ---------------------------------------------------------------------------
# Precedence row: bound == hand arithmetic
# ---------------------------------------------------------------------------

def test_pr_row_bound_matches_hand_arithmetic():
    audit = ra.compliance_bound(_bank([_pr_row()]))
    assert audit.k == 1 and bool(audit.is_precedence[0])
    assert audit.t_comply[0].tolist() == [7, 8]        # |dt| >= 1 structural
    assert audit.t_defect[0].tolist() == [8, 7]
    assert float(audit.matching0[0]) == 12.0
    for cls in (0, 1):
        assert abs(float(audit.delta[0, cls]) - PR_DELTA[cls]) < 1e-9
    assert abs(audit.bound - min(PR_DELTA)) < 1e-9
    assert audit.argmin_class == 1
    assert audit.class_label(0, 0) == "PR0"


# ---------------------------------------------------------------------------
# Exactness against the TRUE reward: roll the compliant RB plan
# ---------------------------------------------------------------------------

def test_compliant_return_matches_rolled_trajectory():
    """audit.g_comply[RB, RB0] == the discounted return of the compliant
    plan stepped through grid_core + rewards -- ties the closed form (and
    its telescoped-shaping term ``lambda * M0``) to the implemented reward.
    """
    row = _rb_row()
    bank = _bank([row])
    audit = ra.compliance_bound(bank)

    occ = bank.occ.to(torch.int8)                      # (1, R, C)
    df = bank.dist_field                               # (1, 3, R, C)
    tv = bank.target_valid                             # (1, 3)
    tgt = bank.target                                  # (1, 3, 2)
    pos = bank.spawn[:, :2].clone()                    # (1, 2, 2) int16
    latched = torch.zeros((1, 2), dtype=torch.bool)
    latch_slot = torch.full((1, 2), -1, dtype=torch.int8)
    latch_time = torch.full((1, 2), -1, dtype=torch.int16)

    lf, rt, up = grid_core.LEFT, grid_core.RIGHT, grid_core.UP
    plan = [(lf, rt), (lf, rt), (up, up)]              # r0->(5,1), r1->(5,10)

    phi_prev = grid_core.matching_potential(df, pos, tv)
    assert float(phi_prev[0]) == -6.0                  # Phi_0 = -M0
    ret = 0.0
    for t, (a0, a1) in enumerate(plan):
        act = torch.tensor([[a0, a1]], dtype=torch.long)
        pos, hit_obs, hit_rob = grid_core.step_positions(
            pos, act, occ, latched, (L.R, L.C))
        assert not bool(hit_obs.any()) and not bool(hit_rob.any())
        prev_latched = latched.clone()
        grid_core.latch_update(pos, tgt, tv, latched, latch_slot,
                               latch_time, t)
        newly = latched & ~prev_latched
        done = latched.all(dim=1)
        rew, phi_prev = rewards.compute_rewards(
            ("robot_0", "robot_1"), df, pos, tv, phi_prev, done,
            correct=torch.ones(1), hit_obstacle=hit_obs, hit_robot=hit_rob,
            newly_latched=newly)
        # per-team scalar broadcast: both agents identical (no collisions)
        assert torch.equal(rew["robot_0"], rew["robot_1"])
        ret += (GAMMA ** t) * float(rew["robot_0"][0])

    assert bool(latched.all())                         # T = 3, Y = 1 (RB0)
    assert latch_time.tolist() == [[2, 2]]
    assert abs(ret - float(audit.g_comply[0, 0])) < 1e-4   # float32 rewards
    assert abs(ret - _g_hand(3, 12.0, 6, latch=(3, 3))) < 1e-4


# ---------------------------------------------------------------------------
# End-to-end from a bank FILE through the production loader
# ---------------------------------------------------------------------------

def test_bound_from_bank_file():
    rows = [_rb_row(), _pr_row()]
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "tiny_bank_for_reward_audit.pt")
        torch.save(_payload(rows), path)
        audit = ra.compliance_bound_file(path)
    assert audit.k == 2
    assert audit.is_precedence.tolist() == [False, True]
    hand = [RB_DELTA, PR_DELTA]
    for k in range(2):
        for cls in (0, 1):
            assert abs(float(audit.delta[k, cls]) - hand[k][cls]) < 1e-9
    expect = min(RB_DELTA + PR_DELTA)                  # == RB_DELTA[1]
    assert abs(audit.bound - expect) < 1e-9
    assert audit.argmin_row == 0 and audit.argmin_class == 1
    text = audit.summary()
    assert "bound" in text and "RoleBinding" in text and "Precedence" in text


# ---------------------------------------------------------------------------
# The margin gate
# ---------------------------------------------------------------------------

def test_assert_compliance_bound_gates_on_margin():
    bank = _bank([_rb_row()])
    lo = min(RB_DELTA)                                 # ~= 9.177
    audit = ra.assert_compliance_bound(bank, margin=lo - 0.1)   # passes
    assert abs(audit.bound - lo) < 1e-9
    try:
        ra.assert_compliance_bound(bank, margin=lo + 0.1)
    except RuntimeError as exc:
        assert "compliance bound" in str(exc)
    else:
        raise AssertionError("bound <= margin must raise")


# ---------------------------------------------------------------------------
# Lambda cancels in the bound; gamma is respected
# ---------------------------------------------------------------------------

def test_lambda_cancels_and_gamma_respected():
    bank = _bank([_rb_row()])
    a_lam0 = ra.compliance_bound(bank, shaping_lambda=0.0)
    a_lam1 = ra.compliance_bound(bank, shaping_lambda=0.1)
    # shaping telescopes identically on both sides: delta is lambda-free
    assert torch.allclose(a_lam0.delta, a_lam1.delta, atol=1e-12, rtol=0)
    # ...but the absolute returns shift by exactly lambda * M0 = 0.1 * 6
    assert torch.allclose(a_lam1.g_comply - a_lam0.g_comply,
                          torch.full((1, 2), 0.6, dtype=torch.float64),
                          atol=1e-12, rtol=0)
    # gamma flows through to the hand formula
    a_g95 = ra.compliance_bound(bank, gamma=0.95)
    expect = (_g_hand(8, 12.0, 6, gamma=0.95, latch=(8, 8))
              - _g_hand(3, 2.0, 6, gamma=0.95, latch=(3, 3)))
    assert abs(float(a_g95.delta[0, 1]) - expect) < 1e-9


# ---------------------------------------------------------------------------
# Model-precondition refusals
# ---------------------------------------------------------------------------

def _expect_refusal(bank, needle):
    try:
        ra.compliance_bound(bank)
    except RuntimeError as exc:
        assert needle in str(exc), \
            "wrong refusal message: %s (wanted %r)" % (exc, needle)
    else:
        raise AssertionError("must refuse: %s" % needle)


def test_unreachable_spawn_refused():
    bank = _bank([_rb_row()])
    bank.dist_field[0, 0, 6, 3] = -1                   # r0 spawn cut off
    _expect_refusal(bank, "unreachable")


def test_three_valid_targets_refused():
    bank = _bank([_rb_row()])
    bank.target_valid[0, 2] = True                     # M1 closed form is 2-station
    _expect_refusal(bank, "2 valid stations")


def test_precedence_station_distance_mismatch_refused():
    bank = _bank([_pr_row()])
    bank.dist_field[0, 1, 5, 1] += 1                   # breaks mouth-transit equality
    _expect_refusal(bank, "unique mouth")


def test_delta_gap_drift_refused():
    row = _pr_row()
    row["delta_gap"] = 0                               # truth is -2
    _expect_refusal(_bank([row]), "delta_gap")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

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
