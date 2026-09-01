"""Unit tests for tasks/team_listen/rewards.py on hand-built tensors.

M1_SPEC coverage (section 1.12 and its [FIXED] notes):

* term values      -- step cost -0.01, collisions -0.25/robot/term,
                      completion +2.0, outcome +10*Y, blind expectation
                      +5.0; constants must match the frozen copies in
                      fleet_env.py and the cfg defaults (no drift).
* shaping form     -- gamma-correct ``lambda * (gamma*Phi_t - Phi_{t-1})``
                      [FIXED: not policy-invariant]: differs from the
                      naive difference by exactly ``lambda*(1-gamma)*Phi_t``
                      and telescopes (discounted) to
                      ``lambda*(gamma^T Phi_T - Phi_0)``, path-independent.
* terminal Phi     -- ``Phi(terminal) == 0`` on the both-latched step only.
* latch-aware Phi  -- [FIXED: latch-aware distance field]: shaping over the
                      latch-aware fields penalises stepping toward the
                      wrong station and flags pass-through-trap states as
                      infeasible, where the all-free field would have
                      REWARDED marching into the wrong station.
* assembly         -- per-team scalar broadcast to both agents + per-agent
                      collision terms, dict keyed in possible_agents order;
                      bitwise agreement with fleet_env._get_rewards'
                      inline formula on the same inputs.
* compliance bound -- the spec 1.12/3.4 sanity case: with the telescoping
                      shaping, the discounted return of an optimal
                      compliant plan strictly exceeds the optimal
                      defecting plan's (the +10 outcome bonus dominates
                      the extra step costs and discounting), and both
                      rolled returns match the closed form
                      ``STEP_COST*annuity(T) - lambda*Phi_0
                        + gamma^(T-1)*(COMPLETION + bonus)`` exactly.

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_rewards.py`` discovers and runs its own
tests with a pass/fail summary.
"""

import sys
from collections import deque
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import fleet_env as fe
from tasks.team_listen import grid_core, rewards
from tasks.team_listen.fleet_env_cfg import TeamGridEnvCfg

AGENTS = ("robot_0", "robot_1")
LAM = rewards.SHAPING_LAMBDA
GAMMA = rewards.REWARD_GAMMA
MAX_TARGETS = 3


# ---------------------------------------------------------------------------
# Hand-built BFS distance fields (independent reference implementation)
# ---------------------------------------------------------------------------

def _bfs_field(free, src, blocked):
    """int16 (R, C) BFS over 4-connected free cells; -1 = unreachable."""
    rows, cols = free.shape
    dist = torch.full((rows, cols), -1, dtype=torch.int16)
    passable = free.clone()
    for (br, bc) in blocked:
        passable[br, bc] = False
    sr, sc = int(src[0]), int(src[1])
    if not bool(passable[sr, sc]):
        return dist
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        d = int(dist[r, c]) + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and bool(passable[nr, nc]) \
                    and int(dist[nr, nc]) < 0:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


def _fields(free, t_cells, valid, latch_aware=True):
    """(1, MAX_TARGETS, R, C) int16 field stack over ``free``.

    latch_aware=True blocks every OTHER valid target cell (spec 1.10);
    False gives the all-free field the [FIXED] note diagnoses as broken.
    Invalid slots stay all -1 (masked out by target_valid anyway).
    """
    rows, cols = free.shape
    out = torch.full((1, MAX_TARGETS, rows, cols), -1, dtype=torch.int16)
    for j in range(MAX_TARGETS):
        if not valid[j]:
            continue
        others = [t_cells[k] for k in range(MAX_TARGETS)
                  if k != j and valid[k]] if latch_aware else []
        out[0, j] = _bfs_field(free, t_cells[j], others)
    return out


def _corridor_free(rows=3, cols=7):
    """Width-1 free corridor along row 1 (rows 0/2 are obstacles)."""
    free = torch.zeros((rows, cols), dtype=torch.bool)
    free[1, :] = True
    return free


def _pos(p0, p1):
    """(1, 2, 2) int16 position tensor for the two robots."""
    return torch.tensor([[list(p0), list(p1)]], dtype=torch.int16)


_VALID2 = torch.tensor([[True, True, False]])


# ---------------------------------------------------------------------------
# Constants (spec 1.12 table; no drift vs fleet_env / cfg)
# ---------------------------------------------------------------------------

def test_constants_match_spec():
    assert rewards.STEP_COST == -0.01
    assert rewards.COLLISION_COST == -0.25
    assert rewards.COMPLETION_BONUS == 2.0
    assert rewards.OUTCOME_BONUS == 10.0
    assert rewards.BLIND_EXPECTED_BONUS == 5.0
    assert rewards.SHAPING_LAMBDA == 0.1
    assert rewards.REWARD_GAMMA == 0.99


def test_constants_match_fleet_env_and_cfg():
    # rewards.py cannot import fleet_env (purity-pinned import surface), so
    # the frozen values are stated twice; this is the anti-drift check the
    # rewards.py docstring promises.
    assert rewards.STEP_COST == fe.STEP_COST
    assert rewards.COLLISION_COST == fe.COLLISION_COST
    assert rewards.COMPLETION_BONUS == fe.COMPLETION_BONUS
    assert rewards.OUTCOME_BONUS == fe.OUTCOME_BONUS
    assert rewards.BLIND_EXPECTED_BONUS == fe.BLIND_EXPECTED_BONUS
    assert rewards.FIRST_LATCH_BONUS == fe.FIRST_LATCH_BONUS
    # lambda/gamma defaults must equal the cfg defaults (spec 1.12; gamma
    # must in turn equal the YAML discount_factor 0.99).
    assert rewards.SHAPING_LAMBDA == TeamGridEnvCfg.shaping_lambda
    assert rewards.REWARD_GAMMA == TeamGridEnvCfg.reward_gamma


# ---------------------------------------------------------------------------
# Shaping term (spec 1.12 [FIXED: not policy-invariant])
# ---------------------------------------------------------------------------

def test_shaping_gamma_correct_form():
    phi_now = torch.tensor([-3.0, 0.0, -7.5])
    phi_prev = torch.tensor([-5.0, -1.0, -7.5])
    s = rewards.shaping_reward(phi_now, phi_prev)
    expect = LAM * (GAMMA * phi_now - phi_prev)
    assert torch.allclose(s, expect)
    assert abs(float(s[0]) - 0.203) < 1e-6
    # the naive undiscounted difference deviates by exactly
    # lambda*(1-gamma)*Phi_t -- the anti-waiting confound of spec 3.4
    naive = LAM * (phi_now - phi_prev)
    assert torch.allclose(s - naive, -LAM * (1.0 - GAMMA) * phi_now, atol=1e-7)
    # custom lambda/gamma are honoured
    s2 = rewards.shaping_reward(phi_now, phi_prev, 0.2, 0.5)
    assert torch.allclose(s2, 0.2 * (0.5 * phi_now - phi_prev))


def test_terminal_potential_zeroing():
    phi = torch.tensor([-3.0, -4.0, -1.0])
    done = torch.tensor([False, True, False])
    out = rewards.terminal_potential(phi, done)
    assert out.tolist() == [-3.0, 0.0, -1.0]
    # on the terminal step the shaping collapses to -lambda * Phi_{t-1}
    s = rewards.shaping_reward(out, torch.tensor([-2.0, -2.0, -2.0]))
    assert abs(float(s[1]) - (-LAM * -2.0)) < 1e-7


def test_shaping_telescopes_path_independent():
    # discounted sum over any Phi path == lambda*(gamma^T Phi_T - Phi_0):
    # different random interiors with shared endpoints give one total.
    g = torch.Generator().manual_seed(7)
    T = 10
    phi0, phiT = -6.0, 0.0
    totals = []
    for _ in range(3):
        mids = (-8.0 * torch.rand((T - 1,), generator=g)).tolist()
        seq = [phi0] + mids + [phiT]
        total, disc = 0.0, 1.0
        for t in range(1, T + 1):
            s = rewards.shaping_reward(torch.tensor([seq[t]]),
                                       torch.tensor([seq[t - 1]]))
            total += disc * float(s[0])
            disc *= GAMMA
        totals.append(total)
    expect = LAM * (GAMMA ** T * phiT - phi0)
    for total in totals:
        assert abs(total - expect) < 1e-5, (total, expect)


# ---------------------------------------------------------------------------
# Terminal bonuses and collision terms
# ---------------------------------------------------------------------------

def test_completion_reward_gating():
    done = torch.tensor([True, True, False, False])
    out = rewards.completion_reward(done)
    assert out.tolist() == [2.0, 2.0, 0.0, 0.0]
    assert out.dtype == torch.float32


def test_outcome_reward_modes():
    done = torch.tensor([True, True, False, False])
    correct = torch.tensor([True, False, True, False])
    # stochastic +10*Y (every arm but Blind/Mute, all Leaky cells included)
    out = rewards.outcome_reward(done, correct)
    assert out.tolist() == [10.0, 0.0, 0.0, 0.0]
    # Blind/Mute: exact conditional expectation +5.0, Y ignored (may be None)
    exp = rewards.outcome_reward(done, None, use_expected=True)
    assert exp.tolist() == [5.0, 5.0, 0.0, 0.0]
    exp2 = rewards.outcome_reward(done, correct, use_expected=True)
    assert torch.equal(exp, exp2), "use_expected must ignore Y entirely"
    # missing Y without use_expected is a hard error, not silent zeros
    try:
        rewards.outcome_reward(done, None)
    except AssertionError:
        pass
    else:
        raise AssertionError("outcome_reward accepted correct=None")


def test_collision_reward_values():
    ho = torch.tensor([[True, False], [False, False], [True, True]])
    hr = torch.tensor([[True, True], [False, False], [False, True]])
    out = rewards.collision_reward(ho, hr)
    assert out.tolist() == [[-0.5, -0.25], [0.0, 0.0], [-0.25, -0.5]]
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Assembly: team scalar broadcast + per-agent collision terms
# ---------------------------------------------------------------------------

def test_per_agent_dict_assembly():
    team = torch.tensor([1.0, -2.0])
    ho = torch.tensor([[True, False], [False, False]])
    hr = torch.tensor([[False, False], [False, True]])
    out = rewards.per_agent_rewards(AGENTS, team, ho, hr)
    assert list(out.keys()) == list(AGENTS), "dict must keep agent order"
    assert out["robot_0"].tolist() == [0.75, -2.0]
    assert out["robot_1"].tolist() == [1.0, -2.25]
    for name in AGENTS:
        assert out[name].dtype == torch.float32
    # the team scalar is SHARED: rewards differ only by collision terms
    coll = rewards.collision_reward(ho, hr)
    assert torch.allclose(out["robot_0"] - coll[:, 0],
                          out["robot_1"] - coll[:, 1])


def test_compute_rewards_matches_fleet_env_formula():
    # bitwise-level agreement with fleet_env._get_rewards' inline arithmetic
    # on identical hand-built inputs (fleet_env is first-wave and green; the
    # two implementations must never diverge).
    free = _corridor_free()
    df = _fields(free, [(1, 1), (1, 5), (0, 0)], [True, True, False])
    df = df.expand(4, MAX_TARGETS, 3, 7).contiguous()
    pos = torch.tensor([[[1, 0], [1, 2]],
                        [[1, 1], [1, 5]],
                        [[1, 3], [1, 4]],
                        [[1, 2], [1, 6]]], dtype=torch.int16)
    valid = _VALID2.expand(4, MAX_TARGETS)
    phi_prev = torch.tensor([-4.0, -1.0, -3.5, -2.0])
    done = torch.tensor([False, True, False, True])
    correct = torch.tensor([False, True, True, False])
    ho = torch.tensor([[True, False], [False, False],
                       [False, True], [False, False]])
    hr = torch.tensor([[False, False], [False, False],
                       [True, True], [False, True]])

    for use_expected in (False, True):
        got, phi = rewards.compute_rewards(
            AGENTS, df, pos, valid, phi_prev, done, correct, ho, hr,
            LAM, GAMMA, use_expected)
        # --- fleet_env._get_rewards transcription, fe constants ---------
        phi_ref = grid_core.matching_potential(df, pos, valid)
        phi_now = torch.where(done, torch.zeros_like(phi_ref), phi_ref)
        shaping = LAM * (GAMMA * phi_now - phi_prev)
        if use_expected:
            bonus = torch.full_like(shaping, fe.BLIND_EXPECTED_BONUS)
        else:
            bonus = fe.OUTCOME_BONUS * correct.float()
        team = fe.STEP_COST + shaping + torch.where(
            done, fe.COMPLETION_BONUS + bonus, torch.zeros_like(shaping))
        assert torch.equal(phi, phi_ref)
        for i, name in enumerate(AGENTS):
            expect = (team + fe.COLLISION_COST * ho[:, i].float()
                      + fe.COLLISION_COST * hr[:, i].float())
            assert torch.allclose(got[name], expect, atol=1e-6), \
                (name, use_expected, got[name], expect)


def test_compute_rewards_returns_raw_phi():
    # the returned Phi is the RAW potential (fleet_env stores raw _phi for
    # the next step's phi_prev); the terminal zeroing lives only inside the
    # reward arithmetic.
    free = _corridor_free()
    df = _fields(free, [(1, 1), (1, 5), (0, 0)], [True, True, False])
    pos = _pos((1, 2), (1, 4))                 # off-target, Phi = -2
    done = torch.tensor([True])                # artificial terminal
    phi_prev = torch.tensor([-3.0])
    got, phi = rewards.compute_rewards(
        AGENTS, df, pos, _VALID2, phi_prev, done, torch.tensor([True]),
        torch.zeros((1, 2), dtype=torch.bool),
        torch.zeros((1, 2), dtype=torch.bool))
    assert float(phi[0]) == -2.0, "raw Phi must pass through un-zeroed"
    # reward used the ZEROED terminal Phi: step + lambda*(0 - phi_prev)
    #   + completion + outcome
    expect = (rewards.STEP_COST + LAM * (0.0 - -3.0)
              + rewards.COMPLETION_BONUS + rewards.OUTCOME_BONUS)
    assert abs(float(got["robot_0"][0]) - expect) < 1e-6


# ---------------------------------------------------------------------------
# Latch-aware potential in the shaping (spec 1.10/1.12 [FIXED])
# ---------------------------------------------------------------------------

def test_latch_aware_shaping_signs():
    # corridor row with a station at each end: under the min-matching, a
    # step TOWARD the wrong station is strictly penalised, a step onto the
    # matched station is rewarded -- the shaping cannot pull a robot the
    # wrong way.
    free = _corridor_free()
    df = _fields(free, [(1, 1), (1, 5), (0, 0)], [True, True, False])
    zeros = torch.zeros((1, 2), dtype=torch.bool)
    nd = torch.tensor([False])
    no_y = torch.tensor([False])

    phi_a = grid_core.matching_potential(df, _pos((1, 4), (1, 2)), _VALID2)
    assert float(phi_a[0]) == -2.0            # matched: r0->right, r1->left

    # r0 steps LEFT toward the WRONG station: Phi -2 -> -3, shaping < 0
    got, phi_b = rewards.compute_rewards(
        AGENTS, df, _pos((1, 3), (1, 2)), _VALID2, phi_a, nd, no_y,
        zeros, zeros)
    assert float(phi_b[0]) == -3.0
    expect_wrong = rewards.STEP_COST + LAM * (GAMMA * -3.0 - -2.0)
    assert abs(float(got["robot_0"][0]) - expect_wrong) < 1e-6
    assert float(got["robot_0"][0]) < rewards.STEP_COST, \
        "step toward the wrong station must be shaped NEGATIVE"

    # r0 steps RIGHT onto its MATCHED station: Phi -2 -> -1, shaping > 0
    got, phi_c = rewards.compute_rewards(
        AGENTS, df, _pos((1, 5), (1, 2)), _VALID2, phi_a, nd, no_y,
        zeros, zeros)
    assert float(phi_c[0]) == -1.0
    expect_right = rewards.STEP_COST + LAM * (GAMMA * -1.0 - -2.0)
    assert abs(float(got["robot_0"][0]) - expect_right) < 1e-6
    assert float(got["robot_0"][0]) > 0.0, \
        "step onto the matched station must be shaped POSITIVE"


def test_latch_aware_field_flags_pass_through_trap():
    # spec 1.10 [FIXED: latch-aware distance field]: both stations in one
    # width-1 row with the far station only reachable THROUGH the near one.
    # The all-free field calls the doomed march "progress" (positive
    # shaping); the latch-aware field scores the state -inf (no perfect
    # matching exists without passing through an absorbing station), so the
    # corrected shaping cannot reward driving into the wrong station.  The
    # bank builder rejects such rows outright (alcove topology, spec 2.1);
    # tests/test_bank_distfields.py asserts latch-aware == all-free there.
    free = _corridor_free()
    t_cells = [(1, 4), (1, 6), (0, 0)]
    valid = [True, True, False]
    pos_before = _pos((1, 0), (1, 2))
    pos_after = _pos((1, 0), (1, 3))          # r1 marches toward t1 via t0

    aware = _fields(free, t_cells, valid, latch_aware=True)
    phi_aware = grid_core.matching_potential(pos=pos_before,
                                             dist_field=aware,
                                             target_valid=_VALID2)
    assert torch.isinf(phi_aware).all() and float(phi_aware[0]) < 0, \
        "latch-aware Phi must flag the pass-through trap as infeasible"
    phi_aware2 = grid_core.matching_potential(aware, pos_after, _VALID2)
    assert torch.isinf(phi_aware2).all(), "no gradient INTO the trap either"

    allfree = _fields(free, t_cells, valid, latch_aware=False)
    phi_free = grid_core.matching_potential(allfree, pos_before, _VALID2)
    phi_free2 = grid_core.matching_potential(allfree, pos_after, _VALID2)
    assert float(phi_free[0]) == -8.0 and float(phi_free2[0]) == -7.0
    s = rewards.shaping_reward(phi_free2, phi_free)
    assert float(s[0]) > 0.0, \
        "the broken all-free field REWARDS marching into the wrong station"


# ---------------------------------------------------------------------------
# Compliance-bound sanity case (spec 1.12 [FIXED: reward-fairness] / 3.4)
# ---------------------------------------------------------------------------

def _roll_return(df, valid, t_cells, pos_seq, correct):
    """Roll a scripted 2-robot trajectory through compute_rewards.

    ``pos_seq[0]`` is the spawn (supplies Phi_0); steps t = 1..T follow.
    ``done`` fires when both robots sit on (distinct) valid stations --
    the scripted analogue of the both-latched terminal.  No collisions.
    Returns (discounted return of robot_0, Phi_0, T).
    """
    stations = [tuple(t_cells[j]) for j in range(len(t_cells)) if valid[j]]
    zeros = torch.zeros((1, 2), dtype=torch.bool)
    phi_prev = grid_core.matching_potential(
        df, _pos(*pos_seq[0]), _VALID2)
    phi0 = float(phi_prev[0])
    total, disc, steps = 0.0, 1.0, 0
    for t in range(1, len(pos_seq)):
        p0, p1 = pos_seq[t]
        done = torch.tensor([tuple(p0) in stations and tuple(p1) in stations])
        got, phi = rewards.compute_rewards(
            AGENTS, df, _pos(p0, p1), _VALID2, phi_prev, done,
            torch.tensor([bool(correct)]), zeros, zeros)
        assert torch.allclose(got["robot_0"], got["robot_1"]), \
            "no collisions scripted: the team scalar must be shared"
        total += disc * float(got["robot_0"][0])
        disc *= GAMMA
        phi_prev = phi
        steps = t
        if bool(done[0]):
            break
    return total, phi0, steps


def _closed_form(T, phi0, bonus):
    """STEP_COST*annuity(T) + telescoped shaping + discounted terminal."""
    annuity = sum(GAMMA ** t for t in range(T))
    return (rewards.STEP_COST * annuity - LAM * phi0
            + GAMMA ** (T - 1) * (rewards.COMPLETION_BONUS + bonus))


def test_compliance_bound_sanity():
    # Open 12x12 map, stations tA=(5,2) / tB=(5,9), spawns r0=(5,4),
    # r1=(6,7).  Geometric default ("defect"): r0->tA, r1->tB, done at T=3,
    # outcome bonus 0.  Compliant plan: the instructed CROSSED matching
    # r0->tB, r1->tA, done at T=7, outcome bonus +10.  The shaping
    # telescopes identically in both lanes (-lambda*Phi_0, same spawn), so
    # G_comply - G_defect is pure discounted-bonus-vs-step-cost -- and it
    # must be decisively positive: compliance is unambiguously optimal
    # (spec 3.4), which harness/reward_audit.py later certifies bank-wide.
    free = torch.ones((12, 12), dtype=torch.bool)
    t_cells = [(5, 2), (5, 9), (0, 0)]
    valid = [True, True, False]
    df = _fields(free, t_cells, valid)

    defect = [
        ((5, 4), (6, 7)),                      # spawn
        ((5, 3), (6, 8)),
        ((5, 2), (6, 9)),                      # r0 latches tA
        ((5, 2), (5, 9)),                      # r1 latches tB -> done, Y=0
    ]
    comply = [
        ((5, 4), (6, 7)),                      # spawn (same Phi_0)
        ((4, 4), (6, 6)),
        ((4, 5), (6, 5)),
        ((4, 6), (6, 4)),
        ((4, 7), (6, 3)),
        ((4, 8), (6, 2)),
        ((4, 9), (5, 2)),                      # r1 latches tA
        ((5, 9), (5, 2)),                      # r0 latches tB -> done, Y=1
    ]

    g_defect, phi0_d, t_defect = _roll_return(df, valid, t_cells, defect,
                                              correct=False)
    g_comply, phi0_c, t_comply = _roll_return(df, valid, t_cells, comply,
                                              correct=True)

    assert phi0_d == phi0_c == -5.0, "lanes must share the spawn potential"
    assert (t_defect, t_comply) == (3, 7)

    # rolled returns match the telescoped closed form exactly
    expect_d = _closed_form(3, -5.0, 0.0)
    expect_c = _closed_form(7, -5.0, rewards.OUTCOME_BONUS)
    assert abs(g_defect - expect_d) < 1e-5, (g_defect, expect_d)
    assert abs(g_comply - expect_c) < 1e-5, (g_comply, expect_c)

    # the bound itself: comply beats defect by a decisive margin
    margin = g_comply - g_defect
    assert margin > 1.0, "compliance must be unambiguously optimal " \
        "(G_comply - G_defect = %r)" % margin
    assert abs(margin - (expect_c - expect_d)) < 1e-5
    # and the shaping contributed IDENTICALLY to both lanes (telescoping):
    # the margin is exactly the bonus/step-cost/discount arithmetic.
    annuity = lambda T: sum(GAMMA ** t for t in range(T))
    pure = (GAMMA ** 6 * (rewards.COMPLETION_BONUS + rewards.OUTCOME_BONUS)
            - GAMMA ** 2 * rewards.COMPLETION_BONUS
            + rewards.STEP_COST * (annuity(7) - annuity(3)))
    assert abs(margin - pure) < 1e-5, (margin, pure)


# ---------------------------------------------------------------------------
# First-latch bonus (DECISIONS.md terminal-credit amendment)
# ---------------------------------------------------------------------------

def _latch_transition(pos, target, valid, latched, slot, time_, t):
    """Mirror fleet_env._pre_physics_step's transition-mask construction."""
    prev = latched.clone()
    grid_core.latch_update(pos, target, valid, latched, slot, time_, t)
    return latched & ~prev


def _fresh_latch_state(E=1):
    target = torch.tensor([[[2, 2], [9, 9], [0, 0]]],
                          dtype=torch.int16).repeat(E, 1, 1)
    valid = torch.tensor([[True, True, False]]).repeat(E, 1)
    latched = torch.zeros((E, 2), dtype=torch.bool)
    slot = torch.full((E, 2), -1, dtype=torch.int8)
    time_ = torch.full((E, 2), -1, dtype=torch.int16)
    return target, valid, latched, slot, time_


def test_first_latch_fires_once_per_agent():
    """+2.0 on each agent's FIRST latch step only; absorbing latch means the
    transition mask (and hence the bonus) can never fire twice."""
    target, valid, latched, slot, time_ = _fresh_latch_state()
    t = torch.zeros(1, dtype=torch.long)
    # step 1: robot_0 enters station 0, robot_1 is in open space
    pos = torch.tensor([[[2, 2], [5, 5]]], dtype=torch.int16)
    newly = _latch_transition(pos, target, valid, latched, slot, time_, t)
    assert newly.tolist() == [[True, False]]
    r = rewards.first_latch_reward(newly)
    assert r[0, 0].item() == rewards.FIRST_LATCH_BONUS and r[0, 1].item() == 0.0
    # step 2: robot_0 still on its station (no re-fire), robot_1 latches
    pos2 = torch.tensor([[[2, 2], [9, 9]]], dtype=torch.int16)
    newly2 = _latch_transition(pos2, target, valid, latched, slot, time_,
                               t + 1)
    assert newly2.tolist() == [[False, True]]
    # step 3: both latched, nothing fires ever again
    newly3 = _latch_transition(pos2, target, valid, latched, slot, time_,
                               t + 2)
    assert newly3.tolist() == [[False, False]]


def test_first_latch_wrong_station_paid_identically():
    """The bonus reads the latch TRANSITION only: latching onto station 1
    (the 'wrong' one for any given instruction) pays exactly what latching
    onto station 0 pays -- correctness stays priced by the outcome term."""
    rs = []
    for station_cell in ([2, 2], [9, 9]):
        target, valid, latched, slot, time_ = _fresh_latch_state()
        pos = torch.tensor([[station_cell, [5, 5]]], dtype=torch.int16)
        newly = _latch_transition(pos, target, valid, latched, slot, time_,
                                  torch.zeros(1, dtype=torch.long))
        assert newly[0, 0].item() is True or bool(newly[0, 0])
        rs.append(rewards.first_latch_reward(newly))
    assert torch.equal(rs[0], rs[1])


def test_first_latch_assembly_and_instruction_independence():
    """per_agent_rewards wires the bonus per agent, and the whole reward
    dict is invariant to the instruction context (correct=0 vs correct=1)
    on non-terminal steps -- the bonus adds no instruction dependence."""
    E = 2
    team_kwargs = dict(
        phi=torch.tensor([-4.0, -6.0]), phi_prev=torch.tensor([-5.0, -6.0]),
        done=torch.zeros(E, dtype=torch.bool))
    hit_o = torch.zeros((E, 2), dtype=torch.bool)
    hit_r = torch.zeros((E, 2), dtype=torch.bool)
    newly = torch.tensor([[True, False], [False, False]])
    outs = []
    for correct in (torch.zeros(E, dtype=torch.bool),
                    torch.ones(E, dtype=torch.bool)):
        team = rewards.team_reward(correct=correct, **team_kwargs)
        outs.append(rewards.per_agent_rewards(AGENTS, team, hit_o, hit_r,
                                              newly))
    for name in AGENTS:
        assert torch.equal(outs[0][name], outs[1][name])
    base = rewards.per_agent_rewards(
        AGENTS, rewards.team_reward(
            correct=torch.zeros(E, dtype=torch.bool), **team_kwargs),
        hit_o, hit_r, None)
    got = outs[0]
    diff0 = (got["robot_0"] - base["robot_0"])
    diff1 = (got["robot_1"] - base["robot_1"])
    assert torch.allclose(diff0, torch.tensor([rewards.FIRST_LATCH_BONUS, 0.0]))
    assert torch.allclose(diff1, torch.zeros(E))


def test_first_latch_none_matches_zeros():
    """newly_latched=None (pre-amendment call sites) is exactly the
    all-zeros mask -- backward compatibility is byte-exact."""
    E = 3
    team = torch.randn(E)
    hit_o = torch.rand((E, 2)) > 0.5
    hit_r = torch.rand((E, 2)) > 0.5
    a = rewards.per_agent_rewards(AGENTS, team, hit_o, hit_r, None)
    b = rewards.per_agent_rewards(AGENTS, team, hit_o, hit_r,
                                  torch.zeros((E, 2), dtype=torch.bool))
    for name in AGENTS:
        assert torch.equal(a[name], b[name])


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
