"""Divergence primitives and coordination metrics over rollout records.

M1_SPEC section 7 file-plan entry for this module: "C, ITT outcome, Y, TV
divergence (per-agent + joint), decision-point masking, truncation to
min(T_A, T_B)".  RESEARCH_PLAN.md's evaluation plan adds the coordination
metrics reported alongside success throughout: deadlock rate, idle time,
redundant-action rate, cross-agent interference, role stability,
assignment accuracy.

Everything here is PURE TORCH, batched over the episode dimension E, and
operates on the tensor layout of ``harness.rollout.RolloutRecord``:

* trajectory tensors are step-major ``(T, E, ...)`` and masked to each
  env's FIRST episode (``active`` (T, E) bool marks the live steps and is
  a PREFIX mask -- True exactly for ``t < episode_length``, the driver's
  ``mask_after_done`` contract);
* terminal tensors are ``(E, ...)``: ``latch_slot`` (E, N) int8 (-1
  unlatched), ``latch_time`` (E, N) int16 (-1 unlatched, 0-based decision
  index in the SAME frame as the trajectory step index -- fleet_env calls
  ``latch_update`` with ``episode_length_buf`` pre-increment), ``completed``
  (E,) bool, ``correct`` (E,) bool, ``outcome`` (E,) int8 ITT code.

Divergences between two lanes of a counterfactual pair (spec 6.1/6.4):

* ``traj_divergence`` -- CSI_traj's object: per-step mean position
  divergence, truncated to min(T_A, T_B) via the common prefix of the two
  ``active`` masks (spec 6.4 length control).
* ``action_tv`` -- CSI_step's object: mean TV distance between the two
  lanes' Categorical action distributions, per-agent marginal and joint,
  with optional decision-point masking (spec 6.1 [FIXED: restricted to
  instruction-relevant decision points]).  The decision-point mask itself
  is an INPUT here: it is defined as "timesteps where the compliant and
  greedy scripted planners' optimal actions differ" and therefore belongs
  to ``harness/planners.py`` (not yet in the repo); this module only
  applies it.
* ``action_mismatch`` -- Hamming divergence on realised or INTENDED
  action sequences (CSI_intent's object: the revert rule creates
  action-effect degeneracy, so intent-level divergence is reported
  alongside, spec 6.1).
* ``outcome_codes`` / ``outcome_flip`` -- CSI_flip's object: the realised
  assignment (role binding: which slot each robot latched) or the
  realised order (precedence: sign of the latch-time gap), with the flip
  indicator defined only where BOTH lanes completed.

NaN convention: a per-episode statistic whose denominator is empty (no
common active step, no completed pair, no pre-latch step) is NaN, never a
silent 0 -- degenerate-denominator handling is the caller's decision
(``harness/csi.py`` implements the spec 6.1 regularisation).
"""

import os
import sys

import torch

try:
    from tasks.team_listen import grid_core
    from tasks.team_listen import obs_layout as L
except ImportError:  # standalone import: put the repo root on sys.path
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from tasks.team_listen import grid_core
    from tasks.team_listen import obs_layout as L

STAY = grid_core.STAY

#: Position-divergence norms (grid world: "l1" Manhattan is the default;
#: "hamming" is cell mismatch in {0,1}; "l2" Euclidean).
TRAJ_NORMS = ("l1", "l2", "hamming")

#: Outcome-code variants (spec 2.3 / 3.3).
VARIANTS = ("RoleBinding", "Precedence")


# ---------------------------------------------------------------------------
# Pairing / truncation (spec 6.4 length control)
# ---------------------------------------------------------------------------

def episode_lengths(active):
    """(E,) long: recorded first-episode length from the prefix mask."""
    return active.long().sum(dim=0)


def pair_mask(active_a, active_b):
    """(T, E) bool: steps live in BOTH lanes.

    Because ``active`` is a prefix mask (True exactly for
    ``t < episode_length``), the elementwise AND is exactly the spec 6.4
    truncation of every pair to ``min(T_A, T_B)``.
    """
    assert active_a.shape == active_b.shape, (active_a.shape, active_b.shape)
    return active_a & active_b


def masked_step_mean(values, mask):
    """(E,) per-episode mean of ``values`` (T, E) over ``mask`` (T, E).

    Empty mask -> NaN (0/0), never a silent 0: degenerate denominators are
    surfaced to the caller (spec 6.1).
    """
    m = mask.to(torch.float32)
    v = values.to(torch.float32)
    return (v * m).sum(dim=0) / m.sum(dim=0)


# ---------------------------------------------------------------------------
# Trajectory-level divergence (CSI_traj's object, spec 6.1)
# ---------------------------------------------------------------------------

def position_distance(pos_a, pos_b, norm="l1"):
    """Per-agent position distance, consuming the trailing (row, col) dim.

    Args:
        pos_a, pos_b: (..., 2) integer positions (any matching shape).
        norm: "l1" (Manhattan), "l2" (Euclidean) or "hamming" (cell
              mismatch in {0, 1}).

    Returns:
        (...,) float32.
    """
    if norm not in TRAJ_NORMS:
        raise ValueError("norm %r not in %r" % (norm, TRAJ_NORMS))
    diff = (pos_a.long() - pos_b.long()).float()
    if norm == "l1":
        return diff.abs().sum(dim=-1)
    if norm == "l2":
        return diff.pow(2).sum(dim=-1).sqrt()
    return (diff != 0).any(dim=-1).float()                 # hamming


def traj_divergence(pos_a, pos_b, active_a, active_b, norm="l1"):
    """Per-step mean position divergence over the common prefix (spec 6.1).

    Args:
        pos_a, pos_b:       (T, E, N, 2) integer post-conflict positions.
        active_a, active_b: (T, E) bool prefix masks.
        norm: see ``position_distance``.

    Returns:
        (E,) float32: mean over min(T_A, T_B) steps of the per-step
        agent-mean distance; NaN where the common prefix is empty.
    """
    assert pos_a.shape == pos_b.shape, (pos_a.shape, pos_b.shape)
    d = position_distance(pos_a, pos_b, norm=norm).mean(dim=-1)   # (T, E)
    return masked_step_mean(d, pair_mask(active_a, active_b))


# ---------------------------------------------------------------------------
# Action-distribution divergence (CSI_step's object, spec 6.1/6.2)
# ---------------------------------------------------------------------------

def tv_distance(p, q):
    """Total-variation distance between categoricals on the last dim.

    ``0.5 * sum_a |p_a - q_a|`` in [0, 1]; any matching leading shape.
    """
    assert p.shape == q.shape, (p.shape, q.shape)
    return 0.5 * (p.float() - q.float()).abs().sum(dim=-1)


def joint_probs(probs):
    """(..., N, A) per-agent marginals -> (..., A**N) factored joint.

    The joint action distribution of independent per-agent policies is the
    product of the marginals (spec 6.1 CSI_step "per-agent marginal and
    joint"; a factored joint roughly halves TV when only one agent
    changes, which is exactly why both are reported).
    """
    n = probs.shape[-2]
    out = probs[..., 0, :]
    for i in range(1, n):
        out = (out.unsqueeze(-1) * probs[..., i, :].unsqueeze(-2))
        out = out.reshape(out.shape[:-2] + (-1,))
    return out


def action_tv(logits_a, logits_b, active_a, active_b, joint=False,
              decision_mask=None):
    """Per-episode mean TV between two lanes' action distributions.

    Args:
        logits_a, logits_b: (T, E, N, A) Categorical logits.
        active_a, active_b: (T, E) bool prefix masks.
        joint: TV on the factored joint over agents (A**N outcomes)
               instead of the mean of per-agent marginal TVs.
        decision_mask: optional (T, E) bool restricting the average to
               instruction-relevant decision points (spec 6.1 [FIXED]);
               produced by the scripted planners, applied here.

    Returns:
        (E,) float32 in [0, 1]; NaN where the (masked) common prefix is
        empty -- an episode with no decision point has no defined
        statistic, and the all-timestep average is a SEPARATE call with
        ``decision_mask=None`` (secondary per spec 6.1).
    """
    assert logits_a.shape == logits_b.shape, (logits_a.shape, logits_b.shape)
    p = torch.softmax(logits_a.float(), dim=-1)
    q = torch.softmax(logits_b.float(), dim=-1)
    if joint:
        tv = tv_distance(joint_probs(p), joint_probs(q))          # (T, E)
    else:
        tv = tv_distance(p, q).mean(dim=-1)                       # (T, E)
    mask = pair_mask(active_a, active_b)
    if decision_mask is not None:
        assert decision_mask.shape == mask.shape
        mask = mask & decision_mask.bool()
    return masked_step_mean(tv, mask)


def action_mismatch(act_a, act_b, active_a, active_b, decision_mask=None):
    """Per-episode Hamming divergence between two action sequences.

    Use on ``record.intended`` for CSI_intent (pre-revert intent, spec
    6.1: realised-only reporting understates sensitivity) or on
    ``record.executed`` for the realised counterpart.

    Args:
        act_a, act_b: (T, E, N) integer actions.
        active_a, active_b: (T, E) bool prefix masks.
        decision_mask: optional (T, E) bool decision-point restriction.

    Returns:
        (E,) float32 in [0, 1]: masked mean over steps of the per-step
        agent-mean disagreement; NaN where the masked prefix is empty.
    """
    assert act_a.shape == act_b.shape, (act_a.shape, act_b.shape)
    dis = (act_a.long() != act_b.long()).float().mean(dim=-1)     # (T, E)
    mask = pair_mask(active_a, active_b)
    if decision_mask is not None:
        assert decision_mask.shape == mask.shape
        mask = mask & decision_mask.bool()
    return masked_step_mean(dis, mask)


# ---------------------------------------------------------------------------
# Outcome-level divergence (CSI_flip's object, spec 6.1; outcomes per
# spec 2.3 / 3.3)
# ---------------------------------------------------------------------------

def realised_assignment(latch_slot):
    """(E,) long code of the realised role-binding assignment.

    Encodes BOTH live agents' latch slots into one integer:
    ``(slot0 + 1) * (MAX_TARGETS + 1) + (slot1 + 1)`` (unlatched -1 maps
    to 0), so "the realised outcome flipped" is a single integer
    inequality.  At N=2 with distinct alcoves, robot 0's slot alone
    determines the assignment, but encoding both is robust to padded or
    partial latches.
    """
    s = latch_slot.long() + 1
    return s[:, 0] * (L.MAX_TARGETS + 1) + s[:, 1]


def realised_order(latch_time):
    """(E,) long realised precedence order: sign(latch_time_r1 - r0).

    +1 robot_0 first, -1 robot_1 first, 0 tie (structurally impossible
    under the unique-mouth topology, spec 3.1, but representable).
    """
    return torch.sign(latch_time[:, 1].long() - latch_time[:, 0].long())


def outcome_codes(latch_slot, latch_time, variant):
    """(E,) long realised-outcome code for ``variant`` (spec 2.3 / 3.3)."""
    if variant not in VARIANTS:
        raise ValueError("variant %r not in %r" % (variant, VARIANTS))
    if variant == "Precedence":
        return realised_order(latch_time)
    return realised_assignment(latch_slot)


def outcome_flip(code_a, code_b, valid_a, valid_b):
    """Paired outcome-flip indicator (CSI_flip's per-episode object).

    Args:
        code_a, code_b: (E,) long realised-outcome codes.
        valid_a, valid_b: (E,) bool -- typically ``completed`` per lane;
            the outcome is defined only where C = 1 (spec 2.3).

    Returns:
        flip:  (E,) float32 -- 1.0 flipped / 0.0 unchanged, NaN where the
               pair is not valid (either lane incomplete).
        valid: (E,) bool -- both lanes completed.
    """
    valid = valid_a.bool() & valid_b.bool()
    flip = (code_a.long() != code_b.long()).float()
    return torch.where(valid, flip, torch.full_like(flip, float("nan"))), valid


# ---------------------------------------------------------------------------
# Outcome aggregates (spec file plan: "C, ITT outcome, Y")
# ---------------------------------------------------------------------------

def outcome_rates(record):
    """Scalar outcome aggregates from one record (spec 2.3 estimands).

    ``p_correct_itt`` is the primary ITT estimand P(O = correct) over ALL
    episodes; ``y_given_c`` is the clearly-labelled secondary conditional
    E[Y | C = 1] (NaN when no episode completed).
    """
    completed = record.completed.float()
    outcome = record.outcome.long()
    c_rate = completed.mean()
    y_given_c = (record.correct[record.completed].float().mean()
                 if bool(record.completed.any())
                 else torch.tensor(float("nan")))
    return {
        "completion_rate": float(c_rate),
        "p_correct_itt": float((outcome == 0).float().mean()),
        "p_wrong_itt": float((outcome == 1).float().mean()),
        "p_incomplete_itt": float((outcome == 2).float().mean()),
        "y_given_c": float(y_given_c),
    }


# ---------------------------------------------------------------------------
# Coordination metrics (RESEARCH_PLAN.md evaluation plan; definitions are
# fixed HERE -- the plan names the metrics without defining them)
# ---------------------------------------------------------------------------

def prelatch_mask(active, latch_time):
    """(T, E, N) bool: live steps strictly BEFORE each agent's latch.

    The latch step itself (t == latch_time) is the arrival move and every
    later step is the forced overwrite to "stay" (spec 1.11), so neither
    belongs in idleness/redundancy denominators.  Never-latched agents
    (latch_time == -1) contribute every live step.
    """
    T = active.shape[0]
    t_idx = torch.arange(T, device=active.device).view(T, 1, 1)
    lt = latch_time.long().unsqueeze(0)                    # (1, E, N)
    pre = (lt < 0) | (t_idx < lt)
    return active.unsqueeze(-1) & pre


def idle_fraction(executed, active, latch_time):
    """(E, N) float32: fraction of pre-latch live steps spent on STAY.

    "Idle time" per RESEARCH_PLAN.md.  Post-latch forced stays are
    excluded (see ``prelatch_mask``).  NaN where an agent has no
    pre-latch step (latched at t = 0).
    """
    pre = prelatch_mask(active, latch_time)
    stay = (executed.long() == STAY) & pre
    return stay.float().sum(dim=0) / pre.float().sum(dim=0)


def redundant_action_fraction(executed, positions, active, latch_time,
                              spawn=None):
    """(E, N) float32: fraction of pre-latch steps that were redundant.

    "Redundant-action rate" per RESEARCH_PLAN.md; defined here as an
    executed action that produced no progress:

    * blocked: a non-STAY executed action that left the position
      unchanged (obstacle/conflict revert or off-grid clamp), or
    * backtrack: a move that returned the agent to its position of two
      steps earlier (immediate A -> B -> A oscillation).

    ``positions[t]`` is the POST-step position, so the pre-step position
    at t is ``positions[t-1]`` -- and at t = 0 it is the spawn, which the
    record does not carry.  Pass ``spawn`` (E, N, 2) to include step 0
    (and step-1 backtracks); with ``spawn=None`` the denominator starts
    at t = 1 (t = 2 for backtracks).

    Returns NaN where an agent has no considered step.
    """
    T, E, N = executed.shape
    pos = positions.long()
    pre = prelatch_mask(active, latch_time)
    t_idx = torch.arange(T, device=executed.device).view(T, 1, 1)

    if spawn is not None:
        spawn_row = spawn.long().unsqueeze(0)                        # (1,E,N,2)
        prev = torch.cat([spawn_row, pos[:-1]], dim=0)               # (T,E,N,2)
        prev2 = (torch.cat([spawn_row, spawn_row, pos[:-2]], dim=0)
                 if T >= 2 else spawn_row.expand(T, E, N, 2))
        considered = pre
        backtrack_ok = t_idx >= 1
    else:
        prev = torch.cat([pos[:1], pos[:-1]], dim=0)     # t=0 self (masked out)
        prev2 = torch.cat([pos[:2], pos[:-2]], dim=0) if T >= 2 else pos
        considered = pre & (t_idx >= 1)
        backtrack_ok = t_idx >= 2

    non_stay = executed.long() != STAY
    blocked = non_stay & (pos == prev).all(dim=-1)
    backtrack = ((pos == prev2).all(dim=-1) & (pos != prev).any(dim=-1)
                 & backtrack_ok)
    redundant = (blocked | backtrack) & considered
    return redundant.float().sum(dim=0) / considered.float().sum(dim=0)


def deadlocked(positions, active, completed, window=8):
    """(E,) bool: incomplete episode whose final steps are frozen.

    "Deadlock rate" per RESEARCH_PLAN.md: the episode did NOT complete
    (C = 0, so it ran to timeout) and NO agent changed position over the
    last ``min(window, length)`` live steps.  Episodes shorter than 2
    steps cannot exhibit movement and are never flagged.
    """
    T, E = active.shape
    length = episode_lengths(active)                       # (E,)
    pos = positions.long()
    moved = torch.zeros((T, E), dtype=torch.bool, device=active.device)
    if T >= 2:
        moved[1:] = (pos[1:] != pos[:-1]).any(dim=-1).any(dim=-1)
    t_idx = torch.arange(T, device=active.device).unsqueeze(1)     # (T, 1)
    lo = (length - int(window)).clamp(min=1)
    tail = (t_idx >= lo.unsqueeze(0)) & (t_idx < length.unsqueeze(0))
    any_move_tail = (moved & tail).any(dim=0)
    return (~completed.bool()) & (length >= 2) & ~any_move_tail


def role_stability(positions, targets, target_valid, active):
    """(E,) float32 in [0, 1]: 1 - realised role-switch rate.

    "Role stability" per RESEARCH_PLAN.md.  The realised role of an agent
    at step t is proxied by its NEAREST valid target under Manhattan
    distance (ties -> lowest slot); a switch is a change of nearest
    target between consecutive live steps.  Stability = 1 - switches /
    agent-transitions; NaN where no transition exists (single-step
    episode).  A latched agent's position is frozen, so its role is
    constant and dilutes nothing.

    Targets are NOT part of the rollout record -- pass ``env.target`` /
    ``env.target_valid`` (E, MAX_TARGETS, 2) / (E, MAX_TARGETS) captured
    at reset (per-episode constants, spec 1.10).
    """
    T = positions.shape[0]
    pos = positions.long()                                 # (T, E, N, 2)
    tgt = targets.long().view(1, targets.shape[0], 1,
                              targets.shape[1], 2)         # (1, E, 1, MT, 2)
    d = (pos.unsqueeze(3) - tgt).abs().sum(dim=-1)         # (T, E, N, MT)
    big = torch.iinfo(torch.long).max
    d = d.masked_fill(~target_valid.view(1, targets.shape[0], 1, -1), big)
    role = d.argmin(dim=-1)                                # (T, E, N)
    if T < 2:
        return torch.full(positions.shape[1:2], float("nan"))
    trans = active[1:] & active[:-1]                       # (T-1, E)
    switch = (role[1:] != role[:-1]) & trans.unsqueeze(-1)
    n_switch = switch.float().sum(dim=0).sum(dim=-1)       # (E,)
    n_trans = trans.float().sum(dim=0) * positions.shape[2]
    return 1.0 - n_switch / n_trans


def interference_rate(n_robot_collisions, active):
    """(E,) float32: robot-robot collision flags per agent-step.

    "Cross-agent interference" per RESEARCH_PLAN.md, from the record's
    per-agent collision counters (spec 6.6 columns).
    """
    length = episode_lengths(active).float()
    n = n_robot_collisions.shape[1]
    return n_robot_collisions.float().sum(dim=1) / (length * n)


def coordination_summary(record, targets=None, target_valid=None,
                         spawn=None, window=8):
    """Scalar coordination metrics from one record (RESEARCH_PLAN.md list).

    NaN-aware means over episodes/agents (an episode with an undefined
    per-episode value is excluded, not zero-counted).  ``role_stability``
    requires reset-time ``targets`` / ``target_valid`` and is omitted
    from the dict when they are not given; ``spawn`` sharpens the
    redundant-action denominator (see ``redundant_action_fraction``).
    ``assignment_accuracy`` is E[Y | C = 1] -- the secondary conditional
    (spec 2.3); the ITT rates are included alongside.
    """
    out = dict(outcome_rates(record))
    out["assignment_accuracy"] = out.pop("y_given_c")
    out["deadlock_rate"] = float(
        deadlocked(record.positions, record.active, record.completed,
                   window=window).float().mean())
    out["idle_time"] = float(torch.nanmean(
        idle_fraction(record.executed, record.active, record.latch_time)))
    out["redundant_action_rate"] = float(torch.nanmean(
        redundant_action_fraction(record.executed, record.positions,
                                  record.active, record.latch_time,
                                  spawn=spawn)))
    out["interference_rate"] = float(
        interference_rate(record.n_robot_collisions, record.active).mean())
    if targets is not None:
        assert target_valid is not None, \
            "role_stability needs target_valid alongside targets"
        out["role_stability"] = float(torch.nanmean(
            role_stability(record.positions, targets, target_valid,
                           record.active)))
    return out


__all__ = [
    "STAY", "TRAJ_NORMS", "VARIANTS",
    "episode_lengths", "pair_mask", "masked_step_mean",
    "position_distance", "traj_divergence",
    "tv_distance", "joint_probs", "action_tv", "action_mismatch",
    "realised_assignment", "realised_order", "outcome_codes",
    "outcome_flip", "outcome_rates",
    "prelatch_mask", "idle_fraction", "redundant_action_fraction",
    "deadlocked", "role_stability", "interference_rate",
    "coordination_summary",
]
