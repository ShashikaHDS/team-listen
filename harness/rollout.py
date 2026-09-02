"""Rollout driver: the CSI measurement backbone (M1_SPEC 4.3 / 6.3 / sec. 7).

What lives here, per the spec's file plan ("raw env.unwrapped driver:
argmax + stochastic, 5-lane batching, identical-init and
paired-trajectory-equality assertions" + "per-episode record emission"):

* **Action selection** -- ``mode="argmax"``: deterministic argmax over the
  Categorical logits (spec 4.3).  NEVER play.py's
  ``outputs[-1][a].get("mean_actions", outputs[0][a])`` idiom:
  ``"mean_actions"`` is a Gaussian key absent for Categorical, silently
  returning the SAMPLED action.  ``mode="stochastic"``: Categorical
  sampling from the same logits through an explicit ``torch.Generator``
  so stochastic rollouts are reproducible.  ``torch.argmax`` breaks ties
  to the lowest index (0 = up); the per-step top-2 logit margin is
  recorded so tie-driven steps are identifiable post hoc (spec 4.3
  [FIXED: argmax tie-break]).
* **Lane batching (spec 6.3)** -- counterfactual pairs are LANES OF THE
  SAME VECTORISED BATCH, never two processes.  ``make_five_lane_plan``
  builds the L0..L4 layout (factual / counterfactual / seed / spawn /
  blank); ``make_paired_plan`` builds the two-lane within-scenario paired
  manifest layout of spec 4.3 (each base scenario under both instruction
  classes), which is the same machinery restricted to L0/L1.
* **Identical-init assertion (spec 6.3)** -- ``torch.equal`` on the
  non-language observation slices at t = 0 for every group, every
  rollout (spawn-intervention lanes excluded: they differ by design).
* **Paired-lane trajectory-equality assertion (spec 4.3)** -- for a blind
  arm (``lang_gain == 0``) the factual and counterfactual lanes receive
  byte-identical observations on a byte-identical env with identical
  stored slips, so the FULL recorded trajectory tensors must be
  ``torch.equal``; any nonzero deviation is direct proof of an
  observation-channel leak, with zero sampling noise.  Applied
  automatically for blind arms in argmax mode (in stochastic mode the two
  lanes draw independent samples, so the identity does not hold and the
  check is skipped by design).
* **Per-episode record emission** -- ``episode_records`` flattens a
  ``RolloutRecord`` into the per-episode scalars the spec 6.6 parquet
  schema consumes (``harness/records.py``'s job to serialise).

Lane logic is PURE TORCH and env-agnostic: it drives any object exposing
the ``TeamGridEnv`` driver surface (see ``resolve_env``), so it is
testable on ``harness/cpu_env.CPUFleetEnv`` without Isaac and runs
unchanged against the real env on the 5090.  The Isaac path is guarded by
construction -- this module NEVER imports isaaclab; on the 5090 pass
``env.unwrapped`` (the raw ``DirectMARLEnv``), never the skrl wrapper
(``IsaacLabMultiAgentWrapper.reset()`` has a ``_reset_once`` flag and
returns cached observations on repeat calls, spec 4.3).

WHY the driver steps the env's sub-calls manually instead of ``step()``:
``DirectMARLEnv.step`` auto-resets terminated envs BEFORE observations
are rebuilt, wiping the terminal latch state the spec 6.6 record needs
(``latch_time_*``, ``latch_slot_*``, Y).  The driver therefore runs the
documented step order itself -- ``_pre_physics_step`` (the whole grid
transition, spec 1.1), ``episode_length_buf += 1`` BEFORE ``_get_dones``
(spec 1.4), ``_get_rewards``, NO reset -- and reads the terminal fields
after the loop.  This is behaviourally identical for the first episode of
every env: the zero-asset ``sim.step()`` is a no-op over zero actors
(spec 1.1), a both-latched env is frozen by the latch rule (spec 1.11),
and every env's first episode ends by t = T_DECISION.  Recorded tensors
are masked to each env's first episode (``active``).
"""

import dataclasses
import os
import sys
from typing import Optional, Tuple

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


# ---------------------------------------------------------------------------
# Lane identities (spec 6.3): five lanes of one batch.
# ---------------------------------------------------------------------------

LANE_FACTUAL = 0         # L0: (scenario s_i, instr I_i,          slip stream 0)
LANE_COUNTERFACTUAL = 1  # L1: (scenario s_i, minimal_pair(I_i),  slip stream 0)
LANE_SEED = 2            # L2: (scenario s_i, instr I_i,          slip stream 1)
LANE_SPAWN = 3           # L3: (spawn_alt of s_i, instr I_i,      slip stream 0)
LANE_BLANK = 4           # L4: (scenario s_i, LANG zeroed,        slip stream 0)

LANE_NAMES = ("factual", "counterfactual", "seed", "spawn", "blank")
N_LANES_FULL = 5

#: Action-selection modes.  Numerator and denominator of any given CSI
#: contrast must use the SAME mode (spec 6.2).
MODES = ("argmax", "stochastic", "paired_stochastic")


# ---------------------------------------------------------------------------
# Lane plan: pure-torch layout of one vectorised eval batch
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LanePlan:
    """Lane-major layout of ``n_lanes * n_base`` envs (spec 6.3).

    Env ``lane_pos * n_base + g`` runs base scenario (group) ``g`` under
    the lane's intervention.  All tensors are flat ``(n_envs,)`` in that
    order.  ``instr_rows`` may be None: ``run_lanes`` then derives a
    deterministic per-class default row (``default_instruction_rows``).
    """

    lanes: Tuple[int, ...]           # lane ids, in batch order
    n_base: int
    scenarios: torch.Tensor          # (E,) long   bank row per env
    streams: torch.Tensor            # (E,) long   stored slip stream per env
    instr_classes: torch.Tensor      # (E,) long   semantic class per env
    instr_rows: Optional[torch.Tensor]  # (E,) long or None
    blank: torch.Tensor              # (E,) bool   LANG zeroed after reset
    spawn_alt: torch.Tensor          # (E,) bool   bank.spawn_alt instead of spawn

    @property
    def n_lanes(self):
        return len(self.lanes)

    @property
    def n_envs(self):
        return self.n_lanes * self.n_base

    def lane_pos(self, lane_id):
        """Position of ``lane_id`` in the batch (index into ``lanes``)."""
        return self.lanes.index(lane_id)

    def lane_slice(self, lane_id):
        p = self.lane_pos(lane_id)
        return slice(p * self.n_base, (p + 1) * self.n_base)

    def view(self, x):
        """Reshape a flat (E, ...) tensor to (n_lanes, n_base, ...)."""
        return x.reshape((self.n_lanes, self.n_base) + tuple(x.shape[1:]))

    def lane_of_env(self):
        """(E,) long: lane id of every env."""
        ids = torch.tensor(self.lanes, dtype=torch.long)
        return ids.repeat_interleave(self.n_base)

    def group_of_env(self):
        """(E,) long: base-scenario group index of every env."""
        return torch.arange(self.n_base, dtype=torch.long).repeat(self.n_lanes)

    def validate(self):
        E = self.n_envs
        assert len(set(self.lanes)) == len(self.lanes), "duplicate lane ids"
        for name in ("scenarios", "streams", "instr_classes", "blank",
                     "spawn_alt"):
            t = getattr(self, name)
            assert torch.is_tensor(t) and t.numel() == E, \
                "%s must be a flat (%d,) tensor" % (name, E)
        if self.instr_rows is not None:
            assert self.instr_rows.numel() == E
        # scenario ids are lane-invariant within a group (spec 6.3: "same
        # physics" is scenario_id equality; L3 differs by SPAWN, not row).
        v = self.view(self.scenarios.long())
        assert bool((v == v[0]).all()), \
            "scenario ids differ across lanes within a group (spec 6.3)"
        return self


def _const(value, n, dtype=torch.long):
    return torch.full((n,), value, dtype=dtype)


def make_paired_plan(base_sids, classes_a=None, classes_b=None,
                     rows_a=None, rows_b=None, stream=0):
    """The two-lane within-scenario paired manifest layout (spec 4.3).

    Each base scenario runs under BOTH instruction classes on identical
    physics (same bank row, same stored slip stream): lane L0 carries
    ``classes_a`` (default: all class 0), lane L1 ``classes_b`` (default:
    the complement).  For any blind arm the two lanes then receive
    byte-identical observations, so trajectories must be bit-identical and
    exactly one lane of each competent pair scores Y = 1.
    """
    base_sids = base_sids.reshape(-1).long()
    K = base_sids.numel()
    if classes_a is None:
        classes_a = torch.zeros((K,), dtype=torch.long)
    classes_a = classes_a.reshape(-1).long()
    if classes_b is None:
        classes_b = 1 - classes_a
    classes_b = classes_b.reshape(-1).long()
    rows = None
    if rows_a is not None or rows_b is not None:
        assert rows_a is not None and rows_b is not None, \
            "give rows for both lanes or neither"
        rows = torch.cat([rows_a.reshape(-1).long(),
                          rows_b.reshape(-1).long()])
    return LanePlan(
        lanes=(LANE_FACTUAL, LANE_COUNTERFACTUAL),
        n_base=K,
        scenarios=base_sids.repeat(2),
        streams=_const(int(stream), 2 * K),
        instr_classes=torch.cat([classes_a, classes_b]),
        instr_rows=rows,
        blank=torch.zeros((2 * K,), dtype=torch.bool),
        spawn_alt=torch.zeros((2 * K,), dtype=torch.bool),
    ).validate()


def make_five_lane_plan(base_sids, factual_classes=None, pair_classes=None,
                        factual_rows=None, pair_rows=None,
                        base_stream=0, alt_stream=1):
    """The full L0..L4 CSI layout of spec 6.3 (5 lanes x n_base envs).

    L0 factual, L1 counterfactual (minimal-pair instruction), L2 seed
    (slip stream ``alt_stream``, everything else frozen -- the paper's
    definitional physics-seed denominator), L3 spawn (``bank.spawn_alt``,
    matched BFS difficulty), L4 blank (language slice zeroed at eval;
    factual class kept for scoring).
    """
    base_sids = base_sids.reshape(-1).long()
    K = base_sids.numel()
    if factual_classes is None:
        factual_classes = torch.zeros((K,), dtype=torch.long)
    factual_classes = factual_classes.reshape(-1).long()
    if pair_classes is None:
        pair_classes = 1 - factual_classes
    pair_classes = pair_classes.reshape(-1).long()

    classes = torch.cat([factual_classes, pair_classes, factual_classes,
                         factual_classes, factual_classes])
    rows = None
    if factual_rows is not None or pair_rows is not None:
        assert factual_rows is not None and pair_rows is not None, \
            "give factual_rows and pair_rows together or not at all"
        fr = factual_rows.reshape(-1).long()
        pr = pair_rows.reshape(-1).long()
        rows = torch.cat([fr, pr, fr, fr, fr])

    streams = _const(int(base_stream), 5 * K)
    streams[2 * K:3 * K] = int(alt_stream)                    # L2 seed lane
    blank = torch.zeros((5 * K,), dtype=torch.bool)
    blank[4 * K:5 * K] = True                                 # L4 blank lane
    spawn = torch.zeros((5 * K,), dtype=torch.bool)
    spawn[3 * K:4 * K] = True                                 # L3 spawn lane
    return LanePlan(
        lanes=(LANE_FACTUAL, LANE_COUNTERFACTUAL, LANE_SEED, LANE_SPAWN,
               LANE_BLANK),
        n_base=K,
        scenarios=base_sids.repeat(5),
        streams=streams,
        instr_classes=classes,
        instr_rows=rows,
        blank=blank,
        spawn_alt=spawn,
    ).validate()


# ---------------------------------------------------------------------------
# Action selection (spec 4.3)
# ---------------------------------------------------------------------------

def select_actions(logits, mode, generator=None, paired_u=None):
    """Actions from Categorical logits (..., N_ACTIONS).

    ``argmax``: deterministic argmax (ties -> lowest index, i.e. 0 = up;
    stated in the paper, margins recorded -- spec 4.3 [FIXED]).
    ``stochastic``: Categorical sampling; pass a ``torch.Generator`` for
    reproducible draws.  Never a ``mean_actions`` lookup (Gaussian-only
    key; absent for Categorical it silently returns the sampled action).
    """
    if mode == "argmax":
        return logits.argmax(dim=-1)
    if mode == "stochastic":
        probs = torch.softmax(logits.float(), dim=-1)
        flat = probs.reshape(-1, probs.shape[-1])
        picks = torch.multinomial(flat, 1, generator=generator)
        return picks.reshape(probs.shape[:-1])
    if mode == "paired_stochastic":
        # Inverse-CDF sampling from EXTERNAL uniforms (round-4 eval-mode
        # study): the caller draws one uniform per (base scenario, agent)
        # and TILES it across a plan's lanes, so identical logits in two
        # lanes yield bit-identical samples and the blind-arm paired
        # machine check (spec 4.3) survives sampling-mode eval.
        if paired_u is None:
            raise ValueError("paired_stochastic needs paired_u")
        probs = torch.softmax(logits.float(), dim=-1)
        cum = probs.cumsum(dim=-1)
        picks = (paired_u.unsqueeze(-1) >= cum).sum(dim=-1)
        return picks.clamp_(max=probs.shape[-1] - 1)
    raise ValueError("mode %r not in %r" % (mode, MODES))


def top2_margin(logits):
    """Top-1 minus top-2 logit, (...,): near-zero flags argmax tie-break
    sensitivity (spec 4.3 [FIXED: argmax tie-break])."""
    top2 = logits.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


# ---------------------------------------------------------------------------
# Env surface (Isaac path guarded: duck-typed, no isaaclab import ever)
# ---------------------------------------------------------------------------

#: Driver-facing attributes shared by fleet_env.TeamGridEnv and
#: harness.cpu_env.CPUFleetEnv.
_ENV_SURFACE = (
    "cfg", "num_envs", "device", "episode_length_buf", "max_episode_length",
    "pos", "latched", "latch_slot", "latch_time", "scenario_id",
    "slip_stream", "instr_id", "instr_class", "lang_vec",
    "force_scenarios", "force_slip_stream", "_reset_idx",
    "_pre_physics_step", "_get_dones", "_get_rewards", "_get_observations",
    "_outcome_correct", "_write_lang_vec", "_bank", "_ALL",
)


def resolve_env(env):
    """Unwrap to the raw driver surface and verify it (spec 4.3).

    Accepts a ``TeamGridEnv`` (pass the DirectMARLEnv, i.e. what
    ``env.unwrapped`` returns on the 5090 -- NEVER the skrl wrapper) or a
    ``CPUFleetEnv``.  Raises with a 5090-deferral message otherwise.
    """
    base = getattr(env, "unwrapped", env)
    missing = [a for a in _ENV_SURFACE if not hasattr(base, a)]
    if missing:
        raise TypeError(
            "rollout driver needs the TeamGridEnv/CPUFleetEnv driver "
            "surface; %r is missing %r. On the 5090 pass env.unwrapped "
            "(the raw DirectMARLEnv), never the skrl wrapper (its reset() "
            "caches observations, M1_SPEC 4.3); on this box use "
            "harness.cpu_env.CPUFleetEnv (Isaac-dependent evaluation is "
            "5090-deferred)." % (type(base).__name__, missing))
    return base


# ---------------------------------------------------------------------------
# Post-reset lane interventions (spec 6.3: _reset_idx writes identical
# initial state into every lane; the driver then applies the lane deltas)
# ---------------------------------------------------------------------------

def default_instruction_rows(env, classes):
    """Deterministic surface-form row per class: the first train row of the
    class for lang/placebo arms, the class id itself for the 2-code table
    arms (symbol/zero).  Explicit rows (minimal pairs, spec 5.4) should be
    passed through the plan instead where they matter."""
    classes = classes.to(env.device, torch.long)
    rows_by_class = getattr(env, "_rows_by_class", None)
    if rows_by_class is not None:
        r0 = rows_by_class[0][0]
        r1 = rows_by_class[1][0]
        return torch.where(classes == 0, r0, r1)
    return classes.clone()


def force_instructions(env, rows, classes, env_ids=None):
    """Overwrite the drawn instruction with the plan's (per-lane) one.

    Uses the env's own ``_write_lang_vec`` (lang_gain and padded-slot
    semantics stay the env's, spec 1.10) and re-runs the unconditional
    arm-consistency invariant (spec 1.6).
    """
    if env_ids is None:
        env_ids = env._ALL
    n = env_ids.numel()
    rows = rows.to(env.device, torch.long).reshape(-1)
    classes = classes.to(env.device, torch.long).reshape(-1)
    assert rows.numel() == n and classes.numel() == n
    instr = rows.unsqueeze(1).expand(n, L.MAX_AGENTS).contiguous()
    env.instr_id[env_ids] = instr
    env.instr_class[env_ids] = classes
    env._write_lang_vec(env_ids, instr)
    env._assert_arm_consistency(env_ids)


def blank_language(env, mask):
    """L4 blank lane: zero the language slice at eval, everything else
    frozen (``D_blank``, spec 6.1).  ``instr_class`` keeps the factual
    class so Y is still scored."""
    ids = mask.to(env.device, torch.bool).nonzero(as_tuple=False).reshape(-1)
    if ids.numel() == 0:
        return
    env.lang_vec[ids] = 0.0


def apply_spawn_alt(env, mask):
    """L3 spawn lane: the difficulty-matched nuisance intervention
    (``bank.spawn_alt``, spec 1.10/6.1).  Mirrors the tail of
    ``_reset_idx`` for the moved robots: initial fog reveal from scratch
    plus Phi_0 at the new spawn."""
    ids = mask.to(env.device, torch.bool).nonzero(as_tuple=False).reshape(-1)
    if ids.numel() == 0:
        return
    N = L.N_AGENTS
    sid = env.scenario_id[ids]
    env.pos[ids] = env._bank.spawn_alt[sid]
    kf = torch.zeros((ids.numel(), L.R, L.C), dtype=torch.bool,
                     device=env.device)
    ko = torch.zeros_like(kf)
    grid_core.reveal(kf, ko, env.occ[ids], env.pos[ids][:, :N],
                     env._reveal_offsets)
    env.known_free[ids] = kf
    env.known_obs[ids] = ko
    env._phi[ids] = grid_core.matching_potential(
        env.dist_field[ids], env.pos[ids][:, :N], env.target_valid[ids])


# ---------------------------------------------------------------------------
# Identical-init assertion (spec 6.3)
# ---------------------------------------------------------------------------

def nonlang_mask(width, device=None):
    """(width,) bool: True on every non-language observation dimension.

    Widths: OBS_DIM (partial, spec 1.5 LANG_SLICE), FULL_STATE_OBS_DIM
    (full-state arms: state + agent-id, spec 4.1; language lives at
    STATE_LANG_SLICE), STATE_DIM (critic state, spec 1.6).
    """
    mask = torch.ones((width,), dtype=torch.bool, device=device)
    if width == L.OBS_DIM:
        mask[L.LANG_SLICE] = False
    elif width in (L.STATE_DIM, L.FULL_STATE_OBS_DIM):
        mask[L.STATE_LANG_SLICE] = False
    else:
        raise ValueError(
            "obs width %d is none of OBS_DIM=%d / STATE_DIM=%d / "
            "FULL_STATE_OBS_DIM=%d" % (width, L.OBS_DIM, L.STATE_DIM,
                                       L.FULL_STATE_OBS_DIM))
    return mask


def assert_identical_init(obs, plan):
    """``torch.equal`` on the non-language slices at t = 0 for every group
    (spec 6.3), across all lanes that share the base spawn (the spawn
    lane differs by design and is excluded)."""
    lane_spawn = plan.view(plan.spawn_alt).any(dim=1)          # (n_lanes,)
    keep = (~lane_spawn).nonzero(as_tuple=False).reshape(-1).tolist()
    assert keep, "no non-spawn lane to compare"
    ref = keep[0]
    for name, tensor in obs.items():
        v = plan.view(tensor)
        mask = nonlang_mask(v.shape[-1], device=v.device)
        base = v[ref][:, mask]
        for j in keep[1:]:
            other = v[j][:, mask]
            if not torch.equal(other, base):
                bad = (other != base).any(dim=-1).nonzero(
                    as_tuple=False).reshape(-1)
                raise AssertionError(
                    "identical-init violated (spec 6.3): agent %r, lane "
                    "%s vs %s, %d/%d groups differ on non-language dims "
                    "(first: group %d) -- _reset_idx did not write the "
                    "identical initial state into every lane"
                    % (name, LANE_NAMES[plan.lanes[j]],
                       LANE_NAMES[plan.lanes[ref]], bad.numel(),
                       plan.n_base, int(bad[0])))


# ---------------------------------------------------------------------------
# Rollout record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RolloutRecord:
    """One eval batch's recorded trajectories + terminal fields.

    Trajectory tensors are (T, E, ...) and masked to each env's FIRST
    episode (steps after ``first_done`` zeroed; ``active`` marks the live
    steps), so "the full recorded trajectory tensors" (spec 4.3) is a
    well-defined object for ``torch.equal``.
    """

    agents: Tuple[str, ...]
    mode: str
    t_steps: int                     # recorded steps (<= T_DECISION)
    positions: torch.Tensor          # (T, E, N, 2) int16 post-conflict
    intended: torch.Tensor           # (T, E, N) int8 policy action (pre-slip)
    executed: torch.Tensor           # (T, E, N) int8 post-slip (env._act)
    margins: torch.Tensor            # (T, E, N) float32 top-2 logit margin
    rewards: torch.Tensor            # (T, E, N) float32 per-agent reward
    active: torch.Tensor             # (T, E) bool  step within first episode
    first_done: torch.Tensor         # (E,) int16 step index of episode end (-1: none)
    completed: torch.Tensor          # (E,) bool  C: both latched before timeout
    correct: torch.Tensor            # (E,) bool  Y (defined where C)
    outcome: torch.Tensor            # (E,) int8  ITT O: 0 correct/1 wrong/2 incomplete
    latch_time: torch.Tensor         # (E, N) int16
    latch_slot: torch.Tensor         # (E, N) int8
    scenario_id: torch.Tensor        # (E,) long
    slip_stream: torch.Tensor        # (E,) int8
    instr_class: torch.Tensor        # (E,) long
    episode_return: torch.Tensor     # (E, N) float32 undiscounted, first episode
    n_obstacle_collisions: torch.Tensor  # (E, N) int16
    n_robot_collisions: torch.Tensor     # (E, N) int16
    logits: Optional[torch.Tensor]   # (T, E, N, A) float32 when recorded

    #: Trajectory-identity fields compared by assert_paired_lane_identity.
    #: Rewards are EXCLUDED deliberately: the Leaky arm keeps the real
    #: +-10*Y bonus (spec 1.12 [FIXED]) whose terminal value differs
    #: between complementary-instruction lanes even for a blind policy.
    IDENTITY_FIELDS = ("positions", "intended", "executed", "margins",
                       "active", "first_done", "completed", "latch_time",
                       "latch_slot")


# Absolute tolerance for the float `margins` diagnostic when the record
# lives on CPU.  CPU GEMM kernels route different row positions of one
# batch through different microkernel paths (measured on torch
# 2.7.0+cu128, oneDNN, single thread: byte-identical observation rows at
# different lane offsets yield logits differing by ~1e-5), so a float
# diagnostic derived from policy logits cannot be compared bitwise across
# lane positions on CPU.  This is policy-side float noise, not an env
# leak: the integer trajectory fields stay bitwise and are always
# compared exactly.  CUDA GEMM is row-position-invariant on this hardware
# (verified empirically), so the bitwise comparison is kept there.
MARGINS_CPU_ATOL = 1e-4


def assert_paired_lane_identity(record, plan, lane_a=LANE_FACTUAL,
                                lane_b=LANE_COUNTERFACTUAL):
    """The spec 4.3 machine check: bit-identical trajectories between two
    lanes.  For a blind arm any nonzero deviation is direct proof of an
    observation-channel leak, with zero sampling noise."""
    pa, pb = plan.lane_pos(lane_a), plan.lane_pos(lane_b)
    trajectory_fields = ("positions", "intended", "executed", "margins",
                         "active")
    for name in RolloutRecord.IDENTITY_FIELDS:
        t = getattr(record, name)
        if name in trajectory_fields:
            # (T, E, ...) -> (E, T, ...) -> lane slabs
            t = t.transpose(0, 1).contiguous()
        va = plan.view(t)[pa]
        vb = plan.view(t)[pb]
        if (name == "margins" and va.device.type == "cpu"
                and torch.allclose(va, vb, rtol=0.0,
                                   atol=MARGINS_CPU_ATOL)):
            continue
        if not torch.equal(va, vb):
            diff = (va != vb)
            while diff.dim() > 1:
                diff = diff.any(dim=-1)
            bad = diff.nonzero(as_tuple=False).reshape(-1)
            raise AssertionError(
                "paired-lane identity violated (spec 4.3): field %r "
                "differs between lanes %s and %s in %d/%d groups (first: "
                "group %d). For a blind arm this is direct proof of an "
                "observation-channel leak." % (
                    name, LANE_NAMES[lane_a], LANE_NAMES[lane_b],
                    bad.numel(), va.shape[0], int(bad[0])))


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def run_lanes(env, policy, plan=None, mode="argmax", generator=None,
              max_steps=None, assert_init=True, auto_assert_blind=True,
              record_logits=False, mask_after_done=True):
    """Roll one eval batch and return a :class:`RolloutRecord`.

    Args:
        env:      TeamGridEnv (pass ``env.unwrapped`` on the 5090) or
                  CPUFleetEnv; see ``resolve_env``.
        policy:   callable ``obs_dict -> {agent: (E, 5) logits}``.  Give it
                  Categorical LOGITS; action selection happens here
                  (spec 4.3 -- never a mean_actions lookup).
        plan:     optional :class:`LanePlan`; None rolls the env's own
                  reset draw with no forcing, no interventions, no lane
                  assertions (plain rollout mode).
        mode:     "argmax" (deterministic, the eval-manifest mode) or
                  "stochastic" (Categorical sampling via ``generator``).
        generator: torch.Generator for stochastic mode (reproducibility).
        max_steps: cap on decision steps (default T_DECISION = 128).
        assert_init: run the spec 6.3 identical-init assertion.
        auto_assert_blind: in argmax mode on a blind arm
                  (``cfg.lang_gain == 0``), automatically run the paired
                  trajectory-equality assertion for every lane that shares
                  L0's physics (L1, and L4 when present).
        record_logits: additionally keep the full (T, E, N, 5) logits.
        mask_after_done: zero recorded steps after each env's first
                  episode (keeps "full trajectory tensors" well-defined).
    """
    env = resolve_env(env)
    agents = list(env.cfg.possible_agents)
    N = len(agents)
    E = env.num_envs
    dev = env.device
    if mode not in MODES:
        raise ValueError("mode %r not in %r" % (mode, MODES))

    if plan is not None:
        plan.validate()
        if plan.n_envs != E:
            raise ValueError(
                "plan covers %d envs (%d lanes x %d base scenarios) but "
                "env.num_envs=%d" % (plan.n_envs, plan.n_lanes,
                                     plan.n_base, E))
        env.force_scenarios(plan.scenarios.to(dev, torch.long))
        env.force_slip_stream(plan.streams.to(dev, torch.int8))
    try:
        env._reset_idx(None)
        if plan is not None:
            rows = plan.instr_rows
            if rows is None:
                rows = default_instruction_rows(env, plan.instr_classes)
            force_instructions(env, rows, plan.instr_classes)
            apply_spawn_alt(env, plan.spawn_alt)
            blank_language(env, plan.blank)     # after arm-consistency check
        obs = env._get_observations()
        if plan is not None and assert_init:
            assert_identical_init(obs, plan)

        # snapshots (reset-time; instr/slip are per-episode constants in M1)
        scenario_id = env.scenario_id.clone()
        slip_stream = env.slip_stream.clone()
        instr_class = env.instr_class.clone()

        T = int(max_steps) if max_steps is not None else L.T_DECISION
        assert 1 <= T <= L.T_DECISION

        positions = torch.zeros((T, E, N, 2), dtype=torch.int16, device=dev)
        intended = torch.zeros((T, E, N), dtype=torch.int8, device=dev)
        executed = torch.zeros((T, E, N), dtype=torch.int8, device=dev)
        margins = torch.zeros((T, E, N), dtype=torch.float32, device=dev)
        rewards = torch.zeros((T, E, N), dtype=torch.float32, device=dev)
        active = torch.zeros((T, E), dtype=torch.bool, device=dev)
        logits_rec = (torch.zeros((T, E, N, grid_core.N_ACTIONS),
                                  dtype=torch.float32, device=dev)
                      if record_logits else None)

        done_before = torch.zeros((E,), dtype=torch.bool, device=dev)
        first_done = torch.full((E,), -1, dtype=torch.int16, device=dev)
        completed = torch.zeros((E,), dtype=torch.bool, device=dev)
        n_obstacle = torch.zeros((E, N), dtype=torch.int16, device=dev)
        n_robot = torch.zeros((E, N), dtype=torch.int16, device=dev)
        t_run = 0

        for t in range(T):
            logit_d = policy(obs)
            lg = torch.stack([logit_d[a].float() for a in agents], dim=1)
            assert lg.shape == (E, N, grid_core.N_ACTIONS), lg.shape
            if mode == "paired_stochastic":
                n_u = plan.n_base if plan is not None else E
                u = torch.rand((n_u, N), generator=generator)
                if plan is not None:
                    u = u.repeat(plan.n_lanes, 1)
                chosen = select_actions(lg, mode, paired_u=u.to(dev)).long()
            else:
                chosen = select_actions(lg, mode, generator).long()  # (E, N)
            env._pre_physics_step(
                {a: chosen[:, i] for i, a in enumerate(agents)})
            env.episode_length_buf += 1                # BEFORE dones, spec 1.4
            term_d, trunc_d = env._get_dones()
            rew_d = env._get_rewards()
            term = term_d[agents[0]]
            trunc = trunc_d[agents[0]]

            alive = ~done_before
            m = alive if mask_after_done else torch.ones_like(alive)
            mN = m.unsqueeze(-1)
            positions[t] = torch.where(mN.unsqueeze(-1), env.pos[:, :N],
                                       torch.zeros_like(env.pos[:, :N]))
            intended[t] = chosen.to(torch.int8) * mN
            executed[t] = env._act.to(torch.int8) * mN
            margins[t] = top2_margin(lg) * mN
            rewards[t] = torch.stack([rew_d[a].float() for a in agents],
                                     dim=1) * mN
            active[t] = m
            n_obstacle += (env._hit_obstacle & mN).to(torch.int16)
            n_robot += (env._hit_robot & mN).to(torch.int16)
            if logits_rec is not None:
                logits_rec[t] = lg * mN.unsqueeze(-1)

            newly = alive & (term | trunc)
            first_done[newly] = t
            completed |= alive & term          # terminated == both latched
            done_before |= term | trunc
            t_run = t + 1
            if bool(done_before.all()):
                break
            obs = env._get_observations()

        # terminal fields: no auto-reset happened, so latch state is intact
        y = env._outcome_correct() & completed
        outcome = torch.where(
            completed,
            torch.where(y, torch.zeros_like(first_done, dtype=torch.int8),
                        torch.ones_like(first_done, dtype=torch.int8)),
            torch.full((E,), 2, dtype=torch.int8, device=dev))

        record = RolloutRecord(
            agents=tuple(agents),
            mode=mode,
            t_steps=t_run,
            positions=positions[:t_run],
            intended=intended[:t_run],
            executed=executed[:t_run],
            margins=margins[:t_run],
            rewards=rewards[:t_run],
            active=active[:t_run],
            first_done=first_done,
            completed=completed,
            correct=y,
            outcome=outcome,
            latch_time=env.latch_time[:, :N].clone(),
            latch_slot=env.latch_slot[:, :N].clone(),
            scenario_id=scenario_id,
            slip_stream=slip_stream,
            instr_class=instr_class,
            episode_return=rewards[:t_run].sum(dim=0),
            n_obstacle_collisions=n_obstacle,
            n_robot_collisions=n_robot,
            logits=logits_rec[:t_run] if logits_rec is not None else None,
        )
    finally:
        if plan is not None:
            env.force_scenarios(None)
            env.force_slip_stream(None)

    if plan is not None and auto_assert_blind \
            and mode in ("argmax", "paired_stochastic") \
            and float(getattr(env.cfg, "lang_gain", 1.0)) == 0.0:
        # spec 4.3: every blind-arm pair must be bit-identical; L4 shares
        # L0's physics too (blank == zero == blind slice).
        if LANE_COUNTERFACTUAL in plan.lanes:
            assert_paired_lane_identity(record, plan, LANE_FACTUAL,
                                        LANE_COUNTERFACTUAL)
        if LANE_BLANK in plan.lanes:
            assert_paired_lane_identity(record, plan, LANE_FACTUAL,
                                        LANE_BLANK)
    return record


# ---------------------------------------------------------------------------
# Per-episode record emission (spec 6.6 consumers: harness/records.py)
# ---------------------------------------------------------------------------

def episode_records(record, plan=None):
    """Flatten a RolloutRecord into per-episode (E,) tensors -- the spec
    6.6 scalar columns this driver owns.  Blob columns
    (action_logits_blob, realised_positions_blob, ...) are the record's
    trajectory tensors; parquet serialisation is records.py's job."""
    E = record.scenario_id.numel()
    T_ep = torch.where(record.first_done >= 0,
                       record.first_done.long() + 1,
                       torch.full_like(record.first_done.long(), -1))
    out = {
        "scenario_id": record.scenario_id,
        "slip_stream": record.slip_stream.long(),
        "instr_class": record.instr_class,
        "T": T_ep,
        "C": record.completed.long(),
        "Y": record.correct.long(),
        "O_itt": record.outcome.long(),
    }
    for i, name in enumerate(record.agents):
        out["latch_time_%s" % name] = record.latch_time[:, i].long()
        out["latch_slot_%s" % name] = record.latch_slot[:, i].long()
        out["return_%s" % name] = record.episode_return[:, i]
        out["n_obstacle_collisions_%s" % name] = \
            record.n_obstacle_collisions[:, i].long()
        out["n_robot_collisions_%s" % name] = \
            record.n_robot_collisions[:, i].long()
    if plan is not None:
        out["lane"] = plan.lane_of_env().to(record.scenario_id.device)
        out["group"] = plan.group_of_env().to(record.scenario_id.device)
    for key, value in out.items():
        assert value.numel() == E, key
    return out


__all__ = [
    "LANE_FACTUAL", "LANE_COUNTERFACTUAL", "LANE_SEED", "LANE_SPAWN",
    "LANE_BLANK", "LANE_NAMES", "N_LANES_FULL", "MODES",
    "LanePlan", "make_paired_plan", "make_five_lane_plan",
    "select_actions", "top2_margin", "resolve_env",
    "default_instruction_rows", "force_instructions", "blank_language",
    "apply_spawn_alt", "nonlang_mask", "assert_identical_init",
    "RolloutRecord", "assert_paired_lane_identity", "run_lanes",
    "episode_records",
]
