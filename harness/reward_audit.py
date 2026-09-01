"""Exact bank-computed compliance bound (M1_SPEC 1.12 / 3.4 / section 7).

[FIXED: the reward-fairness bound was double-counted and hand-waved.]  The
old draft quoted a hand-estimated "+5.0 left on the table"; this module
computes the bound the paper quotes **from the bank, exactly**: for every
bank row and every instruction class it solves the optimal compliant and
the optimal non-compliant (defecting) plan by BFS under the TRUE reward and
gamma, and ``assert_compliance_bound`` asserts

    min over rows k, classes c of  (G_comply[k, c] - G_defect[k, c]) > margin.

Why a closed form is exact here
-------------------------------
Under the spec 1.12 reward the gamma-correct shaping
``lambda * (gamma * Phi_t - Phi_{t-1})`` telescopes over any trajectory that
terminates (both robots latched, ``Phi(terminal) == 0``) to

    lambda * (gamma^T * 0 - Phi_0)  =  lambda * M0,

where ``M0 = -Phi_0`` is the min-cost perfect matching of latch-aware BFS
distances at spawn -- PATH-INDEPENDENT, identical for the compliant and the
defecting plan of the same row.  An optimal plan is collision-free (both
collision terms are strictly negative and never shorten a latch-aware BFS
path), so the discounted return of an optimal plan completing on decision
step T (1-based; terminal reward discounted by gamma^(T-1), matching
``fleet_env``'s 0-based ``episode_length_buf`` latch stamping) is exactly

    G(T, B) = STEP_COST * (1 - gamma^T) / (1 - gamma)          # step bleed
            + shaping_lambda * M0                              # telescoped
            + gamma^(T-1) * B                                  # terminal

with ``B = COMPLETION_BONUS + OUTCOME_BONUS`` (= 2 + 10) for the compliant
plan and ``B = COMPLETION_BONUS`` (= 2) for the defecting plan.  G is
strictly decreasing in T (each extra step costs ``-STEP_COST * gamma^T +
B * gamma^(T-1) * (1 - gamma) > 0``), so each side's optimum is its MINIMAL
feasible completion time, which is what BFS delivers:

* Role binding (``mouth == (-1, -1)``): assignment a maps robot i to a
  distinct station; ``T(a) = max_i d[i, a(i)]`` with ``d`` the bank's
  latch-aware BFS distances gathered at the spawn cells.  Class c fixes the
  compliant assignment (RB0: robot_0 -> the LEFT station = lexicographic
  min by (col, row); RB1: the flip); the defecting plan uses the other one.
* Precedence (valid ``mouth``): every latch transits the unique mouth m and
  both stations are leaves on m, so ``d[i, j] = d(spawn_i, m) + 1`` for
  both j (asserted) and ``dm_i = d[i, 0] - 1``.  With convoy motion legal
  (spec 1.8 ``test_follow_vacated_cell_allowed``) and the mouth a hard
  one-at-a-time gate, ordering "F first, S second" completes in

      T(F) = max(dm_S, dm_F + 1) + 1

  (S either paces itself into perfect convoy one cell behind F, or -- the
  yield case dm_S <= dm_F -- waits and enters the mouth on the step F
  vacates it into its station).  Class PR0: robot_0 first; PR1: the flip.

"Defect by never completing" is dominated, and machine-checked: any
non-completing plan's return is at most ``STEP_COST * annuity(T_DECISION) +
shaping_lambda * M0`` (since ``Phi_T <= 0``), which ``compliance_bound``
asserts is strictly below every ``G_defect``; completing later than the
minimal T is dominated because G is decreasing in T.  So the defecting
optimum really is the fastest wrong completion.

Model assumptions (the bank builder's guarantees, spec 2.1 / 3.1, verified
by ``tests/test_bank_latch_reachability.py`` / ``test_bank_distfields.py``,
NOT re-proved here): shortest paths of the two robots can be realised
without mutual blocking (alcove topology + per-scenario spawn constraints
for role binding; "both orderings are always feasible" for precedence).
This module additionally requires every gathered distance to be reachable,
both completion times to fit inside ``T_DECISION``, spawns off the mouth,
and ``delta_gap`` consistent with the recomputed ``dm`` on precedence rows.

The instruction enters ONLY as the class index c selecting which completion
time is "compliant" -- never through ``lang_vec`` / ``instr_id`` /
``leak_bit`` -- so the audit itself is leak-clean.  (The trajectory-level
joint-involution invariance check of spec 2.4 rides on sampled rollouts and
ships with the rollout harness, not here.)

Usage::

    from harness import reward_audit
    audit = reward_audit.compliance_bound_file("data/scenario_bank_....pt")
    print(audit.summary())          # audit.bound is the number the paper quotes
    reward_audit.assert_compliance_bound(bank, margin=0.0)

    python -m harness.reward_audit data/scenario_bank_RoleBinding_<sha>.pt

Pure torch + stdlib; runs on the dev box (no Isaac, no simulator).
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:            # direct-script execution only
    sys.path.insert(0, str(_REPO_ROOT))

from tasks.team_listen import obs_layout as L
from tasks.team_listen import rewards
from tasks.team_listen import scenario_bank

#: Class labels per row kind, index 0/1 == class 0/1 (spec 2.2 / 3.2).
#: Class 0 == RB0 "robot_0 -> left station" / PR0 "robot_0 first";
#: class 1 is the flip.  "Left" = lexicographic min by (col, row); the
#: bound ``min`` over both classes is invariant to this labelling.
RB_CLASSES = ("RB0", "RB1")
PR_CLASSES = ("PR0", "PR1")

#: Terminal bonus of the optimal compliant / defecting plan (spec 1.12).
COMPLY_BONUS = rewards.COMPLETION_BONUS + rewards.OUTCOME_BONUS   # 2 + 10
DEFECT_BONUS = rewards.COMPLETION_BONUS                           # 2 (Y = 0)


def _require(ok, why):
    if not ok:
        raise RuntimeError("reward_audit: %s (M1_SPEC 1.12 / 3.4)" % (why,))


def annuity(t, gamma):
    """``(1 - gamma^t) / (1 - gamma)``: discounted sum of t unit payments."""
    return (1.0 - gamma ** t) / (1.0 - gamma)


def plan_return(t_complete, terminal_bonus, matching0, gamma, shaping_lambda,
                latch_times=None):
    """Exact discounted return G(T, B) of an optimal collision-free plan.

    Args:
        t_complete:     integer tensor of completion steps T (1-based).
        terminal_bonus: scalar B -- ``COMPLY_BONUS`` or ``DEFECT_BONUS``.
        matching0:      float tensor M0 (broadcastable) -- spawn min-matching
                        cost, i.e. ``-Phi_0``.
        gamma:          discount in (0, 1).
        shaping_lambda: spec 1.12 lambda.
        latch_times:    optional (..., 2) integer tensor of the two agents'
                        1-based latch steps under this plan; when given the
                        DECISIONS.md first-latch amendment adds
                        ``FIRST_LATCH_BONUS * sum_i gamma^(tau_i - 1)``.
                        Unlike the terminal bonus this term does NOT cancel
                        in ``delta``: the compliant plan latches later
                        (yield cost), so the amendment strictly REDUCES the
                        compliance bound via discounting.

    Returns:
        float64 tensor ``STEP_COST * annuity(T) + lambda * M0
        + gamma^(T-1) * B  [+ latch term]`` (module docstring derivation).
    """
    t = t_complete.to(torch.float64)
    g = (rewards.STEP_COST * annuity(t, gamma)
         + shaping_lambda * matching0.to(torch.float64)
         + (gamma ** (t - 1.0)) * float(terminal_bonus))
    if latch_times is not None:
        # Per-agent MEAN: the broadcast G is a per-agent return, and the
        # latch bonus is the one per-agent-timed term, so the mean keeps G
        # in per-agent units (== either agent's return when latches are
        # simultaneous).  The BOUND uses the stricter min-over-agents delta
        # computed in compliance_bound, not this mean.
        tau = latch_times.to(torch.float64)
        g = g + rewards.FIRST_LATCH_BONUS * (gamma ** (tau - 1.0)).mean(dim=-1)
    return g


@dataclasses.dataclass
class ComplianceAudit:
    """Result of :func:`compliance_bound` over one bank.

    ``bound`` (== ``delta.min()``) is the number the paper quotes.  All
    per-row tensors are CPU; class index c is 0 = RB0/PR0, 1 = RB1/PR1.
    """

    bound: float                    # min_k,c (G_comply - G_defect)
    argmin_row: int                 # bank row attaining the bound
    argmin_class: int               # class index attaining the bound
    delta: torch.Tensor             # (K, 2) float64  G_comply - G_defect
    g_comply: torch.Tensor          # (K, 2) float64
    g_defect: torch.Tensor          # (K, 2) float64
    t_comply: torch.Tensor          # (K, 2) int64 completion steps
    t_defect: torch.Tensor          # (K, 2) int64
    matching0: torch.Tensor         # (K,)   float64  M0 = -Phi_0 at spawn
    is_precedence: torch.Tensor     # (K,)   bool
    gamma: float
    shaping_lambda: float

    @property
    def k(self):
        return int(self.delta.shape[0])

    def class_label(self, row, cls):
        names = PR_CLASSES if bool(self.is_precedence[row]) else RB_CLASSES
        return names[cls]

    def summary(self):
        n_pr = int(self.is_precedence.long().sum())
        r, c = self.argmin_row, self.argmin_class
        lines = [
            "reward_audit compliance bound (M1_SPEC 1.12 / 3.4)",
            "  rows: %d  (%d RoleBinding, %d Precedence)"
            % (self.k, self.k - n_pr, n_pr),
            "  gamma=%.6g  shaping_lambda=%.6g" % (self.gamma,
                                                   self.shaping_lambda),
            "  bound = min_k,c (G_comply - G_defect) = %.12g" % self.bound,
            "  attained at row %d, class %s: G_comply=%.12g (T=%d), "
            "G_defect=%.12g (T=%d)"
            % (r, self.class_label(r, c), float(self.g_comply[r, c]),
               int(self.t_comply[r, c]), float(self.g_defect[r, c]),
               int(self.t_defect[r, c])),
            "  delta over rows x classes: mean=%.6g  max=%.6g"
            % (float(self.delta.mean()), float(self.delta.max())),
            "  completion steps: comply max=%d  defect max=%d  (cap T=%d)"
            % (int(self.t_comply.max()), int(self.t_defect.max()),
               L.T_DECISION),
        ]
        return "\n".join(lines)


def spawn_distances(bank):
    """(K, 2, 2) int64: ``d[k, i, j]`` = latch-aware BFS distance of live
    robot i's spawn to valid station j, gathered from ``bank.dist_field``."""
    spawn = bank.spawn[:, :L.N_AGENTS].long()                    # (K, 2, 2)
    k = spawn.shape[0]
    idx = spawn[..., 0] * L.C + spawn[..., 1]                    # (K, 2)
    flat = bank.dist_field.long().reshape(k, L.MAX_TARGETS, L.R * L.C)
    d_all = flat.gather(2, idx.unsqueeze(1).expand(k, L.MAX_TARGETS,
                                                   L.N_AGENTS))
    return d_all[:, :2].permute(0, 2, 1).contiguous()            # (K, 2, 2)


def compliance_bound(bank, gamma=rewards.REWARD_GAMMA,
                     shaping_lambda=rewards.SHAPING_LAMBDA):
    """Compute the exact per-row, per-class compliance bound over a bank.

    Args:
        bank: a :class:`tasks.team_listen.scenario_bank.ScenarioBank` (or
              any object with ``spawn``, ``target``, ``target_valid``,
              ``dist_field``, ``mouth``, ``delta_gap`` tensors of the spec
              1.10 schema, plus an optional ``meta`` dict).
        gamma:          discount, must lie in (0, 1) (spec 1.12: 0.99).
        shaping_lambda: shaping coefficient (spec 1.12: 0.1, swept OPEN(5)).
                        NOTE: lambda cancels in ``delta`` (the telescoped
                        shaping is identical on both sides); it only shifts
                        the reported absolute returns.

    Returns:
        :class:`ComplianceAudit`.

    Raises:
        RuntimeError: on any violated model precondition (docstring list).
    """
    gamma = float(gamma)
    shaping_lambda = float(shaping_lambda)
    _require(0.0 < gamma < 1.0, "gamma must lie in (0, 1); got %r" % gamma)

    meta = getattr(bank, "meta", {}) or {}
    n_agents = int(meta.get("n_agents", L.N_AGENTS))
    _require(n_agents == 2,
             "closed-form bound covers exactly N=2 live agents (spec 1.8 "
             "scope); bank has n_agents=%d" % n_agents)

    tv = bank.target_valid.to("cpu")
    k = int(tv.shape[0])
    _require(k >= 1, "empty bank")
    _require(bool(tv[:, :2].all()) and bool((~tv[:, 2:]).all()),
             "closed-form bound covers exactly 2 valid stations per row "
             "(M1); found rows with != 2 valid targets")

    d = spawn_distances(bank).to("cpu")                          # (K, 2, 2)
    _require(bool((d >= 0).all()),
             "a spawn cell is unreachable from a station in the latch-aware "
             "field (dist_field sentinel < 0 gathered at spawn)")

    mouth = bank.mouth.long().to("cpu")
    is_pr = ~(mouth == -1).all(dim=1)                            # (K,)

    # -- role binding: T(assignment) = max of the two BFS distances --------
    t_a0 = torch.maximum(d[:, 0, 0], d[:, 1, 1])                 # r0->s0, r1->s1
    t_a1 = torch.maximum(d[:, 0, 1], d[:, 1, 0])                 # r0->s1, r1->s0
    tgt = bank.target[:, :2].long().to("cpu")
    lexi = tgt[..., 1] * L.R + tgt[..., 0]                       # (col, row) key
    slot0_left = lexi[:, 0] < lexi[:, 1]                         # station 0 is LEFT
    t_rb_c0 = torch.where(slot0_left, t_a0, t_a1)                # RB0 compliant
    t_rb_c1 = torch.where(slot0_left, t_a1, t_a0)                # RB1 compliant

    # -- precedence: unique-mouth serialisation ----------------------------
    if bool(is_pr.any()):
        _require(bool((d[is_pr, :, 0] == d[is_pr, :, 1]).all()),
                 "precedence rows must satisfy d(spawn, s0) == d(spawn, s1) "
                 "(every latch transits the unique mouth, spec 3.1)")
    dm = d[:, :, 0] - 1                                          # (K, 2)
    if bool(is_pr.any()):
        _require(bool((dm[is_pr] >= 1).all()),
                 "a precedence spawn sits on or beside the mouth "
                 "(d(spawn, m) < 1); the convoy model needs spawns off the "
                 "mouth (bank builder enforces d >= 3, spec 3.1)")
        gap = bank.delta_gap.long().to("cpu")
        _require(bool(((dm[:, 0] - dm[:, 1]) == gap)[is_pr].all()),
                 "bank delta_gap disagrees with d(r0,m) - d(r1,m) "
                 "recomputed from dist_field (bank drift)")
    t_pr_c0 = torch.maximum(dm[:, 1], dm[:, 0] + 1) + 1          # PR0: r0 first
    t_pr_c1 = torch.maximum(dm[:, 0], dm[:, 1] + 1) + 1          # PR1: r1 first

    t_comply = torch.stack([torch.where(is_pr, t_pr_c0, t_rb_c0),
                            torch.where(is_pr, t_pr_c1, t_rb_c1)], dim=1)
    t_defect = t_comply.flip(dims=[1])       # defect == the other class's plan

    # -- per-agent latch times per plan (first-latch amendment) ------------
    # role binding, assignment A0 (r0->s0, r1->s1) / A1 (r0->s1, r1->s0):
    lat_a0 = torch.stack([d[:, 0, 0], d[:, 1, 1]], dim=-1)       # (K, 2)
    lat_a1 = torch.stack([d[:, 0, 1], d[:, 1, 0]], dim=-1)
    sl = slot0_left.unsqueeze(-1)
    lat_rb_c0 = torch.where(sl, lat_a0, lat_a1)
    lat_rb_c1 = torch.where(sl, lat_a1, lat_a0)
    # precedence, class 0 (r0 first): r0 latches at dm0+1, r1 at T
    lat_pr_c0 = torch.stack([dm[:, 0] + 1, t_pr_c0], dim=-1)
    lat_pr_c1 = torch.stack([t_pr_c1, dm[:, 1] + 1], dim=-1)
    pr = is_pr.unsqueeze(-1)
    lat_c0 = torch.where(pr, lat_pr_c0, lat_rb_c0)               # (K, 2)
    lat_c1 = torch.where(pr, lat_pr_c1, lat_rb_c1)
    lat_comply = torch.stack([lat_c0, lat_c1], dim=1)            # (K, 2cls, 2ag)
    lat_defect = lat_comply.flip(dims=[1])
    _require(bool((t_comply <= L.T_DECISION).all())
             and bool((t_defect <= L.T_DECISION).all()),
             "a minimal completion plan exceeds T_DECISION=%d steps; the "
             "bound is undefined on rows that cannot finish in-horizon"
             % L.T_DECISION)

    # -- spawn min-matching M0 = -Phi_0 (cancels in delta; kept for G) -----
    m0 = torch.minimum(d[:, 0, 0] + d[:, 1, 1],
                       d[:, 0, 1] + d[:, 1, 0]).to(torch.float64)

    g_comply = plan_return(t_comply, COMPLY_BONUS, m0.unsqueeze(1),
                           gamma, shaping_lambda, latch_times=lat_comply)
    g_defect = plan_return(t_defect, DEFECT_BONUS, m0.unsqueeze(1),
                           gamma, shaping_lambda, latch_times=lat_defect)

    # -- machine-check the dominance argument: completing (even wrongly)
    #    beats any never-completing plan, whose return is bounded above by
    #    full-horizon step bleed + lambda * M0 (Phi_T <= 0).
    # a never-completing plan can still latch ONE robot (both latched ==
    # done), so the timeout upper bound gains at most one undiscounted
    # first-latch bonus under the amendment.
    g_timeout_ub = (rewards.STEP_COST
                    * annuity(torch.tensor(float(L.T_DECISION),
                                           dtype=torch.float64), gamma)
                    + shaping_lambda * m0
                    + rewards.FIRST_LATCH_BONUS)
    _require(bool((g_defect > g_timeout_ub.unsqueeze(1)).all()),
             "internal dominance check failed: a defecting completion did "
             "not beat the timeout upper bound -- formula edited?")

    # Bound convention (first-latch amendment): each agent's OWN return is
    # the broadcast scalar plus ITS discounted latch bonus, so "compliance
    # is unambiguously optimal" must hold PER AGENT.  delta is therefore
    # the min-over-agents per-agent difference -- at least as tight as the
    # mean-based ``g_comply - g_defect`` (equal when latches are
    # simultaneous on both plans).
    disc_c = torch.pow(gamma, lat_comply.to(torch.float64) - 1.0)
    disc_d = torch.pow(gamma, lat_defect.to(torch.float64) - 1.0)
    broadcast_delta = (
        (g_comply - rewards.FIRST_LATCH_BONUS * disc_c.mean(dim=-1))
        - (g_defect - rewards.FIRST_LATCH_BONUS * disc_d.mean(dim=-1)))
    delta = broadcast_delta + rewards.FIRST_LATCH_BONUS * (
        (disc_c - disc_d).min(dim=-1).values)
    flat_arg = int(delta.reshape(-1).argmin())
    return ComplianceAudit(
        bound=float(delta.reshape(-1)[flat_arg]),
        argmin_row=flat_arg // 2,
        argmin_class=flat_arg % 2,
        delta=delta, g_comply=g_comply, g_defect=g_defect,
        t_comply=t_comply, t_defect=t_defect, matching0=m0,
        is_precedence=is_pr, gamma=gamma, shaping_lambda=shaping_lambda,
    )


def compliance_bound_file(path, gamma=rewards.REWARD_GAMMA,
                          shaping_lambda=rewards.SHAPING_LAMBDA,
                          arm=None, expected_sha=""):
    """Load a bank artifact through the production loader (all spec 1.10
    integrity gates included) and compute :func:`compliance_bound` on it.

    ``arm`` is forwarded to :func:`scenario_bank.load_bank`; auditing a
    leaky bank therefore requires passing ``arm="Leaky"`` explicitly, the
    same gate training obeys (spec 4.1).
    """
    bank = scenario_bank.load_bank(path, device="cpu", arm=arm,
                                   expected_sha=expected_sha)
    return compliance_bound(bank, gamma=gamma, shaping_lambda=shaping_lambda)


def assert_compliance_bound(bank, margin=0.0, gamma=rewards.REWARD_GAMMA,
                            shaping_lambda=rewards.SHAPING_LAMBDA):
    """Compute the bound and assert ``min_k,c (G_comply - G_defect) > margin``
    (spec 1.12 [FIXED: reward-fairness bound]).  Returns the audit."""
    audit = compliance_bound(bank, gamma=gamma, shaping_lambda=shaping_lambda)
    if not audit.bound > float(margin):
        r, c = audit.argmin_row, audit.argmin_class
        raise RuntimeError(
            "reward_audit: compliance bound %.12g <= margin %.12g at bank "
            "row %d, class %s (G_comply=%.12g @T=%d vs G_defect=%.12g "
            "@T=%d) -- compliance is NOT unambiguously optimal under the "
            "true reward (M1_SPEC 1.12 / 3.4)"
            % (audit.bound, float(margin), r, audit.class_label(r, c),
               float(audit.g_comply[r, c]), int(audit.t_comply[r, c]),
               float(audit.g_defect[r, c]), int(audit.t_defect[r, c])))
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact bank-computed compliance bound (M1_SPEC 1.12): "
                    "min over rows and instruction classes of "
                    "G_comply - G_defect under the true reward and gamma.")
    parser.add_argument("bank", help="scenario bank .pt artifact (spec 1.10)")
    parser.add_argument("--gamma", type=float, default=rewards.REWARD_GAMMA)
    parser.add_argument("--shaping-lambda", type=float,
                        default=rewards.SHAPING_LAMBDA, dest="shaping_lambda")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="fail (exit 1) unless bound > margin")
    parser.add_argument("--arm", default=None,
                        help="requesting arm for the loader's leaky-bank "
                             "gate (spec 4.1); leaky banks need --arm Leaky")
    parser.add_argument("--expected-sha", default="",
                        help="optional expected bank SHA-256 (prefix ok)")
    args = parser.parse_args(argv)

    audit = compliance_bound_file(args.bank, gamma=args.gamma,
                                  shaping_lambda=args.shaping_lambda,
                                  arm=args.arm,
                                  expected_sha=args.expected_sha)
    print(audit.summary())
    ok = audit.bound > args.margin
    print("RESULT: %s (bound %.12g %s margin %.12g)"
          % ("PASS" if ok else "FAIL", audit.bound,
             ">" if ok else "<=", args.margin))
    return 0 if ok else 1


__all__ = [
    "RB_CLASSES", "PR_CLASSES", "COMPLY_BONUS", "DEFECT_BONUS",
    "annuity", "plan_return", "spawn_distances", "ComplianceAudit",
    "compliance_bound", "compliance_bound_file", "assert_compliance_bound",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
