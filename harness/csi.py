"""CSI v1: the Coordination Sensitivity Index (M1_SPEC section 6).

Estimand (spec 6.1 [FIXED: CSI_flip's ratio was a divergent estimand]):

* **Primary, bounded, monotone:**
  ``CSI_share = D_lang / (D_lang + D_nuis)`` in [0, 1], equal to 0.5 when
  language and nuisance are equally influential.  This IS the spec's
  degenerate-denominator regularisation: a grounded policy that is
  invariant to the nuisance intervention drives ``D_nuis -> 0``, where
  the raw ratio diverges with no finite mean -- precisely the regime the
  project exists to detect -- while the share saturates at 1, finitely.
* **Display-only:** the ratio ``CSI_ratio = D_lang / D_nuis`` as a Hajek
  ratio of means with both terms recomputed inside each bootstrap
  replicate (spec 6.4).  It is NEVER silently epsilon-regularised: a zero
  pooled nuisance sum yields ``inf`` (or NaN at 0/0) plus an explicit
  ``nuisance_degenerate`` flag, so no result-conditional fallback can
  hide in the estimator.  ``frac_nuis_zero`` is reported alongside as the
  OPEN(12) concentration diagnostic ("if D_seed concentrates at zero,
  report CSI_share against D_spawn as primary and say so BEFORE
  unblinding" -- switching ``nuisance='spawn'`` here).

``D_lang`` is the counterfactual-instruction divergence (lane L1 vs L0);
the definitional physics-seed denominator ``D_seed`` is lane L2 (slip
stream 0 -> 1, everything else frozen -- spec 1.10 [FIXED: no physics
seed]); ``D_spawn`` is L3.  ``D_blank`` (L4) is published as a THIRD
COLUMN, never a ratio denominator (spec 6.1: the LIBERO-Plus asymmetry
argument), so it is a marginal here and not a ``nuisance`` choice.
Marginal distributions of every divergence are returned raw, not only as
summaries (spec 6.1).

Mode discipline (spec 6.2): numerator and denominator of any contrast
must use the SAME action-selection mode.  Both are computed from lanes of
ONE :class:`harness.rollout.RolloutRecord`, which was rolled in a single
mode, so the discipline is structural rather than procedural.

Inference: episode-clustered bootstrap percentile CIs (the base scenario
is the clustering unit, spec 4.3 ``clustering_unit: base_scenario``) and
the paired sign/permutation test of spec 6.4(a), randomising the sign of
each episode's paired difference ``D_lang,i - D_nuis,i`` within the
frozen pairing.  Placebo calibration (spec 6.4(b)) is not special code:
run this same pipeline on the ``Placebo`` arm's record; a calibrated
``CSI_share`` sits at 0.5 there and the ratio form at 1.

NOTE ON THE BOOTSTRAP: the spec 7 file plan puts the shared
``hierarchical_bootstrap`` in ``harness/stats.py``, which does not exist
yet.  ``_bootstrap`` below is a LOCAL, episode(cluster)-level percentile
bootstrap -- clearly scoped so it can be unified with (or replaced by)
``harness.stats`` when that module lands.  The two-level scenarios-x-seeds
hierarchical interval of spec 4.4 operates ACROSS records (one per
training seed) and belongs there, not here.
"""

import dataclasses
import os
import sys
from typing import Dict, Optional, Tuple

import torch

try:
    from harness import metrics
    from harness import rollout as ro
except ImportError:  # standalone import: put the repo root on sys.path
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from harness import metrics
    from harness import rollout as ro

#: Divergence kinds (the spec 6.1 CSI variants, on the record's objects):
#: "traj" = CSI_traj (per-step mean position divergence), "flip" =
#: CSI_flip (realised-outcome flip), "tv_step"/"tv_joint" = CSI_step
#: (per-agent-marginal / factored-joint TV between action distributions;
#: needs ``record.logits``), "intent" = CSI_intent's object (Hamming on
#: pre-revert intended actions), "executed" = its realised counterpart.
DIVERGENCES = ("traj", "flip", "tv_step", "tv_joint", "intent", "executed")

#: Ratio-denominator lanes (spec 6.1).  "blank" is deliberately ABSENT:
#: D_blank is a third column, never a denominator.
NUISANCE_LANES = {"seed": ro.LANE_SEED, "spawn": ro.LANE_SPAWN}

#: Marginals published when the lane is present in the plan (spec 6.1).
_MARGINAL_LANES = (("d_lang", ro.LANE_COUNTERFACTUAL),
                   ("d_seed", ro.LANE_SEED),
                   ("d_spawn", ro.LANE_SPAWN),
                   ("d_blank", ro.LANE_BLANK))


# ---------------------------------------------------------------------------
# Lane extraction (record layout of harness/rollout.py)
# ---------------------------------------------------------------------------

def lane_tensor(x, plan, lane, trajectory=False):
    """Slice one lane's slab out of a record tensor.

    ``trajectory=True`` for step-major (T, E, ...) tensors -> (T, K, ...);
    otherwise (E, ...) -> (K, ...).  K = plan.n_base.
    """
    if trajectory:
        x = x.transpose(0, 1)
    v = plan.view(x)[plan.lane_pos(lane)]
    if trajectory:
        v = v.transpose(0, 1).contiguous()
    return v


def lane_divergence(record, plan, lane_b, divergence, lane_a=ro.LANE_FACTUAL,
                    variant=None, norm="l1", decision_mask=None):
    """(K,) per-base-scenario divergence between two lanes of one record.

    NaN marks per-episode degeneracy (no common live step under the
    decision mask; an incomplete lane for "flip") -- see harness.metrics.

    Args:
        record: harness.rollout.RolloutRecord (or duck-typed equal).
        plan: the LanePlan the record was rolled under.
        lane_b: intervention lane id (e.g. ro.LANE_COUNTERFACTUAL).
        divergence: one of DIVERGENCES.
        lane_a: reference lane (default L0 factual).
        variant: "RoleBinding" | "Precedence" -- required for "flip"
            (defines the realised-outcome code, spec 2.3 / 3.3).
        norm: position norm for "traj" (metrics.TRAJ_NORMS).
        decision_mask: optional (T, K) bool decision-point restriction for
            "tv_*"/"intent"/"executed" (spec 6.1 [FIXED]; the mask is
            produced by harness/planners.py when it lands -- it is an
            input here, not computed here).
    """
    if divergence not in DIVERGENCES:
        raise ValueError("divergence %r not in %r" % (divergence, DIVERGENCES))

    if divergence == "flip":
        if variant not in metrics.VARIANTS:
            raise ValueError(
                "divergence='flip' needs variant in %r (defines the "
                "realised-outcome code, spec 2.3/3.3); got %r"
                % (metrics.VARIANTS, variant))
        code = metrics.outcome_codes(record.latch_slot, record.latch_time,
                                     variant)
        flip, _ = metrics.outcome_flip(
            lane_tensor(code, plan, lane_a),
            lane_tensor(code, plan, lane_b),
            lane_tensor(record.completed, plan, lane_a),
            lane_tensor(record.completed, plan, lane_b))
        return flip

    act_a = lane_tensor(record.active, plan, lane_a, trajectory=True)
    act_b = lane_tensor(record.active, plan, lane_b, trajectory=True)

    if divergence == "traj":
        return metrics.traj_divergence(
            lane_tensor(record.positions, plan, lane_a, trajectory=True),
            lane_tensor(record.positions, plan, lane_b, trajectory=True),
            act_a, act_b, norm=norm)

    if divergence in ("tv_step", "tv_joint"):
        if record.logits is None:
            raise ValueError(
                "divergence %r needs recorded logits; re-run run_lanes "
                "with record_logits=True" % (divergence,))
        return metrics.action_tv(
            lane_tensor(record.logits, plan, lane_a, trajectory=True),
            lane_tensor(record.logits, plan, lane_b, trajectory=True),
            act_a, act_b, joint=(divergence == "tv_joint"),
            decision_mask=decision_mask)

    field = record.intended if divergence == "intent" else record.executed
    return metrics.action_mismatch(
        lane_tensor(field, plan, lane_a, trajectory=True),
        lane_tensor(field, plan, lane_b, trajectory=True),
        act_a, act_b, decision_mask=decision_mask)


def divergence_marginals(record, plan, divergence, variant=None, norm="l1",
                         decision_mask=None):
    """Raw (K,) marginals of every available lane divergence (spec 6.1:
    published as marginal distributions, not only as a summary)."""
    out = {}
    for name, lane in _MARGINAL_LANES:
        if lane in plan.lanes:
            out[name] = lane_divergence(
                record, plan, lane, divergence, variant=variant, norm=norm,
                decision_mask=decision_mask)
    return out


# ---------------------------------------------------------------------------
# Point estimators (spec 6.1 / 6.4)
# ---------------------------------------------------------------------------

def _paired_valid(d_lang, d_nuis):
    """Pairwise-complete mask: episodes where BOTH divergences are defined.

    The estimand is paired (frozen-scenario design, spec 6.3), so an
    episode degenerate in either term is dropped from both."""
    return torch.isfinite(d_lang) & torch.isfinite(d_nuis)


def csi_point(d_lang, d_nuis):
    """Pooled point estimates over the pairwise-complete episodes.

    Returns (share, ratio, degenerate, nuisance_degenerate):

    * share = sum(D_lang) / (sum(D_lang) + sum(D_nuis)) -- the Hajek
      (ratio-of-means) form of CSI_share; NEVER a mean of per-episode
      0/0 shares.
    * ratio = sum(D_lang) / sum(D_nuis), display-only; ``inf`` when the
      nuisance sum is zero with a nonzero numerator, NaN at 0/0.
    * degenerate: the SHARE's pooled denominator is zero (both channels
      inert; CSI unidentifiable) -> share is NaN.
    * nuisance_degenerate: the nuisance sum alone is zero -> the ratio is
      unbounded and the share is pinned at 1.0 (finite by construction --
      the bounded regularisation working as specified) but the flag makes
      the OPEN(12) denominator switch actionable.
    """
    valid = _paired_valid(d_lang, d_nuis)
    dl = d_lang[valid].double()
    dn = d_nuis[valid].double()
    s_l = float(dl.sum()) if dl.numel() else 0.0
    s_n = float(dn.sum()) if dn.numel() else 0.0
    total = s_l + s_n
    degenerate = (total == 0.0) or (dl.numel() == 0)
    nuisance_degenerate = (s_n == 0.0)
    share = float("nan") if degenerate else s_l / total
    if s_n > 0.0:
        ratio = s_l / s_n
    elif s_l > 0.0:
        ratio = float("inf")
    else:
        ratio = float("nan")
    return share, ratio, degenerate, nuisance_degenerate


# ---------------------------------------------------------------------------
# LOCAL bootstrap + paired sign test.
# TODO(harness/stats.py): the spec 7 file plan owns hierarchical_bootstrap
# and paired_sign_test in harness/stats.py, which does not exist at the
# time of writing.  When it lands, these become thin wrappers (or are
# deleted) -- keep the resampling unit (the CLUSTER = base scenario) and
# the percentile method (spec: percentile, not BCa) identical.
# ---------------------------------------------------------------------------

def _cluster_sums(dl, dn, clusters):
    """Per-cluster sums of the two divergences plus pair counts.

    Resampling clusters with replacement is equivalent to weighting each
    cluster's sums by its multinomial count, which is what lets the
    bootstrap below be fully vectorised for ratio-of-sums statistics.
    """
    uniq, inv = torch.unique(clusters, return_inverse=True)
    m = uniq.numel()
    s_l = torch.zeros(m, dtype=torch.float64).index_add_(0, inv, dl.double())
    s_n = torch.zeros(m, dtype=torch.float64).index_add_(0, inv, dn.double())
    cnt = torch.zeros(m, dtype=torch.float64).index_add_(
        0, inv, torch.ones_like(dl, dtype=torch.float64))
    return s_l, s_n, cnt


def _percentile_ci(reps, ci_level):
    """(lo, hi, n_nonfinite): percentile CI over the FINITE replicates."""
    finite = torch.isfinite(reps)
    n_bad = int((~finite).sum())
    good = reps[finite]
    if good.numel() == 0:
        return float("nan"), float("nan"), n_bad
    alpha = (1.0 - ci_level) / 2.0
    lo = float(torch.quantile(good, alpha))
    hi = float(torch.quantile(good, 1.0 - alpha))
    return lo, hi, n_bad


def _bootstrap(d_lang, d_nuis, clusters, n_boot, seed, ci_level, chunk=2048):
    """Episode(cluster)-resampled percentile bootstrap of share and ratio.

    LOCAL implementation pending harness/stats.py (see module TODO).
    Clusters are resampled with replacement (M draws of M clusters per
    replicate); both terms are recomputed INSIDE each replicate (spec
    6.4).  Degenerate replicates (zero resampled denominator) are
    excluded from the percentile and counted, never imputed.
    """
    valid = _paired_valid(d_lang, d_nuis)
    dl, dn, cl = d_lang[valid], d_nuis[valid], clusters[valid]
    if dl.numel() == 0:
        nan = float("nan")
        return (nan, nan), (nan, nan), 0, 0
    s_l, s_n, _ = _cluster_sums(dl, dn, cl)
    m = s_l.numel()
    g = torch.Generator().manual_seed(int(seed))
    shares, ratios = [], []
    done = 0
    while done < n_boot:
        b = min(int(chunk), n_boot - done)
        idx = torch.randint(0, m, (b, m), generator=g)
        counts = torch.zeros((b, m), dtype=torch.float64).scatter_add_(
            1, idx, torch.ones((b, m), dtype=torch.float64))
        rl = counts @ s_l
        rn = counts @ s_n
        shares.append(rl / (rl + rn))          # 0/0 -> NaN, kept explicit
        ratios.append(rl / rn)                 # x/0 -> inf, 0/0 -> NaN
        done += b
    shares = torch.cat(shares)
    ratios = torch.cat(ratios)
    s_lo, s_hi, s_bad = _percentile_ci(shares, ci_level)
    r_lo, r_hi, r_bad = _percentile_ci(ratios, ci_level)
    return (s_lo, s_hi), (r_lo, r_hi), s_bad, r_bad


def paired_sign_test(d_lang, d_nuis, clusters=None, n_perm=10000, seed=0):
    """Spec 6.4(a): sign-randomisation test on the paired differences.

    Statistic: mean over pairwise-complete episodes of
    ``D_lang,i - D_nuis,i``; the null randomises the SIGN of each
    cluster's contribution within the frozen pairing (preserving the
    design, unlike the mis-specified permutation null the spec deletes).

    Returns (p_greater, p_two_sided): one-sided "language more influential
    than the nuisance" and two-sided, both with the +1 add-one correction.
    """
    valid = _paired_valid(d_lang, d_nuis)
    if int(valid.sum()) == 0:
        return float("nan"), float("nan")
    dl, dn = d_lang[valid], d_nuis[valid]
    cl = (clusters[valid] if clusters is not None
          else torch.arange(int(valid.sum())))
    s_l, s_n, cnt = _cluster_sums(dl, dn, cl)
    s_d = s_l - s_n                                        # per-cluster sums
    n = float(cnt.sum())
    obs = float(s_d.sum()) / n
    g = torch.Generator().manual_seed(int(seed))
    signs = (torch.randint(0, 2, (int(n_perm), s_d.numel()),
                           generator=g).double() * 2.0 - 1.0)
    perm = (signs @ s_d) / n
    p_greater = (1.0 + float((perm >= obs).sum())) / (n_perm + 1.0)
    p_two = (1.0 + float((perm.abs() >= abs(obs)).sum())) / (n_perm + 1.0)
    return p_greater, p_two


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CSIEstimate:
    """One CSI estimate (aggregate or per-category), spec 6.1/6.4 form."""

    n_pairs: int                      # pairwise-complete episodes
    d_lang_mean: float                # unnormalised numerator mean
    d_nuis_mean: float                # unnormalised denominator mean
    frac_lang_zero: float             # concentration diagnostics
    frac_nuis_zero: float             # (OPEN(12) trigger readout)
    share: float                      # PRIMARY: bounded CSI_share
    share_ci: Tuple[float, float]
    ratio: float                      # DISPLAY-ONLY Hajek ratio
    ratio_ci: Tuple[float, float]
    degenerate: bool                  # share's pooled denominator == 0
    nuisance_degenerate: bool         # nuisance sum == 0 (ratio unbounded)
    n_boot: int
    n_degenerate_share_reps: int      # excluded (not imputed) replicates
    n_degenerate_ratio_reps: int
    p_sign_greater: float             # spec 6.4(a) paired sign test
    p_sign_two_sided: float


def estimate(d_lang, d_nuis, clusters=None, n_boot=10000, seed=0,
             ci_level=0.95, n_perm=10000):
    """Full CSIEstimate from paired per-episode divergences.

    Args:
        d_lang, d_nuis: (K,) float tensors; NaN = per-episode degeneracy
            (dropped pairwise).
        clusters: optional (K,) integer cluster ids (base scenario ids);
            default: every episode its own cluster.  The bootstrap and the
            sign test both operate at cluster level (spec 4.3
            ``clustering_unit: base_scenario``).
        n_boot / seed / ci_level: percentile bootstrap controls (spec 6.4:
            10,000 percentile replicates).
        n_perm: sign-test randomisations.
    """
    d_lang = d_lang.reshape(-1).float()
    d_nuis = d_nuis.reshape(-1).float()
    assert d_lang.numel() == d_nuis.numel(), \
        (d_lang.numel(), d_nuis.numel())
    if clusters is None:
        clusters = torch.arange(d_lang.numel())
    clusters = clusters.reshape(-1).long()
    assert clusters.numel() == d_lang.numel()

    valid = _paired_valid(d_lang, d_nuis)
    n_pairs = int(valid.sum())
    dl, dn = d_lang[valid], d_nuis[valid]
    share, ratio, degenerate, nuis_degen = csi_point(d_lang, d_nuis)
    share_ci, ratio_ci, s_bad, r_bad = _bootstrap(
        d_lang, d_nuis, clusters, n_boot, seed, ci_level)
    p_greater, p_two = paired_sign_test(d_lang, d_nuis, clusters,
                                        n_perm=n_perm, seed=seed)
    nan = float("nan")
    return CSIEstimate(
        n_pairs=n_pairs,
        d_lang_mean=float(dl.mean()) if n_pairs else nan,
        d_nuis_mean=float(dn.mean()) if n_pairs else nan,
        frac_lang_zero=float((dl == 0).float().mean()) if n_pairs else nan,
        frac_nuis_zero=float((dn == 0).float().mean()) if n_pairs else nan,
        share=share, share_ci=share_ci,
        ratio=ratio, ratio_ci=ratio_ci,
        degenerate=degenerate, nuisance_degenerate=nuis_degen,
        n_boot=int(n_boot),
        n_degenerate_share_reps=s_bad, n_degenerate_ratio_reps=r_bad,
        p_sign_greater=p_greater, p_sign_two_sided=p_two,
    )


@dataclasses.dataclass
class CSIResult:
    """compute_csi output: aggregate + per-semantic-category estimates plus
    the raw published marginals (spec 6.1)."""

    divergence: str
    nuisance: str
    mode: str                          # record's action-selection mode
    n_base: int
    aggregate: CSIEstimate
    per_category: Dict[int, CSIEstimate]
    marginals: Dict[str, torch.Tensor]     # raw (K,) D_* distributions
    categories: torch.Tensor               # (K,) category of each pair


def compute_csi(record, plan, divergence="traj", nuisance="seed",
                variant=None, categories=None, norm="l1",
                decision_mask=None, n_boot=10000, seed=0, ci_level=0.95,
                n_perm=10000):
    """CSI from one lane-batched rollout record (spec 6.1/6.3/6.4).

    Args:
        record: harness.rollout.RolloutRecord from ``run_lanes``.
        plan: the LanePlan it was rolled under; must contain the
            counterfactual lane L1 and the chosen nuisance lane.
        divergence: one of DIVERGENCES (see module constant).
        nuisance: "seed" (the definitional physics-seed denominator,
            default) or "spawn" (the OPEN(12) fallback -- switching this
            argument IS the pre-registered denominator switch).  "blank"
            is deliberately not accepted: D_blank is a third column, not
            a ratio (spec 6.1); it still appears in ``marginals``.
        variant: required for divergence="flip".
        categories: optional (K,) integer semantic category per base
            scenario; default: the FACTUAL lane's ``instr_class`` (the
            spec's semantic class).  Estimates are returned per category
            and aggregate.
        norm / decision_mask: forwarded to the divergence.
        n_boot, seed, ci_level, n_perm: inference controls.
    """
    if ro.LANE_COUNTERFACTUAL not in plan.lanes:
        raise ValueError(
            "plan has no counterfactual lane L1; CSI needs the paired or "
            "five-lane layout (spec 6.3)")
    if nuisance not in NUISANCE_LANES:
        raise ValueError(
            "nuisance %r not in %r ('blank' is a third column, never a "
            "denominator -- spec 6.1)" % (nuisance, sorted(NUISANCE_LANES)))
    nuis_lane = NUISANCE_LANES[nuisance]
    if nuis_lane not in plan.lanes:
        raise ValueError(
            "plan lanes %r lack the %r nuisance lane (lane id %d); roll "
            "with make_five_lane_plan (spec 6.3)"
            % (plan.lanes, nuisance, nuis_lane))

    marginals = divergence_marginals(record, plan, divergence,
                                     variant=variant, norm=norm,
                                     decision_mask=decision_mask)
    d_lang = marginals["d_lang"]
    d_nuis = marginals["d_" + nuisance]

    if categories is None:
        categories = lane_tensor(record.instr_class, plan, ro.LANE_FACTUAL)
    categories = categories.reshape(-1).long()
    assert categories.numel() == plan.n_base, \
        (categories.numel(), plan.n_base)

    # cluster = base scenario id (spec 4.3 clustering_unit)
    clusters = lane_tensor(plan.scenarios.long(), plan, ro.LANE_FACTUAL)

    aggregate = estimate(d_lang, d_nuis, clusters=clusters, n_boot=n_boot,
                         seed=seed, ci_level=ci_level, n_perm=n_perm)
    per_category = {}
    for i, cat in enumerate(torch.unique(categories).tolist()):
        sel = categories == cat
        per_category[int(cat)] = estimate(
            d_lang[sel], d_nuis[sel], clusters=clusters[sel],
            n_boot=n_boot, seed=seed + 7919 * (i + 1), ci_level=ci_level,
            n_perm=n_perm)

    return CSIResult(
        divergence=divergence, nuisance=nuisance,
        mode=getattr(record, "mode", "?"), n_base=plan.n_base,
        aggregate=aggregate, per_category=per_category,
        marginals=marginals, categories=categories,
    )


__all__ = [
    "DIVERGENCES", "NUISANCE_LANES",
    "lane_tensor", "lane_divergence", "divergence_marginals",
    "csi_point", "paired_sign_test",
    "CSIEstimate", "estimate", "CSIResult", "compute_csi",
]
