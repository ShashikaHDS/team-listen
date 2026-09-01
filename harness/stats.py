"""Statistical toolkit for the M1 audit (M1_SPEC 2.3 / 4.4 / 6.4 / section 7).

Spec file-plan entry: ``stats.py  # tost, wilson(display), hierarchical_
bootstrap, iqm, icc_deff, mcnemar, paired_sign_test, power_analysis``.

Design points, straight from the spec:

* **Hierarchical (episode-clustered) bootstrap is the PRIMARY interval**
  (spec 4.4 [FIXED: the primary interval structurally excluded the dominant
  error term]): resample base scenarios AND training seeds with replacement
  in the same replicate, percentile method (not BCa, per rliable's few-run
  guidance), 10,000 replicates.  A seed-level-only interval is pre-registered
  as inadmissible.  Saravanan et al.: naive pooling reaches 46-96%
  false-positive rates on clustered data; for an EQUIVALENCE claim the
  failure is mirrored -- a spuriously tight pooled CI makes CI-inclusion
  trivially pass.  ``tests/test_stats.py`` demonstrates the too-narrow
  naive CI on synthetic clustered data.
* **TOST at alpha = 0.05 <=> a 90% CI inside [0.45, 0.55]** (spec 4.4),
  justified by coverage properties only; the CI is the hierarchical
  bootstrap CI at confidence 1 - 2*alpha.
* **Wilson / Clopper-Pearson intervals and the exact binomial test are
  DISPLAY-ONLY** (spec 4.4: with a frozen manifest and argmax actions a
  per-seed accuracy is an exact finite-population count with no binomial
  sampling model).  They are exact known-answer testable, so they also
  anchor ``tests/test_stats.py``.
* **Paired bootstrap for arm deltas**: the study is fully paired (same
  (scenario, instruction, slip-stream) tuples across every arm and seed,
  spec 4.3), so an arm contrast resamples base scenarios ONCE per replicate
  and differences the two arms' statistics inside the replicate.
* **Holm correction** across a perturbation family (spec OPEN(10); the
  family is fixed in month 4 -- the helper ships now).
* **icc_deff** (spec 2.3): observed ICC and the Kish design effect,
  reported so the "one binary per episode, never per agent" clustering
  decision is auditable.
* **power_analysis** (spec 4.4 [FIXED: no power analysis existed]):
  simulates the TOST rule at true accuracies {0.50, 0.53, 0.55, 0.60}
  under a given between-seed SD and reports pass/detection probability.

Pure numpy + torch (torch only for ``erfinv``); NO scipy, NO sklearn, NO
Isaac imports.  Everything is deterministic given ``seed``.
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _as_np(x, dtype=np.float64):
    """torch tensor / list / scalar -> 1-D numpy array of ``dtype``."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    if dtype is not None:
        a = a.astype(dtype)
    return a.reshape(-1)


def normal_ppf(q):
    """Standard-normal quantile via torch.erfinv (no scipy)."""
    q = float(q)
    assert 0.0 < q < 1.0, q
    return float(math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0,
                                                            dtype=torch.float64)))


def _log_factorials(n):
    """log(k!) for k = 0..n, exact to float64 (cumsum of logs)."""
    if n == 0:
        return np.zeros(1)
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n + 1,
                                                             dtype=np.float64)))])


def binom_logpmf(k, n, p):
    """log P(X = k), X ~ Binomial(n, p); k may be an array."""
    k = np.asarray(k, dtype=np.int64)
    if p <= 0.0:
        return np.where(k == 0, 0.0, -np.inf)
    if p >= 1.0:
        return np.where(k == n, 0.0, -np.inf)
    lf = _log_factorials(n)
    return (lf[n] - lf[k] - lf[n - k]
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_pmf(k, n, p):
    return np.exp(binom_logpmf(k, n, p))


def binom_cdf(k, n, p):
    """P(X <= k), exact summation (float64)."""
    k = int(k)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(np.exp(binom_logpmf(np.arange(k + 1), n, p)).sum())


def binom_test(k, n, p=0.5, alternative="two-sided"):
    """Exact binomial test (no scipy).

    ``two-sided`` uses the minlike convention (sum of all outcome
    probabilities <= the observed one, with a 1 + 1e-7 relative slack) --
    the same definition as scipy.stats.binomtest / R binom.test, so the
    known-answer tests can quote published values.
    """
    k, n = int(k), int(n)
    assert 0 <= k <= n and 0.0 <= p <= 1.0
    if alternative == "greater":
        return float(min(1.0, 1.0 - binom_cdf(k - 1, n, p)))
    if alternative == "less":
        return float(min(1.0, binom_cdf(k, n, p)))
    assert alternative == "two-sided", alternative
    if p in (0.0, 1.0):
        obs_ok = (k == 0) if p == 0.0 else (k == n)
        return 1.0 if obs_ok else 0.0
    pmf = np.exp(binom_logpmf(np.arange(n + 1), n, p))
    d = pmf[k] * (1.0 + 1e-7)
    return float(min(1.0, pmf[pmf <= d].sum()))


# ---------------------------------------------------------------------------
# Binomial intervals for at-chance assignment accuracy (display, spec 4.4)
# ---------------------------------------------------------------------------

def wilson_interval(k, n, conf=0.95):
    """Wilson score interval (display-only per spec 4.4)."""
    k, n = int(k), int(n)
    assert 0 <= k <= n and n > 0
    z = normal_ppf(1.0 - (1.0 - conf) / 2.0)
    phat = k / n
    z2n = z * z / n
    center = (phat + z2n / 2.0) / (1.0 + z2n)
    half = (z / (1.0 + z2n)) * math.sqrt(phat * (1.0 - phat) / n
                                         + z2n / (4.0 * n))
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson_interval(k, n, conf=0.95):
    """Exact (Clopper-Pearson) interval via bisection on the exact binomial
    CDF -- no scipy beta functions needed."""
    k, n = int(k), int(n)
    assert 0 <= k <= n and n > 0
    a2 = (1.0 - conf) / 2.0

    def _bisect(f, lo, hi):
        # f monotone increasing with a sign change on [lo, hi]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    if k == 0:
        lo = 0.0
    else:
        # smallest p with P(X >= k | p) = a2 ; P(X >= k | p) increases in p
        lo = _bisect(lambda p: (1.0 - binom_cdf(k - 1, n, p)) - a2, 0.0, 1.0)
    if k == n:
        hi = 1.0
    else:
        # largest p with P(X <= k | p) = a2 ; P(X <= k | p) decreases in p
        hi = _bisect(lambda p: a2 - binom_cdf(k, n, p), 0.0, 1.0)
    return lo, hi


# ---------------------------------------------------------------------------
# Episode-clustered / hierarchical bootstrap (the PRIMARY interval, spec 4.4)
# ---------------------------------------------------------------------------

def _cell_sums(num, den, cluster_ids, seed_ids):
    num = _as_np(num)
    den = np.ones_like(num) if den is None else _as_np(den)
    n = num.shape[0]
    assert den.shape[0] == n
    if cluster_ids is None:
        cidx = np.arange(n)
        G = n
    else:
        _, cidx = np.unique(_as_np(cluster_ids, dtype=None), return_inverse=True)
        G = int(cidx.max()) + 1 if n else 0
    if seed_ids is None:
        sidx = np.zeros(n, dtype=np.int64)
        S = 1
    else:
        _, sidx = np.unique(_as_np(seed_ids, dtype=None), return_inverse=True)
        S = int(sidx.max()) + 1 if n else 0
    A = np.zeros((S, G))
    B = np.zeros((S, G))
    np.add.at(A, (sidx, cidx), num)
    np.add.at(B, (sidx, cidx), den)
    return A, B, S, G


def hierarchical_bootstrap_ci(num, den=None, cluster_ids=None, seed_ids=None,
                              n_boot=10000, conf=0.95, seed=0, chunk=2048):
    """Percentile CI for a ratio-of-sums statistic under the two-level
    hierarchical bootstrap of spec 4.4.

    The statistic is ``sum(num) / sum(den)`` over the resampled rows
    (``den=None`` -> a plain mean; accuracy is ``num = Y*C, den = C``).
    Each replicate resamples base scenarios (``cluster_ids``) AND training
    seeds (``seed_ids``) with replacement IN THE SAME REPLICATE.
    ``seed_ids=None`` degrades to the one-level episode-clustered
    bootstrap; ``cluster_ids=None`` additionally degrades to iid.

    Ratio-of-sums makes every replicate computable from per-(seed, scenario)
    sufficient statistics, so 10,000 replicates over 2000 x 10 cells are a
    few vectorised matmuls (Hajek ratio of means recomputed inside each
    replicate, spec 6.4).
    """
    A, B, S, G = _cell_sums(num, den, cluster_ids, seed_ids)
    tot_num, tot_den = float(A.sum()), float(B.sum())
    point = tot_num / tot_den if tot_den > 0 else float("nan")
    rng = np.random.default_rng(seed)
    reps = []
    n_dropped = 0
    remaining = int(n_boot)
    p_s = np.full(S, 1.0 / S)
    p_c = np.full(G, 1.0 / G)
    while remaining > 0:
        m = min(chunk, remaining)
        remaining -= m
        if S > 1:
            ws = rng.multinomial(S, p_s, size=m).astype(np.float64)
        else:
            ws = np.ones((m, 1))
        wc = rng.multinomial(G, p_c, size=m).astype(np.float64)
        ta = ws @ A                                    # (m, G)
        tb = ws @ B
        num_r = (ta * wc).sum(axis=1)
        den_r = (tb * wc).sum(axis=1)
        ok = den_r > 0
        n_dropped += int((~ok).sum())
        reps.append(num_r[ok] / den_r[ok])
    reps = np.concatenate(reps) if reps else np.array([])
    if reps.size == 0:
        lo = hi = float("nan")
    else:
        a2 = (1.0 - conf) / 2.0
        lo, hi = np.quantile(reps, [a2, 1.0 - a2])
    return {
        "point": point, "lo": float(lo), "hi": float(hi), "conf": float(conf),
        "n": int(_as_np(num).shape[0]), "n_clusters": int(G),
        "n_seeds": int(S), "n_boot": int(n_boot),
        "n_dropped": int(n_dropped), "method": "percentile",
        "level": ("hierarchical_scenarios_x_seeds" if S > 1
                  else "episode_clustered"),
    }


def clustered_bootstrap_ci(num, den=None, cluster_ids=None, n_boot=10000,
                           conf=0.95, seed=0, chunk=2048):
    """One-level episode-clustered bootstrap CI (resample clusters with
    replacement, keep every member episode).  ``clustering_unit:
    base_scenario`` per the pre-registration (spec 4.4)."""
    return hierarchical_bootstrap_ci(num, den=den, cluster_ids=cluster_ids,
                                     seed_ids=None, n_boot=n_boot, conf=conf,
                                     seed=seed, chunk=chunk)


def bootstrap_ci_stat(stat_fn, arrays, cluster_ids, n_boot=2000, conf=0.95,
                      seed=0):
    """Cluster bootstrap for an ARBITRARY statistic (e.g. AUC).

    ``arrays`` is a sequence of aligned (n, ...) arrays; each replicate
    resamples clusters with replacement, concatenates their member rows and
    calls ``stat_fn(*resampled_arrays)``.  Replicates where ``stat_fn``
    returns nan (e.g. a single-class AUC resample) are dropped and counted.
    """
    arrays = [np.asarray(a) if not torch.is_tensor(a)
              else a.detach().cpu().numpy() for a in arrays]
    n = arrays[0].shape[0]
    for a in arrays:
        assert a.shape[0] == n
    _, cidx = np.unique(_as_np(cluster_ids, dtype=None), return_inverse=True)
    G = int(cidx.max()) + 1 if n else 0
    order = np.argsort(cidx, kind="stable")
    bounds = np.searchsorted(cidx[order], np.arange(G + 1))
    members = [order[bounds[g]:bounds[g + 1]] for g in range(G)]
    point = float(stat_fn(*arrays))
    rng = np.random.default_rng(seed)
    reps = []
    n_dropped = 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, G, size=G)
        rows = np.concatenate([members[g] for g in draw])
        val = float(stat_fn(*[a[rows] for a in arrays]))
        if math.isnan(val):
            n_dropped += 1
        else:
            reps.append(val)
    if not reps:
        lo = hi = float("nan")
    else:
        a2 = (1.0 - conf) / 2.0
        lo, hi = np.quantile(np.asarray(reps), [a2, 1.0 - a2])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "conf": float(conf), "n": int(n), "n_clusters": int(G),
            "n_boot": int(n_boot), "n_dropped": int(n_dropped),
            "method": "percentile"}


def paired_delta_bootstrap_ci(num_a, num_b, den_a=None, den_b=None,
                              cluster_ids=None, n_boot=10000, conf=0.95,
                              seed=0, chunk=2048):
    """Paired cluster bootstrap for an ARM DELTA (spec 4.3: the study is
    fully paired over (scenario, instruction, slip-stream) tuples).

    ``num_a/den_a`` and ``num_b/den_b`` are the two arms' per-episode
    numerators/denominators over the SAME cluster ids.  Each replicate
    resamples base scenarios once and computes
    ``ratio_a - ratio_b`` inside the replicate, so the between-scenario
    variance common to both arms cancels -- the whole point of pairing.
    """
    Aa, Ba, _, Ga = _cell_sums(num_a, den_a, cluster_ids, None)
    Ab, Bb, _, Gb = _cell_sums(num_b, den_b, cluster_ids, None)
    assert Ga == Gb, "arms must share the cluster structure (paired design)"
    aa, ba, ab, bb = Aa[0], Ba[0], Ab[0], Bb[0]
    pa = float(aa.sum() / ba.sum()) if ba.sum() > 0 else float("nan")
    pb = float(ab.sum() / bb.sum()) if bb.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    reps = []
    n_dropped = 0
    remaining = int(n_boot)
    p_c = np.full(Ga, 1.0 / Ga)
    while remaining > 0:
        m = min(chunk, remaining)
        remaining -= m
        wc = rng.multinomial(Ga, p_c, size=m).astype(np.float64)
        da, db = wc @ ba, wc @ bb
        ok = (da > 0) & (db > 0)
        n_dropped += int((~ok).sum())
        reps.append((wc @ aa)[ok] / da[ok] - (wc @ ab)[ok] / db[ok])
    reps = np.concatenate(reps) if reps else np.array([])
    if reps.size == 0:
        lo = hi = float("nan")
    else:
        a2 = (1.0 - conf) / 2.0
        lo, hi = np.quantile(reps, [a2, 1.0 - a2])
    return {"point": pa - pb, "point_a": pa, "point_b": pb,
            "lo": float(lo), "hi": float(hi), "conf": float(conf),
            "n_clusters": int(Ga), "n_boot": int(n_boot),
            "n_dropped": int(n_dropped), "method": "percentile"}


# ---------------------------------------------------------------------------
# TOST equivalence (the gating at-chance test, spec 4.4)
# ---------------------------------------------------------------------------

def tost_from_ci(lo, hi, center=0.5, delta=0.05):
    """TOST at alpha <=> the (1 - 2*alpha) CI lies inside
    [center - delta, center + delta] (spec 4.4)."""
    return bool(lo >= center - delta and hi <= center + delta)


def tost_equivalence(num, den=None, cluster_ids=None, seed_ids=None,
                     center=0.5, delta=0.05, alpha=0.05, n_boot=10000,
                     seed=0):
    """The spec 4.4 at-chance equivalence test: hierarchical-bootstrap
    (1 - 2*alpha) CI inside [center - delta, center + delta]."""
    ci = hierarchical_bootstrap_ci(num, den=den, cluster_ids=cluster_ids,
                                   seed_ids=seed_ids, n_boot=n_boot,
                                   conf=1.0 - 2.0 * alpha, seed=seed)
    ci.update({
        "test": "TOST", "alpha": float(alpha), "center": float(center),
        "delta": float(delta),
        "band": (float(center - delta), float(center + delta)),
        "equivalent": tost_from_ci(ci["lo"], ci["hi"], center, delta),
    })
    return ci


# ---------------------------------------------------------------------------
# Descriptives: IQM, ICC / design effect
# ---------------------------------------------------------------------------

def iqm(values):
    """Interquartile mean (middle 50% trimmed mean, rliable-style):
    discard floor(n/4) values from each end of the sorted sample."""
    v = np.sort(_as_np(values))
    n = v.shape[0]
    cut = n // 4
    kept = v[cut:n - cut] if n - 2 * cut > 0 else v
    return float(kept.mean())


def icc_deff(values, cluster_ids):
    """One-way-ANOVA ICC estimate and the Kish design effect
    DEFF = 1 + (m_bar - 1) * ICC (spec 2.3: reported so the one-binary-
    per-episode clustering decision is auditable)."""
    y = _as_np(values)
    _, cidx = np.unique(_as_np(cluster_ids, dtype=None), return_inverse=True)
    n = y.shape[0]
    G = int(cidx.max()) + 1
    counts = np.bincount(cidx, minlength=G).astype(np.float64)
    sums = np.bincount(cidx, weights=y, minlength=G)
    means = sums / counts
    grand = y.mean()
    ssb = float((counts * (means - grand) ** 2).sum())
    ssw = float(((y - means[cidx]) ** 2).sum())
    if G <= 1 or n <= G:
        return {"icc": 0.0, "deff": 1.0, "n": n, "n_clusters": G,
                "mean_cluster_size": n / max(G, 1)}
    msb = ssb / (G - 1)
    msw = ssw / (n - G)
    n0 = (n - float((counts ** 2).sum()) / n) / (G - 1)
    denom = msb + (n0 - 1.0) * msw
    icc = (msb - msw) / denom if denom > 0 else 0.0
    mbar = n / G
    deff = 1.0 + (mbar - 1.0) * max(0.0, icc)
    return {"icc": float(icc), "deff": float(deff), "n": int(n),
            "n_clusters": int(G), "mean_cluster_size": float(mbar)}


# ---------------------------------------------------------------------------
# Paired categorical tests (CSI consumers, spec 6.1 / 6.4)
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Exact McNemar test on the discordant counts (spec 6.1: CSI_flip is
    a pair of paired proportions with McNemar on the discordant counts)."""
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    return binom_test(b, n, 0.5, alternative="two-sided")


def paired_sign_permutation_test(diffs, n_perm=10000, seed=0):
    """Sign-randomisation test on paired differences (spec 6.4(a)):
    randomise the sign of each episode's paired difference within the
    frozen pairing; two-sided p for the mean with the add-one rule."""
    d = _as_np(diffs)
    n = d.shape[0]
    if n == 0:
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(int(n_perm), n)) * 2 - 1
    perm = np.abs((signs * d).mean(axis=1))
    return float((1 + int((perm >= obs - 1e-12).sum())) / (1 + int(n_perm)))


# ---------------------------------------------------------------------------
# Holm correction (spec OPEN(10))
# ---------------------------------------------------------------------------

def holm_correction(pvalues, alpha=0.05):
    """Holm step-down correction.  Returns (adjusted_pvalues, reject) in
    the ORIGINAL order; adjusted p-values are monotone (step-down max)."""
    p = _as_np(pvalues)
    m = p.shape[0]
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    reject = adj <= alpha
    return adj, reject


# ---------------------------------------------------------------------------
# Power analysis of the pre-registered rule (spec 4.4 [FIXED])
# ---------------------------------------------------------------------------

def power_analysis(true_accuracies=(0.50, 0.53, 0.55, 0.60),
                   n_scenarios=2000, n_seeds=10, between_seed_sd=0.0,
                   delta=0.05, alpha=0.05, n_sim=200, n_boot=2000, seed=0):
    """Simulate the TOST decision rule and report its operating profile.

    For each true accuracy: draw per-seed accuracies
    ``clip(acc + N(0, between_seed_sd))``, per-(seed, scenario) Bernoulli
    outcomes, apply the hierarchical-bootstrap TOST, and record the pass
    rate.  ``pass_prob`` is P(rule declares at-chance); ``detect_prob`` is
    its complement -- the Type-II profile the spec requires the audit to
    state.
    """
    rng = np.random.default_rng(seed)
    scen = np.tile(np.arange(n_scenarios), n_seeds)
    seeds_ix = np.repeat(np.arange(n_seeds), n_scenarios)
    out = {}
    for acc in true_accuracies:
        passes = 0
        for s in range(int(n_sim)):
            p_s = np.clip(acc + rng.normal(0.0, between_seed_sd, n_seeds),
                          0.01, 0.99)
            y = (rng.random((n_seeds, n_scenarios))
                 < p_s[:, None]).astype(np.float64).reshape(-1)
            res = tost_equivalence(
                y, den=None, cluster_ids=scen,
                seed_ids=seeds_ix if n_seeds > 1 else None,
                center=0.5, delta=delta, alpha=alpha, n_boot=n_boot,
                seed=int(rng.integers(0, 2 ** 31 - 1)))
            passes += int(res["equivalent"])
        rate = passes / float(n_sim)
        out[float(acc)] = {"pass_prob": rate, "detect_prob": 1.0 - rate}
    return out


__all__ = [
    "normal_ppf", "binom_logpmf", "binom_pmf", "binom_cdf", "binom_test",
    "wilson_interval", "clopper_pearson_interval",
    "hierarchical_bootstrap_ci", "clustered_bootstrap_ci",
    "bootstrap_ci_stat", "paired_delta_bootstrap_ci",
    "tost_from_ci", "tost_equivalence",
    "iqm", "icc_deff", "mcnemar_exact", "paired_sign_permutation_test",
    "holm_correction", "power_analysis",
]
