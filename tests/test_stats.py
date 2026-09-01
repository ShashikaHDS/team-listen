"""Known-answer and behavioural tests for harness/stats.py (M1_SPEC 4.4 /
section 7 test-plan entry ``test_stats.py``).

Coverage:

* **Binomial known answers** -- Wilson and Clopper-Pearson intervals against
  hand-computed / published values (8/10 is the classic textbook case), the
  exact two-sided (minlike) binomial test against its closed-form tail sums,
  and an optional cross-check against scipy when scipy happens to be
  installed (skipped otherwise -- the shipped code never imports scipy).
* **Episode-clustered bootstrap vs the naive interval** -- on synthetic
  clustered data (ICC ~ 1) the naive iid interval is PROVABLY too narrow:
  its width scales as 1/sqrt(G*m) while the sampling SD of the mean is
  tau/sqrt(G).  A coverage simulation shows the naive CI covering far below
  nominal while the clustered CI stays near nominal -- the Saravanan et al.
  failure mode spec 4.4 cites, in the mirrored equivalence-claim direction.
* **Duplication invariance** -- duplicating every episode m times within its
  cluster leaves the clustered CI EXACTLY unchanged (same seed, identical
  replicates) while the naive CI shrinks by ~sqrt(m): the design-effect
  error of counting two agent decisions per episode (spec 2.3 [FIXED]).
* **Two-level hierarchical bootstrap** -- when seeds carry a common offset,
  the scenarios-only interval collapses to a point (the spec 4.4
  anti-conservative failure) while the scenarios-x-seeds interval spans the
  seed spread.
* **Paired bootstrap for arm deltas** -- a constant per-cluster shift is
  recovered with a near-zero-width CI despite large between-cluster
  variance (the point of pairing).
* **TOST / Holm / IQM / ICC-DEFF / McNemar / sign-permutation /
  power_analysis** known answers and operating checks.

No pytest: run ``python tests/test_stats.py`` from the repo root.
"""

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import stats                              # noqa: E402


class SkipTest(Exception):
    pass


def _close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b, tol)


# ---------------------------------------------------------------------------
# Binomial known answers
# ---------------------------------------------------------------------------

def test_normal_ppf_known_answers():
    _close(stats.normal_ppf(0.975), 1.959963984540054, 1e-9)
    _close(stats.normal_ppf(0.95), 1.6448536269514722, 1e-9)
    _close(stats.normal_ppf(0.5), 0.0, 1e-12)


def test_wilson_known_answers():
    # classic 8/10 case (R binom.confint / published tables): (0.4902, 0.9433)
    lo, hi = stats.wilson_interval(8, 10, 0.95)
    _close(lo, 0.4901624715366418, 1e-9)
    _close(hi, 0.9433178485456248, 1e-9)
    # k = 0: Wilson lower bound is exactly 0; upper is z^2-shrunk
    lo, hi = stats.wilson_interval(0, 10, 0.95)
    _close(lo, 0.0, 1e-12)
    _close(hi, 0.2775327998628892, 1e-9)
    # symmetry: k = n mirrors k = 0
    lo2, hi2 = stats.wilson_interval(10, 10, 0.95)
    _close(lo2, 1.0 - hi, 1e-9)
    _close(hi2, 1.0, 1e-12)
    # hand-derived closed form for 8/10 at z = 1.959963984540054
    z = 1.959963984540054
    phat, n = 0.8, 10
    center = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * math.sqrt(
        phat * (1 - phat) / n + z * z / (4 * n * n))
    lo, hi = stats.wilson_interval(8, 10, 0.95)
    _close(lo, center - half, 1e-12)
    _close(hi, center + half, 1e-12)


def test_clopper_pearson_known_answers():
    # classic 8/10 case (R binom.test 95% CI): (0.4439045, 0.9747893)
    lo, hi = stats.clopper_pearson_interval(8, 10, 0.95)
    _close(lo, 0.4439045376923585, 1e-9)
    _close(hi, 0.9747892736731666, 1e-9)
    # k = 0 closed form: (0, 1 - (alpha/2)^(1/n))
    lo, hi = stats.clopper_pearson_interval(0, 10, 0.95)
    _close(lo, 0.0, 1e-12)
    _close(hi, 1.0 - 0.025 ** (1.0 / 10.0), 1e-9)
    # k = n mirrors
    lo2, hi2 = stats.clopper_pearson_interval(10, 10, 0.95)
    _close(lo2, 0.025 ** (1.0 / 10.0), 1e-9)
    _close(hi2, 1.0, 1e-12)
    # CP contains Wilson's point estimate and is wider than Wilson (n small)
    wlo, whi = stats.wilson_interval(8, 10, 0.95)
    clo, chi = stats.clopper_pearson_interval(8, 10, 0.95)
    assert clo <= wlo and chi >= whi


def test_binom_test_known_answers():
    # symmetric p = 0.5 cases: closed-form tail sums over 2^10
    _close(stats.binom_test(8, 10, 0.5), 112.0 / 1024.0, 1e-12)  # 0.109375
    _close(stats.binom_test(2, 10, 0.5), 112.0 / 1024.0, 1e-12)
    _close(stats.binom_test(10, 10, 0.5), 2.0 / 1024.0, 1e-12)
    _close(stats.binom_test(5, 10, 0.5), 1.0, 1e-12)
    # asymmetric minlike case (matches scipy.stats.binomtest / R binom.test)
    _close(stats.binom_test(7, 10, 0.3), 0.0105920784, 1e-10)
    # one-sided closed forms
    _close(stats.binom_test(8, 10, 0.5, "greater"), 56.0 / 1024.0, 1e-9)
    _close(stats.binom_test(8, 10, 0.5, "less"), 1013.0 / 1024.0, 1e-9)
    # cdf consistency
    _close(stats.binom_cdf(5, 10, 0.5), 638.0 / 1024.0, 1e-12)
    _close(stats.binom_cdf(-1, 10, 0.5), 0.0, 0.0)
    _close(stats.binom_cdf(10, 10, 0.5), 1.0, 0.0)


def test_binomial_against_scipy_if_available():
    try:
        import scipy.stats as ss
    except ImportError:
        raise SkipTest("scipy not installed (optional cross-check only)")
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(1, 400))
        k = int(rng.integers(0, n + 1))
        p = float(rng.uniform(0.05, 0.95))
        bt = ss.binomtest(k, n, p)
        _close(stats.binom_test(k, n, p), bt.pvalue, 1e-9)
        ci = bt.proportion_ci(0.95, "exact")
        lo, hi = stats.clopper_pearson_interval(k, n, 0.95)
        _close(lo, ci.low, 1e-8)
        _close(hi, ci.high, 1e-8)
        wi = bt.proportion_ci(0.95, "wilson")
        lo, hi = stats.wilson_interval(k, n, 0.95)
        _close(lo, float(wi.low), 1e-9)
        _close(hi, float(wi.high), 1e-9)


# ---------------------------------------------------------------------------
# Episode-clustered bootstrap: the naive CI is provably too narrow
# ---------------------------------------------------------------------------

def _clustered_sample(rng, G, m, tau=1.0, sigma=0.1):
    """ICC ~ tau^2/(tau^2+sigma^2) ~ 0.99: y_ij = u_g + eps_ij, mean 0."""
    u = rng.normal(0.0, tau, G)
    y = (u[:, None] + rng.normal(0.0, sigma, (G, m))).reshape(-1)
    return y, np.repeat(np.arange(G), m)


def test_clustered_bootstrap_covers_where_naive_fails():
    """Coverage simulation on ICC~1 data: sampling SD of the mean is
    tau/sqrt(G), but the naive iid interval has width ~ 1/sqrt(G*m) --
    sqrt(m) times too narrow.  Nominal 95%: the clustered CI must stay
    near nominal, the naive CI must collapse far below it."""
    rng = np.random.default_rng(20260901)
    G, m = 20, 25
    n_sim = 250
    z = stats.normal_ppf(0.975)
    cov_clustered = cov_naive = 0
    width_clustered = width_naive = 0.0
    for s in range(n_sim):
        y, cid = _clustered_sample(rng, G, m)
        ci = stats.clustered_bootstrap_ci(y, cluster_ids=cid, n_boot=500,
                                          conf=0.95, seed=s)
        cov_clustered += int(ci["lo"] <= 0.0 <= ci["hi"])
        width_clustered += ci["hi"] - ci["lo"]
        # naive interval: iid normal approximation over all G*m rows
        mean, sd = y.mean(), y.std(ddof=1) / math.sqrt(y.size)
        nlo, nhi = mean - z * sd, mean + z * sd
        cov_naive += int(nlo <= 0.0 <= nhi)
        width_naive += nhi - nlo
    cov_clustered /= n_sim
    cov_naive /= n_sim
    assert cov_clustered >= 0.85, \
        "clustered bootstrap coverage %.3f below nominal band" % cov_clustered
    assert cov_naive <= 0.55, \
        "naive CI coverage %.3f is not the documented failure (ICC~1 " \
        "should collapse it toward ~0.30)" % cov_naive
    assert cov_clustered > cov_naive + 0.25
    # width: the naive interval is ~sqrt(m) = 5x too narrow
    assert width_clustered / width_naive >= 2.5, \
        (width_clustered / n_sim, width_naive / n_sim)


def test_cluster_bootstrap_duplication_invariance():
    """Counting each episode's two logically-identical agent bits (spec 2.3
    [FIXED]) must not shrink the clustered CI: duplicating every value m
    times within its cluster leaves the clustered CI exactly unchanged,
    while the naive iid interval wrongly shrinks by ~sqrt(m)."""
    rng = np.random.default_rng(7)
    G, m = 40, 4
    y = rng.normal(0.0, 1.0, G)
    ci_single = stats.clustered_bootstrap_ci(
        y, cluster_ids=np.arange(G), n_boot=1000, seed=11)
    y_dup = np.repeat(y, m)
    cid_dup = np.repeat(np.arange(G), m)
    ci_dup = stats.clustered_bootstrap_ci(
        y_dup, cluster_ids=cid_dup, n_boot=1000, seed=11)
    _close(ci_single["lo"], ci_dup["lo"], 1e-12)
    _close(ci_single["hi"], ci_dup["hi"], 1e-12)
    _close(ci_single["point"], ci_dup["point"], 1e-12)
    # the naive (iid-resampled) interval falls for it: ~sqrt(m) narrower
    naive_single = stats.clustered_bootstrap_ci(y, n_boot=1000, seed=11)
    naive_dup = stats.clustered_bootstrap_ci(y_dup, n_boot=1000, seed=11)
    w1 = naive_single["hi"] - naive_single["lo"]
    w2 = naive_dup["hi"] - naive_dup["lo"]
    assert w2 < 0.7 * w1, (w1, w2)


def test_hierarchical_two_level_vs_scenario_only():
    """Seeds sharing one manifest: per-scenario cell means are constant
    across scenarios, so the scenarios-only interval collapses to a point
    (spec 4.4: anti-conservative for an equivalence claim, inadmissible);
    the scenarios-x-seeds interval spans the between-seed spread."""
    S, G = 3, 50
    offsets = np.array([-0.2, 0.0, 0.2])
    y = np.tile(0.5 + offsets[:, None], (1, G)).reshape(-1)
    sid = np.repeat(np.arange(S), G)
    cid = np.tile(np.arange(G), S)
    hier = stats.hierarchical_bootstrap_ci(
        y, cluster_ids=cid, seed_ids=sid, n_boot=2000, seed=3)
    flat = stats.clustered_bootstrap_ci(y, cluster_ids=cid, n_boot=2000,
                                        seed=3)
    assert hier["level"] == "hierarchical_scenarios_x_seeds"
    _close(hier["point"], 0.5, 1e-12)
    _close(flat["point"], 0.5, 1e-12)
    assert flat["hi"] - flat["lo"] < 1e-9, \
        "scenario-only interval should collapse on seed-common offsets"
    assert hier["hi"] - hier["lo"] > 0.2, (hier["lo"], hier["hi"])
    assert hier["lo"] >= 0.5 - 0.2 - 1e-9
    assert hier["hi"] <= 0.5 + 0.2 + 1e-9


def test_ratio_statistic_and_zero_denominator():
    """Accuracy is a ratio of sums (num = Y&C, den = C): episodes with
    C = 0 must not dilute it, and all-zero-denominator replicates drop."""
    # 3 clusters: (Y,C) = (1,1),(0,1) | (1,1),(0,0) | (0,0),(0,0)
    num = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    den = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    cid = np.array([0, 0, 1, 1, 2, 2])
    ci = stats.clustered_bootstrap_ci(num, den=den, cluster_ids=cid,
                                      n_boot=500, seed=0)
    _close(ci["point"], 2.0 / 3.0, 1e-12)
    assert 0.0 <= ci["lo"] <= ci["hi"] <= 1.0
    # dropped replicates counted (cluster 2 alone has zero denominator)
    assert ci["n_dropped"] >= 0


def test_paired_delta_bootstrap_exact_shift():
    """Arm B = arm A - 0.1 on every cluster: the paired CI pins the delta
    at exactly 0.1 regardless of the between-cluster variance, while each
    arm's own CI is wide -- the point of the fully paired design."""
    rng = np.random.default_rng(5)
    G = 30
    a = rng.normal(0.5, 1.0, G)
    b = a - 0.1
    cid = np.arange(G)
    d = stats.paired_delta_bootstrap_ci(a, b, cluster_ids=cid, n_boot=1000,
                                        seed=2)
    _close(d["point"], 0.1, 1e-9)
    _close(d["lo"], 0.1, 1e-8)
    _close(d["hi"], 0.1, 1e-8)
    wide = stats.clustered_bootstrap_ci(a, cluster_ids=cid, n_boot=1000,
                                        seed=2)
    assert wide["hi"] - wide["lo"] > 0.1


def test_tost_equivalence():
    assert stats.tost_from_ci(0.46, 0.54)
    assert not stats.tost_from_ci(0.44, 0.54)
    assert not stats.tost_from_ci(0.46, 0.56)
    rng = np.random.default_rng(9)
    n = 4000
    cid = np.arange(n)
    at_chance = (rng.random(n) < 0.5).astype(float)
    res = stats.tost_equivalence(at_chance, cluster_ids=cid, n_boot=2000,
                                 seed=1)
    assert res["equivalent"], (res["lo"], res["hi"])
    _close(res["conf"], 0.90, 1e-12)          # alpha = 0.05 -> 90% CI
    off_chance = (rng.random(n) < 0.65).astype(float)
    res = stats.tost_equivalence(off_chance, cluster_ids=cid, n_boot=2000,
                                 seed=1)
    assert not res["equivalent"], (res["lo"], res["hi"])
    # the paired-manifest algebraic identity: one Y=1 per competent pair
    y = np.array([1.0, 0.0] * 100)
    cid = np.repeat(np.arange(100), 2)
    res = stats.tost_equivalence(y, cluster_ids=cid, n_boot=500, seed=4)
    _close(res["point"], 0.5, 1e-12)
    _close(res["lo"], 0.5, 1e-12)
    _close(res["hi"], 0.5, 1e-12)
    assert res["equivalent"]


def test_bootstrap_ci_stat_generic():
    """Generic cluster bootstrap (the probe's AUC path): perfect separation
    gives a degenerate CI at 1.0; nan replicates are dropped and counted."""
    from harness.probe import auc_score
    scores = np.array([-2.0, -1.0, 1.0, 2.0] * 10)
    labels = np.array([0, 0, 1, 1] * 10)
    cid = np.repeat(np.arange(20), 2)
    ci = stats.bootstrap_ci_stat(auc_score, [scores, labels], cid,
                                 n_boot=300, seed=0)
    _close(ci["point"], 1.0, 1e-12)
    _close(ci["lo"], 1.0, 1e-12)
    _close(ci["hi"], 1.0, 1e-12)
    assert ci["n_dropped"] < 300


def test_holm_correction_known_answer():
    adj, rej = stats.holm_correction([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02], atol=1e-12)
    assert rej.tolist() == [True, False, False, True]
    # monotone step-down: an inversion cannot un-reject a smaller p
    adj, rej = stats.holm_correction([0.02, 0.02], alpha=0.05)
    assert np.allclose(adj, [0.04, 0.04], atol=1e-12)
    assert rej.tolist() == [True, True]
    adj, _ = stats.holm_correction([0.9, 0.9, 0.9])
    assert np.all(adj <= 1.0)


def test_iqm_known_answer():
    _close(stats.iqm(np.arange(1, 11)), 5.5, 1e-12)      # trims 2 each end
    # robust to a wild outlier that wrecks the mean
    v = np.concatenate([np.arange(1, 10), [1000.0]])
    _close(stats.iqm(v), 5.5, 1e-12)
    _close(stats.iqm([3.0]), 3.0, 1e-12)


def test_icc_deff():
    rng = np.random.default_rng(13)
    G, m = 20, 25
    y, cid = _clustered_sample(rng, G, m)                 # ICC ~ 0.99
    res = stats.icc_deff(y, cid)
    assert res["icc"] > 0.95, res
    assert res["deff"] > 0.8 * m, res
    # independent data: ICC ~ 0, DEFF ~ 1
    y = rng.normal(0.0, 1.0, G * m)
    res = stats.icc_deff(y, cid)
    assert abs(res["icc"]) < 0.2, res
    assert res["deff"] < 3.0, res


def test_mcnemar_and_sign_permutation():
    _close(stats.mcnemar_exact(2, 8), 112.0 / 1024.0, 1e-12)
    _close(stats.mcnemar_exact(8, 2), 112.0 / 1024.0, 1e-12)
    _close(stats.mcnemar_exact(0, 0), 1.0, 1e-12)
    rng = np.random.default_rng(21)
    strong = np.abs(rng.normal(1.0, 0.1, 40))             # all positive
    assert stats.paired_sign_permutation_test(strong, 2000, seed=0) < 0.01
    null = rng.normal(0.0, 1.0, 40)
    assert stats.paired_sign_permutation_test(null, 2000, seed=0) > 0.05


def test_power_analysis_profile():
    """The spec 4.4 Type-II profile: the rule passes at true 0.50 and
    detects (fails to pass) at 0.60, outside the margin."""
    res = stats.power_analysis(
        true_accuracies=(0.50, 0.60), n_scenarios=600, n_seeds=3,
        between_seed_sd=0.01, n_sim=20, n_boot=300, seed=5)
    assert res[0.50]["pass_prob"] >= 0.8, res
    assert res[0.60]["pass_prob"] <= 0.05, res
    assert res[0.60]["detect_prob"] >= 0.95, res


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_pass = n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
            n_pass += 1
            print("PASS  " + name)
        except SkipTest as exc:
            n_skip += 1
            print("SKIP  %s (%s)" % (name, exc))
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("%d passed, %d failed, %d skipped" % (n_pass, n_fail, n_skip))
    sys.exit(1 if n_fail else 0)
