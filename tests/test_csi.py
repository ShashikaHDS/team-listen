"""Tests for harness/csi.py on synthetic records with ANALYTICALLY KNOWN
CSI (M1_SPEC 6.1-6.4), plus one real-record integration pass on the CPU
stand-in env.

The analytic ladder (all on hand-built five-lane records where every
per-episode divergence is exact by construction):

* identical behaviour under the instruction swap  -> share = 0, ratio = 0;
* behaviour changing exactly as much as seed noise -> share = 0.5,
  ratio = 1 (the "equally influential" calibration point of spec 6.1);
* a large instructed change                        -> ratio >> 1, share
  near 1 (bounded);
* the degenerate denominator the red team demanded (spec 6.1 [FIXED:
  divergent estimand] / OPEN(12)): D_seed == 0 with D_lang > 0 gives the
  PRIMARY share a finite 1.0 with ``nuisance_degenerate`` flagged and a
  ratio of inf -- never a crash, never a silent epsilon; both channels
  inert gives share NaN with ``degenerate`` flagged.

Plus: per-semantic-category vs aggregate consistency, episode-clustered
bootstrap CI behaviour (including degenerate-replicate counting), the
spec 6.4(a) paired sign test, the flip/TV/intent divergence kinds, and
the guard rails (flip needs a variant; 'blank' is a third column, never a
denominator; TV needs recorded logits; nuisance lane must exist).

Integration: a Blind-arm five-lane rollout on ``CPUFleetEnv`` (the smoke
bank) must yield CSI_traj share == 0 EXACTLY (the paired-lane machine
check guarantees D_lang == 0 bit-for-bit) with a live seed denominator,
and a SymbolPO arm with a language-weighted policy must yield D_lang > 0.

pytest-compatible; standalone: ``python tests/test_csi.py``.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from harness import csi                              # noqa: E402
from harness import rollout as ro                    # noqa: E402
from tasks.team_listen import obs_layout as L        # noqa: E402

N_BOOT = 800          # spec production value is 10,000; small here for speed
N_PERM = 800


def _isnan(x):
    return math.isnan(float(x))


# ---------------------------------------------------------------------------
# Synthetic five-lane records (RolloutRecord-shaped, hand-built)
# ---------------------------------------------------------------------------

def _blank_record(T, plan, N=2):
    E = plan.n_envs
    return SimpleNamespace(
        agents=("robot_0", "robot_1"), mode="argmax", t_steps=T,
        positions=torch.zeros((T, E, N, 2), dtype=torch.int16),
        intended=torch.zeros((T, E, N), dtype=torch.int8),
        executed=torch.zeros((T, E, N), dtype=torch.int8),
        margins=torch.zeros((T, E, N), dtype=torch.float32),
        rewards=torch.zeros((T, E, N), dtype=torch.float32),
        active=torch.ones((T, E), dtype=torch.bool),
        first_done=torch.full((E,), T - 1, dtype=torch.int16),
        completed=torch.ones((E,), dtype=torch.bool),
        correct=torch.zeros((E,), dtype=torch.bool),
        outcome=torch.zeros((E,), dtype=torch.int8),
        latch_time=torch.full((E, N), -1, dtype=torch.int16),
        latch_slot=torch.full((E, N), -1, dtype=torch.int8),
        scenario_id=plan.scenarios.clone(),
        slip_stream=plan.streams.to(torch.int8),
        instr_class=plan.instr_classes.clone(),
        episode_return=torch.zeros((E, N), dtype=torch.float32),
        n_obstacle_collisions=torch.zeros((E, N), dtype=torch.int16),
        n_robot_collisions=torch.zeros((E, N), dtype=torch.int16),
        logits=None,
    )


def _set_row_offset(rec, plan, lane, offsets):
    """Give lane ``lane``'s agent-0 row a constant per-group offset.

    ``offsets`` is per-group (K,); a constant row offset of d yields a
    per-step L1 position divergence vs the factual lane of exactly d for
    agent 0, hence an agent-mean of d/2 -- so per-episode
    D = offsets/2, EXACTLY, for every group.
    """
    sl = plan.lane_slice(lane)
    off = torch.as_tensor(offsets, dtype=torch.int16).reshape(1, -1)
    rec.positions[:, sl, 0, 0] = off


def _five_lane_case(K=8, T=6, lang=0.0, seed_noise=1.0,
                    factual_classes=None):
    """Synthetic record with exact per-episode divergences.

    ``lang`` / ``seed_noise`` are per-group row offsets (scalar or (K,)),
    so D_lang = lang/2 and D_seed = seed_noise/2 per episode, exactly.
    """
    base = torch.arange(K)
    plan = ro.make_five_lane_plan(base, factual_classes=factual_classes)
    rec = _blank_record(T, plan)
    lang_t = torch.as_tensor(lang).expand(K) if torch.as_tensor(
        lang).dim() == 0 else torch.as_tensor(lang)
    seed_t = torch.as_tensor(seed_noise).expand(K) if torch.as_tensor(
        seed_noise).dim() == 0 else torch.as_tensor(seed_noise)
    _set_row_offset(rec, plan, ro.LANE_COUNTERFACTUAL, lang_t)
    _set_row_offset(rec, plan, ro.LANE_SEED, seed_t)
    return rec, plan


def _agg(rec, plan, **kw):
    kw.setdefault("n_boot", N_BOOT)
    kw.setdefault("n_perm", N_PERM)
    return csi.compute_csi(rec, plan, **kw)


# ---------------------------------------------------------------------------
# The analytic ladder
# ---------------------------------------------------------------------------

def test_csi_zero_when_instruction_inert():
    # identical behaviour under the instruction swap: D_lang == 0
    rec, plan = _five_lane_case(lang=0.0, seed_noise=1.0)
    res = _agg(rec, plan, divergence="traj")
    a = res.aggregate
    assert a.n_pairs == plan.n_base
    assert a.share == 0.0 and a.ratio == 0.0
    assert not a.degenerate and not a.nuisance_degenerate
    assert a.d_lang_mean == 0.0
    assert abs(a.d_nuis_mean - 0.5) < 1e-6
    assert a.share_ci == (0.0, 0.0)
    # the sign test must NOT reject: language is not more influential
    assert a.p_sign_greater > 0.5
    # published marginals carry the raw distributions (spec 6.1)
    assert set(res.marginals) == {"d_lang", "d_seed", "d_spawn", "d_blank"}
    assert torch.equal(res.marginals["d_lang"],
                       torch.zeros(plan.n_base))


def test_csi_one_when_equal_to_seed_noise():
    # behaviour changes exactly as much as seed noise: the calibration
    # point where language and nuisance are equally influential
    rec, plan = _five_lane_case(lang=1.0, seed_noise=1.0)
    a = _agg(rec, plan, divergence="traj").aggregate
    assert abs(a.share - 0.5) < 1e-9
    assert abs(a.ratio - 1.0) < 1e-9
    assert not a.degenerate and not a.nuisance_degenerate
    # every replicate resamples identical per-group divergences -> the CI
    # degenerates onto the point (a property, not an accident)
    assert abs(a.share_ci[0] - 0.5) < 1e-9
    assert abs(a.share_ci[1] - 0.5) < 1e-9


def test_csi_large_instructed_change():
    # a large instructed change: ratio >> 1, share near (bounded by) 1
    rec, plan = _five_lane_case(lang=25.0, seed_noise=1.0)
    a = _agg(rec, plan, divergence="traj").aggregate
    assert abs(a.ratio - 25.0) < 1e-9
    assert a.ratio > 10.0
    assert abs(a.share - 25.0 / 26.0) < 1e-9
    assert a.share < 1.0
    # the paired sign test rejects: every pair has D_lang > D_seed, so
    # only the all-plus sign assignment reaches the observed statistic
    assert a.p_sign_greater < 0.05


def test_csi_degenerate_denominator_regularised():
    # the red-team case (spec 6.1 [FIXED] / OPEN(12)): a grounded,
    # nuisance-invariant policy drives D_seed -> 0 where the RATIO
    # diverges; the primary share must stay finite and flagged.
    rec, plan = _five_lane_case(lang=2.0, seed_noise=0.0)
    a = _agg(rec, plan, divergence="traj").aggregate
    assert a.share == 1.0                      # bounded regularisation
    assert a.nuisance_degenerate               # OPEN(12) trigger readout
    assert not a.degenerate                    # share itself is defined
    assert a.ratio == float("inf")             # display form, honest inf
    assert a.frac_nuis_zero == 1.0
    # every bootstrap ratio replicate is degenerate: counted, not imputed
    assert a.n_degenerate_ratio_reps == a.n_boot
    assert _isnan(a.ratio_ci[0]) and _isnan(a.ratio_ci[1])
    assert a.share_ci == (1.0, 1.0)


def test_csi_fully_degenerate_is_nan_not_crash():
    # both channels inert: 0/0 -- CSI unidentifiable, said out loud
    rec, plan = _five_lane_case(lang=0.0, seed_noise=0.0)
    a = _agg(rec, plan, divergence="traj").aggregate
    assert a.degenerate and a.nuisance_degenerate
    assert _isnan(a.share) and _isnan(a.ratio)
    assert a.n_degenerate_share_reps == a.n_boot
    # all-zero paired differences: the sign test cannot reject
    assert a.p_sign_greater == 1.0


def test_csi_bootstrap_ci_covers_truth():
    # heterogeneous groups with equal pooled sums: true share 0.5, but
    # replicates vary, so the CI must straddle 0.5 with nonzero width
    K = 16
    lang = torch.tensor([1.0, 2.0] * (K // 2))
    seed = torch.tensor([2.0, 1.0] * (K // 2))
    rec, plan = _five_lane_case(K=K, lang=lang, seed_noise=seed)
    a = _agg(rec, plan, divergence="traj").aggregate
    assert abs(a.share - 0.5) < 1e-9
    lo, hi = a.share_ci
    assert lo < 0.5 < hi
    assert 0.0 < hi - lo < 0.4
    ral, rah = a.ratio_ci
    assert ral < 1.0 < rah


def test_csi_per_category_and_aggregate():
    # category 0: equal influence (share .5, ratio 1); category 1: lang
    # 3x seed (share .75, ratio 3); aggregate pools the sums (2/3, 2)
    K = 8
    cats = torch.tensor([0] * 4 + [1] * 4)
    lang = torch.tensor([1.0] * 4 + [3.0] * 4)
    rec, plan = _five_lane_case(K=K, lang=lang, seed_noise=1.0,
                                factual_classes=cats)
    res = _agg(rec, plan, divergence="traj")     # categories default to
    assert torch.equal(res.categories, cats)     # the factual instr_class
    assert set(res.per_category) == {0, 1}
    assert abs(res.per_category[0].share - 0.5) < 1e-9
    assert abs(res.per_category[0].ratio - 1.0) < 1e-9
    assert abs(res.per_category[1].share - 0.75) < 1e-9
    assert abs(res.per_category[1].ratio - 3.0) < 1e-9
    assert abs(res.aggregate.share - 2.0 / 3.0) < 1e-9
    assert abs(res.aggregate.ratio - 2.0) < 1e-9
    assert res.per_category[0].n_pairs == 4
    # explicit categories override the default
    res2 = _agg(rec, plan, divergence="traj",
                categories=torch.zeros(K, dtype=torch.long))
    assert set(res2.per_category) == {0}
    assert abs(res2.per_category[0].share - res2.aggregate.share) < 1e-12


def test_csi_flip_divergence():
    K = 4
    plan = ro.make_five_lane_plan(torch.arange(K))
    rec = _blank_record(6, plan)
    # factual assignment everywhere: r0 -> slot 0, r1 -> slot 1
    rec.latch_slot[:, 0] = 0
    rec.latch_slot[:, 1] = 1
    cf = plan.lane_slice(ro.LANE_COUNTERFACTUAL)
    sd = plan.lane_slice(ro.LANE_SEED)
    # counterfactual: groups 0, 1 flip; group 2 unchanged; group 3
    # incomplete (dropped pairwise)
    rec.latch_slot[cf.start + 0] = torch.tensor([1, 0], dtype=torch.int8)
    rec.latch_slot[cf.start + 1] = torch.tensor([1, 0], dtype=torch.int8)
    rec.completed[cf.start + 3] = False
    # seed lane: only group 0 flips
    rec.latch_slot[sd.start + 0] = torch.tensor([1, 0], dtype=torch.int8)
    res = _agg(rec, plan, divergence="flip", variant="RoleBinding")
    a = res.aggregate
    # valid pairs {0,1,2}: D_lang = [1,1,0], D_seed = [1,0,0]
    assert a.n_pairs == 3
    assert abs(a.share - 2.0 / 3.0) < 1e-9
    assert abs(a.ratio - 2.0) < 1e-9
    dl = res.marginals["d_lang"]
    assert dl[:3].tolist() == [1.0, 1.0, 0.0] and _isnan(dl[3])
    # precedence codes flip on latch-ORDER, not slots
    rec.latch_time[:, 0] = 2
    rec.latch_time[:, 1] = 5
    rec.latch_time[cf.start + 0] = torch.tensor([5, 2], dtype=torch.int16)
    d_prec = csi.lane_divergence(rec, plan, ro.LANE_COUNTERFACTUAL,
                                 "flip", variant="Precedence")
    assert float(d_prec[0]) == 1.0 and float(d_prec[1]) == 0.0


def test_csi_tv_and_intent_divergences():
    K, T, N, A = 4, 6, 2, 5
    plan = ro.make_five_lane_plan(torch.arange(K))
    rec = _blank_record(T, plan)
    big = 50.0
    logits = torch.zeros((T, plan.n_envs, N, A))
    logits[..., 0] = big                       # everyone prefers action 0
    cf = plan.lane_slice(ro.LANE_COUNTERFACTUAL)
    sd = plan.lane_slice(ro.LANE_SEED)
    # counterfactual: BOTH agents flip to action 1 -> marginal TV 1
    logits[:, cf, :, 0] = 0.0
    logits[:, cf, :, 1] = big
    # seed: only agent 0 flips -> marginal TV mean 0.5
    logits[:, sd, 0, 0] = 0.0
    logits[:, sd, 0, 1] = big
    rec.logits = logits
    a = _agg(rec, plan, divergence="tv_step").aggregate
    assert abs(a.d_lang_mean - 1.0) < 1e-4
    assert abs(a.d_nuis_mean - 0.5) < 1e-4
    assert abs(a.share - 2.0 / 3.0) < 1e-3
    assert abs(a.ratio - 2.0) < 1e-3
    # joint TV: any single-agent flip already makes the joints disjoint
    aj = _agg(rec, plan, divergence="tv_joint").aggregate
    assert abs(aj.d_lang_mean - 1.0) < 1e-4
    assert abs(aj.d_nuis_mean - 1.0) < 1e-4
    assert abs(aj.share - 0.5) < 1e-3
    # intent (Hamming on pre-revert intended actions, spec 6.1)
    rec.intended[:, cf, :] = 1                 # both agents differ: D = 1
    rec.intended[:, sd, 0] = 1                 # one agent differs: D = 0.5
    ai = _agg(rec, plan, divergence="intent").aggregate
    assert abs(ai.d_lang_mean - 1.0) < 1e-6
    assert abs(ai.d_nuis_mean - 0.5) < 1e-6
    assert abs(ai.share - 2.0 / 3.0) < 1e-6
    # decision-point mask restricted to zero steps -> all pairs NaN ->
    # fully degenerate estimate, no crash
    empty = torch.zeros((T, K), dtype=torch.bool)
    a0 = _agg(rec, plan, divergence="intent", decision_mask=empty).aggregate
    assert a0.n_pairs == 0 and a0.degenerate and _isnan(a0.share)


def test_estimate_clusters_and_nan_pairs():
    d_lang = torch.tensor([2.0, 2.0, 4.0, 4.0, float("nan"), 6.0])
    d_nuis = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0, float("nan")])
    # NaN in EITHER term drops the pair (paired estimand)
    est = csi.estimate(d_lang, d_nuis, n_boot=N_BOOT, n_perm=N_PERM)
    assert est.n_pairs == 4
    assert abs(est.ratio - 2.0) < 1e-9
    assert abs(est.share - 2.0 / 3.0) < 1e-9
    # clustered resampling: same point estimate, still no crash, and the
    # cluster ids collapse 4 pairs into 2 resampling units
    cl = torch.tensor([0, 0, 1, 1, 2, 3])
    est_c = csi.estimate(d_lang, d_nuis, clusters=cl, n_boot=N_BOOT,
                         n_perm=N_PERM)
    assert abs(est_c.share - est.share) < 1e-12
    assert abs(est_c.ratio - est.ratio) < 1e-12


def test_paired_sign_test_calibration():
    # symmetric differences: no rejection
    d_lang = torch.tensor([1.0, 2.0, 1.0, 2.0])
    d_nuis = torch.tensor([2.0, 1.0, 2.0, 1.0])
    p_g, p_2 = csi.paired_sign_test(d_lang, d_nuis, n_perm=2000, seed=3)
    assert p_g > 0.2 and p_2 > 0.2
    # uniformly positive differences over 12 pairs: strong rejection
    d_lang = torch.full((12,), 3.0)
    d_nuis = torch.full((12,), 1.0)
    p_g, p_2 = csi.paired_sign_test(d_lang, d_nuis, n_perm=4000, seed=3)
    assert p_g < 0.01
    # empty input -> NaN
    p_g, p_2 = csi.paired_sign_test(torch.full((3,), float("nan")),
                                    torch.ones(3))
    assert _isnan(p_g) and _isnan(p_2)


def test_compute_csi_guards():
    rec, plan = _five_lane_case()
    # flip needs a variant (defines the outcome code, spec 2.3/3.3)
    try:
        csi.compute_csi(rec, plan, divergence="flip", n_boot=10, n_perm=10)
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("flip without variant accepted")
    # blank is a third column, never a denominator (spec 6.1)
    try:
        csi.compute_csi(rec, plan, nuisance="blank", n_boot=10, n_perm=10)
    except ValueError as exc:
        assert "third column" in str(exc)
    else:
        raise AssertionError("blank accepted as a denominator")
    # TV needs recorded logits
    try:
        csi.compute_csi(rec, plan, divergence="tv_step", n_boot=10,
                        n_perm=10)
    except ValueError as exc:
        assert "logits" in str(exc)
    else:
        raise AssertionError("tv_step without logits accepted")
    # unknown divergence
    try:
        csi.compute_csi(rec, plan, divergence="wasserstein", n_boot=10,
                        n_perm=10)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown divergence accepted")
    # a two-lane paired plan has no seed lane to normalise by
    plan2 = ro.make_paired_plan(torch.arange(4))
    rec2 = _blank_record(6, plan2)
    try:
        csi.compute_csi(rec2, plan2, n_boot=10, n_perm=10)
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("missing nuisance lane accepted")


def test_lane_tensor_layout():
    rec, plan = _five_lane_case(K=4, T=3)
    K = plan.n_base
    # (E,) field: factual slab is the first K entries
    assert torch.equal(csi.lane_tensor(rec.scenario_id, plan,
                                       ro.LANE_FACTUAL),
                       rec.scenario_id[:K])
    # (T, E, ...) field: lane slab preserves the step-major layout
    seed_pos = csi.lane_tensor(rec.positions, plan, ro.LANE_SEED,
                               trajectory=True)
    sl = plan.lane_slice(ro.LANE_SEED)
    assert torch.equal(seed_pos, rec.positions[:, sl])


# ---------------------------------------------------------------------------
# Integration: real records from the CPU stand-in env (smoke bank)
# ---------------------------------------------------------------------------

def test_csi_on_real_blind_and_symbolpo_records():
    import test_paired_lane_identity as tpl    # reuses the smoke bank +
    K = tpl.N_BASE                             # policies of spec sec. 7
    plan = ro.make_five_lane_plan(torch.arange(K))

    # Blind arm + state-reading greedy controller: run_lanes' machine
    # check (spec 4.3) asserts L0 == L1 bit-for-bit, so D_lang == 0
    # EXACTLY and CSI must be identically zero with a live denominator.
    env = tpl._make_env("Blind", plan.n_envs)
    rec = ro.run_lanes(env, tpl.GreedyPolicy(env), plan, mode="argmax")
    res = csi.compute_csi(rec, plan, divergence="traj", n_boot=300,
                          n_perm=300)
    a = res.aggregate
    assert a.n_pairs == K
    assert float(res.marginals["d_lang"].abs().sum()) == 0.0
    assert float(res.marginals["d_blank"].abs().sum()) == 0.0
    assert a.d_nuis_mean > 0.0, \
        "slip stream 0 vs 1 produced no divergence: D_seed has no " \
        "referent on this bank (spec 1.10 [FIXED])"
    assert a.share == 0.0 and a.ratio == 0.0
    assert not a.degenerate and not a.nuisance_degenerate
    # the spawn intervention moved someone too (marginal published)
    assert float(res.marginals["d_spawn"].sum()) > 0.0

    # SymbolPO + language-weighted policy: the counterfactual lane must
    # actually diverge, so D_lang > 0 and the share leaves zero.
    env_s = tpl._make_env("SymbolPO", plan.n_envs)
    policy = tpl.LinearPolicy(L.OBS_DIM, env_s.cfg.possible_agents,
                              seed=123, lang_weight=50.0)
    rec_s = ro.run_lanes(env_s, policy, plan, mode="argmax")
    res_s = csi.compute_csi(rec_s, plan, divergence="traj", n_boot=300,
                            n_perm=300)
    assert res_s.aggregate.d_lang_mean > 0.0
    assert res_s.aggregate.share > 0.0
    # per-category keys are the factual semantic classes present
    assert set(res_s.per_category) == set(
        torch.unique(res_s.categories).tolist())


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
