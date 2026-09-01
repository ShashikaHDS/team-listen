"""Certificate decision-procedure tests on the CPU stand-in env
(M1_SPEC 2.4 / 4.3 / 4.4; harness/certificate.py + harness/probe.py).

What is proven here, on real ``run_lanes`` records over a smoke scenario
bank (built by ``scripts/build_scenario_bank.py`` into a temp dir):

* **A truly blind policy PASSES** -- Blind arm, scripted geometry-reading
  controller, paired within-scenario manifest: the machine check holds,
  the recomputed accuracy is exactly the paired-manifest identity 0.5,
  the TOST equivalence passes, the leak probe's held-out AUC CI contains
  0.5, the planted-leak positive control is detected, and the verdict is
  PASS.
* **A leaky record FAILS VIA THE PROBE** -- same blind policy, natural
  (unpaired) manifest whose instruction classes are drawn correlated with
  the scenario's geometric default (the spec 4.1 Leaky/canary coupling,
  rho ~ 0.9): the trajectory probe detects the coupling (AUC CI excludes
  0.5) and the verdict is FAIL, with the at-chance TOST also rejecting.
* **The positive control FAILS the certificate** -- a language-sensitive
  policy (SymbolPO arm, instruction-weighted linear policy) audited as if
  blind: the paired-lane trajectory-equality machine check catches the
  divergence and the verdict is FAIL.
* Probe unit behaviour (AUC known answers, planted control sensitivity),
  ground-truth recomputation on synthetic latch tensors for both
  variants, and the pyyaml-free preregistration reader.

No pytest: run ``python tests/test_certificate.py`` from the repo root.
"""

import atexit
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import certificate as ce_mod              # noqa: E402
from harness import cpu_env as ce                      # noqa: E402
from harness import probe as pr                        # noqa: E402
from harness import rollout as ro                      # noqa: E402
from tasks.team_listen import grid_core                # noqa: E402
from tasks.team_listen import obs_layout as L          # noqa: E402


class SkipTest(Exception):
    pass


SMOKE_K = 32
SMOKE_SEED = 20260831        # same smoke bank as test_paired_lane_identity
N_BASE = 24

_TMP = None
_BANK_PATHS = {}


def _tmpdir():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="team_listen_certificate_")
        atexit.register(shutil.rmtree, _TMP, True)
    return _TMP


def _bank_path(variant="RoleBinding"):
    if variant not in _BANK_PATHS:
        spec = importlib.util.spec_from_file_location(
            "team_listen_bsb_certificate",
            str(REPO_ROOT / "scripts" / "build_scenario_bank.py"))
        bsb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bsb)
        payload = bsb.build_bank(variant, SMOKE_K,
                                 bsb.derive_seed_list(SMOKE_SEED, SMOKE_K))
        out_dir = os.path.join(_tmpdir(), "bank_%s" % variant)
        pt_path, _, _ = bsb.save_bank(payload, out_dir,
                                      {"mode": "derived",
                                       "master_seed": SMOKE_SEED})
        _BANK_PATHS[variant] = pt_path
    return _BANK_PATHS[variant]


def _make_env(arm, n_envs, variant="RoleBinding", **cfg_overrides):
    cfg = ce.make_cfg(arm=arm, variant=variant, bank_path=_bank_path(variant),
                      debug_asserts=True, **cfg_overrides)
    return ce.CPUFleetEnv(cfg, num_envs=n_envs)


def _balanced_classes(n, seed=17):
    """Seeded balanced 0/1 labels, decorrelated from scenario order."""
    g = torch.Generator().manual_seed(seed)
    half = n // 2
    cls = torch.cat([torch.zeros(n - half, dtype=torch.long),
                     torch.ones(half, dtype=torch.long)])
    return cls[torch.randperm(n, generator=g)]


def _geom_bit(env, base_sids):
    """Realised-assignment bit of the slot-descending greedy controller:
    0 iff target slot 0 is the LEFT station (r0 -> slot 0 -> left)."""
    bank = env._bank
    col = bank.target[base_sids][..., 1].float()
    valid = bank.target_valid[base_sids]
    left = torch.where(valid, col,
                       torch.full_like(col, float("inf"))).argmin(dim=1)
    return (left != 0).long()


# ---------------------------------------------------------------------------
# Deterministic test policies (mirrors test_paired_lane_identity.py)
# ---------------------------------------------------------------------------

class LinearPolicy:
    """Fixed random-projection logits over the observation; ``lang_weight``
    scales the LANG_SLICE rows so a language-sensitive policy (the
    positive control) is one parameter away."""

    def __init__(self, obs_dim, agents, seed=123, lang_weight=None):
        g = torch.Generator().manual_seed(seed)
        self.w = {}
        for name in agents:
            w = torch.randn(obs_dim, grid_core.N_ACTIONS, generator=g)
            if lang_weight is not None:
                assert obs_dim == L.OBS_DIM
                w[L.LANG_SLICE] *= float(lang_weight)
            self.w[name] = w

    def __call__(self, obs):
        return {name: obs[name] @ self.w[name] for name in obs}


class GreedyPolicy:
    """Scripted geometry reader: robot i descends ``dist_field[slot=i]`` --
    instruction-blind by construction (never touches lang_vec)."""

    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        env = self.env
        E, N = env.num_envs, L.N_AGENTS
        deltas = grid_core.DELTAS
        eidx = torch.arange(E)
        logits = torch.zeros((E, N, grid_core.N_ACTIONS))
        for i in range(N):
            df = env.dist_field[eidx, i].long()
            pos = env.pos[:, i].long()
            for a in range(grid_core.N_ACTIONS):
                cell = pos + deltas[a]
                cell[:, 0].clamp_(0, L.R - 1)
                cell[:, 1].clamp_(0, L.C - 1)
                d = df[eidx, cell[:, 0], cell[:, 1]].float()
                blocked = (env.occ[eidx, cell[:, 0], cell[:, 1]] != 0) | (d < 0)
                logits[:, i, a] = -torch.where(
                    blocked, torch.full_like(d, 1e6), d)
        return {name: logits[:, i]
                for i, name in enumerate(env.cfg.possible_agents)}


# ---------------------------------------------------------------------------
# Probe unit behaviour
# ---------------------------------------------------------------------------

def test_probe_auc_known_answers():
    assert pr.auc_score([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]) == 1.0
    assert pr.auc_score([4.0, 3.0, 2.0, 1.0], [0, 0, 1, 1]) == 0.0
    assert pr.auc_score([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]) == 0.5
    # hand count: pairs won 3.5 of 4 (one tie) -> 0.875
    assert pr.auc_score([1.0, 3.0, 3.0, 5.0], [0, 0, 1, 1]) == 0.875
    assert np.isnan(pr.auc_score([1.0, 2.0], [1, 1]))


def test_probe_null_and_planted_control():
    """Pure-noise features: chance stays in the CI.  Planted label column:
    the identical pipeline must detect it (the audit's teeth)."""
    rng = np.random.default_rng(3)
    n, p = 120, 30
    X = rng.normal(0.0, 1.0, (n, p))
    y = (np.arange(n) % 2).astype(np.int64)
    cid = np.arange(n)                      # one scenario per episode
    null = pr.leak_probe(X, y, cid, n_boot=500, seed=0)
    assert null["ok"] and null["chance_in_ci"] and not null["leak"], null
    planted = pr.planted_leak_control(X, y, cid, n_boot=500, seed=0)
    assert planted["detected"], planted
    assert planted["auc"] > 0.9, planted["auc"]
    # a genuinely leaky feature matrix is flagged
    X_leak = X.copy()
    X_leak[:, 0] += 3.0 * (2.0 * y - 1.0)
    leaky = pr.leak_probe(X_leak, y, cid, n_boot=500, seed=0)
    assert leaky["leak"], leaky
    # single-class labels: not runnable, never a leak claim
    res = pr.leak_probe(X, np.zeros(n, dtype=np.int64), cid)
    assert not res["ok"] and not res["leak"]


def test_probe_cluster_folds_never_split_a_scenario():
    cid = np.repeat(np.arange(10), 2)       # paired episodes per scenario
    folds, k = pr.cluster_folds(cid, 5, seed=1)
    assert k == 5
    for g in range(10):
        rows = folds[cid == g]
        assert np.unique(rows).shape[0] == 1, \
            "scenario %d split across folds (paired-lane leakage)" % g


# ---------------------------------------------------------------------------
# Ground-truth recomputation (synthetic latch tensors, both variants)
# ---------------------------------------------------------------------------

def _fake_record(**kw):
    base = dict(mode="argmax")
    base.update(kw)
    return SimpleNamespace(**base)


def test_realised_assignment_role_binding():
    # bank: scenario 0 has slot0 at col 1 (left), slot1 at col 10;
    #       scenario 1 has slot0 at col 10, slot1 at col 2 (left)
    bank = SimpleNamespace(
        target=torch.tensor([[[5, 1], [5, 10], [0, 0]],
                             [[5, 10], [5, 2], [0, 0]]], dtype=torch.int16),
        target_valid=torch.tensor([[True, True, False],
                                   [True, True, False]]))
    rec = _fake_record(
        completed=torch.tensor([True, True, True, False]),
        latch_slot=torch.tensor([[0, 1], [1, 0], [0, 1], [-1, -1]],
                                dtype=torch.int8),
        latch_time=torch.tensor([[3, 5], [4, 6], [2, 9], [-1, -1]],
                                dtype=torch.int16),
        scenario_id=torch.tensor([0, 0, 1, 1]),
        instr_class=torch.tensor([0, 1, 0, 0]),
        correct=torch.tensor([True, True, False, False]))
    A = ce_mod.realised_assignment(rec, bank, "RoleBinding")
    # ep0: r0 in slot0 = col1 = left -> A=0; ep1: r0 in slot1 = col10 -> A=1
    # ep2: r0 in slot0 = col10 (right in scenario 1) -> A=1; ep3 incomplete
    assert A.tolist() == [0, 1, 1, -1]
    Y, C, _ = ce_mod.recompute_outcomes(rec, bank, "RoleBinding")
    assert Y.tolist() == [True, True, False, False]
    assert torch.equal(Y, rec.correct)


def test_realised_assignment_precedence():
    rec = _fake_record(
        completed=torch.tensor([True, True, True]),
        latch_slot=torch.zeros((3, 2), dtype=torch.int8),
        latch_time=torch.tensor([[2, 7], [9, 4], [5, 5]], dtype=torch.int16),
        scenario_id=torch.tensor([0, 1, 2]),
        instr_class=torch.tensor([0, 0, 1]),
        correct=torch.tensor([True, False, False]))
    A = ce_mod.realised_assignment(rec, None, "Precedence")
    # dt>0 -> r0 first -> A=0; dt<0 -> A=1; tie -> undefined (-1, Y=0)
    assert A.tolist() == [0, 1, -1]
    Y, _, _ = ce_mod.recompute_outcomes(rec, None, "Precedence")
    assert Y.tolist() == [True, False, False]


def test_preregistration_reader():
    prereg = ce_mod.load_preregistration(path=None) \
        if os.path.isfile(ce_mod.PREREGISTRATION_PATH) \
        else ce_mod.load_preregistration(path="__does_not_exist__")
    for key, value in ce_mod.DEFAULT_PREREGISTRATION.items():
        assert key in prereg, key
    miss = ce_mod.load_preregistration(path="__does_not_exist__")
    assert miss["preregistration_source"] == "embedded_defaults"
    assert miss["alpha"] == 0.05
    assert miss["equivalence_margin_delta"] == 0.05
    # flat-YAML override file (pyyaml-free subset reader)
    path = os.path.join(_tmpdir(), "prereg_override.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# comment line\n"
                 "alpha: 0.10\n"
                 "equivalence_margin_delta: 0.02\n"
                 "bootstrap_method: percentile   # trailing comment\n"
                 "some_flag: true\n"
                 "nested:\n"
                 "  ignored_key: 3\n")
    over = ce_mod.load_preregistration(path=path)
    assert over["alpha"] == 0.10
    assert over["equivalence_margin_delta"] == 0.02
    assert over["bootstrap_method"] == "percentile"
    assert over["some_flag"] is True
    assert "ignored_key" not in over
    assert over["chance_level"] == 0.5          # default survives the merge
    assert over["preregistration_source"] == path


def test_make_natural_plan():
    cls = _balanced_classes(8)
    plan = ce_mod.make_natural_plan(torch.arange(8), cls)
    assert plan.n_lanes == 1 and plan.n_envs == 8
    assert plan.lanes == (ro.LANE_FACTUAL,)
    assert torch.equal(plan.instr_classes, cls)
    assert not bool(plan.blank.any()) and not bool(plan.spawn_alt.any())


# ---------------------------------------------------------------------------
# The three headline cases
# ---------------------------------------------------------------------------

def test_truly_blind_policy_passes():
    """Blind arm + geometry-reading scripted policy on the paired manifest:
    every clause holds and the verdict is PASS."""
    env = _make_env("Blind", 2 * N_BASE)
    base = torch.arange(N_BASE)
    plan = ro.make_paired_plan(base, classes_a=_balanced_classes(N_BASE))
    rec = ro.run_lanes(env, GreedyPolicy(env), plan, mode="argmax")
    result, report = ce_mod.run_certificate(
        rec, plan, env._bank, "RoleBinding", seed=0)

    cl = result["clauses"]
    assert cl["paired_lane_identity"]["checked"]
    assert cl["paired_lane_identity"]["passed"]
    assert cl["scorer_consistency"]["passed"]
    # the paired-manifest algebraic identity (spec 4.3): accuracy == 0.5
    at = cl["at_chance"]
    assert at["n_scored"] > 0
    assert abs(at["accuracy"] - 0.5) < 1e-12
    assert at["tost"]["lo"] == 0.5 and at["tost"]["hi"] == 0.5
    assert at["passed"]
    assert at["binom_p_two_sided"] > 0.9        # display: dead-centre
    nd = cl["non_degeneracy"]
    assert nd["in_range"], nd
    for name, res in cl["probe"].items():
        assert res["ok"], (name, res)
        assert not res["leak"], (name, res)
    assert cl["positive_control"]["detected"], cl["positive_control"]
    assert result["verdict"] == "PASS", (result["verdict"],
                                         result["fail_reasons"],
                                         result["uninformative_reasons"])
    assert "VERDICT: PASS" in report
    assert "INSTRUCTION-LEAKAGE AUDIT" in report


def test_leaky_record_fails_via_probe():
    """Blind arm on a NATURAL manifest whose instruction classes were drawn
    correlated with the geometric default (the spec 4.1 canary coupling,
    rho ~ 0.9): the trajectory probe must flag the leak and the at-chance
    test must reject; verdict FAIL."""
    env = _make_env("Blind", N_BASE)
    base = torch.arange(N_BASE)
    geom = _geom_bit(env, base)
    g = torch.Generator().manual_seed(99)
    flip = torch.rand(N_BASE, generator=g) < 0.1
    classes = torch.where(flip, 1 - geom, geom)
    plan = ce_mod.make_natural_plan(base, classes)
    rec = ro.run_lanes(env, GreedyPolicy(env), plan, mode="argmax")
    result, report = ce_mod.run_certificate(
        rec, plan, env._bank, "RoleBinding", seed=0)

    cl = result["clauses"]
    assert cl["scorer_consistency"]["passed"]
    # the probe (the audit's primary instrument) fires on the trajectory
    assert cl["probe"]["trajectory"]["leak"], cl["probe"]["trajectory"]
    assert cl["probe"]["trajectory"]["auc"] > 0.7
    # the behavioural at-chance test rejects too (accuracy ~ 0.9)
    at = cl["at_chance"]
    assert at["accuracy"] > 0.7
    assert not at["passed"]
    assert at["binom_p_two_sided"] < 0.01
    assert result["verdict"] == "FAIL", result["verdict"]
    assert any("leak probe" in r for r in result["fail_reasons"])
    assert "VERDICT: FAIL" in report


def test_positive_control_fails_certificate():
    """A language-sensitive policy (SymbolPO arm, instruction-weighted
    logits) audited as if blind: the paired-lane trajectory-equality
    machine check catches the divergence and the certificate FAILS."""
    env = _make_env("SymbolPO", 2 * N_BASE)
    base = torch.arange(N_BASE)
    plan = ro.make_paired_plan(base, classes_a=_balanced_classes(N_BASE))
    policy = LinearPolicy(L.OBS_DIM, env.cfg.possible_agents, seed=123,
                          lang_weight=50.0)
    rec = ro.run_lanes(env, policy, plan, mode="argmax")
    result, report = ce_mod.run_certificate(
        rec, plan, env._bank, "RoleBinding", seed=0)

    mc = result["clauses"]["paired_lane_identity"]
    assert mc["checked"]
    assert mc["passed"] is False, \
        "machine check did not catch a language-sensitive policy: the " \
        "certificate cannot fail and is vacuous"
    assert result["verdict"] == "FAIL", result["verdict"]
    assert any("machine check" in r for r in result["fail_reasons"])
    assert "VERDICT: FAIL" in report


def test_certificate_without_plan_clusters_by_scenario():
    """plan=None (plain rollout mode): episodes cluster by scenario_id and
    the certificate still runs end to end."""
    env = _make_env("Blind", N_BASE)
    base = torch.arange(N_BASE)
    classes = _balanced_classes(N_BASE)
    plan = ce_mod.make_natural_plan(base, classes)
    rec = ro.run_lanes(env, GreedyPolicy(env), plan, mode="argmax")
    result, _ = ce_mod.run_certificate(
        rec, None, env._bank, "RoleBinding", seed=0)
    assert result["clauses"]["paired_lane_identity"]["checked"] is False
    assert result["clauses"]["scorer_consistency"]["passed"]
    assert result["verdict"] in ("PASS", "UNINFORMATIVE", "FAIL")
    # blind policy + independent balanced labels: no leak claim
    assert not result["clauses"]["probe"]["trajectory"]["leak"]


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
