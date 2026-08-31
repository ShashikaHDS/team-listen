"""Paired-lane identity tests on the CPU stand-in env (M1_SPEC 4.3 / 6.3).

M1_SPEC coverage (spec section 7 test list entry ``test_paired_lane_identity``
"blind-arm lanes byte-identical across the full trajectory", plus the
"resolvable only on the 5090" empirical-determinism question, answered here
for the CPU stand-in):

* **Lane plans (spec 6.3)** -- the five-lane L0..L4 layout and the two-lane
  paired-manifest layout are pure-torch and carry the right per-lane
  scenario / slip-stream / instruction / blank / spawn assignments.
* **Identical init (spec 6.3)** -- ``torch.equal`` on the non-language
  observation slices at t = 0 for every group; the spawn lane is excluded
  by design; the assertion has TEETH (a hand-planted non-language
  perturbation is caught, a language-slice difference is not).
* **Paired-lane identity (spec 4.3)** -- on a blind arm, factual and
  counterfactual lanes produce bit-identical trajectories, and exactly one
  lane of each competent pair scores Y = 1 (the paired-manifest algebraic
  identity ``E[Y | C=1] == 0.5`` exactly).  On a language-sensitive arm
  (SymbolPO) the lanes diverge and ``assert_paired_lane_identity`` catches
  it -- the machine check can actually fail.
* **Cross-process determinism** -- same bank row + same stored slip stream
  reproduce BIT-IDENTICAL trajectories across two separate python
  processes (subprocess-spawned), which is what makes the paired-lane
  ``torch.equal`` assertion a meaningful audit instrument rather than a
  same-process tautology.  Different slip streams (the spec 1.10 seed
  intervention, CSI's ``D_seed`` referent) change the trajectory.
* **Action modes (spec 4.3)** -- argmax is deterministic with lowest-index
  tie-break and recorded top-2 margins; stochastic mode is reproducible
  from an explicit ``torch.Generator``.

The smoke bank (32 scenarios) is built by ``scripts/build_scenario_bank.py``
into a temp dir -- never into data/.

pytest-compatible; also standalone: ``python tests/test_paired_lane_identity.py``
(the ``--child`` argv form is the cross-process worker, not a test).
"""

import atexit
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import cpu_env as ce                    # noqa: E402
from harness import rollout as ro                    # noqa: E402
from tasks.team_listen import grid_core              # noqa: E402
from tasks.team_listen import obs_layout as L        # noqa: E402

SMOKE_K = 32
SMOKE_SEED = 20260831
N_BASE = 8

_TMP = None
_BANK_PATHS = {}


def _tmpdir():
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="team_listen_paired_lane_")
        atexit.register(shutil.rmtree, _TMP, True)
    return _TMP


def _bank_path(variant="RoleBinding"):
    """Build (once) and return the smoke bank artifact for ``variant``."""
    if variant not in _BANK_PATHS:
        spec = importlib.util.spec_from_file_location(
            "team_listen_bsb_paired_lane",
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


# ---------------------------------------------------------------------------
# Deterministic test policies
# ---------------------------------------------------------------------------

class LinearPolicy:
    """Fixed random-projection logits: a pure function of the OBSERVATION,
    so any obs difference between lanes propagates into actions -- the
    paired-lane assertion has teeth against this policy.  ``lang_weight``
    scales the language-slice rows so a language-sensitive policy is one
    parameter away (the SymbolPO negative control)."""

    def __init__(self, obs_dim, agents, seed=123, lang_weight=None):
        g = torch.Generator().manual_seed(seed)
        self.w = {}
        for name in agents:
            w = torch.randn(obs_dim, grid_core.N_ACTIONS, generator=g)
            if lang_weight is not None:
                assert obs_dim == L.OBS_DIM, \
                    "lang_weight targets the partial-obs LANG_SLICE"
                w[L.LANG_SLICE] *= float(lang_weight)
            self.w[name] = w

    def __call__(self, obs):
        return {name: obs[name] @ self.w[name] for name in obs}


class GreedyPolicy:
    """Scripted state-reading controller: robot i descends
    ``dist_field[slot=i]`` (obstacle / unreachable neighbours cost +inf).
    Reaches both alcoves in ~15 steps on the smoke bank, so C = 1 pairs
    exist and the terminal-field bookkeeping (latch, Y) is exercised."""

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
# Pure-torch lane logic
# ---------------------------------------------------------------------------

def test_lane_plan_layout():
    base = torch.arange(N_BASE)
    plan = ro.make_five_lane_plan(base)
    assert plan.n_lanes == 5 and plan.n_envs == 5 * N_BASE
    assert plan.lanes == (ro.LANE_FACTUAL, ro.LANE_COUNTERFACTUAL,
                          ro.LANE_SEED, ro.LANE_SPAWN, ro.LANE_BLANK)
    # scenario ids tiled per lane (spec 6.3: same physics row in every lane)
    v = plan.view(plan.scenarios)
    assert bool((v == base).all())
    # slip streams: only the seed lane runs the alternate stream
    s = plan.view(plan.streams)
    assert bool((s[plan.lane_pos(ro.LANE_SEED)] == 1).all())
    for lane in (ro.LANE_FACTUAL, ro.LANE_COUNTERFACTUAL, ro.LANE_SPAWN,
                 ro.LANE_BLANK):
        assert bool((s[plan.lane_pos(lane)] == 0).all())
    # blank / spawn masks hit exactly their lanes
    assert bool(plan.view(plan.blank)[plan.lane_pos(ro.LANE_BLANK)].all())
    assert int(plan.blank.long().sum()) == N_BASE
    assert bool(plan.view(plan.spawn_alt)[plan.lane_pos(ro.LANE_SPAWN)].all())
    assert int(plan.spawn_alt.long().sum()) == N_BASE
    # instruction classes: counterfactual lane is the complement
    c = plan.view(plan.instr_classes)
    assert bool((c[plan.lane_pos(ro.LANE_COUNTERFACTUAL)]
                 == 1 - c[plan.lane_pos(ro.LANE_FACTUAL)]).all())
    for lane in (ro.LANE_SEED, ro.LANE_SPAWN, ro.LANE_BLANK):
        assert bool((c[plan.lane_pos(lane)]
                     == c[plan.lane_pos(ro.LANE_FACTUAL)]).all())
    # lane_slice / lane_of_env / group_of_env agree
    sl = plan.lane_slice(ro.LANE_SEED)
    assert (sl.start, sl.stop) == (2 * N_BASE, 3 * N_BASE)
    assert bool((plan.lane_of_env()[sl] == ro.LANE_SEED).all())
    assert bool((plan.group_of_env()[sl] == base).all())

    paired = ro.make_paired_plan(base)
    assert paired.n_lanes == 2 and paired.n_envs == 2 * N_BASE
    assert not bool(paired.blank.any()) and not bool(paired.spawn_alt.any())
    pc = paired.view(paired.instr_classes)
    assert bool((pc[0] == 0).all()) and bool((pc[1] == 1).all())


def test_select_actions_and_margins():
    logits = torch.tensor([[0.0, 3.0, 3.0, -1.0, 2.0],
                           [5.0, 1.0, 0.0, 0.0, 4.5]])
    a = ro.select_actions(logits, "argmax")
    # torch.argmax tie-break: LOWEST index wins (spec 4.3 [FIXED]) -> 1, not 2
    assert a.tolist() == [1, 0]
    m = ro.top2_margin(logits)
    assert torch.allclose(m, torch.tensor([0.0, 0.5]))
    # stochastic: reproducible from an explicit generator
    big = torch.randn(64, grid_core.N_ACTIONS)
    s1 = ro.select_actions(big, "stochastic",
                           torch.Generator().manual_seed(11))
    s2 = ro.select_actions(big, "stochastic",
                           torch.Generator().manual_seed(11))
    assert torch.equal(s1, s2)
    assert bool((s1 >= 0).all()) and bool((s1 < grid_core.N_ACTIONS).all())
    try:
        ro.select_actions(logits, "mean_actions")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown action mode accepted")


def test_nonlang_mask_widths():
    m = ro.nonlang_mask(L.OBS_DIM)
    assert int((~m).long().sum()) == L.LANG_DIM
    assert not bool(m[L.LANG_SLICE].any()) and bool(m[:L.LANG_SLICE.start].all())
    for width in (L.STATE_DIM, L.FULL_STATE_OBS_DIM):
        m = ro.nonlang_mask(width)
        assert int((~m).long().sum()) == L.MAX_AGENTS * L.LANG_DIM
        assert not bool(m[L.STATE_LANG_SLICE].any())
    try:
        ro.nonlang_mask(123)
    except ValueError:
        pass
    else:
        raise AssertionError("bogus obs width accepted")


def test_identical_init_assertion_has_teeth():
    plan = ro.make_paired_plan(torch.arange(4))
    base = torch.randn(4, L.OBS_DIM)
    obs = {"robot_0": torch.cat([base, base.clone()], dim=0)}
    ro.assert_identical_init(obs, plan)                  # identical: passes
    # language-slice difference between lanes is LEGAL (that IS the
    # counterfactual intervention) and must not trip the init assertion
    obs["robot_0"][4:, L.LANG_SLICE] += 1.0
    ro.assert_identical_init(obs, plan)
    # a single non-language deviation must be caught
    obs["robot_0"][5, 0] += 1e-3
    try:
        ro.assert_identical_init(obs, plan)
    except AssertionError as exc:
        assert "identical-init" in str(exc)
    else:
        raise AssertionError("non-language init deviation not caught "
                             "(spec 6.3 assertion is vacuous)")


# ---------------------------------------------------------------------------
# Paired lanes on the CPU stand-in
# ---------------------------------------------------------------------------

def test_blind_paired_lanes_bit_identical():
    env = _make_env("Blind", 2 * N_BASE)
    plan = ro.make_paired_plan(torch.arange(N_BASE))
    # run_lanes auto-runs the spec 4.3 machine check for blind arms in
    # argmax mode; a raise here is a failure of the audit invariant.
    rec = ro.run_lanes(env, GreedyPolicy(env), plan, mode="argmax")
    # explicit lane-level comparisons on the full recorded tensors
    for name in ro.RolloutRecord.IDENTITY_FIELDS:
        t = getattr(rec, name)
        if name in ("positions", "intended", "executed", "margins", "active"):
            t = t.transpose(0, 1).contiguous()
        v = plan.view(t)
        assert torch.equal(v[0], v[1]), name
    # Blind runs the expected +5.0 bonus (spec 1.12 [FIXED]), so even the
    # rewards are lane-identical for this arm
    rv = plan.view(rec.rewards.transpose(0, 1).contiguous())
    assert torch.equal(rv[0], rv[1])
    # paired-manifest identity (spec 4.3): identical trajectories under
    # complementary instructions => exactly one lane of each competent
    # pair scores Y = 1
    y, c = plan.view(rec.correct), plan.view(rec.completed)
    assert bool(c.any()), "no completed pair: the greedy controller or " \
                          "the smoke bank topology regressed"
    assert torch.equal(c[0], c[1])
    assert torch.equal(y[0] ^ y[1], c[0])
    # instruction classes really were complementary across the lanes
    ic = plan.view(rec.instr_class)
    assert bool((ic[1] == 1 - ic[0]).all())
    # forcing hooks released
    assert env._forced_scenarios is None and env._forced_slip_stream is None


def test_language_sensitive_pair_diverges_and_is_caught():
    env = _make_env("SymbolPO", 2 * N_BASE)
    plan = ro.make_paired_plan(torch.arange(N_BASE))
    policy = LinearPolicy(L.OBS_DIM, env.cfg.possible_agents, seed=123,
                          lang_weight=50.0)
    # auto assertion must NOT fire (lang_gain != 0), so this returns
    rec = ro.run_lanes(env, policy, plan, mode="argmax")
    # ...but the same non-language observation prefix held at t=0
    # (assert_identical_init ran inside run_lanes), and the trajectories
    # must now diverge somewhere -- and the machine check must catch it
    try:
        ro.assert_paired_lane_identity(rec, plan)
    except AssertionError as exc:
        assert "paired-lane identity" in str(exc)
    else:
        raise AssertionError(
            "SymbolPO factual/counterfactual lanes were bit-identical: the "
            "paired-lane check cannot fail and is therefore vacuous")


def test_five_lane_semantics_blind():
    env = _make_env("Blind", 5 * N_BASE)
    plan = ro.make_five_lane_plan(torch.arange(N_BASE))
    policy = LinearPolicy(int(env.cfg.observation_spaces["robot_0"]),
                          env.cfg.possible_agents, seed=123)
    # auto-assert covers L1 and L4 against L0 for the blind arm
    rec = ro.run_lanes(env, policy, plan, mode="argmax")
    pos = plan.view(rec.positions.transpose(0, 1).contiguous())
    p_fact = plan.lane_pos(ro.LANE_FACTUAL)
    # L2 seed lane: slip stream 0 -> 1 with everything else frozen is a
    # genuine intervention (the spec 1.10 D_seed referent)
    assert not torch.equal(pos[plan.lane_pos(ro.LANE_SEED)], pos[p_fact]), \
        "slip stream 0 and 1 produced identical trajectories: the seed " \
        "denominator has no referent (spec 1.10 [FIXED])"
    # L3 spawn lane: spawn_alt actually moved someone
    assert not torch.equal(pos[plan.lane_pos(ro.LANE_SPAWN)], pos[p_fact])
    # recorded slip streams follow the plan
    ss = plan.view(rec.slip_stream.long())
    assert bool((ss[plan.lane_pos(ro.LANE_SEED)] == 1).all())
    assert bool((ss[p_fact] == 0).all())
    # blank lane carried a zero language slice at t=0 (D_blank semantics):
    # for a blind arm it is zero everywhere, so check via the env's table
    # on a symbol arm instead
    env_s = _make_env("SymbolPO", 5 * N_BASE)
    env_s.force_scenarios(plan.scenarios)
    env_s.force_slip_stream(plan.streams.to(torch.int8))
    env_s._reset_idx(None)
    ro.force_instructions(env_s, ro.default_instruction_rows(
        env_s, plan.instr_classes), plan.instr_classes)
    ro.blank_language(env_s, plan.blank)
    lv = plan.view(env_s.lang_vec.reshape(env_s.num_envs, -1))
    assert float(lv[plan.lane_pos(ro.LANE_BLANK)].abs().sum()) == 0.0
    assert float(lv[p_fact].abs().sum()) > 0.0
    env_s.force_scenarios(None)
    env_s.force_slip_stream(None)


def test_stochastic_mode_generator_reproducible():
    env = _make_env("Blind", 2 * N_BASE)
    plan = ro.make_paired_plan(torch.arange(N_BASE))
    policy = LinearPolicy(int(env.cfg.observation_spaces["robot_0"]),
                          env.cfg.possible_agents, seed=123)
    r1 = ro.run_lanes(env, policy, plan, mode="stochastic",
                      generator=torch.Generator().manual_seed(7))
    r2 = ro.run_lanes(env, policy, plan, mode="stochastic",
                      generator=torch.Generator().manual_seed(7))
    for name in ("positions", "intended", "executed", "rewards", "active"):
        assert torch.equal(getattr(r1, name), getattr(r2, name)), name
    assert torch.equal(r1.first_done, r2.first_done)
    r3 = ro.run_lanes(env, policy, plan, mode="stochastic",
                      generator=torch.Generator().manual_seed(8))
    assert not torch.equal(r1.intended, r3.intended), \
        "different sampling seeds produced identical stochastic rollouts"


def test_run_lanes_guards():
    # env surface guard: the Isaac path degrades with a clear message
    try:
        ro.resolve_env(object())
    except TypeError as exc:
        assert "5090" in str(exc) or "unwrapped" in str(exc)
    else:
        raise AssertionError("resolve_env accepted a non-env object")
    # plan size guard
    env = _make_env("Blind", 2 * N_BASE)
    plan = ro.make_paired_plan(torch.arange(N_BASE + 1))
    policy = GreedyPolicy(env)
    try:
        ro.run_lanes(env, policy, plan, mode="argmax")
    except ValueError as exc:
        assert "num_envs" in str(exc)
    else:
        raise AssertionError("plan/env size mismatch accepted")
    # lane-inconsistent scenarios are rejected by the plan itself
    bad = ro.make_paired_plan(torch.arange(N_BASE))
    bad.scenarios = torch.cat([torch.arange(N_BASE),
                               torch.arange(N_BASE) + 1])
    try:
        bad.validate()
    except AssertionError as exc:
        assert "scenario ids" in str(exc)
    else:
        raise AssertionError("lane-varying scenario ids accepted")


def test_episode_records_fields():
    env = _make_env("Blind", 2 * N_BASE)
    plan = ro.make_paired_plan(torch.arange(N_BASE))
    rec = ro.run_lanes(env, GreedyPolicy(env), plan, mode="argmax")
    er = ro.episode_records(rec, plan)
    E = 2 * N_BASE
    for key, value in er.items():
        assert value.numel() == E, key
    assert bool((er["lane"] == plan.lane_of_env()).all())
    assert bool((er["group"] == plan.group_of_env()).all())
    # ITT outcome coding (spec 2.3): 0 correct / 1 wrong / 2 incomplete
    o, c, y = er["O_itt"], er["C"].bool(), er["Y"].bool()
    assert bool(((o == 2) == ~c).all())
    assert bool(((o == 0) == (c & y)).all())
    assert bool(((o == 1) == (c & ~y)).all())
    # episode length T: completed episodes ended before truncation
    assert bool((er["T"][c] == rec.first_done[c].long() + 1).all())
    # latch fields round-trip
    assert bool((er["latch_time_robot_0"][c] >= 0).all())
    assert bool((er["latch_slot_robot_0"][c] >= 0).all())


# ---------------------------------------------------------------------------
# Cross-process determinism (the 5090 open question, answered on CPU):
# same bank row + same stored slip stream => bit-identical trajectories
# across two SEPARATE processes.
# ---------------------------------------------------------------------------

_CHILD_N_BASE = 6
_CHILD_POLICY_SEED = 123


def _digest(tensors):
    """Canonical SHA-256 over named tensors (sorted keys, raw bytes)."""
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key]
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _record_tensors(rec, prefix):
    out = {}
    for name in ("positions", "intended", "executed", "margins", "rewards",
                 "active", "first_done", "completed", "correct", "outcome",
                 "latch_time", "latch_slot", "scenario_id", "slip_stream",
                 "instr_class", "episode_return", "n_obstacle_collisions",
                 "n_robot_collisions"):
        out["%s/%s" % (prefix, name)] = getattr(rec, name)
    return out


def _child_main(bank_path, out_path):
    """Cross-process worker: rolls the SAME (bank rows, slip streams,
    instructions, policy) as its sibling process and saves every recorded
    tensor.  Determinism knobs: single-threaded BLAS (rules out reduction
    -order variation; the claim under test is the env+driver pipeline),
    seeded global RNG (hygiene only -- forced scenarios/instructions leave
    nothing trajectory-relevant to the RNG, spec 1.10)."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base = torch.arange(_CHILD_N_BASE)
    tensors = {}
    for stream in (0, 1):
        for policy_kind in ("linear", "greedy"):
            cfg = ce.make_cfg(arm="Blind", variant="RoleBinding",
                              bank_path=bank_path, debug_asserts=True)
            env = ce.CPUFleetEnv(cfg, num_envs=2 * _CHILD_N_BASE)
            plan = ro.make_paired_plan(base, stream=stream)
            if policy_kind == "linear":
                policy = LinearPolicy(
                    int(cfg.observation_spaces["robot_0"]),
                    cfg.possible_agents, seed=_CHILD_POLICY_SEED)
            else:
                policy = GreedyPolicy(env)
            rec = ro.run_lanes(env, policy, plan, mode="argmax")
            tensors.update(_record_tensors(
                rec, "s%d_%s" % (stream, policy_kind)))
    torch.save(tensors, out_path)
    print("DIGEST %s" % _digest(tensors))


def _load_tensors(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def test_cross_process_bit_identical():
    bank = _bank_path("RoleBinding")
    outs, digests = [], []
    for tag in ("a", "b"):
        out = os.path.join(_tmpdir(), "child_%s.pt" % tag)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", bank, out],
            capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT))
        assert proc.returncode == 0, (
            "child process failed (rc=%d)\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, proc.stdout, proc.stderr))
        digest_lines = [ln for ln in proc.stdout.splitlines()
                        if ln.startswith("DIGEST ")]
        assert digest_lines, "child printed no digest:\n%s" % proc.stdout
        digests.append(digest_lines[-1].split(" ", 1)[1])
        outs.append(_load_tensors(out))

    # the audit's primary machine check is only meaningful if this holds:
    # same bank row + same stored slip stream => bit-identical trajectories
    # across separate processes (spec 6.3 / the 5090 open-question list)
    a, b = outs
    assert sorted(a.keys()) == sorted(b.keys())
    for key in sorted(a):
        assert a[key].dtype == b[key].dtype, key
        assert torch.equal(a[key], b[key]), (
            "tensor %r differs across two separate processes: same bank "
            "row + same stored slip stream did NOT reproduce bit-identical "
            "trajectories (spec 1.10 frozen-physics guarantee broken)"
            % (key,))
    assert digests[0] == digests[1]
    assert _digest(a) == digests[0], "parent-side digest mismatch"

    # and the seed intervention is not a no-op: stream 0 vs stream 1 on the
    # same rows must differ somewhere (D_seed has a referent, spec 1.10)
    changed = any(
        not torch.equal(a["s0_%s/positions" % kind],
                        a["s1_%s/positions" % kind])
        for kind in ("linear", "greedy"))
    assert changed, "slip streams 0 and 1 gave identical trajectories on " \
                    "every scenario (spec 1.10 [FIXED: no physics seed])"


# ---------------------------------------------------------------------------
# Standalone runner (child-mode dispatch first)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        _child_main(sys.argv[2], sys.argv[3])
        sys.exit(0)

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
