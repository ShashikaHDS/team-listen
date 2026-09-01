"""Instruction-leakage audit certificate (M1_SPEC 2.4 / 4.3 / 4.4).

The decision procedure for the (renamed) "language-necessity certificate":
given rollout records of an arm AUDITED AS BLIND over a scenario bank
(``harness/rollout.py::run_lanes`` output), emit a structured verdict dict
plus a plain-text report.  Everything is recomputed from the record and the
bank -- never trusted from the env's own bookkeeping (the scorer itself is
cross-checked).

What is decided here (spec 4.4 clause 1, the leakage audit, plus the probe
positive control):

1(a) **Paired-lane machine check** -- when the record carries the paired
     within-scenario manifest (factual + counterfactual lanes, argmax
     mode), the full recorded trajectory tensors of the two lanes must be
     ``torch.equal`` (float margins diagnostic: CPU atol per
     docs/DECISIONS.md 2026-09-01).  Any deviation is direct proof of an
     observation-channel leak, with zero sampling noise.
1(b) **Supervised leak probe** (``harness/probe.py``, the PRIMARY
     instrument): held-out AUC 95% CI must contain 0.5 on every supplied
     feature set -- the trajectory features are always built from the
     record; t=0 observation / full-state matrices are passed by the
     caller when available.
1(d) **At-chance TOST** -- assignment accuracy against the bank's
     ground-truth role assignment, TOST at alpha = 0.05 (<=> the 90%
     episode-clustered hierarchical-bootstrap CI inside [0.45, 0.55]).
     The exact binomial test and Wilson / Clopper-Pearson intervals are
     computed as DISPLAY (spec 4.4: with a frozen manifest and argmax
     actions a per-seed accuracy is an exact finite-population count).
1(e) **Non-degeneracy** -- the realised-assignment marginal
     P(A = r0->left) (precedence: P(r0 first)) must lie in [0.35, 0.65];
     outside it the audit is UNINFORMATIVE, not PASS (a constant-
     assignment oracle scores exactly 0.5 while being blind to any leak).
**Positive control** -- ``probe.planted_leak_control`` plants a label-
     carrying feature and must detect it; a probe that cannot find a
     planted leak proves nothing by finding nothing (UNINFORMATIVE).

Also computed and reported (never gating the leakage verdict): competence
E[C] with its clustered CI against the spec 4.4 clause-3 thresholds
(clause 3 gates the GO/NO-GO on the trained policy's quality, and clauses
2/4 -- SymbolPO ceiling, the rho-canary sweep -- need OTHER arms' runs;
this module's paired-delta hook is ``stats.paired_delta_bootstrap_ci``).

Verdict semantics: **FAIL** on any leak evidence (machine check, TOST,
probe) or a scorer inconsistency; **UNINFORMATIVE** when the audit cannot
speak (degenerate assignment marginal, failed positive control, probe not
runnable, nothing scored); **PASS** otherwise.  FAIL takes precedence over
UNINFORMATIVE.

Pre-registration: the constants above are the spec 4.4
``config/preregistration.yaml`` values, embedded as
``DEFAULT_PREREGISTRATION``.  When ``config/preregistration.yaml`` exists
it is read (flat ``key: value`` YAML subset -- no pyyaml on this box) and
its keys override the defaults; the resolved dict is echoed into the
verdict so every number in the report is traceable to its rule.

Pure numpy/torch; imports ``harness.rollout`` for the lane vocabulary and
the machine check; NO Isaac imports.
"""

import math
import os
import sys

import numpy as np
import torch

try:
    from harness import probe as probe_mod
    from harness import rollout as ro
    from harness import stats
except ImportError:  # standalone import: put the repo root on sys.path
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from harness import probe as probe_mod
    from harness import rollout as ro
    from harness import stats


# ---------------------------------------------------------------------------
# Pre-registration (spec 4.4 block, embedded; file overrides when present)
# ---------------------------------------------------------------------------

DEFAULT_PREREGISTRATION = {
    "equivalence_test": "TOST",
    "alpha": 0.05,                      # <=> 90% CI inside [0.45, 0.55]
    "equivalence_margin_delta": 0.05,
    "chance_level": 0.5,
    "clustering_unit": "base_scenario",
    "primary_interval": "hierarchical_bootstrap_scenarios_x_seeds",
    "bootstrap_replicates": 10000,
    "bootstrap_method": "percentile",   # not BCa, per rliable few-run guidance
    "primary_outcome": "itt_three_valued",
    # clause 1(e) non-degeneracy band for the realised-assignment marginal
    "assignment_marginal_lo": 0.35,
    "assignment_marginal_hi": 0.65,
    # clause 1(b): probe held-out AUC CI confidence
    "probe_confidence": 0.95,
    "probe_folds": 5,
    "probe_ridge": 0.1,
    "probe_bootstrap_replicates": 2000,
    # clause 3 (reported, not gating the leakage verdict)
    "competence_min": 0.90,
    "competence_ci_lo_min": 0.85,
}

PREREGISTRATION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "preregistration.yaml")


def _parse_scalar(text):
    text = text.strip()
    if not text:
        return ""
    if text[0] in "\"'" and text[-1:] == text[0]:
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def load_preregistration(path=None):
    """DEFAULT_PREREGISTRATION, overridden by the flat ``key: value`` lines
    of ``config/preregistration.yaml`` when that file exists (a pyyaml-free
    subset reader: comments and nested blocks are ignored)."""
    out = dict(DEFAULT_PREREGISTRATION)
    path = PREREGISTRATION_PATH if path is None else path
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].rstrip()
                if not line or line != line.lstrip() or ":" not in line:
                    continue                     # skip nested/blank lines
                key, _, value = line.partition(":")
                value = value.strip()
                if value:
                    out[key.strip()] = _parse_scalar(value)
        out["preregistration_source"] = path
    else:
        out["preregistration_source"] = "embedded_defaults"
    return out


# ---------------------------------------------------------------------------
# Natural (unpaired) manifest plan -- the spec 4.3 SECONDARY manifest as a
# single-lane LanePlan (one instruction per scenario, drawn at build time)
# ---------------------------------------------------------------------------

def make_natural_plan(base_sids, classes, rows=None, stream=0):
    """One-lane plan: each base scenario under ONE instruction class (the
    spec 4.3 secondary "natural" manifest, on which a training-distribution
    leak can express itself -- the paired design necessarily neutralises
    it).  ``classes`` must be supplied by the manifest, never defaulted."""
    base_sids = base_sids.reshape(-1).long()
    K = base_sids.numel()
    classes = classes.reshape(-1).long()
    assert classes.numel() == K
    return ro.LanePlan(
        lanes=(ro.LANE_FACTUAL,),
        n_base=K,
        scenarios=base_sids,
        streams=torch.full((K,), int(stream), dtype=torch.long),
        instr_classes=classes,
        instr_rows=None if rows is None else rows.reshape(-1).long(),
        blank=torch.zeros((K,), dtype=torch.bool),
        spawn_alt=torch.zeros((K,), dtype=torch.bool),
    ).validate()


# ---------------------------------------------------------------------------
# Ground-truth outcome recomputation from the bank (never trusted from env)
# ---------------------------------------------------------------------------

def realised_assignment(record, bank, variant):
    """(E,) long realised assignment A: 0 = class-0-compatible
    (RoleBinding: robot_0 latched the LEFT station, smaller column;
    Precedence: robot_0 docked first), 1 = the class-1-compatible
    assignment, -1 = undefined (episode incomplete, or a structural
    impossibility such as a precedence tie -- which scores Y = 0 for both
    classes, exactly as the env does, spec 3.1/3.3)."""
    completed = record.completed.bool().cpu()
    E = completed.numel()
    A = torch.full((E,), -1, dtype=torch.long)
    if variant == "Precedence":
        dt = (record.latch_time[:, 1].long()
              - record.latch_time[:, 0].long()).cpu()
        A = torch.where(dt > 0, torch.zeros_like(A),
                        torch.where(dt < 0, torch.ones_like(A), A))
    else:
        assert variant == "RoleBinding", variant
        sid = record.scenario_id.long().cpu()
        target = bank.target[sid].cpu()                     # (E, MT, 2)
        valid = bank.target_valid[sid].cpu()
        col = target[..., 1].float()
        left = torch.where(valid, col,
                           torch.full_like(col, float("inf"))).argmin(dim=1)
        r0_slot = record.latch_slot[:, 0].long().cpu()
        A = torch.where(r0_slot == left, torch.zeros_like(A),
                        torch.ones_like(A))
    return torch.where(completed, A, torch.full_like(A, -1))


def recompute_outcomes(record, bank, variant):
    """(Y, C, A) recomputed against the bank's ground truth; Y is scored
    only where C (spec 2.3/3.3)."""
    C = record.completed.bool().cpu()
    A = realised_assignment(record, bank, variant)
    Y = (A == record.instr_class.long().cpu()) & C
    return Y, C, A


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------

def run_certificate(record, plan, bank, variant, features=None, prereg=None,
                    seed_ids=None, seed=0):
    """Apply the leakage-audit decision procedure to one rollout record.

    Args:
        record:   ``rollout.RolloutRecord`` from ``run_lanes`` (the arm
                  under audit, evaluated in argmax mode for the machine
                  check to apply).
        plan:     the ``LanePlan`` the record was rolled under (paired,
                  five-lane, or a ``make_natural_plan`` single lane).
                  None treats every episode as its own cluster keyed by
                  ``scenario_id``.
        bank:     scenario-bank namespace (``target``, ``target_valid``
                  used for the RoleBinding ground truth).
        variant:  "RoleBinding" or "Precedence".
        features: optional dict name -> (E, D) matrix aligned with the
                  record's env axis (e.g. ``{"obs_t0": ..., "state_t0":
                  ...}``, spec 2.4(a)/(b)).  Trajectory features (spec
                  2.4(c)) are always built from ``record.positions``.
        prereg:   pre-registration dict (default: ``load_preregistration()``).
        seed_ids: optional per-episode training-seed ids for a multi-seed
                  record (activates the seeds level of the hierarchical
                  bootstrap, spec 4.4).
        seed:     RNG seed for bootstrap/probe fold draws.

    Returns:
        (verdict_dict, report_text)
    """
    prereg = load_preregistration() if prereg is None else dict(prereg)
    alpha = float(prereg["alpha"])
    delta = float(prereg["equivalence_margin_delta"])
    chance = float(prereg["chance_level"])
    n_boot = int(prereg["bootstrap_replicates"])

    clauses = {}
    fail_reasons = []
    uninformative_reasons = []

    E = record.scenario_id.numel()
    if plan is not None:
        plan.validate()
        assert plan.n_envs == E, (plan.n_envs, E)
        clusters = plan.group_of_env().cpu().numpy()
    else:
        clusters = record.scenario_id.long().cpu().numpy()

    # -- clause 1(a): paired-lane trajectory-equality machine check --------
    has_pair = (plan is not None
                and ro.LANE_FACTUAL in plan.lanes
                and ro.LANE_COUNTERFACTUAL in plan.lanes)
    mc = {"checked": bool(has_pair and record.mode == "argmax"),
          "passed": None, "detail": ""}
    if mc["checked"]:
        try:
            ro.assert_paired_lane_identity(record, plan, ro.LANE_FACTUAL,
                                           ro.LANE_COUNTERFACTUAL)
            mc["passed"] = True
        except AssertionError as exc:
            mc["passed"] = False
            mc["detail"] = str(exc)
            fail_reasons.append(
                "paired-lane machine check: factual/counterfactual "
                "trajectories differ (observation-channel leak, spec 4.3)")
    clauses["paired_lane_identity"] = mc

    # -- ground truth recomputation + scorer cross-check -------------------
    Y, C, A = recompute_outcomes(record, bank, variant)
    scorer_ok = bool(torch.equal(Y, record.correct.bool().cpu()))
    clauses["scorer_consistency"] = {"passed": scorer_ok}
    if not scorer_ok:
        fail_reasons.append(
            "scorer inconsistency: bank-recomputed Y disagrees with the "
            "recorded Y (outcome scorer or bank ground truth is wrong)")

    y_np = Y.numpy().astype(np.float64)
    c_np = C.numpy().astype(np.float64)
    n_scored = int(c_np.sum())
    k_correct = int(y_np.sum())

    # -- clause 1(d): at-chance TOST on assignment accuracy ----------------
    at = {"n_episodes": int(E), "n_scored": n_scored,
          "k_correct": k_correct,
          "accuracy": (k_correct / n_scored) if n_scored else float("nan")}
    if n_scored == 0:
        at.update({"tost": None, "passed": None})
        uninformative_reasons.append(
            "no completed episode: assignment accuracy is unscoreable")
    else:
        tost = stats.tost_equivalence(
            y_np, den=c_np, cluster_ids=clusters, seed_ids=seed_ids,
            center=chance, delta=delta, alpha=alpha, n_boot=n_boot,
            seed=seed)
        at["tost"] = tost
        at["passed"] = bool(tost["equivalent"])
        if not at["passed"]:
            fail_reasons.append(
                "at-chance TOST: %.0f%% CI (%.4f, %.4f) not inside "
                "[%.2f, %.2f]" % (100 * (1 - 2 * alpha), tost["lo"],
                                  tost["hi"], chance - delta, chance + delta))
        # display-only exact/binomial numbers (spec 4.4)
        at["binom_p_two_sided"] = stats.binom_test(k_correct, n_scored,
                                                   chance)
        at["wilson_95"] = stats.wilson_interval(k_correct, n_scored, 0.95)
        at["clopper_pearson_95"] = stats.clopper_pearson_interval(
            k_correct, n_scored, 0.95)
        at["icc_deff"] = stats.icc_deff(y_np[c_np > 0],
                                        clusters[c_np > 0])
    clauses["at_chance"] = at

    # -- clause 1(e): non-degeneracy of the realised assignment ------------
    a_np = A.numpy()
    scored_mask = a_np >= 0
    lo_band = float(prereg["assignment_marginal_lo"])
    hi_band = float(prereg["assignment_marginal_hi"])
    if scored_mask.any():
        marginal = float((a_np[scored_mask] == 0).mean())
        in_range = bool(lo_band <= marginal <= hi_band)
        if not in_range:
            uninformative_reasons.append(
                "assignment marginal P(A=class0)=%.3f outside [%.2f, %.2f]:"
                " a (near-)constant-assignment policy scores 0.5 while "
                "blind to any leak (spec 4.4 clause 1(e))"
                % (marginal, lo_band, hi_band))
    else:
        marginal, in_range = float("nan"), None
    clauses["non_degeneracy"] = {"marginal_class0": marginal,
                                 "band": (lo_band, hi_band),
                                 "in_range": in_range}

    # -- clause 1(b): supervised leak probe (PRIMARY instrument) -----------
    if plan is not None and ro.LANE_FACTUAL in plan.lanes:
        sl = plan.lane_slice(ro.LANE_FACTUAL)
        probe_rows = np.arange(sl.start, sl.stop)
    else:
        probe_rows = np.arange(E)
    labels = record.instr_class.long().cpu().numpy()[probe_rows]
    probe_clusters = clusters[probe_rows]
    feature_sets = {"trajectory": probe_mod.trajectory_features(
        record.positions, rows=probe_rows)}
    for name, mat in (features or {}).items():
        feature_sets[name] = probe_mod.matrix_features(mat, rows=probe_rows)

    probe_conf = float(prereg["probe_confidence"])
    probe_kw = dict(n_folds=int(prereg["probe_folds"]),
                    ridge=float(prereg["probe_ridge"]),
                    n_boot=int(prereg["probe_bootstrap_replicates"]),
                    conf=probe_conf, seed=seed)
    probes = {}
    any_leak = False
    any_unrunnable = False
    for name, X in sorted(feature_sets.items()):
        res = probe_mod.leak_probe(X, labels, probe_clusters, **probe_kw)
        probes[name] = res
        if res["leak"]:
            any_leak = True
            fail_reasons.append(
                "leak probe on %r: held-out AUC %.3f, %.0f%% CI (%.3f, "
                "%.3f) excludes %.1f (spec 2.4: the observation leaks)"
                % (name, res["auc"], 100 * probe_conf, res["lo"],
                   res["hi"], chance))
        if not res["ok"]:
            any_unrunnable = True
            uninformative_reasons.append(
                "leak probe on %r not runnable: %s" % (name, res["note"]))
    clauses["probe"] = probes

    # -- positive control: planted leak must be detected -------------------
    pc = probe_mod.planted_leak_control(
        feature_sets["trajectory"], labels, probe_clusters, **probe_kw)
    clauses["positive_control"] = pc
    if not pc["detected"]:
        uninformative_reasons.append(
            "positive control failed: the probe did not detect a planted "
            "label-carrying feature (AUC %.3f, CI (%.3f, %.3f)); a null "
            "probe result at this n proves nothing" % (pc["auc"], pc["lo"],
                                                       pc["hi"]))

    # -- competence (spec 4.4 clause 3): reported, never gating ------------
    comp_ci = stats.clustered_bootstrap_ci(
        c_np, cluster_ids=clusters, n_boot=n_boot, conf=0.95, seed=seed)
    clauses["competence"] = {
        "rate": comp_ci["point"], "ci95": (comp_ci["lo"], comp_ci["hi"]),
        "meets_clause3": bool(
            comp_ci["point"] >= float(prereg["competence_min"])
            and comp_ci["lo"] > float(prereg["competence_ci_lo_min"])),
        "gating": False,
    }

    # -- verdict ------------------------------------------------------------
    if fail_reasons:
        verdict = "FAIL"
    elif uninformative_reasons:
        verdict = "UNINFORMATIVE"
    else:
        verdict = "PASS"

    result = {
        "verdict": verdict,
        "variant": variant,
        "fail_reasons": fail_reasons,
        "uninformative_reasons": uninformative_reasons,
        "clauses": clauses,
        "preregistration": prereg,
        "n_episodes": int(E),
        "mode": record.mode,
    }
    return result, format_report(result)


# ---------------------------------------------------------------------------
# Plain-text report
# ---------------------------------------------------------------------------

def _fmt_ci(pair):
    return "(%.4f, %.4f)" % (pair[0], pair[1])


def format_report(result):
    c = result["clauses"]
    lines = []
    push = lines.append
    push("=" * 72)
    push("INSTRUCTION-LEAKAGE AUDIT CERTIFICATE (M1_SPEC 4.4 clause 1)")
    push("=" * 72)
    push("variant        %s" % result["variant"])
    push("episodes       %d   (mode %s)" % (result["n_episodes"],
                                            result["mode"]))
    push("prereg source  %s" % result["preregistration"].get(
        "preregistration_source", "embedded_defaults"))
    push("")

    mc = c["paired_lane_identity"]
    if mc["checked"]:
        push("[1a] paired-lane machine check     %s"
             % ("PASS (bit-identical)" if mc["passed"] else "FAIL"))
        if mc["detail"]:
            push("     %s" % mc["detail"])
    else:
        push("[1a] paired-lane machine check     not applicable "
             "(no paired argmax lanes)")

    sc = c["scorer_consistency"]
    push("[--] scorer cross-check            %s"
         % ("PASS (bank-recomputed Y == recorded Y)" if sc["passed"]
            else "FAIL"))

    at = c["at_chance"]
    push("[1d] at-chance assignment accuracy")
    push("     scored n = %d / %d episodes, correct k = %d, acc = %s"
         % (at["n_scored"], at["n_episodes"], at["k_correct"],
            ("%.4f" % at["accuracy"]) if at["n_scored"] else "n/a"))
    if at.get("tost"):
        t = at["tost"]
        push("     TOST: %.0f%% %s CI %s inside [%.2f, %.2f] -> %s"
             % (100 * t["conf"], t["level"], _fmt_ci((t["lo"], t["hi"])),
                t["band"][0], t["band"][1],
                "PASS" if at["passed"] else "FAIL"))
        push("     display: exact binomial p = %.4g | Wilson95 %s | CP95 %s"
             % (at["binom_p_two_sided"], _fmt_ci(at["wilson_95"]),
                _fmt_ci(at["clopper_pearson_95"])))
        icc = at["icc_deff"]
        push("     clustering: ICC = %.3f, DEFF = %.2f over %d clusters"
             % (icc["icc"], icc["deff"], icc["n_clusters"]))

    nd = c["non_degeneracy"]
    push("[1e] non-degeneracy: P(A = class0) = %s in [%.2f, %.2f] -> %s"
         % (("%.4f" % nd["marginal_class0"])
            if not math.isnan(nd["marginal_class0"]) else "n/a",
            nd["band"][0], nd["band"][1],
            {True: "OK", False: "DEGENERATE", None: "n/a"}[nd["in_range"]]))

    push("[1b] supervised leak probe (primary instrument)")
    for name, res in sorted(c["probe"].items()):
        if res["ok"]:
            push("     %-12s AUC %.3f  %.0f%% CI %s  -> %s"
                 % (name, res["auc"], 100 * res["conf"],
                    _fmt_ci((res["lo"], res["hi"])),
                    "LEAK" if res["leak"] else "chance in CI"))
        else:
            push("     %-12s NOT RUNNABLE (%s)" % (name, res["note"]))

    pc = c["positive_control"]
    push("[pc] planted-leak positive control AUC %.3f CI %s -> %s"
         % (pc["auc"], _fmt_ci((pc["lo"], pc["hi"])),
            "DETECTED (audit has teeth)" if pc["detected"]
            else "NOT DETECTED (audit uninformative)"))

    comp = c["competence"]
    push("[3 ] competence (reported, non-gating): E[C] = %.4f, 95%% CI %s,"
         % (comp["rate"], _fmt_ci(comp["ci95"])))
    push("     clause-3 thresholds met: %s" % comp["meets_clause3"])

    push("")
    push("VERDICT: %s" % result["verdict"])
    for reason in result["fail_reasons"]:
        push("  FAIL: %s" % reason)
    for reason in result["uninformative_reasons"]:
        push("  UNINFORMATIVE: %s" % reason)
    push("=" * 72)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PREREGISTRATION", "PREREGISTRATION_PATH",
    "load_preregistration", "make_natural_plan", "realised_assignment",
    "recompute_outcomes", "run_certificate", "format_report",
]
