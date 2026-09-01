# M1 first science runs — training campaign report (5090)

Per experiments-f1's work order, 2026-09-01. Config: MAPPO @ 9b6c927 YAMLs
(num_envs 8192, timesteps 18304 = 149,946,368 env steps/run), seeds 0–4,
banks RoleBinding 68d025a6… / Precedence 0838c7c9…, lang cache 985f159f….
Total campaign wall clock ≈ 89 min for 20 launched runs.

**HEADLINE: 3 of 20 runs completed. 16 died with an identical CUDA
device-side assert mid-training; 1 failed at boot. All 10 Lang runs died.**
Per the work order I finished the campaign and did not debug solo; the
characterization below is from logs/TB only. Mitigation is a joint decision.

## Run table

| Run | Outcome | Death step (of 18304) | Wall s | Log dir (logs/skrl/runs/) |
|---|---|---|---|---|
| RoleBinding_Blind_s0 | CRASH | 8687 (47%) | 215 | 2026-09-01_01-23-59_mappo_torch |
| RoleBinding_Blind_s1 | **COMPLETE** | — | 446 | 2026-09-01_01-27-33_mappo_torch |
| RoleBinding_Blind_s2 | **COMPLETE** | — | 447 | 2026-09-01_01-34-59_mappo_torch |
| RoleBinding_Blind_s3 | CRASH | 9311 (51%) | 232 | 2026-09-01_01-42-26_mappo_torch |
| RoleBinding_Blind_s4 | CRASH | 9279 (51%) | 229 | 2026-09-01_01-46-17_mappo_torch |
| RoleBinding_Lang_s0 | CRASH | 9839 (54%) | 225 | 2026-09-01_01-50-07_mappo_torch |
| RoleBinding_Lang_s1 | CRASH | 8719 (48%) | 199 | 2026-09-01_01-53-51_mappo_torch |
| RoleBinding_Lang_s2 | CRASH | 8767 (48%) | 201 | 2026-09-01_01-57-11_mappo_torch |
| RoleBinding_Lang_s3 | CRASH | 10447 (57%) | 238 | 2026-09-01_02-00-32_mappo_torch |
| RoleBinding_Lang_s4 | BOOT CRASH | 0 | 2 | (kit breakpad dump; see below) |
| Precedence_Blind_s0 | CRASH | 12383 (68%) | 286 | 2026-09-01_02-04-32_mappo_torch |
| Precedence_Blind_s1 | CRASH | 10895 (60%) | 261 | 2026-09-01_02-09-18_mappo_torch |
| Precedence_Blind_s2 | CRASH | 14687 (80%) | 352 | 2026-09-01_02-13-39_mappo_torch |
| Precedence_Blind_s3 | CRASH | 9871 (54%) | 239 | 2026-09-01_02-19-32_mappo_torch |
| Precedence_Blind_s4 | **COMPLETE** | — | 441 | 2026-09-01_02-23-30_mappo_torch |
| Precedence_Lang_s0 | CRASH | 10863 (59%) | 243 | 2026-09-01_02-30-50_mappo_torch |
| Precedence_Lang_s1 | CRASH | 11151 (61%) | 249 | 2026-09-01_02-34-54_mappo_torch |
| Precedence_Lang_s2 | CRASH | 14831 (81%) | 326 | 2026-09-01_02-39-03_mappo_torch |
| Precedence_Lang_s3 | CRASH | 10319 (56%) | 229 | 2026-09-01_02-44-29_mappo_torch |
| Precedence_Lang_s4 | CRASH | 13007 (71%) | 289 | 2026-09-01_02-48-18_mappo_torch |

Checkpoints for every run (including partial checkpoints of crashed runs up
to their death step) are local and untracked under the listed log dirs.

## Crash characterization (from logs/TB only, no re-runs)

- **Identical failure everywhere**: `RuntimeError: CUDA error: device-side
  assert triggered`, surfacing in `mappo.update` at `policy_loss.item()`
  (the first sync point — the true faulting kernel is earlier and
  asynchronous). Same traceback in all 16 mid-run crashes.
- **Sharp onset**: across 19 booted runs, NO death before step 8687 (47%);
  all deaths in 47–81% of budget. Something accumulates or drifts for
  ~70M env steps before the hazard turns on.
- **Stochastic, not deterministic**: same-arm seeds die at different steps;
  3 runs finish the identical binary/config to completion.
- **Arm gradient**: Lang 0/10 completed vs Blind 3/10. The arms differ only
  in observation width (376 vs 645) and a nonzero LANG_SLICE — if the
  eventual root cause is logit explosion, the Lang arm's near-constant
  32-d slice under no preprocessor is a plausible accelerant.
- **No precursor in TB**: for the inspected crash (RB_Blind_s0), value loss
  stable ~0.76, branch entropy healthy, rewards ~−5.6, no NaN in any
  logged scalar at the last write before death. **KLAdaptiveLR had climbed
  3e-4 → ~1.1e-3 (3.7×)** by then; a single-minibatch logit explosion
  between TB writes fits all observations, but is unproven.
- The one boot crash (RB_Lang_s4) is environmental: kit segfaulted 22 ms
  into boot, launched 1 s after the previous run's device-assert teardown
  (breakpad dump at kit/data/Kit/Isaac-Sim/5.1/fbc04fb2….dmp). A
  retry-once-on-boot-failure policy in the campaign runner would absorb it.

## Completed runs — final metrics (mean over final 10% of writes)

| Run | Final return (mean) | Final value loss (r0) | Wall |
|---|---|---|---|
| RoleBinding_Blind_s1 | −1.13 | 0.021 | 7.4 min |
| RoleBinding_Blind_s2 | −7.05 | 0.746 | 7.5 min |
| Precedence_Blind_s4 | −2.03 | 0.706 | 7.4 min |

**The watch-item return shape is confirmed at full budget**: mean returns
end NEGATIVE in all three completed runs (a competent blind oracle should
collect ≈ +2 completion + 5 expected bonus − small costs). Between-seed
spread is large (−1.1 vs −7.1 on the same cell). No assignment-accuracy or
competence claim is made here (the certificate harness measures that), but
these curves say the pre-registered optimiser config is likely not reaching
the competence clause within budget as it stands — relevant to §4.4
clause 3 and the λ/ε pilot's scope.

## For the joint decision (observations, not actions taken)

1. Cheapest diagnosis: one repro run with `CUDA_LAUNCH_BLOCKING=1` (and a
   TORCH_USE_CUDA_DSA build if available) to name the faulting kernel;
   second cheapest: wrap the update in a logit/advantage finiteness check.
2. If the LR-spike hypothesis holds, candidate mitigations to *evaluate
   jointly* (they touch pre-registered training config): an LR ceiling on
   KLAdaptiveLR, or OPEN(3)'s slice-aware preprocessor fallback.
3. The Lang-vs-Blind hazard gap is itself diagnostic signal — worth
   preserving in whatever instrumented repro is run.
4. Paper note: a systematic mid-training instability of skrl MAPPO +
   Discrete at this scale is reportable training-details material either
   way (spec's own "genuinely untested territory" flag was prescient).

---

## Addendum (same day): certificate eval of the three completed checkpoints

Wave-3 harness driven on the real Isaac env (`scripts/eval_checkpoints.py`,
committed): paired manifest, first 2000 eval-split rows per variant, argmax.
Full verdict JSONs in runs/diag/certeval_*.json (untracked).

| Checkpoint | Verdict | Completed episodes (of 4000) | Paired-lane identity | Leak probes |
|---|---|---|---|---|
| RB_Blind_s1 | UNINFORMATIVE (constant assignment) | 2 | PASS (bitwise) | AUC ≈ 0.5, no leak |
| RB_Blind_s2 | UNINFORMATIVE (constant assignment) | 4 | PASS (bitwise) | AUC ≈ 0.5, no leak |
| Prec_Blind_s4 | UNINFORMATIVE (nothing to score) | 0 | PASS (bitwise) | AUC ≈ 0.5, no leak |

Two conclusions:
1. **The audit machinery is validated end-to-end on GPU**: the spec-4.3
   machine check (bitwise lane equality) holds on the real TeamGridEnv at
   4000 envs, scorer consistency passes, probes are calibrated. This was
   the paired design's biggest untested risk and it is now retired.
2. **The negative TB returns were real incompetence, not reward
   accounting**: E[C] ≈ 0.0005/0.001/0.0 versus the pre-registered
   E[C] ≥ 0.90 competence clause. With the crash mechanism now diagnosed
   (docs/CRASH_DIAGNOSIS.md) the milestone's binding constraint is
   LEARNING — credit assignment/exploration under the 128-step horizon —
   which is λ-sweep / task-design territory (spec 4.5 ladder, OPEN(5)/(8)),
   not infrastructure.
