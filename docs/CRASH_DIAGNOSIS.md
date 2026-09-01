# Crash diagnosis — the campaign's CUDA device-side assert

Per experiments-f1's diagnosis work order, 2026-09-01. Instrument:
`scripts/diag_repro.py` (committed) — finiteness tripwire wrapped around
`MAPPO.update` (skrl untouched), per-update LR logging, `--mitigation`
in-process probe overrides, `--no-tripwire` control mode.

**ROOT CAUSE (named): non-finite policy logits arising MID-UPDATE, killing
`torch.distributions.Categorical.sample()` with a device-side assert on the
next minibatch forward.** Faulting site, from a deterministic repro under
`CUDA_LAUNCH_BLOCKING=1`:

```
mappo.py:433 update → skrl categorical.py:52 act
  → torch/distributions/categorical.py:135 sample
  → RuntimeError: CUDA error: device-side assert triggered
```

Proposed mechanism (fits every observation): KLAdaptiveLR grows the LR
(3e-4 → ~1.1e-3, 3.7×, logged) until one large-LR minibatch step on this
5-way Categorical explodes the policy weights; the following minibatch's
forward yields NaN logits and the sampling kernel asserts. The explosion is
*intra-update*: per-update tripwires and TB both see healthy values right
up to death because the update that dies never finishes to write anything.

## Evidence chain (chronological, 6 GPU runs ≈ 45 min)

1. **Tripwire run, CUDA_LAUNCH_BLOCKING=1** (RB-Lang seed 0, campaign death
   step 9839): ran to completion, 0 trips in 2×1144 update sweeps.
2. **Tripwire run, non-blocking**: completed again, 0 trips.
3. **Control (no tripwire, no blocking)**: died at **exactly 9839** — the
   campaign's death step for this seed. The crash is **deterministic per
   (seed, code path)**; the campaign's varying death steps were
   seed-dependence, not scheduling noise.
   - Runs 1–2 survived because the tripwire's logit sweep called
     `model.act()`, which SAMPLES and consumes global RNG, deterministically
     diverting the trajectory (an instrumentation lesson worth recording:
     probing a policy with `act()` perturbs the run; compute logits without
     sampling, or restore RNG state).
4. **Control + CUDA_LAUNCH_BLOCKING=1**: died at 9839 with the synchronous
   traceback above — the kernel named.
5. **Mitigation probe (a) `KLAdaptiveLR max_lr=5e-4`**, seed 0, no
   tripwire: **completed** (18304/18304, wall 404 s). Mechanistically
   aligned with the root cause, though n=1 and any perturbation reshuffles
   the deterministic trajectory — adoption should rest on the mechanism +
   a small confirmation batch, not this run alone.
6. **Mitigation probe (b) `RunningStandardScaler` obs preprocessor**, seed
   1, no tripwire: **died at 8591** (campaign seed 1 died 8719). Fails to
   prevent the crash → preprocessing is NOT the fix; also ~confirms the
   crash's insensitivity to observation scaling and OPEN(3)'s ordering.

Numerical-NaN-in-rollout, GAE-NaN, and env-side indexing hypotheses are
all falsified by the tripwire sweeps (no non-finite value in any rollout
tensor, GAE output, or pre-update logit across ~4,600 sweeps). The earlier
"async race" reading of runs 1–2 was wrong — the suppressor was RNG
divergence, not synchronization (run 4 crashes *under full serialization*).

## Proposal for ratification (not adopted; DECISIONS.md is the gate)

1. **Adopt `max_lr: 5.0e-4` in `learning_rate_scheduler_kwargs`** for both
   agent YAMLs. Rationale: caps exactly the mechanism that kills runs;
   KLAdaptiveLR's own default max (1e-2) was never designed for a 5-way
   Categorical over hand-normalised features; 5e-4 still allows 1.7× warm-up
   over the base LR. The optimiser config is outside the evaluation prereg
   (DECISIONS.md crash-response policy), so this is ratifiable without a
   prereg amendment.
2. Optional belt-and-braces (cheap, catches any residual explosion):
   train-time `torch.nan_to_num` is NOT proposed (it would mask real
   pathology); instead a per-minibatch weight-finiteness assert in the
   training wrapper can be armed for the re-run campaign only.
3. Boot flakes: 3 further kit boot segfaults today (~200 ms in, three
   different kit plugins: telemetry, crashreporter regex compile, carb
   tasking) — all absorbed by the DECISIONS.md retry-once policy; campaign
   runner now implements it (`diag_probes` runner and the eval runner both
   retry once on a no-DIAG_START boot).

## Re-run cost once ratified

Campaign re-run (20 runs) ≈ 90 min. The three completed no-mitigation runs
(RB s1/s2, Prec s4) remain valid as-is for any analysis that survives the
config change decision; whether mixed-config seeds are poolable for the
paper's certificate table is an experiments-f1 call (recommendation: re-run
all seeds under the ratified config for a homogeneous table — it is cheap).
