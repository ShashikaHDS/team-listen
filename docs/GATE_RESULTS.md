# M1 throughput-gate results (5090)

Executed per docs/GATE_TASK.md on 2026-09-01, Isaac python (lock: 268d220's
environment.lock.yaml). **VERDICT: GATE PASS** — training-mode MAPPO clears
the ≥100k env-steps/s hard gate at every swept `num_envs`, by 1.4–3.9×.

## 1. Test matrix (Isaac Lab python)

All 13 CPU test files green. Two latent issues surfaced on this box and were
fixed in this commit; neither was reachable on the dev box:

| File | Result | Notes |
|---|---|---|
| test_bank_determinism | 11/11 | |
| test_bank_distfields | 7/7 | also re-run vs real banks, below |
| test_bank_latch_reachability | 6/6 | also re-run vs real banks, below |
| test_conflict_parity | 11/11 | |
| test_grid_core | 21/21 | |
| test_obs_layout | 23/23 | |
| test_paired_lane_identity | 11/11 | after fix (a) |
| test_potential_purity | 9/9 | |
| test_reward_audit | 11/11 | |
| test_rewards | 14/14 | |
| test_skrl_compat | 10/10 | |
| test_spaces | 9/9, **0 skipped** | under new tests/sim_fixture.py; fix (b) |
| test_templates | 17/17 | |

**Fix (a) — CPU GEMM row-position nondeterminism.** `test_cross_process_
bit_identical` failed here: `margins` differed between blind-arm lanes while
every integer trajectory field stayed bitwise-equal, and instrumentation
showed the lanes' observations never diverged. Root cause is the platform,
not the code: this box's torch 2.7.0 CPU GEMM routes different row positions
of one batch through different microkernel paths, so byte-identical rows at
different lane offsets produce logits differing ~1e-5 (minimal repro: rows
duplicated in a (12,376)@(376,5) matmul are NOT bitwise equal on CPU;
**bitwise equal on CUDA**, verified — the real eval path is unaffected).
Fix: `assert_paired_lane_identity` compares the float `margins` diagnostic
with atol=1e-4 on CPU only; every integer field and all CUDA comparisons
stay bitwise. The leak-detection test (`test_language_sensitive_pair_
diverges_and_is_caught`) still passes, so the audit's teeth are intact.
Flagged for design ratification since it touches the audit instrument.

**Fix (b) — test_spaces' base-cfg assumption.** The 4 skips activated under
the new SimulationApp fixture and 2 promptly failed: the base
`TeamGridEnvCfg` defaults to `arm="Blind"`, whose `__post_init__` derives
the FULL-STATE width (645), while the tests asserted OBS_DIM (376) — they
had only ever "passed" on the dev box where the cfg cannot instantiate and
class attrs were checked instead. Tests are now arm-explicit and cover both
widths (Lang→376, Blind→645). `spec_to_gym_space({5}) == Discrete(5)` and
`[{5},{5}] == MultiDiscrete` are now empirically confirmed on the pinned
checkout.

`tests/sim_fixture.py` (new): boots a headless SimulationApp then runs a
test module; flushes wrapped output before `app.close()` (kit teardown
otherwise drops buffered stdout — silent-looking green runs).

## 2. Scenario banks (real, full size)

Built by `scripts/build_scenario_bank.py --variant both` (defaults: K=16384,
ε=0.05, seed rule as recorded in the JSON manifests; realized slip rate
0.0499). Untracked per plan; manifests committed alongside in data/.

| Variant | File | sha256 |
|---|---|---|
| RoleBinding | data/scenario_bank_RoleBinding_68d025a618cf.pt | 68d025a618cf298f546f446bd83ed2f1a11378bc00886c11750d51f83356a727 |
| Precedence | data/scenario_bank_Precedence_0838c7c98df3.pt | 0838c7c98df38d3ba6c0c6eb1405ec87b6e1b15ac58be5535b43852d513e27f0 |

- All 13 checks of the two structural bank suites re-run against the REAL
  16384-row banks (not the smoke banks): **13/13 PASS** — degree-1 alcoves,
  unique-mouth topology, latch-aware==all-free field equality, independent
  BFS parity, Δ sign-symmetric stratification, leak_bit geometric defaults,
  both-orderings feasibility.
- Determinism on the real banks: full rebuild to a scratch dir reproduced
  **bit-identical .pt files** (sha256 equal for both variants).

## 3. Throughput (the gate)

`scripts/bench_env.py` (implemented here per GATE_TASK item 3) on
`Isaac-TeamListen-RoleBinding-Lang-Direct-v0`, headless, fresh SimulationApp
per row; stepping = random actions over 3000 timed steps after 300 warm-up;
training = real skrl MAPPO for 320 timesteps (20 updates). VRAM is
process-level peak from a 0.5s nvidia-smi poller (torch-allocator peak in
parentheses). `clone_environments(copy_from_source=False)` on the
zero-asset scene: **works** (OPEN question closed); `filter_collisions=False`
accepted.

| num_envs | stepping env-steps/s | training env-steps/s | update share | peak VRAM MiB (torch) |
|---|---|---|---|---|
| 1024 | 508k | 136k | 0.73 | 1320 (350) |
| 2048 | 987k | 221k | 0.78 | 1679 (624) |
| 4096 | 1.95M | 298k | 0.85 | 2359 (1168) |
| 8192 | 3.73M | **390k** | 0.90 | 3639 (2260) |

- **HARD GATE ≥100k training FPS: PASS at every size.** Training FPS is
  still rising at 8192 → **recommend freezing training num_envs = 8192**
  (matches N_EVAL_ENVS; memory is trivial at 3.6 GiB of 32).
- Stepping floor is ~5–9× §8.1's >400k target; the optimiser dominates wall
  clock exactly as §8.1 predicted (70–90%), so future tuning knobs are
  `mini_batches` / `learning_epochs` / net width, not env kernels.
- Budget implication: 150M steps ≈ **6.4 min/run** at 8192; the ~110-run
  M1 program + IPPO controls + boots lands ≈ **15 GPU-hours**, comfortably
  under §8.1's 30–45 h estimate.
- Cosmetic: one boot emitted a kit `omni.platforminfo` circular-dependency
  error pair at startup (4096 row); run unaffected, results consistent.

## 4. MAPPO + Discrete falsification run (GATE_TASK item 5)

`scripts/train.py --task Isaac-TeamListen-RoleBinding-Blind-Direct-v0
--algorithm MAPPO --num_envs 4096 --headless --max_iterations 50` — the
wrapper's manifest, entry-point resolution (`skrl_mappo_cfg_entry_point`),
and refusal guards all exercised. 800/800 timesteps at ~64 it/s (~262k
env-steps/s sustained through checkpointing/TB I/O).

- **Value loss decreases for both agents** (robot_0 2.41→1.55, robot_1
  2.28→1.58, first-20% vs last-20% of 50 updates): the critic learns.
- **Actions decode correctly end-to-end**: skrl's float32-stored Discrete
  actions round-trip through `Categorical.log_prob` and the env's
  `.reshape(-1).long()` decode; recorded executed actions are integer
  indices; no dtype/shape crash anywhere. The spec's "genuinely untested
  territory" note (§8.2) is now closed by a live run.
- Caveat, expected at 3.3M env steps (~2% of a full run): mean episode
  return drifts down over the window (12.5→−0.4) while min return widens —
  an exploration-phase artifact at this horizon; no conclusion about task
  learnability should be drawn from a 50-iteration probe. Not a gate
  criterion; flagging for the λ/ε pilot to watch.

## 5. API breakage found

None blocking. Platform findings a future editor should know: the CPU-GEMM
row-position nondeterminism and kit stdout-buffering issues above, and (from
day-0, still true) `isaaclab.envs.utils.spaces` is only importable after
SimulationApp boots.
