# OPEN(5) λ-pilot results (5090) — near-zero branch, stopped per rule

Per experiments-e1's work order, 2026-09-01. Config: ratified YAMLs @ 1853dc4
(max_lr 5e-4), RoleBinding Blind, full 149.9M-step budget, weight-finiteness
assert armed (RNG-free), boot-retry active. Runner: `scripts/diag_repro.py
--shaping-lambda … --no-tripwire --weight-check`. TB event dirs under
`runs/26-09-01_*` (untracked), listed per cell below.

**DECISION-RULE VERDICT: every cell is in the "near zero" branch —
E[C] ≈ 1% everywhere against the 0.90 gate — so per the pre-agreed rule I
ran ONE exploratory probe (documented §4) and STOPPED. No campaign re-run.**

## 1. The 3×3 table

Episode-level E[C] derived from the per-step `Info/completion_rate` share:
the env logs per-step means, and per-step `outcome_incomplete_share` ≈ 1/128
calibrates the conversion (episode-level ≈ per-step share × 128; equivalently
completion/(completion+incomplete)). Plateau slope is the OLS slope of the
completion share over the final 30M env steps, in share/100M steps.

| λ | seed | Outcome | E[C] (final, episode-level) | E[C] max | Plateau slope | Wall |
|---|---|---|---|---|---|---|
| 0.05 | 0 | complete | ~1.3% | ~1.3% | ≈ 0 | 446 s |
| 0.05 | 1 | **CRASH @ 12399 (68%)** | ~0–1% | ~1.3% | — | 305 s |
| 0.05 | 2 | **CRASH @ 10031 (55%)** | ~0–1% | ~1.3% | — | 246 s |
| 0.1 | 0 | complete | ~1.3% | ~2.6% | ≈ 0 | 447 s |
| 0.1 | 1 | complete | ~1.3% | ~2.6% | ≈ 0 | 448 s |
| 0.1 | 2 | complete | ~1.3% | ~2.6% | ≈ 0 | 447 s |
| 0.2 | 0 | complete | ~1.3% | ~3.8% | ≈ 0 | 448 s |
| 0.2 | 1 | complete | ~0–1% | ~3.8% | ≈ 0 | 447 s |
| 0.2 | 2 | complete | ~1.3% | ~3.8% | ≈ 0 | 448 s |

λ does not differentiate: all cells plateau at ~1% (max transient ~2–4%,
**peaking mid-run then declining** — the campaign's return-drift shape,
reproduced in completion itself). Branch-step entropy floor: moot at these
competence levels (no cell qualifies for convergence assessment); episode-
mean policy entropy stays ~1.1 nats of 1.61 max — the policies never
committed to a strategy in 150M steps.

**max_lr 5e-4 efficacy (campaign-relevant):** completion-of-run rate 7/9
under the cap vs 3/19 uncapped — the ratified ceiling REDUCES the crash
hazard ~4× but does NOT eliminate it (both residual deaths: identical
Categorical.sample device assert, both in the sparsest-shaping λ=0.05
cells; the weight check never tripped, confirming the explosion stays
intra-update). CRASH_DIAGNOSIS.md's mechanism stands, with LR as a
modulator rather than the sole cause.

## 2. Task validity: the spec-3.1 pre-training smoke number (was missing)

Scripted `GreedyPolicy` (dist-field descent) over the REAL RoleBinding bank,
2048 eval rows, cpu_env: **E[C] = 0.999** (2046/2048), |Δt| mean 4.4,
ties 7% (ties are legal in RoleBinding; the unique-mouth tie exclusion is a
Precedence property). The task is trivially completable by a competent
controller in ~15 steps — **learning, not the task, is broken.**

## 3. ε = 0.05 D_seed non-degeneracy (spec tie-in to this pilot)

GreedyPolicy on identical rows under slip stream 0 vs 1: **77.7% of 512
episodes diverge in their position trajectories, while E[C] is 0.998 in
both streams.** ε = 0.05 yields a non-degenerate CSI seed denominator at
zero competence cost. (Run on the scripted policy: with no competent
trained cell, a trained-policy D_seed would measure noise.) CLOSED.

## 4. The one exploratory probe (exploratory, not adopted)

`--mitigation lr_ceiling --max-lr 1.5e-4` (half base LR), λ = 0.1, seed 0,
full budget: **no crash, no degradation — mean return positive and stable
(~1.8–2.4 all run, vs negative-and-declining at 5e-4) — but E[C] unchanged
at ~1%.** This cleanly separates the two pathologies:

- **Pathology A — optimisation instability** (return/completion decline,
  intra-update explosion): LR-driven; a 1.5e-4 ceiling removes it entirely
  in this run. Proposal for ratification: lower the adopted cap to 1.5e-4
  (or reduce base LR) for all future runs — it also plausibly closes the
  residual λ=0.05 crash hazard.
- **Pathology B — completion is never learned**: the stable return ≈ +1.8
  is almost exactly the full approach-shaping credit (λ|Φ₀| ≈ 1–2): the
  policies learn to approach the stations and park, never exploiting the
  ≈ +7 completion payoff. §3.4's "long-horizon, low-frequency terminal
  credit" prediction applies to COMPETENCE itself, before language enters.

## 5. For the joint decision (per the rule, nothing further run)

The binding constraint is terminal credit / the latch discovery event, and
λ ∈ {0.05–0.2} does not move it. Candidate directions, in rough order of
design-conservatism: (a) raise the completion bonus and/or add a latch-
proximity potential term (touches reward — needs reward_audit re-run);
(b) OPEN(8): lower obstacle density / shorter spawn-to-station distances as
a curriculum that makes accidental latches common enough to learn from;
(c) longer horizon (touches T_DECISION = 128, wide blast radius);
(d) entropy schedule. The GreedyPolicy result guarantees a competent
policy exists at the current task design; the question is purely whether
SGD can find it. GPU cost of this entire pilot + probe + baselines:
~80 min.

---

## Addendum 2: first-latch bonus implemented; compliance bound re-derived

Per experiments-e1's terminal-credit work order. `FIRST_LATCH_BONUS = 2.0`
per agent on its first latch of the episode (absorbing latch ⇒ structurally
once), instruction-free by construction — implemented in rewards.py +
fleet_env/cpu_env (shared `_pre_physics_step` transition mask), 4 new unit
tests (fires once / per agent / wrong-station paid identically / zero
instruction dependence), purity scanner 9/9 over the amended module.

The compliance audit needed a genuine accounting extension, not just a re-
run: the latch bonus is the reward's first per-agent-TIMED term, so (i) the
closed-form plan return gains the discounted per-agent-mean latch term;
(ii) the bound's delta now uses the **min-over-agents** per-agent
difference (each MAPPO agent optimises its own return, so "compliance
unambiguously optimal" must hold per agent — this is stricter than the
mean); (iii) the timeout dominance bound rises by one undiscounted bonus
(a never-completing plan can still latch one robot). The amendment
strictly REDUCES the bound (compliant plans latch later; discounting), and
the hand-arithmetic tests reproduce the audit to 1e-9 under the new
convention (11/11).

**Re-derived exact compliance bounds on the real banks (γ=0.99, λ=0.1):**

| Bank | bound (min over rows × classes × agents) | mean delta | verdict |
|---|---|---|---|
| RoleBinding 68d025a6… | **8.583** | 8.948 | PASS (>0 with margin) |
| Precedence 0838c7c9… | **6.629** | 8.407 | PASS (>0 with margin) |

Compliance remains unambiguously optimal for every agent under the amended
reward. Probe results follow in Addendum 3.

## Addendum 3: first-latch-bonus probe — STOP branch again

3 seeds, RoleBinding Blind, λ=0.1, max_lr 1.5e-4, amended reward, full
budget each. All three completed with no crash and no weight-trip — the
1.5e-4 ceiling is now 4/4 across runs (vs 7/9 at 5e-4, 3/19 uncapped), so
the stability story is closed as far as evidence at this scale can close it.

| Seed | E[C] final | E[C] max (transient) | Final return | Wall |
|---|---|---|---|---|
| 0 | 0.3% | ~1.7% | 2.00 | 446 s |
| 1 | 0.4% | ~2.8% | 2.04 | 449 s |
| 2 | 0.5% | ~2.1% | 2.05 | 448 s |

**Decision rule: "still ~1%" branch → STOPPED, no exploratory probe (as
pre-agreed). The +2.0 first-latch bonus did not unlock completion.** The
~+0.2 return shift vs the pre-amendment probe is consistent with policies
harvesting occasional single latches, but the double-latch event remains
effectively undiscovered, and the mid-run peak-then-fade shape persists at
a stable optimum. The terminal-credit densification at the latch event was
necessary-looking but is evidently not sufficient: the bottleneck now looks
like the JOINT discovery problem (both agents must hold stations
simultaneously within the horizon), which is curriculum territory.

For the OPEN(8) joint design, two requests from the training side:
1. Add a `single_latch_share` scalar to the env's extras log (fraction of
   envs with exactly one latched robot) — the one diagnostic that
   separates "agents rarely latch at all" from "agents latch singly but
   the second never joins", and it is currently not logged.
2. Candidate curriculum axes, cheapest first: spawn-to-station distance
   window (bank builder already parameterises [4,14]); obstacle density /
   cluster count; 12×12 grid held fixed. A bank-level curriculum (a
   sequence of banks) preserves the frozen-scenario audit machinery
   unchanged — nothing in _reset_idx needs to learn about curricula if
   training just swaps TEAM_LISTEN_BANK between phases.

GPU: probe 22 min; cumulative today ≈ 4.5 h.

## Addendum 4: OPEN(8) curriculum probe — hypothesis confirmed, transfer fails

3 seeds × 4 phases (25M/25M/25M/75M; checkpoint continuation across bank
swaps; stamped banks phase1_7ecb4821 / phase2_51724dac / phase3_c6a9e8f1 —
see the loader-stamp integration fix, f6efd34). All 12 phase runs clean.

| Phase (near-window) | E[C] final (s0/s1/s2) | single_latch_share |
|---|---|---|
| 1 [1,3] | **0.688 / 0.695 / 0.695** | 0.37–0.39 |
| 2 [3,6] | 0.395 / 0.395 / 0.379 | 0.39–0.41 |
| 3 [6,10] | 0.139 / 0.105 / 0.101 | 0.22–0.25 |
| 4 certified | **0.007 / 0.006 / 0.005** (plateaued) | 0.11 |

Two findings, both load-bearing:
1. **The joint-discovery hypothesis is CONFIRMED.** With [1,3] spawns the
   policies reach E[C] ≈ 0.69 within 25M steps — completion is learnable,
   the reward works, and the earlier flat-task failures were an
   exploration/discovery problem, exactly as hypothesised. This retires
   the "deeper than exploration" worry the probe was designed to test.
2. **Transfer across windows is the new binding constraint.** Each
   transition halves-or-worse the competence (0.69 → 0.40 → 0.11 → 0.006),
   and the certified-bank endpoint is back at baseline. single_latch_share
   degrades in parallel (0.38 → 0.11): the skill itself erodes rather
   than one agent stranding the other. Under the decision rule this is
   the stop-and-report branch (certified E[C] ≪ 0.3; the "~0 even in
   phase 1" falsification branch explicitly does NOT apply).

Directions for the joint design of round 2, cheapest first: (a) MIXED
phases — sample each episode's bank row from a phase MIXTURE that anneals
(e.g. 70/20/10 easy-mid-hard → … → certified-only), so earlier
competence is continuously rehearsed instead of abandoned at a hard swap
(implementable as bank concatenation at build time: zero loader/env
changes); (b) more, smaller window steps; (c) per-phase LR/entropy warm
restart (the KLAdaptiveLR state currently resets at each phase boundary
by construction — the optimizer LR carries via the checkpoint, but the
schedule's KL memory does not; worth controlling). GPU: 12 runs ≈ 24 min.

Correction note for the record: an interim mid-probe status (sent before
all runs finished) misattributed one seed's phase-1 curve to the certified
phase due to a run-directory alignment error; the table above is from the
complete, order-verified extraction. No pushed number was affected.

---

## Addendum 5: round-2 annealed mixtures — below-0.3 branch; curriculum-by-spawn-distance falsified

3 seeds × stages S1–S4 (mix banks mix1_4e637c2b / mix2_4fc30969 /
mix3_dfb447ea / mix4_184506db; ratios 60/25/10/5 → 5/10/15/70; certified
eval rows embedded split=1 in every stage bank), 25/25/25/75M schedule,
checkpoint continuation, ratified config. All 12 training runs + 3
stage-eval boots clean. Stage-boundary CERTIFIED evals (argmax, 1024 rows)
via the committed scripts/eval_stage_checkpoints.py:

| Stage | train-mixture E[C] (s0/s1/s2) | CERTIFIED eval E[C] | certified share |
|---|---|---|---|
| S1 | 0.46 / 0.46 / 0.48 | 0.009 / 0.016 / 0.014 | 5% |
| S2 | 0.39 / 0.38 / 0.38 | 0.021 / 0.018 / 0.019 | 15% |
| S3 | 0.19 / 0.17 / 0.17 | 0.015 / 0.009 / 0.006 | 40% |
| S4 | 0.05 / 0.04 / 0.05 | **0.000 / 0.001 / 0.002** | 70% |

**Decision rule: below-0.3 branch — stop and reconvene.** And the numbers
say something stronger than "mixtures also fail": train-mixture E[C]
tracks the EASY-row share almost exactly at every stage (S1 0.60×~0.8 ≈
0.46; S4 ≈ 0.05), i.e. the policy completes near-window rows and NOTHING
else, with continuous rehearsal, intermediate windows, and a 70%-certified
training majority all unable to move the certified number off ~0. Round 1
+ round 2 together FALSIFY spawn-distance curriculum as the route to
certified competence in either hard-swap or annealed-mixture form. The
implied competence-vs-approach-distance profile is a cliff: window [1,3]
≈ 0.7–0.8, [3,6] ≈ 0.3–0.4, [6,10] ≈ 0, certified [4,14] ≈ 0.

For the reconvene — what the evidence now constrains:
1. NOT forgetting (round 2 removes swaps; unchanged), NOT discovery in the
   local sense (phase-1 learning is fast and reliable), NOT map visibility
   for this arm (Blind observes the true obstacle plane in its full
   state). The failure scales with APPROACH LENGTH itself.
2. Candidate mechanisms for round 3, for joint prioritisation:
   (i) coordination horizon — both agents must jointly commit over 2–5×
   longer trajectories; branch-step entropy vs distance would test it;
   (ii) value/credit propagation depth at rollouts=16 — the 16-step
   GAE window is shorter than most certified approaches (4–14 steps per
   robot plus conflict detours); rollouts 32/64 is a one-line YAML probe
   and my nominee for cheapest-first;
   (iii) OPEN(4)/architecture (entity encoder) — least conservative.
3. Ops note: eval_stage_checkpoints.py (committed) makes any future
   recipe's certified trajectory a ~2-minute add-on per seed.

GPU: 12 runs + 3 evals ≈ 30 min. Cumulative today ≈ 5.5 h.

---

## Addendum 6: round-3 rollouts-depth probe — falsified; entropy diagnostic reframes the mechanism

**Training probe (order §1-2).** Flat certified training, rollouts {32, 64}
× 2 seeds, `mini_batches` scaled {8, 16} to hold the minibatch at 32768
(baseline 16×8192/4), ratified config otherwise, full 149.9M budget. All
four runs clean; wall ≈ 447–450 s each (throughput insensitive to rollout
depth); VRAM uneventful. Certified-eval trajectories (3 mid checkpoints +
final, scripts/eval_stage_checkpoints.py):

| Cell | E[C] @25% / 50% / 75% / final |
|---|---|
| rollouts 32, s0 | 0.000 / 0.000 / 0.000 / 0.000 |
| rollouts 32, s1 | 0.000 / 0.000 / 0.000 / 0.001 |
| rollouts 64, s0 | 0.000 / 0.002 / 0.004 / 0.001 |
| rollouts 64, s1 | 0.000 / 0.001 / 0.001 / 0.003 |

**Decision rule: "~0 at both" branch — stopped immediately. Mechanism (ii)
(GAE credit-window depth) is falsified**: quadrupling the credit window
moves certified competence by nothing.

**Entropy-vs-distance diagnostic (order §3).** scripts/entropy_vs_distance.py
(committed; early-step proxy for the branch step, limitation documented; an
estimator bug — active-mask normalisation missing the agent axis, caught
because reported values exceeded ln 5 — was fixed and the diagnostic rerun
before any number below was recorded). Round-1 checkpoints, argmax, 512
eval rows per bank:

| Bank (window) | p1-competent ckpt: early-4 H / E[C] | certified-trained ckpt: early-4 H / E[C] |
|---|---|---|
| phase1 [1,3] | 0.897 / 0.303 | 0.305 / 0.021 |
| phase2 [3,6] | 0.783 / 0.080 | 0.335 / 0.002 |
| phase3 [6,10] | 0.613 / 0.010 | 0.367 / 0.002 |
| certified [4,14] | 0.499 / 0.006 | 0.350 / 0.000 |

The signature mechanism (i) predicted — entropy collapsing/dithering at
long distance — is NOT what appears. The competent policy becomes MORE
deterministic as distance grows while completing less, and the
certified-trained policy is near-deterministic everywhere (all-steps H
≈ 0.11, the parked equilibrium). The failure is confident commitment to a
non-completing behaviour, not indecision: a stable SGD attractor.
(Secondary observation, flagged for the eval design: the p1 checkpoint
scores 0.303 under argmax here vs ~0.69 under training-time stochastic
action selection — argmax-vs-sampled evaluation sensitivity worth its own
check before any headline eval.)

**Where this leaves the mechanism space.** Falsified so far: spawn-distance
curriculum (both forms), observation preprocessing, GAE credit depth,
LR instability (fixed separately), local discovery, forgetting, map
visibility. Surviving: (iii) architecture/representation (OPEN(4) entity
encoder), and — newly suggested by the entropy data — the PARKED-ATTRACTOR
question: WHY is near-station parking a stable fixed point when the
reward-audit says completing dominates it by ≥ 8.58 discounted reward?
Optimality is not gradient-reachability; candidate probes for the
reconvene, cheapest first: (a) value-function inspection at
mouth-adjacent states (does the critic ever see the +7?); (b) targeted
exploration at the decisive step; (c) the OPEN(4) encoder. GPU: round 3
≈ 65 min incl. diagnostics.

---

## Addendum 7: round-4 eval-mode + critic studies — measurement bias quantified; parked attractor is critic-blindness

**Study A — eval-mode recalibration.** New `paired_stochastic` rollout mode
(fbf62ea): per-(base scenario, agent, step) uniforms tiled across a plan's
lanes, so identical logits give bit-identical samples and the spec-4.3
blind machine check SURVIVES sampling-mode eval — admissibility proven on
the CPU stand-in before use (blind lanes bitwise-equal under sampling;
different seeds genuinely sample; language-sensitive divergence still
caught; tests committed). 5 checkpoints × 4 banks × 2 modes, 512 rows.
Headline rows (E[C], argmax → paired_stochastic):

| Checkpoint | phase1 | phase2 | phase3 | certified |
|---|---|---|---|---|
| p1-competent | 0.30 → **0.58** | 0.08 → **0.30** | 0.010 → **0.14** | 0.006 → **0.098** |
| flat-16 (original) | 0.016 → 0.027 | — | — | 0.002 → 0.004 |
| flat-32 (round 3) | 0.010 → 0.033 | — | — | 0.000 → 0.002 |
| flat-64 (round 3) | 0.031 → 0.059 | — | — | 0.000 → 0.012 |
| mixture-S4 | 0.065 → 0.066 | — | — | 0.000 → 0.002 |

SAY IT LOUDLY, per the order: **argmax materially understated competence
everywhere** — up to 16× on the certified bank for the competent policy
(0.006 → 0.098). Rounds 1–3's certified numbers were all biased LOW, and
round-1/2 "transfer ≈ 0" should be read as "transfer ≈ 10% native-mode
from the p1 policy". HOWEVER: no decision-rule branch changes — the best
certified native number is 0.098, far below both the 0.9 gate and the 0.3
report line, and certified-TRAINED checkpoints stay ≤ 1.2% native. The
competence gap is real; its measurement was biased. All future evals run
paired_stochastic alongside argmax.

**Study B — critic inspection** (scripts/critic_inspection.py, committed;
values inverse-transformed through each checkpoint's trained
RunningStandardScaler, so REAL return units; yardstick: both_adjacent
completing next step ≈ +10.9, parking ≈ −1):

| Family | p4 (cert-trained) | flat-16 | p1-competent |
|---|---|---|---|
| early_far | 0.77 | 0.88 | 3.35 |
| both_adjacent | 0.92 (n=3) | 0.50 (n=14) | 1.58 (n=16) |
| parked_near | 0.67 | 0.70 | 2.94 |
| one_latched_partner_near | 0.79 | 0.50 | 3.18 |
| one_latched_partner_far | 0.48 | 0.55 | 3.17 |
| both_latched | 0.63 | 0.51 | **4.18** |

**Verdict: CRITIC-BLINDNESS.** The certified-trained critics are flat
(~0.5–0.9) across every family — both_adjacent is priced like parked_near,
and even both_latched states that JUST collected the payoff carry no
premium. The p1-competent critic is informed (both_latched highest). Per
the round-4 dichotomy this puts the parked attractor in EXPLORATION-FIX
territory, not architecture territory. (Caveat recorded: both_adjacent
has tiny visit counts (3–16) precisely because the policies avoid it —
itself evidence.)

**Synthesis for the joint decision.** The mechanism is a self-reinforcing
rarity trap: completions too rare on certified → the critic never encodes
the payoff → no policy gradient toward completing → completions stay rare.
Phase-1 training breaks the loop by making the event common — and the
eval-mode study shows that skill is worth ~10% certified natively — but
CONTINUED training on rare-completion distributions actively erodes it
(p1 0.098 → p4 ≤0.002 native). Candidate round-5 recipes, for joint
ranking: (a) permanent competence floor — never anneal the training
mixture below ~30% easy rows, evaluated native-mode (one bank build + one
probe); (b) warm-start from p1 + floor; (c) entropy schedule. GPU round 4:
≈ 25 min, eval-only as ordered.
