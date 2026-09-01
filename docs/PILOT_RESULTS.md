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
