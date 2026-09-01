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
