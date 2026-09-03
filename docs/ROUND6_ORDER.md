# 5090 work order: round 6 - entity-encoder probe (OPEN(4) pulled forward)

Queued while the 5090 is powered off (standing discipline: work orders
live in git). For the session owning ~/team_listen. Context:
docs/DECISIONS.md 2026-09-02 entries; the policy-layer interference
mechanism is the target.

## What is new in the repo

harness/models.py - DeepSets entity encoder + skrl-2.x MAPPO classes
(CategoricalEntityPolicy / DeterministicEntityValue) + the
build_entity_models factory. 26 CPU tests green (tests/test_models.py);
run them under the Isaac python too. Parameter counts are BELOW the flat
baseline (0.52-0.64x), so throughput should not regress materially.
READ the build_entity_models docstring fully - it contains the exact
manual-construction wiring (the YAML models block cannot express these
classes; replace the Runner with manual MAPPO construction; note the
value_preprocessor size-injection trap and the 2.x kwarg names).

## Probe cells (ratified config otherwise; full 149.9M budget each)

IMPORTANT: encoder state_dicts are NOT interchangeable with flat-MLP
checkpoints, so "warm start" here means a two-stage encoder-only
protocol, not loading old weights.

- Cell A (2 seeds): encoder, two-stage - 25M steps on the phase1 bank,
  then 125M on the floor bank (floor_94854c2c), checkpoint continuation.
  This reproduces round-5's warm protocol with the new architecture.
- Cell B (2 seeds): encoder, floor bank from scratch, full budget.
  Separates "encoder + informed critic suffices" from "encoder still
  needs the easy pretrain".

## Measurements

Native-mode (paired_stochastic) certified eval at 25/50/75/final, argmax
alongside; single_latch_share; one critic_inspection.py pass on the best
final checkpoint; wall-clock and FPS (encoder kernel cost vs flat).

## Decision rule (pre-agreed)

- Certified native E[C] >= 0.9 any cell -> FULL CAMPAIGN with that
  recipe (encoder + its winning protocol, all arms; Lang arms get their
  own phase1 pretrain built identically; Precedence separate as agreed),
  then certificate + CSI eval, verdicts committed. No further approval.
- 0.3-0.9 -> stop and report (per-checkpoint trajectories + critic
  table); the reconvene decides between encoder tuning and a
  supervisor-level design review.
- < 0.3 in both cells -> STOP. That exhausts the pre-planned ladder;
  the next step is a design reconvene with the author and supervisor,
  not another probe.

Commit results as PILOT_RESULTS.md addendum 9, push. ~4 runs x ~8 min
plus evals.
