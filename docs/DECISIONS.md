# Decision log

Running log of design decisions made after M1_SPEC.md was frozen. Newest first.

## 2026-09-01 - Curriculum round 1: hypothesis confirmed, transfer fails; round 2 = annealed mixtures

Probe (PILOT_RESULTS.md addendum 4): phase-1 E[C] 0.69 in 25M steps
CONFIRMS the joint-discovery hypothesis (completion is learnable, the
reward works, prior failures were exploration); but hard bank swaps
erode the skill at every transition (0.69 -> 0.40 -> 0.11 -> 0.006 on
certified). RATIFIED: the 5090's loader-stamp mechanism (SHA-covered
curriculum_near_window payload stamp relaxes ONLY the delta_gap cap and
ONLY for stamped banks; certified gate and bytes unchanged;
roundtrip-tested) - correct design. The interim-status correction note
is acknowledged as model conduct for the deviations log.

ROUND 2 DESIGN (option (a), cheapest): ANNEALED MIXTURE banks built by
row concatenation from {phase1, phase2, phase3, certified-train} - no
loader or env changes. Fixed pre-declared stage ratios over the same
25/25/25/75M schedule:
  S1 60/25/10/5, S2 30/30/25/15, S3 10/20/30/40, S4 5/10/15/70
(percent p1/p2/p3/cert-train; deterministic interleave, stamped, mixture
composition recorded in the manifest with source SHAs). Rationale: the
erosion pattern (single_latch_share fell in parallel) is
forgetting-at-swap; continuous rehearsal with a monotone certified ramp
is the standard fix and costs bank construction only. Success criterion
unchanged: E[C] >= 0.9 on the CERTIFIED EVAL split at end of S4.
KLAdaptiveLR state reset at stage boundaries is retained as-is this
round (logged as a controlled variable for a later ablation if needed).
Implementation delegated to the 5090 session (it owns the stamp/loader
mechanics); 3-seed probe under the standing decision rule.

## 2026-09-01 - Curriculum round 1: hypothesis confirmed, transfer fails; round 2 = annealed mixtures

Probe (PILOT_RESULTS.md addendum 4): phase-1 E[C] 0.69 in 25M steps
CONFIRMS the joint-discovery hypothesis (completion is learnable, the
reward works, prior failures were exploration); but hard bank swaps
erode the skill at every transition (0.69 -> 0.40 -> 0.11 -> 0.006 on
certified). RATIFIED: the 5090's loader-stamp mechanism (SHA-covered
curriculum_near_window payload stamp relaxes ONLY the delta_gap cap and
ONLY for stamped banks; certified gate and bytes unchanged;
roundtrip-tested) - correct design. The interim-status correction note
is acknowledged as model conduct for the deviations log.

ROUND 2 DESIGN (option (a), cheapest): ANNEALED MIXTURE banks built by
row concatenation from {phase1, phase2, phase3, certified-train} - no
loader or env changes. Fixed pre-declared stage ratios over the same
25/25/25/75M schedule:
  S1 60/25/10/5, S2 30/30/25/15, S3 10/20/30/40, S4 5/10/15/70
(percent p1/p2/p3/cert-train; deterministic interleave, stamped, mixture
composition recorded in the manifest with source SHAs). Rationale: the
erosion pattern (single_latch_share fell in parallel) is
forgetting-at-swap; continuous rehearsal with a monotone certified ramp
is the standard fix and costs bank construction only. Success criterion
unchanged: E[C] >= 0.9 on the CERTIFIED EVAL split at end of S4.
KLAdaptiveLR state reset at stage boundaries is retained as-is this
round (logged as a controlled variable for a later ablation if needed).
Implementation delegated to the 5090 session (it owns the stamp/loader
mechanics); 3-seed probe under the standing decision rule.

## 2026-09-01 - OPEN(8) curriculum design (response to the latch-bonus stop)

The first-latch-bonus probe stopped per rule (E[C] 0.3-0.5%): agents
harvest single latches but the JOINT double-latch event is effectively
never discovered - a joint-exploration problem, not a credit-magnitude
problem. DESIGN (implemented dev-side, tests green): a bank-level
curriculum that leaves every audit surface untouched. Three
TRAINING-ONLY RoleBinding banks (scripts/build_curriculum_banks.py,
k=4096 each) with NEAREST-station spawn-distance windows [1,3] / [3,6] /
[6,10], each robot spawning nearest a DIFFERENT station so accidental
double latches are locally discoverable; phase 4 is the certified real
bank unchanged. Curriculum mode is an explicit generator flag
(--near-window, requires --tag) that skips the anti-leak symmetry
machinery - acceptable because these banks are never evaluated on; the
certified generator path is byte-identical when the flag is unset
(regression-tested). Training schedule FIXED for reproducibility, not
adaptive: 25M / 25M / 25M / 75M env steps with checkpoint continuation
across bank swaps. New env diagnostic single_latch_share (fraction of
envs with exactly one latched robot) separates "rarely latch" from
"second never joins". Success criterion unchanged: E[C] >= 0.9 on the
CERTIFIED bank at the end of phase 4.

## 2026-09-01 - Lambda pilot verdict: two pathologies separated; terminal-credit fix is next

Pilot (docs/PILOT_RESULTS.md): E[C] ~ 1% at every lambda in {0.05, 0.1,
0.2} (no differentiation); scripted GreedyPolicy scores 0.999 on the same
bank, so the task is valid and LEARNING is the sole blocker; policies
learn approach-and-park (collect shaping credit, never discover the
terminal latch payoff). The exploratory max_lr 1.5e-4 probe removed the
optimisation instability entirely (stable positive returns, no crash)
without changing competence, cleanly separating Pathology A (LR-driven
instability) from Pathology B (terminal credit never discovered).

RATIFIED: (1) max_lr lowered 5e-4 -> 1.5e-4 in both YAMLs (eliminates
Pathology A in the probe; plausibly closes the residual lambda=0.05
crash hazard). (2) lambda frozen at 0.1 (spec default; no cell
differentiated). (3) epsilon=0.05 D_seed non-degeneracy CLOSED (77.7%
trajectory divergence at zero competence cost). (4) NEXT RUNG, targeting
Pathology B at the reward (most design-conservative of the pilot's
options): add an instruction-free PER-AGENT FIRST-LATCH BONUS (+2, once
per agent per episode) to bridge the last-mile credit gap between
approach shaping and the terminal payoff; completion (+2) and outcome
(+-10*Y) terms unchanged. Gates before any training: purity tests must
stay green (the bonus reads latch state only, never instruction/
assignment) and reward_audit's compliance bound must be recomputed and
remain positive. Then a 3-seed probe at lambda=0.1; E[C] >= 0.9 triggers
the campaign, failure escalates to OPEN(8) curriculum (easier maps).
Spec amendment ledger: this modifies M1_SPEC's reward table; DECISIONS.md
is the amendment record.

## 2026-09-01 - Crash fix ratified; binding constraint moves to learning

Root cause named (docs/CRASH_DIAGNOSIS.md): KLAdaptiveLR growth produces
an intra-update logit explosion in the 5-way Categorical, asserting in
Categorical.sample on the next minibatch; deterministic per seed;
preprocessor probe falsified the observation-scaling hypothesis.
RATIFIED: (1) max_lr 5.0e-4 in learning_rate_scheduler_kwargs of BOTH
agent YAMLs (mechanism-aligned; 1.7x warm-up headroom retained; outside
the evaluation prereg per the crash-response policy); (2) a per-minibatch
weight-finiteness assert armed for campaign runs only; (3) headline
tables use homogeneous-config runs only - the three pre-fix completed
runs are diagnostic material, not table rows; (4) instrumentation lesson
recorded: probing a policy with act() consumes global RNG and diverts
deterministic trajectories - compute logits sample-free or restore RNG
state. ALSO CLOSED: the wave-3 audit machinery is validated end-to-end
on the real Isaac env (bitwise paired-lane equality at 4000 envs, scorer
consistency, calibrated probes).

FINDING: the certificate eval shows completed policies are incompetent
(E[C] ~ 0 vs the 0.90 clause) - the negative returns were real learning
failure, not reward accounting. The binding constraint is now credit
assignment/exploration, which is the spec's pre-registered lambda/epsilon
pilot (OPEN(5), run before any headline) and, if that fails, OPEN(8)
grid-size/density reduction. The 20-run campaign re-run is DEFERRED until
a pilot cell demonstrates competent blind learning.

## 2026-09-01 - Training campaign crash: diagnosis-first, mitigations gated on evidence

First M1 campaign (docs/TRAIN_M1_RESULTS.md): 16/20 runs died from an
identical stochastic CUDA device-side assert with 47-81% onset, Lang
0/10 vs Blind 3/10 completed, and all completed runs ended with negative
returns (the ratified watch item confirmed at full budget). Leading
hypothesis: KLAdaptiveLR spike (3e-4 to 1.1e-3 observed) driving a
single-minibatch logit explosion in the Categorical head, with the Lang
arm's unpreprocessed near-constant 32-d slice as accelerant. DECISIONS:
(1) diagnosis before mitigation - instrumented repro names the faulting
kernel first; (2) candidate mitigations (KLAdaptiveLR ceiling,
slice-aware observation preprocessor per OPEN(3)) are evaluated as
single-run probes, adopted only on evidence, and any adopted change to
the TRAINING config is disclosed here - config/preregistration.yaml
freezes the EVALUATION protocol, not optimiser hyperparameters, and no
trained policy has been evaluated yet; (3) the campaign runner gains a
retry-once-on-boot-failure policy (the single kit segfault was
environmental); (4) the competence question is separated from the crash:
the three completed checkpoints go through the wave-3 certificate
harness's non-gating competence path before any reward-balance change is
considered.

## 2026-09-01 - Throughput gate PASS; training num_envs frozen at 8192

Gate results (docs/GATE_RESULTS.md, be4e3a5): MAPPO training mode sustains
136k/221k/298k/390k env-steps/s at 1024/2048/4096/8192 envs against the
100k hard gate; peak VRAM 3.6 GiB; MAPPO+Discrete proven by a live run.
DECISION: training num_envs frozen at 8192 (same as N_EVAL_ENVS); the M1
program is ~15 GPU-hours. RATIFIED (a): assert_paired_lane_identity
compares the float margins diagnostic with atol=1e-4 on CPU only (torch
CPU GEMM routes batch-row positions through different microkernels; CUDA
verified bitwise-clean; all integer fields bitwise everywhere) - leak
detection teeth verified intact. RATIFIED (b): the base-cfg Blind default
vs OBS_DIM test inconsistency fix (tests now arm-explicit over both
widths). WATCH ITEM for the lambda/epsilon pilot: mean episode return
drifted down (12.5 to -0.4) over the 50-iteration falsification window
while value loss improved - expected at 2% of a run, but if the same
shape appears at 30M+ steps, revisit the shaping/collision balance.

## 2026-08-31 - Language-cache projection: JL finding accepted, LDA-augmented projection to be stored in the artifact

The 5090's 384d-vs-32d ridge probe on the committed lang cache (see
docs/5090_REPORT.md) shows the fixed 32-d JL projection destroys much of
MiniLM's class separability (train acc 0.87-0.94 at 384d drops to
0.64-0.73 at 32d), falsifying M1_SPEC §5.2's "JL trivially preserves
cluster geometry" for minimal-pair-constructed classes. Decision:
build_lang_cache.py gains a class-aware (LDA-augmented) projection stored
ALONGSIDE the JL projection in the same artifact, selected by config, so
OPEN(4)'s escalation is a switch rather than a rebuild. The JL projection
remains the default until the M1 gate says otherwise. Second implication
adopted: held-out-sentence evaluation of the Lang arm is partly bounded by
encoder geometry, so the paper needs attribution language and the month-2
encoder ablation gains priority.

## 2026-08-31 - 5090 session ownership

Two Claude sessions on the 5090 worked overlapping day-0 tasks (dispatch
ambiguity from rotating session names). Resolved by the sessions
directly: day-0 was committed by one (4c119b8, 268d220); from now on the
isaac_sim_vla VS Code session owns ~/team_listen exclusively (wave-2
throughput gate onward). Coordination ground truth is git commits and
docs/, never session registries.

## 2026-08-31 - Environment lock highlights (from day-0)

Isaac Lab main @ b0542fe2d (release/3.0.0 default requires Isaac Sim 6.x;
main supports 5.1). skrl 2.1.0 passes the >=2.0.0 hard gate.
episode_length_s = 12.9 is SAFE on the training box (12.9/0.1 == 129.0
exactly, plus ceil). test_spaces.py cannot cheaply import
isaaclab.envs.utils.spaces (pxr import chain) - needs file-path loading
with a stub package or a SimulationApp fixture. test_yaml_keyset.py must
scope to the agent/memory/trainer/seed blocks (models.policy keys differ
by mixin, legitimately).
