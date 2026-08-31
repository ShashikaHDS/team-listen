# Decision log

Running log of design decisions made after M1_SPEC.md was frozen. Newest first.

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
