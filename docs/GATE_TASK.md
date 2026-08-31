# 5090 work order: milestone-1 throughput gate

For the Claude session owning ~/team_listen on the 5090 (per
docs/DECISIONS.md ownership rule). Everything needed is in this repo;
report results by committing docs/GATE_RESULTS.md and pushing. Plain
commit messages, no Co-Authored-By trailer.

Preconditions (all met as of main@wave-2): environment lock committed,
agent YAMLs in place, wave-2 files present (rewards, scenario bank,
rollout harness, cpu_env), 156 CPU tests green on the dev box.

## Tasks, in order

1. Pull main. Run the full CPU test suite here too (`python
   tests/<file>.py` for all 13) with the Isaac Lab python - expect the 4
   test_spaces skips to ACTIVATE and pass under a SimulationApp fixture
   (write the small fixture the test's skip message describes if absent).
2. Build the real scenario banks: `scripts/build_scenario_bank.py` for
   both variants at the spec's full bank size (M1_SPEC section 2/3), and
   run the three bank tests against the real banks (not the smoke banks).
   Banks stay untracked; record their SHAs in GATE_RESULTS.md.
3. Implement `scripts/bench_env.py` per M1_SPEC section 8.2 (it does not
   exist yet - this box writes it because only this box can iterate on
   it): boots Isaac-TeamListen-RoleBinding-Lang-Direct-v0 headless,
   random-action stepping AND MAPPO-training modes, reports sustained
   env-steps/s and peak VRAM.
4. THE GATE: bench TeamGridEnv at num_envs in {1024, 2048, 4096, 8192},
   stepping-only and MAPPO-training, N=2. HARD GATE: >=100k env-steps/s
   sustained in training mode at the best num_envs. Also confirm
   `scene.clone_environments(copy_from_source=False)` works on the
   zero-asset scene (spec's OPEN question) and record peak VRAM per
   config.
5. The MAPPO+Discrete falsification run (spec section 8.2 step 2): 50
   iterations of real MAPPO training on the role-binding task; confirm
   loss decreases and actions decode correctly (the code trace said yes;
   this is the run that proves it).
6. Commit docs/GATE_RESULTS.md with: test matrix, bank SHAs, the FPS
   table, VRAM table, gate PASS/FAIL, and any API breakage found, then
   push. If the gate FAILS, do not tune - report the numbers and stop;
   the fallback ladder (M1_SPEC section 8.3) is a joint decision.
