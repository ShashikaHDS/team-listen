# Milestone 1 Implementation Spec (FINAL) — "Does the Team Actually Listen?"

**Scope:** Isaac Lab `DirectMARLEnv` port of the grid fleet env at N=2, two language-necessary task variants, the instruction-leakage audit (formerly "language-necessity certificate"), sentence-embedding-conditioned MAPPO, and CSI v1.
**Repo root:** `d:/RL/team_listen/` (Windows dev box; training on the Linux 5090 with `~/isaac-sim-standalone-5.1.0-linux-x86_64` + `~/IsaacLab`).
**Source of truth for grid semantics:** `reference/env_paper.py` + `reference/test_env_paper.py` (verified read).

Changes against the first-draft design are tagged `[FIXED: …]`. `OPEN(n)` items are collected in §9.

---

## 0. Day-0 version pinning (blocking prerequisite)

`scripts/print_versions.py` writes `config/environment.lock.yaml`:

| Key | Source | Why |
|---|---|---|
| `isaac_sim` | `SimulationApp` banner | reproducibility statement |
| `isaaclab` | `isaaclab.__version__` + `git rev-parse HEAD` | the MAPPO YAML **must** be copied from this checkout |
| `skrl` | `skrl.__version__` | see rename table below |
| `gymnasium` | module attr | gymnasium ≥1.0 + skrl <1.4 ⇒ `AttributeError: 'OrderEnforcing' object has no attribute 'state'` on MAPPO. (Dev box currently has gymnasium 1.1.1, torch 2.8.0+cu128, numpy 2.0.2; `isaaclab` is **not** importable from system python — it lives in the Isaac python.) |
| `torch`, `numpy`, `sentence-transformers` | module attrs | embedding-cache reproducibility |

**Hard gate:** `skrl >= 2.0.0`.

`[FIXED: two of the five claimed 1.4→2.x renames were wrong and would have produced a bogus compat shim.]` The **verified real** renames are: `shared_observation_spaces` → `state_spaces`; `shared_state_preprocessor` → `state_preprocessor` (+ new `observation_preprocessor`); `value_loss_scale` default 1.0 → 2.5; `act(timestep=…)` keyword-only. The following are **not** renames and must not be "translated": YAML `lambda` is still accepted (`Runner._check_cfg_compatibility` maps it to `gae_lambda`), and the `models:` block is still **flat** per-role in 2.x (`Runner._generate_models` copies it per agent id). `harness/skrl_compat.py` therefore covers only the four real items and is a no-op on 2.x; its translation table is asserted in `tests/test_skrl_compat.py`.

`[FIXED: §0 previously cross-referenced the MAPPO YAML as "§1.9"; it is §1.14.]`

---

## 1. Environment architecture

### 1.1 Substrate: zero-asset, tensors-only

`TeamGridEnv(DirectMARLEnv)` holds **no physics assets**. `_setup_scene()` calls `self.scene.clone_environments(copy_from_source=False)` and nothing else (dome light only under `cfg.debug_vis`). `_apply_action()` is a no-op; the whole transition runs in `_pre_physics_step()` on `self.device` tensors.

`WHY:` an empty PhysX scene degenerates `sim.step()` to one simulate/fetch over zero actors — a constant sub-millisecond cost independent of `num_envs`. Grid transitions stay exact; the documented "stale values after reset" and kinematic-pose-write (Isaac Sim #251) issues are structurally unreachable. `clone_environments()` is unconditional because with `replicate_physics=True` `scene.env_origins` stays `None` until it runs.

`[FIXED: filter_collisions]` `InteractiveSceneCfg.filter_collisions = False`. `clone_environments()` otherwise passes `enable_env_ids=True` into the cloner, requesting PhysX env-id collision filtering on a collider-free scene.

### 1.2 Module split — the transition core is importable without Isaac

```
tasks/team_listen/grid_core.py   # pure torch, imports only `torch`
tasks/team_listen/fleet_env.py   # thin DirectMARLEnv shell
```

```python
def apply_slip(actions, slip_row)                                   -> actions        # bank-indexed stochasticity
def step_positions(cur, actions, occ, latched, bounds)              -> (nxt, hit_obstacle, hit_robot)
def reveal(known_free, known_obs, occ, pos, offsets)                -> in-place scatter
def latch_update(pos, target_cells, target_valid, latched, latch_slot, latch_time, t)
def matching_potential(dist_field, pos, target_valid)               -> (E,) float     # NO `assign` argument
```

`[FIXED: Φ signature contradiction.]` `matching_potential` takes **no** `assign` / `instr_id` / `lang_vec`. §1.12 asserts Φ is instruction-free; the old draft signature `bfs_potential(dist_field, pos, assign)` contradicted that and would have made the reward a live leak channel. `tests/test_potential_purity.py` statically asserts that `rewards.py` and `grid_core.matching_potential` never reference `self.assign`, `self.instr_id`, `self.lang_vec` or `bank.leak_bit`.

`WHY the split:` CPU unit tests and the numpy differential-parity harness run in seconds without `SimulationApp`; the core drops into a plain vectorised gym env if Isaac Lab throughput disappoints; and the conflict-resolution port — the single most likely silent divergence from `env_paper.py` — becomes directly testable.

### 1.3 Tensorised world state

`R = C = 12`, `P = 144`, `N_AGENTS = 2`, `MAX_AGENTS = 4`, `MAX_TARGETS = 3`, `LANG_DIM = 32`, `T_DECISION = 128`.

`[FIXED: layout foreclosed three downstream arms.]` Agents and targets are **fixed-capacity padded entity slots** with presence masks, and the language vector is **per-agent**. At N=2 the extra slots are zeroed, so M1 numbers are unchanged, but per-agent prompting, the VLM commander's per-agent sub-goal tokens, mid-episode reassignment, a third target, and N=4 all become config/data changes rather than a dimension change that would invalidate every M1 checkpoint.

| Buffer | Shape | dtype | Meaning |
|---|---|---|---|
| `occ` | `(E,R,C)` | int8 | ground truth, 0 free / 1 obstacle |
| `known_free`, `known_obs` | `(E,R,C)` | bool | fog of war |
| `pos` | `(E,MAX_AGENTS,2)` | int16 | (row,col) |
| `agent_valid` | `(E,MAX_AGENTS)` | bool | presence mask |
| `target` | `(E,MAX_TARGETS,2)` | int16 | station cells |
| `target_valid` | `(E,MAX_TARGETS)` | bool | presence mask |
| `latched` | `(E,MAX_AGENTS)` | bool | |
| `latch_slot` | `(E,MAX_AGENTS)` | int8 | −1 unlatched, else target index |
| `latch_time` | `(E,MAX_AGENTS)` | int16 | −1 unlatched |
| `dist_field` | `(E,MAX_TARGETS,R,C)` | int16 | **latch-aware** BFS (§1.10) |
| `scenario_id` | `(E,)` | int64 | the frozen-physics key |
| `slip_stream` | `(E,)` | int8 | which stored stochasticity stream is in force |
| `instr_id` | `(E,MAX_AGENTS)` | int64 | per-agent instruction row |
| `instr_class` | `(E,)` | int64 | semantic class (scoring only, never observed) |
| `lang_vec` | `(E,MAX_AGENTS,32)` | float32 | projected instruction vector, zeroed in blind arms |

Constants: `_deltas = [[-1,0],[1,0],[0,-1],[0,1],[0,0]]` (bit-identical to `RendezvousEnv._MOVES`), `_reveal_offsets` = the `(2r+1)²` Chebyshev window, `r = cfg.lidar_radius = 1`.

### 1.4 Spaces and episode length

```python
@configclass
class TeamGridEnvCfg(DirectMARLEnvCfg):
    decimation = 1
    episode_length_s = 12.9                        # -> max_episode_length = 129 -> 128 decision steps
    possible_agents = ["robot_0", "robot_1"]       # STABLE ORDER — load-bearing
    action_spaces = {"robot_0": {5}, "robot_1": {5}}
    observation_spaces = {a: OBS_DIM for a in possible_agents}
    state_space = STATE_DIM                        # positive int
    action_noise_model = None                      # MUST stay None
    sim = SimulationCfg(dt=0.1, render_interval=1, physx=PhysxCfg(<GPU buffers minimised>))
    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0,
                                replicate_physics=True, filter_collisions=False)
```

- `{5}` (a set of length one) is what `spec_to_gym_space` maps to `Discrete(5)`. `[{5},{5}]` gives `MultiDiscrete`.
- `state_space` is a **positive int**. `0` is a documented trap (`Box(shape=(0,))` while `state()` returns `None`); `-1` would auto-concatenate agent observations, duplicating the language slice in the critic and making blindness implicit rather than auditable.
- `action_noise_model = None`: noise is applied before `_pre_physics_step` and would turn an integer action index into float garbage.
- `is_finite_horizon = False` + `time_limit_bootstrap: True`.

`[FIXED: episode-length off-by-one.]` Isaac Lab increments `episode_length_buf` **before** `_get_dones()`, so the shipped `>= max_episode_length - 1` idiom yields `max_episode_length - 1` decision steps. `episode_length_s = 12.9` ⇒ `max_episode_length = 129` ⇒ exactly **128 decision steps**, matching `T_DECISION` used in every `t/T` normalisation and in the reward arithmetic. `__init__` asserts `self.max_episode_length == T_DECISION + 1`, and `tests/test_episode_length.py` rolls a stay-only policy and asserts truncation lands on step 128 exactly. Any change to `sim.dt` or `decimation` crashes rather than silently rescaling every cross-condition comparison.

### 1.5 Observation layout — `OBS_DIM = 376`

Named slice constants live once in `tasks/team_listen/obs_layout.py`; the blind ablation is a **slice zeroing, never a shape change**.

| Slice | W | Content | Norm |
|---|---|---|---|
| `[0:144)` | 144 | `known_free` plane | {0,1} |
| `[144:288)` | 144 | `known_obs` plane | {0,1} |
| `[288:300)` | 12 | ego block: (r,c), latched, latch-slot one-hot(4), latch-time/T, agent-id one-hot(4) | [−1,1]/{0,1} |
| `[300:327)` | 27 | 3 × teammate block (9): present, (r,c), latched, latch-slot one-hot(4), latch-time/T |
| `[327:342)` | 15 | 3 × target block (5): present, (r,c), occupied, occupied-by-ego |
| `[342:344)` | 2 | `t/T`, time-since-first-latch/T |
| `[344:376)` | **32** | `LANG_SLICE` — this agent's projected instruction vector |

Positions use `2x/(R−1) − 1 ∈ [−1,1]`. Flat `Box`, never a gym `Dict` (Dict breaks `state_space=-1` and `multi_agent_to_single_agent`). No `Discrete` inside the observation (`compute_space_size(Discrete, occupied_size=True) == 1`, so skrl would feed the raw category index to the MLP unembedded).

`[FIXED: taxonomy cell stated explicitly.]` The M1 `Lang` arm writes the **same** vector into every agent's slot: it occupies the *global-broadcast × sentence-embedding* cell of the four-way interface taxonomy. Per-agent atomic prompting is `cfg.per_agent_instruction = True` plus a different `instr_id` per slot — no dimension change.

`[FIXED: mid-episode reassignment foreclosed.]` The instruction is per-episode constant in M1, but the bank carries `instr_switch_time (K,) int16 = -1` and `_pre_physics_step` contains the (M1-inert) switch branch, so month-4 reassignment is a data change.

**Permutation invariance.** Entity blocks are padded to capacity so `OBS_DIM` never changes again. M1 trains with the stock skrl MLP over the flat vector; `harness/models.py` ships a DeepSets-style entity encoder over the same slices, unit-tested and exercised in one smoke run, and becomes the default from month 4 (needed for the held-out-team-size test).

### 1.6 Critic state — `STATE_DIM = 641`, `_get_states()` hand-written

`[0:144)` true obstacle plane (privileged) · `[144:288)` `known_free` · `[288:432)` `known_obs` · `[432:484)` 4 × agent block(13) · `[484:511)` 3 × target block(9, incl. occupier one-hot) · `[511:513)` `t/T`, first-latch time · `[513:641)` **`STATE_LANG_SLICE` = 4 × 32**, per-agent, zeroed in every blind arm.

The state carries language slots so blind and language arms have byte-identical critic input shapes and identical preprocessor behaviour; a critic that sees the instruction while the actor does not is asymmetric actor-critic, not an instruction-blind oracle.

`[FIXED: arm consistency was gated on `cfg.debug_asserts`, which is off for every throughput run.]` `harness/arms.py::assert_arm_consistency` is now **unconditional** — one reduction per reset, free against ~45 kernels/step — asserting `‖LANG_SLICE‖ = ‖STATE_LANG_SLICE‖ = 0` for `Blind`/`Leaky`/`Mute` and `> 0` for `Lang`/`Symbol`/`SymbolPO`/`Placebo`. `instruction_in_obs` / `instruction_in_state` are **no longer independently settable booleans**; both are derived from the arm name in `config/arms.yaml`, so the illegal combinations are unrepresentable. `lang_slice_l2` and `state_lang_slice_l2` are written into every parquet row (§6.6), so a "the team does not listen" headline can be re-verified post hoc without re-running the simulator.

`_get_states()` is implemented even when unused (`@abstractmethod`).

### 1.7 Discrete action decode

```python
def _pre_physics_step(self, actions):
    a = torch.stack([actions[k].reshape(-1).long() for k in self.cfg.possible_agents], 1)  # (E,2)
    a = grid_core.apply_slip(a, self._slip_row())      # bank-indexed, §1.10
    self._act = a; self._advance(a)
def _apply_action(self): pass
```

`WHY .reshape(-1).long():` skrl's `unflatten_tensorized_space(Discrete, x)` returns `(num_envs, 1)` int32/int64, not `(num_envs,)`; an unsqueezed index broadcasts `_deltas` to `(E,1,2)`. `DirectMARLEnv.step()` performs no dtype coercion and no clipping.

### 1.8 Conflict resolution — closed form, N=2

At N=2 the reference `while changed` fixpoint provably terminates in one pass (cases (a) and (b) revert both robots; case (c) reverts only the mover, whose counterpart is already stationary):

```python
delta   = self._deltas[act]
tgt     = (cur + delta).clamp_(lo, hi)                       # off-grid -> stay, no flags (== np.clip)
hit_obs = self.occ[env_idx, tgt[...,0], tgt[...,1]].bool()
tgt     = torch.where(hit_obs.unsqueeze(-1), cur, tgt)
tgt     = torch.where(latched.unsqueeze(-1), cur, tgt)       # latched robots immovable
mov     = (tgt != cur).any(-1)
same  = (tgt[:,0]==tgt[:,1]).all(-1) & mov[:,0] & mov[:,1]
swap  = (tgt[:,0]==cur[:,1]).all(-1) & (tgt[:,1]==cur[:,0]).all(-1) & mov[:,0] & mov[:,1]
into0 = (tgt[:,1]==cur[:,0]).all(-1) & mov[:,1] & ~mov[:,0]
into1 = (tgt[:,0]==cur[:,1]).all(-1) & mov[:,0] & ~mov[:,1]
rev0, rev1 = same|swap|into1, same|swap|into0
nxt        = torch.where(torch.stack([rev0,rev1],1).unsqueeze(-1), cur, tgt)
hit_robot  = torch.stack([rev0, rev1], 1)
```

Verified against the reference tests I read: `test_same_target_conflict` (both flagged), `test_swap_conflict` (both flagged), `test_move_into_stationary` (**only the mover**), `test_follow_vacated_cell_allowed` (**no mask fires — convoy motion is legal**), `test_obstacle_collision` (pre-pass revert, no robot flag). `test_revert_cascade` is N=4; `step_positions` asserts `n_agents == 2` — `OPEN(1)`.

`[FIXED: the precedence story contradicted this rule.]` Because convoy motion is legal, a width-1 corridor does **not** serialise; it only enforces a one-cell offset. §3.1 is re-engineered accordingly.

~22 kernel launches. Kernel count, not FLOPs, is the currency (Isaac Lab's own workflow-overhead statement).

### 1.9 Fog-of-war reveal

`lidar_radius = 1` in `env_paper.py` is a Chebyshev **window slice**, not a raycast. The port is a scatter — no warp mesh, no `RayCaster`, no scene query, `enable_scene_query_support` left `False`:

```python
cells = (pos.unsqueeze(2) + offsets).clamp_(lo, hi)      # (E,N,9,2)
idx   = (cells[...,0]*C + cells[...,1]).reshape(E, -1)
vals  = self.occ.view(E,-1).gather(1, idx)
self.known_free.view(E,-1).scatter_(1, idx, vals == 0)
self.known_obs .view(E,-1).scatter_(1, idx, vals == 1)
```

Clamping is equivalent to the reference's `max(0,·)/min(R,·)` slicing. `RayCaster` raycasts static USD meshes only (killing per-episode map randomisation); the PhysX `raycast_closest` loop in `reference/isaac_env.py` is 72 beams × N × E Python round-trips and stays confined to the month-8 cross-fidelity arm.

### 1.10 Scenario bank, frozen seed, and bank-indexed stochasticity

**No RNG in `_reset_idx`.** It is a pure gather from `data/scenario_bank_{variant}_{sha}.pt`:

```
occ           (K,R,C)                  uint8
spawn         (K,MAX_AGENTS,2)         int16
spawn_alt     (K,MAX_AGENTS,2)         int16   # difficulty-matched nuisance intervention
target        (K,MAX_TARGETS,2)        int16
target_valid  (K,MAX_TARGETS)          bool
dist_field    (K,MAX_TARGETS,R,C)      int16   # LATCH-AWARE BFS
mouth         (K,2)                    int16   # precedence corridor mouth, (-1,-1) for role binding
delta_gap     (K,)                     int8    # d(r0,mouth) - d(r1,mouth)
leak_bit      (K,)                     uint8   # sign of the geometric default; see §4.1
instr_switch_time (K,)                 int16   # -1 in M1
slip          (K, N_STREAMS=2, T=128, MAX_AGENTS) uint8   # 5 = no slip, else forced action
split         (K,)                     uint8
```

K = 16384 (14336 train / 2048 eval). Resident: `dist_field` ≈ 14.2 MB, `slip` ≈ 16.8 MB, `occ` ≈ 2.4 MB — ~35 MB on GPU.

`[FIXED: latch-aware distance field.]` `dist_field[k,j]` is BFS from target *j* over free cells with **every other target cell treated as an obstacle**. The all-free field routed shaping paths *through* the other absorbing station (§2.1's same-row layout guaranteed it), so Φ actively rewarded walking into the wrong station and locking `Y = 0`. Combined with the alcove topology of §2.1/§3.1 (targets are degree-1 leaves), the latch-aware and all-free fields coincide, and `tests/test_bank_distfields.py` asserts that equality as a cheap topology invariant.

`[FIXED: there was no physics seed at all.]` The zero-asset env is fully deterministic with deterministic-argmax eval, so "divergence under a seed change alone" — CSI's defining denominator and the project's stated novelty — had **no referent** in M1, and `CSI_traj`'s only available denominator (policy entropy) would have ranked policies by their entropy coefficient. A per-step **action slip** is now part of the MDP: with probability ε the executed action is replaced by a uniformly drawn action. All slip draws are precomputed into `bank.slip[k, stream, t, n]`, so the env stays exactly reproducible *and* a genuine seed intervention exists (`slip_stream: 0 → 1`, everything else frozen). ε is pre-registered at 0.05 with a sensitivity check at {0.0, 0.05, 0.10}. Because slips are keyed by `(scenario_id, slip_stream, t)`, two lanes of a counterfactual pair receive **identical** slips, so the frozen-physics guarantee is unaffected.

```python
def _reset_idx(self, env_ids):
    if env_ids is None: env_ids = self._ALL
    super()._reset_idx(env_ids)
    sid = self._draw_scenarios(env_ids)                 # train: randint over split==0; eval: manifest cursor
    self.scenario_id[env_ids] = sid
    self.occ[env_ids] = bank.occ[sid]; ...              # pure gathers
    self.instr_id[env_ids], self.instr_class[env_ids] = \
        self._draw_instructions(env_ids, bank.leak_bit[sid], self.cfg.leak_rho)
    self.lang_vec[env_ids] = LANG_TABLE[self.instr_id[env_ids]] * ARM.lang_gain
    self.known_free[env_ids] = False; self.known_obs[env_ids] = False
    reveal(...)                                         # initial reveal at spawn, as in reference reset()
    assert_arm_consistency(self, env_ids)               # unconditional
```

`WHY a bank instead of RNG:` `env.reset(seed=k)` cannot deliver "frozen physics seed under counterfactual instruction" — `DirectMARLEnv.seed` is a `@staticmethod` seeding the *global* RNG, and any `sample_uniform` inside `_reset_idx` couples every env's layout to the whole draw history. "Same physics" is now literally `scenario_id[i] == scenario_id[j] and slip_stream[i] == slip_stream[j]`, an equality assertion rather than trust in Isaac Lab's conditionally-qualified determinism. It also removes MapGen's rejection sampling and Python DFS flood-fill from the hot path.

`[FIXED: the "guards" made the canary unimplementable and the independence test vacuous.]` `_draw_instructions` now takes an explicit `leak_bit` and `leak_rho`, drawing `I = leak_bit` with probability ρ and uniform otherwise. `cfg.leak_rho` defaults to 0.0 and is settable **only** by the `Leaky` arm registry entry. This keeps the canary on the *audited* code path (the previous design required a separate unguarded implementation, so the canary validated a different program than the one under audit). The static "signature has no scenario argument" check is replaced by a measurable invariant: at ρ = 0, mutual information between `instr_class` and `(scenario_id, leak_bit, spawn, delta_gap)` is zero over 10⁴ resets; at ρ = 0.8 the measured coupling is ≈0.8. The import-graph test (`scenario_bank.py` must not import `lang_cache.py`) survives intact, since the bank stores only a geometric scalar.

### 1.11 Termination

**Latch rule.** A robot whose post-conflict cell equals a target cell latches: `latched=True`, `latch_slot=j`, `latch_time=t`. A latched robot's action is overwritten to "stay" and it remains a blocking obstacle. Because §2.1/§3.1 make every target a **degree-1 alcove**, a robot can never accidentally pass *through* a station, and a latched robot can never block the other station's approach.

```python
def _get_dones(self):
    done      = self.latched[:, :N].all(dim=1)
    time_out  = self.episode_length_buf >= self.max_episode_length - 1
    return ({a: done for a in P}, {a: time_out for a in P})
```

`[FIXED: dead `gap` code removed.]` The draft computed a `gap` mask in the precedence branch and discarded it, reading as an unfinished rule that the next editor would "fix", silently changing the outcome definition mid-study. With the §3.1 unique-mouth topology, `|Δt| ≥ 1` is **structural** (both targets' only free neighbour is the mouth, which holds at most one robot), so ties are impossible and no gap condition is needed. `G = 1` and `debug_asserts` verifies `|Δt| ≥ 1` on every terminal.

One shared tensor broadcast to both agents: `step()` computes `reset_buf = prod(terminated) | prod(time_out)`, an elementwise **AND** across agents, so per-agent termination would leave envs running to timeout. `_get_observations()` always emits both agents (`self.agents` is recomputed as `[a for a in possible_agents if a in obs_dict]`; a missing key would silently change the state layout mid-episode).

### 1.12 Reward

Per-team scalar broadcast to both agents, plus a per-agent collision term.

| Term | Value |
|---|---|
| step cost | −0.01/step |
| obstacle collision | −0.25 per robot per step |
| robot–robot collision | −0.25 per robot per step |
| shaping | `λ(γΦ_t − Φ_{t−1})`, λ = 0.1, γ = 0.99, `Φ = −min-cost perfect matching of latch-aware BFS distances`, `Φ(terminal) ≡ 0` |
| completion | +2.0 on the terminal both-latched step |
| outcome bonus | `+10·Y` (§2.3/§3.3) — the only instruction-dependent term |

`[FIXED: the shaping was not policy-invariant.]` The draft used `λ(Φ_t − Φ_{t−1})` and called it "Ng-style". At γ = 0.99 the invariant form is `λ(γΦ_t − Φ_{t−1})`; the undiscounted difference adds `λ(1−γ)Φ_t` per step, a genuine penalty proportional to remaining distance — i.e. exactly the anti-waiting confound §3.4 promises to have excluded.

`[FIXED: the reward-fairness bound was double-counted and hand-waved.]` Under the correct potential form the shaping telescopes to `λ(γ^T Φ_T − Φ_0)` and is path-independent given identical terminal Φ, so the old `λ·Δd_extra` term is ≈0 and the "+5.0 left on the table" figure was undiscounted. `harness/reward_audit.py` now computes the bound **from the bank, exactly**: for every row and every instruction class it solves the optimal compliant and optimal non-compliant plans by BFS under the true reward and γ, and asserts `min_k (G_comply − G_defect) > margin`. That number, not a hand estimate, is what the paper quotes.

`[FIXED: the blind oracle was trained against an unpredictable ±10 term.]` For `Blind` and `Mute` the outcome bonus is replaced by its **exact conditional expectation** `+5.0` on the both-latched step. For an instruction-blind policy `E[10·Y | C=1] = 5.0` (§2.4), so this is bias-free and preserves the return scale, while removing a variance-25 terminal term the policy structurally cannot predict — which would otherwise inflate the value-loss floor (`value_loss_scale: 2.0`) and perturb `KLAdaptiveLR`, risking the competence clause for reasons unrelated to task design. Every other arm — including the whole `Leaky` ρ-sweep, whose ρ = 0 cell is the matched internal control — keeps the real stochastic bonus, so the canary is never confounded by reward form.

`WHY Φ is instruction-free:` an instruction-conditioned potential is policy-invariant only in the MDP where the instruction is in the state; for the blind oracle it degenerates into zero-mean high-variance noise. Keeping Φ instruction-free also *deliberately* encodes the geometric shortcut pressure the audit is about.

### 1.13 Visualisation without touching training FPS

`cfg.debug_vis = False`. When True, `_set_debug_vis_impl` creates **one** `VisualizationMarkers` (`UsdGeom.PointInstancer`, so all glyphs across all envs are a single prim) driven from `pos` + `scene.env_origins`.

`[FIXED: the render guard named the wrong mechanism.]` `--enable_cameras` does not itself set `/isaaclab/render/rtx_sensors` — that carb flag is set when a `Camera`/`TiledCamera` is *instantiated*, so an `__init__`-time `has_rtx_sensors()` assert cannot catch a sensor created later, and the actual leak path in the shipped `train.py` is `--video` (which sets `enable_cameras=True`, creates the env with `render_mode="rgb_array"`, and lets `RecordVideo` call `env.render()` → `sim.render()`). The guards are therefore: (1) `scripts/train.py` **refuses** `--video` and `--enable_cameras` outright and forces `cfg.debug_vis=False`; (2) `__init__` asserts `cfg.sim.render_mode is None` and `render_mode is None` in training mode; (3) a per-step assert under `debug_asserts` that `not (sim.has_gui() or sim.has_rtx_sensors())`; (4) `bench_env.py` asserts the same empirically. A single stray camera otherwise costs a 20× throughput collapse with no error.

### 1.14 MAPPO YAML (`agents/skrl_mappo_cfg.yaml`)

`[FIXED: the YAML dropped `*_preprocessor_kwargs`, a live crash-or-misconfigure path.]` `Runner._generate_agent` injects preprocessor shape via `agent_cfg.get("<name>_preprocessor_kwargs", {}).update({"size":…, "device":…})`. On an **absent** key `.get` returns a throwaway dict and the injection is silently discarded; `_process_cfg` only rescues the key-present-but-`null` case — which is exactly why every shipped Isaac Lab config carries the explicit `null`s. **Procedure:** copy `source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/agents/skrl_mappo_cfg.yaml` from the pinned checkout **verbatim** and edit only the intended deviations. `tests/test_yaml_keyset.py` diffs the project YAML against the checkout's and asserts the key **set** is identical, only values differ — the machine-checked form of §0's discipline.

```yaml
seed: 42
models:
  separate: True
  policy: {class: CategoricalMixin, unnormalized_log_prob: True,
           network: [{name: net, input: OBSERVATIONS, layers: [512,256,128], activations: elu}],
           output: ACTIONS}
  value:  {class: DeterministicMixin, clip_actions: False,
           network: [{name: net, input: STATES,       layers: [512,256,128], activations: elu}],
           output: ONE}
memory: {class: RandomMemory, memory_size: -1}
agent:
  class: MAPPO
  rollouts: 16
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95                       # correct on 1.4.x AND 2.x; Runner maps it to gae_lambda
  random_timesteps: 0
  learning_starts: 0
  learning_rate: 3.0e-04
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs: {kl_threshold: 0.008}
  observation_preprocessor: null
  observation_preprocessor_kwargs: null
  state_preprocessor: null
  state_preprocessor_kwargs: null
  value_preprocessor: RunningStandardScaler
  value_preprocessor_kwargs: null
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: True
  entropy_loss_scale: 0.008          # NOT the 0.0 default
  value_loss_scale: 2.0
  kl_threshold: 0.0
  rewards_shaper_scale: 1.0
  time_limit_bootstrap: True
  experiment: {directory: runs, experiment_name: "", write_interval: auto, checkpoint_interval: auto}
trainer: {class: SequentialTrainer, timesteps: <set by §8 gate>, environment_info: log}
```

Two deliberate deviations, both load-bearing: **`entropy_loss_scale: 0.008`** (the Isaac Lab default 0.0 is tuned for Gaussian continuous control; on a deliberately symmetric 5-way Categorical it collapses the policy to a fixed left/right convention early — the exact degeneracy the audit exists to detect, which we must not induce by an optimiser default); and **all observation/state preprocessors `null`** with features hand-normalised (the 32-d language slice takes ~4 distinct values ever and is constant within an episode; `RunningStandardScaler` dividing by `sqrt(var+1e-8)` on near-degenerate dimensions amplifies numerical noise into huge features). `value_preprocessor` is kept — it acts on returns, never on observations. `OPEN(3)`.

Registration (`tasks/team_listen/__init__.py`), one entry point, many cfgs, over `variant ∈ {RoleBinding, Precedence}` × `arm ∈ {Lang, Blind, Symbol, SymbolPO, Leaky, Mute, Placebo}`, each with `env_cfg_entry_point`, `skrl_mappo_cfg_entry_point` and `skrl_ippo_cfg_entry_point`. `train.py` resolves `f"skrl_{algorithm.lower()}_cfg_entry_point"`, so registering only the default key makes `--algorithm MAPPO` fail to find a config.

**Never `--algorithm PPO`.** `train.py` unconditionally applies `multi_agent_to_single_agent(env)` for PPO on a `DirectMARLEnv`, and that shim is broken for discrete spaces: `flatten_space(Tuple([Discrete(5),Discrete(5)])) = Box(shape=(10,))`, so the agent emits 10 floats where the env expects an integer index. `scripts/train.py` refuses it with an explanatory error.

`[FIXED: the MAPPO-vs-IPPO sanity check was vacuous where it was applied.]` For `Blind`/`Symbol`/`Leaky` the actor observes the full state, so actor and critic are informationally identical and MAPPO ≡ IPPO. The paired IPPO control is therefore run on the **partial-observation** arms (`Lang`, `SymbolPO`, `Mute`), where "if MAPPO does not beat IPPO the state vector is probably wrong" actually has content.

Diagnostics go through `self.extras["log"]["<name>"] = <scalar tensor>` with `trainer.environment_info: log` (`extras` is one plain dict shared across agents; non-scalars break the writer). Logged: shell-completion rate, ITT outcome shares, latch-gap histogram, **branch-step entropy** (§4.4), realised-assignment marginal, `lang_slice_l2`.

---

## 2. Task variant A — identical-target role binding

### 2.1 Scene layout

12×12 grid, obstacles from a scaled-down `MapGen` (`num_clusters=3`, `cluster_size_range=(2,6)`, `min_cluster_distance=3.0`), free space 4-connected by MapGen's flood fill (verified present in `reference/env_paper.py`).

`[FIXED: stations were pass-through cells in the same row, five cells apart.]` Both stations are now **degree-1 alcoves**: each station cell has exactly one free neighbour, and the two alcove mouths are ≥5 columns apart on opposite sides of the map. Consequences: (i) a robot can never accidentally latch while *passing*, so the absorbing latch stops being a trap; (ii) no shortest path to one station runs through the other, so `dist_field` is unambiguous and the reward-fairness bound is finite by construction; (iii) a latched robot blocks nothing. The bank builder **rejects** any scenario failing the leaf property, and rejects any row where the latch-aware distance from either spawn to either station is infinite.

Stations are **identical in every observable respect** — same target-block encoding (coordinates + occupancy only; no type/colour/id feature), same reward, same dynamics — distinguishable only by spatial relation, which is what makes "left"/"right" a referring expression.

Spawn cells drawn from free cells with latch-aware BFS distance to each station in [4,14]. The bank builder additionally randomises which physical alcove occupies target slot 0, randomises which robot index spawns where, and — as **defence-in-depth, not a correctness requirement** (§2.4) — rejects rows whose spawn pair has a station-distance asymmetry beyond ±1 **per scenario** (`[FIXED: the draft enforced this only "in aggregate across the bank", leaving per-scenario asymmetry that inflates seed variance and propagates straight into CSI's spawn denominator]`).

### 2.2 Instruction vocabulary

Classes `RB0` (robot_0 → left) and `RB1` (robot_0 → right), generated by a template grammar in `harness/templates.py` — never free LLM generation, so the split can be template-family-disjoint.

| Slot | Card | Examples |
|---|---|---|
| agent referring expression | 5 | "robot one", "the first robot", "unit 1", "R1", "the lead robot" |
| verb phrase | 4 | "takes", "goes to", "docks at", "should occupy" |
| station referring expression | **6** | "the left station", "the station on the left", "the westward dock", "the left-hand bay", "the port-side dock", "the leftmost berth" |
| sentential frame | 4 | `{A} {V} {S}, {B} {V} {S'}.` / `Send {A} to {S}; {B} to {S'}.` / `{S} is for {A} and {S'} is for {B}.` / `{A} — {S}. {B} — {S'}.` |

`[FIXED: 3 held-out families is too few to support any interval.]` A family = a (frame, station-ref) pair ⇒ **24 families, split 18 train / 6 held-out**. 50 train + 10 held-out sentences per class, sampled family-disjointly. Held-out statistics are clustered **by family (n = 6)**, not by sentence — the same design-effect error the design correctly diagnoses for agents-within-episode.

`[FIXED: the 1-bit channel forecloses the months 6–7 headline.]` The grammar also emits **composed** classes (role binding × precedence in one sentence: "robot one takes the left station and docks first"), 4 composed classes with one composition held out entirely. These are built into the cache and the bank schema **now** (cost: one grammar file and seconds of encoder time), trained from month 2. M1 trains only the single-axis variants; the residual concern that a 2-class channel carries exactly 1 bit is `OPEN(11)`.

### 2.3 Outcome definitions

`[FIXED: conditioning on competence is a collider.]` The headline is **intention-to-treat over all episodes** with a three-valued outcome `O ∈ {correct, wrong, incomplete}`:

- `C ∈ {0,1}`: both robots latched before timeout.
- `Y ∈ {0,1}`, defined when `C = 1`: `latch_slot[r0]` equals the instructed slot.
- **Primary estimand: `P(O = correct)` over all episodes.** `E[Y | C=1]` is reported as a clearly-labelled *secondary* conditional, interpretable for cross-arm comparison **only when the arms' C rates are statistically indistinguishable**, which `certificate.py` tests and gates. §2.4's independence proof establishes `C ⟂ I` for a *blind* policy only; for `Lang` the instruction affects C (an attempted-but-short yield scores C=0 and is dropped), so the conditional systematically discards a partially-grounded policy's near-misses while retaining essentially all of a geometry-rider's episodes.

`[FIXED: the stated clustering justification was inverted.]` One binary per episode, never per agent: with N=2 and distinct latches, if r0 holds its instructed station then r1 necessarily holds its instructed station too — the two agent-level bits are **logically identical**, not complements (the draft said complements). Either way ρ = 1 and the Kish design effect is 2 at m = 2, so counting two agent decisions per episode would shrink the CI by √2 with zero added information. Observed ICC and implied DEFF are computed and reported (`harness/stats.py::icc_deff`).

### 2.4 Why an instruction-blind policy is at chance — and what that does and does not buy

> **Claim.** Let π be any instruction-blind policy. Let `I` be the instruction class, drawn independently of the scenario at evaluation. Then `P(Y=1 | C=1) = 1/2` exactly.
> **Proof.** π's action distribution is a function of the observation only; the observation, the state, the initial conditions, the stored slip stream and the transition kernel contain no function of `I`. `I` enters only through the terminal reward, which affects training but not rollout. So the realised trajectory — hence `C` and the realised assignment `A` — is independent of `I` at evaluation. With `Y = 1[A matches I]` and `I ⟂ (A,C)`, `P(Y=1|C=1) = Σ_a P(A=a|C=1)·½ = ½`. ∎

`[FIXED: the milestone hung a 6-week go/no-go on a statement that is true by construction.]` Deliverable (c) is renamed **"instruction-leakage audit"**. The paper states plainly that chance-level blind accuracy is an *identity* under a correct implementation; the scientific content is the **sensitivity calibration** (the ρ-sweep, §4.1/§4.4) and the **supervised leak probe** (below), not the point estimate. The go/no-go is redefined in §4.4 around clauses that can actually fail.

`[FIXED: the load-bearing independence test could not fail.]` The draft's `test_instruction_independence` monkeypatched the bank to a *single constant scenario* and checked the instruction marginal was uniform — guaranteed with nothing to correlate against, and passing even for a maximally leaky bank. It is replaced by two instruments:

1. **Empirical independence test** (`tests/test_instruction_independence.py`): χ² and mutual information between `instr_class` and a feature vector of `(scenario_id, spawn cells, target cells, per-robot latch-aware distances, delta_gap, leak_bit, obstacle-map hash)` over the **full 16384-row bank** and over each eval manifest separately.
2. **Supervised leak probe** (`harness/probe.py`) — the audit's *primary* instrument. A small classifier predicts `instr_class` from (a) the blind arm's t=0 observation, (b) the full 641-d state, (c) the entire recorded trajectory (`realised_positions_blob`). Requirement: **held-out AUC 95% CI contains 0.5.** The probe is orders of magnitude more sample-efficient than MAPPO at extracting a weak coupling; the RL oracle is demoted to behavioural confirmation. (The draft's power was bounded not by n but by MAPPO's ability to *exploit* a leak: at ρ ≈ 0.55 the coupling is worth ~+0.5 reward against a 128-step, shaping-opposed, 1-bit terminal credit problem §3.4 itself predicts the policy will fail to solve even at ρ = 1.)

**Forbidden leaks and their guards:**

| Leak | Guard |
|---|---|
| Instruction drawn conditional on geometry | `cfg.leak_rho == 0.0` for every non-`Leaky` arm; MI test at ρ=0; run-manifest records ρ and the bank SHA |
| Curriculum/rejection sampling filtering scenarios by instruction-conditional solvability | bank built before any instruction exists; `tests/test_import_graph.py` forbids `scenario_bank.py` importing `lang_cache.py` |
| Instruction-derived feature in a blind observation/state | **unconditional** `assert_arm_consistency` (§1.6) + `lang_slice_l2` in every parquet row |
| Instruction-dependent shaping | `tests/test_potential_purity.py` (§1.2) + reward_audit's joint-involution check |
| Asymmetric reward between stations | `[FIXED]` `reward_audit.py` asserts invariance under the **joint** involution (station relabel ∧ instruction-class flip ∧ robot-index swap) on 10⁴ sampled trajectories. A bare station relabel is *not* an invariance — the +10 bonus provably breaks it — so the draft's assertion had to fail or be silently weakened |
| Manifest imbalance | §4.3's paired design makes balance structural; the secondary manifest is asserted balanced **jointly** over `I × sign(geometric default)` |

---

## 3. Task variant B — precedence

### 3.1 Scene layout — unique-mouth airlock

`[FIXED: two independent structural defects.]` The draft specified a width-1 corridor into a bay "2 free cells wide" without fixing the topology. (i) If the two bay cells were collinear with the corridor, the first robot latches on the only route to the second cell and competence is **identically zero** — MapGen's flood fill certifies connectivity over free cells and is blind to latch semantics. (ii) The claim "the conflict rule makes simultaneous corridor occupancy impossible" is **false**: `reference/test_env_paper.py::test_follow_vacated_cell_allowed` (robots at (3,2),(3,3) both move right, both succeed, zero collisions) shows convoy motion is legal, so a width-1 corridor enforces only a one-cell offset and realised gaps cluster at |Δt| ∈ {0,1}, below the draft's G = 2.

**Final topology.** A width-1 corridor of length ≥3 leads to a single **mouth cell** `m`. The two station cells are the *only* other free neighbours of `m`, and each is degree-1 (a leaf). Therefore:

- `m` holds at most one robot, and every latch must transit `m` ⇒ **|Δt| ≥ 1 structurally, ties are impossible**, so `G = 1` and competence needs no learned delay skill (the draft's G = 2 turned C into a learned-waiting skill and put the 0.15 tie cap out of reach).
- Neither latched robot can block the other's approach ⇒ **both orderings are always feasible** (the draft never checked this; asymmetric feasibility would have capped `E[Y|C=1]` and inflated `1−E[C]` for reasons unrelated to language).
- The serialisation story survives intact: the mouth is a hard one-at-a-time gate, so reactive collision avoidance still produces a strict, geometry-determined ordering for free — which is precisely the effect the audit exists to expose.

Bank builder assertions (`tests/test_bank_latch_reachability.py`): stations are leaves attached to `m`; `m` is reachable from every spawn; corridor length ≥3; latch-aware BFS from `m` with the *other* station blocked reaches both stations (the check MapGen's flood fill does not do). Spawn cells are outside the corridor with `d(spawn, m) ≥ 3` `[FIXED: the draft constrained only Δ, permitting a spawn inside the bay that latches at t=0 and makes the ordering degenerate]`.

Δ = `d(r0,m) − d(r1,m)` is drawn uniformly from {0,±1,…,±6}, sign-symmetric, stored as `delta_gap`. It is the **yield cost**: compliance against geometry requires waiting ≈|Δ| steps, so stratifying by Δ turns the headline into a dose–response curve, and Δ = 0 is the cleanest cell.

**Pre-training smoke number:** `harness/planners.py`'s geometric (greedy) scripted controller is rolled over the whole bank *before any training*, and its realised `E[C]` and |Δt| histogram are recorded. If `E[C]` is not near 1 the layout, not the policy, is wrong.

### 3.2 Instruction vocabulary

Classes `PR0` (robot_0 first) / `PR1` (robot_1 first). Same 5 × 4 × 6 × 4 grammar with the station-ref slot replaced by an ordering-connective slot ("first, then", "before", "only after … has docked", "wait until … is docked, then", "… precedes …", "let … go ahead of …"), same 24 families split 18/6, same 50 + 10 per class. Which station each robot ends at is not scored.

`[FIXED: minimal pairs were not comparable across variants.]` The draft flipped a *station* referring expression in role binding and an *agent* referring expression in precedence — different edit types, systematically different embedding deltas, a third confound on top of map and horizon in the cross-variant dissociation. Both variants now ship **both** flip types (role binding: flip the station ref *or* the agent ref; precedence: flip the agent ref *or* the connective polarity), and CSI is stratified by flip type.

### 3.3 Outcome definitions

- `C ∈ {0,1}`: both robots latched before timeout (|Δt| ≥ 1 is structural).
- `Y ∈ {0,1}` when `C=1`: `sign(latch_time[r1] − latch_time[r0])` matches the instruction.
- Primary estimand is the same ITT three-valued outcome as §2.3, with the realised |Δt| distribution reported. Pre-registered cap `1 − E[C] ≤ 0.15`.

### 3.4 Why collision avoidance fakes precedence

**Mechanism.** The unique mouth physically forbids simultaneous docking, so *any* competent policy — grounded or not — produces a strict ordering. A policy that never reads the instruction therefore achieves aggregate success ≈100%, a *consistent, geometry-determined* order (smaller `d(·,m)` first), and order-correctness at exactly 50%.

**Consequence.** Reporting only aggregate team success — what every current multi-robot VLA/VLM result does — makes this policy indistinguishable from a perfectly grounded one. The headline is precise and falsifiable: **a team can be provably language-blind on the ordering while scoring at ceiling on aggregate success.**

**Why compliance might not be learned even though it is rewarded.** The signal is 1 bit delivered ~30–60 steps after the decisive yield; the geometric default already collects it on half the episodes, diluting the advantage estimate. Compliance is worth ≈+5 in expectation against a yield cost `reward_audit.py` computes exactly from the bank (§1.12), so it is unambiguously optimal, but it is long-horizon, low-frequency 1-bit credit assignment.

`[FIXED: the dissociation inference rested on a two-variable confound.]` The draft's positive control `Symbol` observed the **full state including the privileged obstacle plane**, while `Lang` observed 336-d partial observations under fog — two variables changed at once, so "Symbol ≥ 0.90 and Lang ≈ 0.5 ⇒ grounding failure" did not follow (a partial-observability or exploration failure produces the identical pattern). The gating control is now **`SymbolPO`**: byte-identical observation space to `Lang` (376-d, fogged), with a 2-code orthonormal symbol in `LANG_SLICE`. Restated inference: `SymbolPO ≥ 0.90` and `Lang ≈ 0.5` ⇒ grounding/encoder failure; `SymbolPO ≈ 0.5` too ⇒ partial-observability or credit-assignment failure and grounding is untestable at this observation space — itself a reportable result that triggers `OPEN(6)` (`lidar_radius = 12`) as the remedy rather than an encoder ablation. Full-state `Symbol` is retained as the task-solvability ceiling.

The reward-artefact objection is answered by the triple: (i) the exact bank-computed fairness bound, (ii) `Y` vs Δ (grounded ⇒ near-flat in Δ; geometry-riding ⇒ ≈1 with the instructed order and ≈0 against), (iii) `SymbolPO`.

---

## 4. Instruction-leakage audit protocol

### 4.1 Arms

All arms share one env class, one reward family, one bank (except `Leaky`), one manifest. `Blind`/`Symbol`/`Leaky` are byte-identical networks differing only in `LANG_SLICE` content.

| Arm | Actor obs | `LANG_SLICE` | Role | Seeds |
|---|---|---|---|---|
| `Blind` | full state + agent-id (645) | zeros | leakage oracle: competence, non-degeneracy, behavioural confirmation | 10 |
| `Symbol` | full state + agent-id (645) | 2 orthonormal codes | task-solvability ceiling | 5 |
| **`SymbolPO`** | **partial (376)** | 2 orthonormal codes | **gating positive control (clause 2)** `[FIXED]` | 10 |
| `Leaky` | full state + agent-id (645) | zeros | **canary sweep** ρ ∈ {0.0, 0.55, 0.60, 0.70, 0.80} | 3 each, RoleBinding only |
| `Mute` | partial (376) | zeros | language-free floor — **scored on the ITT outcome**, the only chance-level baseline at `Lang`'s own observation space `[FIXED]` | 5 |
| `Lang` | partial (376) | cached projected MiniLM | policy of interest | 10 |
| **`Placebo`** | partial (376) | task-irrelevant 32-d vector from the same cache (randomly relabelled instruction id) | **CSI normaliser calibration** `[FIXED]` | 3 |
| `CompliantPlanner` / `GreedyPlanner` | scripted over `grid_core` | n/a | **CSI ceiling / floor**, no training `[FIXED]` | — |

`WHY the blind oracle is full-state but still MAPPO:` `multi_agent_to_single_agent(env, state_as_observation=True)` is the natural-looking construction and is broken for discrete actions (§1.14). We keep the MARL interface and set `cfg.actor_observes_state = True`.

`WHY `Symbol` uses a 2-d code inside the same 32-d slot:` identical parameter count, first-layer fan-in and preprocessor behaviour, so "at chance" cannot be dismissed as a capacity artefact.

`[FIXED: the canary was one point at ρ = 0.8, ~6× the equivalence margin, and bypassed the very guards it was meant to validate.]` It is now a **sweep** implemented through `cfg.leak_rho` on the audited `_draw_instructions` path (§1.10), with leaky banks produced only by `scripts/build_leaky_bank.py` under a distinct SHA that the production loader **refuses** for any non-`Leaky` arm (`tests/test_leaky_bank_refusal.py`). Reported output: the **smallest ρ at which the decision rule rejects** — the audit's empirical detection threshold, compared directly against δ = 0.05 — and the probe's threshold at every ρ, which is the number justifying making the probe primary. ρ = 0.0 is the matched internal control (same code path, same reward form, only ρ varies).

### 4.2 Training budget and convergence definition

| Parameter | Value |
|---|---|
| `num_envs` (training) | selected by the §8.2 sweep, then frozen for the milestone |
| transitions/update | `rollouts(16) × num_envs` |
| total env steps | 150 M per run, **subject to the §8 training-FPS gate** (documented reduction ladder if the gate binds) |
| seeds | 10 for `Blind`, `Lang`, `SymbolPO`; 5 for `Symbol`, `Mute`; 3 for `Placebo` and each `Leaky` ρ cell |
| checkpoints | every 10 M steps, all retained; audit evaluated at the final and 3 earlier checkpoints |
| λ-sensitivity | λ ∈ {0.05, 0.1, 0.2} × 3 seeds on RoleBinding, **run before the headline** `[FIXED: the draft budgeted this as a post-hoc fallback, but λ co-determines compliance and CSI]` |

**"Trained to convergence" is operationalised:** (i) completion rate has a plateau (slope of a linear fit over the last 30 M steps < 0.01/100 M) and (ii) `[FIXED]` the entropy floor is applied at the **branch step** — the last step at which the assignment is still reversible (first entry into the mouth cell for precedence; the last step from which both alcoves remain reachable for role binding) — not to the episode mean. A policy can carry high entropy over ~120 irrelevant approach steps, be fully deterministic at the one decisive state, and still clear an episode-mean 0.25-nat floor. Floor: 0.25 nats at the branch step.

### 4.3 Evaluation manifests

`[FIXED: the frozen one-instruction-per-scenario manifest made the blind oracle's accuracy a fixed function of its geometric default.]` With 2000 distinct scenarios each carrying **one** instruction by a fixed permutation, deterministic argmax and a deterministic env, blind accuracy is not a random variable — it is the empirical agreement rate between `sign(Δ)` (or `sign(d(r0,s_L) − d(r1,s_L))`) and the frozen instruction vector. Only the *marginal* 1000/1000 balance was asserted, never the joint; a Δ-stratified bank written in index order could put that agreement rate at 0.7 and fail the gate for a pure bookkeeping reason, with a tight CI and no diagnostic.

**Primary manifest — within-scenario paired.** 2000 base eval scenarios drawn once from `split == 1` and frozen in `config/eval_manifest_{variant}.pt`; **each is run under both instruction classes** (4000 episodes), reusing the §6.3 two-lane machinery. For any blind arm the two lanes receive byte-identical observations on a byte-identical env with identical stored slips, so trajectories are identical, C is identical, and exactly one lane of each competent pair scores `Y = 1`:

> **`E[Y | C=1] ≡ 0.5` and `P(O = correct) ≡ E[C]/2`, exactly, by construction.**

Clauses 1–2 therefore become an **algebraic identity plus a machine check**, not a noisy statistical test: `harness/rollout.py` asserts `torch.equal` on the two lanes' full recorded trajectory tensors for every blind-arm pair. Any nonzero deviation is direct proof of an observation-channel leak, with zero sampling noise.

**Secondary manifest — natural (unpaired).** One instruction per scenario, drawn independently at manifest-build time with a recorded seed, **asserted balanced jointly over `I × sign(geometric default)`** with Δ = 0 as its own stratum. This is the manifest on which a *training-distribution* leak can express itself (the paired design necessarily neutralises it, since the blind policy's assignment is the same in both lanes) and on which the ρ-canary is measured. Both manifests are reported.

Common to both: `clustering_unit: base_scenario`; the same (scenario, instruction, slip-stream) tuples across **every arm and seed**, so the study is fully paired; held-out template families for `Lang` at eval with the seen-template score alongside; actions by **deterministic argmax over the Categorical logits** via `harness/rollout.py` — never `play.py`'s `outputs[-1][a].get("mean_actions", outputs[0][a])` idiom, in which `"mean_actions"` is a Gaussian key absent for Categorical, silently returning the *sampled* action; rollouts drive `env.unwrapped` directly (`IsaacLabMultiAgentWrapper.reset()` has a `_reset_once` flag and returns cached observations on repeat calls).

`[FIXED: argmax tie-break is a structural bias.]` `torch.argmax` breaks ties to the lowest index (0 = up). On a deliberately symmetric task with a symmetric net, near-exact logit ties at the branch state are plausible. The **top-2 logit margin at the branch step** is recorded per episode so tie-driven episodes are identifiable post hoc, and the tie-break rule is stated in the paper.

### 4.4 Decision rule

`[FIXED: the interval was chosen because it made the claim easier to establish.]` The draft justified Wilson over Clopper–Pearson on the grounds that an over-wide CI "makes the certificate harder to pass" — selecting an estimator to favour the pre-registered conclusion, and the first thing a reviewer quotes. That rationale is deleted. The equivalence test is stated explicitly as **TOST at α = 0.05 (⇔ a 90% CI inside [0.45, 0.55])**, justified by coverage properties only. Wilson intervals and the per-seed forest plot are **display-only**, captioned to note that with a frozen manifest and argmax actions a per-seed accuracy is an exact finite-population count with no binomial sampling model.

`[FIXED: the primary interval structurally excluded the dominant error term.]` A percentile bootstrap over 10 *seed-level* accuracies cannot extend beyond [min, max] of the observed seeds; since all seeds share one manifest and one deterministic env and are predicted to converge to the same convention, its width collapses toward zero — anti-conservative for an equivalence claim — and it cannot in principle capture the manifest-permutation error, a fixed offset common to all seeds. **Primary interval: a two-level hierarchical bootstrap resampling base scenarios *and* seeds with replacement in the same replicate** (10,000 percentile replicates). A seed-level-only interval is pre-registered as inadmissible; the between-seed SD is reported alongside. `[FIXED: no power analysis existed.]` `harness/stats.py` simulates the rule at true accuracies {0.50, 0.53, 0.55, 0.60} under the realised between-seed SD and ships the detection probability, so the audit has a stated Type-II profile. The draft's "n = 2000 → ±0.0219" arithmetic is episode-level and does not describe the pre-registered estimator; note also that Y is scored only on C = 1, so at the competence floor the scored n is 1800 (±0.0231), and the **scored subset's** balance — not just the manifest's — is what matters (structural under the paired design).

**Pre-registered (`config/preregistration.yaml`, committed before any training run):**

```yaml
equivalence_test: TOST
alpha: 0.05                       # <=> 90% CI inside [0.45, 0.55]
equivalence_margin_delta: 0.05
manifest: paired_within_scenario  # 2000 base scenarios x 2 instruction classes
n_base_scenarios: 2000
clustering_unit: base_scenario
primary_interval: hierarchical_bootstrap_scenarios_x_seeds
bootstrap_replicates: 10000
bootstrap_method: percentile      # not BCa, per rliable's few-run guidance
n_training_seeds_blind: 10
slip_epsilon: 0.05
primary_outcome: itt_three_valued
```

**GO/NO-GO (redefined so that the things that can actually fail are the things being decided):**

1. **Leakage audit passes.** (a) paired-lane trajectory equality holds for every blind-arm pair (machine check); (b) probe held-out AUC 95% CI contains 0.5 on t=0 observation, full state and trajectory; (c) bank/manifest MI test at ρ=0 is null; (d) on the secondary natural manifest, the TOST equivalence test on `Blind` passes; (e) **non-degeneracy** `[FIXED]`: the realised-assignment marginal `P(A = r0→left)` (precedence: `P(r0 first)`) lies in [0.35, 0.65] for **every** seed, and the branch-step entropy floor holds. Outside that range the audit is declared **UNINFORMATIVE, not PASS**, and the run repeats with a higher entropy coefficient — a constant-assignment oracle scores exactly 0.5 while being blind to *any* leak, which is the exact regime the entropy default drives the policy into.
2. **`SymbolPO` ≥ 0.90 with hierarchical-bootstrap CI lower bound > 0.90**, on both variants. `[FIXED: the draft paired a strong point bar with a vacuous interval bar ("CI excluding 0.55") and never named the clustering level.]` This establishes that `Y` is achievable at the policy's own observation space and that the scorer is correct.
3. **Competence and tie caps:** `E[C] ≥ 0.90` with CI lower bound > 0.85, `1 − E[C] ≤ 0.15` for precedence, on the eval manifest, with both orderings verified feasible per scenario by the bank builder.
4. **Canary:** the reported detection threshold ρ* is finite and stated; the rule must reject at ρ = 0.80 and must not reject at ρ = 0.0.

`[FIXED: the draft's clause 2 (max_s |acc_s − 0.5| < 0.10) was dead — implied by clause 1 whenever seeds are not wildly dispersed — and the rule lacked the assignment-marginal clause it actually needed.]` It is replaced by clause 1(e).

Always reported: full hierarchical bootstrap (Saravanan et al. measured 46% false-positive rates for naive pooling, rising toward 96% with within-cluster size, versus ~5% for per-cluster averaging and the hierarchical bootstrap; for an *equivalence* claim the failure mode is mirrored — a spuriously tight pooled CI makes CI-inclusion trivially pass). Display: per-seed Wilson forest, IQM alongside the mean, observed ICC/DEFF, |Δt| histogram, branch-step top-2 logit margin.

### 4.5 Redesign loop (ordered by cost)

0a. Probe AUC and the bank/manifest MI test. 0b. Paired-lane trajectory-equality assertion. 1. `cfg.leak_rho` and bank SHA in the run manifest. 2. Manifest joint-balance assertion + argmax (not sampled) action path. 3. Dump `LANG_SLICE`/`STATE_LANG_SLICE` over 10⁴ resets (or just read `lang_slice_l2` from the parquet). 4. `reward_audit.py` joint-involution check. 5. Only then treat it as task design: audit the bank for residual per-scenario asymmetry, tighten the builder, regenerate, retrain. **Budget: 2 weeks.** A failed first design is itself reportable evidence that hand-designed "language-necessary" tasks usually are not.

---

## 5. Sentence-embedding conditioning

### 5.1 Encoder

**`sentence-transformers/all-MiniLM-L6-v2`, 384-d, frozen, cached offline.** (i) CALVIN ships precomputed MiniLM embeddings, so offline caching is the mainstream pattern. (ii) HULC's frozen-encoder ablation on the 5-chain metric: paraphrase-MiniLM-L3-v2 **28.3%** > Distilroberta-SBERT (768-d) 27.5% > CLIP text tower 23.2% > raw BERT 14.9% — the smallest encoder won. (iii) 22 M params; the whole cache builds in seconds on CPU. (iv) For 480 sentences forming a handful of clusters that need only be linearly separable, MTEB rank is irrelevant.

Rejected for M1, retained as the month-2 encoder-robustness ablation (one cache rebuild + one run each): CLIP/SigLIP text towers (23.2% vs 28.3%; with zero camera rendering the multimodal alignment buys nothing), raw `AutoModel` + mean-pool BERT (14.9% — writing a custom cache builder silently reproduces this), Qwen3-Embedding-0.6B and EmbeddingGemma-300m (prompt-prefix and MRL-renormalisation footguns).

### 5.2 Dimension and projection

Cache at 384-d, then a **fixed, seeded Gaussian random projection to 32-d + L2 re-normalisation**, computed once offline and stored in the artifact.

`WHY project:` the non-language observation is 344 dims; a raw 384-d embedding would be >50% of the actor's first-layer fan-in, and dimensional dominance perversely makes CSI look good while the policy overfits a handful of clusters. 32-d puts language at ~9% of fan-in.

`WHY fixed rather than trainable `nn.Linear(384,32)`:` a trainable projection needs a custom skrl model that slices the observation, whereas the YAML `Runner` path builds a plain MLP over the flat vector. M1's job is to validate plumbing on the stock `Runner`; JL preserves a few-cluster geometry trivially at 32-d. `harness/models.py` ships the trainable-projection and FiLM models as registered month-2 ablations. `OPEN(4)`.

`[FIXED: the L2 renormalisation sits outside the JL guarantee invoked to justify it, and CSI was regressed on a post-nonlinearity distance.]` Both the raw 384-d cosine distance and the post-projection 32-d cosine distance are stored per minimal pair; **CSI is regressed on the 384-d distance** (primary) with the 32-d distance reported alongside, and the caveat is stated in text.

### 5.3 Caching pipeline

`scripts/build_lang_cache.py` → `data/lang_cache_{sha}.pt`:

```
sentences list[str] (480: 240 single-axis + 240 composed) · instr_id · class_id · variant
family_id · split(0 train / 1 held-out) · minimal_pair_id · flip_type(station|agent|connective)
composed_role_class · composed_order_class · emb384 (480,384) · emb32 (480,32)
cos384_to_pair · cos32_to_pair · proj_seed
encoder_name / encoder_revision / builder_git_sha / artifact_sha256
```

Built once, float32, `SentenceTransformer(...).encode(..., convert_to_numpy=True, normalize_embeddings=True, batch_size=64)`. Loaded from disk for training and for **every** evaluation; never re-encoded on the fly. Encoding the same sentence under a different batch size, padding config or fp16 autocast gives bitwise-different vectors, silently decorrelating training-time and CSI-time embeddings; MRL truncation without renormalisation and prompt-prefix mismatch are the same class of bug. `fleet_env` asserts the stored SHA-256 at construction. Resident cost `480 × 32 × 4 B = 61 KB`; per step one `index_select` — **that number is the argument for caching over on-the-fly encoding and should be stated.**

### 5.4 Paraphrase sizes, splits, and the pre-training probe

| | Variant A | Variant B | Composed (built, trained month 2) |
|---|---|---|---|
| semantic classes | 2 | 2 | 4 |
| train / held-out per class | 50 / 10 | 50 / 10 | 50 / 10 |
| families (train/held-out) | 24 (18/6) | 24 (18/6) | shared grammar |

**Minimal-pair counterfactuals.** Same sentence with only the role-bearing token flipped, same family, same frame — deliberately *minimising* the embedding distance so a high CSI cannot be explained by "the input vector changed a lot". Mean pair cosine distance is a table column next to CSI.

`[FIXED: the held-out-paraphrase test measures the frozen encoder, not the policy.]` `scripts/build_lang_cache.py` reports, **before any training step**, the offline linear-probe accuracy of the 32-d cache on held-out families. If it is 100%, the paper states plainly that held-out-sentence transfer is a property of MiniLM's paraphrase geometry and not evidence about what the policy learned. Held-out statistics are clustered by family (n = 6).

---

## 6. CSI v1

### 6.1 Quantities

`[FIXED: CSI_flip's ratio was a divergent estimand.]` Its denominator, `P(flip | spawn swapped, instruction frozen)`, is *anticorrelated* with the numerator: a perfectly grounded policy is invariant to the spawn swap, so denominator → 0 while numerator → 1 and the ratio diverges **precisely in the regime the project exists to detect**, with no finite mean. "Ratio of means inside the bootstrap" does not repair a divergent estimand, and the draft's stated mitigation ("if D_nuis concentrates at zero we fall back") silently changes the headline definition conditional on the result.

**Primary form (bounded, monotone):**

> `CSI_share = D_lang / (D_lang + D_nuis) ∈ [0,1]`, equal to 0.5 when language and nuisance are equally influential.

`D_lang`, `D_nuis` and every other divergence are **published as marginal distributions**, not only as a summary. The ratio form is **display-only and never the primary estimand**, so no result-conditional fallback is possible.

| Name | Object | Action mode | Notes |
|---|---|---|---|
| `CSI_step` | mean TV distance between action distributions on a **matched observation stream** (roll out under A; intervene only on the 32-d slice), reported **per-agent marginal and joint** | distributions, no sampling | `[FIXED]` restricted to **instruction-relevant decision points** — timesteps where the compliant and greedy scripted planners' optimal actions differ — as the primary statistic, with the all-timestep average secondary. The draft averaged over all t, where both instructions usually imply the same action (shared prefix, forced "stay" after latch), diluting a perfectly grounded policy to ~0.05 — and diluting it *differently per variant* (role binding diverges over a whole route in both marginals; precedence diverges in one marginal at a few wait steps, and a factored joint roughly halves TV when only one agent changes), which made the pre-registered cross-variant dissociation unfalsifiable as instrumented. `[FIXED]` also **symmetrised**: computed on both the A-stream and the B-stream (lane B is already rolled out, so this is free) |
| `CSI_flip` | `P(realised outcome flips)` | argmax | reported as a **pair of paired proportions** with McNemar on the discordant counts; headline effect = difference of paired proportions, not a quotient |
| `CSI_traj` | per-step mean position divergence over `min(T_A,T_B)` | stochastic | reported with the compounding caveat and with realised entropy printed next to it |
| `CSI_intent` | `CSI_step` on **intended** actions before conflict revert | argmax of intent | the revert rule creates action-effect degeneracy: an agent can change its intent and have it reverted to a no-op, so realised-only reporting **understates** sensitivity |

**Denominators** (all lanes below): `D_seed` (slip stream 0→1, scenario and instruction frozen) — `[FIXED]` this is the paper's *definitional* physics-seed denominator, which did not exist in the draft's fully deterministic env; `D_spawn` (spawn_alt, matched BFS difficulty); `D_blank` (language slice zeroed at eval on the trained `Lang` policy); `D_policy` (two independent sampled rollouts, everything frozen). `D_blank` is reported as a **third column, not a ratio**: LIBERO-Plus's most transferable finding is the *asymmetry* — removing language entirely left OpenVLA-OFT "largely unchanged" while goal replacement dropped it "nearly to zero" — which is the signature of language as a distribution-shift trigger rather than a grounded referent, and a swap-only CSI cannot distinguish that from real grounding.

`[FIXED: CSI had a noise floor but no ceiling, and there was no positive control anywhere.]` The draft budgeted CSI only for `Lang` and `Mute`, and `Mute`'s `D_lang` is **identically zero by construction** (the slice is multiplied by zero), so it validates plumbing, not measurement. Every reported CSI is now bracketed by three references computed on the identical pipeline: **`CompliantPlanner`** (scripted, reads the instruction — the ceiling), **`GreedyPlanner`** (scripted, consumes the language slice and ignores it — a non-vacuous floor), and **`SymbolPO`** (trained, known to use its channel). Cross-variant numbers are normalised by that variant's scripted ceiling, so a role-binding-vs-precedence difference is not an artefact of the measure's dilution.

`[FIXED: cross-variant comparison was confounded by map, horizon and competence definition.]` A **layout control** is added: role binding run on the corridor+mouth map (same geometry as precedence, different semantics), so the dissociation is measured with layout held fixed.

### 6.2 Why the denominators differ

- `CSI_step`/`CSI_intent` are **matched-observation** interventions — the stream is literally identical and only the 32-d slice changes — so they are reported **unnormalised** (already bounded in [0,1]) and additionally in `CSI_share` form against `D_seed`.
- `CSI_flip` is outcome-level, so `D_seed` and `D_spawn` are both well defined.
- `CSI_traj`'s matched denominators are `D_seed` and `D_policy`.
- **Numerator and denominator of any given contrast use the same action-selection mode.** Mixing an argmax numerator with a sampled denominator pushes measured CSI toward 1 ("no more sensitive to language than to noise") — a false negative for grounding produced entirely by an evaluation bug.

### 6.3 Intervention procedure — five lanes of one batch

Counterfactual pairs are **lanes of the same vectorised batch**, never two processes:

```
[0,K)    L0 factual        (scenario s_i, instr I_i,            slip stream 0)
[K,2K)   L1 counterfactual (scenario s_i, minimal_pair(I_i),    slip stream 0)
[2K,3K)  L2 seed           (scenario s_i, instr I_i,            slip stream 1)
[3K,4K)  L3 spawn          (spawn_alt of s_i, instr I_i,        slip stream 0)
[4K,5K)  L4 blank          (scenario s_i, LANG zeroed,          slip stream 0)
```

`_reset_idx` **explicitly writes** the identical initial state into every lane of a group rather than relying on the RNG; `harness/rollout.py` asserts `torch.equal` on the non-language slices at t = 0 for every group, every rollout. Isaac Lab guarantees determinism only for identical hardware and versions and warns that GPU work scheduling can reorder operations under runtime parameter changes; two lanes of one batch makes the guarantee structural, and the zero-asset substrate plus stored slip draws removes PhysX nondeterminism entirely.

`[FIXED: §6.3 hard-coded num_envs = 8192 while §8.2 said to sweep and freeze it, and the episode arithmetic was wrong.]` The **training** `num_envs` comes from the §8.2 sweep and is frozen for training. **Evaluation** uses a separately frozen `N_EVAL_ENVS = 8192`, identical for every lane, arm, seed and variant — the determinism concern is lanes differing *within* a comparison, not train-vs-eval. 5 lanes × 2000 base scenarios = **10,000 episodes = 2 batches** of 128 steps (the draft said 4 lanes × 2000 = "two batches"; it was one).

### 6.4 Estimation

- **`CSI_share` first; ratios display-only.** Where a ratio is shown it is a Hájek ratio of means with both terms recomputed inside each bootstrap replicate over resampled base scenarios, 10,000 percentile replicates, IQM alongside the mean.
- **Marginals of `D_lang`, `D_seed`, `D_spawn`, `D_blank` are published.**
- `[FIXED: the permutation null was mis-specified.]` The draft shuffled the instruction→episode mapping and asserted permuted CSI "concentrates near 1 by construction" under the null "the policy ignores language". It does not: if the policy ignores language `D_lang ≈ 0` under both mappings, so permuted CSI concentrates near **0** and the test has no power against its own null; with two classes, permuting is close to re-flipping half the pairs, so permuted CSI ≈ CSI/2 — a scaled copy of the statistic, not a null — and it destroys the paired frozen-scenario design that is CSI's identification premise. Replaced by: **(a) p-value** — a paired sign/permutation test randomising the sign of each episode's paired difference `(D_lang,i − D_nuis,i)` within the frozen pairing, preserving the design; **(b) normaliser calibration** — the **`Placebo`** arm (a 32-d vector from the same cache carrying no task-relevant bit), whose measured CSI is the empirical "language is a nuisance covariate" reference: a correctly calibrated `CSI_share` sits at 0.5 there and the ratio form at 1.
- **Length control:** every pair truncated to `min(T_A,T_B)`; divergences reported as per-step means.
- **Stratification:** by variant, by flip type, by Δ, **crossed over {seen, held-out family} × {seen, held-out scenario}** `[FIXED: the draft reported these as separate margins; a policy sensitive on seen sentences and inert on unseen ones is the actual grounding question]`, and regressed on the 384-d minimal-pair cosine distance.

### 6.5 Episode budget

Per (arm, seed, variant): 5 lanes × 2000 = 10,000 rollouts = 2 eval batches, seconds of GPU. Trained arms contributing CSI: `Lang` 10 + `SymbolPO` 10 + `Symbol` 5 + `Mute` 5 + `Blind` 10 + `Placebo` 3 = 43 seeds × 2 variants → ~860,000 rollouts, tens of minutes of GPU total. `[FIXED: the draft's 320,000 figure assumed 10 seeds for Mute while §4.2 allocated 5.]` The cost is entirely in analysis, not simulation — reported as an explicit episodes-and-GPU-hours-per-audit column.

### 6.6 Per-episode record

`harness/records.py` writes one flat parquet row per episode:

```
arm, variant, training_seed, checkpoint_step, lane, slip_stream, leak_rho, bank_sha, lang_cache_sha,
scenario_id, instr_id_per_agent, instr_class, family_id, split, flip_type, env_id, delta_gap,
T, C, O(itt), Y, latch_time_*, latch_slot_*, success, return, n_obstacle_collisions,
n_robot_collisions, branch_step, branch_entropy, branch_top2_margin, entropy_mean,
lang_slice_l2, state_lang_slice_l2, action_logits_blob, intended_actions_blob, realised_positions_blob
```

Every number in the audit and every CSI variant is derivable from this table **without re-running the simulator**, which is what makes the pre-registered rule auditable and lets a reviewer recompute intervals under a different clustering assumption. `[FIXED: the draft's schema recorded no field from which the language slice's content could be recovered, so a mis-set arm flag would have been undetectable post hoc.]`

---

## 7. File plan

```
config/
  environment.lock.yaml        # §0 pinned versions, generated
  preregistration.yaml         # TOST, delta, n, seeds, decision rule — committed BEFORE training
  arms.yaml                    # 7 arms x 2 variants; lang_gain and leak_rho DERIVED from arm name

tasks/team_listen/
  __init__.py                  # gym.register; skrl_mappo/ippo entry points
  grid_core.py                 # PURE TORCH: apply_slip, step_positions (closed-form N=2), reveal,
                               #   latch_update, matching_potential(dist_field,pos,target_valid)
  obs_layout.py                # OBS_DIM=376 / STATE_DIM=641 + every named slice; build_obs/build_state
  scenario_bank.py             # dataclass, loader, GPU residency, SHA gate (refuses leaky SHA off-arm)
  fleet_env.py                 # TeamGridEnv(DirectMARLEnv)
  fleet_env_cfg.py             # @configclass base + 14 arm/variant cfgs
  rewards.py                   # reward terms, gamma-correct potential shaping
  vis.py                       # VisualizationMarkers debug layer (import-guarded)
  agents/skrl_mappo_cfg.yaml   # copied verbatim from the pinned checkout, then edited
  agents/skrl_ippo_cfg.yaml

harness/
  templates.py                 # grammar, 24 families, family-disjoint split, minimal pairs, flip types
  lang_cache.py                # encoder call, JL projection, artifact write/load, SHA, offline probe
  arms.py                      # arm registry + UNCONDITIONAL assert_arm_consistency()
  planners.py                  # scripted CompliantPlanner (ceiling) / GreedyPlanner (floor) on grid_core
  models.py                    # DeepSets entity encoder, trainable projection, FiLM (month-2 ablations)
  rollout.py                   # raw env.unwrapped driver: argmax + stochastic, 5-lane batching,
                               #   identical-init and paired-trajectory-equality assertions
  metrics.py                   # C, ITT outcome, Y, TV divergence (per-agent + joint), decision-point
                               #   masking, truncation to min(T_A,T_B)
  stats.py                     # tost, wilson(display), hierarchical_bootstrap, iqm, icc_deff, mcnemar,
                               #   paired_sign_test, power_analysis
  probe.py                     # supervised leak probe (obs / state / trajectory) + AUC CIs
  certificate.py               # reads preregistration.yaml, applies the 4-clause rule, canary sweep
  csi.py                       # CSI_share + 4 variants + bootstrap + paired sign test + placebo calib
  records.py                   # parquet schema + writer/reader
  reward_audit.py              # joint-involution invariance + exact bank-computed compliance bound
  skrl_compat.py               # only the 4 verified 1.4.x<->2.x items

scripts/
  print_versions.py  build_scenario_bank.py  build_leaky_bank.py  build_lang_cache.py
  train.py           # refuses --algorithm PPO, --video, --enable_cameras; forces debug_vis=False
  eval_audit.py  eval_csi.py  bench_env.py  parity_check.py  make_tables.py

tests/                         # ALL CPU, no SimulationApp unless noted
  test_spaces.py               # spec_to_gym_space({5})==Discrete(5); [{5},{5}]==MultiDiscrete
  test_action_decode.py        # (E,1) int32 -> (E,) long -> (E,2) delta, shape+dtype
  test_conflict_parity.py      # differential vs RendezvousEnv._resolve_conflicts
  test_reveal_parity.py        # differential vs RendezvousEnv._reveal
  test_slices.py               # OBS/STATE arithmetic; no overlap or gap
  test_episode_length.py       # stay-only rollout truncates at exactly 128 decision steps
  test_potential_purity.py     # rewards.py / matching_potential never read assign|instr_id|lang_vec
  test_instruction_independence.py  # empirical chi2/MI over the full bank AND each manifest
  test_import_graph.py         # scenario_bank.py must not import lang_cache.py
  test_leaky_bank_refusal.py   # production loader refuses the leaky SHA for non-Leaky arms
  test_bank_latch_reachability.py   # stations are leaves off the mouth; both orderings feasible
  test_bank_distfields.py      # latch-aware field == all-free field (alcove topology invariant)
  test_bank_determinism.py     # bank SHA stable across rebuilds
  test_lang_cache.py           # re-encode == cached, bitwise, at 3 batch sizes
  test_yaml_keyset.py          # project YAML key set == pinned checkout's; only values differ
  test_skrl_compat.py          # the 4 verified renames, asserted
  test_stats.py                # TOST/Wilson vs published tables; bootstrap coverage on clustered data
  test_paired_lane_identity.py # blind-arm lanes byte-identical across the full trajectory
  test_arm_consistency.py      # lang L2 norms per arm, unconditionally
```

**`scripts/parity_check.py` deserves explicit schedule time.** It steps both implementations from identical layouts with identical joint action sequences and asserts identical positions, fog maps, collision flags and rewards for ≥10⁵ steps over randomised maps. Porting the conflict rule — including swap, same-cell, and the "following a robot into the cell it vacates is allowed" clause I verified in the reference tests — is where the port will actually go wrong, and **it is invisible in reward curves.**

---

## 8. Throughput plan

### 8.1 Expected numbers, honestly stated

`[FIXED: the cartpole extrapolation spanned ~2 orders of magnitude of network cost and was presented as if comparable.]` The published 1.1 M / 510 k FPS for `Isaac-Cartpole-Direct-v0` at 4096 envs on a 4090 was measured with **RL-Games, a [32,32] MLP over a 4-dim observation, one policy+value pair (~2.5 k params)**. This design runs **skrl MAPPO with `separate: True` over 2 agents = four networks of 512-256-128 over 376/641-dim inputs (~1.5 M params)**, with the update loop run per agent: `rollouts: 16` collects 16 env-steps per update but backprops 2 agents × 5 epochs × (16 × num_envs) samples. **The optimiser step, not the environment, is expected to be 70–90% of wall clock.**

| Stage | Target | Reasoning |
|---|---|---|
| empty `DirectMARLEnv`, 4096 envs, stepping-only | > 800 k FPS (**diagnostic, not a gate**) | `decimation=1`, zero actors |
| + grid kernels | > 400 k FPS | ~45 launches/step; launch-bound, not compute-bound |
| **+ MAPPO training** | **≥ 100 k FPS (HARD GATE)** | see §8.2 step 2 |
| 150 M steps/run at 150 k FPS | ~17 min | |
| runs | Blind 20, Lang 20, SymbolPO 20, Symbol 10, Mute 10, Placebo 6, Leaky 15, λ-sweep 9 ≈ **110** + IPPO controls (+~30%) | |
| **total** | **30–45 GPU-hours** | `[FIXED: the draft's 10–20 h excluded Isaac Sim startup (~40–90 s × runs), the four mandated per-run checkpoint evaluations, the IPPO controls, and convergence re-runs]` |

Memory: MAPPO allocates a **separate `Memory` per agent**, each with its own 641-dim `states` tensor — `[FIXED: the draft's 217 MB was one agent's]` — so `2 × 16 × 4096 × (376+641+6) × 4 B ≈ 536 MB`, trivial on 32 GB and leaving room for `N_EVAL_ENVS = 8192`.

### 8.2 Benchmark order

`[FIXED: the ladder's only hard gate measured the quantity that will not matter.]` "If the empty-env floor is not comfortably above ~500 k FPS, nothing downstream matters" is backwards — the floor could be 5 M FPS and the run would still be update-bound, and no gate on *training* FPS existed anywhere, so the 150 M × ~110 runs budget was never validated before committing it.

1. **Minutes, no simulator:** `test_spaces.py`, `test_action_decode.py`, `test_slices.py`, `test_conflict_parity.py`. Highest-risk, lowest-cost integration points (`{5}` → `Discrete(5)`; `(E,1)` int32 action tensors; the conflict masks).
2. **Hour 1 on the 5090 — the real gate.** `train.py --task Isaac-TeamListen-RoleBinding-Blind-Direct-v0 --headless --algorithm MAPPO --num_envs 4096 --max_iterations 20`. `[FIXED: this was step 6 and its acceptance criterion was "if this runs".]` It validates `possible_agents`, `state()`, agent-keyed dicts, discrete plumbing and the `skrl_mappo_cfg_entry_point` **all at once**, and its acceptance criterion is a **measured training FPS ≥ 100 k**. Also profile and record the rollout/optimiser split for one update in `docs/THROUGHPUT.md`: if the update dominates as expected, the actionable knobs are `mini_batches`, `learning_epochs`, `rollouts` and the 512-256-128 width — not the env kernels the rest of this ladder optimises. If the gate binds, cut per-run steps then non-gating seeds, in that documented order.
3. Empty-env floor via `bench_env.py` → `benchmark_non_rl.py --num_frames 500 --benchmark_backend JSONFileMetrics` (**diagnostic**), then add kernels one at a time — conflict → reveal → obs → state → reward — recording per-stage FPS deltas in `docs/THROUGHPUT.md`.
4. Smoke assertions inside the benchmark: `not sim.has_gui()`, `not sim.has_rtx_sensors()`, `cfg.sim.render_mode is None`, and that `train.py` rejects `--video`.
5. `num_envs` sweep {1024, 2048, 4096, 8192} with training; pick the training-FPS maximum; **freeze it for training**. `N_EVAL_ENVS = 8192` is frozen independently (§6.3).
6. `--algorithm IPPO` as the paired control on the **partial-observation** arms.

`[NOTE — genuinely untested territory]` skrl MAPPO + `Discrete` actions is not exercised by any shipped Isaac Lab task. Tracing it, it should work: `mappo.py` does `create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)` (size 1 via `compute_space_size(..., occupied_size=True)`), and `torch.distributions.Categorical.log_prob` casts float input to long, so float32 action storage round-trips; `CategoricalMixin` + `unnormalized_log_prob` are accepted by `Runner._generate_models`. This is why step 2 runs first.

### 8.3 Deferred

The "3B VLM in the loop vs symbolic stand-in" FPS comparison is outside this document's four deliverables and runs at the end of the window. It needs `--enable_cameras` and RTX rendering, so it must be a **separate benchmark entry point** against a copy of the env with `debug_vis=True` and a `TiledCamera`, never against the training cfg — adding a camera sensor to the shared scene cfg flips `has_rtx_sensors()` and silently re-enables rendering in every training run.

---

## 9. Open questions (defaults in force)

- **OPEN(1) — general-N conflict resolution.** Closed form proven for N=2 and asserted. N=3–4 (month 4) needs a bounded relaxation (3 iterations of the same masks, provably sufficient for N≤4 chains) or a scatter fixpoint; `reference/test_env_paper.py::test_revert_cascade` is the acceptance test.
- **OPEN(2) — absorbing latch vs re-dockable stations.** Default: absorbing latch on entry. The alcove topology (§2.1/§3.1) removes the pass-through trap that made this dangerous. The alternative ("stay" on a station cell latches; moving through does not) is retained; it would change §2.3's single-binary property and must be re-argued. Revisit only if `SymbolPO` fails to clear 0.90.
- **OPEN(3) — preprocessors.** Default: all observation/state preprocessors `null`. Fallback if MAPPO is unstable: a slice-aware preprocessor standardising only `[0:344)` / `[0:513)` and passing the unit-norm language slices through unchanged — **not** a global `RunningStandardScaler`.
- **OPEN(4) — fixed JL projection vs trainable `nn.Linear(384,32)`.** Default: fixed, to stay on the stock `Runner`. If CSI is low with high task return, escalate to FiLM (per-layer γ/β from the 32-d vector); the concat-CSI vs FiLM-CSI difference is itself a reportable result.
- **OPEN(5) — shaping λ.** Default λ = 0.1 with the γ-correct potential form. The λ ∈ {0.05, 0.1, 0.2} × 3-seed sweep runs **before** the headline, not as a fallback. Note that a correct potential-based term cannot change the optimum — only variance and the credit-assignment path — so the sweep tests the *learning* story, not the optimality story; the paper must not conflate the two.
- **OPEN(6) — is fog-of-war needed in M1?** Faithful to `env_paper.py` and the "LiDAR-revealed partial map" spec, but exploration difficulty could mask language effects. Control: a `lidar_radius = 12` variant, 3 seeds. Triggered automatically if `SymbolPO ≈ 0.5`.
- **OPEN(7) — precedence gap.** `G = 1`, and ties are structurally impossible under the unique-mouth topology. If the realised |Δt| distribution from the scripted geometric controller (measured pre-training) shows an unexpected mass at 1, lengthen the corridor rather than raising G.
- **OPEN(8) — grid size 12×12 and obstacle density.** Primary tuning knob if `SymbolPO` cannot clear 0.90; secondary if blind competence falls below 0.90.
- **OPEN(9) — dynamic obstacles.** `num_dynamic_obstacles = 0` in M1. The reference's sequential order-dependent obstacle walk does not vectorise and adds stochasticity outside the stored-slip design. Month-4 reformulation: a deterministic bank-indexed obstacle trajectory, exactly like `bank.slip`.
- **OPEN(10) — multiple comparisons.** The rule is a conjunction over arms across 2 variants. Primary is the seed-level-plus-scenario hierarchical interval; the conjunction is secondary. Holm correction across the perturbation family is deferred to month 4 when the family is fixed.
- **OPEN(11) — channel capacity.** M1's instruction carries exactly 1 bit per variant, so partial grounding is indistinguishable from unreliable grounding, and there are no unseen instruction-role *compositions* within a variant. The composed grammar and `MAX_TARGETS = 3` capacity are built now so this is a data/config change; the decision on whether to promote a 3-target role-binding variant into the trained set is taken at the month-2 gate, not now.
- **OPEN(12) — slip magnitude.** ε = 0.05 pre-registered, sensitivity at {0.0, 0.05, 0.10}. If ε = 0.05 measurably degrades `SymbolPO`'s ceiling, drop to 0.02 and re-derive `D_seed`; if `D_seed` concentrates at zero, report `CSI_share` against `D_spawn` as primary and say so **before** unblinding.


---

## Open questions resolvable only on the 5090

- Exact version lock: isaaclab __version__ + git rev-parse HEAD of the ~/IsaacLab checkout, skrl.__version__ (the whole spec hard-gates on >=2.0.0), Isaac Sim 5.1 SimulationApp banner, gymnasium/torch/numpy/sentence-transformers inside the Isaac python. The dev box has gymnasium 1.1.1 / torch 2.8.0+cu128 / numpy 2.0.2 in system python and no importable isaaclab, so none of these transfer.
- Does the pinned checkout actually ship source/isaaclab_tasks/.../direct/cart_double_pendulum/agents/skrl_mappo_cfg.yaml, and what is its exact key set? tests/test_yaml_keyset.py cannot be written until that file is read on the 5090. Confirm specifically that observation_preprocessor_kwargs / state_preprocessor_kwargs / value_preprocessor_kwargs appear as explicit nulls.
- Can isaaclab.envs.utils.spaces.spec_to_gym_space be imported WITHOUT booting SimulationApp? tests/test_spaces.py is specified as CPU-only; if the import chain pulls omni modules, that test must move behind a SimulationApp fixture and the §8.2 step-1 ladder loses its cheapest gate.
- Does skrl MAPPO + gym.spaces.Discrete actually run end-to-end? No shipped Isaac Lab task exercises this combination. The trace says yes (actions stored float32 via compute_space_size(...occupied_size=True)==1; Categorical.log_prob casts to long), but §8.2 step 2 exists to falsify it.
- Measured MAPPO training FPS at 4096 envs with the 512-256-128 x 4-network configuration, and the rollout-vs-optimiser wall-clock split for one update. This is the hard gate (>=100k FPS) and it sets the per-run step budget for ~110 runs; everything in §8.1's 30-45 GPU-hour estimate is unvalidated until it is measured.
- Empty zero-asset DirectMARLEnv stepping-only FPS at 4096 envs, and whether scene.clone_environments(copy_from_source=False) succeeds at all on a scene containing zero assets (plus whether filter_collisions=False is accepted by the pinned InteractiveSceneCfg).
- num_envs sweep result {1024, 2048, 4096, 8192}: which value maximises training FPS, and does N_EVAL_ENVS=8192 x 5 lanes fit in 32 GB alongside the ~536 MB per-agent MAPPO memories.
- Does episode_length_s=12.9 with dt=0.1, decimation=1 actually yield max_episode_length==129 on the pinned Isaac Lab (float rounding of 12.9/0.1 = 128.99999999999997)? The __init__ assert will crash if not; the fallback is 12.8 with T_DECISION=127.
- Is the render leak path in the pinned scripts/reinforcement_learning/skrl/train.py really --video (setting enable_cameras + render_mode='rgb_array' + RecordVideo), and does sim.has_rtx_sensors() report False at __init__ time as claimed? The §1.13 guard set depends on this.
- Does the pinned train.py resolve f'skrl_{algorithm.lower()}_cfg_entry_point' as assumed, and does it unconditionally apply multi_agent_to_single_agent for --algorithm PPO on a DirectMARLEnv?
- Is sentence-transformers/all-MiniLM-L6-v2 downloadable on the training machine (network/HF cache), and what encoder_revision hash lands in the cache artifact? The whole language pipeline is blocked on this one download.
- Empirical determinism check: does the same bank row + same stored slip stream reproduce bit-identical trajectories across process restarts on this GPU? The paired-lane torch.equal assertion is the audit's primary machine check and its cost/feasibility depends on this holding.
- Realised E[C] and the |delta t| histogram for the scripted GreedyPlanner over the precedence bank, measured before any training. If ties (|delta t| == 0) appear at all, the unique-mouth topology is not what the bank builder actually produced and G=1 is unsafe.
- Empirical value of the exact bank-computed compliance bound from reward_audit.py (min over rows of G_comply - G_defect at gamma=0.99). The paper quotes this number and it cannot be derived on paper.
- Whether slip epsilon=0.05 produces a non-degenerate D_seed (the CSI denominator) without measurably lowering SymbolPO's ceiling below 0.90 - checked on the 3-seed lambda/epsilon pilot before the headline runs.
- Whether VisualizationMarkers / UsdGeom.PointInstancer debug vis works under Isaac Sim 5.1 at all (Isaac Sim issue #251 on kinematic pose writes is unverified on 5.1); this is eval-only and droppable.

## Implementation order (first files)

1. `d:/RL/team_listen/scripts/print_versions.py` - Day-0 blocking gate: writes config/environment.lock.yaml with Isaac Sim / isaaclab+git-SHA / skrl / gymnasium / torch / numpy / sentence-transformers, so every later assumption is version-anchored.
1. `d:/RL/team_listen/tasks/team_listen/obs_layout.py` - Single source of layout truth: OBS_DIM=376, STATE_DIM=641, every named slice constant, padded entity blocks (MAX_AGENTS=4, MAX_TARGETS=3), build_obs()/build_state() - nothing else can be written until the slices are fixed.
1. `d:/RL/team_listen/tasks/team_listen/grid_core.py` - Pure-torch transition core (apply_slip, closed-form N=2 step_positions, reveal, latch_update, instruction-free matching_potential) importable and testable on CPU with no SimulationApp.
1. `d:/RL/team_listen/tests/test_conflict_parity.py` - Differential test of step_positions against RendezvousEnv._resolve_conflicts on the five verified reference cases plus randomised joint actions - the port's most likely silent divergence, invisible in reward curves.
1. `d:/RL/team_listen/tests/test_spaces.py` - Asserts spec_to_gym_space({5}) == Discrete(5) and [{5},{5}] == MultiDiscrete, plus test_action_decode's (E,1) int32 -> (E,) long -> (E,2) delta contract; the two cheapest, highest-risk integration checks.
1. `d:/RL/team_listen/tasks/team_listen/fleet_env_cfg.py` - @configclass base cfg (dt=0.1, decimation=1, episode_length_s=12.9, filter_collisions=False, action_noise_model=None, positive state_space) plus the 14 arm/variant cfgs with lang_gain and leak_rho derived from the arm name.
1. `d:/RL/team_listen/tasks/team_listen/fleet_env.py` - TeamGridEnv(DirectMARLEnv): zero-asset _setup_scene, no-op _apply_action, _pre_physics_step transition, hand-written _get_states, bank-gather _reset_idx, unconditional assert_arm_consistency.
1. `d:/RL/team_listen/tasks/team_listen/agents/skrl_mappo_cfg.yaml` - Copied verbatim from the pinned checkout's cart_double_pendulum MAPPO config, then edited only for entropy_loss_scale=0.008, null preprocessors (with explicit *_kwargs: null), time_limit_bootstrap and the step budget.
1. `d:/RL/team_listen/tasks/team_listen/__init__.py` - gym.register for 7 arms x 2 variants with env_cfg_entry_point plus BOTH skrl_mappo_cfg_entry_point and skrl_ippo_cfg_entry_point - the registration that makes the §8.2 step-2 training-FPS gate runnable.
1. `d:/RL/team_listen/scripts/train.py` - Thin wrapper over Isaac Lab's skrl train.py that refuses --algorithm PPO, --video and --enable_cameras, forces debug_vis=False, and logs the run manifest (arm, leak_rho, bank SHA, lang cache SHA).
1. `d:/RL/team_listen/scripts/build_scenario_bank.py` - Offline bank builder: alcove/unique-mouth layouts, latch-aware BFS dist_field, spawn_alt, delta_gap stratification, leak_bit, precomputed slip streams, plus the latch-aware reachability and per-scenario symmetry rejections.
1. `d:/RL/team_listen/harness/templates.py` - Template grammar (5 x 4 x 6 x 4), 24 families split 18 train / 6 held-out, minimal pairs with both flip types, and the composed role-x-precedence classes built now for month 2.
1. `d:/RL/team_listen/scripts/build_lang_cache.py` - One-shot MiniLM encode + fixed JL projection + L2 renorm into data/lang_cache_{sha}.pt, and reports the offline held-out-family linear-probe accuracy before any training step.
1. `d:/RL/team_listen/harness/rollout.py` - Raw env.unwrapped driver with deterministic-argmax and stochastic modes, 5-lane batching, identical-init and paired-lane torch.equal assertions, and per-episode record emission - never play.py's mean_actions idiom.
1. `d:/RL/team_listen/harness/reward_audit.py` - Exact bank-computed compliance bound (optimal compliant vs defecting return under the true reward and gamma) plus the joint-involution invariance check; the number the paper quotes and the guard against an instruction-dependent reward.