# 5090 Day-0 Report

Training box: `teal-sutd-X870-EAGLE-WIFI7`, RTX 5090 32 GB, driver 580.173.02. Date: 2026-08-31.
Version lock: `config/environment.lock.yaml` (committed in 4c119b8). Isaac Lab `main` @ `b0542fe2d`
(isaaclab ext 0.54.4), chosen because the repo's current default branch `release/3.0.0-beta2`
requires Isaac Sim 6.0.x while this box runs Isaac Sim 5.1.0-rc.19. skrl 2.1.0 (§0 gate passes),
gymnasium 1.2.1, torch 2.7.0+cu128, numpy 1.26.0, sentence-transformers 6.0.1, bundled Python 3.11.13.

Note for reproducibility: the box's shell auto-activates miniconda base (py3.13) and
`isaaclab.sh` prefers `$CONDA_PREFIX`; every Isaac-python invocation must clear
`CONDA_PREFIX`/`CONDA_DEFAULT_ENV`/`VIRTUAL_ENV` or call `_isaac_sim/python.sh` directly.

## MAPPO example benchmarks (shipped tasks, headless, skrl)

| Task | num_envs | iterations/s (sustained) | env-steps/s | peak VRAM |
|---|---|---|---|---|
| Isaac-Shadow-Hand-Over-Direct-v0 (MAPPO) | 2048 | ~39 it/s (33 avg incl. warm-up; 640 steps / 19 s) | ~80 k | 5201 MiB |
| Isaac-Cart-Double-Pendulum-Direct-v0 (MAPPO) | 4096 | ~140 it/s | ~573 k | 3886 MiB |

The two rows are not comparable to each other (Shadow Hand Over simulates two 24-DoF hands per
env; the pendulum is near-free) and neither predicts TeamGridEnv: the zero-asset substrate has no
per-env physics cost at all, and §8.1 expects the optimiser, not the env, to dominate. The ≥100k
FPS hard gate (§8.2 step 2) remains unmeasured until wave-2 lands `fleet_env.py` + registration.
Cold start (SimulationApp boot to first iteration) is ~60–90 s per run on this box — consistent
with §8.1's per-run overhead line.

## Answers: open questions resolvable only on the 5090

1. **Version lock** — done, see above / `config/environment.lock.yaml`.
2. **Pinned cart_double_pendulum MAPPO YAML** — ships at
   `source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/agents/skrl_mappo_cfg.yaml`
   (IPPO and PPO variants alongside). `observation/state/value_preprocessor_kwargs` all present as
   **explicit `null`s**, as §1.14 requires. Key-set diff vs our committed YAMLs is confined to
   `models.policy` mixin kwargs (ours: `unnormalized_log_prob`; pinned Gaussian: `clip_actions`,
   `clip_log_std`, `min_log_std`, `max_log_std`, `initial_log_std`) — `tests/test_yaml_keyset.py`
   must exempt the mixin-kwargs block and assert strict equality elsewhere.
3. **`spec_to_gym_space` without SimulationApp** — **NO.** Import fails at
   `from pxr import Usd, UsdGeom` (chain enters `isaaclab.envs.__init__`). `tests/test_spaces.py`
   must move behind a SimulationApp fixture; §8.2 step 1 loses that CPU-only gate.
4. **skrl MAPPO + Discrete end-to-end** — trace re-verified on skrl 2.1.0 as installed
   (`compute_space_size(..., occupied_size=True) == 1`; `Categorical.log_prob` casts float to
   long; `CategoricalMixin`+`unnormalized_log_prob` accepted by `Runner._generate_models`), but
   **no live run yet exercises it** — first live falsification is §8.2 step 2 on TeamGridEnv.
5. **`episode_length_s = 12.9` → `max_episode_length == 129`** — **YES.** On this box
   `12.9/0.1 == 129.0` exactly in IEEE double (not 128.999…), and the pinned formula is
   `math.ceil(episode_length_s / (dt · decimation))` (`direct_marl_env.py:284`). No 12.8 fallback.
6. **Render leak path in pinned `train.py`** — confirmed exactly as §1.13 claims: `--video` sets
   `enable_cameras=True` (line 67–68), `render_mode="rgb_array"` iff `--video` (line 203),
   `RecordVideo` wrap (line 219). (`has_rtx_sensors()` at `__init__`-time not yet empirically
   probed; eval-only, deferred.)
7. **Entry-point resolution / PPO shim** — confirmed: line 127 resolves
   `f"skrl_{algorithm}_cfg_entry_point"` (plain `skrl_cfg_entry_point` for PPO), and lines
   206–207 apply `multi_agent_to_single_agent(env)` unconditionally for
   `DirectMARLEnv ∧ algorithm == "ppo"` — §1.14's "never `--algorithm PPO`" stands.
8. **MiniLM download** — works; see language cache below.

Still pending (need wave-2 code): empty-env FPS + `clone_environments` on a zero-asset scene +
`filter_collisions=False` acceptance; training-FPS hard gate; `num_envs` sweep; N_EVAL_ENVS=8192
×5-lane memory fit; cross-restart bit-identity of bank-row trajectories; GreedyPlanner E[C] and
|Δt| histogram over the precedence bank; `reward_audit.py` exact compliance bound; ε=0.05
D_seed non-degeneracy; VisualizationMarkers under 5.1.

## Language cache

Built into `data/lang_cache_985f159fd1f5.pt` (untracked, per plan). Encoder
`sentence-transformers/all-MiniLM-L6-v2` @ revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
builder git SHA 8bb28e5; artifact SHA-256 `985f159f…`; 480 sentences (240 single-axis + 240
composed), emb384 (480×384), emb32 (480×32).

**Offline probe (§5.4), builder's report** (linear probe on the 32-d cache, held-out families):
role_binding train 0.68 / held-out 0.55 (family mean 0.54); precedence 0.75 / 0.65 (0.625);
composed (4-class) 0.675 / 0.425 (0.43).

**Additional 5090-side diagnostic — 384-d vs 32-d ridge probe** (same split, decides whether the
signal is lost by the projection or absent in the encoder):

| Variant | 384-d train / held-out | 32-d train / held-out | chance |
|---|---|---|---|
| role_binding | 0.87 / 0.60 | 0.64 / 0.60 | 0.50 |
| precedence | 0.89 / 0.70 | 0.73 / 0.65 | 0.50 |
| composed | 0.94 / 0.625 | 0.65 / 0.40 | 0.25 |

Two findings the design should absorb:
1. **The fixed 32-d JL projection destroys a large fraction of the class signal** (train accuracy
   drops 0.87–0.94 → 0.64–0.73). §5.2's "JL preserves a few-cluster geometry trivially" does not
   hold here — minimal-pair construction makes the class direction a *small* component of the
   embedding, exactly what an unsupervised random projection dilutes. The M1 Lang arm may still
   train (an MLP over ~100 fixed vectors can exceed a linear probe, and training-family
   memorisation suffices for the in-distribution headline), but this materially raises the odds
   of landing in OPEN(4)'s escalation branch, and argues for storing a class-aware (e.g.
   LDA-augmented) projection alongside JL in the artifact now, so the choice is a config switch.
2. **Held-out-family transfer is weak even at 384-d** (0.60–0.70 vs chance 0.5) — the inverse of
   §5.4's worry (probe at 100%). Held-out-sentence evaluation of the Lang arm will be partly
   bounded by encoder geometry, not policy grounding; the paper must attribute held-out
   degradation accordingly, and the month-2 encoder-robustness ablation gains importance.

## Housekeeping

- Two sessions were double-dispatched on this box earlier today; resolved — this session owns
  `~/team_listen` from here. The day-0 commit 4c119b8 is the other session's work, verified here.
- The README's IsaacLab setup snippet should pin `git checkout main` (default branch now requires
  Isaac Sim 6).
- Installer side effect, for anyone using `_isaac_sim`'s python for other projects: its torch was
  replaced 2.13.0+cu130 → 2.7.0+cu128 during Isaac Lab install.
