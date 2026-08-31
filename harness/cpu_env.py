"""CPU stand-in env for the rollout harness (M1_SPEC 1.2 / 6.3 / section 7).

``CPUFleetEnv`` is the "the core drops into a plain vectorised gym env"
promise of spec 1.2 made concrete: a lightweight, Isaac-free environment
over ``tasks/team_listen/grid_core`` with the same driver-facing surface as
``fleet_env.TeamGridEnv``, so ``harness/rollout.py``'s lane logic (the CSI
measurement backbone, spec 6.3) is testable on this Windows dev box where
``isaaclab`` does not exist.

Design rule -- BORROW, do not re-implement.  Every method that can be
borrowed from ``TeamGridEnv`` is borrowed verbatim as a plain function
(class-level assignment below): ``_advance``, ``_slip_row``,
``_pre_physics_step``, ``_get_dones``, ``_get_rewards``,
``_outcome_correct``, ``_draw_scenarios``, ``_draw_instructions``,
``_write_lang_vec``, ``_load_bank``, ``_init_language``, the obs/state
builders, and the ``force_scenarios`` / ``force_slip_stream`` harness
hooks.  Those functions never call ``super()`` and never touch
``DirectMARLEnv`` internals, so the code paths exercised here ARE
fleet_env's own -- the stand-in cannot silently drift from the shipped
transition, observation, reward or bank-gather logic (the parity concern
of spec 1.2 / section 7 ``parity_check.py``).

Exactly two members are local because their fleet_env counterparts call
``super()``:

* ``__init__`` -- allocates the spec 1.3 buffer set (kept in lockstep with
  ``TeamGridEnv.__init__``) instead of booting ``DirectMARLEnv``.
* ``_reset_idx`` -- a line-for-line port of ``TeamGridEnv._reset_idx``
  whose single ``super()._reset_idx(env_ids)`` call is replaced by the
  only bookkeeping that call performs on a zero-asset scene:
  ``episode_length_buf`` zeroing.  Any edit to fleet_env's ``_reset_idx``
  must be mirrored here (and ``tests/test_paired_lane_identity.py`` will
  catch behavioural drift).

``step()`` mirrors the documented ``DirectMARLEnv.step`` order (spec 1.4:
``episode_length_buf`` increments BEFORE ``_get_dones``; spec 1.11:
``reset_buf`` is the AND across agents; done envs auto-reset).  Note that
``harness/rollout.py`` deliberately does NOT use ``step()`` -- it drives
the sub-calls manually so terminal state survives for record emission --
but ``step()`` exists so the stand-in is also usable as an ordinary env.

Import guard: this module never imports isaaclab.  It imports
``fleet_env``, whose own isaaclab import is guarded (spec section 7); when
the guard trips, ``TeamGridEnv``'s base degrades to ``object`` and the
borrowed methods are exactly the plain functions we want.  The stand-in
itself is fully functional with or without isaaclab installed.
"""

import os
import sys
from types import SimpleNamespace

import torch

try:
    from tasks.team_listen import grid_core
    from tasks.team_listen import obs_layout as L
except ImportError:  # standalone import: put the repo root on sys.path
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from tasks.team_listen import grid_core
    from tasks.team_listen import obs_layout as L

from tasks.team_listen import fleet_env as _fe                  # noqa: E402
from tasks.team_listen.fleet_env_cfg import (                   # noqa: E402
    ARM_SPECS,
    ISAACLAB_AVAILABLE,
    TeamGridEnvCfg,
)


def make_cfg(arm="Blind", variant="RoleBinding", **overrides):
    """A ``TeamGridEnvCfg`` with the arm-derived fields applied.

    On the dev box the ``@configclass`` decorator is an inert shim
    (fleet_env_cfg import guard), so ``__post_init__`` -- which derives
    ``lang_gain`` / ``lang_mode`` / ``actor_observes_state`` /
    ``blind_expected_bonus`` from the arm name and validates the spec 1.4
    invariants -- does not run automatically.  This helper always runs it
    (idempotent under real isaaclab, where configclass already ran it).
    """
    cfg = TeamGridEnvCfg(arm=arm, variant=variant, **overrides)
    cfg.__post_init__()
    return cfg


class CPUFleetEnv:
    """Isaac-free stand-in with ``TeamGridEnv``'s driver-facing surface."""

    # ---- borrowed fleet_env methods: the REAL code paths (see module
    # docstring; none of these call super() or touch DirectMARLEnv) -------
    _load_bank = _fe.TeamGridEnv._load_bank
    _init_language = _fe.TeamGridEnv._init_language
    _pre_physics_step = _fe.TeamGridEnv._pre_physics_step
    _apply_action = _fe.TeamGridEnv._apply_action
    _slip_row = _fe.TeamGridEnv._slip_row
    _advance = _fe.TeamGridEnv._advance
    _apply_instruction_switch = _fe.TeamGridEnv._apply_instruction_switch
    _get_dones = _fe.TeamGridEnv._get_dones
    _outcome_correct = _fe.TeamGridEnv._outcome_correct
    _get_rewards = _fe.TeamGridEnv._get_rewards
    _log_step = _fe.TeamGridEnv._log_step
    _draw_scenarios = _fe.TeamGridEnv._draw_scenarios
    _draw_instructions = _fe.TeamGridEnv._draw_instructions
    _lang_mode_is = _fe.TeamGridEnv._lang_mode_is
    _write_lang_vec = _fe.TeamGridEnv._write_lang_vec
    _assert_arm_consistency = _fe.TeamGridEnv._assert_arm_consistency
    force_scenarios = _fe.TeamGridEnv.force_scenarios
    force_slip_stream = _fe.TeamGridEnv.force_slip_stream
    _common_features = _fe.TeamGridEnv._common_features
    _build_state = _fe.TeamGridEnv._build_state
    _build_obs_agent = _fe.TeamGridEnv._build_obs_agent
    _get_observations = _fe.TeamGridEnv._get_observations
    _get_states = _fe.TeamGridEnv._get_states

    def __init__(self, cfg, num_envs=64, device="cpu"):
        # Arm-derived fields + spec 1.4 validation (idempotent; the shim
        # configclass never ran __post_init__, see make_cfg).
        cfg.__post_init__()
        if cfg.arm not in ARM_SPECS:
            raise ValueError("unknown arm %r" % (cfg.arm,))
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.render_mode = None
        # _pre_physics_step's debug_asserts render guard probes these
        # (spec 1.13); the stand-in has no renderer by construction.
        self.sim = SimpleNamespace(has_gui=lambda: False,
                                   has_rtx_sensors=lambda: False,
                                   render_mode=None)

        # spec 1.4 [FIXED: episode-length off-by-one]: 129 -> 128 decision
        # steps, same arithmetic TeamGridEnv.__init__ asserts against.
        self.max_episode_length = L.T_DECISION + 1
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device)
        self.extras = {}

        E, dev = self.num_envs, self.device
        N = L.N_AGENTS
        assert len(self.cfg.possible_agents) == N == 2

        # ---- world state buffers: KEPT IN LOCKSTEP with
        # TeamGridEnv.__init__ (spec 1.3 table, exact dtypes) -------------
        self.occ = torch.zeros((E, L.R, L.C), dtype=torch.int8, device=dev)
        self.known_free = torch.zeros((E, L.R, L.C), dtype=torch.bool, device=dev)
        self.known_obs = torch.zeros((E, L.R, L.C), dtype=torch.bool, device=dev)
        self.pos = torch.zeros((E, L.MAX_AGENTS, 2), dtype=torch.int16, device=dev)
        self.agent_valid = torch.zeros((E, L.MAX_AGENTS), dtype=torch.bool, device=dev)
        self.agent_valid[:, :N] = True
        self.target = torch.zeros((E, L.MAX_TARGETS, 2), dtype=torch.int16, device=dev)
        self.target_valid = torch.zeros((E, L.MAX_TARGETS), dtype=torch.bool, device=dev)
        self.latched = torch.zeros((E, L.MAX_AGENTS), dtype=torch.bool, device=dev)
        self.latch_slot = torch.full((E, L.MAX_AGENTS), -1, dtype=torch.int8, device=dev)
        self.latch_time = torch.full((E, L.MAX_AGENTS), -1, dtype=torch.int16, device=dev)
        self.dist_field = torch.zeros((E, L.MAX_TARGETS, L.R, L.C),
                                      dtype=torch.int16, device=dev)
        self.scenario_id = torch.zeros((E,), dtype=torch.int64, device=dev)
        self.slip_stream = torch.zeros((E,), dtype=torch.int8, device=dev)
        self.instr_id = torch.zeros((E, L.MAX_AGENTS), dtype=torch.int64, device=dev)
        self.instr_class = torch.zeros((E,), dtype=torch.int64, device=dev)
        self.lang_vec = torch.zeros((E, L.MAX_AGENTS, L.LANG_DIM),
                                    dtype=torch.float32, device=dev)
        self._instr_switch_time = torch.full((E,), -1, dtype=torch.int16, device=dev)

        # step artifacts consumed by _get_rewards
        self._act = torch.zeros((E, N), dtype=torch.long, device=dev)
        self._hit_obstacle = torch.zeros((E, N), dtype=torch.bool, device=dev)
        self._hit_robot = torch.zeros((E, N), dtype=torch.bool, device=dev)
        self._phi = torch.zeros((E,), dtype=torch.float32, device=dev)
        self._phi_prev = torch.zeros((E,), dtype=torch.float32, device=dev)

        # constants
        self._ALL = torch.arange(E, dtype=torch.long, device=dev)
        self._reveal_offsets = grid_core.chebyshev_offsets(
            self.cfg.lidar_radius).to(dev)
        self._agent_id_oh = torch.eye(L.MAX_AGENTS, dtype=torch.float32,
                                      device=dev)

        # harness hooks (five-lane batching, spec 6.3)
        self._forced_scenarios = None
        self._forced_slip_stream = None

        # ---- artifacts (same flow as TeamGridEnv.__init__) --------------
        self._bank = self._load_bank()
        self._switch_active = bool(
            (self._bank.instr_switch_time >= 0).any().item())
        self._train_pool = (self._bank.split == 0).nonzero(as_tuple=False).reshape(-1)
        eval_pool = (self._bank.split == 1).nonzero(as_tuple=False).reshape(-1)
        if self.cfg.eval_mode and self.cfg.eval_manifest_path:
            manifest = _fe._torch_load(self.cfg.eval_manifest_path)
            if isinstance(manifest, dict):
                manifest = manifest["scenario_id"]
            eval_pool = manifest.reshape(-1).to(dev, torch.long)
        self._eval_pool = eval_pool
        self._eval_cursor = 0
        if self._train_pool.numel() == 0 and not self.cfg.eval_mode:
            raise RuntimeError("bank has no split==0 (train) rows")

        self._init_language()

        try:
            from harness.arms import assert_arm_consistency
            self._arm_consistency_fn = assert_arm_consistency
        except Exception:
            self._arm_consistency_fn = None

    # ------------------------------------------------------------------
    # Driver-facing surface
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        """Mirrors gym's ``.unwrapped`` so rollout.resolve_env is uniform."""
        return self

    @property
    def agents(self):
        return list(self.cfg.possible_agents)

    def reset(self, seed=None):
        """Full reset; returns (obs_dict, extras) like DirectMARLEnv.reset.

        ``seed`` seeds the GLOBAL torch RNG, exactly like the
        ``DirectMARLEnv.seed`` staticmethod it stands in for (spec 1.10 WHY:
        scenario CONTENT is bank-gathered, RNG picks only row/instruction).
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        self._reset_idx(None)
        return self._get_observations(), self.extras

    def step(self, actions):
        """One decision step in the documented DirectMARLEnv order:
        transition (in ``_pre_physics_step``), ``episode_length_buf`` +1
        BEFORE ``_get_dones`` (spec 1.4), rewards, AND-reduced
        ``reset_buf`` with auto-reset (spec 1.11), then observations.
        """
        agents = self.cfg.possible_agents
        self._pre_physics_step(actions)
        self._apply_action()                      # no-op (zero-asset, spec 1.1)
        self.episode_length_buf += 1
        terminated, truncated = self._get_dones()
        self.reset_buf = terminated[agents[0]] | truncated[agents[0]]
        rewards = self._get_rewards()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).reshape(-1)
        if reset_ids.numel() > 0:
            self._reset_idx(reset_ids)
        obs = self._get_observations()
        return obs, rewards, terminated, truncated, self.extras

    # ------------------------------------------------------------------
    # _reset_idx: line-for-line port of TeamGridEnv._reset_idx.  The ONLY
    # difference is the super()._reset_idx(env_ids) call, replaced by the
    # episode_length_buf zeroing it performs on a zero-asset scene.  Keep
    # in lockstep with fleet_env (see module docstring).
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._ALL
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                                      device=self.device)
        self.episode_length_buf[env_ids] = 0          # super()._reset_idx stand-in

        N = L.N_AGENTS
        bank = self._bank
        sid = self._draw_scenarios(env_ids)
        self.scenario_id[env_ids] = sid

        # pure gathers from the frozen bank (spec 1.10)
        self.occ[env_ids] = bank.occ[sid].to(torch.int8)
        self.pos[env_ids] = bank.spawn[sid]
        self.target[env_ids] = bank.target[sid]
        self.target_valid[env_ids] = bank.target_valid[sid]
        self.dist_field[env_ids] = bank.dist_field[sid]
        self._instr_switch_time[env_ids] = bank.instr_switch_time[sid]

        if self._forced_slip_stream is not None:
            self.slip_stream[env_ids] = self._forced_slip_stream[env_ids]
        else:
            self.slip_stream[env_ids] = int(self.cfg.slip_stream)

        self.latched[env_ids] = False
        self.latch_slot[env_ids] = -1
        self.latch_time[env_ids] = -1

        instr_id, instr_class = self._draw_instructions(
            env_ids, bank.leak_bit[sid], float(self.cfg.leak_rho))
        self.instr_id[env_ids] = instr_id
        self.instr_class[env_ids] = instr_class
        self._write_lang_vec(env_ids, instr_id)

        # fog of war: all-unknown, then the initial reveal at spawn (spec 1.9)
        kf = torch.zeros((env_ids.numel(), L.R, L.C), dtype=torch.bool,
                         device=self.device)
        ko = torch.zeros_like(kf)
        grid_core.reveal(kf, ko, self.occ[env_ids],
                         self.pos[env_ids][:, :N], self._reveal_offsets)
        self.known_free[env_ids] = kf
        self.known_obs[env_ids] = ko

        # Phi_0 at spawn (consumed as Phi_{t-1} by the first step's shaping)
        self._phi[env_ids] = grid_core.matching_potential(
            self.dist_field[env_ids], self.pos[env_ids][:, :N],
            self.target_valid[env_ids])

        self._assert_arm_consistency(env_ids)         # UNCONDITIONAL (spec 1.6)


__all__ = ["CPUFleetEnv", "make_cfg", "ISAACLAB_AVAILABLE"]
