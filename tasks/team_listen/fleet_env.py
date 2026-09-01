"""TeamGridEnv: zero-asset DirectMARLEnv shell over the pure-torch grid core.

Authoritative reference: docs/M1_SPEC.md sections 1.1 (zero-asset substrate),
1.4 (spaces/episode length), 1.5/1.6 (observation and state layouts --
hand-written against tasks/team_listen/obs_layout.py), 1.7 (discrete action
decode), 1.9 (reveal), 1.10 (scenario bank + bank-indexed slip; RNG-free
scenario CONTENT), 1.11 (latch + termination), 1.12 (reward), 1.13 (render
guards), 5.3 (lang-cache SHA assert at construction).

Design contract (spec 1.1/1.2):

* The scene holds NO physics assets. ``_setup_scene`` calls
  ``self.scene.clone_environments(copy_from_source=False)`` UNCONDITIONALLY
  (with ``replicate_physics=True`` ``scene.env_origins`` stays None until it
  runs) and, only under ``cfg.debug_vis``, adds a dome light.
* ``_apply_action`` is a no-op; the whole transition runs in
  ``_pre_physics_step`` on ``self.device`` tensors via ``grid_core``.
* ``_get_states`` is hand-written (positive ``state_space``; NEVER the
  ``-1`` auto-concatenation) and implemented even where unused.

Import guard: ``isaaclab`` is unavailable on the dev box and pre-app on the
5090; the guarded import below keeps plain ``import`` working everywhere,
and construction raises a clear error when the shim is active.  Do NOT
import this module on the 5090 before ``SimulationApp`` boots in a process
that will later ``gym.make`` the env (sys.modules would cache the shim).
"""

import hashlib
import os
from collections import deque
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from . import grid_core
from . import obs_layout as L
from .fleet_env_cfg import TeamGridEnvCfg, VARIANT_TO_TEMPLATE_KEY

try:
    from isaaclab.envs import DirectMARLEnv

    _ISAACLAB_AVAILABLE = True
    _ISAACLAB_IMPORT_ERROR = None
except Exception as _exc:  # ImportError, or partial omni failures pre-app
    DirectMARLEnv = object  # type: ignore[assignment,misc]
    _ISAACLAB_AVAILABLE = False
    _ISAACLAB_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Reward constants (spec 1.12). shaping_lambda / reward_gamma live on the cfg
# (lambda is swept pre-headline); these are frozen for the study.
# ---------------------------------------------------------------------------

STEP_COST = -0.01               # per team, per step
COLLISION_COST = -0.25          # per robot, per step, obstacle AND robot-robot
COMPLETION_BONUS = 2.0          # on the terminal both-latched step
OUTCOME_BONUS = 10.0            # +10 * Y, the ONLY instruction-dependent term
BLIND_EXPECTED_BONUS = 5.0      # E[10*Y | C=1] for an instruction-blind policy
FIRST_LATCH_BONUS = 2.0         # per agent, first latch of the episode
                                # (DECISIONS.md terminal-credit amendment;
                                #  MUST match tasks/team_listen/rewards.py)

#: Environment variable set by scripts/train.py: forces cfg.debug_vis = False
#: and arms the training-mode render guards (spec 1.13 guard set).
_TRAIN_ENV_FLAG = "TEAM_LISTEN_FORCE_DEBUG_VIS_OFF"

#: Required tensor keys of a scenario-bank artifact (spec 1.10 schema).
_BANK_KEYS = (
    "occ", "spawn", "spawn_alt", "target", "target_valid", "dist_field",
    "mouth", "delta_gap", "leak_bit", "instr_switch_time", "slip", "split",
)


# ---------------------------------------------------------------------------
# Artifact loading helpers (module-level so they are testable without Isaac)
# ---------------------------------------------------------------------------

def _file_sha256(path):
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _torch_load(path):
    """torch.load tolerant of the torch>=2.6 weights_only default.

    Both artifacts (bank, lang cache) are our own files of tensors, python
    scalars, strings and containers; try the safe path first.
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _check_sha(path, expected, what):
    """Compare the file hash against an expected (prefix) hex digest."""
    if not expected:
        return
    got = _file_sha256(path)
    if not got.startswith(expected.lower()):
        raise RuntimeError(
            "%s SHA mismatch for %s: expected %s..., file hashes to %s "
            "(the bank/cache SHA gate of M1_SPEC 1.10/5.3)"
            % (what, path, expected, got))


def _load_bank_fallback(path, cfg, device):
    """Interim loader for ``data/scenario_bank_{variant}_{sha}.pt``.

    ``tasks/team_listen/scenario_bank.py`` (spec section 7) is the
    production loader (dataclass + GPU residency + full SHA gate); until it
    lands, this fallback enforces the same critical guards: the SHA check
    and the leaky-bank refusal for non-Leaky arms (spec 4.1 [FIXED:
    canary]; tests/test_leaky_bank_refusal.py targets the production
    loader).
    """
    _check_sha(path, cfg.bank_sha, "scenario bank")
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        payload = getattr(payload, "__dict__", None) or {}
    missing = [k for k in _BANK_KEYS if k not in payload]
    if missing:
        raise RuntimeError("scenario bank %s is missing keys %r (spec 1.10 "
                           "schema)" % (path, missing))
    if bool(payload.get("leaky", False)) and cfg.arm != "Leaky":
        raise RuntimeError(
            "bank %s is a LEAKY bank (build_leaky_bank.py); it is refused "
            "for arm %r -- only the Leaky arm may load it (M1_SPEC 4.1)"
            % (path, cfg.arm))
    ns = SimpleNamespace(**{k: payload[k].to(device) for k in _BANK_KEYS})
    ns.meta = {k: v for k, v in payload.items() if not torch.is_tensor(v)}
    return ns


def _bfs_dist(free, src, blocked):
    """int16 (R, C) BFS distance field over 4-connected free cells.

    ``free``: (R, C) bool; ``src``: (r, c); ``blocked``: extra obstacle
    cells (the OTHER target cells -- the latch-aware rule of spec 1.10).
    Unreachable cells hold -1 (the sentinel ``matching_potential`` maps to
    +inf).  Offline/CPU only (synthetic smoke bank).
    """
    rows, cols = free.shape
    dist = torch.full((rows, cols), -1, dtype=torch.int16)
    passable = free.clone()
    for (br, bc) in blocked:
        passable[br, bc] = False
    sr, sc = int(src[0]), int(src[1])
    if not bool(free[sr, sc]):
        return dist
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        d = int(dist[r, c]) + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and passable[nr, nc] \
                    and int(dist[nr, nc]) < 0:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


def _synthetic_bank(device, epsilon=0.05, seed=1234):
    """Deterministic in-memory smoke bank (cfg.allow_synthetic_bank ONLY).

    Exists so the section 8.2 step-2 training-FPS gate can run before
    ``scripts/build_scenario_bank.py`` lands.  Obstacle-free 12x12 map, two
    targets, two spawns, latch-aware BFS fields, two genuine slip streams.
    NEVER valid for results: no alcove/mouth topology, no stratification.
    """
    K, R, C = 2, L.R, L.C
    occ = torch.zeros((K, R, C), dtype=torch.uint8)
    target = torch.zeros((K, L.MAX_TARGETS, 2), dtype=torch.int16)
    target[:, 0] = torch.tensor([1, 1], dtype=torch.int16)
    target[:, 1] = torch.tensor([1, 10], dtype=torch.int16)
    target_valid = torch.zeros((K, L.MAX_TARGETS), dtype=torch.bool)
    target_valid[:, 0] = target_valid[:, 1] = True
    spawn = torch.zeros((K, L.MAX_AGENTS, 2), dtype=torch.int16)
    spawn[:, 0] = torch.tensor([10, 2], dtype=torch.int16)
    spawn[:, 1] = torch.tensor([10, 9], dtype=torch.int16)
    spawn_alt = spawn.clone()
    spawn_alt[:, 0, 1] = 3
    spawn_alt[:, 1, 1] = 8

    free = occ[0] == 0
    cells = [tuple(target[0, j].tolist()) for j in range(L.MAX_TARGETS)]
    dist = torch.full((K, L.MAX_TARGETS, R, C), -1, dtype=torch.int16)
    for j in range(L.MAX_TARGETS):
        if bool(target_valid[0, j]):
            others = [c for jj, c in enumerate(cells)
                      if jj != j and bool(target_valid[0, jj])]
            dist[:, j] = _bfs_dist(free, cells[j], others)

    gen = torch.Generator().manual_seed(seed)
    u = torch.rand((K, 2, L.T_DECISION, L.MAX_AGENTS), generator=gen)
    forced = torch.randint(0, grid_core.N_ACTIONS,
                           (K, 2, L.T_DECISION, L.MAX_AGENTS), generator=gen)
    slip = torch.where(u < epsilon, forced,
                       torch.full_like(forced, grid_core.NO_SLIP))

    ns = SimpleNamespace(
        occ=occ, spawn=spawn, spawn_alt=spawn_alt, target=target,
        target_valid=target_valid, dist_field=dist,
        mouth=torch.full((K, 2), -1, dtype=torch.int16),
        delta_gap=torch.zeros((K,), dtype=torch.int8),
        leak_bit=torch.zeros((K,), dtype=torch.uint8),
        instr_switch_time=torch.full((K,), -1, dtype=torch.int16),
        slip=slip.to(torch.uint8),
        split=torch.tensor([0, 1], dtype=torch.uint8),
    )
    for k in _BANK_KEYS:
        setattr(ns, k, getattr(ns, k).to(device))
    ns.meta = {"synthetic": True, "epsilon": epsilon, "seed": seed}
    return ns


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

class TeamGridEnv(DirectMARLEnv):
    """Zero-asset grid fleet env (spec 1.1): tensors-only, grid_core transition."""

    cfg: TeamGridEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        if not _ISAACLAB_AVAILABLE:
            raise RuntimeError(
                "TeamGridEnv requires Isaac Lab (DirectMARLEnv); the import "
                "failed with: %r. On the 5090, construct it only after "
                "SimulationApp boots (via the shipped train.py workflow)."
                % (_ISAACLAB_IMPORT_ERROR,))

        # scripts/train.py sets this to force-disable debug vis (spec 1.13
        # guard 1) and to arm the training-mode render asserts (guard 2).
        training_mode = bool(os.environ.get(_TRAIN_ENV_FLAG))
        if training_mode:
            cfg.debug_vis = False

        super().__init__(cfg, render_mode, **kwargs)

        # ---- spec 1.4 [FIXED: episode-length off-by-one] -----------------
        assert self.max_episode_length == L.T_DECISION + 1, (
            "max_episode_length=%d != T_DECISION+1=%d; dt/decimation/"
            "episode_length_s drifted (M1_SPEC 1.4; fallback 12.8 / "
            "T_DECISION=127 must be taken EXPLICITLY, not silently)"
            % (self.max_episode_length, L.T_DECISION + 1))

        # ---- spec 1.13 render guards (2) ---------------------------------
        if training_mode:
            assert self.render_mode is None, (
                "training mode forbids render_mode=%r (the --video leak "
                "path, M1_SPEC 1.13)" % (self.render_mode,))
            assert getattr(self.cfg.sim, "render_mode", None) is None, \
                "training mode forbids cfg.sim.render_mode (M1_SPEC 1.13)"

        E, dev = self.num_envs, self.device
        N = L.N_AGENTS
        assert len(self.cfg.possible_agents) == N == 2

        # ---- world state buffers (spec 1.3 table, exact dtypes) ----------
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

        # per-env instruction switch times, gathered at reset (M1: all -1)
        self._instr_switch_time = torch.full((E,), -1, dtype=torch.int16, device=dev)

        # step artifacts consumed by _get_rewards
        self._act = torch.zeros((E, N), dtype=torch.long, device=dev)
        self._hit_obstacle = torch.zeros((E, N), dtype=torch.bool, device=dev)
        self._hit_robot = torch.zeros((E, N), dtype=torch.bool, device=dev)
        self._newly_latched = torch.zeros((E, N), dtype=torch.bool, device=dev)
        self._phi = torch.zeros((E,), dtype=torch.float32, device=dev)
        self._phi_prev = torch.zeros((E,), dtype=torch.float32, device=dev)

        # constants
        self._ALL = torch.arange(E, dtype=torch.long, device=dev)
        self._reveal_offsets = grid_core.chebyshev_offsets(
            self.cfg.lidar_radius).to(dev)
        self._agent_id_oh = torch.eye(L.MAX_AGENTS, dtype=torch.float32,
                                      device=dev)

        # harness hooks (five-lane batching, spec 6.3): None = env defaults
        self._forced_scenarios = None       # (E,) long or None
        self._forced_slip_stream = None     # (E,) or None

        # ---- artifacts ---------------------------------------------------
        self._bank = self._load_bank()
        self._switch_active = bool(
            (self._bank.instr_switch_time >= 0).any().item())
        self._train_pool = (self._bank.split == 0).nonzero(as_tuple=False).reshape(-1)
        eval_pool = (self._bank.split == 1).nonzero(as_tuple=False).reshape(-1)
        if self.cfg.eval_mode and self.cfg.eval_manifest_path:
            manifest = _torch_load(self.cfg.eval_manifest_path)
            if isinstance(manifest, dict):
                manifest = manifest["scenario_id"]
            eval_pool = manifest.reshape(-1).to(dev, torch.long)
        self._eval_pool = eval_pool
        self._eval_cursor = 0
        if self._train_pool.numel() == 0 and not self.cfg.eval_mode:
            raise RuntimeError("bank has no split==0 (train) rows")

        self._init_language()

        # arm-consistency checker: harness/arms.py when present, otherwise a
        # local fallback with the same spec 1.6 invariant. UNCONDITIONAL.
        try:
            from harness.arms import assert_arm_consistency
            self._arm_consistency_fn = assert_arm_consistency
        except Exception:
            self._arm_consistency_fn = None

        # optional debug-vis layer (spec 1.13; eval-only, vis.py owns it)
        if self.cfg.debug_vis and hasattr(self, "set_debug_vis"):
            self.set_debug_vis(True)

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_bank(self):
        cfg = self.cfg
        path = cfg.bank_path or os.environ.get("TEAM_LISTEN_BANK", "")
        if path:
            try:
                from . import scenario_bank  # production loader (spec sec. 7)
            except Exception:
                scenario_bank = None
            if scenario_bank is not None and hasattr(scenario_bank, "load_bank"):
                return scenario_bank.load_bank(path, device=self.device,
                                               arm=cfg.arm)
            return _load_bank_fallback(path, cfg, self.device)
        if cfg.allow_synthetic_bank or os.environ.get("TEAM_LISTEN_SYNTHETIC_BANK"):
            return _synthetic_bank(self.device)
        raise RuntimeError(
            "no scenario bank: set cfg.bank_path (or $TEAM_LISTEN_BANK) to "
            "data/scenario_bank_{variant}_{sha}.pt, or set "
            "cfg.allow_synthetic_bank=True for a plumbing smoke run "
            "(M1_SPEC 1.10; NEVER for results)")

    def _init_language(self):
        """LANG_TABLE + per-class sentence pools (spec 1.10 / 5.3 / 4.1).

        Arms without a language cache (Blind/Leaky/Mute/Symbol/SymbolPO) use
        a fixed 2-row orthonormal code table indexed by class: identical
        parameter count and fan-in across arms (spec 4.1), and the zero-gain
        arms multiply it away entirely.
        """
        cfg, dev = self.cfg, self.device
        if cfg.lang_mode in ("lang", "placebo"):
            path = cfg.lang_cache_path or os.environ.get("TEAM_LISTEN_LANG_CACHE", "")
            if not path:
                raise RuntimeError(
                    "arm %r needs the language cache: set cfg.lang_cache_path "
                    "(or $TEAM_LISTEN_LANG_CACHE) to data/lang_cache_{sha}.pt "
                    "(M1_SPEC 5.3)" % (cfg.arm,))
            payload = _torch_load(path)
            stored = str(payload.get("artifact_sha256", ""))
            # spec 5.3: fleet_env asserts the stored SHA-256 at construction.
            if cfg.lang_cache_sha:
                if not stored.startswith(cfg.lang_cache_sha.lower()):
                    raise RuntimeError(
                        "lang cache artifact_sha256=%s does not match "
                        "cfg.lang_cache_sha=%s (M1_SPEC 5.3)"
                        % (stored, cfg.lang_cache_sha))
            self._lang_table = payload["emb32"].to(dev, torch.float32)
            variant_key = VARIANT_TO_TEMPLATE_KEY[cfg.variant]
            variants = payload["variant"]           # list[str] per row
            class_id = payload["class_id"].tolist()
            split = payload["split"].tolist()
            rows = [i for i, v in enumerate(variants)
                    if v == variant_key and split[i] == 0]
            by_class = []
            for c in (0, 1):
                sel = [i for i in rows if class_id[i] == c]
                if not sel:
                    raise RuntimeError(
                        "lang cache has no train rows for variant %r class %d"
                        % (variant_key, c))
                by_class.append(torch.tensor(sel, dtype=torch.long, device=dev))
            self._rows_by_class = by_class
            # Placebo: any train row of this variant, class-independent
            # ("randomly relabelled instruction id", spec 4.1).
            self._placebo_rows = torch.tensor(rows, dtype=torch.long, device=dev)
            self._lang_cache_sha = stored
        else:
            codes = torch.zeros((2, L.LANG_DIM), dtype=torch.float32, device=dev)
            codes[0, 0] = 1.0
            codes[1, 1] = 1.0                       # 2 orthonormal codes
            self._lang_table = codes
            self._rows_by_class = None
            self._placebo_rows = None
            self._lang_cache_sha = ""

    # ------------------------------------------------------------------
    # Scene (spec 1.1)
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Unconditional: with replicate_physics=True, scene.env_origins stays
        # None until clone_environments runs. copy_from_source=False on a
        # zero-asset scene. filter_collisions=False comes from the scene cfg
        # ([FIXED: filter_collisions]).
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.debug_vis:
            import isaaclab.sim as sim_utils
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0,
                                               color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

    def _set_debug_vis_impl(self, debug_vis):
        """Eval-only marker layer (spec 1.13); owned by tasks/team_listen/vis.py."""
        if not debug_vis:
            return
        try:
            from . import vis
        except ImportError as exc:
            raise RuntimeError(
                "cfg.debug_vis=True but tasks/team_listen/vis.py is not "
                "available (spec section 7 file plan)") from exc
        vis.setup_markers(self)

    # ------------------------------------------------------------------
    # Action path (spec 1.7)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions):
        # skrl's unflatten_tensorized_space(Discrete, .) returns (E, 1)
        # int32/int64; .reshape(-1).long() is load-bearing (spec 1.7).
        a = torch.stack(
            [actions[k].reshape(-1).long() for k in self.cfg.possible_agents], 1)
        a = grid_core.apply_slip(a, self._slip_row())   # bank-indexed, spec 1.10
        self._act = a
        if self.cfg.debug_asserts:
            # spec 1.13 guard (3): a single stray camera costs 20x silently.
            assert not (self.sim.has_gui() or self.sim.has_rtx_sensors()), \
                "rendering active during training (M1_SPEC 1.13)"
        self._advance(a)

    def _apply_action(self):
        # No physics assets; the transition already ran in _pre_physics_step.
        pass

    def _slip_row(self):
        """(E, MAX_AGENTS) uint8 slip row for (scenario, stream, t) (spec 1.10)."""
        t = self.episode_length_buf.clamp(max=L.T_DECISION - 1).long()
        return self._bank.slip[self.scenario_id, self.slip_stream.long(), t]

    def _advance(self, a):
        """One grid transition: latch-stay, move+conflicts, latch, reveal, Phi."""
        N = L.N_AGENTS
        # spec 1.11: a latched robot's action is overwritten to "stay" (after
        # slip -- a docked robot cannot slip off its station).
        a = torch.where(self.latched[:, :N],
                        torch.full_like(a, grid_core.STAY), a)

        nxt, hit_obs, hit_rob = grid_core.step_positions(
            self.pos[:, :N], a, self.occ, self.latched[:, :N], (L.R, L.C))
        self.pos[:, :N] = nxt

        # latch on the POST-CONFLICT cell; t is the 0-based decision index
        # (episode_length_buf pre-increment).
        prev_latched = self.latched[:, :N].clone()
        grid_core.latch_update(self.pos[:, :N], self.target, self.target_valid,
                               self.latched[:, :N], self.latch_slot[:, :N],
                               self.latch_time[:, :N], self.episode_length_buf)
        # latch-state transition for the first-latch bonus (absorbing latch
        # => True at most once per agent per episode; instruction-free).
        self._newly_latched = self.latched[:, :N] & ~prev_latched

        grid_core.reveal(self.known_free, self.known_obs, self.occ,
                         self.pos[:, :N], self._reveal_offsets)

        self._hit_obstacle = hit_obs
        self._hit_robot = hit_rob
        self._phi_prev = self._phi
        self._phi = grid_core.matching_potential(
            self.dist_field, self.pos[:, :N], self.target_valid)

        # M1-inert mid-episode reassignment branch (spec 1.5 [FIXED]): the
        # bank ships instr_switch_time = -1 everywhere, so _switch_active is
        # False and this costs nothing; month-4 reassignment is a data change.
        if self._switch_active:
            self._apply_instruction_switch()

    def _apply_instruction_switch(self):
        t = self.episode_length_buf.to(self._instr_switch_time.dtype)
        fire = (self._instr_switch_time >= 0) & (self._instr_switch_time == t)
        ids = fire.nonzero(as_tuple=False).reshape(-1)
        if ids.numel() == 0:
            return
        leak_bit = self._bank.leak_bit[self.scenario_id[ids]]
        instr_id, instr_class = self._draw_instructions(
            ids, leak_bit, float(self.cfg.leak_rho))
        self.instr_id[ids] = instr_id
        self.instr_class[ids] = instr_class
        self._write_lang_vec(ids, instr_id)

    # ------------------------------------------------------------------
    # Termination (spec 1.11) and reward (spec 1.12)
    # ------------------------------------------------------------------

    def _get_dones(self):
        N = L.N_AGENTS
        done = self.latched[:, :N].all(dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        agents = self.cfg.possible_agents
        # One shared tensor broadcast to both agents: step() reduces with an
        # elementwise AND across agents (spec 1.11).
        return ({a: done for a in agents}, {a: time_out for a in agents})

    def _outcome_correct(self):
        """Y per spec 2.3 / 3.3 (meaningful only where both robots latched)."""
        if self.cfg.variant == "Precedence":
            # PR0 (class 0): robot_0 docks first. Ties are structurally
            # impossible (unique-mouth topology); dt == 0 scores Y = 0.
            dt = self.latch_time[:, 1].long() - self.latch_time[:, 0].long()
            return torch.where(self.instr_class == 0, dt > 0, dt < 0)
        # RoleBinding: RB0 (class 0): robot_0 -> LEFT station (smaller col).
        # The bank randomises which physical alcove occupies slot 0, so
        # left/right is resolved geometrically per scenario.
        col = self.target[..., 1].float()
        left = torch.where(self.target_valid, col,
                           torch.full_like(col, float("inf"))).argmin(dim=1)
        right = torch.where(self.target_valid, col,
                            torch.full_like(col, float("-inf"))).argmax(dim=1)
        instructed = torch.where(self.instr_class == 0, left, right)
        return self.latch_slot[:, 0].long() == instructed

    def _get_rewards(self):
        N = L.N_AGENTS
        done = self.latched[:, :N].all(dim=1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # gamma-correct potential shaping (spec 1.12 [FIXED: not
        # policy-invariant]): lambda * (gamma * Phi_t - Phi_{t-1}),
        # Phi(terminal) == 0 (identically true on the both-latched terminal;
        # made explicit here). Timeout is truncation, not termination: Phi
        # stays live and time_limit_bootstrap handles the value.
        phi_now = torch.where(done, torch.zeros_like(self._phi), self._phi)
        shaping = self.cfg.shaping_lambda * (
            self.cfg.reward_gamma * phi_now - self._phi_prev)

        y = self._outcome_correct()
        if self.cfg.blind_expected_bonus:
            # Blind/Mute: exact conditional expectation E[10*Y | C=1] = 5.0
            # (spec 1.12 [FIXED: blind oracle vs +-10]); bias-free, kills a
            # variance-25 terminal term the policy cannot predict.
            bonus = torch.full_like(shaping, BLIND_EXPECTED_BONUS)
        else:
            bonus = OUTCOME_BONUS * y.float()

        team = STEP_COST + shaping + torch.where(
            done, COMPLETION_BONUS + bonus, torch.zeros_like(shaping))

        rewards = {}
        for i, name in enumerate(self.cfg.possible_agents):
            rewards[name] = (team
                             + COLLISION_COST * self._hit_obstacle[:, i].float()
                             + COLLISION_COST * self._hit_robot[:, i].float()
                             + FIRST_LATCH_BONUS
                             * self._newly_latched[:, i].float())

        if self.cfg.debug_asserts and self.cfg.variant == "Precedence":
            # spec 1.11 [FIXED: dead gap code]: |dt| >= 1 is structural under
            # the unique-mouth topology; verify on every terminal.
            gap = (self.latch_time[:, 1].long()
                   - self.latch_time[:, 0].long()).abs()
            assert bool((gap[done] >= 1).all()), \
                "latch tie observed: unique-mouth topology violated (spec 3.1)"

        self._log_step(done, time_out, y)
        return rewards

    def _log_step(self, done, time_out, y):
        """Scalar diagnostics via extras['log'] (spec 1.14; trainer
        environment_info: log). Branch-step entropy is policy-side and is
        computed in harness/rollout.py, not here."""
        N = L.N_AGENTS
        n_done = done.float().sum().clamp(min=1.0)
        gap = (self.latch_time[:, 1].long() - self.latch_time[:, 0].long())
        if self.cfg.variant == "Precedence":
            assign = (gap > 0)                     # realised: robot_0 first
        else:
            col = self.target[..., 1].float()
            left = torch.where(self.target_valid, col,
                               torch.full_like(col, float("inf"))).argmin(dim=1)
            assign = self.latch_slot[:, 0].long() == left   # r0 -> left
        lang_l2 = self.lang_vec[:, :N].norm(dim=-1).mean()
        state_lang_l2 = self.lang_vec.reshape(self.num_envs, -1).norm(dim=-1).mean()
        log = self.extras.setdefault("log", {})
        log["completion_rate"] = done.float().mean()
        # curriculum diagnostic (DECISIONS 2026-09-01): separates "agents
        # rarely latch at all" from "one latches, the second never joins"
        n_latched = (self.latched & self.agent_valid).sum(dim=1)
        log["single_latch_share"] = (n_latched == 1).float().mean()
        log["outcome_correct_share"] = (done & y).float().mean()
        log["outcome_wrong_share"] = (done & ~y).float().mean()
        log["outcome_incomplete_share"] = (time_out & ~done).float().mean()
        log["latch_gap_mean"] = (gap.abs().float() * done.float()).sum() / n_done
        log["assign_marginal"] = (assign & done).float().sum() / n_done
        log["lang_slice_l2"] = lang_l2
        log["state_lang_slice_l2"] = state_lang_l2

    # ------------------------------------------------------------------
    # Reset (spec 1.10: pure gather; RNG only for WHICH row / instruction)
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._ALL
        elif not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long,
                                      device=self.device)
        super()._reset_idx(env_ids)

        N = L.N_AGENTS
        bank = self._bank
        sid = self._draw_scenarios(env_ids)
        self.scenario_id[env_ids] = sid

        # pure gathers from the frozen bank
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

        # fog of war: all-unknown, then the initial reveal at spawn (spec 1.9;
        # reference reset() semantics). Advanced indexing copies, so reveal
        # into the copies and write back.
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

        self._assert_arm_consistency(env_ids)      # UNCONDITIONAL (spec 1.6)

    def _draw_scenarios(self, env_ids):
        n = env_ids.numel()
        if self._forced_scenarios is not None:
            return self._forced_scenarios[env_ids]
        if self.cfg.eval_mode:
            # deterministic manifest cursor (spec 1.10)
            m = self._eval_pool.numel()
            idx = (self._eval_cursor
                   + torch.arange(n, device=self.device)) % m
            self._eval_cursor = int((self._eval_cursor + n) % m)
            return self._eval_pool[idx]
        pool = self._train_pool
        return pool[torch.randint(0, pool.numel(), (n,), device=self.device)]

    def _draw_instructions(self, env_ids, leak_bit, leak_rho):
        """(instr_id (n, MAX_AGENTS), instr_class (n,)) per spec 1.10 [FIXED].

        Class: I = leak_bit with probability rho, uniform otherwise -- the
        canary lives on THIS audited path; rho != 0 is representable only for
        the Leaky arm (cfg gate).
        """
        n = env_ids.numel()
        dev = self.device
        cls = torch.randint(0, 2, (n,), device=dev)
        if leak_rho > 0.0:
            take = torch.rand(n, device=dev) < leak_rho
            cls = torch.where(take, leak_bit.long(), cls)

        def draw_rows(c):
            if self._lang_mode_is("lang"):
                r0, r1 = self._rows_by_class
                i0 = torch.randint(0, r0.numel(), (n,), device=dev)
                i1 = torch.randint(0, r1.numel(), (n,), device=dev)
                return torch.where(c == 0, r0[i0], r1[i1])
            if self._lang_mode_is("placebo"):
                i = torch.randint(0, self._placebo_rows.numel(), (n,), device=dev)
                return self._placebo_rows[i]        # relabelled: class-independent
            return c.clone()        # symbol / zero: row == class id (2-code table)

        if self.cfg.per_agent_instruction:
            # month-4 per-agent atomic prompting (spec 1.5 taxonomy cell):
            # same class, independently drawn surface form per agent slot.
            rows = torch.stack([draw_rows(cls) for _ in range(L.MAX_AGENTS)], 1)
        else:
            # M1 global broadcast: the SAME row in every agent slot.
            rows = draw_rows(cls).unsqueeze(1).expand(n, L.MAX_AGENTS).contiguous()
        return rows, cls

    def _lang_mode_is(self, mode):
        return self.cfg.lang_mode == mode

    def _write_lang_vec(self, env_ids, instr_id):
        vec = self._lang_table[instr_id] * float(self.cfg.lang_gain)
        # padded agent slots stay zero (spec 1.3: extra slots zeroed at N=2)
        vec = vec * self.agent_valid[env_ids].unsqueeze(-1).float()
        self.lang_vec[env_ids] = vec

    def _assert_arm_consistency(self, env_ids):
        if self._arm_consistency_fn is not None:
            self._arm_consistency_fn(self, env_ids)
            return
        # Fallback (harness/arms.py not present yet): the spec 1.6 invariant
        # verbatim -- ||LANG|| == 0 for Blind/Leaky/Mute, > 0 otherwise.
        # One reduction per reset; unconditional by design.
        l2 = float(self.lang_vec[env_ids].square().sum().item())
        if self.cfg.arm in ("Blind", "Leaky", "Mute"):
            assert l2 == 0.0, (
                "arm %r must have a zero language slice; ||lang||^2=%r "
                "(M1_SPEC 1.6 assert_arm_consistency)" % (self.cfg.arm, l2))
        else:
            assert l2 > 0.0, (
                "arm %r must have a nonzero language slice; got 0 "
                "(M1_SPEC 1.6 assert_arm_consistency)" % (self.cfg.arm,))

    # ------------------------------------------------------------------
    # Harness hooks (spec 6.3 five-lane batching drives env.unwrapped)
    # ------------------------------------------------------------------

    def force_scenarios(self, scenario_ids):
        """Pin per-env scenario rows for the next resets (None to release)."""
        if scenario_ids is not None:
            scenario_ids = scenario_ids.to(self.device, torch.long)
            assert scenario_ids.numel() == self.num_envs
        self._forced_scenarios = scenario_ids

    def force_slip_stream(self, streams):
        """Pin per-env slip streams for the next resets (None to release)."""
        if streams is not None:
            streams = streams.to(self.device, torch.int8)
            assert streams.numel() == self.num_envs
        self._forced_slip_stream = streams

    # ------------------------------------------------------------------
    # Observations (spec 1.5) and critic state (spec 1.6) -- hand-written
    # ------------------------------------------------------------------

    def _common_features(self):
        """Per-step tensors shared by the obs and state builders."""
        E = self.num_envs
        N = L.N_AGENTS
        f = SimpleNamespace()
        f.kf = self.known_free.reshape(E, L.P).float()
        f.ko = self.known_obs.reshape(E, L.P).float()
        f.occ = self.occ.reshape(E, L.P).float()
        f.presentf = self.agent_valid.float()                       # (E, A)
        f.pr = L.norm_pos(self.pos[..., 0].float(), L.R)            # (E, A)
        f.pc = L.norm_pos(self.pos[..., 1].float(), L.C)
        f.latchedf = self.latched.float()
        slot = self.latch_slot.long().clamp(min=0)
        oh = F.one_hot(slot, L.LATCH_SLOT_ONEHOT_W).float()
        f.slot_oh = oh * (self.latch_slot >= 0).unsqueeze(-1).float()  # (E, A, 4)
        # latch-time/T, literal: -1 (unlatched) -> -1/T (spec 1.5 "latch-time/T")
        f.ltime = self.latch_time.float() / L.T_DECISION            # (E, A)
        f.tvf = self.target_valid.float()                           # (E, T)
        f.tpr = L.norm_pos(self.target[..., 0].float(), L.R) * f.tvf
        f.tpc = L.norm_pos(self.target[..., 1].float(), L.C) * f.tvf
        # occupancy: robot i sits on valid target j (a robot on a target is
        # latched by the time observations are built)
        match = (self.target.long().unsqueeze(2)
                 == self.pos.long().unsqueeze(1)).all(-1)           # (E, T, A)
        match = match & self.agent_valid.unsqueeze(1) & self.target_valid.unsqueeze(2)
        f.occupier = match.float()                                  # (E, T, A)
        f.occupied = match.any(-1).float()                          # (E, T)
        f.occ_ltime = (f.occupier * f.ltime.unsqueeze(1)).sum(-1)   # (E, T)
        f.t_norm = (self.episode_length_buf.float() / L.T_DECISION).unsqueeze(-1)
        lt = self.latch_time[:, :N].float()
        first = torch.where(self.latched[:, :N], lt,
                            torch.full_like(lt, float("inf"))).min(dim=1).values
        has = torch.isfinite(first)
        zeros = torch.zeros_like(first)
        f.since_latch = torch.where(
            has, (self.episode_length_buf.float() - first) / L.T_DECISION,
            zeros).unsqueeze(-1)                                    # obs field
        f.first_latch = torch.where(has, first / L.T_DECISION,
                                    zeros).unsqueeze(-1)            # state field
        return f

    def _build_state(self, f=None):
        """(E, STATE_DIM=641) critic state per obs_layout's state table."""
        if f is None:
            f = self._common_features()
        E = self.num_envs
        pieces = [f.occ, f.kf, f.ko]                        # privileged + fog
        for j in range(L.MAX_AGENTS):
            p = f.presentf[:, j:j + 1]
            pieces += [
                p,
                (f.pr[:, j:j + 1]) * p,
                (f.pc[:, j:j + 1]) * p,
                f.latchedf[:, j:j + 1],
                f.slot_oh[:, j],
                f.ltime[:, j:j + 1] * p,
                self._agent_id_oh[j].expand(E, L.MAX_AGENTS) * p,
            ]
        for k in range(L.MAX_TARGETS):
            pieces += [
                f.tvf[:, k:k + 1],
                f.tpr[:, k:k + 1],
                f.tpc[:, k:k + 1],
                f.occupied[:, k:k + 1],
                f.occupier[:, k],                           # one-hot over agent slots
                f.occ_ltime[:, k:k + 1],
            ]
        pieces += [f.t_norm, f.first_latch,
                   self.lang_vec.reshape(E, L.MAX_AGENTS * L.LANG_DIM)]
        state = torch.cat(pieces, dim=-1)
        assert state.shape == (E, L.STATE_DIM), state.shape
        return state

    def _build_obs_agent(self, i, f):
        """(E, OBS_DIM=376) partial observation for live agent slot ``i``."""
        E = self.num_envs
        pieces = [f.kf, f.ko]
        # ego block (12)
        pieces += [
            f.pr[:, i:i + 1], f.pc[:, i:i + 1],
            f.latchedf[:, i:i + 1], f.slot_oh[:, i], f.ltime[:, i:i + 1],
            self._agent_id_oh[i].expand(E, L.MAX_AGENTS),
        ]
        # 3 teammate blocks (9 each): non-ego slots in ascending index order
        for j in range(L.MAX_AGENTS):
            if j == i:
                continue
            p = f.presentf[:, j:j + 1]
            pieces += [
                p,
                f.pr[:, j:j + 1] * p,
                f.pc[:, j:j + 1] * p,
                f.latchedf[:, j:j + 1],
                f.slot_oh[:, j],
                f.ltime[:, j:j + 1] * p,
            ]
        # 3 target blocks (5 each)
        for k in range(L.MAX_TARGETS):
            pieces += [
                f.tvf[:, k:k + 1],
                f.tpr[:, k:k + 1],
                f.tpc[:, k:k + 1],
                f.occupied[:, k:k + 1],
                f.occupier[:, k, i:i + 1],                  # occupied-by-ego
            ]
        pieces += [f.t_norm, f.since_latch, self.lang_vec[:, i]]
        obs = torch.cat(pieces, dim=-1)
        assert obs.shape == (E, L.OBS_DIM), obs.shape
        return obs

    def _get_observations(self):
        """Always emits BOTH agents (spec 1.11: a missing key would silently
        change the state layout mid-episode)."""
        f = self._common_features()
        E = self.num_envs
        if self.cfg.actor_observes_state:
            # full-state arms (spec 4.1): 641-d state + own agent-id one-hot
            state = self._build_state(f)
            obs = {}
            for i, name in enumerate(self.cfg.possible_agents):
                aid = self._agent_id_oh[i].expand(E, L.MAX_AGENTS)
                o = torch.cat([state, aid], dim=-1)
                assert o.shape == (E, L.FULL_STATE_OBS_DIM), o.shape
                obs[name] = o
            return obs
        return {name: self._build_obs_agent(i, f)
                for i, name in enumerate(self.cfg.possible_agents)}

    def _get_states(self):
        # Implemented even when unused (@abstractmethod, spec 1.6); the
        # critic state carries the (possibly zeroed) per-agent language
        # slots so every arm has byte-identical critic input shapes.
        return self._build_state()


__all__ = ["TeamGridEnv", "STEP_COST", "COLLISION_COST", "COMPLETION_BONUS",
           "OUTCOME_BONUS", "BLIND_EXPECTED_BONUS"]
