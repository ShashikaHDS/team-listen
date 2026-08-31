"""Isaac Lab env configs for the Team Listen grid fleet (M1_SPEC 1.4, 4.1, 7).

One ``@configclass`` base (``TeamGridEnvCfg``) plus the 14 concrete
variant x arm configs registered by ``tasks/team_listen/__init__.py``:

    variant in {RoleBinding, Precedence} x arm in {Lang, Blind, Symbol,
    SymbolPO, Leaky, Mute, Placebo}

Spec decisions hard-coded here (do not "fix" them without reading the spec):

* ``decimation = 1``, ``sim.dt = 0.1``, ``episode_length_s = 12.9``
  [FIXED: episode-length off-by-one]: Isaac Lab increments
  ``episode_length_buf`` BEFORE ``_get_dones()``, so
  ``max_episode_length = ceil(12.9 / 0.1) = 129`` yields exactly 128
  decision steps == ``T_DECISION``.  ``_episode_length_guard`` re-derives
  that ceil (float rounding: 12.9/0.1 = 128.99999999999997 -> ceil 129)
  at import AND in ``__post_init__``, and ``TeamGridEnv.__init__`` asserts
  the live ``max_episode_length`` too; any dt/decimation edit crashes
  loudly instead of silently rescaling every cross-condition comparison.
  (Documented fallback if the pinned Isaac Lab computes 128 instead of
  129: episode_length_s = 12.8 with T_DECISION = 127 -- see the spec's
  "resolvable only on the 5090" list.)
* ``action_spaces = {agent: {5}}`` -- a SET of length one is what
  ``spec_to_gym_space`` maps to ``Discrete(5)`` (a list ``[{5},{5}]``
  would give ``MultiDiscrete``).
* ``state_space`` is a POSITIVE int (spec 1.4): ``0`` is a documented trap
  and ``-1`` would auto-concatenate agent observations, duplicating the
  language slice in the critic and making blindness implicit.
* ``action_noise_model = None`` MUST stay None: noise is applied before
  ``_pre_physics_step`` and would turn an integer action index into float
  garbage.
* ``scene.filter_collisions = False`` [FIXED: filter_collisions]:
  ``clone_environments()`` otherwise passes ``enable_env_ids=True`` into
  the cloner, requesting PhysX env-id collision filtering on a
  collider-free scene.
* PhysX GPU buffers are minimised: the scene holds ZERO physics assets
  (spec 1.1), so the default multi-MB contact/pair buffers are waste.
  Values are verified/adjustable on the 5090 only.
* ``instruction_in_obs`` / ``instruction_in_state`` DO NOT EXIST as
  settable fields [FIXED: arm consistency]: everything language-related
  (``lang_gain``, ``lang_mode``, ``actor_observes_state``,
  ``blind_expected_bonus``, the ``leak_rho`` settability) is DERIVED from
  the arm name in ``__post_init__`` via ``ARM_SPECS``, so the illegal
  combinations are unrepresentable.

Import guard: ``isaaclab`` is not importable on the Windows dev box (and
not importable on the 5090 before ``SimulationApp`` boots).  This module
must still import for pure-python tests and for ``scripts/train.py``'s
pre-boot bookkeeping, so all isaaclab imports are guarded with inert
shims.  NEVER import this module on the 5090 before the app boots from
code that will later ``gym.make`` the env in the same process -- the
shimmed classes would be cached in ``sys.modules``.  ``fleet_env`` raises
at construction when the shim is active.
"""

import math
from typing import Dict, List, NamedTuple

from .obs_layout import (
    FULL_STATE_OBS_DIM,
    OBS_DIM,
    STATE_DIM,
    T_DECISION,
)

# ---------------------------------------------------------------------------
# Guarded Isaac Lab imports (see module docstring).
# ---------------------------------------------------------------------------

try:
    from isaaclab.envs import DirectMARLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import PhysxCfg, SimulationCfg
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
    ISAACLAB_IMPORT_ERROR = None
except Exception as _exc:  # ImportError, or partial omni failures pre-app
    ISAACLAB_AVAILABLE = False
    ISAACLAB_IMPORT_ERROR = _exc

    def configclass(cls):  # type: ignore[misc]
        """Inert stand-in: leaves the class untouched (dev box only)."""
        return cls

    class _ShimCfg:
        """Kwargs bag so class bodies below still evaluate without isaaclab."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DirectMARLEnvCfg(_ShimCfg):  # type: ignore[no-redef]
        pass

    class InteractiveSceneCfg(_ShimCfg):  # type: ignore[no-redef]
        pass

    class SimulationCfg(_ShimCfg):  # type: ignore[no-redef]
        pass

    class PhysxCfg(_ShimCfg):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Episode-length float-rounding guard (spec 1.4 [FIXED] + 5090 open question)
# ---------------------------------------------------------------------------

def _episode_length_guard(episode_length_s, dt, decimation):
    """Replicate Isaac Lab's ``max_episode_length`` ceil arithmetic exactly.

    ``DirectMARLEnv.max_episode_length = math.ceil(episode_length_s /
    (sim.dt * decimation))``.  We require exactly ``T_DECISION + 1`` (129):
    the shipped ``>= max_episode_length - 1`` idiom then yields exactly
    ``T_DECISION`` (128) decision steps.
    """
    steps = math.ceil(episode_length_s / (dt * decimation))
    if steps != T_DECISION + 1:
        raise ValueError(
            "episode_length_s=%r, dt=%r, decimation=%r give "
            "max_episode_length=%d, need exactly T_DECISION+1=%d for %d "
            "decision steps (M1_SPEC 1.4 [FIXED: episode-length "
            "off-by-one]; fallback: episode_length_s=12.8 with "
            "T_DECISION=127)" % (episode_length_s, dt, decimation, steps,
                                 T_DECISION + 1, T_DECISION)
        )
    return steps


# Import-time self-check of the spec constants (pure math, runs everywhere):
# ceil(12.9 / 0.1) = ceil(128.99999999999997) = 129.
_episode_length_guard(12.9, 0.1, 1)


# ---------------------------------------------------------------------------
# Arm registry mirror (spec 4.1 table; runtime authority is config/arms.yaml
# via harness/arms.py -- keep the two in sync).
# ---------------------------------------------------------------------------

class ArmSpec(NamedTuple):
    """Per-arm derivations (spec 4.1 / 1.6 / 1.12).

    full_state:        actor observes the 641-d state + agent-id (645) instead
                       of the partial 376-d observation (spec 4.1; implemented
                       as cfg.actor_observes_state, NEVER via the broken
                       multi_agent_to_single_agent shim).
    lang_mode:         what fills LANG_SLICE: "lang" (cached projected MiniLM),
                       "symbol" (2 orthonormal codes), "placebo" (relabelled
                       cache row), "zero" (gain-zeroed).
    lang_gain:         multiplier on the LANG_TABLE gather (spec 1.10);
                       0.0 <=> the slice is exactly zero (Blind/Leaky/Mute).
    leak_rho_settable: only the Leaky arm may set cfg.leak_rho != 0
                       (spec 1.10 [FIXED: guards]).
    expected_bonus:    Blind/Mute replace the +-10*Y outcome bonus with its
                       exact conditional expectation +5.0 (spec 1.12 [FIXED]);
                       every other arm (ALL Leaky rho cells included) keeps
                       the real stochastic bonus.
    """

    full_state: bool
    lang_mode: str
    lang_gain: float
    leak_rho_settable: bool
    expected_bonus: bool


ARM_SPECS = {
    "Lang":     ArmSpec(False, "lang",    1.0, False, False),
    "Blind":    ArmSpec(True,  "zero",    0.0, False, True),
    "Symbol":   ArmSpec(True,  "symbol",  1.0, False, False),
    "SymbolPO": ArmSpec(False, "symbol",  1.0, False, False),
    "Leaky":    ArmSpec(True,  "zero",    0.0, True,  False),
    "Mute":     ArmSpec(False, "zero",    0.0, False, True),
    "Placebo":  ArmSpec(False, "placebo", 1.0, False, False),
}

VARIANTS = ("RoleBinding", "Precedence")

#: Registry-name -> harness/templates.py variant key (lang-cache rows).
VARIANT_TO_TEMPLATE_KEY = {
    "RoleBinding": "role_binding",
    "Precedence": "precedence",
}


# ---------------------------------------------------------------------------
# Base config (spec 1.4, verbatim where the spec gives code)
# ---------------------------------------------------------------------------

@configclass
class TeamGridEnvCfg(DirectMARLEnvCfg):
    """Base config for TeamGridEnv. Concrete arm/variant classes below."""

    # -- core Isaac Lab fields (spec 1.4) ---------------------------------
    decimation: int = 1
    episode_length_s: float = 12.9      # -> max_episode_length = 129 -> 128 decision steps
    is_finite_horizon: bool = False     # + time_limit_bootstrap: True in the agent YAML
    possible_agents: List[str] = ["robot_0", "robot_1"]   # STABLE ORDER -- load-bearing
    action_spaces: Dict[str, object] = {"robot_0": {5}, "robot_1": {5}}   # {5} -> Discrete(5)
    # Overwritten per-arm in __post_init__ (full-state arms observe 645).
    observation_spaces: Dict[str, int] = {"robot_0": OBS_DIM, "robot_1": OBS_DIM}
    state_space: int = STATE_DIM        # POSITIVE int; never 0, never -1 (spec 1.4)
    action_noise_model: object = None   # MUST stay None (spec 1.4)

    # Zero-asset scene: one simulate/fetch over zero actors per step.
    sim: SimulationCfg = SimulationCfg(
        dt=0.1,
        render_interval=1,
        physx=PhysxCfg(
            # GPU buffers minimised for a scene with ZERO physics assets
            # (spec 1.4 "<GPU buffers minimised>"); values validated on the
            # 5090 only -- raise them there if PhysX complains, never here.
            gpu_max_rigid_contact_count=2 ** 10,
            gpu_max_rigid_patch_count=2 ** 10,
            gpu_found_lost_pairs_capacity=2 ** 10,
            gpu_found_lost_aggregate_pairs_capacity=2 ** 10,
            gpu_total_aggregate_pairs_capacity=2 ** 10,
            gpu_collision_stack_size=2 ** 16,
            gpu_heap_capacity=2 ** 20,
            gpu_temp_buffer_capacity=2 ** 16,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        filter_collisions=False,        # [FIXED: filter_collisions] spec 1.1
    )

    # -- task identity -----------------------------------------------------
    variant: str = "RoleBinding"        # "RoleBinding" | "Precedence"
    arm: str = "Blind"                  # key into ARM_SPECS

    # -- world / instruction knobs (spec 1.3, 1.5, 1.10) -------------------
    lidar_radius: int = 1               # Chebyshev window radius (spec 1.9)
    per_agent_instruction: bool = False  # M1: global broadcast (spec 1.5)
    leak_rho: float = 0.0               # settable ONLY by the Leaky arm
    slip_stream: int = 0                # default stored stochasticity stream

    # -- reward constants (spec 1.12) --------------------------------------
    shaping_lambda: float = 0.1         # swept {0.05, 0.1, 0.2} pre-headline
    reward_gamma: float = 0.99          # MUST equal the YAML discount_factor

    # -- artifacts ---------------------------------------------------------
    # "" -> fall back to $TEAM_LISTEN_BANK / $TEAM_LISTEN_LANG_CACHE env vars
    # (set by scripts/train.py callers), then error.
    bank_path: str = ""
    bank_sha: str = ""                  # expected SHA-256 (prefix ok); "" = skip
    allow_synthetic_bank: bool = False  # plumbing-smoke only, NEVER results
    lang_cache_path: str = ""
    lang_cache_sha: str = ""            # expected artifact_sha256; "" = skip

    # -- evaluation --------------------------------------------------------
    eval_mode: bool = False             # scenario draw: manifest cursor, not randint
    eval_manifest_path: str = ""        # tensor of scenario ids; "" -> split==1 rows

    # -- debug -------------------------------------------------------------
    debug_vis: bool = False             # spec 1.13; train.py force-disables
    debug_asserts: bool = False         # per-step asserts (render leak, |dt|>=1)

    # -- DERIVED from `arm` in __post_init__; NEVER set these by hand ------
    lang_gain: float = 0.0
    lang_mode: str = "zero"
    actor_observes_state: bool = False
    blind_expected_bonus: bool = False

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if parent_post is not None:
            parent_post()

        # Arm-derived fields: unconditional overwrite so illegal combinations
        # are unrepresentable (spec 1.6 [FIXED: arm consistency]).
        if self.arm not in ARM_SPECS:
            raise ValueError("unknown arm %r; legal: %r"
                             % (self.arm, sorted(ARM_SPECS)))
        if self.variant not in VARIANTS:
            raise ValueError("unknown variant %r; legal: %r"
                             % (self.variant, VARIANTS))
        spec = ARM_SPECS[self.arm]
        self.lang_gain = spec.lang_gain
        self.lang_mode = spec.lang_mode
        self.actor_observes_state = spec.full_state
        self.blind_expected_bonus = spec.expected_bonus

        # leak_rho is a Leaky-only knob (spec 1.10 [FIXED: guards]).
        if not 0.0 <= float(self.leak_rho) <= 1.0:
            raise ValueError("leak_rho=%r outside [0, 1]" % (self.leak_rho,))
        if float(self.leak_rho) != 0.0 and not spec.leak_rho_settable:
            raise ValueError(
                "cfg.leak_rho=%r is settable ONLY by the Leaky arm "
                "(M1_SPEC 1.10); arm=%r must keep 0.0"
                % (self.leak_rho, self.arm))

        # Observation width follows the arm (spec 4.1: full-state arms 645).
        obs_dim = FULL_STATE_OBS_DIM if spec.full_state else OBS_DIM
        self.observation_spaces = {a: obs_dim for a in self.possible_agents}

        # Spec 1.4 invariants.
        if list(self.possible_agents) != ["robot_0", "robot_1"]:
            raise ValueError(
                "possible_agents must be ['robot_0', 'robot_1'] in stable "
                "order (M1_SPEC 1.4, load-bearing); got %r"
                % (self.possible_agents,))
        if self.state_space != STATE_DIM:
            raise ValueError("state_space must be STATE_DIM=%d (positive int, "
                             "spec 1.4); got %r" % (STATE_DIM, self.state_space))
        if self.action_noise_model is not None:
            raise ValueError("action_noise_model MUST stay None (spec 1.4): "
                             "noise runs before _pre_physics_step and would "
                             "corrupt integer action indices")
        _episode_length_guard(self.episode_length_s, self.sim.dt,
                              self.decimation)


# ---------------------------------------------------------------------------
# The 14 concrete variant x arm configs (spec section 7 file plan).
# Registered by tasks/team_listen/__init__.py as
# "Isaac-TeamListen-{variant}-{arm}-Direct-v0".
# ---------------------------------------------------------------------------

@configclass
class RoleBindingLangEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "Lang"


@configclass
class RoleBindingBlindEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "Blind"


@configclass
class RoleBindingSymbolEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "Symbol"


@configclass
class RoleBindingSymbolPOEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "SymbolPO"


@configclass
class RoleBindingLeakyEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "Leaky"
    # rho sweep cells {0.0, 0.55, 0.60, 0.70, 0.80} are launch-time overrides
    # (spec 4.1); the default is the matched internal control rho = 0.


@configclass
class RoleBindingMuteEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "Mute"


@configclass
class RoleBindingPlaceboEnvCfg(TeamGridEnvCfg):
    variant: str = "RoleBinding"
    arm: str = "Placebo"


@configclass
class PrecedenceLangEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "Lang"


@configclass
class PrecedenceBlindEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "Blind"


@configclass
class PrecedenceSymbolEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "Symbol"


@configclass
class PrecedenceSymbolPOEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "SymbolPO"


@configclass
class PrecedenceLeakyEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "Leaky"
    # NOTE: the canary sweep itself is RoleBinding-only (spec 4.1 table);
    # this cfg exists so the 2x7 registration grid is complete.


@configclass
class PrecedenceMuteEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "Mute"


@configclass
class PrecedencePlaceboEnvCfg(TeamGridEnvCfg):
    variant: str = "Precedence"
    arm: str = "Placebo"


__all__ = [
    "ISAACLAB_AVAILABLE", "ISAACLAB_IMPORT_ERROR",
    "ArmSpec", "ARM_SPECS", "VARIANTS", "VARIANT_TO_TEMPLATE_KEY",
    "TeamGridEnvCfg",
    "RoleBindingLangEnvCfg", "RoleBindingBlindEnvCfg",
    "RoleBindingSymbolEnvCfg", "RoleBindingSymbolPOEnvCfg",
    "RoleBindingLeakyEnvCfg", "RoleBindingMuteEnvCfg",
    "RoleBindingPlaceboEnvCfg",
    "PrecedenceLangEnvCfg", "PrecedenceBlindEnvCfg",
    "PrecedenceSymbolEnvCfg", "PrecedenceSymbolPOEnvCfg",
    "PrecedenceLeakyEnvCfg", "PrecedenceMuteEnvCfg",
    "PrecedencePlaceboEnvCfg",
]
