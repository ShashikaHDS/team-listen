"""DeepSets entity-encoder models for MAPPO (M1_SPEC 1.5 / section 7 / OPEN(4)).

WHY THIS EXISTS (DECISIONS.md 2026-09-02, round 5): the permanent floor +
warm start provably fixed the CRITIC layer (floor-trained critics price
both_adjacent correctly) yet certified competence still eroded from 0.098
to ~0.01 -- the failure is isolated to the POLICY layer: representation
interference in the shared 512-256-128 flat MLP, where easy-row gradients
reshape shared features away from long-approach competence.  Training-
distribution and optimiser levers are exhausted (LR, GAE depth, curriculum
x2, floor, warm start -- all falsified for that layer).  The ratified fix
candidate is spec 1.5's month-4 default pulled forward: a DeepSets-style
entity encoder over the SAME padded slots, whose inductive bias keeps
per-entity features in a shared phi (one gradient signal per entity kind,
not per slot position) and keeps the map, entity, scalar and language
streams in separate parameter blocks so easy-row map/entity gradients
cannot overwrite long-approach entity features wholesale.

WHAT IT IS
----------
``EntityEncoder`` (a plain ``torch.nn.Module``, importable and testable
with NO skrl and NO Isaac) splits the flat observation / state vector via
the named slices in ``tasks/team_listen/obs_layout.py`` -- the single
source of layout truth -- into streams:

* map planes  -> flatten-MLP embedding (see MAP ENCODER CHOICE below);
* padded entity slots (teammates/agents, targets) -> one SHARED per-entity
  MLP ``phi`` per entity kind, then presence-mask-weighted pooling
  (masked mean by default, masked sum selectable);
* the ego block (obs layouts), the time scalars, the 32-d ``LANG_SLICE``
  (obs layout) and the trailing agent-id one-hot (full-state layout)
  pass straight into the trunk;
* concat(map emb, pooled entities, ego, scalars, lang) -> trunk MLP.

Three layouts, selected by input width (all three actor/critic widths that
exist in the study -- spec 1.5, 1.6, 4.1):

* ``"obs"``            OBS_DIM            = 376  (partial-obs actors:
                                                  Lang/SymbolPO/Mute/Placebo)
* ``"state"``          STATE_DIM          = 641  (the central critic)
* ``"full_state_obs"`` FULL_STATE_OBS_DIM = 645  (full-state actors:
                                                  Blind/Symbol/Leaky --
                                                  state + own agent-id
                                                  one-hot; fleet_env
                                                  ``_get_observations``)

The 645 layout is included deliberately: the round-5 encoder probe runs on
the Blind competence arm, whose actor observes 645 dims -- an encoder that
only accepted 376 could not run the decided experiment.

PERMUTATION INVARIANCE (what is and is not invariant)
-----------------------------------------------------
* obs layout: teammate slots and target slots are pooled with a shared phi
  -> the representation is invariant to permuting slot ORDER within each
  group.  Teammate blocks carry no identity features (spec 1.5), so no
  information is discarded.  The EGO block is a separate, un-pooled stream
  and carries the agent-id one-hot, so per-agent identity is preserved
  exactly as the spec's ego block prescribes.
* state layouts: each 13-wide agent block is concatenated with ITS OWN
  32-d ``STATE_LANG_SLICE`` vector (the (agent, instruction) binding made
  explicit) before phi -- the representation is invariant under a JOINT
  permutation of (agent block, lang vector) pairs; the agent-id one-hot
  travels inside the block, so identity is not lost.  Caveat: the state
  target blocks contain an occupier one-hot indexed by agent SLOT, so slot
  indices are not fully erasable semantics -- the function-level invariance
  above still holds exactly (target blocks are untouched by the agent-slot
  permutation).

MAP ENCODER CHOICE (documented decision -- the spec is silent)
--------------------------------------------------------------
Spec 1.5 prescribes only "a DeepSets-style entity encoder over the same
slices"; neither 1.5 nor OPEN(4) mandates conv vs flatten-MLP for the map
planes (OPEN(4) is about the LANGUAGE projection: fixed JL vs trainable
vs FiLM).  Flatten-MLP is chosen because (a) the interference mechanism
being fixed (DECISIONS 2026-09-02) lives in the entity/feature streams,
not map encoding; (b) it matches the flat baseline's treatment of the
planes, isolating the entity inductive bias as the single changed
variable; (c) kernel-launch budget is the throughput currency (spec 1.8 /
8.1) and a conv stack adds launches for no hypothesised benefit.  A conv
variant would consume ``planes.view(B, n_planes, R, C)``; the split point
is ``EntityEncoder._encode_map``.

POOLING CHOICE (documented decision -- the spec is silent)
----------------------------------------------------------
Masked MEAN by default (``pool="mean"``), masked SUM selectable
(``pool="sum"``).  Spec 1.5's stated reason the encoder becomes default is
the held-out-team-size test; mean keeps the pooled embedding's scale
independent of the number of valid entities (N=2 -> N=4 is then a data
change, matching the padded-capacity philosophy), while sum -- canonical
DeepSets -- scales activations with team size.  Both are exercised by
tests/test_models.py.

PARAMETER-COUNT PARITY vs the flat 512-256-128 baseline (spec 1.14 YAML)
------------------------------------------------------------------------
Default cfg, measured (regenerate: ``python harness/models.py``):

  policy  obs(376)->5 :   227,333  vs flat  357,893   (0.64x)
  policy  full(645)->5:   256,517  vs flat  495,621   (0.52x)
  value   state(641)->1:  254,977  vs flat  493,057   (0.52x)

Same order of magnitude by design; ``parameter_parity_report()``
recomputes these live and tests/test_models.py asserts the parity band,
so the numbers above cannot silently drift.

SKRL WIRING (the 5090 session owns the training glue)
-----------------------------------------------------
skrl imports are GUARDED: this module imports cleanly without skrl (the
dev box has none); ``CategoricalEntityPolicy`` / ``DeterministicEntityValue``
/ ``build_entity_models`` raise a clear ImportError only on use.  The YAML
``models:`` block cannot express these classes (skrl's model instantiator
builds flat MLPs only), so the Runner path is REPLACED by manual agent
construction -- see ``build_entity_models`` for the exact recipe.

MRO note: the task sheet writes ``CategoricalEntityPolicy(Model,
CategoricalMixin)``; the actual bases are ``(CategoricalMixin, Model)`` --
skrl's documented convention -- because ``Model.act`` raises
NotImplementedError and the mixin's ``act`` must win the MRO.

Month-2 ablations from the section-7 file plan (trainable nn.Linear(384,32)
projection, FiLM conditioning) are NOT implemented here yet -- OPEN(4)'s
default (fixed JL projection, language as a plain input stream) is in
force; this file is their future home.
"""

import os
import sys

import torch
import torch.nn as nn

try:
    from tasks.team_listen import obs_layout as L
except ImportError:  # standalone import: put the repo root on sys.path
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from tasks.team_listen import obs_layout as L

# ---------------------------------------------------------------------------
# Guarded skrl import (spec section 0 hard-gates skrl >= 2.0.0 at runtime on
# the 5090; the dev box has NO skrl and this module must import cleanly).
# ---------------------------------------------------------------------------

try:
    from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
    SKRL_AVAILABLE = True
except ImportError:
    CategoricalMixin = DeterministicMixin = Model = None
    SKRL_AVAILABLE = False

_SKRL_MISSING_MSG = (
    "skrl is not installed: %s requires skrl >= 2.0.0 (M1_SPEC section 0 "
    "hard gate). The raw encoder (harness.models.EntityEncoder) works "
    "without skrl; only the skrl Model subclasses and build_entity_models "
    "need it. Install skrl on the training box (the 5090's Isaac python "
    "already has 2.1.0 per config/environment.lock.yaml)."
)


# ---------------------------------------------------------------------------
# Actions (spec 1.7: Discrete(5) -- the grid_core delta set)
# ---------------------------------------------------------------------------

N_ACTIONS = 5

#: input width -> layout name (the three widths that exist in the study)
LAYOUT_BY_DIM = {
    L.OBS_DIM: "obs",
    L.STATE_DIM: "state",
    L.FULL_STATE_OBS_DIM: "full_state_obs",
}

# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------

#: Default hyper-parameters. Tuple entries are MLP hidden sizes; the LAST
#: entry is the stream's embedding width. Overridable per-key via the
#: ``cfg`` argument of EntityEncoder / build_entity_models; unknown keys
#: raise (typo guard).
DEFAULT_CFG = {
    "map_hidden": (256, 128),    # flatten-MLP over the concatenated planes
    "phi_hidden": (64, 64),      # SHARED per-entity MLP, per entity kind
    "trunk_hidden": (256, 128),  # post-concat trunk; last = encoder out_dim
    "pool": "mean",              # "mean" (default, see docstring) or "sum"
    "activation": "elu",         # matches the flat baseline (spec 1.14)
}

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def resolve_cfg(cfg=None):
    """Merge ``cfg`` over DEFAULT_CFG; reject unknown keys and bad values."""
    out = dict(DEFAULT_CFG)
    if cfg:
        unknown = sorted(set(cfg) - set(DEFAULT_CFG))
        if unknown:
            raise ValueError(
                "unknown EntityEncoder cfg keys %r (valid: %r)"
                % (unknown, sorted(DEFAULT_CFG)))
        out.update(cfg)
    if out["pool"] not in ("mean", "sum"):
        raise ValueError("pool must be 'mean' or 'sum', got %r" % (out["pool"],))
    if out["activation"] not in _ACTIVATIONS:
        raise ValueError("activation must be one of %r, got %r"
                         % (sorted(_ACTIVATIONS), out["activation"]))
    for key in ("map_hidden", "phi_hidden", "trunk_hidden"):
        sizes = tuple(int(w) for w in out[key])
        if not sizes or any(w <= 0 for w in sizes):
            raise ValueError("%s must be a non-empty tuple of positive ints, "
                             "got %r" % (key, out[key]))
        out[key] = sizes
    return out


def _mlp(in_dim, hidden, activation):
    """[in -> h1 -> ... -> hk], activation after EVERY layer (the last
    hidden width is the stream's embedding, kept nonlinear like the
    baseline instantiator's hidden stack)."""
    act = _ACTIVATIONS[activation]
    layers, prev = [], in_dim
    for width in hidden:
        layers += [nn.Linear(prev, width), act()]
        prev = width
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Layout stream tables (all reads go through obs_layout's named slices --
# the input contract; nothing here re-derives an offset)
# ---------------------------------------------------------------------------

class _LayoutStreams(object):
    """Slice bookkeeping for one input layout. Pure data, torch-free."""

    def __init__(self, layout):
        if layout in ("state", "full_state_obs"):
            self.map_slices = [L.STATE_OBSTACLE_SLICE, L.STATE_KNOWN_FREE_SLICE,
                               L.STATE_KNOWN_OBS_SLICE]
            self.ego_slice = None
            self.agent_block = (L.STATE_AGENTS_SLICE.start, L.MAX_AGENTS,
                                L.STATE_AGENT_W)
            self.agent_present_off = 0            # spec 1.6 block order
            # (agent, instruction) binding: each agent block is paired with
            # its own STATE_LANG_SLICE vector before phi (module docstring).
            self.agent_lang_block = (L.STATE_LANG_SLICE.start, L.MAX_AGENTS,
                                     L.LANG_DIM)
            self.target_block = (L.STATE_TARGETS_SLICE.start, L.MAX_TARGETS,
                                 L.STATE_TARGET_W)
            self.target_present_off = L.TGT_PRESENT_OFF
            self.scalar_slices = [L.STATE_TIME_SLICE]
            self.lang_slice = None                # folded into agent entities
            self.dim = L.STATE_DIM
            if layout == "full_state_obs":
                # trailing own agent-id one-hot (fleet_env._get_observations)
                self.scalar_slices = [L.STATE_TIME_SLICE,
                                      slice(L.STATE_DIM, L.FULL_STATE_OBS_DIM)]
                self.dim = L.FULL_STATE_OBS_DIM
        elif layout == "obs":
            self.map_slices = [L.OBS_KNOWN_FREE_SLICE, L.OBS_KNOWN_OBS_SLICE]
            self.ego_slice = L.OBS_EGO_SLICE      # un-pooled: keeps agent-id
            self.agent_block = (L.OBS_TEAMMATES_SLICE.start,
                                L.N_TEAMMATE_SLOTS, L.OBS_TEAMMATE_W)
            self.agent_present_off = L.TM_PRESENT_OFF
            self.agent_lang_block = None
            self.target_block = (L.OBS_TARGETS_SLICE.start, L.MAX_TARGETS,
                                 L.OBS_TARGET_W)
            self.target_present_off = L.TGT_PRESENT_OFF
            self.scalar_slices = [L.OBS_TIME_SLICE]
            self.lang_slice = L.LANG_SLICE        # ego agent's own 32-d
            self.dim = L.OBS_DIM
        else:
            raise ValueError("unknown layout %r (valid: %r)"
                             % (layout, sorted(LAYOUT_BY_DIM.values())))
        self.layout = layout

    @property
    def map_in(self):
        return sum(sl.stop - sl.start for sl in self.map_slices)

    @property
    def agent_phi_in(self):
        width = self.agent_block[2]
        if self.agent_lang_block is not None:
            width += self.agent_lang_block[2]
        return width

    @property
    def scalar_w(self):
        w = sum(sl.stop - sl.start for sl in self.scalar_slices)
        if self.ego_slice is not None:
            w += self.ego_slice.stop - self.ego_slice.start
        if self.lang_slice is not None:
            w += self.lang_slice.stop - self.lang_slice.start
        return w


# ---------------------------------------------------------------------------
# The encoder core (NO skrl dependency)
# ---------------------------------------------------------------------------

class EntityEncoder(nn.Module):
    """DeepSets entity encoder over the obs_layout named slices.

    forward(x): x is ``(..., D)`` with D the layout width (376/641/645);
    returns ``(..., out_dim)``.  Streams and design decisions: module
    docstring.  Presence-mask weighting: each entity's phi output is
    multiplied by its own present bit BEFORE pooling, so a padded slot's
    content -- zeros or garbage -- contributes exactly 0.0 to the pooled
    embedding (bit-identical output; tests/test_models.py asserts it).
    """

    def __init__(self, layout="obs", cfg=None):
        super().__init__()
        self.cfg = resolve_cfg(cfg)
        self.streams = s = _LayoutStreams(layout)
        self.layout = s.layout
        self.in_dim = s.dim
        act = self.cfg["activation"]

        self.map_mlp = _mlp(s.map_in, self.cfg["map_hidden"], act)
        # SHARED per-entity MLPs -- one per entity KIND (shared across that
        # kind's slots), separate between kinds (different widths/semantics).
        self.phi_agent = _mlp(s.agent_phi_in, self.cfg["phi_hidden"], act)
        self.phi_target = _mlp(s.target_block[2], self.cfg["phi_hidden"], act)

        embed = self.cfg["phi_hidden"][-1]
        trunk_in = (self.cfg["map_hidden"][-1] + 2 * embed + s.scalar_w)
        self.trunk = _mlp(trunk_in, self.cfg["trunk_hidden"], act)
        self.out_dim = self.cfg["trunk_hidden"][-1]

    # -- streams ----------------------------------------------------------

    def _encode_map(self, x):
        """Flatten-MLP over the concatenated {0,1} planes (docstring: MAP
        ENCODER CHOICE). A conv variant would instead consume
        ``planes.view(B, n_planes, R, C)`` built from the same slices."""
        planes = torch.cat([x[:, sl] for sl in self.streams.map_slices], dim=-1)
        return self.map_mlp(planes)

    def _pool(self, phi, ent, present_off):
        """Masked pool of ``phi`` over entity slots; ``ent`` is (B, S, W_in),
        the present bit sits at ``present_off`` inside each slot."""
        h = phi(ent)                                          # (B, S, E)
        mask = ent[..., present_off:present_off + 1]          # (B, S, 1)
        pooled = (h * mask).sum(dim=1)                        # (B, E)
        if self.cfg["pool"] == "mean":
            pooled = pooled / mask.sum(dim=1).clamp(min=1.0)  # (B, 1) bcast
        return pooled

    def _entities(self, x, block):
        start, n_slots, width = block
        return x[:, start:start + n_slots * width].view(-1, n_slots, width)

    # -- forward ----------------------------------------------------------

    def forward(self, x):
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                "EntityEncoder(layout=%r) expects input width %d, got %r "
                "(widths: obs=%d, state=%d, full_state_obs=%d)"
                % (self.layout, self.in_dim, tuple(x.shape),
                   L.OBS_DIM, L.STATE_DIM, L.FULL_STATE_OBS_DIM))
        lead = x.shape[:-1]
        x = x.reshape(-1, self.in_dim)
        s = self.streams

        agent_ent = self._entities(x, s.agent_block)
        if s.agent_lang_block is not None:
            lang_ent = self._entities(x, s.agent_lang_block)
            agent_ent = torch.cat([agent_ent, lang_ent], dim=-1)
        target_ent = self._entities(x, s.target_block)

        pieces = [
            self._encode_map(x),
            self._pool(self.phi_agent, agent_ent, s.agent_present_off),
            self._pool(self.phi_target, target_ent, s.target_present_off),
        ]
        if s.ego_slice is not None:
            pieces.append(x[:, s.ego_slice])
        for sl in s.scalar_slices:
            pieces.append(x[:, sl])
        if s.lang_slice is not None:
            pieces.append(x[:, s.lang_slice])

        out = self.trunk(torch.cat(pieces, dim=-1))
        return out.reshape(lead + (self.out_dim,))


# ---------------------------------------------------------------------------
# Raw actor/critic heads (NO skrl dependency; the skrl classes wrap these).
# tests/test_models.py exercises these when skrl is absent.
# ---------------------------------------------------------------------------

class EntityPolicyNet(nn.Module):
    """EntityEncoder + linear logits head: (..., D) -> (..., N_ACTIONS)."""

    def __init__(self, layout="obs", cfg=None, n_actions=N_ACTIONS):
        super().__init__()
        self.encoder = EntityEncoder(layout=layout, cfg=cfg)
        self.head = nn.Linear(self.encoder.out_dim, n_actions)

    def forward(self, x):
        return self.head(self.encoder(x))


class EntityValueNet(nn.Module):
    """EntityEncoder + linear value head: (..., D) -> (..., 1)."""

    def __init__(self, layout="state", cfg=None):
        super().__init__()
        self.encoder = EntityEncoder(layout=layout, cfg=cfg)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# skrl 2.x Model subclasses (constructible only where skrl exists)
# ---------------------------------------------------------------------------

if SKRL_AVAILABLE:

    class CategoricalEntityPolicy(CategoricalMixin, Model):
        """skrl-2.x Categorical policy over the entity encoder.

        Bases are (CategoricalMixin, Model) -- skrl's convention; the mixin
        supplies ``act`` (module docstring: MRO note).  ``compute`` returns
        UNNORMALIZED logits; ``unnormalized_log_prob=True`` matches the
        YAML baseline (spec 1.14).

        ``layout`` must match the arm's actor width (spec 4.1): "obs" for
        the 376-d partial-obs arms, "full_state_obs" for the 645-d
        full-state arms.  No weight sharing with the value model: each
        instance owns a fresh EntityEncoder.
        """

        def __init__(self, observation_space, action_space, device=None,
                     unnormalized_log_prob=True, layout="obs", cfg=None):
            Model.__init__(self, observation_space, action_space, device)
            CategoricalMixin.__init__(self, unnormalized_log_prob)
            self.net = EntityPolicyNet(layout=layout, cfg=cfg,
                                       n_actions=N_ACTIONS)

        def compute(self, inputs, role=""):
            return self.net(inputs["states"]), {}

    class DeterministicEntityValue(DeterministicMixin, Model):
        """skrl-2.x central-critic value model over the entity encoder.

        Input is the 641-d critic state (MAPPO feeds the shared state under
        ``inputs["states"]``); layout is fixed to "state".  Owns its own
        EntityEncoder instance -- NO weight sharing with the policy.
        """

        def __init__(self, observation_space, action_space, device=None,
                     clip_actions=False, cfg=None):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)
            self.net = EntityValueNet(layout="state", cfg=cfg)

        def compute(self, inputs, role=""):
            return self.net(inputs["states"]), {}

else:

    class CategoricalEntityPolicy(object):  # noqa: D401 -- stub
        """Stub: skrl is not installed; construction raises ImportError."""

        def __init__(self, *args, **kwargs):
            raise ImportError(_SKRL_MISSING_MSG % "CategoricalEntityPolicy")

    class DeterministicEntityValue(object):
        """Stub: skrl is not installed; construction raises ImportError."""

        def __init__(self, *args, **kwargs):
            raise ImportError(_SKRL_MISSING_MSG % "DeterministicEntityValue")


# ---------------------------------------------------------------------------
# Space plumbing helpers for the factory (duck-typed: gymnasium spaces,
# plain ints, or skrl wrapper accessors -- skrl itself not required here)
# ---------------------------------------------------------------------------

def _flat_dim(space):
    """Flat width of an observation/state space: int, Box-like (.shape),
    or Discrete-like (.n)."""
    if isinstance(space, int):
        return space
    shape = getattr(space, "shape", None)
    if shape:
        n = 1
        for s in shape:
            n *= int(s)
        return n
    if getattr(space, "n", None) is not None:
        return int(space.n)
    raise TypeError("cannot infer flat width of space %r" % (space,))


def _n_discrete_actions(space):
    """Discrete action count from Discrete(.n), int, or {5}-style set."""
    if isinstance(space, int):
        return space
    if getattr(space, "n", None) is not None:
        return int(space.n)
    if isinstance(space, (set, frozenset)) and len(space) == 1:
        return int(next(iter(space)))
    raise TypeError("cannot infer discrete action count from %r" % (space,))


def _agent_space(env, kind, agent_id):
    """Fetch env's per-agent space: tries the dict property
    (``env.<kind>_spaces[agent_id]``, the skrl-2.x wrapper surface; item 1
    of harness/skrl_compat.py covers the 1.4 name), then the method form
    (``env.<kind>_space(agent_id)``, the PettingZoo-style surface)."""
    names = [kind + "_spaces"]
    if kind == "state":
        names.append("shared_observation_spaces")   # 1.4.x attr name
    for name in names:
        spaces = getattr(env, name, None)
        if spaces is not None and hasattr(spaces, "__getitem__"):
            try:
                return spaces[agent_id]
            except (KeyError, TypeError, IndexError):
                pass
    meth = getattr(env, kind + "_space", None)
    if callable(meth):
        return meth(agent_id)
    raise AttributeError(
        "env %r exposes neither .%s_spaces[agent] nor .%s_space(agent)"
        % (type(env).__name__, kind, kind))


# ---------------------------------------------------------------------------
# Factory -- the documented entry point for the 5090's training glue
# ---------------------------------------------------------------------------

def build_entity_models(env, device=None, cfg_overrides=None):
    """Build per-agent {"policy", "value"} entity-encoder models for MAPPO.

    Returns ``{agent_id: {"policy": CategoricalEntityPolicy,
    "value": DeterministicEntityValue}}`` with FRESH instances per agent
    and per role (MAPPO ``separate: True`` semantics; no weight sharing
    anywhere).  The policy layout is inferred from each agent's observation
    width (376 -> "obs", 645 -> "full_state_obs"); the value model always
    takes the 641-d critic state.

    Parameters
    ----------
    env : the skrl-wrapped multi-agent env (must expose ``possible_agents``
        and per-agent observation/action/state spaces; both the dict- and
        method-style accessors are handled).
    device : torch device for the models (default: ``env.device`` if
        present, else "cpu").
    cfg_overrides : optional dict merged over ``DEFAULT_CFG`` (keys:
        map_hidden, phi_hidden, trunk_hidden, pool, activation).

    5090 WIRING (they own the glue; the YAML ``models:`` block CANNOT build
    these classes -- skrl's instantiator only emits flat MLPs -- so replace
    ``Runner`` with manual construction; everything else comes from the
    project YAML unchanged)::

        import yaml, torch
        from skrl.envs.wrappers.torch import wrap_env
        from skrl.memories.torch import RandomMemory
        from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        from skrl.resources.schedulers.torch import KLAdaptiveLR
        from skrl.trainers.torch import SequentialTrainer
        from skrl.utils import set_seed

        from harness.models import build_entity_models

        raw = yaml.safe_load(open("tasks/team_listen/agents/skrl_mappo_cfg.yaml"))
        set_seed(raw["seed"])              # BEFORE building models: init
                                           # must be seed-reproducible
        env = wrap_env(isaac_env, wrapper="isaaclab-multi-agent")

        models = build_entity_models(env, device=env.device)

        memories = {aid: RandomMemory(memory_size=raw["agent"]["rollouts"],
                                      num_envs=env.num_envs, device=env.device)
                    for aid in env.possible_agents}

        cfg = MAPPO_DEFAULT_CONFIG.copy()
        for k, v in raw["agent"].items():   # the Runner did this for you:
            if k in ("class",):
                continue
            cfg[k] = v
        cfg["gae_lambda"] = raw["agent"]["lambda"]  # Runner's lambda mapping
        cfg.pop("lambda", None)
        cfg["learning_rate_scheduler"] = KLAdaptiveLR          # class, not str
        cfg["learning_rate_scheduler_kwargs"] = dict(
            raw["agent"]["learning_rate_scheduler_kwargs"])
        # spec 1.14: observation/state preprocessors stay None; the value
        # preprocessor is real and needs the size injection the Runner
        # normally performs (spec 1.14 [FIXED: *_preprocessor_kwargs]):
        cfg["observation_preprocessor"] = None
        cfg["state_preprocessor"] = None
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}
        cfg["experiment"] = dict(raw["agent"]["experiment"])

        agent = MAPPO(possible_agents=env.possible_agents,
                      models=models,
                      memories=memories,
                      cfg=cfg,
                      observation_spaces=env.observation_spaces,
                      action_spaces=env.action_spaces,
                      device=env.device,
                      state_spaces=env.state_spaces)
        # ^ 2.x names per harness/skrl_compat.py item 1 (1.4.x called both
        #   the wrapper attr and the kwarg shared_observation_spaces).
        #   If the pinned skrl's MAPPO.__init__ still takes the old kwarg,
        #   check inspect.signature(MAPPO.__init__) once and record it in
        #   DECISIONS.md -- do not guess.

        trainer = SequentialTrainer(
            cfg={"timesteps": raw["trainer"]["timesteps"],
                 "environment_info": raw["trainer"]["environment_info"]},
            env=env, agents=agent)
        trainer.train()

    Checkpoint note: these state_dicts are NOT loadable into the flat
    512-256-128 baseline or vice versa -- warm starts from p1-competent
    flat checkpoints (DECISIONS 2026-09-02 round-5 recipe (b)) do not
    transfer to the encoder; the encoder probe's warm-start cell means
    "encoder trained on the floor bank that warm start produced", not
    "encoder initialised from flat weights".
    """
    if not SKRL_AVAILABLE:
        raise ImportError(_SKRL_MISSING_MSG % "build_entity_models")
    agents = list(getattr(env, "possible_agents"))
    if device is None:
        device = getattr(env, "device", "cpu")
    cfg = dict(cfg_overrides) if cfg_overrides else None

    models = {}
    for aid in agents:
        obs_space = _agent_space(env, "observation", aid)
        act_space = _agent_space(env, "action", aid)
        state_space = _agent_space(env, "state", aid)

        obs_dim = _flat_dim(obs_space)
        if obs_dim not in LAYOUT_BY_DIM:
            raise ValueError(
                "agent %r observation width %d matches no known layout %r"
                % (aid, obs_dim, LAYOUT_BY_DIM))
        state_dim = _flat_dim(state_space)
        if state_dim != L.STATE_DIM:
            raise ValueError("agent %r state width %d != STATE_DIM %d"
                             % (aid, state_dim, L.STATE_DIM))
        n_act = _n_discrete_actions(act_space)
        if n_act != N_ACTIONS:
            raise ValueError("agent %r action count %d != %d"
                             % (aid, n_act, N_ACTIONS))

        policy = CategoricalEntityPolicy(
            obs_space, act_space, device=device, unnormalized_log_prob=True,
            layout=LAYOUT_BY_DIM[obs_dim], cfg=cfg)
        value = DeterministicEntityValue(
            state_space, act_space, device=device, clip_actions=False,
            cfg=cfg)
        models[aid] = {"policy": policy.to(device), "value": value.to(device)}
    return models


# ---------------------------------------------------------------------------
# Parameter-count parity vs the flat 512-256-128 baseline (spec 1.14)
# ---------------------------------------------------------------------------

#: The YAML baseline's hidden stack (spec 1.14 models block).
BASELINE_LAYERS = (512, 256, 128)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def flat_mlp_param_count(in_dim, layers=BASELINE_LAYERS, out_dim=1):
    """Exact parameter count of the skrl-instantiated flat MLP baseline."""
    total, prev = 0, in_dim
    for width in tuple(layers) + (out_dim,):
        total += prev * width + width
        prev = width
    return total


def parameter_parity_report(cfg=None):
    """Live parity table: entity models vs the flat baseline, per layout.

    Returns ``(text, rows)`` where rows is a list of
    ``(name, entity_params, baseline_params)``.
    """
    rows = [
        ("policy obs(376)->5",
         count_parameters(EntityPolicyNet(layout="obs", cfg=cfg)),
         flat_mlp_param_count(L.OBS_DIM, out_dim=N_ACTIONS)),
        ("policy full(645)->5",
         count_parameters(EntityPolicyNet(layout="full_state_obs", cfg=cfg)),
         flat_mlp_param_count(L.FULL_STATE_OBS_DIM, out_dim=N_ACTIONS)),
        ("value state(641)->1",
         count_parameters(EntityValueNet(layout="state", cfg=cfg)),
         flat_mlp_param_count(L.STATE_DIM, out_dim=1)),
    ]
    lines = ["entity-encoder models vs flat %s baseline (spec 1.14):"
             % (list(BASELINE_LAYERS),)]
    for name, ours, base in rows:
        lines.append("  %-22s %9s  vs %9s  (%.2fx)"
                     % (name, "{:,}".format(ours), "{:,}".format(base),
                        ours / float(base)))
    return "\n".join(lines), rows


__all__ = [
    "SKRL_AVAILABLE", "N_ACTIONS", "LAYOUT_BY_DIM", "DEFAULT_CFG",
    "resolve_cfg", "EntityEncoder", "EntityPolicyNet", "EntityValueNet",
    "CategoricalEntityPolicy", "DeterministicEntityValue",
    "build_entity_models", "BASELINE_LAYERS", "count_parameters",
    "flat_mlp_param_count", "parameter_parity_report",
]


if __name__ == "__main__":
    print(parameter_parity_report()[0])
