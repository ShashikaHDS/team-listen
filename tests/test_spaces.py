"""Space-spec decoding + discrete action-decode contract (M1_SPEC 1.4 / 1.7).

The two cheapest, highest-risk integration points of the section 8.2 ladder
(step 1):

* ``spec_to_gym_space`` decoding -- ``{5}`` (a SET of length one) must map
  to ``Discrete(5)`` and ``[{5}, {5}]`` to ``MultiDiscrete([5, 5])``
  (spec 1.4); an int spec maps to a flat ``Box`` (the observation / state
  contract: never a Dict, no Discrete inside the observation).
* the action-decode contract of spec 1.7 -- skrl's
  ``unflatten_tensorized_space(Discrete, .)`` hands the env ``(E, 1)``
  int32/int64, NOT ``(E,)``; the load-bearing idiom is
  ``actions[k].reshape(-1).long()`` -> ``(E,)`` int64 -> stacked ``(E, 2)``
  -> delta lookup ``(E, 2, 2)``.  An unsqueezed index instead broadcasts
  ``DELTAS`` to ``(E, 1, 2)`` -- the documented trap, asserted here so it
  cannot be "simplified" back in.

Isaac Lab guard: ``isaaclab`` is NOT importable on the Windows dev box
(M1_SPEC section 0), and whether ``isaaclab.envs.utils.spaces`` imports
without booting ``SimulationApp`` is itself a 5090-open-question.  Every
``spec_to_gym_space`` test therefore SKIPS (never fails) when the import
is unavailable, with the underlying error in the skip message; on the 5090
the whole file runs.  The action-decode contract is pure torch and runs
everywhere, pinned against the actual ``fleet_env`` source.

pytest-compatible (plain ``test_*`` functions, bare asserts; skips raise
``unittest.SkipTest``); also runnable standalone:
``python tests/test_spaces.py`` prints a pass/fail/skip summary and exits
nonzero only on failures (skips count as green on the dev box).
"""

import inspect
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import fleet_env
from tasks.team_listen import grid_core
from tasks.team_listen.fleet_env_cfg import TeamGridEnvCfg
from tasks.team_listen.obs_layout import OBS_DIM, STATE_DIM

AGENTS = ("robot_0", "robot_1")                 # STABLE ORDER (spec 1.4)

# ---------------------------------------------------------------------------
# Guarded isaaclab import (spec section 0: not importable on the dev box;
# open question: importable pre-SimulationApp on the 5090?)
# ---------------------------------------------------------------------------

spec_to_gym_space = None
_IMPORT_ERROR = None
try:
    from isaaclab.envs.utils.spaces import spec_to_gym_space  # noqa: F811
except Exception as _exc:                       # ImportError / omni failures
    _IMPORT_ERROR = _exc
    try:                                        # fallback location
        from isaaclab.utils.spaces import spec_to_gym_space  # noqa: F811
        _IMPORT_ERROR = None
    except Exception:
        spec_to_gym_space = None


def _need_spec_to_gym_space():
    if spec_to_gym_space is None:
        raise unittest.SkipTest(
            "isaaclab spec_to_gym_space not importable on this box "
            "(%r) -- Isaac-dependent checks run on the 5090 "
            "(M1_SPEC section 0 / 8.2 step 1)" % (_IMPORT_ERROR,))


def _base_cfg():
    """A TeamGridEnvCfg view that works on both the dev-box shim (plain
    class attrs) and the real @configclass (instantiate)."""
    try:
        return TeamGridEnvCfg()
    except Exception:
        return TeamGridEnvCfg


# ---------------------------------------------------------------------------
# spec_to_gym_space decoding (spec 1.4) -- SKIPPED without isaaclab
# ---------------------------------------------------------------------------

def test_spec_set5_is_discrete_5():
    """{5} -- a set of length one -- decodes to Discrete(5) (spec 1.4)."""
    _need_spec_to_gym_space()
    import gymnasium as gym

    space = spec_to_gym_space({5})
    assert isinstance(space, gym.spaces.Discrete), \
        "{5} decoded to %r, not Discrete" % (space,)
    assert space == gym.spaces.Discrete(5)


def test_spec_list_of_sets_is_multidiscrete():
    """[{5}, {5}] decodes to MultiDiscrete([5, 5]) (spec 1.4)."""
    _need_spec_to_gym_space()
    import gymnasium as gym
    import numpy as np

    space = spec_to_gym_space([{5}, {5}])
    assert isinstance(space, gym.spaces.MultiDiscrete), \
        "[{5}, {5}] decoded to %r, not MultiDiscrete" % (space,)
    assert np.array_equal(np.asarray(space.nvec).reshape(-1), [5, 5])


def test_cfg_action_spaces_decode_to_discrete_5():
    """The ACTUAL cfg values decode to Discrete(5) per agent."""
    _need_spec_to_gym_space()
    import gymnasium as gym

    cfg = _base_cfg()
    for name in AGENTS:
        space = spec_to_gym_space(cfg.action_spaces[name])
        assert space == gym.spaces.Discrete(5), \
            "cfg.action_spaces[%r] decoded to %r" % (name, space)


def test_cfg_observation_and_state_specs_decode_to_flat_box():
    """Int specs (OBS_DIM / positive STATE_DIM) decode to flat Boxes --
    never Dict, never Discrete-inside-observation (spec 1.4/1.5)."""
    _need_spec_to_gym_space()
    import gymnasium as gym

    cfg = _base_cfg()
    for name in AGENTS:
        space = spec_to_gym_space(cfg.observation_spaces[name])
        assert isinstance(space, gym.spaces.Box)
        assert tuple(space.shape) == (OBS_DIM,)
    state = spec_to_gym_space(cfg.state_space)
    assert isinstance(state, gym.spaces.Box)
    assert tuple(state.shape) == (STATE_DIM,)


# ---------------------------------------------------------------------------
# Cfg spec SHAPES are the exact objects spec_to_gym_space keys on
# (pure python -- runs on the dev box)
# ---------------------------------------------------------------------------

def test_cfg_action_space_spec_shape():
    """action_spaces must stay {agent: {5}} -- a SET of length one holding
    the int 5.  Anything else ({5} "simplified" to 5, or a prebuilt
    Discrete) changes what spec_to_gym_space builds (spec 1.4)."""
    cfg = _base_cfg()
    assert list(cfg.possible_agents) == list(AGENTS)
    assert set(cfg.action_spaces) == set(AGENTS)
    for name in AGENTS:
        spec = cfg.action_spaces[name]
        assert isinstance(spec, set) and spec == {5} and len(spec) == 1, \
            "action_spaces[%r] must be the set {5}; got %r" % (name, spec)
    for name in AGENTS:
        assert cfg.observation_spaces[name] == OBS_DIM
        assert isinstance(cfg.observation_spaces[name], int)
    assert cfg.state_space == STATE_DIM and STATE_DIM > 0   # POSITIVE int
    assert cfg.action_noise_model is None                   # MUST stay None


# ---------------------------------------------------------------------------
# Action-decode contract (spec 1.7) -- pure torch, runs everywhere
# ---------------------------------------------------------------------------

def test_action_decode_e1_int32_to_e2_long():
    """(E, 1) int32 -> .reshape(-1).long() -> (E,) int64 -> stacked (E, 2)
    -> DELTAS lookup (E, 2, 2), exactly the fleet_env._pre_physics_step
    idiom (spec 1.7)."""
    E = 7
    gen = torch.Generator().manual_seed(0)
    actions = {
        # skrl's unflatten_tensorized_space(Discrete, .) shape/dtypes:
        "robot_0": torch.randint(0, 5, (E, 1), generator=gen,
                                 dtype=torch.int32),
        "robot_1": torch.randint(0, 5, (E, 1), generator=gen,
                                 dtype=torch.int64),
    }
    a = torch.stack([actions[k].reshape(-1).long() for k in AGENTS], 1)
    assert a.shape == (E, 2) and a.dtype == torch.int64
    assert torch.equal(a[:, 0], actions["robot_0"].reshape(-1).long())
    assert torch.equal(a[:, 1], actions["robot_1"].reshape(-1).long())

    delta = grid_core.DELTAS[a]
    assert delta.shape == (E, 2, 2), \
        "delta lookup must be (E, 2, 2); got %r" % (tuple(delta.shape),)
    # spot-check the delta table against the action encoding (spec 1.3)
    for i in range(E):
        for j, name in enumerate(AGENTS):
            act = int(a[i, j])
            assert delta[i, j].tolist() == grid_core.DELTAS[act].tolist()

    # an already-flat (E,) tensor round-trips through the same idiom
    flat = torch.randint(0, 5, (E,), generator=gen, dtype=torch.int64)
    assert torch.equal(flat.reshape(-1).long(), flat)


def test_unsqueezed_index_is_the_documented_trap():
    """WHY .reshape(-1) is load-bearing (spec 1.7): indexing DELTAS with
    the raw (E, 1) tensor broadcasts to (E, 1, 2), silently mis-shaping
    every downstream position update."""
    E = 4
    raw = torch.zeros((E, 1), dtype=torch.int32)         # as skrl hands it
    trap = grid_core.DELTAS[raw.long()]
    assert trap.shape == (E, 1, 2), \
        "the (E,1) broadcast trap changed shape -- update spec 1.7 WHY"
    assert trap.shape != (E, 2)


def test_decode_idiom_pinned_in_fleet_env_source():
    """The exact load-bearing idiom must live in TeamGridEnv._pre_physics_step
    (statically checked -- the env itself cannot be constructed off-Isaac)."""
    src = inspect.getsource(fleet_env.TeamGridEnv._pre_physics_step)
    assert ".reshape(-1).long()" in src, \
        "spec 1.7 decode idiom missing from _pre_physics_step"
    assert "possible_agents" in src, \
        "decode must iterate cfg.possible_agents (STABLE ORDER, spec 1.4)"
    src_apply = inspect.getsource(fleet_env.TeamGridEnv._apply_action)
    assert src_apply.strip().splitlines()[-1].strip() == "pass", \
        "_apply_action must stay a no-op (spec 1.1/1.7)"


def test_decode_through_slip_and_step_positions():
    """End-to-end: (E,1) dict -> decode -> apply_slip (NO_SLIP row) ->
    step_positions on an open grid moves each robot by DELTAS[action]."""
    E = 5
    occ = torch.zeros((E, 12, 12), dtype=torch.int8)
    latched = torch.zeros((E, 2), dtype=torch.bool)
    pos = torch.tensor([[3, 3], [8, 8]], dtype=torch.int16) \
        .unsqueeze(0).repeat(E, 1, 1)
    gen = torch.Generator().manual_seed(1)
    actions = {name: torch.randint(0, 5, (E, 1), generator=gen,
                                   dtype=torch.int32) for name in AGENTS}

    a = torch.stack([actions[k].reshape(-1).long() for k in AGENTS], 1)
    slip_row = torch.full((E, 4), grid_core.NO_SLIP, dtype=torch.uint8)
    a = grid_core.apply_slip(a, slip_row)                # identity here
    assert a.dtype == torch.int64 and a.shape == (E, 2)

    nxt, hit_obs, hit_rob = grid_core.step_positions(
        pos, a, occ, latched, (12, 12))
    assert nxt.shape == (E, 2, 2) and nxt.dtype == pos.dtype
    assert not bool(hit_obs.any()) and not bool(hit_rob.any())
    expect = pos.long() + grid_core.DELTAS[a]            # robots never meet
    assert torch.equal(nxt.long(), expect)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  " + name)
        except unittest.SkipTest as exc:
            n_skip += 1
            print("SKIP  {}  ({})".format(name, exc))
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("{} passed, {} failed, {} skipped".format(
        len(tests) - n_fail - n_skip, n_fail, n_skip))
    sys.exit(1 if n_fail else 0)
