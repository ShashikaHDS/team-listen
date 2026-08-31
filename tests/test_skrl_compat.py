"""Tests for harness/skrl_compat.py: the four verified skrl 1.4.x<->2.x items.

M1_SPEC section 0 [FIXED: bogus renames]: the shim must cover EXACTLY the
four verified differences --

1. attr rename   shared_observation_spaces -> state_spaces
2. cfg rename    shared_state_preprocessor -> state_preprocessor (+ _kwargs;
                 2.x additionally ADDS observation_preprocessor, which is
                 new, not a rename target)
3. default change value_loss_scale 1.0 -> 2.5
4. act(timestep=...) keyword-only in 2.x

-- and must NOT translate the two documented traps: YAML ``lambda`` (still
accepted on 2.x) and the flat per-role ``models:`` block. The shim is a
no-op on 2.x-shaped configs, and the project hard-gates skrl >= 2.0.0.

Pure-python, no skrl/isaaclab needed: this file must pass on the dev box.

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_skrl_compat.py`` discovers and runs its own
tests with a pass/fail summary.
"""

import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import skrl_compat as sc


# ---------------------------------------------------------------------------
# The translation table covers the four items -- no more, no fewer
# ---------------------------------------------------------------------------

def test_table_is_exactly_the_four_verified_items():
    # item 1: exactly one attribute rename
    assert sc.ATTR_RENAMES_1X_TO_2X == {
        "shared_observation_spaces": "state_spaces"}
    # item 2: exactly the preprocessor key pair
    assert sc.CFG_KEY_RENAMES_1X_TO_2X == {
        "shared_state_preprocessor": "state_preprocessor",
        "shared_state_preprocessor_kwargs": "state_preprocessor_kwargs"}
    # ...and observation_preprocessor is NEW in 2.x, never a rename target
    assert set(sc.NEW_IN_2X) == {"observation_preprocessor",
                                 "observation_preprocessor_kwargs"}
    assert not set(sc.NEW_IN_2X) & set(sc.CFG_KEY_RENAMES_1X_TO_2X.values())
    # item 3: exactly one changed default, with the verified values
    assert sc.CHANGED_DEFAULTS == {
        "value_loss_scale": {"1.x": 1.0, "2.x": 2.5}}
    # item 4: exactly act(timestep=...) keyword-only
    assert sc.KEYWORD_ONLY_IN_2X == {"act": ("timestep",)}


def test_non_renames_documented_and_untouched():
    """The two spec-0 traps are recorded as NON-renames and pass through."""
    assert set(sc.NOT_RENAMED) == {"lambda", "models"}
    # neither may appear anywhere in the rename table
    renamed = (set(sc.ATTR_RENAMES_1X_TO_2X)
               | set(sc.ATTR_RENAMES_1X_TO_2X.values())
               | set(sc.CFG_KEY_RENAMES_1X_TO_2X)
               | set(sc.CFG_KEY_RENAMES_1X_TO_2X.values()))
    assert not {"lambda", "models", "gae_lambda"} & renamed
    cfg = {"lambda": 0.95, "models": {"separate": True}}
    out = sc.translate_agent_cfg_1x_to_2x(cfg)
    assert out["lambda"] == 0.95                    # NOT mapped to gae_lambda
    assert "gae_lambda" not in out
    assert out["models"] == {"separate": True}      # NOT restructured


# ---------------------------------------------------------------------------
# Attribute rename (item 1)
# ---------------------------------------------------------------------------

def test_attr_rename():
    assert sc.translate_attr_name("shared_observation_spaces") == "state_spaces"
    # identity everywhere else, including the already-2.x name
    assert sc.translate_attr_name("state_spaces") == "state_spaces"
    assert sc.translate_attr_name("observation_spaces") == "observation_spaces"


# ---------------------------------------------------------------------------
# Cfg translation (items 2 + 3)
# ---------------------------------------------------------------------------

def test_translate_1x_cfg():
    cfg_1x = {
        "rollouts": 16,
        "lambda": 0.95,
        "shared_state_preprocessor": "RunningStandardScaler",
        "shared_state_preprocessor_kwargs": None,
        "state_preprocessor_kwargs_unrelated": 1,   # not in the table: kept
    }
    out = sc.translate_agent_cfg_1x_to_2x(cfg_1x)
    assert cfg_1x["shared_state_preprocessor"] == "RunningStandardScaler", \
        "input must not be mutated"
    assert "shared_state_preprocessor" not in out
    assert out["state_preprocessor"] == "RunningStandardScaler"
    assert "shared_state_preprocessor_kwargs" not in out
    assert out["state_preprocessor_kwargs"] is None
    assert out["rollouts"] == 16 and out["lambda"] == 0.95
    assert out["state_preprocessor_kwargs_unrelated"] == 1
    # item 3: absent on 1.4.x meant 1.0; pinned so the 2.x default (2.5)
    # cannot silently change behaviour
    assert out["value_loss_scale"] == 1.0
    # the 2.x-only observation_preprocessor is never synthesised
    assert "observation_preprocessor" not in out
    assert "observation_preprocessor_kwargs" not in out


def test_translate_preserves_explicit_value_loss_scale():
    out = sc.translate_agent_cfg_1x_to_2x({"value_loss_scale": 2.0})
    assert out["value_loss_scale"] == 2.0


def test_translate_rejects_conflicting_key_pair():
    try:
        sc.translate_agent_cfg_1x_to_2x({
            "shared_state_preprocessor": "A", "state_preprocessor": "B"})
    except ValueError:
        pass
    else:
        raise AssertionError("both 1.x and 2.x forms present must raise")


def test_noop_on_2x_cfg():
    """A 2.x-authored cfg (our YAML's shape) passes through as the SAME object."""
    cfg_2x = {
        "class": "MAPPO",
        "rollouts": 16,
        "lambda": 0.95,                             # correct on 2.x too
        "observation_preprocessor": None,
        "observation_preprocessor_kwargs": None,
        "state_preprocessor": None,
        "state_preprocessor_kwargs": None,
        "value_preprocessor": "RunningStandardScaler",
        "value_preprocessor_kwargs": None,
        "entropy_loss_scale": 0.008,
        "value_loss_scale": 2.0,
        "time_limit_bootstrap": True,
    }
    assert not sc.looks_like_1x(cfg_2x)
    out = sc.maybe_translate_agent_cfg(cfg_2x, "2.0.0")
    assert out is cfg_2x, "must be a true no-op on 2.x-shaped configs"


def test_maybe_translate_ports_1x_shape_forward():
    cfg = {"shared_state_preprocessor": None}
    assert sc.looks_like_1x(cfg)
    out = sc.maybe_translate_agent_cfg(cfg, "2.0.2")
    assert out is not cfg and out["state_preprocessor"] is None


# ---------------------------------------------------------------------------
# Version gate (spec 0: hard gate skrl >= 2.0.0)
# ---------------------------------------------------------------------------

def test_version_parse_and_gate():
    assert sc.parse_version("2.0.0") == (2, 0, 0)
    assert sc.parse_version("2.0.0rc1") == (2, 0, 0)
    assert sc.parse_version("1.4.3-post0") == (1, 4, 3)
    assert sc.parse_version("2.1") == (2, 1, 0)
    assert sc.SKRL_MIN_VERSION == (2, 0, 0)
    assert sc.is_supported("2.0.0") and sc.is_supported("2.1.5")
    assert not sc.is_supported("1.4.3")
    assert sc.is_2x("2.0.0rc1") and not sc.is_2x("1.9.9")
    sc.assert_supported("2.0.0")                    # must not raise
    try:
        sc.assert_supported("1.4.3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("1.4.3 must fail the hard gate")
    try:
        sc.maybe_translate_agent_cfg({}, "1.4.3")
    except RuntimeError:
        pass
    else:
        raise AssertionError("maybe_translate must enforce the hard gate")


# ---------------------------------------------------------------------------
# act() call convention (item 4)
# ---------------------------------------------------------------------------

def test_call_act_uses_keywords_on_both_signatures():
    calls = []

    class Agent2x:                       # 2.x: timestep keyword-only
        def act(self, states, *, timestep, timesteps):
            calls.append(("2x", states, timestep, timesteps))
            return "ok2"

    class Agent1x:                       # 1.4.x: positional-or-keyword
        def act(self, states, timestep, timesteps):
            calls.append(("1x", states, timestep, timesteps))
            return "ok1"

    assert sc.call_act(Agent2x(), "s", 3, 100) == "ok2"
    assert sc.call_act(Agent1x(), "s", 3, 100) == "ok1"
    assert calls == [("2x", "s", 3, 100), ("1x", "s", 3, 100)]
    # the helper itself must pass timestep/timesteps as KEYWORDS -- verify
    # statically that its body contains no positional forwarding
    src = inspect.getsource(sc.call_act)
    assert "timestep=timestep" in src and "timesteps=timesteps" in src


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  " + name)
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("{} passed, {} failed".format(len(tests) - n_fail, n_fail))
    sys.exit(1 if n_fail else 0)
