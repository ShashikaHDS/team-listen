"""skrl 1.4.x <-> 2.x compatibility shim: ONLY the four VERIFIED items.

Authoritative reference: docs/M1_SPEC.md section 0 [FIXED: two of the five
claimed 1.4->2.x renames were wrong and would have produced a bogus compat
shim]. The verified REAL differences between skrl 1.4.x and 2.x are exactly:

1. env/wrapper attribute rename:  ``shared_observation_spaces`` ->
   ``state_spaces``.
2. agent-cfg key rename: ``shared_state_preprocessor`` ->
   ``state_preprocessor`` (with its ``_kwargs`` companion); 2.x also ADDS a
   new, unrelated ``observation_preprocessor`` (no 1.4.x counterpart --
   nothing maps onto it).
3. changed default: ``value_loss_scale`` 1.0 (1.4.x) -> 2.5 (2.x).
4. call convention: ``agent.act(..., timestep=..., timesteps=...)`` --
   ``timestep`` is keyword-only in 2.x.

Explicit NON-renames (spec 0: "must not be translated"; asserted by
tests/test_skrl_compat.py):

* YAML ``lambda`` is still the accepted key on 2.x
  (``Runner._check_cfg_compatibility`` maps it to ``gae_lambda``).
* the ``models:`` block is still FLAT per-role on 2.x
  (``Runner._generate_models`` copies it per agent id).

The project hard-gates ``skrl >= 2.0.0`` (spec 0), so at runtime this module
is a NO-OP: ``maybe_translate_agent_cfg`` returns 2.x-shaped configs
untouched (the very same object). The 1.x->2.x translators exist to port
1.4-era artifacts/snippets forward and to pin the translation table that
``tests/test_skrl_compat.py`` asserts.

Pure python (no skrl import needed), runnable on the dev box.
"""

import re

# ---------------------------------------------------------------------------
# The hard version gate (spec 0)
# ---------------------------------------------------------------------------

SKRL_MIN_VERSION = (2, 0, 0)


def parse_version(version):
    """'2.0.0' / '2.0.0rc1' / '1.4.3-post0' -> integer tuple (2, 0, 0)."""
    if isinstance(version, tuple):
        return tuple(int(x) for x in version)
    m = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(version))
    if not m:
        raise ValueError("unparseable skrl version: %r" % (version,))
    return tuple(int(g) if g is not None else 0 for g in m.groups())


def is_supported(version):
    """True iff ``version`` satisfies the hard gate skrl >= 2.0.0."""
    return parse_version(version) >= SKRL_MIN_VERSION


def assert_supported(version):
    if not is_supported(version):
        raise RuntimeError(
            "skrl %r < %s: the project hard-gates skrl >= 2.0.0 (M1_SPEC 0; "
            "gymnasium >= 1.0 + skrl < 1.4 breaks MAPPO with "
            "\"'OrderEnforcing' object has no attribute 'state'\", and this "
            "shim does NOT make 1.4.x supported -- it only translates 1.4-era "
            "artifacts forward)." % (version, ".".join(map(str, SKRL_MIN_VERSION))))


def is_2x(version):
    return parse_version(version)[0] >= 2


# ---------------------------------------------------------------------------
# Item 1: env/wrapper attribute rename
# ---------------------------------------------------------------------------

ATTR_RENAMES_1X_TO_2X = {
    "shared_observation_spaces": "state_spaces",
}


def translate_attr_name(name):
    """1.4.x attribute name -> 2.x name (identity for everything else)."""
    return ATTR_RENAMES_1X_TO_2X.get(name, name)


# ---------------------------------------------------------------------------
# Item 2: agent-cfg key rename (+ the new 2.x-only observation_preprocessor)
# ---------------------------------------------------------------------------

CFG_KEY_RENAMES_1X_TO_2X = {
    "shared_state_preprocessor": "state_preprocessor",
    "shared_state_preprocessor_kwargs": "state_preprocessor_kwargs",
}

#: New in 2.x with NO 1.4.x counterpart: never a rename target.
NEW_IN_2X = ("observation_preprocessor", "observation_preprocessor_kwargs")

# ---------------------------------------------------------------------------
# Item 3: changed default
# ---------------------------------------------------------------------------

CHANGED_DEFAULTS = {
    "value_loss_scale": {"1.x": 1.0, "2.x": 2.5},
}

# ---------------------------------------------------------------------------
# Item 4: call convention (keyword-only ``timestep`` in 2.x)
# ---------------------------------------------------------------------------

KEYWORD_ONLY_IN_2X = {
    "act": ("timestep",),
}


def call_act(agent, states, timestep, timesteps):
    """Version-proof ``agent.act`` call: always pass keywords.

    2.x makes ``timestep`` keyword-only; 1.4.x accepted it positionally but
    also accepts the keyword form, so keywords are correct on both.
    """
    return agent.act(states, timestep=timestep, timesteps=timesteps)


# ---------------------------------------------------------------------------
# Explicit NON-renames (spec 0: translating these would be a bug)
# ---------------------------------------------------------------------------

NOT_RENAMED = {
    "lambda": ("still the accepted YAML key on 2.x; "
               "Runner._check_cfg_compatibility maps it to gae_lambda"),
    "models": ("still a FLAT per-role block on 2.x; "
               "Runner._generate_models copies it per agent id"),
}


# ---------------------------------------------------------------------------
# Cfg translation
# ---------------------------------------------------------------------------

def looks_like_1x(agent_cfg):
    """True iff the dict uses any 1.4.x-only key this table covers."""
    return any(k in agent_cfg for k in CFG_KEY_RENAMES_1X_TO_2X)


def translate_agent_cfg_1x_to_2x(agent_cfg):
    """Port a 1.4-era agent-cfg dict to 2.x naming/behaviour. Returns a NEW
    dict; the input is never mutated.

    * renames ``shared_state_preprocessor(_kwargs)`` ->
      ``state_preprocessor(_kwargs)`` (key order preserved);
    * pins ``value_loss_scale`` to the 1.x default 1.0 when absent, so the
      2.x default change (2.5) cannot silently alter behaviour;
    * touches NOTHING else -- in particular ``lambda`` and ``models`` pass
      through untranslated (they are not renames, spec 0), and the 2.x-only
      ``observation_preprocessor`` keys are never synthesised from 1.x keys.
    """
    out = {}
    for key, value in agent_cfg.items():
        new_key = CFG_KEY_RENAMES_1X_TO_2X.get(key, key)
        if new_key in out:
            raise ValueError(
                "cfg carries both %r and its 2.x form %r" % (key, new_key))
        out[new_key] = value
    if "value_loss_scale" not in out:
        # absent on 1.4.x meant 1.0; on 2.x it would mean 2.5 -- pin it.
        out["value_loss_scale"] = CHANGED_DEFAULTS["value_loss_scale"]["1.x"]
    return out


def maybe_translate_agent_cfg(agent_cfg, skrl_version):
    """The runtime entry point: a NO-OP on 2.x-shaped configs.

    Enforces the hard version gate, then translates only if the cfg still
    uses 1.4.x naming (a ported artifact); a 2.x-authored cfg is returned
    as the SAME object, untouched.
    """
    assert_supported(skrl_version)
    if looks_like_1x(agent_cfg):
        return translate_agent_cfg_1x_to_2x(agent_cfg)
    return agent_cfg


__all__ = [
    "SKRL_MIN_VERSION", "parse_version", "is_supported", "assert_supported",
    "is_2x",
    "ATTR_RENAMES_1X_TO_2X", "translate_attr_name",
    "CFG_KEY_RENAMES_1X_TO_2X", "NEW_IN_2X",
    "CHANGED_DEFAULTS", "KEYWORD_ONLY_IN_2X", "call_act",
    "NOT_RENAMED", "looks_like_1x",
    "translate_agent_cfg_1x_to_2x", "maybe_translate_agent_cfg",
]
