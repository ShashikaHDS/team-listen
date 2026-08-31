"""Team Listen task package: layout truth + the gym registration grid.

Importing this package (a) runs ``obs_layout.assert_layout()`` at import so
the slice arithmetic of M1_SPEC 1.5/1.6 is guaranteed before anything is
built on it, and (b) registers ONE entry point per (variant x arm) cell --
14 ids total (M1_SPEC 1.14 / section 7):

    id = f"Isaac-TeamListen-{variant}-{arm}-Direct-v0"
    variant in ("RoleBinding", "Precedence")
    arm     in ("Lang", "Blind", "Symbol", "SymbolPO", "Leaky", "Mute", "Placebo")

e.g. "Isaac-TeamListen-RoleBinding-Blind-Direct-v0" (the section 8.2 step-2
training-FPS gate task).

Each registration carries THREE entry-point kwargs. BOTH skrl entry points
are mandatory: the shipped train.py resolves
``f"skrl_{algorithm.lower()}_cfg_entry_point"``, so registering only the
default key makes ``--algorithm MAPPO`` fail to find a config (spec 1.14).
Never register or run ``--algorithm PPO``: ``multi_agent_to_single_agent``
is broken for discrete spaces (spec 1.14); ``scripts/train.py`` refuses it.

Registration is pure metadata (string entry points, resolved lazily by
``gym.make``), so importing this package NEVER imports isaaclab and is safe
on the dev box and pre-SimulationApp on the 5090.
"""

import gymnasium as gym

from . import obs_layout

# Re-export the two dimensions everything downstream keys on.
from .obs_layout import OBS_DIM, STATE_DIM  # noqa: F401

VARIANTS = ("RoleBinding", "Precedence")
ARMS = ("Lang", "Blind", "Symbol", "SymbolPO", "Leaky", "Mute", "Placebo")

#: task id -> registration metadata; consumed by scripts/train.py for the
#: run manifest WITHOUT importing fleet_env_cfg (which must not be imported
#: pre-SimulationApp in a process that later constructs the env).
TASK_TO_CFG = {}

_AGENTS = __name__ + ".agents"

for _variant in VARIANTS:
    for _arm in ARMS:
        _task_id = "Isaac-TeamListen-%s-%s-Direct-v0" % (_variant, _arm)
        _cfg_entry = "%s.fleet_env_cfg:%s%sEnvCfg" % (__name__, _variant, _arm)
        gym.register(
            id=_task_id,
            entry_point="%s.fleet_env:TeamGridEnv" % __name__,
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _cfg_entry,
                "skrl_mappo_cfg_entry_point": _AGENTS + ":skrl_mappo_cfg.yaml",
                "skrl_ippo_cfg_entry_point": _AGENTS + ":skrl_ippo_cfg.yaml",
            },
        )
        TASK_TO_CFG[_task_id] = {
            "variant": _variant,
            "arm": _arm,
            "env_cfg_entry_point": _cfg_entry,
        }

__all__ = ["obs_layout", "OBS_DIM", "STATE_DIM", "VARIANTS", "ARMS",
           "TASK_TO_CFG"]
