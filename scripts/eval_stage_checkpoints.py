#!/usr/bin/env python
"""Stage-boundary certified eval for curriculum probes (round-2 order §3).

One SimulationApp boot evaluates every listed checkpoint on the CERTIFIED
eval rows.  The env is built on a mixture bank (whose split=1 pool IS the
certified eval rows, embedded verbatim by build_mixture_banks.py), so no
bank swap is needed between checkpoints.  Reports per checkpoint:
episode-level E[C] and end-of-episode single-latch share, argmax mode.

Usage:
    TEAM_LISTEN_BANK=data/scenario_bank_RoleBinding_mix4_*.pt \\
    <isaac python> scripts/eval_stage_checkpoints.py \\
        --task Isaac-TeamListen-RoleBinding-Blind-Direct-v0 \\
        --n_base 1024 --headless ckpt_stage1.pt ckpt_stage2.pt ...
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--task", default="Isaac-TeamListen-RoleBinding-Blind-Direct-v0")
parser.add_argument("--n_base", type=int, default=1024)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import tasks.team_listen  # noqa: F401, E402
from harness import certificate as ce_mod  # noqa: E402
from harness import rollout as ro  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.n_base
env = gym.make(args.task, cfg=env_cfg).unwrapped
assert not env.sim.has_gui() and not env.sim.has_rtx_sensors()

agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(SkrlVecEnvWrapper(env), agent_cfg)
policies = runner.agent.policies
STATE_W = int(env.cfg.state_space)

eval_pool = (env._bank.split == 1).nonzero(as_tuple=False).reshape(-1)
assert eval_pool.numel() >= args.n_base, "bank eval pool too small"
base = eval_pool[: args.n_base].cpu()
g = torch.Generator().manual_seed(17)
half = args.n_base // 2
classes = torch.cat([torch.zeros(args.n_base - half, dtype=torch.long),
                     torch.ones(half, dtype=torch.long)])
classes = classes[torch.randperm(args.n_base, generator=g)]
plan = ce_mod.make_natural_plan(base, classes)


def policy(obs):
    out = {}
    with torch.no_grad():
        for a, o in obs.items():
            dummy = torch.zeros(o.shape[0], STATE_W, device=o.device)
            _, extra = policies[a].act(
                {"observations": o, "states": dummy}, role="policy")
            out[a] = extra["net_output"]
    return out


for ck in args.checkpoints:
    runner.agent.load(ck)
    for m in policies.values():
        m.eval()
    rec = ro.run_lanes(env, policy, plan, mode="argmax")
    n_latched = (rec.latch_time >= 0).sum(dim=1)
    ec = rec.completed.float().mean()
    single = (n_latched == 1).float().mean()
    print("STAGE_EVAL ckpt=%s E_C=%.4f single_latch=%.4f n=%d"
          % (ck, ec, single, args.n_base), flush=True)

env.close()
app.close()
