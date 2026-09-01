#!/usr/bin/env python
"""Policy entropy versus approach distance (round-3 work order item 3).

Proxy operationalisation, stated plainly: the harness does not yet
implement the strict spec-4.2 branch step (last step from which both
alcoves remain reachable), so this reports the policy's mean Categorical
entropy over the EARLY approach steps (first --k_steps decision steps,
active episodes only) and over all steps, per bank.  Run it once per
distance-window bank (TEAM_LISTEN_BANK selects the bank; the eval split
supplies the rows) and per checkpoint; the mechanism-(i) signature would
be early-step entropy collapsing as the spawn window widens for the
competent policy, versus flat/high entropy for the incompetent one.

Usage (one boot per bank):
    TEAM_LISTEN_BANK=data/scenario_bank_RoleBinding_phase1_*.pt \\
    <isaac python> scripts/entropy_vs_distance.py --headless \\
        runs/diag/curr_ckpt_s0_p1.pt runs/diag/curr_ckpt_s0_p4.pt
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--task", default="Isaac-TeamListen-RoleBinding-Blind-Direct-v0")
parser.add_argument("--n_base", type=int, default=512)
parser.add_argument("--k_steps", type=int, default=4)

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

bank_name = os.path.basename(os.environ.get("TEAM_LISTEN_BANK", "?"))

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

pool = (env._bank.split == 1).nonzero(as_tuple=False).reshape(-1)
base = pool[: args.n_base].cpu()
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


def entropy(logits):
    """(T, E, N, A) logits -> (T, E, N) Categorical entropy in nats."""
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


for ck in args.checkpoints:
    runner.agent.load(ck)
    for m in policies.values():
        m.eval()
    rec = ro.run_lanes(env, policy, plan, mode="argmax", record_logits=True)
    h = entropy(rec.logits)                      # (T, E, N)
    act = rec.active.unsqueeze(-1).float()       # (T, E, 1)
    k = min(args.k_steps, h.shape[0])
    early = float((h[:k] * act[:k]).sum() / act[:k].sum().clamp(min=1))
    overall = float((h * act).sum() / act.sum().clamp(min=1))
    ec = float(rec.completed.float().mean())
    print("ENTROPY bank=%s ckpt=%s early%d_H=%.3f all_H=%.3f E_C=%.3f"
          % (bank_name, ck, k, early, overall, ec), flush=True)

env.close()
app.close()
