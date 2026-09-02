#!/usr/bin/env python
"""Critic inspection at hand-classified state families (round-4 study B).

Rolls one checkpoint's ARGMAX policy on the certified bank (the env's own
reset draw over certified train rows), and at every decision step queries
the value heads on the true critic state, inverse-transforming through the
checkpoint's trained RunningStandardScaler so values are in REAL return
units.  Steps are classified into families from env internals:

  early_far                 t < 5, both robots >= 4 cells from a station
  both_adjacent             both 1 cell away, none latched
  parked_near               t >= 20, none latched, both within 1..3 cells
  one_latched_partner_near  exactly one latched, other <= 2 cells
  one_latched_partner_far   exactly one latched, other >= 4 cells
  both_latched              terminal frozen state (reference)

Yardstick (gamma=0.99): from both_adjacent, completing next step is worth
~ +2+2+2+5 = +11 discounted one step (~ +10.9) plus residual shaping;
parking forever is worth ~ -0.01 x remaining steps (~ -1).  A critic that
prices both_adjacent >> parked_near KNOWS about the payoff (policy-trap
story); one that prices them alike never learned it (critic-blindness).

Usage:
    TEAM_LISTEN_BANK=<certified> <isaac python> scripts/critic_inspection.py \\
        --headless runs/diag/curr_ckpt_s0_p4.pt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--task", default="Isaac-TeamListen-RoleBinding-Blind-Direct-v0")
parser.add_argument("--num_envs", type=int, default=512)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import tasks.team_listen  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env = gym.make(args.task, cfg=env_cfg).unwrapped
assert not env.sim.has_gui() and not env.sim.has_rtx_sensors()

agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(SkrlVecEnvWrapper(env), agent_cfg)
agent = runner.agent
STATE_W = int(env.cfg.state_space)
AGENTS = list(env.cfg.possible_agents)
N = len(AGENTS)


def robot_station_dist():
    """(E, N) latch-aware BFS distance of each robot to its NEAREST valid
    station, from the env's resident dist_field."""
    pos = env.pos[:, :N].long()                       # (E, N, 2)
    df = env.dist_field.long()                        # (E, T, R, C)
    E = pos.shape[0]
    d = []
    for i in range(N):
        r, c = pos[:, i, 0], pos[:, i, 1]
        per_t = df[torch.arange(E), :, r, c]          # (E, T)
        per_t = torch.where(
            env.target_valid & (per_t >= 0), per_t,
            torch.full_like(per_t, 10_000))
        d.append(per_t.min(dim=1).values)
    return torch.stack(d, dim=1)                      # (E, N)


for ck in args.checkpoints:
    agent.load(ck)
    for m in list(agent.policies.values()) + list(agent.values.values()):
        m.eval()
    env._reset_idx(None)
    env.episode_length_buf.zero_()
    sums = defaultdict(float)
    counts = defaultdict(int)
    for t in range(int(env.max_episode_length) - 1):
        obs = env._get_observations()
        state = env._get_states()
        with torch.no_grad():
            # value in real return units: inverse the trained scaler
            uid = AGENTS[0]
            st = state[uid] if isinstance(state, dict) else state
            v_norm, _ = agent.values[uid].act(
                {"observations": obs[uid],
                 "states": st}, role="value")
            v = agent._value_preprocessor[uid](v_norm, inverse=True)
            v = v.reshape(-1)
            logits = {}
            for a in AGENTS:
                dummy = torch.zeros(obs[a].shape[0], STATE_W,
                                    device=obs[a].device)
                _, extra = agent.policies[a].act(
                    {"observations": obs[a], "states": dummy}, role="policy")
                logits[a] = extra["net_output"]
        d = robot_station_dist()
        latched = env.latched[:, :N]
        nl = latched.sum(dim=1)
        fams = {
            "early_far": (t < 5) & (d >= 4).all(dim=1) & (nl == 0)
            if t < 5 else torch.zeros_like(nl, dtype=torch.bool),
            "both_adjacent": (d == 1).all(dim=1) & (nl == 0),
            "parked_near": ((t >= 20) & (nl == 0)
                            & (d >= 1).all(dim=1) & (d <= 3).all(dim=1))
            if t >= 20 else torch.zeros_like(nl, dtype=torch.bool),
            "one_latched_partner_near": (nl == 1)
            & (torch.where(latched, torch.zeros_like(d), d).max(dim=1).values
               <= 2),
            "one_latched_partner_far": (nl == 1)
            & (torch.where(latched, torch.zeros_like(d), d).max(dim=1).values
               >= 4),
            "both_latched": nl == N,
        }
        for name, mask in fams.items():
            if mask.any():
                sums[name] += float(v[mask].sum())
                counts[name] += int(mask.sum())
        acts = {a: logits[a].argmax(dim=-1) for a in AGENTS}
        env._pre_physics_step(acts)
        env.episode_length_buf += 1
        env._get_dones()
    for name in ("early_far", "both_adjacent", "parked_near",
                 "one_latched_partner_near", "one_latched_partner_far",
                 "both_latched"):
        if counts[name]:
            print("CRITIC ckpt=%s family=%s V=%.3f n=%d"
                  % (ck, name, sums[name] / counts[name], counts[name]),
                  flush=True)
        else:
            print("CRITIC ckpt=%s family=%s V=nan n=0 (never visited)"
                  % (ck, name), flush=True)

env.close()
app.close()
