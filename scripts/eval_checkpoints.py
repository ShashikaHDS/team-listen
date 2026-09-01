#!/usr/bin/env python
"""Competence/leakage eval of a trained checkpoint on the Isaac env
(experiments-f1 diagnosis work order item 3).

Loads a skrl MAPPO checkpoint, drives ``harness.rollout.run_lanes`` over
the variant's real bank on a PAIRED manifest (eval split, argmax mode) and
feeds the record to ``harness.certificate.run_certificate``, printing the
full verdict dict.  Base scenarios are the first ``--n_base`` eval-split
rows in ascending bank order (deterministic; the frozen
``config/eval_manifest_{variant}.pt`` builder is not in the repo yet — noted
in the output so the final audit can re-run against the frozen manifest).

Usage (Isaac python, repo root; num_envs is derived as 2 x n_base):

    <isaac python> scripts/eval_checkpoints.py \
        --task Isaac-TeamListen-RoleBinding-Blind-Direct-v0 \
        --checkpoint logs/skrl/runs/<dir>/checkpoints/best_agent.pt --headless
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--n_base", type=int, default=2000)
parser.add_argument("--tag", default="")

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

variant = args.task.split("-")[2]

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 2 * args.n_base
env = gym.make(args.task, cfg=env_cfg).unwrapped
assert not env.sim.has_gui() and not env.sim.has_rtx_sensors()

agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
agent_cfg["trainer"]["close_environment_at_exit"] = False
agent_cfg["agent"]["experiment"]["write_interval"] = 0
agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
runner = Runner(SkrlVecEnvWrapper(env), agent_cfg)
runner.agent.load(str(REPO_ROOT / args.checkpoint))
policies = runner.agent.policies
for m in policies.values():
    m.eval()


STATE_W = int(env.cfg.state_space)


def policy(obs):
    # The compiled policy consumes inputs["observations"]; "states" must
    # still unflatten cleanly against the 641-d state space, so feed a
    # correctly-shaped zero placeholder (its value is unused by the actor).
    out = {}
    with torch.no_grad():
        for a, o in obs.items():
            dummy = torch.zeros(o.shape[0], STATE_W, device=o.device)
            _, extra = policies[a].act(
                {"observations": o, "states": dummy}, role="policy")
            out[a] = extra["net_output"]
    return out


eval_pool = (env._bank.split == 1).nonzero(as_tuple=False).reshape(-1)
assert eval_pool.numel() >= args.n_base, "eval split smaller than n_base"
base = eval_pool[: args.n_base].cpu()

g = torch.Generator().manual_seed(17)
half = args.n_base // 2
classes = torch.cat([torch.zeros(args.n_base - half, dtype=torch.long),
                     torch.ones(half, dtype=torch.long)])
classes = classes[torch.randperm(args.n_base, generator=g)]

plan = ro.make_paired_plan(base, classes_a=classes)
rec = ro.run_lanes(env, policy, plan, mode="argmax")
verdict, report = ce_mod.run_certificate(rec, plan, env._bank, variant, seed=0)

tag = args.tag or Path(args.checkpoint).parent.parent.name
out_path = REPO_ROOT / "runs" / "diag" / ("certeval_%s.json" % tag)
out_path.parent.mkdir(parents=True, exist_ok=True)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.tolist() if x.numel() <= 8 else "tensor%s" % (tuple(x.shape),)
    if isinstance(x, float) and x != x:
        return "nan"
    return x


out_path.write_text(json.dumps(
    {"task": args.task, "checkpoint": args.checkpoint, "n_base": args.n_base,
     "verdict": _clean(verdict)}, indent=1, default=str))
print("CERT_EVAL_JSON %s" % out_path, flush=True)
print("CERT_REPORT_BEGIN\n%s\nCERT_REPORT_END" % report, flush=True)
print("CERT_VERDICT %s" % json.dumps(_clean(verdict), default=str)[:2000],
      flush=True)

env.close()
app.close()
