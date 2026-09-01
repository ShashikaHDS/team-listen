#!/usr/bin/env python
"""Instrumented crash repro + mitigation probes (experiments-f1 diagnosis
work order, 2026-09-01; findings in docs/CRASH_DIAGNOSIS.md).

Re-runs one training cell with a finiteness tripwire wrapped around skrl
``MAPPO.update`` (skrl itself untouched) and per-update LR logging.  Launch
with ``CUDA_LAUNCH_BLOCKING=1`` so a device-side assert names its kernel in
the traceback.  The tripwire fires at each update() entry per agent —
per-minibatch granularity is redundant once the faulting kernel is named by
blocking mode — checking every float rollout tensor plus fresh policy
logits over the stored observations; on a trip it dumps stats and the
offending tensors to runs/diag/ and exits 3 BEFORE the bad kernel launches.

Mitigation probes (evidence only, never YAML edits — adoption is a joint
decision in docs/DECISIONS.md):

    --mitigation lr_ceiling   KLAdaptiveLR max_lr 5e-4 (in-process override)
    --mitigation obs_scaler   RunningStandardScaler observation preprocessor
    --mitigation none         plain repro (default)

Usage:
    CUDA_LAUNCH_BLOCKING=1 <isaac python> scripts/diag_repro.py \
        --task Isaac-TeamListen-RoleBinding-Lang-Direct-v0 --seed 0 --headless
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for tok in sys.argv[1:]:
    if tok.split("=", 1)[0] in ("--video", "--enable_cameras"):
        sys.exit("diag_repro.py: %r refused (spec 1.13)" % tok)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-TeamListen-RoleBinding-Lang-Direct-v0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--timesteps", type=int, default=None,
                    help="override trainer timesteps (default: YAML value)")
parser.add_argument("--mitigation", choices=("none", "lr_ceiling", "obs_scaler"),
                    default="none")
parser.add_argument("--logit_sample", type=int, default=16384,
                    help="stored observations to sweep for finite logits")
parser.add_argument("--no-tripwire", action="store_true", dest="no_tripwire",
                    help="skip the finiteness wrapper entirely (control runs "
                         "and mitigation probes: the tripwire's per-update "
                         "syncs SUPPRESS the crash, see CRASH_DIAGNOSIS.md)")

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

DIAG_DIR = REPO_ROOT / "runs" / "diag"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env = gym.make(args.task, cfg=env_cfg).unwrapped
assert not env.sim.has_gui() and not env.sim.has_rtx_sensors()

agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
agent_cfg["seed"] = args.seed
if args.timesteps is not None:
    agent_cfg["trainer"]["timesteps"] = args.timesteps
agent_cfg["trainer"]["close_environment_at_exit"] = False
if args.mitigation == "lr_ceiling":
    agent_cfg["agent"].setdefault("learning_rate_scheduler_kwargs", {})
    agent_cfg["agent"]["learning_rate_scheduler_kwargs"]["max_lr"] = 5.0e-4
elif args.mitigation == "obs_scaler":
    agent_cfg["agent"]["observation_preprocessor"] = "RunningStandardScaler"
    agent_cfg["agent"]["observation_preprocessor_kwargs"] = None

runner = Runner(SkrlVecEnvWrapper(env), agent_cfg)
agent = runner.agent
orig_update = agent.update
# Pre-update: rollout-filled tensors only.  "returns"/"advantages" are
# UNINITIALIZED until update() runs GAE, so they are checked post-update
# (their non-finiteness pre-update on the very first update is expected —
# tripwire v1 false-alarmed on exactly that).
PRE_TENSORS = ("observations", "states", "actions", "rewards", "log_prob",
               "values")
POST_TENSORS = ("returns", "advantages")
update_no = {"n": 0}


def _stats(t):
    t = t.detach()
    finite = torch.isfinite(t)
    return {"shape": tuple(t.shape), "n_nonfinite": int((~finite).sum()),
            "min": float(t[finite].min()) if finite.any() else None,
            "max": float(t[finite].max()) if finite.any() else None,
            "absmax": float(t[finite].abs().max()) if finite.any() else None}


def _trip(report, bad, timestep, uid, lr, phase):
    out = DIAG_DIR / ("trip_%s_s%d_t%d_%s_%s.pt"
                      % (args.task.split("-")[2], args.seed, timestep, uid,
                         phase))
    torch.save({"timestep": timestep, "uid": uid, "lr": lr, "phase": phase,
                "update_no": update_no["n"], "report": report,
                "bad_tensors": bad, "mitigation": args.mitigation}, out)
    print("TRIPWIRE phase=%s t=%d uid=%s lr=%.3e -> %s"
          % (phase, timestep, uid, lr, out), flush=True)
    print("TRIP_REPORT %s" % report, flush=True)
    sys.exit(3)


def _sweep(mem, names, report, bad):
    for name in names:
        try:
            t = mem.get_tensor_by_name(name)
        except Exception:
            continue
        report[name] = _stats(t)
        if report[name]["n_nonfinite"]:
            bad[name] = t.detach().to("cpu")


def guarded_update(*, timestep, timesteps, uid):
    update_no["n"] += 1
    lr = agent.optimizers[uid].param_groups[0]["lr"]
    agent.track_data("Diag / LR (%s)" % uid, lr)
    mem = agent.memories[uid]
    report, bad = {}, {}
    _sweep(mem, PRE_TENSORS, report, bad)
    try:
        # skrl 2.x models take BOTH keys (mappo.py update wiring): the
        # instantiated network consumes the one its `input:` token names.
        obs = mem.get_tensor_by_name("observations")
        states = mem.get_tensor_by_name("states")
        n = args.logit_sample
        flat_o = obs.reshape(-1, obs.shape[-1])[:n]
        flat_s = states.reshape(-1, states.shape[-1])[:n]
        with torch.no_grad():
            _, outputs = agent.policies[uid].act(
                {"observations": agent._observation_preprocessor[uid](flat_o),
                 "states": agent._state_preprocessor[uid](flat_s)},
                role="policy")
        logits = outputs.get("net_output")
        if logits is not None:
            report["policy_logits"] = _stats(logits)
            if report["policy_logits"]["n_nonfinite"]:
                bad["policy_logits"] = logits.detach().to("cpu")
    except Exception as exc:                     # the sweep itself faulted
        report["logit_sweep_error"] = repr(exc)
        bad["logit_sweep_error"] = repr(exc)
    if bad:
        _trip(report, bad, timestep, uid, lr, "pre")
    result = orig_update(timestep=timestep, timesteps=timesteps, uid=uid)
    report, bad = {}, {}
    _sweep(mem, POST_TENSORS, report, bad)       # GAE outputs now filled
    if bad:
        _trip(report, bad, timestep, uid, lr, "post")
    return result


if not args.no_tripwire:
    agent.update = guarded_update
print("DIAG_START task=%s seed=%d mitigation=%s timesteps=%s cuda_blocking=%s"
      % (args.task, args.seed, args.mitigation,
         agent_cfg["trainer"]["timesteps"],
         __import__("os").environ.get("CUDA_LAUNCH_BLOCKING", "unset")),
      flush=True)
t0 = time.time()
try:
    runner.run()
    print("DIAG_COMPLETE wall_s=%.0f" % (time.time() - t0), flush=True)
finally:
    sys.stdout.flush()
    sys.stderr.flush()
env.close()
app.close()
