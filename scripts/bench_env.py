#!/usr/bin/env python
"""TeamGridEnv throughput benchmark (M1_SPEC 8.2; GATE_TASK item 3).

Boots one TeamListen task headless and measures sustained env-steps/s and
peak VRAM in two modes:

* ``stepping``: random discrete actions straight into the env's step loop
  (the "+ grid kernels" diagnostic floor, spec 8.1 row 2).
* ``training``: real skrl MAPPO/IPPO training via the same
  ``SkrlVecEnvWrapper`` + ``Runner`` path the pinned train.py uses.  This
  is the number the >=100k env-steps/s HARD GATE reads (spec 8.2 step 2).

Usage (Isaac python, repo root):

    _isaac_sim/python.sh scripts/bench_env.py --task Isaac-TeamListen-RoleBinding-Lang-Direct-v0 \
        --num_envs 4096 --mode both --headless

Env vars: TEAM_LISTEN_BANK / TEAM_LISTEN_LANG_CACHE resolve the data
artifacts when cfg paths are empty (fleet_env_cfg fallback).

Each mode prints a human table plus one machine-readable line:
    BENCH_RESULT {json}
Rollout-vs-optimiser wall-clock split (spec 8.2 step 2) is derived as
``update_share = 1 - training_fps / stepping_fps`` when both modes run.

Render guards (spec 1.13 / 8.2 step 4): ``--video`` and
``--enable_cameras`` are refused outright; after boot the benchmark
asserts ``not sim.has_gui()`` and ``not sim.has_rtx_sensors()``.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BANNED = ("--video", "--video_length", "--video_interval", "--enable_cameras")
for tok in sys.argv[1:]:
    if tok.split("=", 1)[0] in BANNED:
        sys.exit("bench_env.py: %r is refused (spec 1.13 render guard; a "
                 "single stray camera is a silent 20x collapse)" % tok)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-TeamListen-RoleBinding-Lang-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--mode", choices=("stepping", "training", "both"),
                    default="both")
parser.add_argument("--algorithm", choices=("MAPPO", "IPPO"), default="MAPPO")
parser.add_argument("--steps", type=int, default=2000,
                    help="timed decision steps in stepping mode")
parser.add_argument("--warmup", type=int, default=200,
                    help="untimed warm-up steps in stepping mode")
parser.add_argument("--timesteps", type=int, default=320,
                    help="training timesteps (rollouts=16 -> 20 updates)")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below runs under the booted app ---------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import tasks.team_listen  # noqa: F401, E402  (gym registration)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402


class VramPoller:
    """Samples process-level GPU memory via nvidia-smi in a daemon thread."""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                self.peak_mib = max(self.peak_mib, int(out.split()[0]))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False


def emit(payload):
    print("BENCH_RESULT " + json.dumps(payload), flush=True)


def make_env():
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    # spec 8.2 step 4 smoke assertions -- fail loudly, never silently render
    assert not env.sim.has_gui(), "sim.has_gui() is True in a benchmark"
    assert not env.sim.has_rtx_sensors(), "rtx sensors active in a benchmark"
    return env


def bench_stepping(env):
    agents = list(env.cfg.possible_agents)
    E = env.num_envs
    dev = env.device
    gen = torch.Generator(device="cpu").manual_seed(0)

    def rand_actions():
        return {a: torch.randint(0, 5, (E,), generator=gen).to(dev)
                for a in agents}

    env.reset()
    for _ in range(args.warmup):
        env.step(rand_actions())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with VramPoller() as poller:
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(rand_actions())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    fps = args.steps * E / elapsed
    payload = {
        "mode": "stepping", "task": args.task, "num_envs": E,
        "steps": args.steps, "elapsed_s": round(elapsed, 3),
        "env_steps_per_s": round(fps),
        "torch_peak_alloc_mib": round(
            torch.cuda.max_memory_allocated() / 2**20),
        "process_peak_vram_mib": poller.peak_mib,
    }
    emit(payload)
    return payload


def bench_training(env):
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from skrl.utils.runner.torch import Runner

    entry = "skrl_%s_cfg_entry_point" % args.algorithm.lower()
    agent_cfg = load_cfg_from_registry(args.task, entry)
    agent_cfg["trainer"]["timesteps"] = args.timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0      # no TB I/O
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # no ckpt I/O

    wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped, agent_cfg)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with VramPoller() as poller:
        t0 = time.perf_counter()
        runner.run()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    E = env.num_envs
    fps = args.timesteps * E / elapsed
    payload = {
        "mode": "training", "task": args.task, "algorithm": args.algorithm,
        "num_envs": E, "timesteps": args.timesteps,
        "elapsed_s": round(elapsed, 3), "env_steps_per_s": round(fps),
        "torch_peak_alloc_mib": round(
            torch.cuda.max_memory_allocated() / 2**20),
        "process_peak_vram_mib": poller.peak_mib,
        "gate_100k": "PASS" if fps >= 100_000 else "FAIL",
    }
    emit(payload)
    return payload


def main():
    env = make_env()
    step_p = train_p = None
    if args.mode in ("stepping", "both"):
        step_p = bench_stepping(env)
    if args.mode in ("training", "both"):
        train_p = bench_training(env)
    if step_p and train_p:
        split = 1.0 - train_p["env_steps_per_s"] / step_p["env_steps_per_s"]
        emit({"mode": "split", "num_envs": env.num_envs,
              "update_share_of_wallclock": round(split, 3)})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
