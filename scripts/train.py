#!/usr/bin/env python
"""Thin training wrapper over the pinned Isaac Lab skrl train.py (M1_SPEC 1.13/1.14).

Usage (inside the Isaac python on the 5090):

    <isaac python> scripts/train.py --task Isaac-TeamListen-RoleBinding-Blind-Direct-v0 \
        --algorithm MAPPO --num_envs 4096 --headless [--max_iterations N] ...

What this wrapper does, and nothing else:

1. REFUSES ``--algorithm PPO`` (spec 1.14: the shipped train.py
   unconditionally applies ``multi_agent_to_single_agent`` for PPO on a
   DirectMARLEnv, and that shim is broken for discrete spaces --
   ``flatten_space(Tuple([Discrete(5), Discrete(5)])) = Box(shape=(10,))``,
   so the agent emits 10 floats where the env expects an integer index).
   ``--algorithm`` is REQUIRED and must be MAPPO or IPPO (the shipped
   default is PPO, so omitting the flag would silently take the broken path).
2. REFUSES ``--video`` and ``--enable_cameras`` outright (spec 1.13 [FIXED:
   render guard]: ``--video`` sets enable_cameras, creates the env with
   render_mode="rgb_array" and lets RecordVideo call env.render() ->
   sim.render(); a single stray camera costs a 20x throughput collapse with
   no error).
3. Forces ``cfg.debug_vis = False`` by exporting TEAM_LISTEN_FORCE_DEBUG_VIS_OFF,
   which TeamGridEnv.__init__ honours and which arms its training-mode
   render asserts.
4. Writes a run manifest (task, variant, arm, argv, leak_rho note, bank and
   lang-cache SHA-256 when locatable) under runs/manifests/ BEFORE training.
5. Delegates to the pinned checkout's skrl train.py via runpy in-process
   (so this wrapper must itself run under the Isaac python).

This wrapper imports ``tasks.team_listen`` (pure-metadata gym registration)
but deliberately NEVER imports fleet_env / fleet_env_cfg / isaaclab: those
must first be imported after SimulationApp boots inside the delegated
script, or sys.modules would cache their inert dev-box shims.
"""

import datetime
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: env var honoured by TeamGridEnv.__init__ (spec 1.13 guard 1).
FORCE_DEBUG_VIS_OFF = "TEAM_LISTEN_FORCE_DEBUG_VIS_OFF"

#: refused argv token families (exact token or --token=value form).
BANNED_TOKENS = ("--video", "--video_length", "--video_interval",
                 "--enable_cameras")

ALLOWED_ALGORITHMS = ("MAPPO", "IPPO")


def _die(msg):
    sys.stderr.write("scripts/train.py: ERROR: %s\n" % msg)
    raise SystemExit(2)


def _flag_value(argv, name):
    """Value of ``--name X`` or ``--name=X`` in argv, else None."""
    for i, tok in enumerate(argv):
        if tok == name:
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return None


def _refuse_banned(argv):
    for tok in argv:
        base = tok.split("=", 1)[0]
        if base in BANNED_TOKENS:
            _die(
                "%r is refused for every training run (M1_SPEC 1.13): "
                "--video/--enable_cameras re-enable rendering (RecordVideo "
                "-> env.render() -> sim.render()) and silently cost ~20x "
                "throughput. Rendering belongs to the separate month-8 "
                "benchmark entry point, never to training." % tok)


def _check_algorithm(argv):
    algo = _flag_value(argv, "--algorithm")
    if algo is None:
        _die(
            "--algorithm is required (MAPPO or IPPO). The shipped skrl "
            "train.py defaults to PPO, which is refused: "
            "multi_agent_to_single_agent is broken for discrete action "
            "spaces (M1_SPEC 1.14).")
    if algo.upper() == "PPO":
        _die(
            "--algorithm PPO is refused (M1_SPEC 1.14): the shipped train.py "
            "unconditionally applies multi_agent_to_single_agent(env) for "
            "PPO on a DirectMARLEnv, and flatten_space(Tuple([Discrete(5), "
            "Discrete(5)])) = Box(shape=(10,)) makes the agent emit 10 "
            "floats where the env expects an integer action index. Use "
            "--algorithm MAPPO (or IPPO for the paired control on the "
            "partial-observation arms).")
    if algo.upper() not in ALLOWED_ALGORITHMS:
        _die("--algorithm %r is not supported; use MAPPO or IPPO." % algo)
    return algo.upper()


def _sha256_or_none(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(argv, task_id, algorithm):
    """Run manifest (spec section 7: train.py 'logs the run manifest')."""
    import tasks.team_listen as tl  # pure metadata; never imports isaaclab

    meta = tl.TASK_TO_CFG.get(task_id)
    if meta is None:
        _die("unknown --task %r; registered Team Listen tasks:\n  %s"
             % (task_id, "\n  ".join(sorted(tl.TASK_TO_CFG))))

    bank_path = os.environ.get("TEAM_LISTEN_BANK", "")
    cache_path = os.environ.get("TEAM_LISTEN_LANG_CACHE", "")
    manifest = {
        "timestamp": datetime.datetime.now().isoformat(),
        "task": task_id,
        "variant": meta["variant"],
        "arm": meta["arm"],
        "env_cfg_entry_point": meta["env_cfg_entry_point"],
        "algorithm": algorithm,
        "argv": argv,
        # leak_rho is 0.0 for every non-Leaky arm by construction (the cfg
        # raises otherwise, M1_SPEC 1.10); Leaky rho cells are launch-time
        # cfg overrides and appear in argv above.
        "leak_rho_note": ("Leaky arm: rho set via cfg override, see argv"
                          if meta["arm"] == "Leaky"
                          else "0.0 (enforced by cfg __post_init__)"),
        "bank_path": bank_path or None,
        "bank_sha256": _sha256_or_none(bank_path),
        "lang_cache_path": cache_path or None,
        "lang_cache_sha256": _sha256_or_none(cache_path),
        "debug_vis": "forced False (%s=1)" % FORCE_DEBUG_VIS_OFF,
    }
    out_dir = Path.cwd() / "runs" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = out_dir / ("%s_%s.json" % (stamp, task_id))
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[team_listen] run manifest: %s" % out)
    return manifest


def _find_isaaclab_train_script():
    root = Path(os.environ.get("ISAACLAB_PATH", "~/IsaacLab")).expanduser()
    candidates = [
        root / "scripts" / "reinforcement_learning" / "skrl" / "train.py",
        root / "source" / "standalone" / "workflows" / "skrl" / "train.py",
    ]
    for c in candidates:
        if c.is_file():
            return c
    _die("cannot find the pinned Isaac Lab skrl train.py; set $ISAACLAB_PATH "
         "(tried: %s)" % ", ".join(str(c) for c in candidates))


def main():
    argv = sys.argv[1:]

    _refuse_banned(argv)
    algorithm = _check_algorithm(argv)
    task_id = _flag_value(argv, "--task")
    if not task_id:
        _die("--task is required (e.g. "
             "Isaac-TeamListen-RoleBinding-Blind-Direct-v0)")

    # spec 1.13 guard (1): force debug_vis off for the whole process; the
    # env also arms its training-mode render asserts off this flag.
    os.environ[FORCE_DEBUG_VIS_OFF] = "1"
    # make the repo importable for the delegated script and any subprocess
    os.environ["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))

    _write_manifest(argv, task_id, algorithm)

    script = _find_isaaclab_train_script()
    print("[team_listen] delegating to %s" % script)
    # tasks.team_listen is already imported (registrations live in this
    # process); the delegated script boots SimulationApp, then gym.make
    # resolves our string entry points, importing fleet_env post-boot.
    sys.argv = [str(script)] + argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    except ModuleNotFoundError as exc:
        _die("delegated script failed to import %r -- scripts/train.py must "
             "run under the Isaac Lab python (e.g. "
             "~/IsaacLab/isaaclab.sh -p scripts/train.py ...)" % exc.name)


if __name__ == "__main__":
    main()
