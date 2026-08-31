#!/usr/bin/env python3
"""Day-0 version gate (docs/M1_SPEC.md section 0) -- blocking prerequisite.

Writes `config/environment.lock.yaml` recording the versions every later
assumption is anchored to:

    isaac_sim              SimulationApp banner / install VERSION file
    isaaclab               isaaclab.__version__ + `git rev-parse HEAD`
    skrl                   skrl.__version__   (HARD GATE: >= 2.0.0)
    gymnasium              module attr (gymnasium>=1.0 + skrl<1.4 breaks MAPPO)
    torch / numpy / sentence-transformers    module attrs (embedding-cache repro)

Degrades gracefully on machines where a package is absent (the Windows dev
box has no importable isaaclab/skrl -- they live in the Isaac python on the
5090): the key is recorded as "absent" and, for skrl, the hard gate is
reported as UNCHECKED rather than failed. The gate only FAILS (exit 1) when
skrl is present and older than 2.0.0.

The authoritative isaac_sim source is the SimulationApp banner, but booting
SimulationApp costs 40-90 s and is impossible off the training box, so by
default only lightweight probes run (pip metadata, install VERSION file).
Pass --boot-sim on the 5090 to boot a headless SimulationApp and read the
version from inside it.

Usage:
    python scripts/print_versions.py [--out PATH] [--boot-sim]
"""

import argparse
import datetime
import importlib
import importlib.util
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

ABSENT = "absent"
SKRL_MIN = (2, 0, 0)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "config" / "environment.lock.yaml"


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def _dist_version(dist_name):
    """Version from pip metadata, without importing the package (cheap and
    safe: importing e.g. isaaclab outside the Isaac python raises)."""
    try:
        from importlib import metadata
        return metadata.version(dist_name)
    except Exception:
        return None


def _module_version(mod_name):
    """Version from `<module>.__version__`; None on any failure."""
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def probe_version(dist_name, mod_name):
    """Best-effort version string, or ABSENT."""
    v = _dist_version(dist_name)
    if v is None:
        v = _module_version(mod_name)
    return v if v is not None else ABSENT


def _git_sha(directory):
    """`git rev-parse HEAD` for the repo containing `directory`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def probe_isaaclab():
    """(version, git_sha) for the Isaac Lab checkout, or ("absent", "absent").

    The git SHA matters because the MAPPO YAML must be copied verbatim from
    this exact checkout (spec 0 / 1.14). The module path is located with
    find_spec (no import executed -- isaaclab's import chain pulls omni
    modules that fail outside the Isaac python).
    """
    version = _dist_version("isaaclab")
    spec = None
    try:
        spec = importlib.util.find_spec("isaaclab")
    except Exception:
        spec = None
    if version is None and spec is None:
        return ABSENT, ABSENT
    if version is None:
        version = _module_version("isaaclab") or "present (version unreadable)"
    git_sha = None
    if spec is not None and spec.origin:
        git_sha = _git_sha(Path(spec.origin).parent)
    return version, git_sha or "unknown (no git checkout found)"


def _read_version_file(root):
    """Isaac Sim standalone installs ship a VERSION file at the root."""
    try:
        p = Path(root) / "VERSION"
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return text.splitlines()[0].strip()
    except Exception:
        pass
    return None


def probe_isaac_sim(boot):
    """Best-effort Isaac Sim version.

    Order: pip metadata (pip-installed isaacsim) -> VERSION file via
    ISAAC_PATH-style env vars -> VERSION file near the isaacsim module ->
    (only with --boot-sim) boot a headless SimulationApp and ask it.
    """
    v = _dist_version("isaacsim")
    if v:
        return v
    # standalone install: ~/isaac-sim-standalone-5.1.0-linux-x86_64/VERSION
    for env_var in ("ISAAC_PATH", "ISAAC_SIM_PATH", "EXP_PATH", "CARB_APP_PATH"):
        root = os.environ.get(env_var)
        if not root:
            continue
        for candidate in (Path(root), Path(root).parent):
            v = _read_version_file(candidate)
            if v:
                return v
    try:
        spec = importlib.util.find_spec("isaacsim")
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        for candidate in Path(spec.origin).resolve().parents:
            v = _read_version_file(candidate)
            if v:
                return v
        if not boot:
            return "present (version unreadable; rerun with --boot-sim)"
    if boot:
        v = _boot_sim_version()
        if v:
            return v
    return ABSENT


def _boot_sim_version():
    """Boot a headless SimulationApp and read the version from inside it.

    Expensive (spec 0: the banner is the authoritative source); only reached
    under --boot-sim. Returns None if SimulationApp cannot be created.
    """
    SimulationApp = None
    for mod_name in ("isaacsim", "omni.isaac.kit"):
        try:
            SimulationApp = getattr(importlib.import_module(mod_name), "SimulationApp")
            break
        except Exception:
            continue
    if SimulationApp is None:
        return None
    app = SimulationApp({"headless": True})
    try:
        for mod_name in ("isaacsim.core.version", "omni.isaac.version"):
            try:
                ver = importlib.import_module(mod_name).get_version()
                # get_version() returns a sequence whose first element is the
                # full version string.
                if ver:
                    return str(ver[0])
            except Exception:
                continue
        return "booted (version API unavailable)"
    finally:
        try:
            app.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# skrl hard gate
# ---------------------------------------------------------------------------

def _version_tuple(v):
    """Leading numeric components of a version string ('2.0.0b1+x' -> (2,0,0))."""
    nums = re.findall(r"\d+", str(v).split("+")[0])
    return tuple(int(n) for n in nums[:3]) or (0,)


def check_skrl_gate(skrl_version):
    """Returns (status, message); status in {'pass', 'FAIL', 'unchecked'}."""
    if skrl_version == ABSENT:
        return ("unchecked",
                "skrl absent on this machine; the >=2.0.0 hard gate MUST be "
                "re-run and pass on the training box before anything else")
    if _version_tuple(skrl_version) >= SKRL_MIN:
        return "pass", "skrl {} >= 2.0.0".format(skrl_version)
    return ("FAIL",
            "skrl {} < 2.0.0 -- HARD GATE (M1_SPEC section 0); the 1.4.x "
            "rename set (state_spaces, state_preprocessor, ...) is not "
            "supported".format(skrl_version))


# ---------------------------------------------------------------------------
# YAML emission (hand-rolled: flat mapping, no pyyaml dependency)
# ---------------------------------------------------------------------------

def _y(value):
    """Quote a scalar for YAML."""
    s = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return '"' + s + '"'


def build_lock_yaml(info):
    lines = [
        "# Generated by scripts/print_versions.py -- M1_SPEC section 0 day-0 version gate.",
        "# Regenerate on the training box (with --boot-sim) before any training run;",
        "# a lock written on a machine where isaaclab/skrl are 'absent' does NOT",
        "# satisfy the blocking prerequisite.",
        "generated_at: " + _y(info["generated_at"]),
        "hostname: " + _y(info["hostname"]),
        "platform: " + _y(info["platform"]),
        "python: " + _y(info["python"]),
        "isaac_sim: " + _y(info["isaac_sim"]),
        "isaaclab:",
        "  version: " + _y(info["isaaclab_version"]),
        "  git_sha: " + _y(info["isaaclab_git_sha"]),
        "skrl: " + _y(info["skrl"]),
        "skrl_gate: " + _y(info["skrl_gate"]),
        "gymnasium: " + _y(info["gymnasium"]),
        "torch: " + _y(info["torch"]),
        "numpy: " + _y(info["numpy"]),
        "sentence-transformers: " + _y(info["sentence_transformers"]),
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def collect(boot_sim=False):
    isaaclab_version, isaaclab_git_sha = probe_isaaclab()
    skrl = probe_version("skrl", "skrl")
    gate_status, gate_msg = check_skrl_gate(skrl)
    return {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "isaac_sim": probe_isaac_sim(boot_sim),
        "isaaclab_version": isaaclab_version,
        "isaaclab_git_sha": isaaclab_git_sha,
        "skrl": skrl,
        "skrl_gate": "{}: {}".format(gate_status, gate_msg),
        "_skrl_gate_status": gate_status,
        "gymnasium": probe_version("gymnasium", "gymnasium"),
        "torch": probe_version("torch", "torch"),
        "numpy": probe_version("numpy", "numpy"),
        # pip name uses a hyphen, module name an underscore
        "sentence_transformers": probe_version("sentence-transformers",
                                               "sentence_transformers"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="lock file path (default: config/environment.lock.yaml)")
    parser.add_argument("--boot-sim", action="store_true",
                        help="boot a headless SimulationApp to read the Isaac Sim "
                             "version (slow; training box only)")
    args = parser.parse_args(argv)

    info = collect(boot_sim=args.boot_sim)
    text = build_lock_yaml(info)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")

    sys.stdout.write(text)
    sys.stdout.write("\nwrote {}\n".format(args.out))

    if info["_skrl_gate_status"] == "FAIL":
        sys.stderr.write("FATAL: skrl hard gate failed (see skrl_gate above)\n")
        return 1
    if info["_skrl_gate_status"] == "unchecked":
        sys.stderr.write("WARNING: skrl absent -- hard gate UNCHECKED on this "
                         "machine; rerun on the training box.\n")
    for key in ("isaac_sim", "isaaclab_version"):
        if info[key] == ABSENT:
            sys.stderr.write("WARNING: {} recorded as 'absent'; this lock does "
                             "not satisfy the section 0 prerequisite.\n"
                             .format(key.replace("_version", "")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
