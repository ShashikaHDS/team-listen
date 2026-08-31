"""Boot a headless SimulationApp, then run a test module under it.

The 5090-side fixture for tests whose imports need kit (pxr/omni) on the
Isaac python -- day-0 established that ``isaaclab.envs.utils.spaces``
cannot be imported before SimulationApp boots (M1_SPEC "resolvable only
on the 5090" list).  Usage, from the repo root:

    ~/IsaacLab/_isaac_sim/python.sh tests/sim_fixture.py tests/test_spaces.py

Exit code is the wrapped module's exit code; the app is closed either way.
"""

import runpy
import sys

from isaaclab.app import AppLauncher

if len(sys.argv) < 2:
    sys.exit("usage: sim_fixture.py <test_file.py> [args...]")

app = AppLauncher(headless=True).app
target, sys.argv = sys.argv[1], sys.argv[1:]
code = 0
try:
    runpy.run_path(target, run_name="__main__")
except SystemExit as exc:
    code = int(exc.code or 0)
finally:
    # kit's teardown can drop buffered python output; flush before close
    # so the wrapped module's pass/fail summary reaches the caller.
    print("SIM_FIXTURE_EXIT %d" % code, flush=True)
    sys.stderr.flush()
    app.close()
sys.exit(code)
