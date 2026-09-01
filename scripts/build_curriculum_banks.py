"""Build the OPEN(8) curriculum phase banks (DECISIONS 2026-09-01).

    python scripts/build_curriculum_banks.py [--k 4096] [--out-dir data]

Three RoleBinding TRAINING-ONLY banks with progressively wider
NEAREST-station distance windows (each robot spawns near a different
station), so early training makes accidental latches - and crucially
accidental DOUBLE latches - common enough to learn from:

    phase1  nearest-station distance [1, 3]   discovery nearly immediate
    phase2  nearest-station distance [3, 6]   intermediate
    phase3  nearest-station distance [6, 10]  near-final difficulty

Phase 4 of the curriculum IS the certified real bank (window [4, 14]) --
it is not rebuilt here, so the audit machinery and every evaluation
manifest remain untouched. Each phase uses a distinct master seed so the
banks are disjoint by construction. Curriculum banks are for TRAINING
exploration only and are never evaluated on.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_scenario_bank as bsb  # noqa: E402

PHASES = [
    ("phase1", 1, 3, 4101),   # nearest-station distance window
    ("phase2", 3, 6, 4102),
    ("phase3", 6, 10, 4103),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4096,
                    help="rows per phase bank (training-only, smaller than "
                         "the 16384-row certified bank is fine)")
    ap.add_argument("--out-dir", default=str(HERE.parents[0] / "data"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    for tag, lo, hi, seed in PHASES:
        rc = bsb.main([
            "--variant", "RoleBinding",
            "--num-scenarios", str(args.k),
            "--seed", str(seed),
            "--near-window", "%d,%d" % (lo, hi),
            "--tag", tag,
            "--out-dir", args.out_dir,
        ] + (["--quiet"] if args.quiet else []))
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
