#!/usr/bin/env python
"""Annealed mixture banks for curriculum round 2 (DECISIONS.md, OPEN(8)).

Builds four TRAINING-ONLY RoleBinding banks by deterministic row selection
and interleave from the three phase banks plus the certified bank's TRAIN
split, at fixed pre-declared ratios per stage:

    S1  60/25/10/ 5   (phase1/phase2/phase3/certified, percent)
    S2  30/30/25/15
    S3  10/20/30/40
    S4   5/10/15/70

Each stage bank carries 4096 train rows (all ratios satisfiable WITHOUT
replacement from the sources' train splits) plus the certified bank's 2048
EVAL rows appended verbatim with split=1 — so every stage bank contains
the certified eval pool and stage-boundary certified evals need no bank
swap.  Certified EVAL rows never enter any train pool (training draws
split==0 only), and phase eval rows are never used at all.

Mixture banks carry the SHA-covered ``curriculum_near_window`` stamp
(loader delta_gap relaxation, f6efd34) set to the WIDEST source window
[1, 10], plus a ``mixture`` meta block recording ratios, per-source row
counts and the four source SHAs.  Deterministic: same sources + seed ->
bit-identical .pt and SHA.

Usage (Isaac python, repo root):
    <isaac python> scripts/build_mixture_banks.py [--out-dir data]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import build_scenario_bank as bsb  # noqa: E402
from tasks.team_listen import scenario_bank  # noqa: E402

BANK_KEYS = scenario_bank.BANK_KEYS
MASTER_SEED = 52001

#: (tag, ratios over [phase1, phase2, phase3, certified-train]) — percent.
STAGES = [
    ("mix1", (60, 25, 10, 5)),
    ("mix2", (30, 30, 25, 15)),
    ("mix3", (10, 20, 30, 40)),
    ("mix4", (5, 10, 15, 70)),
]
K_TRAIN = 4096
STAMP = [1, 10]                 # widest source near-window (loader gate)


def _find_one(pattern):
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        sys.exit("expected exactly one bank matching %s, found %r"
                 % (pattern, hits))
    return hits[0]


def _alloc(k, ratios):
    """Largest-remainder integer allocation of k rows to the ratios."""
    exact = [k * r / 100.0 for r in ratios]
    base = [int(x) for x in exact]
    rem = k - sum(base)
    order = sorted(range(len(ratios)), key=lambda i: exact[i] - base[i],
                   reverse=True)
    for i in order[:rem]:
        base[i] += 1
    assert sum(base) == k
    return base


def _select(bank, n, gen):
    """First n of a seeded permutation of the bank's TRAIN-split rows."""
    pool = (bank.split == 0).nonzero(as_tuple=False).reshape(-1)
    assert pool.numel() >= n, (
        "source has %d train rows < %d requested" % (pool.numel(), n))
    perm = pool[torch.randperm(pool.numel(), generator=gen)]
    return perm[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--src-dir", default="data",
                    help="directory holding the three phase banks")
    ap.add_argument("--certified",
                    default="data/scenario_bank_RoleBinding_68d025a618cf.pt")
    ap.add_argument("--k-train", type=int, default=K_TRAIN, dest="k_train")
    args = ap.parse_args(argv)
    k_train = int(args.k_train)

    src_paths = {
        "phase1": _find_one(os.path.join(
            args.src_dir, "scenario_bank_RoleBinding_phase1_*.pt")),
        "phase2": _find_one(os.path.join(
            args.src_dir, "scenario_bank_RoleBinding_phase2_*.pt")),
        "phase3": _find_one(os.path.join(
            args.src_dir, "scenario_bank_RoleBinding_phase3_*.pt")),
        "certified": args.certified,
    }
    srcs = {name: scenario_bank.load_bank(p)          # production gate
            for name, p in src_paths.items()}
    names = ("phase1", "phase2", "phase3", "certified")

    cert = srcs["certified"]
    cert_eval = (cert.split == 1).nonzero(as_tuple=False).reshape(-1)

    for si, (tag, ratios) in enumerate(STAGES):
        counts = _alloc(k_train, ratios)
        rows = []                                     # (source, row) picks
        for src_i, (name, n) in enumerate(zip(names, counts)):
            # NB: a stable per-source index, never hash(name) -- python
            # string hashing is salted per process and would break the
            # bit-identical-rebuild guarantee.
            gen = torch.Generator().manual_seed(
                args.seed + 1000 * si + src_i)
            rows.append((name, _select(srcs[name], n, gen)))

        # deterministic interleave of the concatenated train rows
        parts = {key: [] for key in BANK_KEYS}
        for name, idx in rows:
            b = srcs[name]
            for key in BANK_KEYS:
                t = getattr(b, key)[idx]
                if key == "split":
                    t = torch.zeros_like(t)           # all TRAIN
                parts[key].append(t)
        gen = torch.Generator().manual_seed(args.seed + 7000 + si)
        order = torch.randperm(k_train, generator=gen)
        payload = {key: torch.cat(parts[key])[order] for key in BANK_KEYS}

        # append the certified EVAL pool verbatim (split stays 1)
        for key in BANK_KEYS:
            payload[key] = torch.cat(
                [payload[key], getattr(cert, key)[cert_eval]])

        # meta: inherit the certified source's flat meta, then override
        meta = dict(cert.meta)
        meta.update({
            "k": int(payload["occ"].shape[0]),
            "builder": "scripts/build_mixture_banks.py",
            "curriculum_near_window": list(STAMP),
            "mixture": {
                "stage": tag,
                "ratios_percent": dict(zip(names, ratios)),
                "train_counts": dict(zip(names, counts)),
                "k_train": k_train,
                "certified_eval_rows_appended": int(cert_eval.numel()),
                "sources": {n: srcs[n].sha256 for n in names},
                "master_seed": args.seed,
            },
        })
        payload.update(meta)

        data, sha = bsb.serialize_bank(payload)
        os.makedirs(args.out_dir, exist_ok=True)
        pt = os.path.join(args.out_dir,
                          "scenario_bank_RoleBinding_%s_%s.pt" % (tag, sha[:12]))
        with open(pt, "wb") as f:
            f.write(data)
        man = {"file": os.path.basename(pt), "sha256": sha,
               "variant": "RoleBinding", "tag": tag,
               "mixture": meta["mixture"],
               "curriculum_near_window": list(STAMP),
               "schema_version": meta.get("schema_version")}
        mp = os.path.join(args.out_dir,
                          "scenario_bank_RoleBinding_%s.json" % tag)
        with open(mp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=2, sort_keys=True)
            f.write("\n")
        print("wrote %s\n  sha256 %s\n  counts %s +%d certified-eval"
              % (pt, sha, dict(zip(names, counts)), cert_eval.numel()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
