#!/usr/bin/env python
"""One-shot language-cache builder (M1_SPEC 5.1-5.4).

Encodes the 480 template-grammar sentences (240 single-axis + 240 composed,
``harness/templates.py``) with the frozen ``sentence-transformers/
all-MiniLM-L6-v2`` encoder, applies a fixed seeded Gaussian JL projection
384 -> LANG_DIM = 32 with L2 re-normalization, and writes everything into
``data/lang_cache_{sha}.pt``. Built once, float32; training and every
evaluation load this artifact and never re-encode on the fly (encoding the
same sentence under a different batch size / padding / autocast config gives
bitwise-different vectors, M1_SPEC 5.3).

Before writing, it reports the offline linear-probe accuracy of the 32-d
cache on held-out families (M1_SPEC 5.4 [FIXED]): if that is ~100%, held-out
sentence transfer is a property of MiniLM's paraphrase geometry, not
evidence about the policy, and the paper must say so.

This script REQUIRES sentence-transformers and therefore runs on the
training box (the 5090); the Windows dev box intentionally does not have it
and the script fails fast with a clear message there.

Usage (on the 5090):
    python scripts/build_lang_cache.py [--out-dir data] [--device cpu]
                                       [--batch-size 64] [--proj-seed 42]
"""

import argparse
import hashlib
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from harness import templates

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
LANG_DIM = 32                      # M1_SPEC 1.3
DEFAULT_PROJ_SEED = 42
DEFAULT_BATCH_SIZE = 64            # part of the artifact's bitwise identity

# Keys hashed into artifact_sha256, in this exact order. Everything a
# consumer reads is covered; artifact_sha256 itself and the derived
# probe_report are excluded.
_HASH_KEYS = (
    "sentences", "instr_id", "class_id", "class_name", "variant",
    "family_id", "frame_idx", "style_idx", "agent_idx", "verb_idx",
    "split", "minimal_pair_id", "flip_type",
    "composed_role_class", "composed_order_class",
    "emb384", "emb32", "cos384_to_pair", "cos32_to_pair",
    "jl_proj", "proj_seed", "batch_size",
    "encoder_name", "encoder_revision", "builder_git_sha",
    "held_out_composition", "lang_dim", "emb_dim", "sampling_seed",
)


def _require_sentence_transformers():
    """Fail fast, loudly and helpfully, when the encoder stack is absent."""
    try:
        import sentence_transformers  # noqa: F401
        return
    except Exception as exc:
        bar = "=" * 74
        msg = (
            "\n{bar}\n"
            "ERROR: sentence-transformers is not importable in this Python "
            "environment\n"
            "       ({etype}: {err}).\n\n"
            "scripts/build_lang_cache.py performs the one-shot MiniLM encode of "
            "the 480\ntemplate-grammar sentences (M1_SPEC 5.1/5.3) and must run "
            "on the training box\n(the Linux 5090), where "
            "sentence-transformers/all-MiniLM-L6-v2 is available.\n"
            "The Windows dev box intentionally lacks sentence-transformers and "
            "pip installs\nare SSL-blocked there.\n\n"
            "Do NOT substitute a hand-rolled AutoModel + mean-pool encoder: "
            "M1_SPEC 5.1\nrejects it explicitly (it silently reproduces the "
            "14.9% raw-BERT result).\n\n"
            "On the 5090:\n"
            "    pip install sentence-transformers\n"
            "    python scripts/build_lang_cache.py\n"
            "{bar}\n"
        ).format(bar=bar, etype=type(exc).__name__, err=exc)
        sys.stderr.write(msg)
        sys.exit(2)


def encode_sentences(texts, batch_size, device):
    """One-shot MiniLM encode with the exact M1_SPEC 5.3 call signature."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER_NAME, device=device)
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape != (len(texts), EMB_DIM):
        raise RuntimeError("unexpected embedding shape %r" % (emb.shape,))
    return torch.from_numpy(emb)


def jl_project(emb384, proj_seed):
    """Fixed seeded Gaussian JL projection to LANG_DIM + L2 re-norm.

    Computed in float64 for numerical stability, stored as float32
    (M1_SPEC 5.2). Returns (emb32, projection matrix).
    """
    g = torch.Generator().manual_seed(int(proj_seed))
    proj = torch.randn(EMB_DIM, LANG_DIM, generator=g, dtype=torch.float64)
    proj = proj / (LANG_DIM ** 0.5)
    x = emb384.to(torch.float64) @ proj
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x.to(torch.float32), proj.to(torch.float32)


def pair_cos_distance(emb, pair_idx):
    """Cosine distance (1 - cos) from each row to its minimal-pair partner."""
    x = emb.to(torch.float64)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = 1.0 - (x * x[pair_idx]).sum(dim=1)
    return d.to(torch.float32)


def _ridge_probe_accuracy(X_tr, y_tr, X_te, y_te, n_classes, lam=1e-3):
    """Closed-form one-vs-rest ridge probe; returns (train_acc, test_pred)."""
    def with_bias(X):
        return np.hstack([X, np.ones((len(X), 1), dtype=np.float64)])

    Xtr = with_bias(np.asarray(X_tr, dtype=np.float64))
    Y = np.eye(n_classes, dtype=np.float64)[np.asarray(y_tr)]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Y)
    train_pred = (Xtr @ W).argmax(1)
    train_acc = float((train_pred == np.asarray(y_tr)).mean())
    test_pred = (with_bias(np.asarray(X_te, dtype=np.float64)) @ W).argmax(1)
    return train_acc, test_pred


def offline_probe_report(rows, emb32):
    """Linear-probe accuracy of the 32-d cache on held-out families.

    Reported per variant, with the per-family breakdown (held-out statistics
    are clustered by family, n = 6; M1_SPEC 2.2/5.4).
    """
    emb = emb32.numpy()
    report = {}
    for variant in templates.VARIANTS:
        idx = [s.instr_id for s in rows if s.variant == variant]
        classes = sorted({rows[i].class_id for i in idx})
        remap = {c: j for j, c in enumerate(classes)}
        tr = [i for i in idx if rows[i].split == 0]
        te = [i for i in idx if rows[i].split == 1]
        y_tr = [remap[rows[i].class_id] for i in tr]
        y_te = [remap[rows[i].class_id] for i in te]
        train_acc, pred = _ridge_probe_accuracy(
            emb[tr], y_tr, emb[te], y_te, n_classes=len(classes))
        correct = pred == np.asarray(y_te)
        heldout_acc = float(correct.mean())
        by_family = defaultdict(list)
        for ok, i in zip(correct, te):
            by_family[rows[i].family_id].append(bool(ok))
        fam_accs = {fid: float(np.mean(v)) for fid, v in sorted(by_family.items())}
        report[variant] = {
            "n_classes": len(classes),
            "train_acc": train_acc,
            "heldout_acc": heldout_acc,
            "heldout_acc_by_family": fam_accs,
            "heldout_acc_family_mean": float(np.mean(list(fam_accs.values()))),
        }
    return report


def _builder_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _encoder_revision():
    """Best-effort HF commit hash of the pinned encoder, offline-safe.

    Reads refs/main from the local huggingface cache; falls back to
    "unknown" rather than touching the network.
    """
    try:
        hf_home = os.environ.get("HF_HOME")
        base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
        hub = base if base.name == "hub" else base / "hub"
        ref = hub / ("models--" + ENCODER_NAME.replace("/", "--")) / "refs" / "main"
        if ref.is_file():
            rev = ref.read_text(encoding="utf-8").strip()
            if rev:
                return rev
    except Exception:
        pass
    return "unknown"


def compute_artifact_sha(payload):
    """SHA-256 over a canonical byte serialization of the hashed keys."""
    h = hashlib.sha256()
    for key in _HASH_KEYS:
        v = payload[key]
        h.update(key.encode("utf-8"))
        h.update(b"\x1e")
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.cpu().contiguous().numpy().tobytes())
        elif isinstance(v, (list, tuple)):
            h.update("\x1f".join(str(x) for x in v).encode("utf-8"))
        else:
            h.update(repr(v).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def build_payload(args):
    rows = templates.build_all()
    assert len(rows) == templates.N_SENTENCES == 480
    texts = [s.text for s in rows]
    pair_idx = torch.tensor([s.minimal_pair_id for s in rows], dtype=torch.long)

    print("Encoding %d sentences with %s (batch_size=%d, device=%s) ..."
          % (len(texts), ENCODER_NAME, args.batch_size, args.device))
    emb384 = encode_sentences(texts, args.batch_size, args.device)
    emb32, jl_proj = jl_project(emb384, args.proj_seed)

    payload = {
        # --- M1_SPEC 5.3 schema ---
        "sentences": texts,
        "instr_id": torch.tensor([s.instr_id for s in rows], dtype=torch.long),
        "class_id": torch.tensor([s.class_id for s in rows], dtype=torch.long),
        "variant": [s.variant for s in rows],
        "family_id": torch.tensor([s.family_id for s in rows], dtype=torch.long),
        "split": torch.tensor([s.split for s in rows], dtype=torch.long),
        "minimal_pair_id": pair_idx,
        "flip_type": [s.flip_type for s in rows],
        "composed_role_class": torch.tensor(
            [s.composed_role_class for s in rows], dtype=torch.long),
        "composed_order_class": torch.tensor(
            [s.composed_order_class for s in rows], dtype=torch.long),
        "emb384": emb384,
        "emb32": emb32,
        "cos384_to_pair": pair_cos_distance(emb384, pair_idx),
        "cos32_to_pair": pair_cos_distance(emb32, pair_idx),
        "proj_seed": int(args.proj_seed),
        "encoder_name": ENCODER_NAME,
        "encoder_revision": _encoder_revision(),
        "builder_git_sha": _builder_git_sha(),
        # --- provenance extras (deterministic, hashed) ---
        "class_name": [s.class_name for s in rows],
        "frame_idx": torch.tensor([s.frame_idx for s in rows], dtype=torch.long),
        "style_idx": torch.tensor([s.style_idx for s in rows], dtype=torch.long),
        "agent_idx": torch.tensor([s.agent_idx for s in rows], dtype=torch.long),
        "verb_idx": torch.tensor([s.verb_idx for s in rows], dtype=torch.long),
        "jl_proj": jl_proj,
        "batch_size": int(args.batch_size),
        "held_out_composition": tuple(templates.HELD_OUT_COMPOSITION),
        "lang_dim": LANG_DIM,
        "emb_dim": EMB_DIM,
        "sampling_seed": templates._SAMPLING_SEED,
    }

    # Offline held-out-family linear probe: reported BEFORE any training step
    # (derived from hashed fields; not itself part of the artifact identity).
    payload["probe_report"] = offline_probe_report(rows, emb32)
    payload["artifact_sha256"] = compute_artifact_sha(payload)
    return rows, payload


def print_summary(rows, payload):
    cos384 = payload["cos384_to_pair"]
    cos32 = payload["cos32_to_pair"]
    print("\nMinimal-pair cosine distances (mean; the CSI table column):")
    print("  %-14s %-12s %8s %8s %6s" % ("variant", "flip", "cos384", "cos32", "n"))
    for variant in templates.VARIANTS:
        for flip in templates.FLIP_TYPES:
            sel = [s.instr_id for s in rows
                   if s.variant == variant and s.flip_type == flip]
            if not sel:
                continue
            print("  %-14s %-12s %8.4f %8.4f %6d" % (
                variant, flip,
                cos384[sel].mean().item(), cos32[sel].mean().item(), len(sel)))

    print("\nOffline held-out-family linear probe on the 32-d cache "
          "(clustered by family, n=6):")
    for variant, rep in payload["probe_report"].items():
        print("  %-14s train_acc=%.3f  heldout_acc=%.3f  family_mean=%.3f"
              % (variant, rep["train_acc"], rep["heldout_acc"],
                 rep["heldout_acc_family_mean"]))
        fams = "  ".join("f%02d=%.2f" % (fid, acc)
                         for fid, acc in rep["heldout_acc_by_family"].items())
        print("      per-family: %s" % fams)
    print("  NOTE (M1_SPEC 5.4 [FIXED]): if held-out accuracy is ~1.0, "
          "held-out-sentence\n  transfer measures MiniLM's paraphrase "
          "geometry, not the policy.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data"),
                        help="output directory (default: <repo>/data)")
    parser.add_argument("--device", default="cpu",
                        help="encode device; cpu is seconds for 480 sentences")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="encoder batch size; part of the artifact's "
                             "bitwise identity, do not casually change")
    parser.add_argument("--proj-seed", type=int, default=DEFAULT_PROJ_SEED,
                        help="seed of the fixed Gaussian JL projection")
    args = parser.parse_args(argv)

    try:  # keep unicode sentence text printable on cp1252 consoles
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    _require_sentence_transformers()

    rows, payload = build_payload(args)
    print_summary(rows, payload)

    sha = payload["artifact_sha256"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("lang_cache_%s.pt" % sha[:12])
    torch.save(payload, str(out_path))

    print("\nWrote %s" % out_path)
    print("  artifact_sha256  %s" % sha)
    print("  encoder          %s @ %s" % (ENCODER_NAME, payload["encoder_revision"]))
    print("  builder_git_sha  %s" % payload["builder_git_sha"])
    print("  emb384 %s  emb32 %s  proj_seed %d  batch_size %d"
          % (tuple(payload["emb384"].shape), tuple(payload["emb32"].shape),
             payload["proj_seed"], payload["batch_size"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
