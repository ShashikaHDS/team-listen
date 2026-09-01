"""Supervised leak probe -- the audit's PRIMARY instrument (M1_SPEC 2.4).

Spec 2.4 [FIXED: the load-bearing independence test could not fail],
instrument 2: a small classifier predicts ``instr_class`` from (a) the
blind arm's t = 0 observation, (b) the full 641-d state, (c) the entire
recorded trajectory (``realised_positions_blob``).  Requirement for the
certificate: **held-out AUC 95% CI contains 0.5** on every feature set.
The probe is orders of magnitude more sample-efficient than MAPPO at
extracting a weak coupling, so the RL oracle is demoted to behavioural
confirmation and this probe gates the audit (spec 4.4 clause 1(b)).

Implementation constraints (this box): pure numpy + torch -- **no
sklearn** (not installed), no scipy, no Isaac imports.  The probe is a
closed-form ridge regression on +-1 labels (deterministic, seed-free fit;
the only randomness is the cluster fold assignment and the bootstrap,
both seeded), scored by rank AUC.

Leakage hygiene, both mandatory:

* **Cluster-aware cross-fitting** -- train/held-out folds split by BASE
  SCENARIO (the spec 4.4 clustering unit), never by episode, so a
  scenario can never appear on both sides of the split (the paired
  manifest runs one scenario under both instruction classes).
* **Cluster bootstrap CI** -- the held-out AUC CI resamples base
  scenarios, not episodes (``harness/stats.py::bootstrap_ci_stat``),
  matching the design effect the rest of the audit accounts for.

``planted_leak_control`` is the audit's positive control: it plants a
label-carrying feature into the same X and re-runs the identical
pipeline.  If the planted leak is not detected at this n / dimensionality
/ regularisation, a null result on the real features is UNINFORMATIVE,
not clean -- certificate.py enforces exactly that.
"""

import math
import os
import sys

import numpy as np
import torch

try:
    from harness import stats
except ImportError:  # standalone import: put the repo root on sys.path
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from harness import stats


# ---------------------------------------------------------------------------
# Rank AUC (average ranks for ties; no sklearn)
# ---------------------------------------------------------------------------

def _rankdata(x):
    """Average ranks (1-based), ties averaged -- the Mann-Whitney rank."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc_score(scores, labels):
    """Mann-Whitney AUC of ``scores`` for binary ``labels`` (1 = positive).
    Returns nan when a class is absent (bootstrap replicates drop nan)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    pos = labels == 1
    n1 = int(pos.sum())
    n0 = scores.shape[0] - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------------------
# Ridge probe (closed form; deterministic)
# ---------------------------------------------------------------------------

def _standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)          # constant columns pass through
    return mu, sd


def ridge_fit(X, y, l2):
    """Closed-form ridge on +-1 labels: w = (Xc'Xc + l2 I)^-1 Xc' yc.

    ``X`` (n, p) float64, ``y`` in {0, 1}.  Centering supplies the
    intercept; AUC is shift-invariant so it is never added back.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1) * 2.0 - 1.0
    mu, sd = _standardize_fit(X)
    Xc = (X - mu) / sd
    yc = y - y.mean()
    p = Xc.shape[1]
    A = Xc.T @ Xc + float(l2) * np.eye(p)
    w = np.linalg.solve(A, Xc.T @ yc)
    return {"w": w, "mu": mu, "sd": sd}


def ridge_decision(model, X):
    X = np.asarray(X, dtype=np.float64)
    return ((X - model["mu"]) / model["sd"]) @ model["w"]


# ---------------------------------------------------------------------------
# Cluster-aware cross-fitting
# ---------------------------------------------------------------------------

def cluster_folds(cluster_ids, n_folds, seed=0):
    """(n,) fold index per ROW, assigned by shuffling the unique clusters
    (base scenarios) round-robin over ``n_folds`` folds."""
    cluster_ids = np.asarray(cluster_ids)
    uniq, cidx = np.unique(cluster_ids, return_inverse=True)
    G = uniq.shape[0]
    k = max(2, min(int(n_folds), G))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(G)
    fold_of_cluster = np.empty(G, dtype=np.int64)
    fold_of_cluster[perm] = np.arange(G) % k
    return fold_of_cluster[cidx], k


def crossfit_scores(X, y, cluster_ids, n_folds=5, ridge=0.1, seed=0):
    """Out-of-fold decision scores; the ridge weight is ``ridge * n_train``
    (features are standardised per training fold, so X'X scales with n).
    A training fold with a single class yields a zero model (chance scores)
    rather than an error."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    n = X.shape[0]
    assert y.shape[0] == n
    folds, k = cluster_folds(cluster_ids, n_folds, seed=seed)
    scores = np.zeros(n, dtype=np.float64)
    for f in range(k):
        test = folds == f
        train = ~test
        if not test.any():
            continue
        if np.unique(y[train]).shape[0] < 2:
            scores[test] = 0.0
            continue
        l2 = float(ridge) * float(train.sum())
        model = ridge_fit(X[train], y[train], l2=l2)
        scores[test] = ridge_decision(model, X[test])
    return scores


# ---------------------------------------------------------------------------
# The leak probe (spec 2.4 instrument 2 / spec 4.4 clause 1(b))
# ---------------------------------------------------------------------------

def leak_probe(X, y, cluster_ids, n_folds=5, ridge=0.1, n_boot=2000,
               conf=0.95, seed=0):
    """Train the probe cross-fitted by base scenario, return the held-out
    AUC with its cluster-bootstrap CI and the leak verdict.

    ``leak`` is True iff the CI EXCLUDES chance (0.5) -- the observation
    carries instruction information a probe can extract.  ``chance_in_ci``
    is the certificate's clause 1(b) requirement.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    cluster_ids = np.asarray(cluster_ids)
    n = X.shape[0]
    assert y.shape[0] == n and cluster_ids.shape[0] == n
    if np.unique(y).shape[0] < 2:
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "conf": float(conf), "n": int(n),
                "n_clusters": int(np.unique(cluster_ids).shape[0]),
                "chance_in_ci": None, "leak": False, "ok": False,
                "note": "single-class labels: probe not runnable"}
    scores = crossfit_scores(X, y, cluster_ids, n_folds=n_folds,
                             ridge=ridge, seed=seed)
    ci = stats.bootstrap_ci_stat(auc_score, [scores, y], cluster_ids,
                                 n_boot=n_boot, conf=conf, seed=seed + 1)
    lo, hi = ci["lo"], ci["hi"]
    usable = not (math.isnan(lo) or math.isnan(hi))
    chance_in = bool(lo <= 0.5 <= hi) if usable else None
    return {"auc": ci["point"], "lo": lo, "hi": hi, "conf": float(conf),
            "n": int(n), "n_clusters": ci["n_clusters"],
            "n_boot": ci["n_boot"], "n_dropped": ci["n_dropped"],
            "chance_in_ci": chance_in,
            "leak": bool(usable and not chance_in), "ok": usable,
            "note": ""}


def planted_leak_control(X, y, cluster_ids, noise=0.3, n_folds=5, ridge=0.1,
                         n_boot=2000, conf=0.95, seed=0):
    """POSITIVE CONTROL: append one label-carrying column
    ``(2y - 1) + N(0, noise)`` to X and re-run the identical pipeline.

    ``detected`` must be True for the audit to be informative: it
    certifies the probe's sensitivity at this n, dimensionality and
    regularisation.  A probe that cannot find a planted leak proves
    nothing by finding nothing (certificate.py maps a failed control to
    UNINFORMATIVE, never PASS).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1).astype(np.int64)
    rng = np.random.default_rng(seed + 7919)
    planted = (2.0 * y - 1.0) + rng.normal(0.0, float(noise), y.shape[0])
    Xp = np.concatenate([X, planted[:, None]], axis=1)
    out = leak_probe(Xp, y, cluster_ids, n_folds=n_folds, ridge=ridge,
                     n_boot=n_boot, conf=conf, seed=seed)
    out["detected"] = bool(out["leak"])
    out["noise"] = float(noise)
    return out


# ---------------------------------------------------------------------------
# Feature builders for the three spec 2.4 probe inputs
# ---------------------------------------------------------------------------

def trajectory_features(positions, rows=None):
    """Flatten recorded positions (T, E, N, 2) -> (n_rows, T*N*2) float64:
    the ``realised_positions_blob`` probe input of spec 2.4(c).  ``rows``
    selects episodes (e.g. the factual lane of a paired record)."""
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()
    positions = np.asarray(positions, dtype=np.float64)
    assert positions.ndim == 4, positions.shape
    T, E = positions.shape[0], positions.shape[1]
    feats = positions.transpose(1, 0, 2, 3).reshape(E, -1)
    if rows is not None:
        rows = np.asarray(rows)
        feats = feats[rows]
    return feats


def matrix_features(X, rows=None):
    """(E, D) observation/state matrix -> float64, optionally row-sliced
    (spec 2.4(a) t=0 observation, 2.4(b) full state)."""
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2, X.shape
    if rows is not None:
        X = X[np.asarray(rows)]
    return X


__all__ = [
    "auc_score", "ridge_fit", "ridge_decision", "cluster_folds",
    "crossfit_scores", "leak_probe", "planted_leak_control",
    "trajectory_features", "matrix_features",
]
