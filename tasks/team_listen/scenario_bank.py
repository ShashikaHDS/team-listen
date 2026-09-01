"""Production scenario-bank loader (M1_SPEC 1.10 / section 7).

``load_bank(path, device=..., arm=...)`` is the loader ``fleet_env.TeamGridEnv
._load_bank`` prefers over its interim fallback.  It returns a
:class:`ScenarioBank` dataclass whose attributes are the 12 tensor fields of
the spec 1.10 schema, resident on ``device``, plus ``meta`` / ``sha256``.

Integrity gates, in order:

1. **Hash gate.**  The file's SHA-256 is recomputed and checked against (a)
   an explicit ``expected_sha`` prefix when given, and (b) the SHA embedded
   in the spec 1.10 filename convention ``scenario_bank_{variant}_{sha}.pt``
   when the basename matches it.  A bank artifact whose bytes drift from its
   recorded hash is refused -- reproducibility statements quote this SHA.
2. **Leaky-bank refusal (spec 4.1 [FIXED: canary]).**  A payload flagged
   ``leaky: True`` (written only by ``scripts/build_leaky_bank.py``) is
   refused for every arm except ``Leaky`` -- the production loader is the
   gate ``tests/test_leaky_bank_refusal.py`` targets.
3. **Field presence + exact dtypes/shapes** against the spec 1.10 table and
   the ``obs_layout`` capacities (R=C=12, MAX_AGENTS=4, MAX_TARGETS=3,
   T_DECISION=128, N_STREAMS=2).
4. **Capacity-padding consistency with obs_layout** (spec 1.3 "the extra
   slots are zeroed"): padded agent slots of ``spawn``/``spawn_alt`` are 0,
   padded target slots are 0 with ``target_valid`` False and an all ``-1``
   ``dist_field`` plane, and valid target slots are contiguous from 0.
5. **Cheap value invariants**: occ/split/leak_bit binary; slip values in
   [0, NO_SLIP]; positions in-grid; spawns and stations on free cells;
   ``dist_field == 0`` exactly at each valid station; delta_gap in [-6, 6];
   mouth either (-1,-1) (RoleBinding) or an in-grid free cell adjacent to
   both stations (Precedence).

Deep geometry checks (BFS re-derivation, compliant-completion simulation)
live in ``tests/test_bank_distfields.py`` / ``test_bank_latch_reachability.py``,
not here -- the loader must stay cheap enough to run at every env
construction.

Import discipline: this module imports ONLY stdlib + torch + obs_layout +
grid_core.  It must NEVER import ``lang_cache``/``harness.templates`` --
the bank is built before any instruction exists, and
``tests/test_import_graph.py`` (spec section 7) enforces the edge.
"""

import dataclasses
import hashlib
import os
import re
from typing import Dict

import torch

from . import obs_layout as L
from .grid_core import NO_SLIP

#: Required tensor keys, in the spec 1.10 table order (must stay equal to
#: ``fleet_env._BANK_KEYS``).
BANK_KEYS = (
    "occ", "spawn", "spawn_alt", "target", "target_valid", "dist_field",
    "mouth", "delta_gap", "leak_bit", "instr_switch_time", "slip", "split",
)

#: Streams of precomputed slip draws (spec 1.10: N_STREAMS = 2).
N_STREAMS = 2

#: Exact dtypes of the spec 1.10 schema.
BANK_DTYPES = {
    "occ": torch.uint8,
    "spawn": torch.int16,
    "spawn_alt": torch.int16,
    "target": torch.int16,
    "target_valid": torch.bool,
    "dist_field": torch.int16,
    "mouth": torch.int16,
    "delta_gap": torch.int8,
    "leak_bit": torch.uint8,
    "instr_switch_time": torch.int16,
    "slip": torch.uint8,
    "split": torch.uint8,
}

_SHA_IN_NAME = re.compile(r"^scenario_bank_.+_([0-9a-f]{8,64})\.pt$")


def file_sha256(path):
    """SHA-256 hex digest of a file's bytes (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_from_filename(path):
    """The SHA prefix embedded in ``scenario_bank_{variant}_{sha}.pt``,
    or '' when the basename does not follow the spec 1.10 convention."""
    match = _SHA_IN_NAME.match(os.path.basename(path))
    return match.group(1) if match else ""


@dataclasses.dataclass
class ScenarioBank:
    """The frozen-physics artifact of spec 1.10, resident on one device."""

    occ: torch.Tensor                  # (K, R, C)          uint8
    spawn: torch.Tensor                # (K, MAX_AGENTS, 2) int16
    spawn_alt: torch.Tensor            # (K, MAX_AGENTS, 2) int16
    target: torch.Tensor               # (K, MAX_TARGETS, 2) int16
    target_valid: torch.Tensor         # (K, MAX_TARGETS)   bool
    dist_field: torch.Tensor           # (K, MAX_TARGETS, R, C) int16
    mouth: torch.Tensor                # (K, 2)             int16
    delta_gap: torch.Tensor            # (K,)               int8
    leak_bit: torch.Tensor             # (K,)               uint8
    instr_switch_time: torch.Tensor    # (K,)               int16 (-1 in M1)
    slip: torch.Tensor                 # (K, N_STREAMS, T, MAX_AGENTS) uint8
    split: torch.Tensor                # (K,)               uint8
    meta: Dict[str, object]            # every non-tensor payload entry
    sha256: str                        # full file SHA-256
    path: str

    @property
    def k(self):
        return int(self.occ.shape[0])


def _fail(path, why):
    raise RuntimeError(
        "scenario bank %s failed integrity checks: %s (M1_SPEC 1.10 schema; "
        "rebuild with scripts/build_scenario_bank.py rather than editing "
        "the artifact)" % (path, why))


def _torch_load(path):
    """torch.load tolerant of the torch>=2.6 weights_only default; the bank
    is our own artifact of tensors + python scalars/containers."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _check_schema(path, payload):
    """Presence, dtype and shape checks against spec 1.10 + obs_layout."""
    missing = [key for key in BANK_KEYS if key not in payload]
    if missing:
        _fail(path, "missing keys %r" % (missing,))
    for key in BANK_KEYS:
        if not torch.is_tensor(payload[key]):
            _fail(path, "field %r is not a tensor" % (key,))
        if payload[key].dtype != BANK_DTYPES[key]:
            _fail(path, "field %r has dtype %s, expected %s"
                  % (key, payload[key].dtype, BANK_DTYPES[key]))
    k = payload["occ"].shape[0]
    if k < 1:
        _fail(path, "empty bank (k=0)")
    expected_shapes = {
        "occ": (k, L.R, L.C),
        "spawn": (k, L.MAX_AGENTS, 2),
        "spawn_alt": (k, L.MAX_AGENTS, 2),
        "target": (k, L.MAX_TARGETS, 2),
        "target_valid": (k, L.MAX_TARGETS),
        "dist_field": (k, L.MAX_TARGETS, L.R, L.C),
        "mouth": (k, 2),
        "delta_gap": (k,),
        "leak_bit": (k,),
        "instr_switch_time": (k,),
        "slip": (k, N_STREAMS, L.T_DECISION, L.MAX_AGENTS),
        "split": (k,),
    }
    for key, shape in expected_shapes.items():
        if tuple(payload[key].shape) != shape:
            _fail(path, "field %r has shape %r, expected %r (capacity "
                        "padding must match obs_layout: MAX_AGENTS=%d, "
                        "MAX_TARGETS=%d, T_DECISION=%d)"
                  % (key, tuple(payload[key].shape), shape,
                     L.MAX_AGENTS, L.MAX_TARGETS, L.T_DECISION))
    return k


def _check_values(path, payload, k):
    """Cheap vectorised value/padding invariants (deep BFS checks are the
    bank tests' job, spec section 7)."""
    occ = payload["occ"].long()
    if not bool(((occ == 0) | (occ == 1)).all()):
        _fail(path, "occ has values outside {0, 1}")
    for key in ("split", "leak_bit"):
        v = payload[key].long()
        if not bool(((v == 0) | (v == 1)).all()):
            _fail(path, "%s has values outside {0, 1}" % key)
    if not bool((payload["slip"].long() <= NO_SLIP).all()):
        _fail(path, "slip has values > NO_SLIP=%d" % NO_SLIP)
    # Certified banks: delta_gap in [-6, 6] (spec 3.1).  Training-only
    # CURRICULUM banks (OPEN(8), DECISIONS.md) carry the SHA-covered
    # "curriculum_near_window" stamp and legitimately exceed it (near-window
    # spawn asymmetry); they keep a geometric sanity cap only.  A bank
    # WITHOUT the stamp is held to the certified range unchanged.
    dg_cap = 6 if payload.get("curriculum_near_window") is None \
        else 2 * (L.R + L.C)
    if not bool((payload["delta_gap"].long().abs() <= dg_cap).all()):
        _fail(path, "delta_gap outside [-%d, %d] (spec 3.1%s)"
              % (dg_cap, dg_cap,
                 "" if dg_cap == 6 else "; curriculum sanity cap"))
    if not bool((payload["instr_switch_time"].long() >= -1).all()):
        _fail(path, "instr_switch_time below -1")

    n_agents = int(payload.get("n_agents", L.N_AGENTS))
    if not 1 <= n_agents <= L.MAX_AGENTS:
        _fail(path, "meta n_agents=%r outside [1, MAX_AGENTS]" % n_agents)

    # live agent slots: in-grid spawns on free cells, distinct per row
    for key in ("spawn", "spawn_alt"):
        pos = payload[key].long()
        live = pos[:, :n_agents]
        if not bool(((live[..., 0] >= 0) & (live[..., 0] < L.R)
                     & (live[..., 1] >= 0) & (live[..., 1] < L.C)).all()):
            _fail(path, "%s live slots out of grid" % key)
        rows = torch.arange(k).unsqueeze(1)
        if not bool((occ[rows, live[..., 0], live[..., 1]] == 0).all()):
            _fail(path, "%s live slots on obstacle cells" % key)
        if n_agents >= 2 and bool(
                (live.unsqueeze(2) == live.unsqueeze(1)).all(-1)
                .triu(1).any()):
            _fail(path, "%s has coincident live agent slots" % key)
        # capacity padding (spec 1.3: extra slots zeroed)
        if not bool((pos[:, n_agents:] == 0).all()):
            _fail(path, "%s padded agent slots are not zeroed "
                        "(obs_layout capacity-padding rule)" % key)

    # target slots: valid ones contiguous from 0, in-grid, free, dist 0
    tv = payload["target_valid"]
    n_valid = tv.long().sum(dim=1)
    if not bool((n_valid >= 2).all()):
        _fail(path, "rows with fewer than 2 valid targets")
    idx = torch.arange(L.MAX_TARGETS).unsqueeze(0)
    if not bool((tv == (idx < n_valid.unsqueeze(1))).all()):
        _fail(path, "valid target slots are not contiguous from slot 0")
    tgt = payload["target"].long()
    in_grid = ((tgt[..., 0] >= 0) & (tgt[..., 0] < L.R)
               & (tgt[..., 1] >= 0) & (tgt[..., 1] < L.C))
    if not bool(in_grid[tv].all()):
        _fail(path, "valid target slots out of grid")
    rows = torch.arange(k).unsqueeze(1)
    if not bool((occ[rows, tgt[..., 0], tgt[..., 1]][tv] == 0).all()):
        _fail(path, "valid target slots on obstacle cells")
    if not bool((tgt[~tv] == 0).all()):
        _fail(path, "padded target slots are not zeroed")

    df = payload["dist_field"].long()
    flat = df.reshape(k, L.MAX_TARGETS, L.R * L.C)
    cell = (tgt[..., 0] * L.C + tgt[..., 1]).clamp(0, L.R * L.C - 1)
    at_target = flat.gather(2, cell.unsqueeze(-1)).squeeze(-1)
    if not bool((at_target[tv] == 0).all()):
        _fail(path, "dist_field != 0 at its own valid target cell")
    if not bool((df[~tv] == -1).all()):
        _fail(path, "dist_field planes of padded target slots are not "
                    "all -1")
    if not bool((df >= -1).all()):
        _fail(path, "dist_field has values below -1")

    # mouth: (-1,-1) for RoleBinding rows; an in-grid free cell adjacent to
    # both stations for Precedence rows (spec 1.10 table / 3.1)
    mouth = payload["mouth"].long()
    is_rb = (mouth == -1).all(dim=1)
    is_pr = ~is_rb
    if bool(is_pr.any()):
        pm = mouth[is_pr]
        if not bool(((pm[:, 0] >= 0) & (pm[:, 0] < L.R)
                     & (pm[:, 1] >= 0) & (pm[:, 1] < L.C)).all()):
            _fail(path, "precedence mouth cells out of grid")
        if not bool((occ[is_pr.nonzero(as_tuple=False).reshape(-1),
                         pm[:, 0], pm[:, 1]] == 0).all()):
            _fail(path, "precedence mouth on an obstacle cell")
        # both valid stations are 4-adjacent to the mouth (unique-mouth
        # airlock, spec 3.1)
        l1 = (tgt[is_pr] - pm.unsqueeze(1)).abs().sum(-1)
        if not bool((l1[tv[is_pr]] == 1).all()):
            _fail(path, "precedence stations not adjacent to the mouth")
    variant = payload.get("variant", None)
    if variant == "RoleBinding" and bool(is_pr.any()):
        _fail(path, "variant=RoleBinding but %d rows carry a mouth cell"
              % int(is_pr.sum()))
    if variant == "Precedence" and bool(is_rb.any()):
        _fail(path, "variant=Precedence but %d rows have mouth=(-1,-1)"
              % int(is_rb.sum()))


def load_bank(path, device="cpu", arm=None, expected_sha=""):
    """Load, verify and device-place a scenario bank (spec 1.10 / 4.1).

    Args:
        path:         the ``scenario_bank_{variant}_{sha}.pt`` artifact.
        device:       target device for GPU residency (~35 MB at K=16384).
        arm:          the requesting arm name; every arm except ``"Leaky"``
                      refuses a leaky bank (spec 4.1 [FIXED: canary]).
        expected_sha: optional expected SHA-256 (prefix ok), e.g. from
                      ``cfg.bank_sha`` or a run manifest.

    Returns:
        :class:`ScenarioBank`.

    Raises:
        RuntimeError: on any failed gate (hash, leaky refusal, schema,
        padding, value invariants).
    """
    if not os.path.isfile(path):
        raise RuntimeError(
            "scenario bank not found: %s (build it with "
            "scripts/build_scenario_bank.py, M1_SPEC 1.10)" % (path,))

    sha = file_sha256(path)
    if expected_sha and not sha.startswith(expected_sha.lower()):
        _fail(path, "SHA mismatch: expected %s..., file hashes to %s"
              % (expected_sha, sha))
    name_sha = sha_from_filename(path)
    if name_sha and not sha.startswith(name_sha):
        _fail(path, "filename claims SHA %s... but the bytes hash to %s "
                    "(artifact edited or truncated after build)"
              % (name_sha, sha))

    payload = _torch_load(path)
    if not isinstance(payload, dict):
        _fail(path, "payload is %r, expected a flat dict"
              % (type(payload).__name__,))

    # Leaky-bank refusal BEFORE any use (spec 4.1): only the Leaky arm may
    # consume an instruction-geometry-coupled bank.
    if bool(payload.get("leaky", False)) and arm != "Leaky":
        raise RuntimeError(
            "bank %s is a LEAKY bank (scripts/build_leaky_bank.py); it is "
            "refused for arm %r -- only the Leaky arm may load it "
            "(M1_SPEC 4.1)" % (path, arm))

    k = _check_schema(path, payload)
    _check_values(path, payload, k)

    tensors = {key: payload[key].to(device) for key in BANK_KEYS}
    meta = {key: value for key, value in payload.items()
            if not torch.is_tensor(value)}
    return ScenarioBank(meta=meta, sha256=sha, path=str(path), **tensors)


__all__ = [
    "BANK_KEYS", "BANK_DTYPES", "N_STREAMS", "ScenarioBank",
    "load_bank", "file_sha256", "sha_from_filename",
]
