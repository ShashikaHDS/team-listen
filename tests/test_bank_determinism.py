"""Bank determinism tests: same seeds -> bit-identical bank.

M1_SPEC coverage (sections 1.10, 4.1; spec section 7 test list):

* The builder is DETERMINISTIC FROM AN EXPLICIT SEED LIST: rebuilding with
  the same (variant, k, seed_list, epsilon, eval_frac) yields tensor-level
  bit-identical payloads, byte-identical serialized .pt files, and
  therefore a stable SHA-256 -- the "bank SHA stable across rebuilds"
  property the run manifests and the reproducibility statement quote.
* The per-row seed derivation rule (sha256 mix, recorded in the manifest)
  is pinned against golden values so a silent edit to the rule -- which
  would change every future rebuild's SHA -- fails here first.
* The JSON manifest is deterministic and its recorded SHA matches the
  written artifact.
* Loader round-trip (tasks/team_listen/scenario_bank.py): the production
  loader accepts a freshly built bank, returns bit-identical tensors, and
  REFUSES (a) a byte-corrupted artifact whose filename carries the build
  SHA (hash gate), (b) a schema-incomplete payload (field presence), and
  (c) a ``leaky: True`` bank for any arm but ``Leaky`` (spec 4.1
  [FIXED: canary]).

The smoke banks (32 scenarios per variant) are built by
``scripts/build_scenario_bank.py`` and written ONLY to a temp dir --
never into data/.

pytest-compatible; also standalone: ``python tests/test_bank_determinism.py``.
"""

import atexit
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import scenario_bank as sb           # noqa: E402

SMOKE_K = 32
SMOKE_SEED = 20260831

#: Golden per-row seeds for master seed 20260831 (guards the derivation
#: rule: changing it silently changes every rebuild's SHA).
GOLDEN_SEEDS_3 = [9069733118729195716, 5650956655230199767,
                  5562322761040507985]

_BSB = None
_BUILDS = {}
_TMP = None


def _builder():
    global _BSB
    if _BSB is None:
        path = REPO_ROOT / "scripts" / "build_scenario_bank.py"
        spec = importlib.util.spec_from_file_location(
            "team_listen_build_scenario_bank_det", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BSB = mod
    return _BSB


def _tmpdir():
    """Session temp dir (NEVER data/); best-effort cleanup at exit."""
    global _TMP
    if _TMP is None:
        _TMP = tempfile.mkdtemp(prefix="team_listen_smoke_bank_")
        atexit.register(shutil.rmtree, _TMP, True)
    return _TMP


def _build(variant, tag, seed=SMOKE_SEED, k=SMOKE_K):
    """Cached builds; distinct ``tag`` forces an independent rebuild."""
    key = (variant, tag, seed, k)
    if key not in _BUILDS:
        bsb = _builder()
        seeds = bsb.derive_seed_list(seed, k)
        _BUILDS[key] = bsb.build_bank(variant, k, seeds)
    return _BUILDS[key]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Determinism of the build itself
# ---------------------------------------------------------------------------

def test_seed_derivation_pinned():
    bsb = _builder()
    assert bsb.derive_seed_list(SMOKE_SEED, 3) == GOLDEN_SEEDS_3, \
        "derive_seed_list changed -- every rebuild SHA changes with it " \
        "(spec 1.10 determinism; update the golden ONLY with a spec-level " \
        "decision)"
    # prefix property: the k-list is a prefix of any longer list
    assert bsb.derive_seed_list(SMOKE_SEED, 2) == GOLDEN_SEEDS_3[:2]


def test_same_seeds_bitwise_identical_payload():
    bsb = _builder()
    for variant in ("RoleBinding", "Precedence"):
        a = _build(variant, "a")
        b = _build(variant, "b")           # independent rebuild, same seeds
        keys_a = sorted(a.keys())
        assert keys_a == sorted(b.keys())
        for key in keys_a:
            if torch.is_tensor(a[key]):
                assert a[key].dtype == b[key].dtype, key
                assert torch.equal(a[key], b[key]), \
                    "%s: tensor %r differs across identical-seed rebuilds " \
                    "(spec 1.10 determinism broken)" % (variant, key)
            else:
                assert a[key] == b[key], \
                    "%s: metadata %r differs across rebuilds" % (variant, key)


def test_serialized_bytes_and_sha_stable():
    bsb = _builder()
    for variant in ("RoleBinding", "Precedence"):
        data_a, sha_a = bsb.serialize_bank(_build(variant, "a"))
        data_b, sha_b = bsb.serialize_bank(_build(variant, "b"))
        assert data_a == data_b, \
            "%s: .pt bytes differ across identical-seed rebuilds" % variant
        assert sha_a == sha_b


def test_saved_files_and_manifest_deterministic():
    bsb = _builder()
    variant = "RoleBinding"
    seed_info = {"mode": "derived", "master_seed": SMOKE_SEED}
    out_a = os.path.join(_tmpdir(), "det_a")
    out_b = os.path.join(_tmpdir(), "det_b")
    pt_a, mf_a, sha_a = bsb.save_bank(_build(variant, "a"), out_a, seed_info)
    pt_b, mf_b, sha_b = bsb.save_bank(_build(variant, "b"), out_b, seed_info)
    assert sha_a == sha_b
    assert os.path.basename(pt_a) == os.path.basename(pt_b), \
        "sha-suffixed artifact names must be rebuild-stable (spec 1.10 " \
        "data/scenario_bank_{variant}_{sha}.pt)"
    assert _sha256_file(pt_a) == _sha256_file(pt_b) == sha_a
    with open(mf_a, "rb") as f:
        bytes_a = f.read()
    with open(mf_b, "rb") as f:
        bytes_b = f.read()
    assert bytes_a == bytes_b, "manifest JSON not byte-deterministic"
    manifest = json.loads(bytes_a.decode("utf-8"))
    assert manifest["sha256"] == sha_a
    assert manifest["file"] == os.path.basename(pt_a)
    assert manifest["variant"] == variant
    assert manifest["k"] == SMOKE_K
    assert manifest["leaky"] is False
    stats = manifest["stats"]
    payload = _build(variant, "a")
    assert stats["n_train"] == int((payload["split"] == 0).sum())
    assert stats["n_eval"] == int((payload["split"] == 1).sum())
    assert stats["n_train"] + stats["n_eval"] == SMOKE_K


def test_explicit_seed_list_reproduces_derived_build():
    # The manifest records the seed list/derivation; feeding the derived
    # list back explicitly must reproduce the bank bit-identically.
    bsb = _builder()
    seeds = bsb.derive_seed_list(SMOKE_SEED, SMOKE_K)
    explicit = bsb.build_bank("Precedence", SMOKE_K, list(seeds))
    derived = _build("Precedence", "a")
    for key in sb.BANK_KEYS:
        assert torch.equal(explicit[key], derived[key]), key


def test_different_seeds_change_the_bank():
    bsb = _builder()
    a = _build("RoleBinding", "a")
    c = _build("RoleBinding", "c", seed=SMOKE_SEED + 1)
    _, sha_a = bsb.serialize_bank(a)
    _, sha_c = bsb.serialize_bank(c)
    assert sha_a != sha_c, "different master seeds produced identical banks"
    changed = any(not torch.equal(a[key], c[key]) for key in sb.BANK_KEYS)
    assert changed, "different master seeds left every tensor unchanged"


# ---------------------------------------------------------------------------
# Loader round-trip and integrity gates (tasks/team_listen/scenario_bank.py)
# ---------------------------------------------------------------------------

def test_loader_roundtrip_bit_identical():
    bsb = _builder()
    payload = _build("Precedence", "a")
    out = os.path.join(_tmpdir(), "roundtrip")
    pt_path, _, sha = bsb.save_bank(payload, out,
                                    {"mode": "derived",
                                     "master_seed": SMOKE_SEED})
    bank = sb.load_bank(pt_path, device="cpu", arm="Blind")
    assert bank.sha256 == sha
    assert bank.k == SMOKE_K
    for key in sb.BANK_KEYS:
        assert getattr(bank, key).dtype == sb.BANK_DTYPES[key], key
        assert torch.equal(getattr(bank, key), payload[key]), \
            "loader returned a different %r than was built" % (key,)
    assert bank.meta["variant"] == "Precedence"
    assert bank.meta["leaky"] is False
    assert bank.meta["epsilon"] == payload["epsilon"]
    # M1: the instruction-switch branch is inert (spec 1.5 [FIXED])
    assert bool((bank.instr_switch_time == -1).all())


def test_loader_refuses_corrupted_bytes():
    bsb = _builder()
    out = os.path.join(_tmpdir(), "corrupt")
    pt_path, _, _ = bsb.save_bank(_build("RoleBinding", "a"), out,
                                  {"mode": "derived",
                                   "master_seed": SMOKE_SEED})
    with open(pt_path, "rb") as f:
        data = bytearray(f.read())
    data[len(data) // 2] ^= 0xFF
    bad_dir = os.path.join(_tmpdir(), "corrupt_bad")
    os.makedirs(bad_dir, exist_ok=True)
    bad_path = os.path.join(bad_dir, os.path.basename(pt_path))
    with open(bad_path, "wb") as f:
        f.write(bytes(data))
    try:
        sb.load_bank(bad_path, arm="Blind")
    except RuntimeError as exc:
        assert "SHA" in str(exc) or "hash" in str(exc).lower()
    else:
        raise AssertionError("corrupted bank accepted -- the spec 1.10 "
                             "hash gate is not enforced")


def test_loader_refuses_missing_fields():
    payload = dict(_build("RoleBinding", "a"))
    payload.pop("slip")                       # break the spec 1.10 schema
    path = os.path.join(_tmpdir(), "missing_slip.pt")
    torch.save(payload, path)
    try:
        sb.load_bank(path, arm="Blind")
    except RuntimeError as exc:
        assert "slip" in str(exc)
    else:
        raise AssertionError("schema-incomplete bank accepted")


def test_loader_refuses_leaky_bank_for_non_leaky_arms():
    # spec 4.1 [FIXED: canary]: leaky banks carry a distinct flag/SHA and
    # the production loader refuses them for every arm but Leaky.
    payload = dict(_build("RoleBinding", "a"))
    payload["leaky"] = True
    payload["leak_rho"] = 0.8
    path = os.path.join(_tmpdir(), "leaky_probe.pt")
    torch.save(payload, path)
    for arm in ("Blind", "Lang", "SymbolPO", "Mute", None):
        try:
            sb.load_bank(path, arm=arm)
        except RuntimeError as exc:
            assert "LEAKY" in str(exc).upper()
        else:
            raise AssertionError(
                "leaky bank accepted for arm %r (spec 4.1 refusal)" % (arm,))
    bank = sb.load_bank(path, arm="Leaky")
    assert bank.meta["leaky"] is True


def test_smoke_bank_padding_matches_obs_layout_capacities():
    # capacity-padding consistency (spec 1.3): fixed-capacity slots so
    # OBS_DIM/STATE_DIM never change; extra slots zeroed.
    from tasks.team_listen import obs_layout as L
    for variant in ("RoleBinding", "Precedence"):
        payload = _build(variant, "a")
        assert payload["spawn"].shape == (SMOKE_K, L.MAX_AGENTS, 2)
        assert payload["slip"].shape == (SMOKE_K, 2, L.T_DECISION,
                                         L.MAX_AGENTS)
        assert bool((payload["spawn"][:, L.N_AGENTS:] == 0).all())
        assert bool((payload["spawn_alt"][:, L.N_AGENTS:] == 0).all())
        assert bool((payload["target"][:, 2:] == 0).all())
        assert not bool(payload["target_valid"][:, 2:].any())


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  " + name)
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("{}/{} tests passed".format(len(tests) - n_fail, len(tests)))
    sys.exit(1 if n_fail else 0)
