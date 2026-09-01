"""Curriculum phase banks (DECISIONS 2026-09-01): window respected,
naming does not clobber the certified bank, builds are deterministic.

Run: python tests/test_curriculum_banks.py
"""
import os
import sys
import json
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import torch  # noqa: E402
import build_scenario_bank as bsb  # noqa: E402


K = 16  # smoke size


def _build(tmp, lo, hi, tag, seed=4901):
    rc = bsb.main(["--variant", "RoleBinding", "--num-scenarios", str(K),
                   "--seed", str(seed), "--near-window", "%d,%d" % (lo, hi),
                   "--tag", tag, "--out-dir", tmp, "--quiet"])
    assert rc == 0
    manifest = json.load(open(os.path.join(
        tmp, "scenario_bank_RoleBinding_%s.json" % tag)))
    payload_path = os.path.join(tmp, manifest["file"])
    return manifest, payload_path


def test_window_recorded_and_respected():
    with tempfile.TemporaryDirectory() as tmp:
        man, pt = _build(tmp, 1, 3, "phase1")
        assert man["constraints"]["rb_near_window"] == [1, 3], man[
            "constraints"]["rb_near_window"]
        assert man["tag"] == "phase1"
        bank = torch.load(pt, weights_only=True)
        occ, spawn, target = bank["occ"], bank["spawn"], bank["target"]
        for i in range(K):
            grid = occ[i].tolist()
            # latch-aware fields exactly as the builder computes them
            st0 = tuple(int(v) for v in target[i, 0])
            st1 = tuple(int(v) for v in target[i, 1])
            d0 = bsb._bfs(grid, st0, blocked=(st1,))
            d1 = bsb._bfs(grid, st1, blocked=(st0,))
            near = []
            for a in range(2):
                r, c = int(spawn[i, a, 0]), int(spawn[i, a, 1])
                da, db = d0[r][c], d1[r][c]
                assert da != db, (i, a, "tie should be excluded")
                assert 1 <= min(da, db) <= 3, (i, a, da, db)
                assert max(da, db) <= 14, (i, a, da, db)
                near.append(da < db)
            # the two robots spawn nearest DIFFERENT stations
            assert near[0] != near[1], (i, near)


def test_default_window_unchanged():
    # no args -> the certified window and the tagless legacy naming
    with tempfile.TemporaryDirectory() as tmp:
        rc = bsb.main(["--variant", "RoleBinding", "--num-scenarios", str(K),
                       "--seed", "4902", "--out-dir", tmp, "--quiet"])
        assert rc == 0
        man = json.load(open(os.path.join(
            tmp, "scenario_bank_RoleBinding.json")))
        assert man["constraints"]["rb_spawn_dist_window"] == [4, 14]
        assert man["tag"] == ""
        assert man["file"].startswith("scenario_bank_RoleBinding_")


def test_no_clobber_between_tags():
    with tempfile.TemporaryDirectory() as tmp:
        man1, pt1 = _build(tmp, 1, 3, "phase1")
        man2, pt2 = _build(tmp, 3, 6, "phase2", seed=4903)
        assert os.path.exists(pt1) and os.path.exists(pt2)
        assert man1["file"] != man2["file"]
        # both manifests coexist
        assert os.path.exists(os.path.join(
            tmp, "scenario_bank_RoleBinding_phase1.json"))
        assert os.path.exists(os.path.join(
            tmp, "scenario_bank_RoleBinding_phase2.json"))


def test_deterministic_rebuild():
    with tempfile.TemporaryDirectory() as t1, \
         tempfile.TemporaryDirectory() as t2:
        man1, _ = _build(t1, 1, 3, "phase1")
        man2, _ = _build(t2, 1, 3, "phase1")
        assert man1["sha256"] == man2["sha256"]


def test_bad_window_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        for bad_args in (
            ["--near-window", "9,3", "--tag", "x"],   # lo > hi
            ["--near-window", "1,3"],                  # missing --tag
            ["--dist-min", "9", "--dist-max", "3"],    # certified lo > hi
        ):
            try:
                bsb.main(["--variant", "RoleBinding", "--num-scenarios",
                          "4", "--seed", "1", "--out-dir", tmp,
                          "--quiet"] + bad_args)
                raise AssertionError("expected argparse error: %s" % bad_args)
            except SystemExit as e:
                assert e.code != 0


def test_curriculum_stamp_and_loader_roundtrip():
    """5090 integration fix: the production loader refused curriculum banks
    (delta_gap outside the certified [-6, 6], spec 3.1).  Curriculum banks
    now carry the SHA-covered "curriculum_near_window" payload stamp and
    load with a relaxed geometric sanity cap; an UNSTAMPED bank with the
    same wide delta_gap is still refused, and the stamp never appears on
    the certified path (byte-identity preserved)."""
    from tasks.team_listen import scenario_bank

    with tempfile.TemporaryDirectory() as tmp:
        _, path = _build(tmp, 1, 3, "stamproundtrip")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload.get("curriculum_near_window") == [1, 3]
        assert int(payload["delta_gap"].long().abs().max()) > 6, \
            "smoke bank unexpectedly within the certified range; widen K"
        bank = scenario_bank.load_bank(path)          # must NOT refuse
        assert bank.meta["curriculum_near_window"] == [1, 3]

        # strip the stamp -> the certified gate must refuse the same data
        del payload["curriculum_near_window"]
        stripped = os.path.join(tmp, "scenario_bank_RoleBinding_00.pt")
        torch.save(payload, stripped)
        try:
            scenario_bank.load_bank(stripped)
        except RuntimeError as exc:
            assert "delta_gap" in str(exc)
        else:
            raise AssertionError("unstamped wide-delta_gap bank loaded")

        # certified builder path stays stamp-free
        rc = bsb.main(["--variant", "RoleBinding", "--num-scenarios",
                       str(K), "--seed", "77", "--out-dir", tmp, "--quiet"])
        assert rc == 0
        cert = [f for f in os.listdir(tmp)
                if f.startswith("scenario_bank_RoleBinding_")
                and f.endswith(".pt") and "stamproundtrip" not in f
                and f != "scenario_bank_RoleBinding_00.pt"]
        cp = torch.load(os.path.join(tmp, cert[0]), map_location="cpu",
                        weights_only=False)
        assert "curriculum_near_window" not in cp


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %r" % (fn.__name__, e))
    print("%d passed, %d failed, 0 skipped" % (passed, failed))
    sys.exit(1 if failed else 0)
