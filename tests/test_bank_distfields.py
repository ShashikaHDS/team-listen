"""Bank distance-field tests: BFS fields consistent with occ.

M1_SPEC coverage (sections 1.10, 2.1, 3.1; spec section 7 test list):

* ``dist_field[k, j]`` is the LATCH-AWARE BFS from target j over free
  cells with every OTHER valid target cell treated as an obstacle
  (spec 1.10 [FIXED: latch-aware distance field]).  Verified here by an
  INDEPENDENT BFS implementation (this file deliberately re-implements
  BFS instead of importing the builder's), cell-for-cell.
* Alcove-topology invariant (the spec's "cheap topology invariant"):
  because every station is a degree-1 leaf, the latch-aware field equals
  the all-free field at EVERY cell except the other valid station cells
  themselves, where the latch-aware field holds -1 (the blocked leaf).
* Consistency with occ: obstacle cells hold -1; the field is 0 exactly at
  its own station; every free cell other than the other station is
  reachable (free space is 4-connected THROUGH the latch-aware mask);
  padded target slots hold an all -1 plane.
* delta_gap consistency: Precedence rows store d(r0, m) - d(r1, m) of the
  stored mouth (recomputed by BFS); RoleBinding rows store the signed
  matching-cost asymmetry, capped to +-1 (spec 2.1 [FIXED]).

The smoke bank (32 scenarios per variant) is built in memory by
``scripts/build_scenario_bank.py`` -- never written into data/.

pytest-compatible; also standalone: ``python tests/test_bank_distfields.py``.
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import obs_layout as L               # noqa: E402

R, C = L.R, L.C
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

SMOKE_K = 32
SMOKE_SEED = 20260831

_BSB = None
_BANKS = {}


def _builder():
    global _BSB
    if _BSB is None:
        path = REPO_ROOT / "scripts" / "build_scenario_bank.py"
        spec = importlib.util.spec_from_file_location(
            "team_listen_build_scenario_bank_df", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BSB = mod
    return _BSB


def _bank(variant):
    if variant not in _BANKS:
        bsb = _builder()
        seeds = bsb.derive_seed_list(SMOKE_SEED, SMOKE_K)
        _BANKS[variant] = bsb.build_bank(variant, SMOKE_K, seeds)
    return _BANKS[variant]


# ---------------------------------------------------------------------------
# Independent BFS (differential against the builder's implementation)
# ---------------------------------------------------------------------------

def _bfs_ref(grid, src, blocked=()):
    """Reference BFS: -1 sentinel, 4-connected, extra ``blocked`` cells
    treated as obstacles.  Independent of the builder's `_bfs`."""
    dist = [[-1] * C for _ in range(R)]
    blocked = set(blocked)
    sr, sc = src
    if grid[sr][sc] != 0 or (sr, sc) in blocked:
        return dist
    dist[sr][sc] = 0
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0 \
                    and (nr, nc) not in blocked and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                queue.append((nr, nc))
    return dist


def _row(payload, k):
    grid = payload["occ"][k].long().tolist()
    valid = payload["target_valid"][k]
    stations = [tuple(payload["target"][k, j].tolist())
                for j in range(L.MAX_TARGETS) if bool(valid[j])]
    return grid, stations


def _stored_field(payload, k, j):
    return payload["dist_field"][k, j].long().tolist()


# ---------------------------------------------------------------------------
# Tests (each runs over both variants -- the field semantics are shared)
# ---------------------------------------------------------------------------

def test_dist_fields_match_independent_latch_aware_bfs():
    for variant in ("RoleBinding", "Precedence"):
        payload = _bank(variant)
        for k in range(SMOKE_K):
            grid, stations = _row(payload, k)
            for j, st in enumerate(stations):
                others = [s for i, s in enumerate(stations) if i != j]
                expect = _bfs_ref(grid, st, blocked=others)
                got = _stored_field(payload, k, j)
                assert got == expect, \
                    "%s row %d slot %d: stored dist_field != independent " \
                    "latch-aware BFS (spec 1.10)" % (variant, k, j)


def test_latch_aware_equals_all_free_except_other_station():
    # The spec 1.10 "cheap topology invariant": with degree-1 alcove
    # stations the latch-aware and all-free fields coincide -- the ONLY
    # cells where they may differ are the other valid station cells,
    # where latch-aware is -1 (blocked leaf) and all-free is finite.
    for variant in ("RoleBinding", "Precedence"):
        payload = _bank(variant)
        for k in range(SMOKE_K):
            grid, stations = _row(payload, k)
            for j, st in enumerate(stations):
                others = {s for i, s in enumerate(stations) if i != j}
                aware = _stored_field(payload, k, j)
                free = _bfs_ref(grid, st)
                for r in range(R):
                    for c in range(C):
                        if (r, c) in others:
                            assert aware[r][c] == -1, \
                                "%s row %d: latch-aware field not blocked " \
                                "at the other station" % (variant, k)
                            assert free[r][c] >= 0, \
                                "%s row %d: other station unreachable in " \
                                "the all-free field" % (variant, k)
                        else:
                            assert aware[r][c] == free[r][c], \
                                "%s row %d slot %d cell %r: latch-aware " \
                                "%d != all-free %d -- alcove topology " \
                                "broken (spec 1.10/2.1)" \
                                % (variant, k, j, (r, c),
                                   aware[r][c], free[r][c])


def test_fields_consistent_with_occ():
    for variant in ("RoleBinding", "Precedence"):
        payload = _bank(variant)
        for k in range(SMOKE_K):
            grid, stations = _row(payload, k)
            for j, st in enumerate(stations):
                others = {s for i, s in enumerate(stations) if i != j}
                field = _stored_field(payload, k, j)
                assert field[st[0]][st[1]] == 0, \
                    "%s row %d: dist != 0 at its own station" % (variant, k)
                for r in range(R):
                    for c in range(C):
                        if grid[r][c] == 1:
                            assert field[r][c] == -1, \
                                "%s row %d: finite dist on an obstacle " \
                                "cell %r" % (variant, k, (r, c))
                        elif (r, c) in others:
                            assert field[r][c] == -1
                        else:
                            # every free cell is reachable through the
                            # latch-aware mask: no spawn or shaping state
                            # can ever see an infinite distance (spec 2.1)
                            assert field[r][c] >= 0, \
                                "%s row %d: free cell %r unreachable in " \
                                "the latch-aware field" % (variant, k, (r, c))


def test_padded_target_slots_are_all_minus_one():
    for variant in ("RoleBinding", "Precedence"):
        payload = _bank(variant)
        valid = payload["target_valid"]
        assert bool((payload["dist_field"][~valid] == -1).all()), \
            "%s: padded dist_field planes must be all -1 (capacity " \
            "padding, spec 1.3/1.10)" % variant
        assert bool((payload["target"][~valid] == 0).all()), \
            "%s: padded target slots must be zeroed" % variant
        assert bool(valid[:, :2].all()) and not bool(valid[:, 2:].any()), \
            "%s: M1 rows carry exactly target slots 0 and 1" % variant


def test_pr_delta_gap_matches_mouth_bfs():
    payload = _bank("Precedence")
    for k in range(SMOKE_K):
        grid, stations = _row(payload, k)
        mouth = tuple(payload["mouth"][k].tolist())
        dm = _bfs_ref(grid, mouth, blocked=stations)
        d = [dm[tuple(payload["spawn"][k, a].tolist())[0]]
             [tuple(payload["spawn"][k, a].tolist())[1]]
             for a in range(L.N_AGENTS)]
        assert d[0] >= 3 and d[1] >= 3, \
            "row %d spawn-to-mouth distance below 3 (spec 3.1)" % k
        assert d[0] - d[1] == int(payload["delta_gap"][k]), \
            "row %d delta_gap %d != recomputed d(r0,m)-d(r1,m) = %d" \
            % (k, int(payload["delta_gap"][k]), d[0] - d[1])


def test_rb_delta_gap_is_capped_matching_asymmetry():
    payload = _bank("RoleBinding")
    for k in range(SMOKE_K):
        grid, stations = _row(payload, k)
        left = min(stations, key=lambda s: s[1])
        right = max(stations, key=lambda s: s[1])
        d_left = _bfs_ref(grid, left, blocked=(right,))
        d_right = _bfs_ref(grid, right, blocked=(left,))
        s0 = tuple(payload["spawn"][k, 0].tolist())
        s1 = tuple(payload["spawn"][k, 1].tolist())
        asym = ((d_left[s0[0]][s0[1]] + d_right[s1[0]][s1[1]])
                - (d_right[s0[0]][s0[1]] + d_left[s1[0]][s1[1]]))
        assert asym == int(payload["delta_gap"][k]), \
            "row %d delta_gap %d != recomputed matching asymmetry %d" \
            % (k, int(payload["delta_gap"][k]), asym)
        assert abs(asym) <= 1, \
            "row %d asymmetry beyond the +-1 per-scenario cap (spec 2.1)" % k


def test_leak_bit_matches_geometric_default():
    # leak_bit is the sign of the geometric default in instruction-class
    # index space (spec 1.10/4.1); ties may hold either value (the builder
    # alternates them for balance).
    for variant in ("RoleBinding", "Precedence"):
        payload = _bank(variant)
        for k in range(SMOKE_K):
            bit = int(payload["leak_bit"][k])
            assert bit in (0, 1)
            gap = int(payload["delta_gap"][k])
            if variant == "Precedence":
                # Delta < 0: robot_0 closer to the mouth -> PR0 default
                if gap < 0:
                    assert bit == 0, "row %d leak_bit != PR0 default" % k
                elif gap > 0:
                    assert bit == 1, "row %d leak_bit != PR1 default" % k
            else:
                # asym < 0: matching prefers r0->left -> RB0 default
                if gap < 0:
                    assert bit == 0, "row %d leak_bit != RB0 default" % k
                elif gap > 0:
                    assert bit == 1, "row %d leak_bit != RB1 default" % k


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
