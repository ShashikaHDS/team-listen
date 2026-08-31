"""Bank latch-reachability tests: no precedence deadlock, no role trap.

M1_SPEC coverage (sections 2.1, 3.1, 1.10; spec section 7 test list):

* RoleBinding: both stations are DEGREE-1 ALCOVES (leaf cells whose only
  free neighbour is their mouth), mouths >= 5 columns apart on opposite
  halves; spawns sit in the latch-aware [4, 14] distance window with the
  per-scenario +-1 asymmetry cap [FIXED: aggregate-only]; and a COMPLIANT
  COMPLETION EXISTS FOR BOTH ROLE ASSIGNMENTS -- verified by actually
  simulating a serialised compliant plan through ``grid_core`` (the real
  conflict/latch kernels), for spawn AND spawn_alt.
* Precedence: unique-mouth airlock -- the two stations are the ONLY other
  free neighbours of the mouth and each is a leaf, the width-1 corridor
  has length >= 3, spawns are outside the corridor with d(spawn, m) >= 3
  [FIXED: the draft constrained only Delta]; BOTH orderings are always
  feasible and every simulated completion has |dt| >= 1 (ties are
  structurally impossible, so G = 1 needs no learned delay skill).

The simulation is the serialised compliant plan the builder certifies:
the mover follows a BFS path that avoids the stationary robot's cell and
the other station (transiting a station would latch, spec 1.11); the
second robot starts only after the first has latched.  Every step asserts
zero obstacle/robot collision flags from ``grid_core.step_positions``.

The smoke bank (32 scenarios per variant) is built in memory by
``scripts/build_scenario_bank.py`` -- never written into data/.

pytest-compatible; also standalone: ``python tests/test_bank_latch_reachability.py``.
"""

import importlib.util
import sys
from collections import deque
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import obs_layout as L               # noqa: E402
from tasks.team_listen.grid_core import STAY, step_positions  # noqa: E402

R, C = L.R, L.C
T_DECISION = L.T_DECISION
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_OF = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3, (0, 0): 4}

SMOKE_K = 32
SMOKE_SEED = 20260831

_BSB = None
_BANKS = {}


def _builder():
    """Load scripts/build_scenario_bank.py as a module (scripts/ is not a
    package; importlib keeps this robust under pytest and standalone)."""
    global _BSB
    if _BSB is None:
        path = REPO_ROOT / "scripts" / "build_scenario_bank.py"
        spec = importlib.util.spec_from_file_location(
            "team_listen_build_scenario_bank", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BSB = mod
    return _BSB


def _bank(variant):
    """Deterministic 32-row smoke bank per variant, built once, in memory."""
    if variant not in _BANKS:
        bsb = _builder()
        seeds = bsb.derive_seed_list(SMOKE_SEED, SMOKE_K)
        _BANKS[variant] = bsb.build_bank(variant, SMOKE_K, seeds)
    return _BANKS[variant]


# ---------------------------------------------------------------------------
# Independent grid helpers (deliberately NOT imported from the builder)
# ---------------------------------------------------------------------------

def _grid_of(payload, k):
    return payload["occ"][k].long().tolist()


def _in_grid(r, c):
    return 0 <= r < R and 0 <= c < C


def _free_nbrs(grid, cell):
    out = []
    for dr, dc in DIRS4:
        nr, nc = cell[0] + dr, cell[1] + dc
        if _in_grid(nr, nc) and grid[nr][nc] == 0:
            out.append((nr, nc))
    return out


def _bfs(grid, src, blocked=()):
    dist = [[-1] * C for _ in range(R)]
    blocked = set(blocked)
    if grid[src[0]][src[1]] != 0 or src in blocked:
        return dist
    dist[src[0]][src[1]] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (_in_grid(nr, nc) and grid[nr][nc] == 0
                    and (nr, nc) not in blocked and dist[nr][nc] < 0):
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist


def _path(grid, src, dst, blocked=()):
    """A shortest src->dst path avoiding ``blocked``, as a cell list, or
    None.  Greedy descent on the BFS-from-dst field (deterministic
    tie-break: DIRS4 order)."""
    dist = _bfs(grid, dst, blocked=blocked)
    if dist[src[0]][src[1]] < 0:
        return None
    path = [src]
    cur = src
    while cur != dst:
        d = dist[cur[0]][cur[1]]
        for dr, dc in DIRS4:
            nr, nc = cur[0] + dr, cur[1] + dc
            if _in_grid(nr, nc) and dist[nr][nc] == d - 1:
                cur = (nr, nc)
                break
        else:
            raise AssertionError("BFS descent stuck -- inconsistent field")
        path.append(cur)
    return path


def _row_geometry(payload, k):
    """(grid, spawns, alts, stations, valid-slot count) for row k."""
    grid = _grid_of(payload, k)
    spawns = [tuple(payload["spawn"][k, a].tolist())
              for a in range(L.N_AGENTS)]
    alts = [tuple(payload["spawn_alt"][k, a].tolist())
            for a in range(L.N_AGENTS)]
    valid = payload["target_valid"][k]
    stations = [tuple(payload["target"][k, j].tolist())
                for j in range(L.MAX_TARGETS) if bool(valid[j])]
    return grid, spawns, alts, stations


# ---------------------------------------------------------------------------
# Serialised compliant-plan simulation through the REAL grid_core kernels
# ---------------------------------------------------------------------------

def _simulate_serialised(grid, spawns, dests, first):
    """Robot ``first`` walks to dests[first] while the other holds at its
    spawn; the other starts the step after ``first`` docks.  Docked robots
    are fed to ``step_positions`` as latched (immovable, blocking), the
    latch rule of spec 1.11.  Returns (t_dock_first, t_dock_second) or
    None when the plan is infeasible.  Asserts zero collision flags."""
    other = 1 - first
    first_path = _path(grid, spawns[first], dests[first],
                       blocked=(spawns[other], dests[other]))
    if first_path is None:
        return None
    occ1 = torch.tensor(grid, dtype=torch.int8).unsqueeze(0)
    pos = torch.tensor([spawns], dtype=torch.int16)          # (1, 2, 2)
    docked = [None, None]
    paths = {first: first_path}
    idx = {0: 0, 1: 0}
    for t in range(T_DECISION):
        acts = [STAY, STAY]
        for i in (0, 1):
            if docked[i] is not None:
                continue
            if i == other and docked[first] is None:
                continue                                     # hold at spawn
            if i not in paths:
                paths[i] = _path(grid, spawns[i], dests[i],
                                 blocked=(dests[first],))
                if paths[i] is None:
                    return None
            cur = tuple(pos[0, i].tolist())
            nxt = paths[i][idx[i] + 1]
            acts[i] = ACTION_OF[(nxt[0] - cur[0], nxt[1] - cur[1])]
        a = torch.tensor([acts], dtype=torch.long)
        latched = torch.tensor([[docked[0] is not None,
                                 docked[1] is not None]])
        nxt_pos, hit_obs, hit_rob = step_positions(
            pos, a, occ1, latched, (R, C))
        assert not bool(hit_obs.any()), \
            "compliant plan hit an obstacle at t=%d" % t
        assert not bool(hit_rob.any()), \
            "compliant plan produced a robot-robot conflict at t=%d" % t
        for i in (0, 1):
            if docked[i] is None and acts[i] != STAY:
                assert tuple(nxt_pos[0, i].tolist()) == paths[i][idx[i] + 1], \
                    "planned move was reverted at t=%d" % t
                idx[i] += 1
        pos = nxt_pos
        for i in (0, 1):
            if docked[i] is None and tuple(pos[0, i].tolist()) == dests[i]:
                docked[i] = t
        if docked[0] is not None and docked[1] is not None:
            return docked[first], docked[other]
    return None


def _complete_some_order(grid, spawns, dests):
    """RoleBinding helper: the role assignment is fixed by ``dests``; any
    serialisation order may realise it (spec 2.1)."""
    for first in (0, 1):
        out = _simulate_serialised(grid, spawns, dests, first)
        if out is not None:
            return out
    return None


# ---------------------------------------------------------------------------
# RoleBinding (spec 2.1)
# ---------------------------------------------------------------------------

def test_rb_stations_are_degree1_alcoves():
    payload = _bank("RoleBinding")
    for k in range(SMOKE_K):
        grid, _, _, stations = _row_geometry(payload, k)
        assert len(stations) == 2, "M1 rows carry exactly 2 stations"
        mouths = []
        for st in stations:
            nbrs = _free_nbrs(grid, st)
            assert len(nbrs) == 1, \
                "row %d station %r has %d free neighbours; must be a " \
                "degree-1 leaf (spec 2.1 [FIXED: pass-through])" \
                % (k, st, len(nbrs))
            mouths.append(nbrs[0])
        # left/right must be geometrically unambiguous (fleet_env resolves
        # 'left' as the smaller-column station)
        cols = sorted(st[1] for st in stations)
        assert cols[0] != cols[1], "row %d station columns coincide" % k
        left_mouth = min(mouths, key=lambda m: m[1])
        right_mouth = max(mouths, key=lambda m: m[1])
        assert right_mouth[1] - left_mouth[1] >= 5, \
            "row %d mouths %r / %r closer than 5 columns (spec 2.1)" \
            % (k, left_mouth, right_mouth)
        assert left_mouth[1] <= C // 2 - 1 and right_mouth[1] >= C // 2, \
            "row %d mouths not on opposite halves of the map" % k
        # mouth = (-1, -1) sentinel for role binding (spec 1.10 table)
        assert payload["mouth"][k].tolist() == [-1, -1]


def test_rb_spawn_window_asymmetry_and_alt_difficulty():
    payload = _bank("RoleBinding")
    for k in range(SMOKE_K):
        grid, spawns, alts, stations = _row_geometry(payload, k)
        left = min(stations, key=lambda s: s[1])
        right = max(stations, key=lambda s: s[1])
        d_left = _bfs(grid, left, blocked=(right,))      # latch-aware
        d_right = _bfs(grid, right, blocked=(left,))

        for label, pair in (("spawn", spawns), ("spawn_alt", alts)):
            assert pair[0] != pair[1], "row %d %s cells coincide" % (k, label)
            for cell in pair:
                assert grid[cell[0]][cell[1]] == 0, \
                    "row %d %s %r on an obstacle" % (k, label, cell)
                assert cell not in stations, \
                    "row %d %s %r on a station" % (k, label, cell)
                for field in (d_left, d_right):
                    d = field[cell[0]][cell[1]]
                    assert 4 <= d <= 14, \
                        "row %d %s %r latch-aware distance %d outside " \
                        "[4, 14] (spec 2.1)" % (k, label, cell, d)
            dl = [d_left[c[0]][c[1]] for c in pair]
            dr = [d_right[c[0]][c[1]] for c in pair]
            assert abs(dl[0] - dl[1]) <= 1 and abs(dr[0] - dr[1]) <= 1, \
                "row %d %s per-station asymmetry beyond +-1 (spec 2.1 " \
                "[FIXED: per scenario])" % (k, label)
            assert abs((dl[0] + dr[1]) - (dr[0] + dl[1])) <= 1, \
                "row %d %s matching-cost asymmetry beyond +-1" % (k, label)

        # spawn_alt is an actual intervention (at least one robot moved)
        # AND difficulty-matched (spec 1.10 / 6.1 D_spawn): matching cost
        # and signed asymmetry preserved, so delta_gap/leak_bit unchanged
        assert tuple(alts) != tuple(spawns), \
            "row %d spawn_alt identical to spawn -- no intervention" % k

        def costs(pair):
            dl = [d_left[c[0]][c[1]] for c in pair]
            dr = [d_right[c[0]][c[1]] for c in pair]
            return dl[0] + dr[1], dr[0] + dl[1]
        assert costs(alts) == costs(spawns), \
            "row %d spawn_alt is not difficulty-matched" % k


def test_rb_compliant_completion_both_assignments():
    # [FIXED: precedence deadlock class of bugs, RB flavour]: for EVERY row
    # and EVERY spawn set, BOTH role assignments must be completable, so a
    # compliant policy can always score Y = 1 whichever class is drawn.
    payload = _bank("RoleBinding")
    for k in range(SMOKE_K):
        grid, spawns, alts, stations = _row_geometry(payload, k)
        left = min(stations, key=lambda s: s[1])
        right = max(stations, key=lambda s: s[1])
        for label, pair in (("spawn", spawns), ("spawn_alt", alts)):
            for dests in ((left, right), (right, left)):   # RB0 / RB1
                out = _complete_some_order(grid, list(pair), dests)
                assert out is not None, \
                    "row %d %s: assignment r0->%r not completable " \
                    "(spec 2.1 compliant-completion rejection failed)" \
                    % (k, label, dests[0])
                t_a, t_b = out
                assert max(t_a, t_b) < T_DECISION, \
                    "row %d completion after the horizon" % k


# ---------------------------------------------------------------------------
# Precedence (spec 3.1)
# ---------------------------------------------------------------------------

def test_pr_unique_mouth_topology():
    payload = _bank("Precedence")
    for k in range(SMOKE_K):
        grid, spawns, alts, stations = _row_geometry(payload, k)
        assert len(stations) == 2
        mouth = tuple(payload["mouth"][k].tolist())
        assert _in_grid(*mouth) and grid[mouth[0]][mouth[1]] == 0, \
            "row %d mouth %r invalid" % (k, mouth)

        # stations are leaves attached to the mouth
        for st in stations:
            assert _free_nbrs(grid, st) == [mouth], \
                "row %d station %r is not a degree-1 leaf on the mouth " \
                "(free neighbours: %r)" % (k, st, _free_nbrs(grid, st))
        # the mouth's free neighbours are EXACTLY the two stations plus the
        # corridor end -- the unique-mouth property that makes |dt| >= 1
        # structural (spec 3.1)
        m_nbrs = set(_free_nbrs(grid, mouth))
        assert len(m_nbrs) == 3 and set(stations) < m_nbrs, \
            "row %d mouth neighbours %r violate the airlock topology" \
            % (k, m_nbrs)
        corridor_end = next(iter(m_nbrs - set(stations)))

        # width-1 corridor of length >= 3: walk the degree-2 chain
        chain = []
        prev, cur = mouth, corridor_end
        for _ in range(R * C):
            nbrs = _free_nbrs(grid, cur)
            if len(nbrs) != 2:
                break
            chain.append(cur)
            nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
            prev, cur = cur, nxt
        assert len(chain) >= 3, \
            "row %d corridor length %d < 3 (spec 3.1)" % (k, len(chain))

        # latch-aware BFS from the mouth with the OTHER station blocked
        # reaches both stations (the check MapGen's flood fill cannot do)
        for st, other in ((stations[0], stations[1]),
                          (stations[1], stations[0])):
            d = _bfs(grid, mouth, blocked=(other,))
            assert d[st[0]][st[1]] == 1, \
                "row %d station %r unreachable from the mouth with %r " \
                "blocked" % (k, st, other)

        # spawns: outside the corridor, d(spawn, m) >= 3 [FIXED: the draft
        # constrained only Delta]; also holds for spawn_alt
        dm = _bfs(grid, mouth, blocked=tuple(stations))
        assert tuple(alts) != tuple(spawns), \
            "row %d spawn_alt identical to spawn -- no intervention" % k
        bay = {mouth} | set(stations) | \
            {cell for cell in chain if dm[cell[0]][cell[1]] <= 3}
        for label, pair in (("spawn", spawns), ("spawn_alt", alts)):
            assert pair[0] != pair[1]
            for cell in pair:
                assert grid[cell[0]][cell[1]] == 0
                assert cell not in bay, \
                    "row %d %s %r inside the airlock (spec 3.1)" \
                    % (k, label, cell)
                assert dm[cell[0]][cell[1]] >= 3, \
                    "row %d %s %r has d(spawn, m) = %d < 3" \
                    % (k, label, cell, dm[cell[0]][cell[1]])
            # delta_gap == d(r0, m) - d(r1, m), preserved by spawn_alt
            delta = dm[pair[0][0]][pair[0][1]] - dm[pair[1][0]][pair[1][1]]
            assert delta == int(payload["delta_gap"][k]), \
                "row %d %s realises Delta=%d, bank says %d" \
                % (k, label, delta, int(payload["delta_gap"][k]))


def test_pr_compliant_completion_both_orderings():
    # No precedence deadlock (spec 3.1): BOTH orderings are feasible on
    # every row and every spawn set -- compliant completion is always
    # possible whichever class is drawn -- and |dt| >= 1 on every
    # simulated completion (ties structurally impossible, G = 1).
    payload = _bank("Precedence")
    for k in range(SMOKE_K):
        grid, spawns, alts, stations = _row_geometry(payload, k)
        for label, pair in (("spawn", spawns), ("spawn_alt", alts)):
            for first in (0, 1):
                dests = [None, None]
                dests[first] = stations[0]
                dests[1 - first] = stations[1]
                out = _simulate_serialised(grid, list(pair), dests, first)
                assert out is not None, \
                    "row %d %s: ordering robot_%d-first not completable " \
                    "(precedence deadlock -- spec 3.1 topology broken)" \
                    % (k, label, first)
                t_first, t_second = out
                assert t_second - t_first >= 1, \
                    "row %d %s: |dt| < 1 -- mouth serialisation violated" \
                    % (k, label)
                assert t_second < T_DECISION, \
                    "row %d %s completion after the horizon" % (k, label)


def test_pr_delta_stratification_sign_symmetric():
    # Delta drawn sign-symmetrically from {0, +-1, ..., +-6} (spec 3.1)
    payload = _bank("Precedence")
    gaps = payload["delta_gap"].long().tolist()
    assert all(-6 <= g <= 6 for g in gaps)
    for d in range(1, 7):
        assert gaps.count(d) == gaps.count(-d), \
            "Delta=+%d count %d != Delta=-%d count %d (sign symmetry)" \
            % (d, gaps.count(d), d, gaps.count(-d))


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
