#!/usr/bin/env python
"""Offline scenario-bank builder for Team Listen (M1_SPEC 1.10, 2.1, 3.1).

Builds ``data/scenario_bank_{variant}_{sha12}.pt`` plus a JSON manifest
``data/scenario_bank_{variant}.json`` for ``variant`` in {RoleBinding,
Precedence}.  The bank is the frozen-physics artifact of spec 1.10: NO RNG
runs in ``_reset_idx`` -- every map, spawn, target, latch-aware BFS field
and slip draw is precomputed here, deterministically, from an explicit
per-row seed list.

Spec decisions implemented (read the [FIXED: ...] notes before editing):

* RoleBinding (spec 2.1): 12x12 MapGen obstacles (scaled down:
  num_clusters=3, cluster_size_range=(2,6), min_cluster_distance=3.0),
  both stations DEGREE-1 ALCOVES (each station cell has exactly one free
  neighbour, its mouth), the two mouths >= 5 columns apart on opposite
  halves of the map.  Spawns are free cells with latch-aware BFS distance
  to EACH station in [4, 14]; the spawn pair's station-distance asymmetry
  is capped at +-1 PER SCENARIO (both per-station |d(r0,s)-d(r1,s)| <= 1
  and the signed matching-cost asymmetry |c_id - c_sw| <= 1) [FIXED: the
  draft enforced this only in aggregate].  Which physical alcove occupies
  target slot 0 and which robot spawns where are randomised per row.
* Precedence (spec 3.1 unique-mouth airlock): a width-1 corridor of
  length >= 3 leads to a single mouth cell m; the two station cells are
  the ONLY other free neighbours of m and each is degree-1, so |dt| >= 1
  is structural and both orderings are always feasible.  Spawns are
  outside the corridor with d(spawn, m) >= 3 [FIXED: the draft
  constrained only Delta].  Delta = d(r0,m) - d(r1,m) is stratified
  sign-symmetrically over {0,+-1,...,+-6} and stored as ``delta_gap``.
* dist_field is LATCH-AWARE BFS (spec 1.10 [FIXED]): the field for target
  j treats every other valid target cell as an obstacle.  Under the
  alcove topology it coincides with the all-free field everywhere except
  AT the other station cell itself (a leaf); the builder asserts that
  invariant per row and ``tests/test_bank_distfields.py`` re-checks it.
* Compliant-completion certificate (spec 2.1/3.1; tested by
  ``tests/test_bank_latch_reachability.py``): every accepted row -- for
  BOTH role assignments (RoleBinding) / BOTH orderings (Precedence), and
  for spawn_alt as well as spawn -- admits a serialised compliant plan
  (mover's BFS path avoids the stationary robot's cell and the other
  station; the second robot moves after the first has latched).  No
  precedence deadlock, no role-binding trap, well inside T_DECISION.
* spawn_alt (spec 1.10 / 6.1 D_spawn): a difficulty-matched nuisance
  intervention.  Preference ladder: (a) both robots moved with the exact
  per-robot latch-aware distance profile preserved, (b) one robot moved,
  exact profile, (c) [RoleBinding only] equal matching cost AND equal
  signed asymmetry.  Every rung preserves delta_gap and leak_bit and must
  itself pass the compliant-completion certificate.
* leak_bit (spec 1.10/4.1): the sign of the geometric default, in the
  instruction-class index space used by ``fleet_env._outcome_correct``:
  RoleBinding -- 0 iff the min-cost matching sends robot_0 to the LEFT
  (smaller-column) station (RB0), 1 otherwise; Precedence -- 0 iff
  Delta < 0 (robot_0 closer to the mouth, geometric default = PR0),
  1 iff Delta > 0.  Exact ties alternate with row parity (k % 2) so the
  tie stratum stays balanced.  leak_bit is a purely geometric scalar:
  this script imports NOTHING language-side (no harness.templates, no
  lang_cache) -- the bank is built before any instruction exists.
* slip (spec 1.10 [FIXED: no physics seed]): per-row precomputed streams
  ``(N_STREAMS=2, T=128, MAX_AGENTS=4)`` uint8; with probability epsilon
  (default 0.05, pre-registered) the entry is a forced action 0..4,
  otherwise NO_SLIP=5.
* Determinism: every row is built from an explicit per-row seed (either
  given via --seeds or derived from --seed by a SHA-256 mix recorded in
  the manifest); all randomness flows through per-row ``random.Random`` /
  ``torch.Generator`` objects.  Same seed list => bit-identical .pt bytes
  => same SHA (``tests/test_bank_determinism.py``).

The .pt payload is a flat dict: the 12 tensor keys of spec 1.10 (matching
``fleet_env._BANK_KEYS``) plus flat non-tensor metadata -- including
``leaky: False`` at TOP level, which the loaders' leaky-bank refusal keys
on (spec 4.1; leaky banks come only from scripts/build_leaky_bank.py).
"""

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
from collections import deque

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tasks.team_listen import grid_core                     # noqa: E402
from tasks.team_listen import obs_layout as L               # noqa: E402

# ---------------------------------------------------------------------------
# Constants (spec 1.3 / 1.10 / 2.1 / 3.1)
# ---------------------------------------------------------------------------

R, C = L.R, L.C
N_STREAMS = 2
SCHEMA_VERSION = 1
VARIANTS = ("RoleBinding", "Precedence")

# MapGen, scaled down per spec 2.1.
MAPGEN_NUM_CLUSTERS = 3
MAPGEN_CLUSTER_SIZE_RANGE = (2, 6)
MAPGEN_MIN_CLUSTER_DISTANCE = 3.0

# RoleBinding (spec 2.1)
RB_SPAWN_DIST_MIN = 4
RB_SPAWN_DIST_MAX = 14
RB_ASYM_CAP = 1                     # per-station AND matching-cost asymmetry
RB_MOUTH_MIN_COL_SEP = 5
RB_LEFT_COL_MAX = C // 2 - 1        # 5: left-half columns [0, 5]
RB_RIGHT_COL_MIN = C // 2           # 6: right-half columns [6, 11]

# Precedence (spec 3.1)
PR_CORRIDOR_LENGTHS = (3, 4, 5)     # >= 3 per spec
PR_SPAWN_MOUTH_MIN_DIST = 3
PR_DELTA_MAX = 6                    # Delta in {0, +-1, ..., +-6}

# Compliant serialised plan must finish well inside T_DECISION.
PLAN_TIME_BUDGET = 120

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

DEFAULT_K = 16384                   # 14336 train / 2048 eval (spec 1.10)
DEFAULT_EVAL_FRAC = 0.125
DEFAULT_EPSILON = 0.05              # pre-registered slip rate (spec 1.10)
DEFAULT_SEED = 20260831


# ---------------------------------------------------------------------------
# Deterministic seed plumbing
# ---------------------------------------------------------------------------

def _mix(*parts):
    """SHA-256 based 63-bit mixer: deterministic across platforms/runs."""
    text = "team_listen_bank/" + "/".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def derive_seed_list(master_seed, k):
    """The explicit per-row seed list derived from a master seed.

    Recorded in the manifest; passing this exact list via --seeds
    reproduces the bank bit-identically (tests/test_bank_determinism.py).
    """
    return [_mix(master_seed, i) for i in range(k)]


# ---------------------------------------------------------------------------
# Grid helpers (pure python; offline only -- never on the hot path)
# ---------------------------------------------------------------------------

def _in_grid(r, c):
    return 0 <= r < R and 0 <= c < C


def _nbrs4(cell):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4 if _in_grid(r + dr, c + dc)]


def _free_nbrs(grid, cell):
    return [n for n in _nbrs4(cell) if grid[n[0]][n[1]] == 0]


def _bfs(grid, src, blocked=()):
    """(R, C) list-of-lists BFS distance over 4-connected free cells.

    ``blocked``: extra cells treated as obstacles (the latch-aware rule of
    spec 1.10 blocks the OTHER target cells).  Unreachable cells hold -1,
    the sentinel ``matching_potential`` maps to +inf.
    """
    dist = [[-1] * C for _ in range(R)]
    blocked = set(blocked)
    sr, sc = src
    if grid[sr][sc] != 0 or src in blocked:
        return dist
    dist[sr][sc] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        d = dist[r][c] + 1
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (_in_grid(nr, nc) and grid[nr][nc] == 0
                    and (nr, nc) not in blocked and dist[nr][nc] < 0):
                dist[nr][nc] = d
                q.append((nr, nc))
    return dist


def _gen_clusters(rng, protected):
    """MapGen cluster pass (port of reference/env_paper.py MapGen.generate,
    spec 2.1 scaled-down parameters), WITHOUT the pocket fill -- carving
    runs first and the pocket fill must not swallow protected cells.

    ``protected``: cells the generator must leave free (alcove/airlock
    structure -- both its required-free and required-obstacle cells, which
    are applied explicitly afterwards).
    """
    grid = [[0] * C for _ in range(R)]
    centres = []
    tries = 0
    while len(centres) < MAPGEN_NUM_CLUSTERS and tries < 5000:
        tries += 1
        cand = (rng.randrange(1, R - 1), rng.randrange(1, C - 1))
        if cand in protected:
            continue
        if all(math.hypot(cand[0] - o[0], cand[1] - o[1])
               >= MAPGEN_MIN_CLUSTER_DISTANCE for o in centres):
            centres.append(cand)
    dirs8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1))
    lo, hi = MAPGEN_CLUSTER_SIZE_RANGE
    for (cx, cy) in centres:
        size = rng.randrange(lo, hi + 1)
        cells = [(cx, cy)]
        grid[cx][cy] = 1
        stuck = 0
        while len(cells) < size and stuck < 200:
            base = cells[rng.randrange(len(cells))]
            order = list(range(8))
            rng.shuffle(order)
            grew = False
            for k8 in order:
                dx, dy = dirs8[k8]
                nx, ny = base[0] + dx, base[1] + dy
                if (1 <= nx < R - 1 and 1 <= ny < C - 1
                        and grid[nx][ny] == 0 and (nx, ny) not in protected):
                    grid[nx][ny] = 1
                    cells.append((nx, ny))
                    grew = True
                    break
            stuck = 0 if grew else stuck + 1
    return grid


def _fill_pockets(grid, must_keep):
    """Reference MapGen pocket fill, protected-cell aware.

    Keeps the largest 4-connected free component (deterministic tie-break:
    smallest lexicographic cell) and converts every other free pocket to
    obstacles.  Returns False -- caller must reject the attempt -- when any
    ``must_keep`` cell (station/mouth/corridor structure) falls outside the
    kept component; the reference fill is blind to that and would silently
    delete a station.
    """
    seen = [[False] * C for _ in range(R)]
    comps = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 0 and not seen[r][c]:
                comp = []
                stack = [(r, c)]
                seen[r][c] = True
                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    for dx, dy in DIRS4:
                        nx, ny = x + dx, y + dy
                        if (_in_grid(nx, ny) and grid[nx][ny] == 0
                                and not seen[nx][ny]):
                            seen[nx][ny] = True
                            stack.append((nx, ny))
                comps.append(comp)
    if not comps:
        return False
    comps.sort(key=lambda comp: (-len(comp), min(comp)))
    main = set(comps[0])
    if any(cell not in main for cell in must_keep):
        return False
    for comp in comps[1:]:
        for (x, y) in comp:
            grid[x][y] = 1
    return True


def _assert_latch_aware_invariant(grid, stations):
    """Alcove-topology invariant (spec 1.10 [FIXED: latch-aware field]):
    with degree-1 stations, the latch-aware and all-free BFS fields
    coincide at every cell EXCEPT the other station cell itself, where the
    latch-aware field is -1 (blocked leaf)."""
    for j, st in enumerate(stations):
        others = [s for i, s in enumerate(stations) if i != j]
        aware = _bfs(grid, st, blocked=others)
        free = _bfs(grid, st)
        for r in range(R):
            for c in range(C):
                if (r, c) in others:
                    if aware[r][c] != -1:
                        return False
                elif aware[r][c] != free[r][c]:
                    return False
    return True


# ---------------------------------------------------------------------------
# Serialised compliant-plan certificate (spec 2.1 / 3.1; the test simulates
# the same plan through grid_core.step_positions)
# ---------------------------------------------------------------------------

def _serial_ok(grid, spawns, dests, first):
    """Serialised compliant plan feasibility for one order.

    Phase 1: robot ``first`` walks spawn -> its station while the other
    robot HOLDS at its spawn; the path must avoid the holder's cell and
    the other station (latch-aware: transiting a station cell would latch,
    spec 1.11).  Phase 2: the other robot walks while ``first`` sits
    latched in its degree-1 station.  Returns the total plan length or
    None when either leg is disconnected.
    """
    o = 1 - first
    d1 = _bfs(grid, dests[first], blocked=(spawns[o], dests[o]))
    leg1 = d1[spawns[first][0]][spawns[first][1]]
    if leg1 < 0:
        return None
    d2 = _bfs(grid, dests[o], blocked=(dests[first],))
    leg2 = d2[spawns[o][0]][spawns[o][1]]
    if leg2 < 0:
        return None
    total = leg1 + leg2 + 1                 # +1: phase-2 start-up latency
    return total if total <= PLAN_TIME_BUDGET else None


def _rb_feasible(grid, spawns, left, right):
    """RoleBinding: BOTH role assignments must admit a compliant plan
    (some serialisation order each) -- spec 2.1's per-row rejection."""
    for dests in ((left, right), (right, left)):        # r0->left, r0->right
        if (_serial_ok(grid, spawns, dests, 0) is None
                and _serial_ok(grid, spawns, dests, 1) is None):
            return False
    return True


def _pr_feasible(grid, spawns, stations, dm):
    """Precedence: BOTH orderings must be feasible (spec 3.1) -- the order
    is mandated, so the serialisation is fixed: the designated-first robot
    moves first (to station slot 0's cell), the other robot holds, then
    walks to the other station."""
    for first in (0, 1):
        o = 1 - first
        d1 = _bfs(grid, stations[0],
                  blocked=(stations[1], spawns[o]))
        leg1 = d1[spawns[first][0]][spawns[first][1]]
        if leg1 < 0:
            return False
        leg2 = dm[spawns[o][0]][spawns[o][1]]           # stations are leaves:
        if leg2 < 0:                                    # d(spawn, s) = dm + 1
            return False
        if leg1 + (leg2 + 1) + 1 > PLAN_TIME_BUDGET:
            return False
    return True


# ---------------------------------------------------------------------------
# RoleBinding row builder (spec 2.1)
# ---------------------------------------------------------------------------

def _sample_alcove(rng, col_lo, col_hi):
    """One (station, mouth) pair with both cells inside [col_lo, col_hi]."""
    for _ in range(40):
        st = (rng.randrange(R), rng.randrange(col_lo, col_hi + 1))
        dirs = list(DIRS4)
        rng.shuffle(dirs)
        for dr, dc in dirs:
            mo = (st[0] + dr, st[1] + dc)
            if _in_grid(*mo) and col_lo <= mo[1] <= col_hi:
                return st, mo
    return None


def _rb_spawn_alt(rng, grid, candidates, spawns, d_left, d_right,
                  left, right):
    """Difficulty-matched spawn_alt (spec 1.10 ladder, RoleBinding)."""
    prof = [(d_left[s[0]][s[1]], d_right[s[0]][s[1]]) for s in spawns]
    pools = [sorted(c for c in candidates
                    if (d_left[c[0]][c[1]], d_right[c[0]][c[1]]) == prof[i])
             for i in (0, 1)]

    def ok(pair):
        a0, a1 = pair
        return (a0 != a1 and (a0, a1) != tuple(spawns)
                and _rb_feasible(grid, pair, left, right))

    # (a) both moved, exact per-robot profile; (b) one moved, exact profile
    for require_both in (True, False):
        pairs = []
        for a0 in pools[0]:
            for a1 in pools[1]:
                moved0, moved1 = a0 != spawns[0], a1 != spawns[1]
                if require_both and not (moved0 and moved1):
                    continue
                if not require_both and not (moved0 or moved1):
                    continue
                pairs.append((a0, a1))
        rng.shuffle(pairs)
        for pair in pairs[:60]:
            if ok(pair):
                return pair
    # (c) equal matching cost AND equal signed asymmetry
    c_id = prof[0][0] + prof[1][1]
    c_sw = prof[0][1] + prof[1][0]
    for _ in range(200):
        a0 = rng.choice(candidates)
        a1 = rng.choice(candidates)
        if a0 == a1 or (a0, a1) == tuple(spawns):
            continue
        pl = [(d_left[a[0]][a[1]], d_right[a[0]][a[1]]) for a in (a0, a1)]
        if abs(pl[0][0] - pl[1][0]) > RB_ASYM_CAP:
            continue
        if abs(pl[0][1] - pl[1][1]) > RB_ASYM_CAP:
            continue
        if pl[0][0] + pl[1][1] != c_id or pl[0][1] + pl[1][0] != c_sw:
            continue
        if ok((a0, a1)):
            return (a0, a1)
    return None


def _build_role_binding_row(row_seed, row_index, max_attempts):
    for attempt in range(max_attempts):
        rng = random.Random(_mix(row_seed, "map", attempt))

        alc_l = _sample_alcove(rng, 0, RB_LEFT_COL_MAX)
        alc_r = _sample_alcove(rng, RB_RIGHT_COL_MIN, C - 1)
        if alc_l is None or alc_r is None:
            continue
        (st_l, mo_l), (st_r, mo_r) = alc_l, alc_r
        if mo_r[1] - mo_l[1] < RB_MOUTH_MIN_COL_SEP:
            continue

        free_req = {st_l, mo_l, st_r, mo_r}
        obs_req = set()
        for st, mo in ((st_l, mo_l), (st_r, mo_r)):
            obs_req.update(n for n in _nbrs4(st) if n != mo)
        if free_req & obs_req:
            continue

        grid = _gen_clusters(rng, protected=free_req | obs_req)
        for (r, c) in obs_req:
            grid[r][c] = 1
        for (r, c) in free_req:
            grid[r][c] = 0
        if not _fill_pockets(grid, free_req):
            continue

        # leaf property (spec 2.1): exactly one free neighbour, the mouth
        if set(_free_nbrs(grid, st_l)) != {mo_l}:
            continue
        if set(_free_nbrs(grid, st_r)) != {mo_r}:
            continue
        if not _assert_latch_aware_invariant(grid, (st_l, st_r)):
            continue

        d_left = _bfs(grid, st_l, blocked=(st_r,))      # latch-aware fields
        d_right = _bfs(grid, st_r, blocked=(st_l,))
        candidates = sorted(
            (r, c) for r in range(R) for c in range(C)
            if grid[r][c] == 0 and (r, c) not in (st_l, st_r)
            and RB_SPAWN_DIST_MIN <= d_left[r][c] <= RB_SPAWN_DIST_MAX
            and RB_SPAWN_DIST_MIN <= d_right[r][c] <= RB_SPAWN_DIST_MAX)
        if len(candidates) < 2:
            continue

        for _ in range(300):
            r0 = rng.choice(candidates)
            dl0, dr0 = d_left[r0[0]][r0[1]], d_right[r0[0]][r0[1]]
            pool = [c for c in candidates if c != r0
                    and abs(d_left[c[0]][c[1]] - dl0) <= RB_ASYM_CAP
                    and abs(d_right[c[0]][c[1]] - dr0) <= RB_ASYM_CAP]
            pool = [c for c in pool
                    if abs((dl0 + d_right[c[0]][c[1]])
                           - (dr0 + d_left[c[0]][c[1]])) <= RB_ASYM_CAP]
            if not pool:
                continue
            r1 = rng.choice(pool)
            spawns = (r0, r1)
            if not _rb_feasible(grid, spawns, st_l, st_r):
                continue
            alt = _rb_spawn_alt(rng, grid, candidates, spawns,
                                d_left, d_right, st_l, st_r)
            if alt is None:
                continue

            dl1 = d_left[r1[0]][r1[1]]
            dr1 = d_right[r1[0]][r1[1]]
            c_id = dl0 + dr1                    # r0 -> left (RB0)
            c_sw = dr0 + dl1                    # r0 -> right (RB1)
            asym = c_id - c_sw
            if asym < 0:
                leak = 0
            elif asym > 0:
                leak = 1
            else:
                leak = row_index % 2            # tie: alternate, balanced

            # randomise which physical alcove occupies slot 0 (spec 2.1)
            if rng.random() < 0.5:
                slots = [(st_l, d_left), (st_r, d_right)]
            else:
                slots = [(st_r, d_right), (st_l, d_left)]

            return {
                "occ": grid,
                "spawn": spawns,
                "spawn_alt": alt,
                "stations": [s[0] for s in slots],
                "fields": [s[1] for s in slots],
                "mouth": (-1, -1),
                "delta_gap": asym,              # signed matching asymmetry
                "leak_bit": leak,
            }
    raise RuntimeError(
        "RoleBinding row %d (seed %d): no valid scenario in %d map attempts "
        "(M1_SPEC 2.1 rejections too tight for this seed?)"
        % (row_index, row_seed, max_attempts))


# ---------------------------------------------------------------------------
# Precedence row builder (spec 3.1)
# ---------------------------------------------------------------------------

def _pr_structure(rng):
    """Sample the unique-mouth airlock (spec 3.1 final topology)."""
    combos = [(d, lc) for d in DIRS4 for lc in PR_CORRIDOR_LENGTHS]
    rng.shuffle(combos)
    for axis, lc in combos:
        perp = (axis[1], axis[0])
        ms = []
        for r in range(R):
            for c in range(C):
                m = (r, c)
                cells = [m, (r + perp[0], c + perp[1]),
                         (r - perp[0], c - perp[1])]
                cells += [(r - axis[0] * i, c - axis[1] * i)
                          for i in range(1, lc + 2)]
                if all(_in_grid(*cell) for cell in cells):
                    ms.append(m)
        if not ms:
            continue
        m = rng.choice(sorted(ms))
        st_a = (m[0] + perp[0], m[1] + perp[1])
        st_b = (m[0] - perp[0], m[1] - perp[1])
        corridor = [(m[0] - axis[0] * i, m[1] - axis[1] * i)
                    for i in range(1, lc + 1)]
        opening = (m[0] - axis[0] * (lc + 1), m[1] - axis[1] * (lc + 1))

        free_req = {m, st_a, st_b, opening} | set(corridor)
        obs_req = set()
        beyond = (m[0] + axis[0], m[1] + axis[1])
        if _in_grid(*beyond):
            obs_req.add(beyond)
        for st in (st_a, st_b):
            obs_req.update(n for n in _nbrs4(st) if n != m)
        for cell in corridor:
            for sgn in (1, -1):
                w = (cell[0] + sgn * perp[0], cell[1] + sgn * perp[1])
                if _in_grid(*w):
                    obs_req.add(w)
        if free_req & obs_req:
            continue
        return m, st_a, st_b, corridor, opening, free_req, obs_req
    return None


def _pr_spawn_alt(rng, grid, classes, spawns, dm, stations):
    """Difficulty-matched spawn_alt (Precedence): per-robot d(., m)
    profile preserved exactly, so delta_gap and leak_bit are unchanged."""
    prof = [dm[s[0]][s[1]] for s in spawns]
    pools = [sorted(classes.get(prof[i], [])) for i in (0, 1)]

    def ok(pair):
        a0, a1 = pair
        return (a0 != a1 and (a0, a1) != tuple(spawns)
                and _pr_feasible(grid, pair, stations, dm))

    for require_both in (True, False):
        pairs = []
        for a0 in pools[0]:
            for a1 in pools[1]:
                moved0, moved1 = a0 != spawns[0], a1 != spawns[1]
                if require_both and not (moved0 and moved1):
                    continue
                if not require_both and not (moved0 or moved1):
                    continue
                pairs.append((a0, a1))
        rng.shuffle(pairs)
        for pair in pairs[:60]:
            if ok(pair):
                return pair
    return None


def _build_precedence_row(row_seed, row_index, target_delta, max_attempts):
    for attempt in range(max_attempts):
        rng = random.Random(_mix(row_seed, "map", attempt))
        structure = _pr_structure(rng)
        if structure is None:
            continue
        m, st_a, st_b, corridor, opening, free_req, obs_req = structure

        grid = _gen_clusters(rng, protected=free_req | obs_req)
        for (r, c) in obs_req:
            grid[r][c] = 1
        for (r, c) in free_req:
            grid[r][c] = 0
        if not _fill_pockets(grid, free_req):
            continue

        # topology verification (spec 3.1; re-checked by the bank tests)
        if set(_free_nbrs(grid, st_a)) != {m}:
            continue
        if set(_free_nbrs(grid, st_b)) != {m}:
            continue
        if set(_free_nbrs(grid, m)) != {st_a, st_b, corridor[0]}:
            continue
        if any(len(_free_nbrs(grid, cell)) != 2 for cell in corridor):
            continue
        if not _assert_latch_aware_invariant(grid, (st_a, st_b)):
            continue

        dm = _bfs(grid, m, blocked=(st_a, st_b))
        excluded = {m, st_a, st_b} | set(corridor)
        classes = {}
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 0 and (r, c) not in excluded:
                    d = dm[r][c]
                    if d >= PR_SPAWN_MOUTH_MIN_DIST:
                        classes.setdefault(d, []).append((r, c))
        d0_opts = sorted(
            d for d in classes
            if (d - target_delta) in classes
            and (d != d - target_delta or len(classes[d]) >= 2))
        if not d0_opts:
            continue

        for _ in range(300):
            d0 = rng.choice(d0_opts)
            r0 = rng.choice(sorted(classes[d0]))
            pool = [c for c in classes[d0 - target_delta] if c != r0]
            if not pool:
                continue
            r1 = rng.choice(sorted(pool))
            spawns = (r0, r1)

            # randomise which physical station occupies slot 0 (spec 2.1
            # discipline applied to both variants)
            if rng.random() < 0.5:
                stations = [st_a, st_b]
            else:
                stations = [st_b, st_a]
            if not _pr_feasible(grid, spawns, stations, dm):
                continue
            alt = _pr_spawn_alt(rng, grid, classes, spawns, dm, stations)
            if alt is None:
                continue

            if target_delta < 0:
                leak = 0                        # r0 closer -> PR0 default
            elif target_delta > 0:
                leak = 1
            else:
                leak = row_index % 2
            other = {stations[0]: stations[1], stations[1]: stations[0]}
            fields = [_bfs(grid, st, blocked=(other[st],))
                      for st in stations]
            return {
                "occ": grid,
                "spawn": spawns,
                "spawn_alt": alt,
                "stations": stations,
                "fields": fields,
                "mouth": m,
                "delta_gap": target_delta,
                "leak_bit": leak,
            }
    raise RuntimeError(
        "Precedence row %d (seed %d, Delta=%d): no valid scenario in %d "
        "map attempts (M1_SPEC 3.1 rejections too tight for this seed?)"
        % (row_index, row_seed, target_delta, max_attempts))


# ---------------------------------------------------------------------------
# Stratification, split, slip
# ---------------------------------------------------------------------------

def _precedence_strata(k):
    """Sign-symmetric balanced Delta targets over {0, +-1, ..., +-6}
    (spec 3.1).  Remainder rows: 0 first when odd, then (+d, -d) pairs, so
    count(+d) == count(-d) for every d."""
    values = [0]
    for d in range(1, PR_DELTA_MAX + 1):
        values += [d, -d]
    base = k // len(values)
    rem = k - base * len(values)
    counts = {v: base for v in values}
    extra = []
    if rem % 2 == 1:
        extra.append(0)
    pairs = [(d, -d) for d in range(1, PR_DELTA_MAX + 1)]
    i = 0
    while len(extra) < rem:
        extra += [pairs[i % PR_DELTA_MAX][0], pairs[i % PR_DELTA_MAX][1]]
        i += 1
    for v in extra[:rem]:
        counts[v] += 1
    out = []
    for v in values:
        out += [v] * counts[v]
    assert len(out) == k
    assert all(out.count(d) == out.count(-d)
               for d in range(1, PR_DELTA_MAX + 1))
    return out


def _assign_split(delta_gaps, eval_frac):
    """0 train / 1 eval, stratified by delta_gap value: per-group quotas by
    largest remainder, the LAST quota rows of each group (build order)
    marked eval.  Deterministic."""
    k = len(delta_gaps)
    n_eval = int(round(k * eval_frac))
    groups = {}
    for i, v in enumerate(delta_gaps):
        groups.setdefault(v, []).append(i)
    keys = sorted(groups)
    quotas = {v: (len(groups[v]) * n_eval) // k for v in keys}
    rem = n_eval - sum(quotas.values())
    fracs = sorted(keys, key=lambda v: (-((len(groups[v]) * n_eval) % k), v))
    for v in fracs[:rem]:
        quotas[v] += 1
    split = [0] * k
    for v in keys:
        take = min(quotas[v], len(groups[v]))
        for i in groups[v][len(groups[v]) - take:]:
            split[i] = 1
    return split


def _slip_for_row(row_seed, epsilon):
    """(N_STREAMS, T_DECISION, MAX_AGENTS) uint8 slip tensor (spec 1.10)."""
    gen = torch.Generator().manual_seed(_mix(row_seed, "slip"))
    shape = (N_STREAMS, L.T_DECISION, L.MAX_AGENTS)
    u = torch.rand(shape, generator=gen)
    forced = torch.randint(0, grid_core.N_ACTIONS, shape, generator=gen)
    slip = torch.where(u < epsilon, forced,
                       torch.full_like(forced, grid_core.NO_SLIP))
    return slip.to(torch.uint8)


# ---------------------------------------------------------------------------
# Bank assembly
# ---------------------------------------------------------------------------

def build_bank(variant, k, seed_list, epsilon=DEFAULT_EPSILON,
               eval_frac=DEFAULT_EVAL_FRAC, max_attempts=80, verbose=False):
    """Build one bank; returns the flat payload dict of spec 1.10.

    Deterministic: identical (variant, k, seed_list, epsilon, eval_frac)
    yield a bit-identical payload and .pt file.
    """
    if variant not in VARIANTS:
        raise ValueError("variant %r not in %r" % (variant, VARIANTS))
    if len(seed_list) != k:
        raise ValueError("seed list length %d != k=%d" % (len(seed_list), k))

    strata = _precedence_strata(k) if variant == "Precedence" else [None] * k

    occ = torch.zeros((k, R, C), dtype=torch.uint8)
    spawn = torch.zeros((k, L.MAX_AGENTS, 2), dtype=torch.int16)
    spawn_alt = torch.zeros((k, L.MAX_AGENTS, 2), dtype=torch.int16)
    target = torch.zeros((k, L.MAX_TARGETS, 2), dtype=torch.int16)
    target_valid = torch.zeros((k, L.MAX_TARGETS), dtype=torch.bool)
    dist_field = torch.full((k, L.MAX_TARGETS, R, C), -1, dtype=torch.int16)
    mouth = torch.full((k, 2), -1, dtype=torch.int16)
    delta_gap = torch.zeros((k,), dtype=torch.int8)
    leak_bit = torch.zeros((k,), dtype=torch.uint8)
    instr_switch_time = torch.full((k,), -1, dtype=torch.int16)  # M1: inert
    slip = torch.zeros((k, N_STREAMS, L.T_DECISION, L.MAX_AGENTS),
                       dtype=torch.uint8)

    delta_list = []
    for i in range(k):
        seed = seed_list[i]
        if variant == "RoleBinding":
            row = _build_role_binding_row(seed, i, max_attempts)
        else:
            row = _build_precedence_row(seed, i, strata[i], max_attempts)

        occ[i] = torch.tensor(row["occ"], dtype=torch.uint8)
        for a in range(L.N_AGENTS):
            spawn[i, a] = torch.tensor(row["spawn"][a], dtype=torch.int16)
            spawn_alt[i, a] = torch.tensor(row["spawn_alt"][a],
                                           dtype=torch.int16)
        for j, st in enumerate(row["stations"]):
            target[i, j] = torch.tensor(st, dtype=torch.int16)
            target_valid[i, j] = True
            dist_field[i, j] = torch.tensor(row["fields"][j],
                                            dtype=torch.int16)
        mouth[i] = torch.tensor(row["mouth"], dtype=torch.int16)
        delta_gap[i] = int(row["delta_gap"])
        leak_bit[i] = int(row["leak_bit"])
        slip[i] = _slip_for_row(seed, epsilon)
        delta_list.append(int(row["delta_gap"]))
        if verbose and (i + 1) % 256 == 0:
            print("  %s: %d/%d rows" % (variant, i + 1, k))

    split = torch.tensor(_assign_split(delta_list, eval_frac),
                         dtype=torch.uint8)

    payload = {
        # -- the 12 tensor keys of spec 1.10 (== fleet_env._BANK_KEYS) ----
        "occ": occ,
        "spawn": spawn,
        "spawn_alt": spawn_alt,
        "target": target,
        "target_valid": target_valid,
        "dist_field": dist_field,
        "mouth": mouth,
        "delta_gap": delta_gap,
        "leak_bit": leak_bit,
        "instr_switch_time": instr_switch_time,
        "slip": slip,
        "split": split,
        # -- flat metadata (non-tensor; loaders collect it into .meta) ----
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "k": k,
        "grid": [R, C],
        "n_agents": L.N_AGENTS,
        "max_agents": L.MAX_AGENTS,
        "max_targets": L.MAX_TARGETS,
        "n_valid_targets": 2,
        "n_streams": N_STREAMS,
        "t_decision": L.T_DECISION,
        "epsilon": float(epsilon),
        "eval_frac": float(eval_frac),
        "leaky": False,                 # spec 4.1: leaky banks come ONLY
        "leak_rho": 0.0,                # from scripts/build_leaky_bank.py
        "builder": "scripts/build_scenario_bank.py",
        "seed_derivation": ("sha256('team_listen_bank/<seed>/<row>')[:8] "
                            "big-endian mod 2**63"),
        "leak_bit_rule": (
            "RoleBinding: 0 iff min-cost matching sends robot_0 to the "
            "LEFT (smaller-column) station (RB0); Precedence: 0 iff "
            "delta_gap < 0 (robot_0 closer to the mouth, PR0); exact "
            "ties alternate with row parity"),
        "delta_gap_rule": (
            "Precedence: d(r0, mouth) - d(r1, mouth) (spec 3.1); "
            "RoleBinding: signed matching-cost asymmetry "
            "(d(r0,left)+d(r1,right)) - (d(r0,right)+d(r1,left)), "
            "capped to +-1 per scenario (spec 2.1 [FIXED])"),
    }
    return payload


def serialize_bank(payload):
    """torch.save into bytes; returns (bytes, sha256 hexdigest)."""
    buf = io.BytesIO()
    torch.save(payload, buf)
    data = buf.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def bank_stats(payload):
    """Leak-audit summary statistics for the manifest (spec 1.10 / 4.3)."""
    dg = payload["delta_gap"].long()
    lb = payload["leak_bit"].long()
    sp = payload["split"].long()
    hist = {}
    joint = {}
    for v in sorted(set(dg.tolist())):
        mask = dg == v
        hist[str(v)] = int(mask.sum())
        joint[str(v)] = {"leak_bit_0": int((mask & (lb == 0)).sum()),
                         "leak_bit_1": int((mask & (lb == 1)).sum())}
    return {
        "n_train": int((sp == 0).sum()),
        "n_eval": int((sp == 1).sum()),
        "delta_gap_hist": hist,
        "leak_bit_counts": {"0": int((lb == 0).sum()),
                            "1": int((lb == 1).sum())},
        "delta_gap_x_leak_bit": joint,
        "slip_rate_measured": float(
            (payload["slip"] != grid_core.NO_SLIP).float().mean()),
        "obstacle_density_mean": float(payload["occ"].float().mean()),
    }


def save_bank(payload, out_dir, seed_info):
    """Write ``scenario_bank_{variant}_{sha12}.pt`` (spec 1.10 naming) and
    the stable-named JSON manifest ``scenario_bank_{variant}.json``.
    Returns (pt_path, manifest_path, sha256)."""
    os.makedirs(out_dir, exist_ok=True)
    data, sha = serialize_bank(payload)
    variant = payload["variant"]
    pt_name = "scenario_bank_%s_%s.pt" % (variant, sha[:12])
    pt_path = os.path.join(out_dir, pt_name)
    with open(pt_path, "wb") as f:
        f.write(data)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "file": pt_name,
        "sha256": sha,
        "k": payload["k"],
        "epsilon": payload["epsilon"],
        "eval_frac": payload["eval_frac"],
        "leaky": payload["leaky"],
        "seed_info": seed_info,
        "leak_bit_rule": payload["leak_bit_rule"],
        "delta_gap_rule": payload["delta_gap_rule"],
        "fields": {name: {"shape": list(payload[name].shape),
                          "dtype": str(payload[name].dtype)}
                   for name in BANK_TENSOR_KEYS},
        "constraints": {
            "mapgen": {"num_clusters": MAPGEN_NUM_CLUSTERS,
                       "cluster_size_range": list(MAPGEN_CLUSTER_SIZE_RANGE),
                       "min_cluster_distance": MAPGEN_MIN_CLUSTER_DISTANCE},
            "rb_spawn_dist_window": [RB_SPAWN_DIST_MIN, RB_SPAWN_DIST_MAX],
            "rb_asymmetry_cap": RB_ASYM_CAP,
            "rb_mouth_min_col_sep": RB_MOUTH_MIN_COL_SEP,
            "pr_corridor_lengths": list(PR_CORRIDOR_LENGTHS),
            "pr_spawn_mouth_min_dist": PR_SPAWN_MOUTH_MIN_DIST,
            "pr_delta_max": PR_DELTA_MAX,
            "plan_time_budget": PLAN_TIME_BUDGET,
        },
        "stats": bank_stats(payload),
    }
    manifest_path = os.path.join(out_dir,
                                 "scenario_bank_%s.json" % variant)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return pt_path, manifest_path, sha


#: The 12 tensor keys, in the spec 1.10 table order (== fleet_env._BANK_KEYS).
BANK_TENSOR_KEYS = (
    "occ", "spawn", "spawn_alt", "target", "target_valid", "dist_field",
    "mouth", "delta_gap", "leak_bit", "instr_switch_time", "slip", "split",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Offline scenario-bank builder (M1_SPEC 1.10/2.1/3.1). "
                    "Deterministic from an explicit seed list; run it, "
                    "commit the manifest, and never touch _reset_idx.")
    ap.add_argument("--variant", default="both",
                    choices=("RoleBinding", "Precedence", "both"))
    ap.add_argument("--num-scenarios", "-k", type=int, default=DEFAULT_K,
                    help="rows per variant (default %d = 14336 train + "
                         "2048 eval, spec 1.10)" % DEFAULT_K)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="master seed; the explicit per-row seed list is "
                         "derived by the recorded sha256 rule")
    ap.add_argument("--seeds", type=str, default="",
                    help="explicit comma-separated per-row seed list "
                         "(length k); overrides --seed")
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                    help="slip probability per (robot, step) (spec 1.10; "
                         "pre-registered 0.05)")
    ap.add_argument("--eval-frac", type=float, default=DEFAULT_EVAL_FRAC)
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data"))
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    variants = VARIANTS if args.variant == "both" else (args.variant,)
    if args.seeds:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
        if len(seed_list) != args.num_scenarios:
            ap.error("--seeds has %d entries, expected k=%d"
                     % (len(seed_list), args.num_scenarios))
        seed_info = {"mode": "explicit", "seed_list": seed_list}
    else:
        seed_list = derive_seed_list(args.seed, args.num_scenarios)
        seed_info = {"mode": "derived", "master_seed": args.seed,
                     "rule": "sha256('team_listen_bank/<seed>/<row>')[:8] "
                             "big-endian mod 2**63"}
        if args.num_scenarios <= 1024:
            seed_info["seed_list"] = seed_list

    for variant in variants:
        if not args.quiet:
            print("building %s bank: k=%d seed=%s epsilon=%g"
                  % (variant, args.num_scenarios,
                     seed_info.get("master_seed", "explicit"), args.epsilon))
        payload = build_bank(variant, args.num_scenarios, seed_list,
                             epsilon=args.epsilon, eval_frac=args.eval_frac,
                             max_attempts=args.max_attempts,
                             verbose=not args.quiet)
        pt_path, manifest_path, sha = save_bank(payload, args.out_dir,
                                                seed_info)
        if not args.quiet:
            stats = bank_stats(payload)
            print("  wrote %s" % pt_path)
            print("  sha256 %s" % sha)
            print("  manifest %s" % manifest_path)
            print("  train/eval %d/%d  leak_bit %s  slip_rate %.4f"
                  % (stats["n_train"], stats["n_eval"],
                     stats["leak_bit_counts"], stats["slip_rate_measured"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
