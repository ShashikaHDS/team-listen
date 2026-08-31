"""Tests for tasks/team_listen/obs_layout.py (M1_SPEC 1.3, 1.5, 1.6).

Asserts the OBS_DIM = 376 / STATE_DIM = 641 arithmetic against the spec's
slice tables, and -- independently of the module's own ``assert_layout()``
self-check -- verifies slice DISJOINTNESS and COVERAGE by counting, per
index, how many slices claim it (every index must be claimed exactly once).
Also checks the per-slot helper slices tile their parent slices, the sub-field
offsets tile their blocks, the position normaliser, and the spec 1.5/1.6
invariant that the language slices sit LAST (the blind ablation is a slice
zeroing, never a shape change).

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_obs_layout.py`` discovers and runs its own
tests with a pass/fail summary.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import obs_layout as L


# Spec 1.5 observation slice table: (name, start, stop).
OBS_TABLE = [
    ("OBS_KNOWN_FREE_SLICE", 0, 144),
    ("OBS_KNOWN_OBS_SLICE", 144, 288),
    ("OBS_EGO_SLICE", 288, 300),
    ("OBS_TEAMMATES_SLICE", 300, 327),
    ("OBS_TARGETS_SLICE", 327, 342),
    ("OBS_TIME_SLICE", 342, 344),
    ("LANG_SLICE", 344, 376),
]

# Spec 1.6 critic-state slice table.
STATE_TABLE = [
    ("STATE_OBSTACLE_SLICE", 0, 144),
    ("STATE_KNOWN_FREE_SLICE", 144, 288),
    ("STATE_KNOWN_OBS_SLICE", 288, 432),
    ("STATE_AGENTS_SLICE", 432, 484),
    ("STATE_TARGETS_SLICE", 484, 511),
    ("STATE_TIME_SLICE", 511, 513),
    ("STATE_LANG_SLICE", 513, 641),
]


def _slices(table):
    return [(name, getattr(L, name)) for name, _, _ in table]


# ---------------------------------------------------------------------------
# World-state constants (spec 1.3)
# ---------------------------------------------------------------------------

def test_world_constants():
    assert L.R == 12 and L.C == 12
    assert L.P == L.R * L.C == 144
    assert L.N_AGENTS == 2
    assert L.MAX_AGENTS == 4
    assert L.MAX_TARGETS == 3
    assert L.LANG_DIM == 32
    assert L.T_DECISION == 128
    assert L.AGENT_ID_ONEHOT_W == L.MAX_AGENTS == 4
    # Latch-slot one-hot is fixed at 4 (spec 1.5): covers MAX_TARGETS = 3
    # slots plus one spare channel; all-zeros encodes unlatched.
    assert L.LATCH_SLOT_ONEHOT_W == 4 >= L.MAX_TARGETS + 1


# ---------------------------------------------------------------------------
# Dimension arithmetic (spec 1.5 / 1.6)
# ---------------------------------------------------------------------------

def test_obs_block_widths():
    # ego: (r,c) | latched | latch-slot one-hot(4) | latch-time/T | agent-id one-hot(4)
    assert L.OBS_EGO_W == 2 + 1 + L.LATCH_SLOT_ONEHOT_W + 1 + L.AGENT_ID_ONEHOT_W == 12
    # teammate: present | (r,c) | latched | latch-slot one-hot(4) | latch-time/T
    assert L.OBS_TEAMMATE_W == 1 + 2 + 1 + L.LATCH_SLOT_ONEHOT_W + 1 == 9
    assert L.N_TEAMMATE_SLOTS == L.MAX_AGENTS - 1 == 3
    # target: present | (r,c) | occupied | occupied-by-ego
    assert L.OBS_TARGET_W == 1 + 2 + 1 + 1 == 5
    assert L.OBS_TIME_W == 2


def test_obs_dim_arithmetic():
    # 2 map planes + ego + 3 teammates + 3 targets + time + language = 376.
    assert L.OBS_DIM == (2 * L.P
                         + L.OBS_EGO_W
                         + L.N_TEAMMATE_SLOTS * L.OBS_TEAMMATE_W
                         + L.MAX_TARGETS * L.OBS_TARGET_W
                         + L.OBS_TIME_W
                         + L.LANG_DIM) == 376


def test_state_block_widths():
    # agent(13): present | (r,c) | latched | latch-slot(4) | latch-time/T | agent-id(4)
    assert L.STATE_AGENT_W == 1 + 2 + 1 + L.LATCH_SLOT_ONEHOT_W + 1 + L.AGENT_ID_ONEHOT_W == 13
    # target(9): present | (r,c) | occupied | occupier one-hot(4) | occupier latch-time/T
    assert L.STATE_TARGET_W == 1 + 2 + 1 + L.AGENT_ID_ONEHOT_W + 1 == 9
    assert L.STATE_TIME_W == 2


def test_state_dim_arithmetic():
    # 3 map planes + 4 agents + 3 targets + time + 4 per-agent lang = 641.
    assert L.STATE_DIM == (3 * L.P
                           + L.MAX_AGENTS * L.STATE_AGENT_W
                           + L.MAX_TARGETS * L.STATE_TARGET_W
                           + L.STATE_TIME_W
                           + L.MAX_AGENTS * L.LANG_DIM) == 641


def test_full_state_obs_dim():
    # Spec 4.1: full-state arms observe the 641-d state + own agent-id one-hot.
    assert L.FULL_STATE_OBS_DIM == L.STATE_DIM + L.AGENT_ID_ONEHOT_W == 645


# ---------------------------------------------------------------------------
# Slice boundaries -- spec-exact (spec 1.5 / 1.6 tables)
# ---------------------------------------------------------------------------

def test_obs_slice_boundaries():
    for name, start, stop in OBS_TABLE:
        sl = getattr(L, name)
        assert isinstance(sl, slice), name
        assert sl.step in (None, 1), (name, sl.step)
        assert (sl.start, sl.stop) == (start, stop), (name, sl)


def test_state_slice_boundaries():
    for name, start, stop in STATE_TABLE:
        sl = getattr(L, name)
        assert isinstance(sl, slice), name
        assert sl.step in (None, 1), (name, sl.step)
        assert (sl.start, sl.stop) == (start, stop), (name, sl)


# ---------------------------------------------------------------------------
# Disjointness and coverage -- computed by counting claims per index,
# independently of the boundary table and of obs_layout.assert_layout()
# ---------------------------------------------------------------------------

def _claims(slices, dim):
    """claims[i] = how many of ``slices`` contain flat index i."""
    claims = [0] * dim
    for name, sl in slices:
        assert 0 <= sl.start < sl.stop <= dim, (name, sl, dim)
        for i in range(sl.start, sl.stop):
            claims[i] += 1
    return claims


def test_obs_disjoint_and_covering():
    claims = _claims(_slices(OBS_TABLE), L.OBS_DIM)
    gaps = [i for i, c in enumerate(claims) if c == 0]
    overlaps = [i for i, c in enumerate(claims) if c > 1]
    assert not gaps, "obs indices claimed by no slice: %r..." % gaps[:8]
    assert not overlaps, "obs indices claimed twice: %r..." % overlaps[:8]


def test_state_disjoint_and_covering():
    claims = _claims(_slices(STATE_TABLE), L.STATE_DIM)
    gaps = [i for i, c in enumerate(claims) if c == 0]
    overlaps = [i for i, c in enumerate(claims) if c > 1]
    assert not gaps, "state indices claimed by no slice: %r..." % gaps[:8]
    assert not overlaps, "state indices claimed twice: %r..." % overlaps[:8]


# ---------------------------------------------------------------------------
# Per-slot helper slices tile their parent slices exactly
# ---------------------------------------------------------------------------

def _assert_tiles(parent, slot_fn, n_slots, width):
    """Slot slices are adjacent, ``width``-wide and exactly cover ``parent``."""
    cursor = parent.start
    for slot in range(n_slots):
        sl = slot_fn(slot)
        assert sl.start == cursor and sl.stop == cursor + width, (slot, sl)
        cursor = sl.stop
    assert cursor == parent.stop, (cursor, parent)


def test_teammate_slots_tile():
    _assert_tiles(L.OBS_TEAMMATES_SLICE, L.obs_teammate_slice,
                  L.N_TEAMMATE_SLOTS, L.OBS_TEAMMATE_W)


def test_obs_target_slots_tile():
    _assert_tiles(L.OBS_TARGETS_SLICE, L.obs_target_slice,
                  L.MAX_TARGETS, L.OBS_TARGET_W)


def test_state_agent_slots_tile():
    _assert_tiles(L.STATE_AGENTS_SLICE, L.state_agent_slice,
                  L.MAX_AGENTS, L.STATE_AGENT_W)


def test_state_target_slots_tile():
    _assert_tiles(L.STATE_TARGETS_SLICE, L.state_target_slice,
                  L.MAX_TARGETS, L.STATE_TARGET_W)


def test_state_lang_slots_tile():
    _assert_tiles(L.STATE_LANG_SLICE, L.state_lang_slice,
                  L.MAX_AGENTS, L.LANG_DIM)


def test_slot_helpers_reject_out_of_range():
    for fn, n in ((L.obs_teammate_slice, L.N_TEAMMATE_SLOTS),
                  (L.obs_target_slice, L.MAX_TARGETS),
                  (L.state_agent_slice, L.MAX_AGENTS),
                  (L.state_target_slice, L.MAX_TARGETS),
                  (L.state_lang_slice, L.MAX_AGENTS)):
        for bad in (-1, n):
            try:
                fn(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("%s(%d) should reject" % (fn.__name__, bad))


# ---------------------------------------------------------------------------
# Sub-field offsets tile their blocks (order fixed once, spec 1.5/1.6)
# ---------------------------------------------------------------------------

def _assert_fields_tile(fields, block_w):
    """(offset, width) fields are in order, contiguous, and fill the block."""
    cursor = 0
    for off, w in fields:
        assert off == cursor, (off, cursor)
        cursor += w
    assert cursor == block_w, (cursor, block_w)


def test_ego_field_offsets_tile():
    _assert_fields_tile([(L.EGO_POS_OFF, 2),
                         (L.EGO_LATCHED_OFF, 1),
                         (L.EGO_LATCH_SLOT_OFF, L.LATCH_SLOT_ONEHOT_W),
                         (L.EGO_LATCH_TIME_OFF, 1),
                         (L.EGO_AGENT_ID_OFF, L.AGENT_ID_ONEHOT_W)],
                        L.OBS_EGO_W)


def test_teammate_field_offsets_tile():
    _assert_fields_tile([(L.TM_PRESENT_OFF, 1),
                         (L.TM_POS_OFF, 2),
                         (L.TM_LATCHED_OFF, 1),
                         (L.TM_LATCH_SLOT_OFF, L.LATCH_SLOT_ONEHOT_W),
                         (L.TM_LATCH_TIME_OFF, 1)],
                        L.OBS_TEAMMATE_W)


def test_target_field_offsets_tile():
    _assert_fields_tile([(L.TGT_PRESENT_OFF, 1),
                         (L.TGT_POS_OFF, 2),
                         (L.TGT_OCCUPIED_OFF, 1),
                         (L.TGT_OCCUPIED_BY_EGO_OFF, 1)],
                        L.OBS_TARGET_W)


# ---------------------------------------------------------------------------
# Language slices: per-agent width, and LAST in both layouts (spec 1.5/1.6 --
# the blind ablation zeroes these slices; they must exist in every arm)
# ---------------------------------------------------------------------------

def test_lang_slices_widths_and_position():
    assert L.LANG_SLICE.stop - L.LANG_SLICE.start == L.LANG_DIM
    assert L.LANG_SLICE.stop == L.OBS_DIM
    assert L.STATE_LANG_SLICE.stop - L.STATE_LANG_SLICE.start == \
        L.MAX_AGENTS * L.LANG_DIM == 128
    assert L.STATE_LANG_SLICE.stop == L.STATE_DIM


# ---------------------------------------------------------------------------
# Normalisation helper (spec 1.5: 2x/(R-1) - 1 in [-1, 1])
# ---------------------------------------------------------------------------

def test_norm_pos_scalar():
    assert L.norm_pos(0, L.R) == -1.0
    assert L.norm_pos(L.R - 1, L.R) == 1.0
    assert L.norm_pos((L.R - 1) / 2.0, L.R) == 0.0
    # strictly monotone over the grid, always inside [-1, 1]
    vals = [L.norm_pos(x, L.C) for x in range(L.C)]
    assert all(-1.0 <= v <= 1.0 for v in vals)
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_norm_pos_tensor():
    # Must promote integer tensors to float (2.0 * int -> float), so
    # build_obs/build_state can feed int16 position buffers straight in.
    import torch
    out = L.norm_pos(torch.tensor([0, 11], dtype=torch.int16), L.R)
    assert out.is_floating_point()
    assert torch.equal(out, torch.tensor([-1.0, 1.0]))


# ---------------------------------------------------------------------------
# Module self-check runs (it already ran once at import; re-run explicitly)
# ---------------------------------------------------------------------------

def test_assert_layout_passes():
    L.assert_layout()


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback
    import unittest

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS  " + name)
        except unittest.SkipTest as exc:
            n_skip += 1
            print("SKIP  {}  ({})".format(name, exc))
        except Exception:
            n_fail += 1
            print("FAIL  " + name)
            traceback.print_exc()
    print("-" * 60)
    print("{} passed, {} failed, {} skipped".format(
        len(tests) - n_fail - n_skip, n_fail, n_skip))
    sys.exit(1 if n_fail else 0)
