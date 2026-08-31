"""Single source of layout truth for the Team Listen observation and critic state.

Authoritative reference: docs/M1_SPEC.md sections 1.3 (world-state constants),
1.5 (observation layout, OBS_DIM = 376) and 1.6 (critic state, STATE_DIM = 641).

Rules enforced here (and relied on everywhere else):

* Entity blocks are FIXED-CAPACITY padded slots with presence masks
  (MAX_AGENTS = 4, MAX_TARGETS = 3), so OBS_DIM/STATE_DIM never change when
  N grows to 4 or a third target is added -- those become config/data changes.
* The blind ablation is a SLICE ZEROING, never a shape change: `LANG_SLICE`
  and `STATE_LANG_SLICE` exist (and are zero-filled) in every arm.
* Flat Box only -- no gym Dict, no Discrete inside the observation (spec 1.5).
* Positions are normalised as 2*x/(R-1) - 1 in [-1, 1]; times as t/T_DECISION.

This module imports NOTHING (not even torch) so every test, the bank builder
and the numpy parity harness can import it without a GPU or SimulationApp.

`build_obs()` / `build_state()` (spec section 7) are implemented alongside
`fleet_env.py` -- they need live world-state tensors; only the layout
arithmetic lives here.
"""

# ---------------------------------------------------------------------------
# World-state constants (spec 1.3) -- [FIXED: layout foreclosed three
# downstream arms]: capacities are frozen at these values.
# ---------------------------------------------------------------------------

R = 12                      # grid rows
C = 12                      # grid cols
P = R * C                   # cells per plane = 144
N_AGENTS = 2                # live agents in M1 (slots beyond this are padding)
MAX_AGENTS = 4              # padded agent capacity -- NEVER changes in-study
MAX_TARGETS = 3             # padded target capacity -- NEVER changes in-study
LANG_DIM = 32               # per-agent projected instruction vector width
T_DECISION = 128            # decision steps per episode (spec 1.4: exactly 128)

# One-hot widths.
# Agent-id one-hot is MAX_AGENTS wide.
AGENT_ID_ONEHOT_W = MAX_AGENTS                      # = 4
# The spec fixes the latch-slot one-hot at width 4 (spec 1.5 ego/teammate
# blocks: "latch-slot one-hot(4)").  It covers target slots 0..2
# (MAX_TARGETS = 3) with one spare channel so a fourth target slot is a data
# change, matching the padded-capacity philosophy; all-zeros <=> unlatched
# (latch_slot == -1).
LATCH_SLOT_ONEHOT_W = 4


def _take(cursor, width):
    """Advance a running cursor; returns (slice, new_cursor)."""
    return slice(cursor, cursor + width), cursor + width


# ---------------------------------------------------------------------------
# Per-agent actor observation -- OBS_DIM = 376 (spec 1.5)
# ---------------------------------------------------------------------------

# Block widths.
OBS_EGO_W = 2 + 1 + LATCH_SLOT_ONEHOT_W + 1 + AGENT_ID_ONEHOT_W        # = 12
#   (r,c) | latched | latch-slot one-hot(4) | latch-time/T | agent-id one-hot(4)
OBS_TEAMMATE_W = 1 + 2 + 1 + LATCH_SLOT_ONEHOT_W + 1                   # = 9
#   present | (r,c) | latched | latch-slot one-hot(4) | latch-time/T
N_TEAMMATE_SLOTS = MAX_AGENTS - 1                                      # = 3
OBS_TARGET_W = 1 + 2 + 1 + 1                                           # = 5
#   present | (r,c) | occupied | occupied-by-ego
OBS_TIME_W = 2
#   t/T | time-since-first-latch/T

_o = 0
OBS_KNOWN_FREE_SLICE, _o = _take(_o, P)                    # [0:144)
OBS_KNOWN_OBS_SLICE, _o = _take(_o, P)                     # [144:288)
OBS_EGO_SLICE, _o = _take(_o, OBS_EGO_W)                   # [288:300)
OBS_TEAMMATES_SLICE, _o = _take(_o, N_TEAMMATE_SLOTS * OBS_TEAMMATE_W)  # [300:327)
OBS_TARGETS_SLICE, _o = _take(_o, MAX_TARGETS * OBS_TARGET_W)           # [327:342)
OBS_TIME_SLICE, _o = _take(_o, OBS_TIME_W)                 # [342:344)
LANG_SLICE, _o = _take(_o, LANG_DIM)                       # [344:376)
OBS_DIM = _o                                               # = 376

# Ego-block sub-offsets, RELATIVE to OBS_EGO_SLICE.start (order per spec 1.5).
EGO_POS_OFF = 0                 # width 2: (r, c) normalised
EGO_LATCHED_OFF = 2             # width 1
EGO_LATCH_SLOT_OFF = 3          # width 4: one-hot, all-zero when unlatched
EGO_LATCH_TIME_OFF = 7          # width 1: latch_time / T_DECISION
EGO_AGENT_ID_OFF = 8            # width 4: one-hot over MAX_AGENTS

# Teammate-block sub-offsets, RELATIVE to each 9-wide teammate slot.
TM_PRESENT_OFF = 0              # width 1
TM_POS_OFF = 1                  # width 2
TM_LATCHED_OFF = 3              # width 1
TM_LATCH_SLOT_OFF = 4           # width 4
TM_LATCH_TIME_OFF = 8           # width 1

# Target-block sub-offsets, RELATIVE to each 5-wide target slot.
TGT_PRESENT_OFF = 0             # width 1
TGT_POS_OFF = 1                 # width 2
TGT_OCCUPIED_OFF = 3            # width 1
TGT_OCCUPIED_BY_EGO_OFF = 4     # width 1


def obs_teammate_slice(slot):
    """Absolute obs slice of teammate slot `slot` (0..N_TEAMMATE_SLOTS-1).

    Teammate slots enumerate the non-ego agent slots in ascending agent-index
    order; absent slots (index >= N_AGENTS) are zero with present = 0.
    """
    assert 0 <= slot < N_TEAMMATE_SLOTS
    start = OBS_TEAMMATES_SLICE.start + slot * OBS_TEAMMATE_W
    return slice(start, start + OBS_TEAMMATE_W)


def obs_target_slice(slot):
    """Absolute obs slice of target slot `slot` (0..MAX_TARGETS-1)."""
    assert 0 <= slot < MAX_TARGETS
    start = OBS_TARGETS_SLICE.start + slot * OBS_TARGET_W
    return slice(start, start + OBS_TARGET_W)


# ---------------------------------------------------------------------------
# Centralised critic state -- STATE_DIM = 641 (spec 1.6), _get_states()
# hand-written; NEVER state_space = -1 auto-concatenation.
# ---------------------------------------------------------------------------

# Block widths.  Spec 1.6 fixes the widths (agent block 13, target block 9
# "incl. occupier one-hot") and every slice boundary; the sub-field order
# inside the two entity blocks is fixed HERE, once, and build_state() must
# follow it.
STATE_AGENT_W = 1 + 2 + 1 + LATCH_SLOT_ONEHOT_W + 1 + AGENT_ID_ONEHOT_W  # = 13
#   present | (r,c) | latched | latch-slot one-hot(4) | latch-time/T
#   | agent-id one-hot(4)   (ego block plus a presence bit)
STATE_TARGET_W = 1 + 2 + 1 + AGENT_ID_ONEHOT_W + 1                       # = 9
#   present | (r,c) | occupied | occupier one-hot(4, over agent slots,
#   all-zero when unoccupied) | occupier latch-time/T
STATE_TIME_W = 2
#   t/T | time-of-first-latch/T

_s = 0
STATE_OBSTACLE_SLICE, _s = _take(_s, P)                    # [0:144) true occ (privileged)
STATE_KNOWN_FREE_SLICE, _s = _take(_s, P)                  # [144:288)
STATE_KNOWN_OBS_SLICE, _s = _take(_s, P)                   # [288:432)
STATE_AGENTS_SLICE, _s = _take(_s, MAX_AGENTS * STATE_AGENT_W)    # [432:484)
STATE_TARGETS_SLICE, _s = _take(_s, MAX_TARGETS * STATE_TARGET_W)  # [484:511)
STATE_TIME_SLICE, _s = _take(_s, STATE_TIME_W)             # [511:513)
STATE_LANG_SLICE, _s = _take(_s, MAX_AGENTS * LANG_DIM)    # [513:641), per-agent
STATE_DIM = _s                                             # = 641


def state_agent_slice(slot):
    """Absolute state slice of agent slot `slot` (0..MAX_AGENTS-1)."""
    assert 0 <= slot < MAX_AGENTS
    start = STATE_AGENTS_SLICE.start + slot * STATE_AGENT_W
    return slice(start, start + STATE_AGENT_W)


def state_target_slice(slot):
    """Absolute state slice of target slot `slot` (0..MAX_TARGETS-1)."""
    assert 0 <= slot < MAX_TARGETS
    start = STATE_TARGETS_SLICE.start + slot * STATE_TARGET_W
    return slice(start, start + STATE_TARGET_W)


def state_lang_slice(agent_slot):
    """Absolute state slice of agent `agent_slot`'s 32-d language vector."""
    assert 0 <= agent_slot < MAX_AGENTS
    start = STATE_LANG_SLICE.start + agent_slot * LANG_DIM
    return slice(start, start + LANG_DIM)


# ---------------------------------------------------------------------------
# Derived arm dimensions (spec 4.1): full-state arms (Blind/Symbol/Leaky with
# cfg.actor_observes_state = True) observe the 641-d state plus their own
# agent-id one-hot -> 645.
# ---------------------------------------------------------------------------

FULL_STATE_OBS_DIM = STATE_DIM + AGENT_ID_ONEHOT_W         # = 645


# ---------------------------------------------------------------------------
# Normalisation helper (spec 1.5): positions map to [-1, 1].
# ---------------------------------------------------------------------------

def norm_pos(x, extent):
    """2*x/(extent-1) - 1.  Works on python numbers and torch tensors alike
    (2.0 * int-tensor promotes to float).  `extent` is R for rows, C for cols.
    """
    return 2.0 * x / (extent - 1) - 1.0


# ---------------------------------------------------------------------------
# Layout self-check -- runs at import; test_slices.py re-checks it under
# pytest.  Any edit that breaks the arithmetic fails at import, not mid-run.
# ---------------------------------------------------------------------------

def assert_layout():
    """Assert the slice tables are gap-free, overlap-free and spec-exact."""
    obs_expected = [
        (OBS_KNOWN_FREE_SLICE, 0, 144),
        (OBS_KNOWN_OBS_SLICE, 144, 288),
        (OBS_EGO_SLICE, 288, 300),
        (OBS_TEAMMATES_SLICE, 300, 327),
        (OBS_TARGETS_SLICE, 327, 342),
        (OBS_TIME_SLICE, 342, 344),
        (LANG_SLICE, 344, 376),
    ]
    state_expected = [
        (STATE_OBSTACLE_SLICE, 0, 144),
        (STATE_KNOWN_FREE_SLICE, 144, 288),
        (STATE_KNOWN_OBS_SLICE, 288, 432),
        (STATE_AGENTS_SLICE, 432, 484),
        (STATE_TARGETS_SLICE, 484, 511),
        (STATE_TIME_SLICE, 511, 513),
        (STATE_LANG_SLICE, 513, 641),
    ]
    for table, dim in ((obs_expected, OBS_DIM), (state_expected, STATE_DIM)):
        cursor = 0
        for sl, start, stop in table:
            # spec-exact boundaries
            assert sl.start == start and sl.stop == stop, (sl, start, stop)
            # contiguity: no gap, no overlap
            assert sl.start == cursor, (sl, cursor)
            cursor = sl.stop
        assert cursor == dim, (cursor, dim)

    assert OBS_DIM == 376, OBS_DIM
    assert STATE_DIM == 641, STATE_DIM
    assert FULL_STATE_OBS_DIM == 645, FULL_STATE_OBS_DIM
    assert OBS_EGO_W == 12 and OBS_TEAMMATE_W == 9 and OBS_TARGET_W == 5
    assert STATE_AGENT_W == 13 and STATE_TARGET_W == 9
    assert N_TEAMMATE_SLOTS * OBS_TEAMMATE_W == 27
    assert MAX_TARGETS * OBS_TARGET_W == 15
    assert MAX_AGENTS * STATE_AGENT_W == 52
    assert MAX_TARGETS * STATE_TARGET_W == 27
    assert MAX_AGENTS * LANG_DIM == 128
    # per-slot helpers tile their parent slices exactly
    assert obs_teammate_slice(0).start == OBS_TEAMMATES_SLICE.start
    assert obs_teammate_slice(N_TEAMMATE_SLOTS - 1).stop == OBS_TEAMMATES_SLICE.stop
    assert obs_target_slice(MAX_TARGETS - 1).stop == OBS_TARGETS_SLICE.stop
    assert state_agent_slice(MAX_AGENTS - 1).stop == STATE_AGENTS_SLICE.stop
    assert state_target_slice(MAX_TARGETS - 1).stop == STATE_TARGETS_SLICE.stop
    assert state_lang_slice(MAX_AGENTS - 1).stop == STATE_LANG_SLICE.stop
    # ego sub-offsets tile the 12-wide ego block
    assert EGO_AGENT_ID_OFF + AGENT_ID_ONEHOT_W == OBS_EGO_W
    assert TM_LATCH_TIME_OFF + 1 == OBS_TEAMMATE_W
    assert TGT_OCCUPIED_BY_EGO_OFF + 1 == OBS_TARGET_W


assert_layout()


__all__ = [
    # world-state constants
    "R", "C", "P", "N_AGENTS", "MAX_AGENTS", "MAX_TARGETS", "LANG_DIM",
    "T_DECISION", "AGENT_ID_ONEHOT_W", "LATCH_SLOT_ONEHOT_W",
    # obs layout
    "OBS_DIM", "OBS_KNOWN_FREE_SLICE", "OBS_KNOWN_OBS_SLICE", "OBS_EGO_SLICE",
    "OBS_TEAMMATES_SLICE", "OBS_TARGETS_SLICE", "OBS_TIME_SLICE", "LANG_SLICE",
    "OBS_EGO_W", "OBS_TEAMMATE_W", "OBS_TARGET_W", "OBS_TIME_W",
    "N_TEAMMATE_SLOTS", "obs_teammate_slice", "obs_target_slice",
    "EGO_POS_OFF", "EGO_LATCHED_OFF", "EGO_LATCH_SLOT_OFF",
    "EGO_LATCH_TIME_OFF", "EGO_AGENT_ID_OFF",
    "TM_PRESENT_OFF", "TM_POS_OFF", "TM_LATCHED_OFF", "TM_LATCH_SLOT_OFF",
    "TM_LATCH_TIME_OFF",
    "TGT_PRESENT_OFF", "TGT_POS_OFF", "TGT_OCCUPIED_OFF",
    "TGT_OCCUPIED_BY_EGO_OFF",
    # state layout
    "STATE_DIM", "STATE_OBSTACLE_SLICE", "STATE_KNOWN_FREE_SLICE",
    "STATE_KNOWN_OBS_SLICE", "STATE_AGENTS_SLICE", "STATE_TARGETS_SLICE",
    "STATE_TIME_SLICE", "STATE_LANG_SLICE",
    "STATE_AGENT_W", "STATE_TARGET_W", "STATE_TIME_W",
    "state_agent_slice", "state_target_slice", "state_lang_slice",
    # derived
    "FULL_STATE_OBS_DIM",
    # helpers
    "norm_pos", "assert_layout",
]
