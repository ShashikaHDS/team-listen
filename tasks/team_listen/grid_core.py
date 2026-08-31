"""Pure-torch transition core for the Team Listen grid fleet environment.

Authoritative reference: docs/M1_SPEC.md sections 1.2 (module split and
function signatures), 1.3 (constants), 1.8 (closed-form N=2 conflict
resolution), 1.9 (fog-of-war reveal), 1.10 (bank-indexed action slip),
1.11 (latch rule) and 1.12 (instruction-free matching potential).

Grid semantics are ported from ``reference/env_paper.py``
(``RendezvousEnv.step`` + ``RendezvousEnv._resolve_conflicts`` +
``RendezvousEnv._reveal``), verified against ``reference/test_env_paper.py``.
Each function's docstring names the reference behaviour it mirrors.

Design rules enforced here (spec 1.2):

* PURE TORCH -- this module imports only ``torch``, so CPU unit tests and
  the numpy differential-parity harness run in seconds without
  ``SimulationApp``, and the conflict-resolution port (the single most
  likely silent divergence from the reference) is directly testable.
* Every function is batched over the env dimension E and works on
  arbitrary device (CPU or CUDA); no host<->device syncs on the hot path.
* ``matching_potential`` takes NO ``assign`` / ``instr_id`` / ``lang_vec``
  argument [FIXED: potential-purity]: the shaping potential must be
  instruction-free, otherwise the reward is a live instruction-leak
  channel (spec 1.2 / 1.12; statically asserted by
  ``tests/test_potential_purity.py``).

Action encoding (bit-identical to ``RendezvousEnv._MOVES``):
0 = up (row-1), 1 = down (row+1), 2 = left (col-1), 3 = right (col+1),
4 = stay.
"""

import torch

# ---------------------------------------------------------------------------
# Constants (spec 1.3)
# ---------------------------------------------------------------------------

UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
N_ACTIONS = 5

#: Sentinel in ``bank.slip`` meaning "no slip this step: keep the policy's
#: action" (spec 1.10; any value 0..4 is the forced replacement action).
NO_SLIP = 5

#: (5, 2) action -> (drow, dcol), bit-identical to ``RendezvousEnv._MOVES``.
DELTAS = torch.tensor(
    [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=torch.long
)

# Per-device cache so the hot path never re-uploads the 5x2 table.
_DELTA_CACHE = {}


def _deltas_on(device):
    """Return ``DELTAS`` resident on ``device`` (cached per device)."""
    key = str(device)
    cached = _DELTA_CACHE.get(key)
    if cached is None:
        cached = DELTAS.to(device)
        _DELTA_CACHE[key] = cached
    return cached


def chebyshev_offsets(radius=1):
    """The (2r+1)^2 x 2 Chebyshev window offsets, row-major.

    Mirrors the cell set of ``RendezvousEnv._reveal``'s window slice
    (spec 1.3 ``_reveal_offsets``; ``r = cfg.lidar_radius = 1`` in M1).
    """
    rng = torch.arange(-int(radius), int(radius) + 1, dtype=torch.long)
    return torch.cartesian_prod(rng, rng)


#: Default 9x2 reveal window for lidar_radius = 1 (CPU; callers move to device).
REVEAL_OFFSETS = chebyshev_offsets(1)


# ---------------------------------------------------------------------------
# apply_slip (spec 1.10)
# ---------------------------------------------------------------------------

def apply_slip(actions, slip_row):
    """Replace policy actions with the bank's precomputed slip draws.

    Implements the [FIXED: no physics seed] action-slip channel of spec
    1.10: with probability epsilon the executed action is replaced by a
    uniformly drawn action.  All draws are precomputed offline into
    ``bank.slip[k, stream, t, n]`` (uint8; ``NO_SLIP`` = 5 keeps the
    policy's action, any 0..4 is the forced replacement), so the env stays
    exactly reproducible and "same physics" is the equality
    ``scenario_id[i] == scenario_id[j] and slip_stream[i] == slip_stream[j]``.
    This has no counterpart in ``reference/env_paper.py`` (that env is
    seeded-RNG stochastic only at reset).

    Args:
        actions:  (E, N) integer tensor of policy actions in [0, 5).
        slip_row: (E, N) or (E, MAX_AGENTS>=N) integer tensor -- the bank
                  row for this (scenario, stream, t); trailing padded agent
                  columns beyond N are ignored.

    Returns:
        (E, N) int64 tensor of executed actions.
    """
    a = actions.long()
    s = slip_row[:, : a.shape[1]].long()
    return torch.where(s == NO_SLIP, a, s)


# ---------------------------------------------------------------------------
# step_positions (spec 1.8)
# ---------------------------------------------------------------------------

def step_positions(cur, actions, occ, latched, bounds):
    """Closed-form N=2 move proposal + conflict resolution.

    Faithful port of ``RendezvousEnv.step``'s proposal phase followed by
    ``RendezvousEnv._resolve_conflicts`` (reference lines: np.clip of the
    target, obstacle revert with per-robot ``obstacle_hit`` flag, then the
    iterative revert of (a) same-target, (b) swap, (c) move-into-a-cell-
    whose-occupant-is-not-leaving conflicts).  At N=2 the reference
    ``while changed`` fixpoint provably terminates in one pass -- cases (a)
    and (b) revert both robots; case (c) reverts only the mover, whose
    counterpart is already stationary -- so the masks below are exact
    (spec 1.8; general N is OPEN(1), hence the hard N==2 assert).

    Reference semantics mirrored, per the verified reference tests:

    * off-grid move -> clamped to stay, NO flag  (== ``np.clip``);
    * obstacle target -> reverted, ``hit_obstacle`` flag, NO robot flag
      (``test_obstacle_collision``); the reverted robot then counts as
      stationary for the robot-conflict masks (the reference cascade's
      "or was reverted" clause);
    * both robots targeting one cell -> both reverted + flagged
      (``test_same_target_conflict``);
    * swap -> both reverted + flagged (``test_swap_conflict``);
    * moving into a robot that stays (or was reverted) -> ONLY the mover
      reverted + flagged (``test_move_into_stationary``);
    * following a robot into the cell it vacates is LEGAL -- no mask fires,
      convoy motion succeeds with zero collisions
      (``test_follow_vacated_cell_allowed``).

    Latch extension (spec 1.11, no reference counterpart): a latched robot
    is immovable and remains a blocking obstacle.  Its target is forced
    back to its current cell BEFORE the movement masks, so it is
    stationary for cases (a)-(c).  Callers should already have overwritten
    latched robots' actions to STAY (spec 1.11); this line is the
    in-kernel guarantee.  Note ``hit_obstacle`` is computed on the clamped
    pre-latch target, exactly as in the spec 1.8 pseudocode.

    Args:
        cur:     (E, 2, 2) integer tensor of current (row, col) positions.
        actions: (E, 2) integer tensor of executed actions (post-slip).
        occ:     (E, R, C) int8 ground-truth grid, 0 free / 1 obstacle.
        latched: (E, 2) bool.
        bounds:  (R, C) python ints -- grid extent, mirroring the
                 reference's ``np.clip(., 0, rows-1 / cols-1)``.

    Returns:
        nxt:          (E, 2, 2) tensor, dtype of ``cur`` -- post-conflict
                      positions.
        hit_obstacle: (E, 2) bool -- per-robot obstacle-collision flag.
        hit_robot:    (E, 2) bool -- per-robot robot-collision flag.
    """
    E, n_agents = cur.shape[0], cur.shape[1]
    assert n_agents == 2, (
        "step_positions is the closed-form N=2 port (spec 1.8 / OPEN(1)); "
        "got n_agents=%d" % n_agents
    )
    rows, cols = int(bounds[0]), int(bounds[1])
    assert occ.shape[-2] == rows and occ.shape[-1] == cols, (
        "bounds %r inconsistent with occ shape %r" % ((rows, cols), tuple(occ.shape))
    )
    device = cur.device
    cur64 = cur.long()

    # -- move proposal: delta lookup + per-axis clamp (== np.clip) ---------
    delta = _deltas_on(device)[actions.long()]           # (E, 2, 2)
    tgt = cur64 + delta
    tgt[..., 0].clamp_(0, rows - 1)                      # off-grid -> stay,
    tgt[..., 1].clamp_(0, cols - 1)                      # no flags

    # -- obstacle pre-pass: revert + flag, no robot flag -------------------
    env_idx = torch.arange(E, device=device).unsqueeze(1)          # (E, 1)
    hit_obstacle = occ[env_idx, tgt[..., 0], tgt[..., 1]] != 0     # (E, 2)
    tgt = torch.where(hit_obstacle.unsqueeze(-1), cur64, tgt)

    # -- latched robots immovable (spec 1.11) ------------------------------
    tgt = torch.where(latched.unsqueeze(-1), cur64, tgt)

    # -- robot-robot conflicts, one exact pass at N=2 ----------------------
    mov = (tgt != cur64).any(-1)                                   # (E, 2)
    same = (tgt[:, 0] == tgt[:, 1]).all(-1) & mov[:, 0] & mov[:, 1]
    swap = ((tgt[:, 0] == cur64[:, 1]).all(-1)
            & (tgt[:, 1] == cur64[:, 0]).all(-1)
            & mov[:, 0] & mov[:, 1])
    into0 = (tgt[:, 1] == cur64[:, 0]).all(-1) & mov[:, 1] & ~mov[:, 0]
    into1 = (tgt[:, 0] == cur64[:, 1]).all(-1) & mov[:, 0] & ~mov[:, 1]
    rev0 = same | swap | into1          # robot 0 reverted
    rev1 = same | swap | into0          # robot 1 reverted
    hit_robot = torch.stack([rev0, rev1], dim=1)                   # (E, 2)

    nxt = torch.where(hit_robot.unsqueeze(-1), cur64, tgt).to(cur.dtype)
    return nxt, hit_obstacle, hit_robot


# ---------------------------------------------------------------------------
# reveal (spec 1.9)
# ---------------------------------------------------------------------------

def reveal(known_free, known_obs, occ, pos, offsets):
    """In-place fog-of-war reveal around each robot -- a batched scatter.

    Mirrors ``RendezvousEnv._reveal``: ``lidar_radius`` in the reference is
    a Chebyshev WINDOW SLICE (``known_map[x0:x1, y0:y1] = grid[x0:x1, y0:y1]``
    with ``max(0,.)/min(R,.)`` truncation), not a raycast.  Clamping the
    offset cells to the grid is equivalent to the reference's window
    truncation -- edge duplicates scatter the same value twice (spec 1.9).
    The reference's single ternary ``known_map`` (-1/0/1) is split into the
    two boolean planes ``known_free`` / ``known_obs`` of spec 1.3
    (unknown == both False), matching the two observation planes.

    Args:
        known_free: (E, R, C) bool -- updated in place.
        known_obs:  (E, R, C) bool -- updated in place.
        occ:        (E, R, C) int8 ground truth, 0 free / 1 obstacle.
        pos:        (E, N, 2) integer positions to reveal around.  Pass
                    only LIVE agent slots -- padded slots would reveal
                    around their placeholder cells.
        offsets:    (K, 2) integer window offsets (``REVEAL_OFFSETS`` for
                    the M1 radius-1 window), any device (moved as needed).

    Returns:
        None (scatters into ``known_free`` / ``known_obs`` in place; both
        must be contiguous -- ``.view`` hard-fails otherwise rather than
        silently writing into a copy).
    """
    E, rows, cols = occ.shape
    off = offsets.to(device=pos.device).long()
    cells = pos.long().unsqueeze(2) + off                # (E, N, K, 2)
    cells[..., 0].clamp_(0, rows - 1)
    cells[..., 1].clamp_(0, cols - 1)
    idx = (cells[..., 0] * cols + cells[..., 1]).view(E, -1)       # (E, N*K)
    vals = occ.view(E, -1).gather(1, idx)                          # int8
    known_free.view(E, -1).scatter_(1, idx, vals == 0)
    known_obs.view(E, -1).scatter_(1, idx, vals == 1)


# ---------------------------------------------------------------------------
# latch_update (spec 1.11)
# ---------------------------------------------------------------------------

def latch_update(pos, target_cells, target_valid, latched, latch_slot,
                 latch_time, t):
    """In-place latch rule: a robot on a target cell latches, permanently.

    Spec 1.11 (no counterpart in the reference rendezvous env): a robot
    whose POST-CONFLICT cell equals a valid target cell latches --
    ``latched = True``, ``latch_slot = j``, ``latch_time = t``.  The latch
    is absorbing (OPEN(2) default): already-latched robots are excluded,
    so ``latch_slot`` / ``latch_time`` keep their first-latch values.
    Making the latched robot stay and block is enforced elsewhere
    (action overwrite in the env + the latch line in ``step_positions``).

    Args:
        pos:          (E, N, 2) integer post-conflict positions (LIVE agent
                      slots only -- padded slots could alias a target cell).
        target_cells: (E, MAX_TARGETS, 2) integer station cells.
        target_valid: (E, MAX_TARGETS) bool presence mask.
        latched:      (E, N) bool     -- updated in place.
        latch_slot:   (E, N) int8     -- updated in place (-1 = unlatched).
        latch_time:   (E, N) int16    -- updated in place (-1 = unlatched).
        t:            python int, or an integer tensor broadcastable over
                      (E, N) -- e.g. the per-env ``episode_length_buf``
                      (E,), which is unsqueezed to (E, 1) here.

    Returns:
        None (in-place).
    """
    # (E, N, MAX_TARGETS): robot i sits exactly on valid target j
    on = (pos.long().unsqueeze(2) == target_cells.long().unsqueeze(1)).all(-1)
    on = on & target_valid.unsqueeze(1)
    on_target = on.any(-1)                               # (E, N)
    slot = on.long().argmax(-1)                          # first matching slot
    new = on_target & ~latched                           # first latch only

    if torch.is_tensor(t):
        tt = t.to(device=latch_time.device, dtype=latch_time.dtype)
        if tt.dim() == 1:
            tt = tt.unsqueeze(1)                         # (E, 1) over agents
    else:
        tt = torch.tensor(int(t), device=latch_time.device,
                          dtype=latch_time.dtype)

    latched |= new
    latch_slot.copy_(torch.where(new, slot.to(latch_slot.dtype), latch_slot))
    latch_time.copy_(torch.where(new, tt, latch_time))


# ---------------------------------------------------------------------------
# matching_potential (spec 1.12)
# ---------------------------------------------------------------------------

def matching_potential(dist_field, pos, target_valid):
    """Phi = -(min-cost perfect matching of latch-aware BFS distances).

    The shaping potential of spec 1.12: each robot is assigned a DISTINCT
    valid target so as to minimise the summed latch-aware BFS distance
    (``bank.dist_field[k, j]`` = BFS from target j with every other target
    cell treated as an obstacle, spec 1.10), and Phi is minus that minimum
    cost.  Used by ``rewards.py`` as ``lambda * (gamma * Phi_t - Phi_{t-1})``
    with ``Phi(terminal) == 0`` handled by the caller.

    INSTRUCTION-FREE BY CONSTRUCTION [FIXED: Phi signature contradiction]:
    no ``assign`` / ``instr_id`` / ``lang_vec`` argument exists, so the
    reward cannot be a leak channel; the min-matching deliberately encodes
    the geometric shortcut pressure the audit is about (spec 1.12).

    Closed form for N=2 (mirrors ``step_positions``'s N=2 scope): the
    matching is an exact enumeration over ordered target pairs (j0 != j1).
    Entries that are negative (an unreachable-cell sentinel in the stored
    int16 field) and invalid target slots cost +inf, so they are never
    selected; the bank guarantees >= 2 valid, mutually reachable targets
    (spec 2.1/3.1), so the result is finite on every legal state.

    Args:
        dist_field:   (E, MAX_TARGETS, R, C) integer latch-aware BFS field.
        pos:          (E, 2, 2) integer robot positions (LIVE slots only).
        target_valid: (E, MAX_TARGETS) bool presence mask.

    Returns:
        (E,) float32 tensor: Phi (non-positive on reachable states).
    """
    E, n_targets = dist_field.shape[0], dist_field.shape[1]
    rows, cols = dist_field.shape[2], dist_field.shape[3]
    n_agents = pos.shape[1]
    assert n_agents == 2, (
        "matching_potential enumerates the N=2 matching in closed form "
        "(spec 1.8/1.12 scope); got n_agents=%d" % n_agents
    )

    # d[e, j, i] = dist_field[e, j, pos[e, i, 0], pos[e, i, 1]]
    idx = pos[..., 0].long() * cols + pos[..., 1].long()           # (E, N)
    flat = dist_field.reshape(E, n_targets, rows * cols)
    d = flat.gather(2, idx.unsqueeze(1).expand(E, n_targets, n_agents))
    d = d.permute(0, 2, 1).float()                                 # (E, N, T)

    inf = float("inf")
    d = torch.where(d < 0, torch.full_like(d, inf), d)             # sentinel
    d = torch.where(target_valid.unsqueeze(1), d, torch.full_like(d, inf))

    # cost[e, j0, j1] = d(robot0, j0) + d(robot1, j1); forbid j0 == j1
    cost = d[:, 0].unsqueeze(2) + d[:, 1].unsqueeze(1)             # (E, T, T)
    diag = torch.eye(n_targets, dtype=torch.bool, device=cost.device)
    cost = cost.masked_fill(diag, inf)
    return -cost.reshape(E, -1).min(dim=1).values


__all__ = [
    "UP", "DOWN", "LEFT", "RIGHT", "STAY", "N_ACTIONS", "NO_SLIP",
    "DELTAS", "REVEAL_OFFSETS", "chebyshev_offsets",
    "apply_slip", "step_positions", "reveal", "latch_update",
    "matching_potential",
]
