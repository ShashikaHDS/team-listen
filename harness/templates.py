"""Template grammar for "Does the Team Actually Listen?" (M1_SPEC 2.2, 3.2, 5.3, 5.4).

Produces the full deterministic sentence inventory consumed by
``scripts/build_lang_cache.py``:

* 240 single-axis sentences
    - role_binding: classes ``RB0`` (robot_0 -> left) / ``RB1`` (robot_0 -> right),
      60 sentences per class (50 train + 10 held-out).
    - precedence: classes ``PR0`` (robot_0 first) / ``PR1`` (robot_1 first),
      60 sentences per class (50 train + 10 held-out).
* 240 composed sentences (role binding x precedence in one instruction),
  4 classes x 60 sentences, built now so month 2 is a data change only.
  One composition, ``HELD_OUT_COMPOSITION``, is held out from training
  entirely (see ``is_trainable``); the family-based ``split`` field is
  unaffected so per-class 50/10 family accounting stays uniform.

Grammar (5 x 4 x 6 x 4, M1_SPEC 2.2):
  5 agent referring-expression styles x 4 verb phrases x
  6 station referring-expression styles (role binding / composed) or
  6 ordering-connective styles (precedence) x 4 sentential frames.

A **family** is a (frame, station-ref style) pair for role binding/composed
and a (frame, connective style) pair for precedence: 4 x 6 = 24 families,
split **18 train / 6 held-out**, with the held-out set covering every
station/connective style exactly once. Sentences are sampled
family-disjointly: train sentences come only from train families and
held-out sentences only from held-out families.

Every sentence belongs to exactly one **minimal pair**: the identical slot
fill with only the role-bearing token(s) flipped -- same family, same
frame, same agent/verb styles. Both flip types per variant (M1_SPEC 3.2
[FIXED]) are exactly counterbalanced within each split:

  role_binding : station | agent        (25/25 train, 5/5 held-out)
  precedence   : agent   | connective   (25/25 train, 5/5 held-out)
  composed     : station | connective   (50/50 train, 10/10 held-out)

The module is pure stdlib (no torch/numpy) so CPU-only tests import it
without any heavyweight dependency, and ``build_all()`` is fully
deterministic (fixed sampling seed, no wall-clock or hash-salt input).
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Identifiers and fixed cardinalities
# ---------------------------------------------------------------------------

ROLE_BINDING = "role_binding"
PRECEDENCE = "precedence"
COMPOSED = "composed"
VARIANTS = (ROLE_BINDING, PRECEDENCE, COMPOSED)

FLIP_STATION = "station"
FLIP_AGENT = "agent"
FLIP_CONNECTIVE = "connective"
FLIP_TYPES = (FLIP_STATION, FLIP_AGENT, FLIP_CONNECTIVE)

N_FRAMES = 4
N_STYLES = 6
N_FAMILIES = N_FRAMES * N_STYLES            # 24
N_TRAIN_FAMILIES = 18
N_HELDOUT_FAMILIES = 6
TRAIN_PER_CLASS = 50
HELDOUT_PER_CLASS = 10
SENT_PER_CLASS = TRAIN_PER_CLASS + HELDOUT_PER_CLASS   # 60

N_SINGLE_AXIS = 2 * 2 * SENT_PER_CLASS      # 240 (two variants x two classes)
N_COMPOSED = 4 * SENT_PER_CLASS             # 240
N_SENTENCES = N_SINGLE_AXIS + N_COMPOSED    # 480

CLASS_NAMES_SINGLE = {
    ROLE_BINDING: ("RB0", "RB1"),
    PRECEDENCE: ("PR0", "PR1"),
}

# The composition (role_class, order_class) held out from training entirely
# (M1_SPEC 2.2 [FIXED]: "4 composed classes with one composition held out
# entirely"). The family-based `split` field is orthogonal to this.
HELD_OUT_COMPOSITION = (1, 1)

# Deterministic sampling seed; changing it changes the sampled sentences and
# therefore the lang-cache SHA, so it is part of the artifact identity.
_SAMPLING_SEED = 20260831
_VARIANT_CODE = {ROLE_BINDING: 1, PRECEDENCE: 2, COMPOSED: 3}

# ---------------------------------------------------------------------------
# Slot inventories (M1_SPEC 2.2 / 3.2)
# ---------------------------------------------------------------------------

# Each agent style gives the referring expressions for (robot_0, robot_1).
AGENT_REFS: Tuple[Tuple[str, str], ...] = (
    ("robot one", "robot two"),
    ("the first robot", "the second robot"),
    ("unit 1", "unit 2"),
    ("R1", "R2"),
    ("the lead robot", "the wing robot"),
)

VERBS_RB: Tuple[str, ...] = ("takes", "goes to", "docks at", "should occupy")

# Each station style gives the referring expressions for (left, right).
STATION_REFS: Tuple[Tuple[str, str], ...] = (
    ("the left station", "the right station"),
    ("the station on the left", "the station on the right"),
    ("the westward dock", "the eastward dock"),
    ("the left-hand bay", "the right-hand bay"),
    ("the port-side dock", "the starboard-side dock"),
    ("the leftmost berth", "the rightmost berth"),
)

# Precedence verb forms: (3rd-person singular, base, gerund).
VERBS_PR: Tuple[Tuple[str, str, str], ...] = (
    ("docks", "dock", "docking"),
    ("moves in", "move in", "moving in"),
    ("goes in", "go in", "going in"),
    ("pulls in", "pull in", "pulling in"),
)

# Ordering-connective styles. Element 0 of each pair is the forward form
# (the agent in the {P} slot docks FIRST); element 1 is the polarity-flipped
# form of the same style (the {P}-slot agent docks SECOND). The connective
# minimal pair toggles the form while leaving both agent tokens in place.
PR_CONNECTIVES: Tuple[Tuple[str, str], ...] = (
    ("{P} {V3} first, then {Q}", "{P} {V3} last, after {Q}"),
    ("{P} {V3} before {Q} does", "{P} {V3} after {Q} does"),
    ("{P} {V3} while {Q} is still waiting", "{P} {V3} only after {Q} has docked"),
    ("{P} {V3} straight away while {Q} waits",
     "{P} waits until {Q} is docked, then {V3}"),
    ("when {VG}, {P} precedes {Q}", "when {VG}, {P} follows {Q}"),
    ("{P} is to {VB} ahead of {Q}", "{P} is to {VB} behind {Q}"),
)

# Composed order tokens appear as the bigram "docks first" / "docks second";
# the connective flip in a composed pair swaps exactly those tokens.
_ORDER_WORDS = ("first", "second")


# ---------------------------------------------------------------------------
# Families and splits
# ---------------------------------------------------------------------------

def family_id(frame_idx: int, style_idx: int) -> int:
    """Family index of a (frame, station-or-connective style) pair."""
    return frame_idx * N_STYLES + style_idx


def family_frame_style(fid: int) -> Tuple[int, int]:
    """Inverse of :func:`family_id`."""
    return divmod(fid, N_STYLES)


# Held-out families: one per station/connective style, frames spread as
# evenly as 6-into-4 allows ([0,1,2,3,0,1]). Fixed, not sampled, so the
# split is stable across rebuilds and documented in one line.
HELDOUT_FAMILIES = frozenset(family_id(s % N_FRAMES, s) for s in range(N_STYLES))
TRAIN_FAMILIES = frozenset(range(N_FAMILIES)) - HELDOUT_FAMILIES

assert len(HELDOUT_FAMILIES) == N_HELDOUT_FAMILIES
assert len(TRAIN_FAMILIES) == N_TRAIN_FAMILIES


# ---------------------------------------------------------------------------
# Sentence record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sentence:
    """One row of the language cache (M1_SPEC 5.3 schema fields + provenance)."""
    instr_id: int
    text: str
    variant: str            # role_binding | precedence | composed
    class_id: int           # RB0/PR0=0, RB1/PR1=1; composed: 2*role + order
    class_name: str         # "RB0", "PR1", "RB1xPR0", ...
    family_id: int          # 0..23 within this variant's grammar
    frame_idx: int          # provenance: sentential frame
    style_idx: int          # provenance: station/connective style
    agent_idx: int          # provenance: agent referring-expression style
    verb_idx: int           # provenance: verb phrase (0 when the frame drops it)
    split: int              # 0 train family / 1 held-out family
    minimal_pair_id: int    # instr_id of the minimal-pair partner
    flip_type: str          # station | agent | connective
    composed_role_class: int    # -1 when not applicable
    composed_order_class: int   # -1 when not applicable


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _render_rb(frame: int, style: int, agent_idx: int, verb_idx: int,
               agents_swapped: bool, stations_swapped: bool) -> Tuple[str, int]:
    """Render a role-binding sentence; returns (text, class_id)."""
    tok0, tok1 = AGENT_REFS[agent_idx]
    a1, a2 = (tok1, tok0) if agents_swapped else (tok0, tok1)
    left, right = STATION_REFS[style]
    s1, s2 = (right, left) if stations_swapped else (left, right)
    v = VERBS_RB[verb_idx]
    if frame == 0:
        text = _cap("{} {} {}, {} {} {}.".format(a1, v, s1, a2, v, s2))
    elif frame == 1:
        text = "Send {} to {}; {} to {}.".format(a1, s1, a2, s2)
    elif frame == 2:
        text = _cap("{} is for {} and {} is for {}.".format(s1, a1, s2, a2))
    else:
        text = "{} — {}. {} — {}.".format(_cap(a1), s1, _cap(a2), s2)
    # robot_0's station is the one paired with its token's slot.
    r0_station = s2 if agents_swapped else s1
    cls = 0 if r0_station == left else 1
    return text, cls


def _render_pr(frame: int, style: int, agent_idx: int, verb_idx: int,
               agents_swapped: bool, rev_form: bool) -> Tuple[str, int]:
    """Render a precedence sentence; returns (text, class_id)."""
    tok0, tok1 = AGENT_REFS[agent_idx]
    p, q = (tok1, tok0) if agents_swapped else (tok0, tok1)
    v3, vb, vg = VERBS_PR[verb_idx]
    pattern = PR_CONNECTIVES[style][1 if rev_form else 0]
    clause = pattern.format(P=p, Q=q, V3=v3, VB=vb, VG=vg)
    if frame == 0:
        text = _cap(clause + ".")
    elif frame == 1:
        text = "This round, {}.".format(clause)
    elif frame == 2:
        text = "The plan is that {}.".format(clause)
    else:
        text = _cap("{} — that is the order.".format(clause))
    p_is_r0 = not agents_swapped
    first_is_r0 = (not p_is_r0) if rev_form else p_is_r0
    cls = 0 if first_is_r0 else 1
    return text, cls


def _render_composed(frame: int, style: int, agent_idx: int, verb_idx: int,
                     stations_swapped: bool, order_swapped: bool
                     ) -> Tuple[str, int, int]:
    """Render a composed sentence; returns (text, role_class, order_class).

    Composed sentences never swap agent tokens (their flip types are station
    and connective only, M1_SPEC 3.2 [FIXED]); robot_0's token always fills
    the first slot, so the order token attached to it encodes the order class
    directly.
    """
    tok0, tok1 = AGENT_REFS[agent_idx]
    left, right = STATION_REFS[style]
    s1, s2 = (right, left) if stations_swapped else (left, right)
    v = VERBS_RB[verb_idx]
    o1, o2 = (_ORDER_WORDS[1], _ORDER_WORDS[0]) if order_swapped else _ORDER_WORDS
    if frame == 0:
        # Inline composition, matching the spec's example sentence shape:
        # "robot one takes the left station and docks first, ..."
        text = _cap("{} {} {} and docks {}, {} {} {} and docks {}.".format(
            tok0, v, s1, o1, tok1, v, s2, o2))
    else:
        body, _ = _render_rb(frame, style, agent_idx, verb_idx,
                             agents_swapped=False,
                             stations_swapped=stations_swapped)
        text = "{} {} docks {}.".format(body, _cap(tok0), o1)
    role = 0 if s1 == left else 1
    order = 0 if o1 == _ORDER_WORDS[0] else 1
    return text, role, order


# ---------------------------------------------------------------------------
# Sampling machinery
# ---------------------------------------------------------------------------

def _family_capacity(variant: str, fid: int) -> int:
    """Number of distinct (agent, verb) fills producing distinct sentences."""
    frame, _ = family_frame_style(fid)
    uses_verb = True if variant == PRECEDENCE else (frame == 0)
    return len(AGENT_REFS) * (len(VERBS_RB) if uses_verb else 1)


def _family_combos(variant: str, fid: int, n: int) -> List[Tuple[int, int]]:
    """Deterministically sample n distinct (agent_idx, verb_idx) fills."""
    frame, _ = family_frame_style(fid)
    uses_verb = True if variant == PRECEDENCE else (frame == 0)
    verb_range = range(len(VERBS_RB)) if uses_verb else (0,)
    combos = [(a, v) for a in range(len(AGENT_REFS)) for v in verb_range]
    if n > len(combos):
        raise ValueError("family %d of %s: need %d combos, capacity %d"
                         % (fid, variant, n, len(combos)))
    rng = random.Random(_SAMPLING_SEED + _VARIANT_CODE[variant] * 65537 + fid * 257)
    rng.shuffle(combos)
    return combos[:n]


def _allocate_pairs(n_pairs: int, families: Iterable[int],
                    capacity: Dict[int, int]) -> List[int]:
    """Deal n_pairs pair slots round-robin over families, capped by capacity."""
    order = sorted(families)
    if n_pairs > sum(capacity[f] for f in order):
        raise ValueError("cannot place %d pairs in families %r" % (n_pairs, order))
    alloc: Counter = Counter()
    dealt: List[int] = []
    while len(dealt) < n_pairs:
        progressed = False
        for f in order:
            if len(dealt) >= n_pairs:
                break
            if alloc[f] < capacity[f]:
                alloc[f] += 1
                dealt.append(f)
                progressed = True
        if not progressed:                        # pragma: no cover - guarded above
            raise RuntimeError("allocation stalled")
    return dealt


def _append_pair(rows: List[Sentence], variant: str, fid: int,
                 agent_idx: int, verb_idx: int, split: int, flip: str,
                 members: List[dict]) -> None:
    """Append the two members of one minimal pair as consecutive rows."""
    frame, style = family_frame_style(fid)
    base_id = len(rows)
    assert len(members) == 2
    for j, m in enumerate(members):
        rows.append(Sentence(
            instr_id=base_id + j,
            text=m["text"],
            variant=variant,
            class_id=m["class_id"],
            class_name=m["class_name"],
            family_id=fid,
            frame_idx=frame,
            style_idx=style,
            agent_idx=agent_idx,
            verb_idx=verb_idx,
            split=split,
            minimal_pair_id=base_id + (1 - j),
            flip_type=flip,
            composed_role_class=m["role"],
            composed_order_class=m["order"],
        ))


def _splits(variant: str):
    """Yield (split, families, n_pairs) for the two family-disjoint splits."""
    yield 0, TRAIN_FAMILIES, (TRAIN_PER_CLASS if variant != COMPOSED
                              else 2 * TRAIN_PER_CLASS)
    yield 1, HELDOUT_FAMILIES, (HELDOUT_PER_CLASS if variant != COMPOSED
                                else 2 * HELDOUT_PER_CLASS)


def _build_single(rows: List[Sentence], variant: str) -> None:
    """Emit the 60 minimal pairs (120 sentences) of one single-axis variant."""
    names = CLASS_NAMES_SINGLE[variant]
    flips = ((FLIP_STATION, FLIP_AGENT) if variant == ROLE_BINDING
             else (FLIP_AGENT, FLIP_CONNECTIVE))
    for split, fams, n_pairs in _splits(variant):
        caps = {f: _family_capacity(variant, f) for f in fams}
        fam_seq = _allocate_pairs(n_pairs, fams, caps)
        need = Counter(fam_seq)
        combo_iter = {f: iter(_family_combos(variant, f, need[f])) for f in need}
        for i, fid in enumerate(fam_seq):
            frame, style = family_frame_style(fid)
            a, v = next(combo_iter[fid])
            flip = flips[i % 2]                 # exact 25/25 (train), 5/5 (held-out)
            base_cls = (i // 2) % 2             # balance surface agent order per class
            if variant == ROLE_BINDING:
                base = _render_rb(frame, style, a, v,
                                  agents_swapped=False,
                                  stations_swapped=(base_cls == 1))
                if flip == FLIP_STATION:
                    part = _render_rb(frame, style, a, v,
                                      agents_swapped=False,
                                      stations_swapped=(base_cls == 0))
                else:  # agent flip
                    part = _render_rb(frame, style, a, v,
                                      agents_swapped=True,
                                      stations_swapped=(base_cls == 1))
            else:
                base = _render_pr(frame, style, a, v,
                                  agents_swapped=False,
                                  rev_form=(base_cls == 1))
                if flip == FLIP_CONNECTIVE:
                    part = _render_pr(frame, style, a, v,
                                      agents_swapped=False,
                                      rev_form=(base_cls == 0))
                else:  # agent flip
                    part = _render_pr(frame, style, a, v,
                                      agents_swapped=True,
                                      rev_form=(base_cls == 1))
            assert {base[1], part[1]} == {0, 1}, "minimal pair must flip the class"
            members = [{"text": t, "class_id": c, "class_name": names[c],
                        "role": (c if variant == ROLE_BINDING else -1),
                        "order": (c if variant == PRECEDENCE else -1)}
                       for t, c in (base, part)]
            _append_pair(rows, variant, fid, a, v, split, flip, members)


# Composed pair kinds, cycled in order: a station flip holds the order class
# fixed and flips the role class; a connective flip holds the role class fixed
# and flips the order class. Cycling i % 4 over an allocation that is a
# multiple of 4 per split (100 train / 20 held-out pairs) gives exactly
# 50 train + 10 held-out sentences per composed class and a 50/50 flip-type
# counterbalance within each split.
_COMPOSED_KINDS = (
    (FLIP_STATION, 0),      # order fixed at 0: pairs RB0xPR0 <-> RB1xPR0
    (FLIP_STATION, 1),      # order fixed at 1: pairs RB0xPR1 <-> RB1xPR1
    (FLIP_CONNECTIVE, 0),   # role fixed at 0:  pairs RB0xPR0 <-> RB0xPR1
    (FLIP_CONNECTIVE, 1),   # role fixed at 1:  pairs RB1xPR0 <-> RB1xPR1
)


def _build_composed(rows: List[Sentence]) -> None:
    """Emit the 120 composed minimal pairs (240 sentences)."""
    for split, fams, n_pairs in _splits(COMPOSED):
        assert n_pairs % len(_COMPOSED_KINDS) == 0
        caps = {f: _family_capacity(COMPOSED, f) for f in fams}
        fam_seq = _allocate_pairs(n_pairs, fams, caps)
        need = Counter(fam_seq)
        combo_iter = {f: iter(_family_combos(COMPOSED, f, need[f])) for f in need}
        for i, fid in enumerate(fam_seq):
            frame, style = family_frame_style(fid)
            a, v = next(combo_iter[fid])
            flip, fixed = _COMPOSED_KINDS[i % len(_COMPOSED_KINDS)]
            if flip == FLIP_STATION:
                specs = [(0, fixed), (1, fixed)]     # (role, order)
            else:
                specs = [(fixed, 0), (fixed, 1)]
            members = []
            for role, order in specs:
                text, r, o = _render_composed(frame, style, a, v,
                                              stations_swapped=(role == 1),
                                              order_swapped=(order == 1))
                assert (r, o) == (role, order)
                members.append({"text": text, "class_id": 2 * r + o,
                                "class_name": "RB{}xPR{}".format(r, o),
                                "role": r, "order": o})
            _append_pair(rows, COMPOSED, fid, a, v, split, flip, members)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_all_cached() -> Tuple[Sentence, ...]:
    rows: List[Sentence] = []
    _build_single(rows, ROLE_BINDING)
    _build_single(rows, PRECEDENCE)
    _build_composed(rows)
    # Structural invariants; the full battery lives in tests/test_templates.py.
    assert len(rows) == N_SENTENCES
    assert all(s.instr_id == i for i, s in enumerate(rows))
    texts_per_variant: Dict[str, set] = {v: set() for v in VARIANTS}
    for s in rows:
        assert s.text not in texts_per_variant[s.variant], "duplicate: " + s.text
        texts_per_variant[s.variant].add(s.text)
    return tuple(rows)


def build_all() -> List[Sentence]:
    """Return the deterministic 480-sentence inventory (cached)."""
    return list(_build_all_cached())


def is_trainable(s: Sentence) -> bool:
    """True when a sentence may appear in training.

    Held-out-family sentences (split == 1) are never trained, and the
    held-out composition is excluded entirely regardless of family.
    """
    if s.split != 0:
        return False
    if s.variant == COMPOSED and \
            (s.composed_role_class, s.composed_order_class) == HELD_OUT_COMPOSITION:
        return False
    return True
