"""Tests for harness/templates.py (M1_SPEC 2.2, 3.2, 5.3, 5.4).

Verifies the exact family counts (24 per variant grammar), the 18/6
train/held-out family split with disjoint, family-disjoint sampling, class
balance (50 train + 10 held-out sentences per class), minimal-pair validity
(same family/frame/slot fill, opposite class, both flip types, exact
flip-type counterbalance), and token-level minimality of every pair.

pytest-compatible (plain ``test_*`` functions, bare asserts); also runnable
standalone: ``python tests/test_templates.py`` discovers and runs its own
tests with a pass/fail summary.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness import templates as T


def _rows():
    return T.build_all()


def _by_variant(variant):
    return [s for s in _rows() if s.variant == variant]


def _pairs(rows):
    """One representative (lower instr_id) per minimal pair."""
    all_rows = _rows()
    return [(s, all_rows[s.minimal_pair_id]) for s in rows
            if s.instr_id < s.minimal_pair_id]


def _swap(text, tok_a, tok_b):
    """Exchange two token strings inside text (placeholder-safe)."""
    ph = "\x00"
    return text.replace(tok_a, ph).replace(tok_b, tok_a).replace(ph, tok_b)


# ---------------------------------------------------------------------------
# Cardinalities and totals
# ---------------------------------------------------------------------------

def test_grammar_cardinalities():
    # 5 x 4 x 6 x 4 grammar (M1_SPEC 2.2), same shape for precedence (3.2).
    assert len(T.AGENT_REFS) == 5
    assert len(T.VERBS_RB) == 4
    assert len(T.STATION_REFS) == 6
    assert len(T.VERBS_PR) == 4
    assert len(T.PR_CONNECTIVES) == 6
    assert T.N_FRAMES == 4
    assert T.N_FAMILIES == 24
    assert T.N_TRAIN_FAMILIES == 18
    assert T.N_HELDOUT_FAMILIES == 6
    # Every referring-expression style is a (robot_0/left, robot_1/right) pair
    # of distinct strings; connectives are (forward, polarity-flipped) pairs.
    for pair in list(T.AGENT_REFS) + list(T.STATION_REFS) + list(T.PR_CONNECTIVES):
        assert len(pair) == 2 and pair[0] != pair[1]


def test_total_counts():
    rows = _rows()
    assert len(rows) == T.N_SENTENCES == 480
    counts = Counter(s.variant for s in rows)
    assert counts[T.ROLE_BINDING] == 120
    assert counts[T.PRECEDENCE] == 120
    assert counts[T.COMPOSED] == 240
    assert all(s.instr_id == i for i, s in enumerate(rows))


def test_build_deterministic():
    a = [ (s.text, s.class_id, s.family_id, s.split, s.minimal_pair_id, s.flip_type)
          for s in T.build_all() ]
    b = [ (s.text, s.class_id, s.family_id, s.split, s.minimal_pair_id, s.flip_type)
          for s in T.build_all() ]
    assert a == b


# ---------------------------------------------------------------------------
# Families and the 18/6 split
# ---------------------------------------------------------------------------

def test_family_split_definition():
    assert T.TRAIN_FAMILIES.isdisjoint(T.HELDOUT_FAMILIES)
    assert len(T.TRAIN_FAMILIES) == 18
    assert len(T.HELDOUT_FAMILIES) == 6
    assert T.TRAIN_FAMILIES | T.HELDOUT_FAMILIES == set(range(24))
    # Held-out families cover every station/connective style exactly once.
    styles = sorted(T.family_frame_style(f)[1] for f in T.HELDOUT_FAMILIES)
    assert styles == list(range(6))


def test_family_id_roundtrip():
    for frame in range(T.N_FRAMES):
        for style in range(T.N_STYLES):
            assert T.family_frame_style(T.family_id(frame, style)) == (frame, style)


def test_family_usage_per_variant():
    for variant in T.VARIANTS:
        rows = _by_variant(variant)
        train_fams = {s.family_id for s in rows if s.split == 0}
        held_fams = {s.family_id for s in rows if s.split == 1}
        assert train_fams == T.TRAIN_FAMILIES, variant
        assert held_fams == T.HELDOUT_FAMILIES, variant
        # Family id encodes this row's (frame, style).
        for s in rows:
            assert T.family_id(s.frame_idx, s.style_idx) == s.family_id


def test_split_disjointness():
    # Family-disjoint sampling: no family in both splits, hence no sentence
    # in both splits; sentences are unique within each variant.
    for variant in T.VARIANTS:
        rows = _by_variant(variant)
        train_fams = {s.family_id for s in rows if s.split == 0}
        held_fams = {s.family_id for s in rows if s.split == 1}
        assert train_fams.isdisjoint(held_fams), variant
        texts = [s.text for s in rows]
        assert len(texts) == len(set(texts)), variant
        train_texts = {s.text for s in rows if s.split == 0}
        held_texts = {s.text for s in rows if s.split == 1}
        assert train_texts.isdisjoint(held_texts), variant


# ---------------------------------------------------------------------------
# Class counts and counterbalancing
# ---------------------------------------------------------------------------

def test_single_axis_class_counts():
    for variant in (T.ROLE_BINDING, T.PRECEDENCE):
        rows = _by_variant(variant)
        per_class = Counter(s.class_id for s in rows)
        assert per_class == {0: 60, 1: 60}, variant
        per_class_split = Counter((s.class_id, s.split) for s in rows)
        assert per_class_split == {(0, 0): 50, (1, 0): 50,
                                   (0, 1): 10, (1, 1): 10}, variant
        names = set(s.class_name for s in rows)
        assert names == set(T.CLASS_NAMES_SINGLE[variant]), variant


def test_composed_class_counts():
    rows = _by_variant(T.COMPOSED)
    per_class = Counter(s.class_id for s in rows)
    assert per_class == {0: 60, 1: 60, 2: 60, 3: 60}
    per_class_split = Counter((s.class_id, s.split) for s in rows)
    for c in range(4):
        assert per_class_split[(c, 0)] == 50
        assert per_class_split[(c, 1)] == 10
    for s in rows:
        assert s.composed_role_class in (0, 1)
        assert s.composed_order_class in (0, 1)
        assert s.class_id == 2 * s.composed_role_class + s.composed_order_class
        assert s.class_name == "RB{}xPR{}".format(s.composed_role_class,
                                                  s.composed_order_class)


def test_per_family_class_balance():
    # Within every (variant, family), classes are exactly balanced for the
    # single-axis variants (each minimal pair contributes one of each class).
    for variant in (T.ROLE_BINDING, T.PRECEDENCE):
        per = defaultdict(Counter)
        for s in _by_variant(variant):
            per[s.family_id][s.class_id] += 1
        for fid, counts in per.items():
            assert counts[0] == counts[1], (variant, fid)


def test_flip_type_counterbalance():
    expected = {
        T.ROLE_BINDING: ((T.FLIP_STATION, T.FLIP_AGENT), 25, 5),
        T.PRECEDENCE: ((T.FLIP_AGENT, T.FLIP_CONNECTIVE), 25, 5),
        T.COMPOSED: ((T.FLIP_STATION, T.FLIP_CONNECTIVE), 50, 10),
    }
    for variant, (flips, n_train, n_held) in expected.items():
        pairs = _pairs(_by_variant(variant))
        counts = Counter((p[0].split, p[0].flip_type) for p in pairs)
        for f in flips:
            assert counts[(0, f)] == n_train, (variant, f, counts)
            assert counts[(1, f)] == n_held, (variant, f, counts)
        used = {p[0].flip_type for p in pairs}
        assert used == set(flips), variant


# ---------------------------------------------------------------------------
# Minimal-pair validity
# ---------------------------------------------------------------------------

def test_minimal_pair_linkage():
    rows = _rows()
    for s in rows:
        p = rows[s.minimal_pair_id]
        assert p.minimal_pair_id == s.instr_id          # symmetric involution
        assert p.instr_id != s.instr_id
        assert p.variant == s.variant
        assert p.family_id == s.family_id               # same family
        assert p.frame_idx == s.frame_idx               # same frame
        assert p.style_idx == s.style_idx
        assert p.agent_idx == s.agent_idx               # same slot fill
        assert p.verb_idx == s.verb_idx
        assert p.split == s.split
        assert p.flip_type == s.flip_type
        assert p.text != s.text


def test_minimal_pair_class_semantics():
    rows = _rows()
    for s in rows:
        p = rows[s.minimal_pair_id]
        if s.variant == T.COMPOSED:
            if s.flip_type == T.FLIP_STATION:
                # Role flips, order fixed.
                assert p.composed_role_class == 1 - s.composed_role_class
                assert p.composed_order_class == s.composed_order_class
            else:
                assert s.flip_type == T.FLIP_CONNECTIVE
                assert p.composed_role_class == s.composed_role_class
                assert p.composed_order_class == 1 - s.composed_order_class
        else:
            assert {s.class_id, p.class_id} == {0, 1}, (s.variant, s.instr_id)


def test_minimal_pair_surface_minimality():
    # The two texts of a pair must differ only in the role-bearing token(s):
    # swapping exactly those tokens maps one member onto the other
    # (case-insensitively; sentence-initial capitalization aside).
    rows = _rows()
    for s, p in _pairs(rows):
        a, b = s.text.lower(), p.text.lower()
        if s.flip_type == T.FLIP_STATION:
            left, right = T.STATION_REFS[s.style_idx]
            assert _swap(a, left.lower(), right.lower()) == b, (a, b)
        elif s.flip_type == T.FLIP_AGENT:
            tok0, tok1 = T.AGENT_REFS[s.agent_idx]
            assert _swap(a, tok0.lower(), tok1.lower()) == b, (a, b)
        elif s.variant == T.COMPOSED:
            # Composed connective flip: only the order bigrams move.
            assert _swap(a, "docks first", "docks second") == b, (a, b)
        else:
            # Precedence connective flip: agent tokens stay in place (and in
            # the same relative order); only the connective's polarity tokens
            # change, which test_minimal_pair_class_semantics ties to a class
            # flip within the same (frame, connective-style) family.
            tok0, tok1 = (t.lower() for t in T.AGENT_REFS[s.agent_idx])
            assert tok0 in a and tok1 in a and tok0 in b and tok1 in b
            assert (a.index(tok0) < a.index(tok1)) == (b.index(tok0) < b.index(tok1))


def test_role_bearing_tokens_present():
    # Every sentence carries both agent tokens; role-binding/composed carry
    # both station tokens of the family's style (a referring expression needs
    # its contrast set realized in the sentence).
    for s in _rows():
        low = s.text.lower()
        tok0, tok1 = (t.lower() for t in T.AGENT_REFS[s.agent_idx])
        assert tok0 in low and tok1 in low, s.text
        if s.variant in (T.ROLE_BINDING, T.COMPOSED):
            left, right = (t.lower() for t in T.STATION_REFS[s.style_idx])
            assert left in low and right in low, s.text
        if s.variant == T.COMPOSED:
            n_order = low.count("docks first") + low.count("docks second")
            if s.frame_idx == 0:      # inline composition orders both agents
                assert low.count("docks first") == 1
                assert low.count("docks second") == 1
            else:                     # appended tail orders robot_0 only
                assert n_order == 1, s.text


# ---------------------------------------------------------------------------
# Field hygiene and training filter
# ---------------------------------------------------------------------------

def test_axis_class_fields():
    for s in _rows():
        if s.variant == T.ROLE_BINDING:
            assert s.composed_role_class == s.class_id
            assert s.composed_order_class == -1
        elif s.variant == T.PRECEDENCE:
            assert s.composed_role_class == -1
            assert s.composed_order_class == s.class_id
        assert s.flip_type in T.FLIP_TYPES
        assert s.split in (0, 1)


def test_held_out_composition():
    r, o = T.HELD_OUT_COMPOSITION
    assert r in (0, 1) and o in (0, 1)
    rows = _rows()
    held_class = [s for s in rows if s.variant == T.COMPOSED
                  and (s.composed_role_class, s.composed_order_class) == (r, o)]
    assert len(held_class) == 60
    # is_trainable: never a held-out family, never the held-out composition.
    for s in rows:
        trainable = T.is_trainable(s)
        if s.split == 1:
            assert not trainable
        elif s.variant == T.COMPOSED and \
                (s.composed_role_class, s.composed_order_class) == (r, o):
            assert not trainable
        else:
            assert trainable
    n_trainable = sum(T.is_trainable(s) for s in rows)
    # 100 RB + 100 PR + 3 trainable compositions x 50.
    assert n_trainable == 100 + 100 + 150


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    try:  # keep unicode sentence text printable on cp1252 consoles
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

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
