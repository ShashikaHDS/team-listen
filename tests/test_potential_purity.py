"""Static potential/reward purity audit (M1_SPEC 1.2, 1.12, 2.4).

The shaping potential and the reward code must be INSTRUCTION-FREE: spec 1.2
[FIXED: Phi signature contradiction] mandates that ``grid_core.matching_potential``
takes no ``assign`` / ``instr_id`` / ``lang_vec`` argument, and the spec 2.4
forbidden-leak table routes the "instruction-dependent shaping" guard through
this file -- ``rewards.py`` and ``matching_potential`` must never reference
``self.assign``, ``self.instr_id``, ``self.lang_vec`` or ``bank.leak_bit``.
(The one legal instruction-dependent reward term, the +10*Y outcome bonus of
spec 1.12, must therefore be computed OUTSIDE rewards.py and handed in as a
plain precomputed tensor.)

Mechanism: the checks are STATIC. Each target source file is parsed with
``ast`` and every identifier-position string (names, attribute accesses,
argument names, keyword names, def/class names, import targets) is scanned
for instruction/assignment/leak vocabulary. String constants and comments are
deliberately NOT scanned -- docstrings legally mention the forbidden fields
when documenting why they are forbidden.

File targets, adapted to what exists at time of writing:

* ``tasks/team_listen/grid_core.py`` -- exists; checked now (function-level
  and module-wide, plus the pure-torch import rule of spec 1.2).
* ``tasks/team_listen/rewards.py`` -- not yet written (spec section 7 file
  plan); its check SKIPS with a clear message and activates automatically,
  with no edit to this file, the moment the file appears.

pytest-compatible (plain ``test_*`` functions, bare asserts; skips raise
``unittest.SkipTest``, which pytest reports as a skip); also runnable
standalone: ``python tests/test_potential_purity.py`` discovers and runs its
own tests with a pass/fail/skip summary.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.team_listen import grid_core

GRID_CORE_PATH = REPO_ROOT / "tasks" / "team_listen" / "grid_core.py"
REWARDS_PATH = REPO_ROOT / "tasks" / "team_listen" / "rewards.py"

#: The exact fields named by spec 1.2 / 2.4, plus the other instruction- and
#: leak-carrying world-state fields of spec 1.3 / 1.10.  Matched exactly
#: (case-insensitive) against every identifier.
EXACT_FORBIDDEN = frozenset({
    "assign", "assignment",
    "instr_id", "instr_class", "instr_switch_time",
    "lang_vec", "lang_table", "lang_gain",
    "lang_slice", "state_lang_slice",
    "leak_bit", "leak_rho",
})

#: Broader vocabulary net (substring, case-insensitive): NO identifier in a
#: purity-audited source may mention instructions, language or leaks at all.
#: (``assign`` stays exact-match only, so e.g. a hypothetical
#: ``sign``/``design`` helper name cannot false-positive.)
SUBSTRING_FORBIDDEN = ("instr", "lang", "leak")

#: The one legal signature (spec 1.2): matching_potential(dist_field, pos,
#: target_valid) -- and nothing else, ever.
LEGAL_PHI_PARAMS = ["dist_field", "pos", "target_valid"]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse(path):
    """Parse a source file to an AST (comments never reach the AST)."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _identifiers(tree):
    """Every identifier-position string in ``tree``, with its line number.

    Covers: variable names (``ast.Name``), attribute accesses
    (``self.instr_id`` / ``bank.leak_bit`` -> ``ast.Attribute.attr``),
    function/lambda parameter names (``ast.arg``), call keyword names
    (``ast.keyword``), def/class names, global/nonlocal statements and
    import targets (module paths, imported names, aliases).  String
    constants -- docstrings included -- are deliberately excluded.
    """
    out = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            out.append((node.id, line))
        elif isinstance(node, ast.Attribute):
            out.append((node.attr, line))
        elif isinstance(node, ast.arg):
            out.append((node.arg, line))
        elif isinstance(node, ast.keyword):
            if node.arg is not None:                      # skip **kwargs
                out.append((node.arg, line))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            out.append((node.name, line))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.extend((n, line) for n in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, line))
                if alias.asname:
                    out.append((alias.asname, line))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append((node.module, line))
            for alias in node.names:
                out.append((alias.name, line))
                if alias.asname:
                    out.append((alias.asname, line))
    return out


def _violations(idents):
    """Forbidden identifiers among ``idents`` -> sorted unique (name, line)."""
    bad = []
    for name, line in idents:
        low = name.lower()
        if low in EXACT_FORBIDDEN or any(s in low for s in SUBSTRING_FORBIDDEN):
            bad.append((name, line))
    return sorted(set(bad))


def _module_function(tree, name):
    """The module-level FunctionDef called ``name`` (asserts it exists)."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("no module-level def %s(...)" % name)


def _fail_msg(where, bad):
    return ("%s references forbidden assignment/instruction/leak "
            "identifiers %r -- the shaping potential and reward code must be "
            "instruction-free (M1_SPEC 1.2/1.12/2.4)" % (where, bad))


# ---------------------------------------------------------------------------
# grid_core.matching_potential -- exists, checked now
# ---------------------------------------------------------------------------

def test_matching_potential_runtime_signature():
    """The imported function's signature is exactly (dist_field, pos, target_valid)."""
    sig = inspect.signature(grid_core.matching_potential)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == LEGAL_PHI_PARAMS, sig
    for p in params:
        # No *args/**kwargs through which an assignment could be smuggled,
        # and no defaults hiding module-level instruction state.
        assert p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD, (p.name, p.kind)
        assert p.default is inspect.Parameter.empty, (p.name, p.default)


def test_matching_potential_static_signature():
    """The SOURCE signature matches too (guards against runtime wrappers)."""
    fn = _module_function(_parse(GRID_CORE_PATH), "matching_potential")
    a = fn.args
    assert [arg.arg for arg in a.args] == LEGAL_PHI_PARAMS
    assert not a.posonlyargs and not a.kwonlyargs
    assert a.vararg is None and a.kwarg is None
    assert not a.defaults and not a.kw_defaults


def test_matching_potential_body_pure():
    """No identifier inside matching_potential mentions assign/instr/lang/leak."""
    fn = _module_function(_parse(GRID_CORE_PATH), "matching_potential")
    bad = _violations(_identifiers(fn))
    assert not bad, _fail_msg("grid_core.matching_potential", bad)


def test_grid_core_module_pure():
    """The whole transition core is instruction-free, module-wide.

    grid_core is the pure-torch substrate (spec 1.2); nothing in it -- not
    just matching_potential -- may touch instruction/assignment/leak fields.
    """
    bad = _violations(_identifiers(_parse(GRID_CORE_PATH)))
    assert not bad, _fail_msg("tasks/team_listen/grid_core.py", bad)


def test_grid_core_imports_only_torch():
    """Spec 1.2: grid_core imports ONLY torch.

    This is itself a purity guard: with no import of obs_layout, the bank
    loader or the language cache, there is no path by which instruction
    state could reach the transition core or the potential.
    """
    mods = set()
    for node in ast.walk(_parse(GRID_CORE_PATH)):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods == {"torch"}, (
        "grid_core.py must import only torch (M1_SPEC 1.2); found imports %r"
        % sorted(mods))


def test_matching_potential_callsites_pure():
    """Every existing call site passes exactly the three legal arguments.

    Scans all .py files under tasks/ and harness/ for calls to
    ``matching_potential`` and asserts none passes an extra positional
    argument, a ``*``/``**`` splat, or a keyword outside the legal three
    (an ``assign=``/``instr_id=`` keyword at a call site would otherwise
    only fail at runtime).  Passes vacuously while no call site exists yet.
    """
    for sub in ("tasks", "harness"):
        root = REPO_ROOT / sub
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            for node in ast.walk(_parse(path)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None)
                if name != "matching_potential":
                    continue
                where = "%s:%d" % (path.relative_to(REPO_ROOT), node.lineno)
                n_args = len(node.args) + len(node.keywords)
                assert n_args == 3, (
                    "%s calls matching_potential with %d args; the legal "
                    "signature is exactly %r (M1_SPEC 1.2)"
                    % (where, n_args, LEGAL_PHI_PARAMS))
                assert not any(isinstance(a, ast.Starred) for a in node.args), \
                    where + " uses *args in a matching_potential call"
                for kw in node.keywords:
                    assert kw.arg is not None, \
                        where + " uses **kwargs in a matching_potential call"
                    assert kw.arg in LEGAL_PHI_PARAMS, (
                        "%s passes illegal keyword %r to matching_potential"
                        % (where, kw.arg))


# ---------------------------------------------------------------------------
# rewards.py -- forward-looking: not yet written (spec section 7 file plan)
# ---------------------------------------------------------------------------

def test_rewards_module_pure():
    """rewards.py, when it exists, must be instruction-free module-wide.

    Spec 1.2/2.4: ``rewards.py`` never references ``self.assign``,
    ``self.instr_id``, ``self.lang_vec`` or ``bank.leak_bit`` (nor any other
    instruction/language/leak identifier).  The +10*Y outcome bonus (spec
    1.12) is legal only as a PRECOMPUTED tensor argument -- computing Y
    inside rewards.py necessarily trips this scan, by design.
    """
    if not REWARDS_PATH.exists():
        raise unittest.SkipTest(
            "tasks/team_listen/rewards.py is not written yet (M1_SPEC "
            "section 7 file plan); this purity check activates automatically "
            "once the file exists -- do NOT delete this test.")
    bad = _violations(_identifiers(_parse(REWARDS_PATH)))
    assert not bad, _fail_msg("tasks/team_listen/rewards.py", bad)


def test_rewards_imports_pure():
    """rewards.py, when it exists, must not import language/bank machinery.

    Companion to the identifier scan: an ``import harness.lang_cache`` or
    ``from .scenario_bank import ...`` inside the reward module would be a
    structural leak channel even before any field is read.  (The identifier
    scan already catches 'lang'/'instr'/'leak' import names; this check
    additionally pins the ALLOWED import surface: torch, grid_core and
    obs_layout only.)
    """
    if not REWARDS_PATH.exists():
        raise unittest.SkipTest(
            "tasks/team_listen/rewards.py is not written yet (M1_SPEC "
            "section 7 file plan); this import-surface check activates "
            "automatically once the file exists -- do NOT delete this test.")

    def _ok(module, names):
        """Allow torch and the two pure sibling modules, in any import form:
        ``import torch``, ``from . import grid_core``,
        ``from .obs_layout import X``, ``from tasks.team_listen import ...``.
        """
        if module == "torch":
            return True
        allowed_leaves = ("grid_core", "obs_layout")
        if module in ("", "tasks.team_listen"):        # from-package import
            return bool(names) and all(n in allowed_leaves for n in names)
        leaf = module.split(".")[-1]
        if leaf in allowed_leaves:                      # from-module import
            prefix = module[: -len(leaf)].rstrip(".")
            return prefix in ("", "tasks.team_listen")
        return False

    illegal = []
    for node in ast.walk(_parse(REWARDS_PATH)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _ok(alias.name, []):
                    illegal.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if not _ok(node.module or "", [a.name for a in node.names]):
                illegal.append("from %s import %s" % (
                    "." * node.level + (node.module or ""),
                    ", ".join(a.name for a in node.names)))
    assert not illegal, (
        "rewards.py imports %r; only torch/grid_core/obs_layout are allowed "
        "(M1_SPEC 1.2 purity + section 7 file plan)" % illegal)


# ---------------------------------------------------------------------------
# Self-test of the scanner: it must be able to fail
# ---------------------------------------------------------------------------

def test_scanner_detects_planted_leaks():
    """The AST scan actually catches every forbidden reference form.

    Guards against the audit going vacuous through a scanner bug (the same
    failure mode spec 2.4 [FIXED] documents for the old independence test).
    """
    planted = (
        "def f(self, bank, assign):\n"
        "    x = self.instr_id\n"
        "    y = bank.leak_bit\n"
        "    z = lang_vec\n"
        "    g(instr_class=1)\n"
        "    import harness.lang_cache\n"
    )
    bad = dict(_violations(_identifiers(ast.parse(planted))))
    for name in ("assign", "instr_id", "leak_bit", "lang_vec",
                 "instr_class", "harness.lang_cache"):
        assert name in bad, (name, bad)
    # ...and does NOT flag clean transition-core vocabulary.
    clean = "def f(dist_field, pos, target_valid):\n    return design(sign(pos))\n"
    assert not _violations(_identifiers(ast.parse(clean)))
    # Docstrings mentioning forbidden fields are legal (they document the ban).
    doc = 'def f():\n    """never reference self.instr_id or bank.leak_bit"""\n'
    assert not _violations(_identifiers(ast.parse(doc)))


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

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
