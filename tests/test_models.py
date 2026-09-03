"""Tests for harness/models.py -- the DeepSets entity encoder (M1_SPEC 1.5,
section 7, OPEN(4); DECISIONS.md 2026-09-02 round 5).

CPU-only; skrl is NOT required: the encoder core (EntityEncoder,
EntityPolicyNet, EntityValueNet) is plain torch and is tested directly.
The skrl Model subclasses are tested for their guard behaviour when skrl
is absent (clear ImportError on CONSTRUCTION, clean module import), and
for real construction/compute when skrl is present (5090).

Invariance scope asserted here (mirrors the models.py docstring):

* obs layout (376): the representation is invariant to permuting TEAMMATE
  slot contents and TARGET slot contents (presence bits travel inside the
  blocks). The EGO block -- which carries the agent-id one-hot -- is a
  separate, un-pooled stream, so per-agent identity is preserved; the
  spec's agent-ID one-hots therefore constrain invariance to the pooled
  groups only, and the ego-sensitivity test pins that down.
* state layouts (641/645): invariant under a JOINT permutation of (13-wide
  agent block, its 32-d STATE_LANG_SLICE vector) pairs -- the agent-id
  one-hot travels inside the block, so nothing identity-bearing is lost.
  Fixtures keep the target blocks' occupier one-hots all-zero so agent-
  slot references stay semantically coherent under the permutation (the
  function-level invariance holds regardless: target blocks are untouched).

Floating point: pooling is a reduction, so a permutation can change the
fp accumulation order; permutation tests use allclose with tight
tolerances. The MASKED-entity tests demand torch.equal (bit-identical):
a masked slot contributes exactly phi(content) * 0.0 = 0.0 in the same
accumulation position, so garbage behind a zero present bit must change
nothing at all.

pytest-compatible (plain ``test_*`` functions, bare asserts); also
runnable standalone: ``python tests/test_models.py``.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from harness import models as M
from tasks.team_listen import obs_layout as L

# fp-reassociation tolerance for permutation tests (see module docstring)
_ATOL = 1e-6
_RTOL = 1e-5

BATCH_SIZES = (1, 2, 7, 33)


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _seeded_encoder(layout, seed=1234, cfg=None):
    torch.manual_seed(seed)
    return M.EntityEncoder(layout=layout, cfg=cfg)


def _rand(g, *shape):
    return torch.rand(*shape, generator=g)


def _unit(g, *shape):
    v = torch.randn(*shape, generator=g)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _garbage(g, *shape):
    """Finite junk, deliberately outside the normalised feature ranges."""
    return _rand(g, *shape) * 6.0 - 3.0


# ---------------------------------------------------------------------------
# Fixture builders (all offsets via obs_layout -- the input contract)
# ---------------------------------------------------------------------------

def make_obs(B, g, teammates_valid=(0,), targets_valid=(0, 1)):
    """Structurally valid 376-d actor observation batch.

    Padded (non-valid) entity slots are all-zero with present = 0; the
    masked-entity fixtures are derived from a clean batch afterwards via
    ``fill_padded_garbage`` so the two differ ONLY inside masked slots.
    """
    x = torch.zeros(B, L.OBS_DIM)
    x[:, L.OBS_KNOWN_FREE_SLICE] = (_rand(g, B, L.P) < 0.5).float()
    x[:, L.OBS_KNOWN_OBS_SLICE] = (_rand(g, B, L.P) < 0.2).float()

    e = L.OBS_EGO_SLICE.start
    x[:, e + L.EGO_POS_OFF:e + L.EGO_POS_OFF + 2] = _rand(g, B, 2) * 2 - 1
    x[:, e + L.EGO_AGENT_ID_OFF] = 1.0                    # ego = agent slot 0

    for slot in range(L.N_TEAMMATE_SLOTS):
        s = L.obs_teammate_slice(slot).start
        if slot in teammates_valid:
            x[:, s + L.TM_PRESENT_OFF] = 1.0
            x[:, s + L.TM_POS_OFF:s + L.TM_POS_OFF + 2] = _rand(g, B, 2) * 2 - 1
            x[:, s + L.TM_LATCH_TIME_OFF] = _rand(g, B)

    for slot in range(L.MAX_TARGETS):
        s = L.obs_target_slice(slot).start
        if slot in targets_valid:
            x[:, s + L.TGT_PRESENT_OFF] = 1.0
            x[:, s + L.TGT_POS_OFF:s + L.TGT_POS_OFF + 2] = _rand(g, B, 2) * 2 - 1
            x[:, s + L.TGT_OCCUPIED_OFF] = (_rand(g, B) < 0.3).float()

    x[:, L.OBS_TIME_SLICE] = _rand(g, B, L.OBS_TIME_W)
    x[:, L.LANG_SLICE] = _unit(g, B, L.LANG_DIM)
    return x


def make_state(B, g, agents_valid=(0, 1), targets_valid=(0, 1)):
    """Structurally valid 641-d critic state batch.

    Occupier one-hots in the target blocks are kept all-zero (unoccupied)
    so agent-slot references stay coherent under agent-slot permutations
    (module docstring). Padded agent slots zero BOTH their 13-wide block
    and their 32-d STATE_LANG_SLICE row.
    """
    x = torch.zeros(B, L.STATE_DIM)
    x[:, L.STATE_OBSTACLE_SLICE] = (_rand(g, B, L.P) < 0.2).float()
    x[:, L.STATE_KNOWN_FREE_SLICE] = (_rand(g, B, L.P) < 0.5).float()
    x[:, L.STATE_KNOWN_OBS_SLICE] = (_rand(g, B, L.P) < 0.2).float()

    for slot in range(L.MAX_AGENTS):
        s = L.state_agent_slice(slot).start
        lang = L.state_lang_slice(slot)
        if slot in agents_valid:
            x[:, s] = 1.0                                   # present (off 0)
            x[:, s + 1:s + 3] = _rand(g, B, 2) * 2 - 1      # (r, c)
            x[:, s + L.STATE_AGENT_W - L.AGENT_ID_ONEHOT_W + slot] = 1.0
            x[:, lang] = _unit(g, B, L.LANG_DIM)

    for slot in range(L.MAX_TARGETS):
        s = L.state_target_slice(slot).start
        if slot in targets_valid:
            x[:, s + L.TGT_PRESENT_OFF] = 1.0
            x[:, s + L.TGT_POS_OFF:s + L.TGT_POS_OFF + 2] = _rand(g, B, 2) * 2 - 1

    x[:, L.STATE_TIME_SLICE] = _rand(g, B, L.STATE_TIME_W)
    return x


def fill_padded_garbage(layout, x, g):
    """Clone ``x`` and write finite junk into every MASKED slot's
    non-present fields (present bits stay 0; nothing else changes) --
    so clean-vs-junk differs ONLY behind zero presence masks."""
    y = x.clone()
    B = y.shape[0]
    if layout == "obs":
        for slot in range(L.N_TEAMMATE_SLOTS):
            s = L.obs_teammate_slice(slot).start
            if y[:, s + L.TM_PRESENT_OFF].eq(0).all():
                y[:, s + 1:s + L.OBS_TEAMMATE_W] = \
                    _garbage(g, B, L.OBS_TEAMMATE_W - 1)
        for slot in range(L.MAX_TARGETS):
            s = L.obs_target_slice(slot).start
            if y[:, s + L.TGT_PRESENT_OFF].eq(0).all():
                y[:, s + 1:s + L.OBS_TARGET_W] = \
                    _garbage(g, B, L.OBS_TARGET_W - 1)
    else:  # state / full_state_obs share the state slice table
        for slot in range(L.MAX_AGENTS):
            s = L.state_agent_slice(slot).start
            if y[:, s].eq(0).all():                 # present is offset 0
                y[:, s + 1:s + L.STATE_AGENT_W] = \
                    _garbage(g, B, L.STATE_AGENT_W - 1)
                y[:, L.state_lang_slice(slot)] = _garbage(g, B, L.LANG_DIM)
        for slot in range(L.MAX_TARGETS):
            s = L.state_target_slice(slot).start
            if y[:, s + L.TGT_PRESENT_OFF].eq(0).all():
                y[:, s + 1:s + L.STATE_TARGET_W] = \
                    _garbage(g, B, L.STATE_TARGET_W - 1)
    return y


def make_full_state_obs(B, g, **kw):
    """645-d full-state actor obs: state + own agent-id one-hot (spec 4.1)."""
    state = make_state(B, g, **kw)
    aid = torch.zeros(B, L.AGENT_ID_ONEHOT_W)
    aid[:, 0] = 1.0
    return torch.cat([state, aid], dim=-1)


def permute_slots(x, parent_slice, n_slots, width, perm):
    """Permute the slot CONTENTS (blocks travel whole, presence included)."""
    y = x.clone()
    blocks = x[:, parent_slice].reshape(-1, n_slots, width)
    y[:, parent_slice] = blocks[:, list(perm)].reshape(x.shape[0], n_slots * width)
    return y


def permute_state_agents(x, perm):
    """JOINT permutation of (agent block, its lang vector) pairs."""
    y = permute_slots(x, L.STATE_AGENTS_SLICE, L.MAX_AGENTS,
                      L.STATE_AGENT_W, perm)
    return permute_slots(y, L.STATE_LANG_SLICE, L.MAX_AGENTS,
                         L.LANG_DIM, perm)


_MAKERS = {
    "obs": make_obs,
    "state": make_state,
    "full_state_obs": make_full_state_obs,
}

_DIMS = {
    "obs": L.OBS_DIM,
    "state": L.STATE_DIM,
    "full_state_obs": L.FULL_STATE_OBS_DIM,
}


# ---------------------------------------------------------------------------
# Import hygiene / layout table
# ---------------------------------------------------------------------------

def test_module_imports_without_skrl():
    # On this dev box skrl is absent and the module must import cleanly
    # (it already did, above); the flag must be a plain bool either way.
    assert isinstance(M.SKRL_AVAILABLE, bool)
    assert "skrl" not in sys.modules or M.SKRL_AVAILABLE


def test_layout_by_dim_table():
    assert M.LAYOUT_BY_DIM == {376: "obs", 641: "state", 645: "full_state_obs"}
    assert M.N_ACTIONS == 5


def test_cfg_validation():
    assert M.resolve_cfg(None) == M.resolve_cfg({})
    for bad in ({"pool": "max"}, {"activation": "gelu"}, {"phony_key": 1},
                {"trunk_hidden": ()}, {"map_hidden": (0,)}):
        try:
            M.resolve_cfg(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("resolve_cfg(%r) should reject" % (bad,))


# ---------------------------------------------------------------------------
# Shapes -- every layout, several batch sizes, extra leading dims
# ---------------------------------------------------------------------------

def test_encoder_shapes_all_layouts():
    for layout, maker in _MAKERS.items():
        enc = _seeded_encoder(layout)
        assert enc.in_dim == _DIMS[layout]
        for B in BATCH_SIZES:
            out = enc(maker(B, _gen(B)))
            assert out.shape == (B, enc.out_dim), (layout, B, out.shape)
            assert torch.isfinite(out).all(), (layout, B)


def test_policy_value_net_shapes():
    torch.manual_seed(0)
    pol = M.EntityPolicyNet(layout="obs")
    val = M.EntityValueNet(layout="state")
    pol_full = M.EntityPolicyNet(layout="full_state_obs")
    for B in BATCH_SIZES:
        g = _gen(100 + B)
        assert pol(make_obs(B, g)).shape == (B, M.N_ACTIONS)
        assert val(make_state(B, g)).shape == (B, 1)
        assert pol_full(make_full_state_obs(B, g)).shape == (B, M.N_ACTIONS)


def test_encoder_leading_dims():
    enc = _seeded_encoder("obs")
    x = make_obs(6, _gen(7)).reshape(2, 3, L.OBS_DIM)
    out = enc(x)
    assert out.shape == (2, 3, enc.out_dim)
    # identical to the flat call, element for element
    flat = enc(x.reshape(6, L.OBS_DIM))
    assert torch.equal(out.reshape(6, enc.out_dim), flat)


def test_encoder_rejects_wrong_width():
    enc = _seeded_encoder("obs")
    for bad in (L.OBS_DIM - 1, L.STATE_DIM, L.FULL_STATE_OBS_DIM):
        try:
            enc(torch.zeros(2, bad))
        except ValueError as exc:
            assert "376" in str(exc)
        else:
            raise AssertionError("width %d should be rejected" % bad)


def test_encoder_deterministic():
    for layout, maker in _MAKERS.items():
        enc = _seeded_encoder(layout)
        x = maker(5, _gen(11))
        assert torch.equal(enc(x), enc(x)), layout


# ---------------------------------------------------------------------------
# Presence-mask correctness (bit-identical -- module docstring)
# ---------------------------------------------------------------------------

def _mask_check(layout, seed, pool):
    maker = _MAKERS[layout]
    enc = _seeded_encoder(layout, seed=seed, cfg={"pool": pool})
    clean = maker(9, _gen(seed))
    junk = fill_padded_garbage(layout, clean, _gen(seed + 1000))
    # the fixtures differ ONLY inside masked (present = 0) slots...
    assert not torch.equal(clean, junk)
    # ...and the encoder must not see the difference AT ALL:
    assert torch.equal(enc(clean), enc(junk)), (layout, pool)


def test_masked_slots_bit_identical_obs():
    for pool in ("mean", "sum"):
        _mask_check("obs", 21, pool)


def test_masked_slots_bit_identical_state():
    # covers garbage in padded agent blocks AND their paired lang rows
    for pool in ("mean", "sum"):
        _mask_check("state", 22, pool)


def test_masked_slots_bit_identical_full_state_obs():
    for pool in ("mean", "sum"):
        _mask_check("full_state_obs", 23, pool)


def test_adding_masked_entity_bit_identical():
    """Writing an entity INTO a padded slot while leaving present = 0
    changes nothing, bit for bit (task: 'adding a masked entity')."""
    enc = _seeded_encoder("obs", seed=31)
    base = make_obs(4, _gen(31))
    added = base.clone()
    s = L.obs_target_slice(2).start                 # slot 2 is padding
    g = _gen(32)
    added[:, s + 1:s + L.OBS_TARGET_W] = _rand(g, 4, L.OBS_TARGET_W - 1)
    assert added[:, s + L.TGT_PRESENT_OFF].eq(0).all()
    assert torch.equal(enc(base), enc(added))
    # control: the SAME content with present = 1 must change the output
    visible = added.clone()
    visible[:, s + L.TGT_PRESENT_OFF] = 1.0
    assert not torch.allclose(enc(base), enc(visible), atol=_ATOL)


# ---------------------------------------------------------------------------
# Permutation invariance (scope: module docstring)
# ---------------------------------------------------------------------------

_PERMS3 = [(1, 0, 2), (2, 0, 1), (2, 1, 0)]
_PERMS4 = [(1, 0, 2, 3), (3, 2, 1, 0), (1, 2, 3, 0)]


def test_obs_target_slot_permutation_invariance():
    for pool in ("mean", "sum"):
        enc = _seeded_encoder("obs", seed=41, cfg={"pool": pool})
        x = make_obs(6, _gen(41), targets_valid=(0, 1, 2))
        out = enc(x)
        for perm in _PERMS3:
            xp = permute_slots(x, L.OBS_TARGETS_SLICE, L.MAX_TARGETS,
                               L.OBS_TARGET_W, perm)
            assert not torch.equal(xp, x)           # the permutation is real
            assert torch.allclose(enc(xp), out, atol=_ATOL, rtol=_RTOL), \
                (pool, perm)
        # teeth: perturbing a valid target's content DOES change the output
        bumped = x.clone()
        bumped[:, L.obs_target_slice(1).start + L.TGT_POS_OFF] += 0.25
        assert not torch.allclose(enc(bumped), out, atol=_ATOL)


def test_obs_teammate_slot_permutation_invariance():
    """Teammate blocks carry no identity features (spec 1.5), so pooling
    them is lossless invariance; two valid teammates exercise it (N=4
    capacity -- the held-out-team-size case the encoder exists for)."""
    for pool in ("mean", "sum"):
        enc = _seeded_encoder("obs", seed=42, cfg={"pool": pool})
        x = make_obs(6, _gen(42), teammates_valid=(0, 1))
        out = enc(x)
        for perm in _PERMS3:
            xp = permute_slots(x, L.OBS_TEAMMATES_SLICE, L.N_TEAMMATE_SLOTS,
                               L.OBS_TEAMMATE_W, perm)
            assert torch.allclose(enc(xp), out, atol=_ATOL, rtol=_RTOL), \
                (pool, perm)
        bumped = x.clone()
        bumped[:, L.obs_teammate_slice(0).start + L.TM_POS_OFF] += 0.25
        assert not torch.allclose(enc(bumped), out, atol=_ATOL)


def test_obs_ego_is_not_pooled():
    """The ego stream (agent-id one-hot included) bypasses pooling: the
    policy stays agent-identifiable, exactly as spec 1.5's ego block
    prescribes -- invariance applies to the pooled groups ONLY."""
    enc = _seeded_encoder("obs", seed=43)
    x = make_obs(4, _gen(43))
    out = enc(x)
    other_id = x.clone()
    e = L.OBS_EGO_SLICE.start + L.EGO_AGENT_ID_OFF
    other_id[:, e:e + L.AGENT_ID_ONEHOT_W] = 0.0
    other_id[:, e + 1] = 1.0                        # ego claims slot 1 instead
    assert not torch.allclose(enc(other_id), out, atol=_ATOL)


def test_state_agent_joint_permutation_invariance():
    """State layout: invariant under the JOINT permutation of (agent
    block, its lang vector); the agent-id one-hot travels inside the
    block, so the spec's IDs are preserved, not discarded."""
    for layout in ("state", "full_state_obs"):
        for pool in ("mean", "sum"):
            enc = _seeded_encoder(layout, seed=44, cfg={"pool": pool})
            x = _MAKERS[layout](5, _gen(44), agents_valid=(0, 1, 2))
            out = enc(x)
            for perm in _PERMS4:
                xp = permute_state_agents(x, perm)
                assert torch.allclose(enc(xp), out, atol=_ATOL, rtol=_RTOL), \
                    (layout, pool, perm)
            # teeth: permuting agent blocks WITHOUT their lang rows must
            # NOT be invariant (the (agent, instruction) binding is real)
            broken = permute_slots(x, L.STATE_AGENTS_SLICE, L.MAX_AGENTS,
                                   L.STATE_AGENT_W, _PERMS4[0])
            assert not torch.allclose(enc(broken), out, atol=_ATOL), \
                (layout, pool)


def test_state_target_slot_permutation_invariance():
    for pool in ("mean", "sum"):
        enc = _seeded_encoder("state", seed=45, cfg={"pool": pool})
        x = make_state(5, _gen(45), targets_valid=(0, 1, 2))
        out = enc(x)
        for perm in _PERMS3:
            xp = permute_slots(x, L.STATE_TARGETS_SLICE, L.MAX_TARGETS,
                               L.STATE_TARGET_W, perm)
            assert torch.allclose(enc(xp), out, atol=_ATOL, rtol=_RTOL), \
                (pool, perm)


# ---------------------------------------------------------------------------
# Language sensitivity (the audit's whole point: the channel must be live)
# ---------------------------------------------------------------------------

def test_obs_lang_slice_sensitivity():
    enc = _seeded_encoder("obs", seed=51)
    x = make_obs(4, _gen(51))
    out = enc(x)
    flipped = x.clone()
    flipped[:, L.LANG_SLICE] = _unit(_gen(52), 4, L.LANG_DIM)
    assert not torch.allclose(enc(flipped), out, atol=_ATOL)
    zeroed = x.clone()                      # the blind ablation: slice zeroing
    zeroed[:, L.LANG_SLICE] = 0.0
    assert not torch.allclose(enc(zeroed), out, atol=_ATOL)


def test_state_lang_sensitivity_valid_vs_masked():
    """Changing a VALID agent's lang vector changes the output; changing a
    MASKED agent's lang vector is bit-invisible (the binding rides the
    presence mask)."""
    enc = _seeded_encoder("state", seed=53)
    x = make_state(4, _gen(53), agents_valid=(0, 1))
    out = enc(x)
    valid = x.clone()
    valid[:, L.state_lang_slice(1)] = _unit(_gen(54), 4, L.LANG_DIM)
    assert not torch.allclose(enc(valid), out, atol=_ATOL)
    masked = x.clone()
    masked[:, L.state_lang_slice(3)] = _unit(_gen(55), 4, L.LANG_DIM)
    assert torch.equal(enc(masked), out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def _grad_check(net, x):
    net.zero_grad(set_to_none=True)
    net(x).square().sum().backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, "no grad for: %r" % (missing,)
    bad = [n for n, p in net.named_parameters()
           if not torch.isfinite(p.grad).all()]
    assert not bad, "non-finite grads: %r" % (bad,)
    zero = [n for n, p in net.named_parameters() if p.grad.abs().sum() == 0]
    return zero


def test_gradient_flow_policy_obs():
    torch.manual_seed(61)
    net = M.EntityPolicyNet(layout="obs")
    zero = _grad_check(net, make_obs(8, _gen(61), teammates_valid=(0, 1)))
    # with valid entities in the batch, every stream must receive signal
    assert not zero, "all-zero grads for: %r" % (zero,)


def test_gradient_flow_value_state():
    torch.manual_seed(62)
    net = M.EntityValueNet(layout="state")
    zero = _grad_check(net, make_state(8, _gen(62)))
    assert not zero, "all-zero grads for: %r" % (zero,)


def test_gradient_flow_policy_full_state():
    torch.manual_seed(63)
    net = M.EntityPolicyNet(layout="full_state_obs")
    zero = _grad_check(net, make_full_state_obs(8, _gen(63)))
    assert not zero, "all-zero grads for: %r" % (zero,)


# ---------------------------------------------------------------------------
# Parameter parity vs the flat 512-256-128 baseline (spec 1.14)
# ---------------------------------------------------------------------------

def test_baseline_param_formula():
    # hand-computed: 376*512+512 + 512*256+256 + 256*128+128 + 128*5+5
    assert M.flat_mlp_param_count(L.OBS_DIM, out_dim=5) == 357893
    assert M.flat_mlp_param_count(L.STATE_DIM, out_dim=1) == 493057
    assert M.flat_mlp_param_count(L.FULL_STATE_OBS_DIM, out_dim=5) == 495621


def test_parameter_parity_same_order_of_magnitude():
    text, rows = M.parameter_parity_report()
    assert len(rows) == 3 and "baseline" in text
    for name, ours, base in rows:
        ratio = ours / float(base)
        assert 0.1 <= ratio <= 10.0, (name, ours, base, ratio)


# ---------------------------------------------------------------------------
# skrl guard / skrl integration (whichever applies on this box)
# ---------------------------------------------------------------------------

class _FakeMarlEnv(object):
    """Duck-typed skrl-wrapper surface for build_entity_models."""

    possible_agents = ["robot_0", "robot_1"]
    num_envs = 4
    device = "cpu"
    observation_spaces = {a: L.OBS_DIM for a in possible_agents}
    action_spaces = {a: M.N_ACTIONS for a in possible_agents}
    state_spaces = {a: L.STATE_DIM for a in possible_agents}


def test_skrl_classes_guarded_or_constructible():
    if not M.SKRL_AVAILABLE:
        for cls in (M.CategoricalEntityPolicy, M.DeterministicEntityValue):
            try:
                cls(L.OBS_DIM, M.N_ACTIONS, device="cpu")
            except ImportError as exc:
                assert "skrl" in str(exc), exc
            else:
                raise AssertionError("%s must raise without skrl" % cls)
        return
    # skrl present (5090): construct and run compute()
    torch.manual_seed(71)
    pol = M.CategoricalEntityPolicy(L.OBS_DIM, M.N_ACTIONS, device="cpu")
    val = M.DeterministicEntityValue(L.STATE_DIM, M.N_ACTIONS, device="cpu")
    g = _gen(71)
    logits, _ = pol.compute({"states": make_obs(3, g)}, role="policy")
    value, _ = val.compute({"states": make_state(3, g)}, role="value")
    assert logits.shape == (3, M.N_ACTIONS) and value.shape == (3, 1)


def test_build_entity_models_guarded_or_working():
    if not M.SKRL_AVAILABLE:
        try:
            M.build_entity_models(_FakeMarlEnv())
        except ImportError as exc:
            assert "skrl" in str(exc), exc
            return
        raise AssertionError("build_entity_models must raise without skrl")
    models = M.build_entity_models(_FakeMarlEnv(), device="cpu")
    assert sorted(models) == ["robot_0", "robot_1"]
    for aid, pair in models.items():
        assert sorted(pair) == ["policy", "value"]
        # no weight sharing anywhere: all encoder objects are distinct
        assert pair["policy"].net.encoder is not pair["value"].net.encoder
    p0 = models["robot_0"]["policy"].net.encoder
    p1 = models["robot_1"]["policy"].net.encoder
    assert p0 is not p1


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
