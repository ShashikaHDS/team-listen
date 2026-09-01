"""Reward assembly for TeamGridEnv (M1_SPEC 1.12) -- pure torch, audit-clean.

Spec 1.12 reward table, per decision step:

    step cost               -0.01 per team per step
    obstacle collision      -0.25 per robot per step
    robot-robot collision   -0.25 per robot per step
    shaping                 lambda * (gamma * Phi_t - Phi_{t-1}),
                            lambda = 0.1, gamma = 0.99,
                            Phi = -(min-cost perfect matching of latch-aware
                            BFS distances), Phi(terminal) == 0
    completion              +2.0 on the terminal both-latched step
    outcome bonus           +10 * Y on that same step -- the ONLY
                            instruction-dependent term (see below)

The per-team scalar is broadcast to both agents and only the two collision
penalties are per-agent (spec 1.12 "per-team scalar broadcast to both
agents, plus a per-agent collision term"); the result is the per-agent
reward dict MAPPO consumes, keyed in ``cfg.possible_agents`` order
(spec 1.4 STABLE ORDER -- load-bearing).

Red-team fixes honoured here:

* [FIXED: the shaping was not policy-invariant] -- the gamma-correct
  potential difference ``lambda * (gamma * Phi_t - Phi_{t-1})``.  The
  undiscounted form ``lambda * (Phi_t - Phi_{t-1})`` adds
  ``lambda * (1 - gamma) * Phi_t`` per step: a genuine penalty
  proportional to remaining distance, i.e. exactly the anti-waiting
  confound spec 3.4 promises to have excluded.  Under the correct form
  the discounted shaping sum telescopes to
  ``lambda * (gamma^T * Phi_T - Phi_0)`` and is path-independent given
  identical terminal Phi (spec 1.12 [FIXED: reward-fairness bound]).
* [FIXED: latch-aware distance field] -- Phi comes from
  ``grid_core.matching_potential`` over the bank's LATCH-AWARE BFS fields
  (``dist_field[k, j]`` is BFS from target j with every OTHER target cell
  treated as an obstacle, spec 1.10).  An all-free field routed shaping
  paths THROUGH the other absorbing station and actively rewarded walking
  into the wrong station, locking Y = 0; the latch-aware field scores such
  routes unreachable instead, so the shaping cannot drive agents into
  wrong stations.
* [FIXED: Phi signature contradiction] -- Phi is instruction-free.  This
  module never references the instruction / assignment / leak fields of
  spec 1.3 (statically enforced by ``tests/test_potential_purity.py``).
  The one legal instruction-dependent term, the +10 * Y outcome bonus,
  therefore enters ONLY as a precomputed tensor argument (``correct``):
  Y itself is computed by the env per spec 2.3 / 3.3, never here.
* [FIXED: the blind oracle was trained against an unpredictable +-10
  term] -- ``use_expected=True`` (the Blind / Mute arms) replaces the
  stochastic outcome bonus with its exact conditional expectation +5.0 on
  the both-latched step, which is bias-free for a policy that cannot
  predict Y and preserves the return scale.  Every other arm keeps the
  real stochastic bonus.
* ``Phi(terminal) == 0`` applies to TERMINATION (both robots latched)
  only.  Timeout is truncation, not termination: Phi stays live and the
  agent's ``time_limit_bootstrap: True`` handles the value (spec 1.4 /
  1.12).

Import surface (pinned by ``tests/test_potential_purity.py``): ``torch``
and the pure sibling ``grid_core`` only.  The constants below mirror the
frozen values in ``fleet_env.py``; ``tests/test_rewards.py`` asserts the
two modules never drift apart (``fleet_env`` cannot import from here
without pulling reward code into the Isaac shell's import graph, and this
module must not import ``fleet_env``, so the values are stated twice and
machine-checked).
"""

import torch

from . import grid_core

# ---------------------------------------------------------------------------
# Frozen reward constants (spec 1.12 table). shaping_lambda / reward_gamma
# defaults live on the cfg too (lambda is swept pre-headline, OPEN(5)); the
# defaults here are the spec values.
# ---------------------------------------------------------------------------

STEP_COST = -0.01               # per team, per step
COLLISION_COST = -0.25          # per robot, per step, obstacle AND robot-robot
COMPLETION_BONUS = 2.0          # on the terminal both-latched step
OUTCOME_BONUS = 10.0            # +10 * Y, the ONLY instruction-dependent term
BLIND_EXPECTED_BONUS = 5.0      # exact E[10 * Y | C=1] for a blind policy
SHAPING_LAMBDA = 0.1            # spec 1.12 lambda (swept {0.05, 0.1, 0.2})
REWARD_GAMMA = 0.99             # spec 1.12 gamma == YAML discount_factor

#: DECISIONS.md amendment (2026-09-01, terminal-credit fix): +2.0 to an
#: agent the FIRST time it latches in an episode.  Instruction-free by
#: construction -- computed from the latch-state transition alone (any
#: valid station counts; a wrong-station latch is paid identically), so
#: correctness stays priced ONLY by the +-10*Y outcome term and the
#: leakage-audit purity argument (spec 2.4) is untouched.  Motivated by
#: docs/PILOT_RESULTS.md section 4: policies learned to approach stations
#: and park, never sampling the completion payoff; this term densifies the
#: terminal credit at the single decisive event (entering an alcove).
FIRST_LATCH_BONUS = 2.0


# ---------------------------------------------------------------------------
# Individual terms
# ---------------------------------------------------------------------------

def terminal_potential(phi, done):
    """Apply the ``Phi(terminal) == 0`` convention (spec 1.12).

    ``done`` is TERMINATION (both robots latched), never timeout: timeout
    is truncation, the potential stays live and ``time_limit_bootstrap``
    handles the value (spec 1.4 / 1.12).

    Args:
        phi:  (E,) float raw potential at t.
        done: (E,) bool both-latched termination mask.

    Returns:
        (E,) float: ``phi`` with terminal entries zeroed.
    """
    return torch.where(done, torch.zeros_like(phi), phi)


def shaping_reward(phi_now, phi_prev, shaping_lambda=SHAPING_LAMBDA,
                   gamma=REWARD_GAMMA):
    """Gamma-correct potential shaping ``lambda * (gamma*Phi_t - Phi_{t-1})``.

    [FIXED: not policy-invariant] -- this is the Ng-style invariant form;
    it telescopes (discounted) to ``lambda * (gamma^T Phi_T - Phi_0)`` and
    cannot change the optimal policy, only variance and the
    credit-assignment path (OPEN(5)).  ``phi_now`` must already carry the
    terminal convention (``terminal_potential``).

    Args:
        phi_now:  (E,) float Phi_t (terminal entries already zeroed).
        phi_prev: (E,) float Phi_{t-1} (Phi_0 at spawn for the first step).

    Returns:
        (E,) float shaping term.
    """
    return shaping_lambda * (gamma * phi_now - phi_prev)


def completion_reward(done):
    """+2.0 on the terminal both-latched step, 0 elsewhere (spec 1.12)."""
    zeros = torch.zeros(done.shape, dtype=torch.float32, device=done.device)
    return torch.where(done, torch.full_like(zeros, COMPLETION_BONUS), zeros)


def outcome_reward(done, correct, use_expected=False):
    """The outcome bonus on the terminal both-latched step (spec 1.12).

    ``correct`` is the precomputed Y tensor (spec 2.3 / 3.3), handed in by
    the caller: this module is statically forbidden from computing it
    (tests/test_potential_purity.py), which is what keeps the reward
    assembly audit-clean.

    * ``use_expected=False``: ``+OUTCOME_BONUS * correct`` -- the real
      stochastic bonus (every arm except Blind / Mute, ALL Leaky rho
      cells included, spec 1.12 [FIXED]).
    * ``use_expected=True``: ``+BLIND_EXPECTED_BONUS`` regardless of
      ``correct`` -- the exact conditional expectation for an
      instruction-blind policy; removes a variance-25 terminal term the
      policy structurally cannot predict.

    Args:
        done:    (E,) bool both-latched termination mask.
        correct: (E,) bool/float Y tensor (ignored, may be None, when
                 ``use_expected`` is True).

    Returns:
        (E,) float32 bonus, nonzero only where ``done``.
    """
    zeros = torch.zeros(done.shape, dtype=torch.float32, device=done.device)
    if use_expected:
        bonus = torch.full_like(zeros, BLIND_EXPECTED_BONUS)
    else:
        assert correct is not None, (
            "outcome_reward needs the precomputed Y tensor unless "
            "use_expected=True (M1_SPEC 1.12)")
        bonus = OUTCOME_BONUS * correct.to(zeros.dtype)
    return torch.where(done, bonus, zeros)


def collision_reward(hit_obstacle, hit_robot):
    """Per-robot collision penalty: -0.25 per flag per step (spec 1.12).

    Args:
        hit_obstacle: (E, N) bool from ``grid_core.step_positions``.
        hit_robot:    (E, N) bool from ``grid_core.step_positions``.

    Returns:
        (E, N) float32 penalty (obstacle and robot terms are additive).
    """
    return COLLISION_COST * (hit_obstacle.float() + hit_robot.float())


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def team_reward(phi, phi_prev, done, correct, shaping_lambda=SHAPING_LAMBDA,
                gamma=REWARD_GAMMA, use_expected=False):
    """The per-team scalar of spec 1.12 (everything except collisions).

    step cost + gamma-correct shaping + (completion + outcome bonus) on
    the terminal both-latched step.  ``phi`` is the RAW potential at t;
    the ``Phi(terminal) == 0`` convention is applied here.

    Args:
        phi:      (E,) float raw ``matching_potential`` value at t.
        phi_prev: (E,) float potential at t-1 (Phi_0 at spawn).
        done:     (E,) bool both-latched termination mask.
        correct:  (E,) precomputed Y tensor (see ``outcome_reward``).

    Returns:
        (E,) float32 team reward.
    """
    phi_now = terminal_potential(phi, done)
    team = (STEP_COST
            + shaping_reward(phi_now, phi_prev, shaping_lambda, gamma)
            + completion_reward(done)
            + outcome_reward(done, correct, use_expected))
    return team.to(torch.float32)


def first_latch_reward(newly_latched):
    """``+FIRST_LATCH_BONUS`` per agent on the step it first latches.

    ``newly_latched`` is the (E, N) bool latch-state TRANSITION mask
    (``latched_after & ~latched_before``): it is True at most once per
    agent per episode because ``latched`` is absorbing (spec 1.11), so the
    bonus structurally fires once.  Reads latch state only -- never
    instruction, assignment, or leak fields (see FIRST_LATCH_BONUS note).
    """
    return FIRST_LATCH_BONUS * newly_latched.float()


def per_agent_rewards(agent_order, team, hit_obstacle, hit_robot,
                      newly_latched=None):
    """Broadcast the team scalar + per-agent collision and first-latch
    terms into the MAPPO reward dict (spec 1.12 + DECISIONS.md amendment).

    Args:
        agent_order:  sequence of agent names, in ``cfg.possible_agents``
                      order (spec 1.4 STABLE ORDER -- load-bearing).
        team:         (E,) float team reward from ``team_reward``.
        hit_obstacle: (E, N) bool.
        hit_robot:    (E, N) bool.
        newly_latched: (E, N) bool latch transition mask, or None (no
                      first-latch term -- pre-amendment behaviour).

    Returns:
        dict name -> (E,) float32, keys in ``agent_order`` order.
    """
    coll = collision_reward(hit_obstacle, hit_robot)
    assert coll.shape[-1] == len(agent_order), (
        "collision flags cover %d agents but agent_order names %d"
        % (coll.shape[-1], len(agent_order)))
    latch = (first_latch_reward(newly_latched) if newly_latched is not None
             else torch.zeros_like(coll, dtype=torch.float32))
    out = {}
    for i, name in enumerate(agent_order):
        out[name] = (team + coll[:, i] + latch[:, i]).to(torch.float32)
    return out


def compute_rewards(agent_order, dist_field, pos, target_valid, phi_prev,
                    done, correct, hit_obstacle, hit_robot,
                    shaping_lambda=SHAPING_LAMBDA, gamma=REWARD_GAMMA,
                    use_expected=False, newly_latched=None):
    """Full spec 1.12 reward assembly for one decision step.

    Computes Phi_t via the instruction-free ``grid_core.matching_potential``
    over the latch-aware fields, assembles the team scalar and the
    per-agent dict, and returns the RAW Phi_t so the caller can store it as
    Phi_{t-1} for the next step (mirroring ``fleet_env``'s ``_phi`` /
    ``_phi_prev`` bookkeeping: the stored value is the raw potential; the
    terminal zeroing lives only inside the reward).

    Args:
        agent_order:  agent names in ``cfg.possible_agents`` order.
        dist_field:   (E, MAX_TARGETS, R, C) latch-aware BFS fields
                      (bank rows gathered per env, spec 1.10).
        pos:          (E, 2, 2) post-conflict robot positions (LIVE slots).
        target_valid: (E, MAX_TARGETS) bool presence mask.
        phi_prev:     (E,) float potential at t-1 (Phi_0 at spawn).
        done:         (E,) bool both-latched termination mask (spec 1.11).
        correct:      (E,) precomputed Y tensor (see ``outcome_reward``);
                      may be None when ``use_expected`` is True.
        hit_obstacle: (E, N) bool from ``grid_core.step_positions``.
        hit_robot:    (E, N) bool from ``grid_core.step_positions``.
        newly_latched: (E, N) bool latch transition mask (or None).

    Returns:
        (rewards, phi): the per-agent reward dict, and the RAW (E,) float32
        Phi_t to carry forward as the next step's ``phi_prev``.
    """
    assert done.dtype == torch.bool, "done must be the bool termination mask"
    phi = grid_core.matching_potential(dist_field, pos, target_valid)
    team = team_reward(phi, phi_prev, done, correct, shaping_lambda, gamma,
                       use_expected)
    return per_agent_rewards(agent_order, team, hit_obstacle, hit_robot,
                             newly_latched), phi


__all__ = [
    "STEP_COST", "COLLISION_COST", "COMPLETION_BONUS", "OUTCOME_BONUS",
    "BLIND_EXPECTED_BONUS", "SHAPING_LAMBDA", "REWARD_GAMMA",
    "FIRST_LATCH_BONUS",
    "terminal_potential", "shaping_reward", "completion_reward",
    "outcome_reward", "collision_reward", "team_reward",
    "first_latch_reward", "per_agent_rewards", "compute_rewards",
]
