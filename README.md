# Team Listen

**Does the Team Actually Listen? A Causal Audit of Language Grounding in
Language-Conditioned Multi-Robot Policies.**

Single-arm audits (LIBERO-Plus, LangGap) showed that Vision-Language-Action
policies largely ignore their instructions. No such audit exists for robot
*teams*, where language carries semantics a single arm cannot express (role
binding over identical targets, precedence, exclusion, mid-episode
reassignment) and where every current multi-robot VLA/VLM system reports only
aggregate success, the exact condition under which the language channel can be
silently inert. This project builds the task family, the construct-validity
device, and the metric that make language grounding in multi-robot policies a
measured quantity.

Fully simulation-based: Isaac Lab (DirectMARLEnv, PettingZoo Parallel API,
skrl MAPPO/HAPPO) on a single RTX 5090. Everything language-side is frozen;
only small MARL policies are trained.

## Contributions (planned)

1. A cooperative Isaac Lab task family where every task ships a
   machine-checked **language-necessity certificate**: an instruction-blind
   full-state MAPPO oracle whose assignment accuracy must sit at chance.
2. The **Coordination Sensitivity Index (CSI)**: behavioural divergence under
   a counterfactual instruction intervention, normalised by divergence under
   a physics-seed change alone, making counterfactual auditing survive MARL
   stochasticity.
3. A pre-registered **dissociation** headline: per-agent prompting is
   language-sensitive on role binding but near-inert on precedence, because
   emergent collision avoidance satisfies precedence for free.
4. A controlled four-way comparison of language-conditioning interfaces
   (global broadcast / per-agent atomic prompts / frozen VLM commander via a
   distilled symbolic sub-goal vocabulary / frozen sentence embedding), with
   evidence that CSI predicts held-out generalisation, released as an open
   harness, plus a cross-fidelity robustness arm (RTX rendering, real VLM in
   the loop at evaluation, raycast sensing and localisation noise).

## Repository layout

| Path | Contents |
|---|---|
| `docs/RESEARCH_PLAN.md` | Full 12-month plan: milestones, gates, evaluation protocol, risks |
| `reference/` | Read-only imports from the rendezvous project: `env_paper.py` (grid fleet env to port), `test_env_paper.py`, `bridge.py` and `isaac_env.py` (Isaac 5.1 continuous harness) |
| `tasks/` | Isaac Lab DirectMARLEnv task family (role binding, precedence, exclusion, reassignment) |
| `harness/` | Certificate trainer, CSI computation, perturbation suite, statistics |
| `scripts/` | Training/evaluation entry points |

## Milestone 1 (months 1 to 1.5), the go/no-go gate

1. Port the grid fleet environment (`reference/env_paper.py` semantics) into
   an Isaac Lab DirectMARLEnv at N=2 with two task variants, identical-target
   role binding and precedence.
2. Train the instruction-blind full-state MAPPO oracle (5 seeds) and iterate
   task design until assignment accuracy sits at chance with a tight CI (the
   certificate).
3. Benchmark env FPS with a 3B VLM in the loop versus the cached symbolic
   stand-in (locks the commander architecture).
4. Train one sentence-embedding-conditioned policy. GO if the certificate
   holds and success is high with near-zero CSI on precedence. PIVOT to the
   architecture-comparison framing if CSI is already high.

## Setup on the training machine (5090)

Isaac Sim 5.1 standalone is already installed
(`~/isaac-sim-standalone-5.1.0-linux-x86_64`). Isaac Lab is required on top:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
ln -s ~/isaac-sim-standalone-5.1.0-linux-x86_64 _isaac_sim
./isaaclab.sh --install skrl
```

Then clone this repo and see `docs/RESEARCH_PLAN.md`.
