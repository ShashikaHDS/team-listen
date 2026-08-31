# Publication plan

Venue study run Aug 2026 (five scouts over journal scopes, precedent, and transactions expectations; full transcripts in the session workflow directory).

## Venue ranking

| Fit | Venue | Note |
|---|---|---|
| 8 | IEEE Transactions on Artificial Intelligence (TAI) | Only IEEE venue with live precedent for a no-new-algorithm evaluation paper (McIntosh et al. 2026; Cheng et al. 2022), no hardware norm, ~12-week first decision, matches the author's IEEE track record - but 10 free / 15 hard pages and a scope list that never says robotics make framing and compression the whole game. |
| 7 | IEEE Transactions on Robot Learning (T-RL) | Best topical fit anywhere (explicitly welcomes benchmarks and explainable robot learning, 12 pages, double-anonymous, ICRA/CoRL presentation), but its Information for Authors states in writing that real-hardware evaluation 'in addition or to complement simulation' is expected - the one sentence that can kill a fully sim-only paper; worth a pre-submission enquiry to the EiC before committing. |
| 6 | IEEE Transactions on Neural Networks and Learning Systems (TNNLS) | Highest impact of the realistic set, with precedent for purely empirical studies and for language-conditioned robot learning, but demands a learning-systems spine (representation probes, gradient pathways per interface) plus 'non-incremental archival theory' - a real reframing cost on top of the same 10/15-page squeeze. |
| 6 | IEEE Transactions on Cognitive and Developmental Systems (TCDS) | Same-society fallback whose scope explicitly names cognitive robotics, multi-agent systems and symbol grounding, so language-grounding audit reads as core rather than as an application - lower visibility and impact than TAI/TNNLS, but the safest CIS landing zone if TAI's AE routing rejects the robotics framing. |
| 6 | Transactions on Machine Learning Research (TMLR) | Perfect shape fit - no page limit, no SOTA requirement, dissociation/negative results explicitly protected by the acceptance criteria - and the only venue where all five arms fit as primary; costs the IEEE branding and the impact factor the author wants, so it is insurance rather than the target. |
| 5 | IEEE Transactions on Automation Science and Engineering (T-ASE) | 12 pages, simulation tolerated, AI/ML and modelling-and-simulation in scope - but the mandatory Note to Practitioners forces an industrial referent onto the task family, and the maximum-one-revise-and-resubmit rule means the cross-fidelity arm must be finished at first submission or never. |
| 5 | Robotics and Autonomous Systems (Elsevier) | No punishing page cap and a publication mix historically friendly to simulation studies and MARL reviews, making it the most mechanically comfortable robotics home - but the sim-only and benchmark policies could not be verified (403 to automated fetch) and it reads as a step down from an IEEE Transactions line for the author's CV. |
| 4 | IEEE Transactions on Robotics (T-RO) | The only IEEE venue that natively fits 12-18 pages of metric theory plus five arms, and its scope sentence lists 'analysis' standing alone - but it accepts no supplementary files during review, has no benchmark article category, and its reviewer pool operationalises 'major advance' as hardware; a single 'no real-robot experiments' review is terminal. |
| 4 | IEEE Robotics and Automation Letters (RA-L) | Fastest and most certain decision (guaranteed within 6 months) and the incumbent journal for language-conditioned robot teams, but 6+2 pages cannot hold certificate + CSI theory + four interfaces + generalisation + cross-fidelity; useful only as a spin-off for one arm. |
| 5 | Journal of Autonomous Agents and Multi-Agent Systems (JAAMAS) | Natural journal counterpart to the AAMAS cooperative-MARL benchmarking literature, no page cap, coordination-semantics framing is native there - but slow, non-IEEE, and near-invisible to the embodied-AI audience the paper argues with. |
| 3 | IEEE Transactions on Cybernetics (T-CYB) | Reviewer pool is bipartite-consensus and prescribed-time-formation control that will ask for stability proofs this paper structurally cannot supply, LLM papers there are non-embodied, and the journal's own stated seven-to-ten-month review would consume the entire project window. |
| 1 | Journal of Field Robotics (JFR) | Scope is robotics validated by extended field experiments in unstructured real environments; a sim-only Isaac Lab audit is a desk rejection by definition - listed only to close it off. |

## Recommendation

Primary target: IEEE TAI, submitted as a measurement-methodology paper - "a validity certificate and a normalised sensitivity measure for language-conditioned multi-agent policies" - with the Isaac Lab family as the instrument's substrate, never as the headline; TAI is the only IEEE venue that has both published a no-new-algorithm evaluation paper (McIntosh et al., TAI Jan 2026) and imposes no hardware norm, and its ~12-week first decision fits a solo 12-month runway in a way T-CYB's stated 7-10 months does not. Run one cheap hedge in parallel at month 5: a pre-submission enquiry to the T-RL EiC asking whether a cross-fidelity sim-to-sim arm with a real VLM in the loop satisfies their written "real hardware expected" clause - if the answer is yes, T-RL becomes primary because its scope explicitly welcomes benchmarks and explainable robot learning and it carries a CoRL/ICRA presentation slot, and if the answer is no or silent, you have lost two weeks and gained certainty. Backup chain on a TAI reject: TCDS (same society, cognitive-robotics and symbol-grounding scope, so an AE-routing rejection at TAI is not a content rejection) → TNNLS only if you are willing to spend 4-6 weeks adding representation-probing analysis so the paper is about the network and not only the protocol → TMLR as the terminal fallback, where the dissociation finding and the un-compressed five-arm structure are actually welcome. Be honest with yourself about two costs: the calendar (12 weeks to first decision plus one or two revision rounds is 9-14 months past submission, so acceptance lands well outside the project window and the arXiv preprint is what the community will actually read), and sim-only exposure (it is a non-issue at TAI/TCDS/TMLR, a probable rejection at T-RO/RA-L/JFR, and an open question at T-RL - which is exactly why the enquiry is worth sending rather than guessing). Deliberately do not chase T-RO: 18 pages of room is seductive, but no supplementary files during review plus a hardware-trained reviewer pool plus a "major advance" bar is the worst possible combination for a compute-constrained solo sim-only audit. Finally, freeze scope by month 6 and post arXiv v1 by month 7: the language-audit subgenre is producing preprints monthly (LIBERO-Plus/PRO/Para, LangGap, RoboSemanticBench), and a 12-month embargo forfeits priority on the one thing that is genuinely yours, which is CSI's predictive validity in a multi-agent coordination setting rather than the already-established observation that policies ignore language.

## Paper outline (transactions length)

### I. Title, Abstract, and Impact Statement

Title leads with the measure and the validity condition, not with Isaac Lab or multi-robot policies - e.g. 'Coordination Sensitivity: A Certified Measure of Language Use in Multi-Agent Policies' - because the single most likely fast-fail at TAI is administrative routing to a RAS journal. Abstract orders the contributions as certificate, measure, predictive validity, and only then the dissociation finding. The mandatory <=150-word Impact Statement addresses assurance: unverified language conditioning in deployed multi-robot systems is a safety claim nobody currently tests.

*Evidence from:* M6 (wording), gated by M2 and M4 results

### II. Introduction and Enumerated Contributions

Frames the gap as a measurement problem: the field ships language-conditioned multi-robot policies with no procedure for deciding whether the language channel is load-bearing, and success-rate deltas conflate task difficulty with instruction sensitivity. Ends with an IEEE-style enumerated contribution list (C1 certificate procedure, C2 CSI and its estimator, C3 formal properties and design-space analysis, C4 four-interface validation study, C5 predictive validity, C6 cross-fidelity reproduction) and an explicit paragraph on why the causal claim is identifiable only under controlled simulation - matched physics seeds across counterfactual instructions cannot be executed on hardware, which is an argument for simulation rather than an apology.

*Evidence from:* M6, framed from M1 and M2

### III. Related Work

Four subsections: language-grounding audits for single-robot VLAs (contrast sets, LIBERO-Plus/PRO/Para, LangGap, RoboSemanticBench, shortcut learning), MARL evaluation protocols and benchmarking (BenchMARL, the NeurIPS-2022 cooperative-MARL protocol, rliable), metric-as-contribution papers as a genre (HOTA, causal-learning evaluation measures), and language-conditioned multi-robot planning. The wedge is stated in one sentence and defended: coordination semantics (role binding, precedence) have no single-robot analogue, and CSI is predictive rather than merely diagnostic. Cite IEEE-published MARL work here, not only CoRL/NeurIPS, or the AE reads venue mismatch.

*Evidence from:* M6, with the comparison set fixed during M3

### IV. Problem Formalisation and Interventional Semantics

Defines the language-conditioned Dec-POMDP with an instruction channel, the conditioning interface as a map from instruction to per-agent policy input, and the counterfactual-instruction intervention as a do-operation on that channel with physics seed, initial state and team composition held fixed. States precisely what is intervened on, what is held fixed, and the identifying assumptions under which a behavioural divergence ratio is a causal effect rather than an association - this section exists specifically to survive the reviewer who objects to the word 'causal'.

*Evidence from:* M1

### V. Language-Necessity Certificates

Presents the instruction-blind full-state oracle as a decision procedure with a stated hypothesis class, a training budget that makes 'the oracle failed' informative, a chance-level null, a one-sided test with confidence intervals, and multiple-comparison correction across tasks in the family. Includes a soundness statement in both directions: a chance-level oracle certifies the task as language-necessary, and explicitly does not certify that any policy uses language. Reports a false-certificate rate estimated on deliberately language-redundant control tasks so the certificate is calibrated rather than asserted.

*Evidence from:* M1

### VI. The Coordination Sensitivity Index: Definition and Estimator

Defines CSI as counterfactual-instruction behavioural divergence normalised by physics-seed divergence, over a named divergence on a named object (state-occupancy or trajectory distribution, decided and defended, not left implicit). Gives the finite-sample estimator as an algorithm box over paired rollouts, with the seed and instruction-pair design that generates each term, plus the regularised form used when the denominator is small.

*Evidence from:* M2

### VII. Formal Properties of CSI

Numbered propositions rather than prose: range and boundedness; behaviour and regularisation in the degenerate near-zero-denominator regime; invariance to agent permutation, episode horizon and policy re-parameterisation, with counterexamples where invariance fails; and an identifiability statement saying exactly what CSI approximately zero entails (an inert channel) versus what it does not (an unidentifiable or saturated regime). Closes with estimator bias and confidence-interval width as a function of the number of counterfactual pairs and physics seeds, plus the sample-size table that justifies the project's seed budget. This is the section that converts the paper from a benchmark into a Transactions contribution and it is not optional.

*Evidence from:* M2, empirically checked in M4

### VIII. Analysing the Design Space of CSI

HOTA-style: one subsection per metric design choice - which divergence, normalise by physics seed versus by a random-instruction baseline, per-timestep versus whole-trajectory, action space versus state occupancy, mean versus geometric mean across agents - each demonstrated on stored rollouts to show why the alternative misranks policies. Terminates in an explicit 'Limitations and drawbacks of CSI' subsection, whose absence journal reviewers read as an unfinished metric.

*Evidence from:* M2 for the choices, M3 for the rollout corpus

### IX. Problems With Existing Language-Sensitivity Measures

Scores prior proxies - instruction-swap success drop, instruction-blind baselines, attention/attribution scores, contrast-set accuracy - against the same property list used in Section VII, with enumerated failure modes computed on the paper's own policies so the criticism is empirical rather than rhetorical. This is what turns 'we proposed a metric' into 'we proposed the right metric', and it is also the cleanest pre-emption of the 'CSI is a normalised success-rate delta' review.

*Evidence from:* M3

### X. Task Family, Interfaces, and Experimental Protocol

Compact specification of the cooperative Isaac Lab family, the role-binding and precedence instruction semantics, and a taxonomy table of the four conditioning interfaces along fixed axes (bottleneck width, per-agent addressability, grounding locus, inference cost, credit-assignment path), plus a second table positioning the family against existing language-conditioned and cooperative-MARL benchmarks. Protocol paragraph states MAPPO budgets held identical across interfaces, the hyperparameter-search protocol shared across arms, seeds per cell, compute per cell, and the timestamped pre-registration with H1..Hk and their falsification criteria. Under a 10-15 page cap, per-task specs and hyperparameter tables move to supplementary; the taxonomy tables stay in body.

*Evidence from:* M3

### XI. Validation Study: Four Interfaces and the Pre-Registered Dissociation

The near-factorial grid of interface x task x team size x seed, reported with IQM and stratified-bootstrap intervals, performance profiles and probability of improvement rather than mean-final-return bars, with any omitted cell justified in text. The dissociation - per-agent prompting language-sensitive on role binding, near-inert on precedence - is presented as a discovery the instrument made, accompanied by a mechanistic hypothesis for why precedence is inert, and it is deliberately not the abstract's opening claim.

*Evidence from:* M2 and M3

### XII. Predictive Validity and a Repair Control

Rank correlation between CSI and held-out generalisation (unseen instruction-role compositions, unseen team size), with the policy as the unit of analysis, a stated number of distinct policies, a permutation null, and the relationship shown to survive partialling out in-distribution success rate - the confounder every reviewer will name. Adds the repair control: at least one targeted intervention that partially recovers precedence sensitivity, raising CSI and, as predicted, the held-out score. This is the strongest journal-grade evidence in the paper and it should be protected in the page budget before anything else.

*Evidence from:* M4

### XIII. Cross-Fidelity Robustness With a Frozen VLM Commander

Render-free training to RTX-rendered evaluation with raycast sensing, localisation noise given as equations with parameter values, and a real 3B VLM producing distilled sub-goal tokens, tested for whether certificate verdicts and CSI rankings are preserved across the fidelity shift. Carries a reproducibility contract: pinned Isaac Lab and renderer versions, exact checkpoint hash, decoding parameters, and a determinism statement. Framed as external-validity evidence and as the answer to 'no hardware', while stating plainly that it is sim-to-sim and not a hardware substitute; this is the first section to be relegated to supplementary if the page budget breaks.

*Evidence from:* M5

### XIV. Threats to Validity, Reproducibility and Release, and Conclusions

Enumerated threats: oracle undertraining as an alternative explanation for certificates, divergence-functional sensitivity, single-lab compute limiting team sizes and seeds, and the sim-only scope of every causal claim. Data and code availability statement with a Zenodo DOI (anonymous mirror during double-anonymous review, including no GitHub links anywhere in the PDF or supplementary), plus the deviations log against the pre-registration. Conclusion restates the instrument and its validity evidence, and names the one thing a practitioner does differently once CSI is measurable.

*Evidence from:* M5 and M6

## Additions versus a conference version

- Formal CSI properties section with numbered propositions - range/boundedness, degenerate-denominator regularisation, permutation and horizon invariance with counterexamples, and an identifiability statement for CSI approximately zero. This is the single biggest delta over a conference version and the thing that makes the paper archival. Cost: 3-4 weeks (analysis and writing, no new compute).
- Finite-sample estimator analysis: bias and variance of the divergence ratio, confidence-interval construction over the paired (instruction-pair, physics-seed) design, and a power analysis with a sample-size table that pre-justifies the seed budget. Do this BEFORE the big runs so the grid is powered at first submission. Cost: 2-3 weeks analysis, plus it sets the compute bill for M3.
- HOTA-style design-space section ablating the metric's own choices (divergence functional, physics-seed vs random-instruction normaliser, per-step vs trajectory, action vs state occupancy, mean vs geometric-mean agent aggregation), each shown to misrank policies, plus an explicit drawbacks subsection. Cheap in GPU time if rollouts are logged in a re-analysable form from day one; expensive if not. Cost: 4-5 weeks (mostly re-analysis and writing).
- A 'problems with existing measures' audit scoring 4-6 prior language-sensitivity proxies against the same property list, computed on the paper's own policies rather than argued rhetorically. Cost: 2-3 weeks.
- Certificate hardening from a script into a decision procedure: oracle hypothesis-class sweep, training-budget sensitivity curve showing the verdict is not an undertraining artifact, multiple-comparison correction across the task family, and a false-certificate rate measured on deliberately language-redundant control tasks. Cost: 3 weeks including the control-task runs.
- Survey-grade taxonomy depth: a fixed-axis comparison table over language-conditioning interfaces and a second table positioning the task family against 12-20 existing language-conditioned and cooperative-MARL benchmarks. Conference versions get away with a paragraph; Transactions reviewers ask for the table. Cost: 2-3 weeks.
- Near-factorial ablation grid at >=10 independent seeds per cell with rliable-grade statistics (IQM, stratified bootstrap, performance profiles, probability of improvement) and identical hyperparameter-search protocol across interfaces so the four-way comparison is not a tuning artifact. Cost: 6-8 weeks of RTX 5090 wall-clock (run in background across other writing), ~1 week of analysis.
- Predictive-validity statistics upgraded from a scatter plot to a design: policy-level unit of analysis, rank correlation with CIs, permutation null, and the relationship shown to survive partialling out in-distribution success rate. Cost: 1-2 weeks.
- A repair/intervention experiment for the inert-precedence result - a mechanistic hypothesis plus at least one intervention that partially restores precedence sensitivity. Without this the dissociation reads to an IEEE reviewer as 'the method does not work'. Cost: 4-6 weeks including training.
- Representation-level analysis of the four interfaces (where the instruction embedding enters actor/critic, gradient pathways, probing classifiers for role and precedence content). Optional for TAI, mandatory if the paper is redirected to TNNLS. Cost: 4-6 weeks - decide by month 6, not month 11.
- Reproducibility contract for the cross-fidelity arm: pinned Isaac Lab/renderer versions, exact VLM checkpoint hash and decoding parameters, determinism statement, sensor and localisation-noise models as equations, plus a Zenodo-DOI release and a de-identified anonymous mirror for double-anonymous review. Cost: 2 weeks.
- IEEE house-style apparatus a conference version does not carry: <=150-word Impact Statement, enumerated contributions list, theorem/proposition environments, algorithm boxes for the certificate check and CSI estimator, standalone related-work section, a supplementary file, a data/code availability statement, and a full de-anonymisation sweep of PDF, figures, supplementary and links. Cost: 1-2 weeks.
- Auditable pre-registration artifact: timestamped OSF or arXiv-v1 protocol enumerating H1..Hk with predictions and falsification criteria, a confirmatory-versus-exploratory table, and a deviations log, explained in one sentence of plain IEEE language rather than as a methodological innovation. Cost: 1 week at month 1, 0.5 week at write-up.

## Writing calendar

- Month 1 - Write the pre-registration protocol and post it timestamped (OSF or arXiv v1). Its technical body IS the first draft of Section IV (problem formalisation and interventional semantics) and Section V (certificate procedure), so registration costs almost nothing extra. Also fix the rollout-logging schema now: every design-space ablation in Section VIII is cheap re-analysis if logs are re-analysable and impossible if they are not.
- Months 2-3 - Draft Sections V and VI to completion as a standalone 'metric specification' document (certificate soundness argument, CSI definition, estimator algorithm box). Run the power analysis here and let it set the seed budget for M3 rather than discovering underpowering in review. Produce the Section X taxonomy table of the four interfaces while designing them.
- Month 4 - Write Section VII (formal properties) alongside M2, proving boundedness, degenerate-denominator behaviour and invariances on paper before the results exist. Build the Section XI results scaffolding with placeholder tables and frozen analysis scripts, so the confirmatory analysis is executable code committed before the data lands.
- Months 5-6 - Write Sections III (related work), IX (problems with existing measures) and X (task family and protocol). Send the T-RL pre-submission enquiry at the start of month 5 and LOCK the venue and page budget by the end of month 6 - the TNNLS representation-probing decision must be made here, not at month 11.
- Month 7 - Assemble and post arXiv v1 of the core measurement paper (Sections I-XI) to claim priority against the monthly preprint churn in the language-audit subgenre. Optionally submit a strict subset (certificate + CSI + dissociation) to AAMAS or CoRL, keeping >=30% new material in reserve for the IEEE version and planning the delta statement now.
- Months 8-9 - Write Section XII (predictive validity and repair control) directly from M4 outputs, and Section VIII (design-space ablation and drawbacks) from the stored rollout corpus. These two sections are the paper's strongest journal-grade evidence and the most likely to be rushed if left to month 11.
- Month 10 - Write Section XIII (cross-fidelity arm) and Section XIV (threats, reproducibility, conclusions) as M5 completes. Assemble the supplementary file, cut the Zenodo release with a DOI, and stand up the anonymous mirror. Decide explicitly which arm gets relegated to supplementary to hit the page cap - default relegation order: cross-fidelity, then task-family specs, then hyperparameters.
- Months 11-12 - Integration, compression to the locked page budget, Impact Statement, full double-anonymisation sweep, and one hostile self-review against the four predicted attacks (degenerate denominator, oracle undertraining, statistical power, VLM reproducibility). Submit at month 12, then immediately draft the TCDS/TNNLS reframe notes and the un-compressed TMLR packaging so a reject at month 15 costs days rather than months.