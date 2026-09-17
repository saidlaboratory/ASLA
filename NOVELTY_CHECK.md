# Novelty check: scaling-law selection as transductive pure exploration

**Verdict: GATE PASSES.** The connection is not made in the literature. Every component
exists separately; no work combines them, and the closest prior art on each axis
*assumes away* exactly the thing our audit measured.

Every claim below was checked against the source (abstract or full text), not against
search snippets. Where a search summary disagreed with the paper, the paper wins and the
discrepancy is recorded.

## 1. What exists

### 1a. Transductive linear bandits — the theory, with no application to scaling
**Fiez, Jain, Jamieson, Ratliff, "Sequential Experimental Design for Transductive Linear
Bandits", NeurIPS 2019 (arXiv:1906.08399).** Verified. Measurement set `X` and target set
`Z`, both finite subsets of R^d; linear model `y = θ*'x + η` with sub-Gaussian noise; goal
is to infer `argmax_{z∈Z} z'θ*` with probability `1−δ` using as few sequentially chosen
measurements as possible. Fixed-confidence. Instance-dependent lower bounds plus a matching
algorithm (up to log factors). Design objective:

    ρ* = argmin_λ max_{z,z'∈Z} ||z − z'||²_{A(λ)^{-1}} / gap²,  A(λ) = Σ_x λ(x) x x'

Motivating applications are drug discovery and recommender systems. **Costs are uniform** —
each measurement costs one sample; heterogeneous per-arm cost is nowhere considered.
**Misspecification is not addressed**; the linear model is assumed correct.

Forward citation graph checked for any training/scaling application: the citing work found
is methodological (e.g. "Experiment Planning with Function Approximation" arXiv:2401.05193,
"Experimental Design for Semiparametric Bandits" arXiv:2506.13390). No application to
scaling laws, pretraining, or recipe selection was found.

### 1b. Active experiment selection for scaling laws — estimation objective, no decision
**Li, Li, Lin, Sun, Talwalkar, Yang, "Spend Less, Fit Better: Budget-Efficient Scaling Law
Fitting via Active Experiment Selection" (arXiv:2604.22753; submitted 2026-04-24, revised
2026-08-12).** Verified against full text. *This is the live competitor and it is a clean
foil.* Its acquisition function is

    S(x) = (ΔV_intra(x) + ΔV_inter(x)) / c(x)^α

where `ΔV_intra + ΔV_inter` decompose the **target-region mean squared prediction error**.
The objective is predictive-variance minimisation for extrapolation accuracy. Each task
fits a **single** scaling law; the paper does not compare competing recipes or select a
winner. It cites classical optimal experiment design (D-optimality, A-optimality) and
**does not cite bandits, best-arm identification, pure exploration, or G-/XY-optimal
design.** No PAC or correctness guarantee. Baselines: random, cheapest, cost-random,
D-opt, V-opt, all-data. On misspecification it concedes only a limitation: "SL2 may be less
reliable under severe model misspecification, particularly when abrupt phase transitions
are neither visible in the pilot region nor represented by the chosen law family."

Note in their favour, and against a naive statement of our novelty: **their budget
constraint is already cost-weighted by FLOPs** (`c_i = c(x_i)`, "for dense-training
settings we use 6ND"). Cost-weighting per se is therefore *not* novel in the scaling-law
design setting. What is novel is cost-weighting inside a *decision* objective — see §3.

### 1c. Other active/efficient scaling-law construction — all estimation
- **"Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided
  Pruning" (arXiv:2605.17234)**, Schram, Hiller, Beck, Cohn. Successive Halving plus
  parametric/non-parametric surrogates to allocate budget for the loss–compute frontier.
  Success metric is its own words: "obtain accurate scaling laws at significantly reduced
  computational costs, saving up to 98.7%". Estimation, not selection.
  *Recorded discrepancy:* an automated summary of this paper claimed it ranks candidate
  interventions and targets "decision quality". The verbatim abstract does not support
  that; it seeks the best-loss frontier for one curve. The summary's reading was rejected.
- **"Amortizing Scaling Law Construction Costs" (arXiv:2609.05016)**. Bayesian optimisation
  with compute slicing and fantasization to recover the loss envelope cheaply. Explicitly
  not a ranking problem, no bandit framing, no guarantees.

### 1d. Hyperparameter optimisation bandits — a different problem
Successive Halving / Hyperband cast HPO as best-arm identification over configurations,
using low-fidelity (short-budget) runs to eliminate candidates. **The difference is precise
and must be stated precisely:** in SH/Hyperband the target budget *is sampled* — the
surviving arms are eventually run at full budget, and the returned arm's performance at the
target fidelity is observed. The known failure mode (SH may discard a configuration that
would win at full budget; Hyperband hedges with multiple brackets) is a *fidelity-ordering*
problem, not a transductive one. In our setting the target budget `C*` is **never sampled
by anyone, ever** — that is the entire premise. No amount of bracketing reaches it. This
makes ours transductive in the Fiez sense and makes the HPO guarantees inapplicable.

### 1e. Misspecification in fixed-confidence identification — assumes the bound is known
**Réda, Tirinzoni, Degenne, "Dealing With Misspecification In Fixed-Confidence Linear Top-m
Identification" (arXiv:2111.01479), NeurIPS 2021.** Verified abstract. Derives a tractable
lower bound for δ-correct Top-m identification under misspecified linear models, gives the
first algorithm that adapts to the *amount* of misspecification, matching the lower bound as
δ→0. Decisive sentence: **"We show that knowing the scale of the deviation from linearity is
necessary to exploit the structure of the problem."**

This is the most important single result for us, and it cuts in our favour. It is a theorem
that the misspecification scale must be known to get structure-exploiting guarantees. The
broader misspecified-bandit literature (arXiv:1704.06880 Misspecified Linear Bandits;
arXiv:2510.00073 ε-best arms in misspecified linear bandits; arXiv:2501.05361 gap-adjusted
misspecification) works predominantly with a **uniform sup-norm bound ε assumed known**,
with "average misspecification" as the main weakening. None estimates a *covariate-indexed*
misspecification function from data and validates it out-of-range.

### 1f. The decision question is posed in the literature — and left open
**"When does a scaling result justify a different allocation? A critical review of
resource-allocation evidence for AI systems" (arXiv:2609.14500).** A critical review, not a
methods paper. It asks our question and states the gap explicitly: no small-to-large audit
measuring whether results reverse at deployment scale, no mis-selection measurement, no
estimator or design proposal, no PAC bounds or abstention conditions. Its §8 research
agenda is, in effect, a call for what we built.

**DataDecide (arXiv:2504.11393)** measures decision accuracy (~80% at 150M→1B) and reports
that scaling-law baselines do not beat the single-scale ranking frontier. It is the source
of our estimand and our baseline, and it poses the compute-to-decision-accuracy frontier as
an empirical curve — but proposes no design, no allocation rule, and no guarantee.

**Conformal/abstention work** (arXiv:2512.09850 conformal bandits under weak arm
separability, verified; arXiv:2605.19779 distribution-free UQ for agent evaluation,
verified) applies at the **observed** scale. Neither extrapolates to an unsampled compute
budget, and neither does experimental design across scales.
*Recorded discrepancy:* a search engine synthesis asserted a paper doing "conformal
selective abstention ... PAC-style guarantees on incorrect ordering" for scaling
comparisons. Following it to the cited sources, no such paper was found; the two candidate
papers both operate at the observed scale. Treated as a search artifact, not prior art.

## 2. Closest prior art

Ranked by how much of our problem it covers:

1. **arXiv:2604.22753 (SL2)** — same domain, same cost-weighted budget, *estimation*
   objective, single law, no guarantees. Closest in setting.
2. **arXiv:1906.08399 (Fiez et al.)** — exactly our decision structure and design
   quantity, *uniform costs*, *well-specified*, unrelated domain. Closest in theory.
3. **arXiv:2111.01479 (Réda et al.)** — fixed-confidence identification adapting to
   misspecification magnitude, but the scale is assumed known and the model is not
   transductive across an unsampled covariate. Closest on misspecification.

No single work covers two of the three axes.

## 3. The precise delta

Stated as what would have to be added to the closest prior art:

- **To SL2:** replace MSPE of the extrapolated loss with the sign of
  `(θ_a − θ_b)'x(C*)` over all pairs — design for the decision, not the estimate. These
  objectives are not monotonically related: our own results show a fit whose projections
  move 24% of the spread while preserving the ordering exactly (Spearman 1.000000), and
  conversely a well-fit curve can flip an ordering inside its error bars. Minimising
  prediction error is neither necessary nor sufficient for selecting correctly.
- **To Fiez et al.:** (i) a **cost-weighted** design constraint where pulling arm
  `(recipe, C)` costs ∝ C, so the allocation trades many cheap low-`C` measurements against
  few expensive high-`C` ones — absent from transductive BAI, where all pulls cost 1;
  (ii) misspecification that is **measured, not assumed away**.
- **To Réda et al.:** the deviation scale is **not known** in our setting, and it is **not
  uniform** — we measured it to be compute-structured (a constant variance inflation φ
  fails: φ varies 2.7–8.9× across ladders, is monotone in lever arm and ladder length, and
  under-predicts held-out structure by 12×). Our contribution is to estimate the deviation
  as a *function of compute* on the fitting range and validate it on a held-out larger
  scale. Their lower bound says knowing the scale is necessary; we supply an estimate of it
  and test whether the estimate is good enough to carry a guarantee.
- **To DataDecide and arXiv:2609.14500:** a design and a stopping rule where they have an
  empirical frontier and an open problem.

Additionally, the abstention contribution has an empirical anchor that does not exist
elsewhere: on the leaderboards we audited, **417 of 444 adjacent orderings (93.9%) are not
resolvable at the seed budget actually used** (93.4% of 468 including the off-trajectory
DataDecide 750M scale, which is excluded elsewhere in this project), and 28 are not
identifiable at any seed budget. A procedure that always returns a ranking is returning
answers it cannot support; abstention is the formally correct output for those instances.

## 4. Risk register

- **Not scooped, but adjacent and moving.** SL2 was revised 2026-08-12 and could add a
  decision objective in a further revision. The contrast should be stated as
  design-for-estimation vs design-for-decision and defended empirically, not merely
  asserted.
- **Our linearity is a modelling assumption we have already refuted in part.** The
  two-parameter log-space form is linear in `x(C) = (1, −log C)`, which is what makes the
  transductive theory apply — but our measurements show residual scatter at 2.90× seed SE,
  compute-structured misspecification, and an unidentified floor on accuracy metrics. The
  honest framing is that the linear theory supplies the design and the guarantee *skeleton*,
  and Task 4 decides whether it survives contact with the measured error structure. If it
  does not, that is a result about the applicability of BAI theory to scaling laws.
- **Negative-result exposure.** Single-scale ranking has 1.33% mis-selection and zero fit
  variance on a fixed ladder. Choosing the ladder is the only remaining lever; if the
  optimal design concentrates at the top rung, the method reduces to the baseline and the
  contribution becomes the theory plus the abstention rule, not a win on mis-selection.
  That outcome must be reportable.
