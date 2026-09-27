# Pre-registration: the misspecification-pattern property, its sweep, measurability, correct-prob's long design, DataDecide against the paper

Written 2026-09-27, before any computation in this round. No paper prose changes.
Outputs: JSON, figures, generated macros.

## Framing, as given

A transferable regime rule cannot be validated with two to three independent
metrics; that is the candidate-count problem at the metric level. This round
characterises the mechanism of metric-specificity. **Anything rule-like here is
exploratory.**

## Definitions

- **Design.** For a metric M and a design (top fitted rung f, target t):
  - mu is the shrunk truth.
  - S is the recipe-specific residual, as in the earlier arms: the residual from
    per-recipe power-law fits over the full ladder, minus its rank-one common
    mode.
- **Recipe-specific misspecification loading at the target.** Delta_r = S(r, t).
  - For a lower-is-better metric, Delta_r > 0 means recipe r is worse at the
    target than its power law says.
- **Alignment (a).** Spearman correlation of Delta with the true target values
  mu(., t).
  - Positive means the misspecification stretches the true gaps. The parametric
    ordering that projection recovers then agrees with the truth, so projection
    should win.
  - Negative means it opposes the true gaps and reverses some of them, so
    projection should lose.
- **(b)** Spearman correlation of Delta with the top-fitted-rung ordering,
  mu(., f).
- **(c) Observable proxy.** Spearman correlation of the in-sample residual at f
  (the per-recipe power-law fit on rungs <= f, evaluated at f, minus its
  across-recipe mean) with mu(., f). It uses no target information.
- **Crossover rate between f and t.** The share of recipe pairs whose true order
  differs between f and t.

## Task 1: the pattern property (exploratory)

- (a), (b) and (c) are computed for each of the four metrics (C4, OLMES,
  correct-prob, S&N) at their short, primary and long designs: 12 own-world
  cells.
- For the 18 structure-swap configurations (Task 2 of the previous round), (a)
  is computed at the primary design from the hybrid's own recipe-specific part
  and truth.
- The sign of (a) is compared with which rule wins: the own-world sign and the
  swap sign from the committed results. Unresolved cells are not tests.

**P1 (exploratory).**
- Across own-world cells, (a) < 0 goes with projection losing and (a) > 0 with
  it winning, in at least 80% of resolved cells.
- The same holds across the resolved swap configurations.

## Task 2: controlled sweep of alignment (C4, physical region)

- **Truth.** mu = mu_P + common + b m^T + E. Recipe-specific amplitude is held at
  measured: sd(b) = sd(a - mean(a)), and the remainder E is unchanged.
- **Loadings.** b is chosen per design so that Delta = b m(t) + E(t) has the
  requested Pearson correlation c with y0 = mu_P(., t) + common(., t), the
  target values before the recipe-specific part:
  - b = sd(a) x sign(m(t)) x (c z_y + sqrt(1 - c^2) z_perp), a direction chosen
    so that the correlation holds exactly, with E(t) included in Delta;
  - z_y is y0 standardised;
  - z_perp is the component of the measured loadings orthogonal to z_y,
    standardised.
- **Achieved correlation.** The Spearman correlation of Delta with the resulting
  true target values is also reported.
- **Grid.** c in {-1, -0.75, ..., 1} (9 points) x 3 designs (short 14M->16M,
  primary 300M->530M, long 8M->1B).
  - 100 fixed-candidate worlds per point at measured noise, with common random
    numbers across c.
  - Each sweep truth is checked for physicality (tolerant monotonicity, one
    seed sd). Unphysical points are reported and excluded from the verdict.
- **Readout.** Projection's true excess over single-scale, with a 95% bootstrap
  interval.

**P2 (as given).**
- Loadings opposing the true gaps make projection lose, and aligned loadings
  make it win.
- The excess falls monotonically in c at each design (Spearman over physical
  grid points <= -0.9).
- It changes sign at a predictable correlation. My operationalisation: c*
  (linear interpolation) lies in [-0.25, 0.25], the neutral point.
- **Refutation.** No monotone relationship. In that case the driving property is
  something else about the pattern, and what the swap data suggests is reported
  instead.

## Task 3: measurable in advance?

- **Per metric and design:** (a), (b) and (c); whether (b) and (a) agree in sign
  and differ by less than 0.2; and the crossover rate.
- **P3-1.** (b) agrees with (a) in sign wherever the crossover rate is below 10%.
- **P3-2.** The observable proxy (c) agrees in sign with (a) in at most 8 of the
  12 cells. In that case the property requires knowing the target (through the
  full-ladder fit) and cannot guide selection; that is reported.

## Task 4: correct-prob's long-design win

- **Readouts.**
  - In correct-prob's own worlds at the long design (8M -> 1B):
    single-scale's mis-selection rate, projection's rate, and the chance rate
    (50% of scored pairs).
  - At 8M: the noiseless single-scale rate (truth at 8M against truth at 1B),
    and the ratio of between-recipe spread to cell-mean seed noise.
  - Correct-prob's truth monotonicity, strict and tolerant, by rung, as was done
    for OLMES.
- **P4.** Single-scale's rate at the long design is at least 40%, near chance.
  The -13.3 is then reported as single-scale ranking at an uninformative rung,
  not as extrapolation succeeding.

## Task 5: DataDecide against the paper

- **Comparison.** The macro-average single-scale and scaling-law decision
  accuracies printed in arXiv:2504.11393v2 (fetched; the table or figure is
  cited) are compared with:
  - (a) their code's mixed-target output: scaling law on the default seed,
    single-scale on the three-seed mean;
  - (b) the consistent three-seed output;
  - also (c) the consistent default-seed output.
- **Released outputs.** If the Drive folder linked from `results/readme.md` is
  publicly shared, its listing is fetched and the single-scale output checked.
  If it is blocked, it is reported as unchecked.
- **S&N agreement figure.** The "93 to 99.7%" pair-agreement figure is verified
  from the committed `results/signal_and_noise/agreement.json`.
- **P5.** The printed numbers match (a), the mixed-target comparison.
