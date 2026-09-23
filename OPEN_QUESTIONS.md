# Open questions, with the experiment that settles each

Each entry names a claim we cannot currently make, why, and the specific
experiment that would resolve it. These are scoped limitations with a resolution
path, not caveats.

---

## OQ1. Is the recipe axis scale-stable?

**The gap.** Every decision-accuracy number in this project is measured on
DataDecide, which varies **data recipes**. Our own H2 predicts that data
interventions are the scale-stable class --- few crossovers, so a small-scale
ranking survives extrapolation. The headline result ("no estimator, objective or
allocation beats ranking at the top rung") is therefore measured in the regime
where our theory says there is little for extrapolation to correct. Single-scale
ranking winning there is what the theory predicts, and it is weaker evidence than
it looks.

The **recipe** class --- optimizer, schedule, architecture, initialisation ---
is the one H2 calls crossover-prone. We have never tested it with seeds, because
no public suite provides a seeded compute ladder on that axis.

**What we do have.** `data/fantastic_optimizers_*` varies optimizers across a
ladder, but with a **single run per cell and no seeds**, so the seed-noise band
cannot be estimated and mis-selection cannot be separated from run-to-run
variation. It is reported as a limitation in `FIRST_AUDIT.md` and remains one.

**The experiment that settles it.**

| requirement | value | why |
| --- | --- | --- |
| suite | compute-multipliers (Han, Lunar Society, 2026) | seven recipes x seven corpora under one frozen protocol; the 2x2 this needs |
| axes | recipe (7 candidates) and corpus (7), crossed | the class contrast under one protocol is the quantity, not either axis alone |
| budgets | >= 4 of {1e17, 3.16e17, 1e18, 3.16e18, 1e19} | >= 3 to fit plus >= 1 held out above the fit range |
| seeds | **20 per cell on the recipe axis**, 5 on the corpus axis | measured, not assumed: at the suite's 3 seeds, 0 of 21 recipe pairs clear a Bonferroni-corrected two-sided test at the endpoint. Half the pairs become resolvable at 20 seeds; the corpus axis reaches the same point at 5 |
| lever arm | up to 100x (1e17 -> 1e19) | comparable to the 4.7x-8.9x we reach on DataDecide, and wider |
| metrics | native NLL, OLMES10, held-out-7, Alt-8 | lets OQ2 be answered on the same runs |

**Status: partially answered.** The campaign ran exactly this (1,397 runs across
five budgets, three seeds per cell). The per-run record is not released --- the
public artifact is 50 runs at the top budget, and the cited code release holding
`runs.csv` returns 404 --- but the *compute-scaling curves* are public by another
route. The write-up embeds Datawrapper charts, and Datawrapper serves each
chart's underlying table, which yields OLMES against compute across all five
budgets on both axes with +/-1 sd bands
(`results/external/compute_multipliers_curves.json`).

What that buys, and what it does not:

- **Answered:** ranking reversals between any fitting budget and the 1e19
  endpoint, on both axes, up to a 100x lever arm, with seed dispersion recovered
  from the plotted bands. The write-up's reported NeoX-over-GPT-2 reversal is
  confirmed in the data as one of exactly two recipe-axis reversals between
  3.16e18 and 1e19.
- **Still blocked:** the per-run rows needed for the cell-wise seed bootstrap our
  primary estimand uses, so mis-selection rates from this source are not
  comparable to the DataDecide numbers.
- **Newly measured, and it changes the ask:** the recipe axis is *underpowered by
  construction* at seven candidates. Recipes span 5.4 seed standard deviations at
  the endpoint against the corpora's 20.7, so 15 of 21 recipe pairs sit inside
  the seed noise versus 4 of 21 corpus pairs. Restricting reversals to pairs
  resolvable at both ends leaves at most one recipe comparison per budget. Six
  years of recipe work simply produced less spread than six years of corpus work
  --- the write-up's own headline --- so **more seeds, not just the per-run
  record, are what this question needs.** Quantified: to resolve half the
  endpoint pairs under a Bonferroni-corrected test, the corpus axis needs 5 seeds
  per cell and the recipe axis needs 20. Returns flatten past 20 (13 of 21 pairs
  at 30 seeds), because some recipe pairs are too close to separate at any
  affordable budget --- which is itself an answer to "is the recipe axis
  crossover-prone": several of its candidates are not distinguishable at all.

**One email may still help**; a draft is in
`docs/outreach/compute_multipliers_request.md`. It should now ask for seed counts
as well as per-run metrics, since the power analysis above says three seeds is
not enough on the recipe axis regardless of what else is shared.

**What we would learn either way.** If the recipe axis shows higher crossover
frequency and higher mis-selection than the corpus axis under one protocol, H2's
untested half is confirmed and the conditional-selection rule has a regime to
select between. If it shows the same null, the elimination argument strengthens
considerably: extrapolation would then fail to beat top-rung ranking even in the
class our theory says should favour it most.

---

## OQ2. Does the metric disagreement grow or shrink with scale?

**The gap.** `results/external/SELECTION_STABILITY.md` measures metric-conditional
selection failure on the corpus axis --- cross-family rank correlations of
-0.107 to +0.214, with three different corpora winning under four metrics. That
is a **single-budget snapshot** at 1e19 FLOPs. Whether the disagreement widens,
narrows or holds as compute grows is unmeasured, and it determines whether
small-scale metric choice is a transient problem or a persistent one.

**The experiment that settles it.** The same released ladder as OQ1. Compute the
cross-family rank correlation at each of the five budgets and test for a trend.
No new training compute is required.

---

## OQ3. Does hyperparameter-transfer quality moderate extrapolation success?

**The gap.** The Delphi result (Held and Marin Community, 2026) extrapolates 300x past its fit to within
0.2%, and its stated enabler is a token-horizon correction to the learning rate
plus an optimizer that removes weight decay from the search
(`results/external/counterexample_delphi.json`). That is a claim about
hyperparameter transfer. Our tuning-confound study measures the same mechanism
from the other side: pairwise agreement falls when a single hyperparameter is
mis-tuned relative to coordinate-descent tuning.

If extrapolation succeeds when hyperparameters transfer correctly and fails when
they do not, transfer regime is a third moderator alongside intervention class
and metric --- and it would reconcile their success with our null.

**Why we cannot test it here.** DataDecide varies corpora under one fixed recipe,
so it contains no hyperparameter-transfer contrast at all. The suite that does
--- compute-multipliers, whose seven recipes differ in exactly the knobs that
govern transfer (optimizer, schedule, initialisation, with later vintages
embodying better transfer practice) --- releases only its top budget.

**The experiment that settles it.** The same released ladder, stratified by
recipe vintage: fit and extrapolate per recipe, and test whether projection error
at a held-out budget decreases with the recipe's transfer quality. A cheaper
partial version: re-run the tuning-confound design with a token-horizon-corrected
learning rate and check whether the 87%-to-76% agreement drop narrows.

---

## What these have in common

All three are blocked on the same artifact: a seeded compute ladder on the recipe
axis. That is one request away, and until it is answered our claims are scoped to
the data axis, to loss-like metrics, and to a fixed-recipe regime. Those scopes
are stated wherever the claims appear.
