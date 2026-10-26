# Stage 4 review: what was not changed, and why

The reviewer saw only `paper/main.pdf` and `results/`. Everything below was left
as it is, or only partly addressed. Valid points not listed here were fixed in
the text.

## New analyses (DataDecide and Marin analyses are frozen)

1. **Lever-arm confounds.** The reviewer asked for a partial Spearman controlling
   for top-rung compute and number of rungs, a stratified permutation null, a
   restriction to lever arms of at most 1.5 decades, and an analytic
   variance-only baseline.
   - Not run: these would be new analyses on frozen data, chosen after seeing the
     result.
   - Instead, the confound is stated in Section 4 and in Limitations.
   - The null's coverage of a constant positive excess is explained: a rank
     correlation is invariant to a shift.
2. **Replication on a second suite** (PolyPythias, OLMo, Fantastic Optimizers
   ladders). Not run: this needs new data and runs.
3. **Equivalence test (TOST) for "hard to beat".** Not run. Instead, the claim
   is reworded as "not significantly better", the power is reported, and the
   text says this is "absence of evidence, not evidence of equivalence".
4. **Trend test and intervals for false certification by metric and lever arm.**
   - Added: a descriptive per-metric table (Table 6), tabulated from the
     committed per-design records.
   - Not added: a test on it, because the split was not pre-registered.
5. **A Marin-specific critical value.** Not calibrated, because Marin is frozen.
   The transferred 2.99 comparison is labelled indicative.
6. **Figure 3 with at least 2000 bootstrap draws.**
   - Not rerun: the run took about 22 hours, which is over the 30-minute limit
     for uncheckpointed jobs.
   - The reviewer said the 4/300 and 16/300 counts are not reproducible. That is
     incorrect for this figure: recomputing at HEAD from `scripts/run_task_b.py`
     gives 4, 16 and 6 flips.
   - The non-reproducible counts in `target_scoring.json` come from a different
     detector (the first audit's FDR crossover detection).
   - The figure's design label was wrong and is fixed (fit up to 300M, target
     1B). The caption now gives the draw count.

## Framing kept by decision

7. **The name "lever-arm law".** Kept, because it is the outline's section name.
   Every statement of it is now scoped to C4.
8. **Moving the selector to the appendix.** Kept in Section 5, as the outline
   specifies ("evidence for the law's consequence, not a proposed method"). The
   failed criterion and the per-metric failure are stated there.
9. **The family of 38 claims, which includes duplicated rows.**
   - Not redefined: changing the family after seeing the verdicts would be a
     post hoc choice.
   - The Table 4 caption now explains the duplicates and the sign conventions.

## Minor, deliberately unchanged

10. **Figure 2 is scored by observed hard mis-selection, not expected error.**
    Not rescored. The text now says the specifications are not independent.
11. **Table 3, Marin single-scale coverage "100" (230 of 231).** This is integer
    rounding, the same convention as every coverage cell.
12. **The variance inflation 8.40 equals 2.90 squared.** That is how it is
    defined; the independent 8.43 is a different pooling.
13. **"55 designs" (lever arm) against "45 designs" (certification).** These are
    different design sets: 55 C4 ladder and target combinations, and 15 fit
    windows times 3 metrics. They are not reconciled in a single sentence;
    Table 6 gives the per-metric counts.
14. **Figure 5's legend wording and truncated submission names.** Marin is
    frozen, and the names are shortened for width.
15. **The contradiction between `expected_error.json` Q4 (m=8 within 2%) and
    `published_comparisons.json` (44% miss at m=8).** Not reconciled here. The
    paper cites the failure, which is the conservative reading. This is an open
    item.
16. **The Impact Statement.** It uses ICML's suggested text for work without
    specific societal concerns.
17. **Appendix K cites script paths.** These are kept for the artifact; the
    repository will be anonymised at submission.
18. **The moderated t test "fails" in Section 3, yet moderated variances are used
    in Section 2.** Unchanged. Appendix H states that the adopted variant uses
    raw degrees of freedom and was chosen by Brier score, not pre-registered.
