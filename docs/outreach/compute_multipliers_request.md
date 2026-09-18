# Draft: data request to the compute-multipliers author

**Status: draft for review. Not sent.**

Recipient: Jerry Han (Lunar Society) --- author of *Compute multipliers: recipe
and corpus effects on language-model pretraining* (2026).

Suggested subject: **Request: per-run metrics across the five budgets (compute-multipliers)**

---

Hi Jerry,

I'm auditing whether small-compute scaling-law extrapolation picks the same
training intervention a target-budget run would pick, and I'd like to test
whether that accuracy differs between recipe and corpus interventions. Your suite
is the only one I know of that crosses both under a single protocol with seed
replicates.

Would you be able to share the per-run metrics across all five budgets (1e17 to
1e19) with seed identifiers? Concretely, one row per run with recipe, corpus,
budget, seed, and the evaluation metrics --- essentially the `runs.csv` the model
card refers to. **I don't need checkpoints or weights**, only the metrics table,
which I realise is the cheap part to share.

The model card points to `github.com/Lunar-Society/compute-multipliers` for the
run record, but that repository currently 404s, which is why I'm asking directly
rather than just cloning it.

One thing I found along the way that may be useful to you: I reconstructed the
compute-scaling curves from the Datawrapper tables behind the write-up's charts,
so I have OLMES against compute on both axes with the sd bands. Working from
those, the recipe axis looks hard to power at three seeds --- 0 of 21 endpoint
pairs clear a Bonferroni-corrected test, against 4 of 21 on the corpus axis,
because the recipes span about 5 seed standard deviations at 1e19 where the
corpora span about 21. If you have cells with more than three seeds, those are
the ones I'd most like to see.

Happy to acknowledge the contribution and cite the suite however you prefer, and
glad to share what I find on the recipe-versus-corpus comparison if it's useful
to you.

Thanks,
[name]

---

## Notes for the sender (not part of the email)

- **Ask is deliberately narrow.** Metrics table only. Checkpoints are 75 GB and
  are not needed; saying so up front removes the main reason to decline.
- **The 404 is stated as fact, not complaint.** It is the reason for the email
  and may simply be an oversight or a repo not yet made public.
- **One-line purpose**, as requested: testing whether small-scale selection
  accuracy differs between recipe and corpus interventions.
- **What this unblocks if answered:** OQ1, OQ2 and OQ3 in `OPEN_QUESTIONS.md`,
  all three of which are blocked on this one artifact.
- **If the answer is no**, the scoped claim in `OPEN_QUESTIONS.md` stands as
  written and nothing needs to change; the limitation is already stated wherever
  the affected claims appear.
- Contact route: the HuggingFace model card
  (`j23h67/compute-multipliers-checkpoints`) and the linked write-up are the two
  public points of contact. The HF repo's community tab is the lowest-friction
  option and makes the request visible to others who may want the same data.
