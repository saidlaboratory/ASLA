# Theory notes: when can small-compute data select the right intervention?

These are paper-ready sketches supporting ASLA's methods. They formalize two
things: (1) a lower bound showing that fit-range data alone *cannot* order
some curve pairs at the target — the formal version of the audit's
"fit is blind" message and the justification for the ensemble reliability
score; (2) the correctness argument for the racing policy's eliminations.

## Setup

- Interventions `k = 1..K` with true mean-loss curves `f_k(C)` (BPB at compute
  `C`), observed at ladder budgets `C_1 < ... < C_R = C_max` with `n` seeds per
  cell and independent sub-Gaussian seed noise with scale `σ`.
- A *selection rule* sees only fit-range data (budgets `≤ C_max`) and picks
  `k̂`; its regret is `f_k̂(C*) − min_k f_k(C*)` at target `C* > C_max`.

## 1. A detectability lower bound (the "fit is blind" lemma)

**Lemma.** Fix any curve dictionary `F` containing two families that admit
pairs `(f, g)` with (i) `|f(C) − g(C)| ≤ ε` for all `C ≤ C_max` and (ii)
`g(C*) − f(C*) ≥ Δ` at the target. Consider the two-intervention instances
`H_1 = (f, h)` and `H_2 = (g, h)` where `h` is any reference curve whose
target value lies between `f(C*)` and `g(C*)` with margin `Δ/2`. Then any
selection rule's worst-case wrong-pick probability over `{H_1, H_2}` is at
least

```
1/2 · (1 − TV(P_1, P_2)) ≥ 1/2 · (1 − sqrt(N/2) · ε/σ)
```

where `N = nR` is the total number of fit-range runs of the ambiguous
intervention and `P_i` its data distribution under `H_i`.

*Proof sketch.* The two hypotheses demand opposite picks (under `H_1` the
ambiguous intervention beats `h` at target, under `H_2` it loses). Any rule's
max error over two hypotheses is bounded below by Le Cam's two-point method:
`inf_k̂ max_i P_i(error) ≥ (1 − TV(P_1, P_2))/2`. With Gaussian noise,
`TV² ≤ KL/2` (Pinsker) and `KL(P_1‖P_2) = Σ_cells n·(f(C_c) − g(C_c))²/(2σ²) ≤
N ε²/(2σ²)`. ∎

**Consequences.**

- For power-law vs. saturating families the premise holds with `ε → 0` at any
  fixed `Δ`: a saturating curve can shadow a power law arbitrarily closely on
  `[C_1, C_max]` while flattening after it (choose `c_half` beyond `C_max`;
  the divergence grows with `C*/C_max`). Hence **no** rule fitting only
  `C ≤ C_max` data — however clever its functional form — can reliably order
  such pairs at the target: to get wrong-pick probability below `1/2 − γ` you
  need `N ≥ 8γ²σ²/ε²` runs, which diverges as `ε → 0`.
- This is exactly what the **ensemble reliability score** `ρ` estimates from
  data: `ρ` is large when families that fit the ladder equally well (LLBO
  losses within noise) disagree at the target by more than the seed-noise
  band — i.e. when the instance is inside the lemma's ambiguous set. The
  audit therefore reports `ρ` as a *decision hazard*, and the only fix is
  more ladder (raise `C_max`), not more seeds (which shrink `σ/√n` but not
  the family gap `ε` — the bound is per-run noise against curve ambiguity).

## 2. Racing correctness and cost

At rung `r ≥ 3`, the race refits every survivor on its accumulated rungs and
forms a target-projection interval `[L_k(r), U_k(r)]`; it eliminates `k` when
`L_k(r) > U_leader(r)` (with `β` scaling half-widths around the point
estimate; `β = 1` uses raw intervals).

**Claim (best-survivor).** If for every surviving `k` and every rung `r` the
intervals cover the truth, `P(f_k(C*) ∈ [L_k(r), U_k(r)]) ≥ 1 − δ`, then the
true best intervention `k*` survives to the final rung with probability at
least `1 − 2KRδ`.

*Proof sketch.* `k*` is eliminated at rung `r` only if `L_{k*}(r) >
U_leader(r) ≥ f_leader(C*) ≥ f_{k*}(C*)` — the second inequality needs the
leader's coverage, the first `k*`'s. Each is a coverage failure with
probability `≤ δ`; union bound over `K` arms and `R` rungs. ∎

**Cost.** The race pays `Σ_k Σ_{r ≤ r_elim(k)} n·C_r` versus the full ladder's
`Σ_k Σ_r n·C_r`. Since rung costs grow geometrically, eliminating an arm at
rung `r` saves a constant fraction of its total ladder cost; arms whose curves
separate early are exactly the ones the interval test removes first. The
audit does not assume this saving — it *measures* `compute_spent` per rule and
reports the (cost, regret) pair, and the Pareto study sweeps `τ` (gate) and
`β` (race) to trace each rule's frontier.

**Caveats stated in the paper.**

- Bootstrap percentile intervals on 3–5 rungs are not exact; coverage is
  studied empirically in the calibration section (conformal intervals are the
  distribution-free alternative, at the cost of wider bands on short
  ladders).
- Under family misspecification the projection intervals of *every* family
  can exclude the truth (Section 1); the race inherits that hazard, which is
  why `ρ` gates whether extrapolation-based elimination is trusted at all.
