"""Cost-aware sequential selection: racing interventions up the budget ladder.

One-shot projection spends the full ladder on every intervention and then
decides. The racing policy instead advances interventions rung by rung and
eliminates an intervention as soon as its *optimistic* projected target BPB
(interval lower bound) is worse than the current leader's *pessimistic* one
(interval upper bound) — the classic racing rule, but applied to
scaling-law extrapolations rather than per-rung means, so evidence from every
rung an intervention has run accumulates into its target projection.

All rules here report ``compute_spent`` — the summed ``budget x runs`` cost of
the cells the rule actually consumed — so decision quality can be compared at
equal cost, not just at equal ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from asla.analysis.fits import normalize_budgets, projection_with_uncertainty
from asla.analysis.gate import evaluation_truth, gate_pick_detailed, largest_single_run_pick, plain_projection_pick
from asla.config import AuditConfig
from asla.data.schema import validate
from asla.models import FitError


@dataclass(frozen=True)
class RungRecord:
    """What happened at one rung of the race."""

    budget: float
    survivors: tuple[str, ...]
    eliminated: tuple[str, ...]
    intervals: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    """A selection rule's pick together with the compute it consumed."""

    pick: str
    compute_spent: float
    rungs: tuple[RungRecord, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "pick": self.pick,
            "compute_spent": float(self.compute_spent),
            "rungs": [
                {
                    "budget": float(r.budget),
                    "survivors": list(r.survivors),
                    "eliminated": list(r.eliminated),
                }
                for r in self.rungs
            ],
        }


def cell_cost(df: pd.DataFrame, intervention: str, budget: float) -> float:
    """Return ``budget x number_of_runs`` for one intervention-budget cell."""

    rows = df[
        (df["intervention"].astype(str) == str(intervention))
        & np.isclose(df["compute"].astype(float), float(budget))
    ]
    return float(budget) * int(len(rows))


def ladder_cost(df: pd.DataFrame, interventions: list[str], budgets: tuple[float, ...]) -> float:
    """Total compute for running every listed intervention at every listed budget."""

    return float(sum(cell_cost(df, iv, b) for iv in interventions for b in budgets))


def plain_projection_result(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> SelectionResult:
    """One-shot projection pick with its full-ladder cost."""

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    interventions = sorted(df["intervention"].astype(str).unique())
    pick = plain_projection_pick(df, fit_budgets, target)
    return SelectionResult(pick=pick, compute_spent=ladder_cost(df, interventions, fit_budgets))


def largest_single_result(df: pd.DataFrame, budgets: tuple[float, ...], target: float) -> SelectionResult:
    """Largest-fitting-budget pick; it only pays for the largest rung."""

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    interventions = sorted(df["intervention"].astype(str).unique())
    pick = largest_single_run_pick(df, fit_budgets, target)
    largest = (float(max(fit_budgets)),)
    return SelectionResult(pick=pick, compute_spent=ladder_cost(df, interventions, largest))


def gate_result(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    intermediate_budget: float,
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> SelectionResult:
    """Uncertainty-gate pick with cost; escalation pays the top-2 intermediate runs."""

    validate(df)
    fit_budgets = normalize_budgets(budgets, target=target)
    interventions = sorted(df["intervention"].astype(str).unique())
    cost = ladder_cost(df, interventions, fit_budgets)
    decision = gate_pick_detailed(df, fit_budgets, target, intermediate_budget, tau, n_boot, rng)
    if decision["escalated"]:
        top1, top2 = decision["top_pair"]  # type: ignore[misc]
        cost += cell_cost(df, top1, intermediate_budget) + cell_cost(df, top2, intermediate_budget)
    return SelectionResult(pick=str(decision["pick"]), compute_spent=cost)


def race_pick(
    df: pd.DataFrame,
    budgets: tuple[float, ...],
    target: float,
    n_boot: int,
    rng: np.random.Generator,
    beta: float = 1.0,
    start_rungs: int = 3,
) -> SelectionResult:
    """Race interventions up the ladder, eliminating by projected-interval dominance.

    All interventions run the first ``start_rungs`` rungs (a power-law fit
    needs three distinct budgets). From then on, after each rung every
    survivor is refit on all rungs it has run and projected to the target
    with bootstrap uncertainty; an intervention is eliminated when its
    interval lower bound exceeds the leader's interval upper bound. ``beta``
    scales interval half-widths around the point projection: ``beta < 1``
    eliminates more aggressively (cheaper, riskier), ``beta > 1`` is more
    conservative. Survivors whose bootstrap fails at a rung are carried, not
    eliminated — lack of evidence is not evidence of being worse.

    Compute is charged for every cell a survivor advances into; the pick is
    the surviving intervention with the lowest final projection.
    """

    validate(df)
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    ladder = normalize_budgets(budgets, target=target)
    if len(ladder) < max(3, start_rungs):
        raise ValueError(f"racing requires at least {max(3, start_rungs)} rungs, got {len(ladder)}")
    survivors = sorted(df["intervention"].astype(str).unique())
    if len(survivors) < 2:
        raise ValueError("racing requires at least two interventions")

    spent = 0.0
    records: list[RungRecord] = []
    last_points: dict[str, float] = {}
    for i, rung in enumerate(ladder):
        spent += sum(cell_cost(df, iv, rung) for iv in survivors)
        if i + 1 < start_rungs or len(survivors) == 1:
            records.append(RungRecord(budget=float(rung), survivors=tuple(survivors), eliminated=()))
            continue
        run_budgets = ladder[: i + 1]
        intervals: dict[str, tuple[float, float, float]] = {}
        for iv in survivors:
            try:
                point, _, lo, hi = projection_with_uncertainty(
                    df[df["intervention"] == iv], run_budgets, target, n_boot, rng
                )
            except FitError:
                continue
            lo_b = point - beta * (point - lo)
            hi_b = point + beta * (hi - point)
            intervals[iv] = (float(point), float(lo_b), float(hi_b))
            last_points[iv] = float(point)
        if len(intervals) < 2:
            records.append(RungRecord(budget=float(rung), survivors=tuple(survivors), eliminated=()))
            continue
        leader = min(intervals, key=lambda name: (intervals[name][0], name))
        leader_hi = intervals[leader][2]
        eliminated = tuple(
            iv for iv in survivors if iv in intervals and iv != leader and intervals[iv][1] > leader_hi
        )
        survivors = [iv for iv in survivors if iv not in eliminated]
        records.append(
            RungRecord(budget=float(rung), survivors=tuple(survivors), eliminated=eliminated, intervals=intervals)
        )
        if len(survivors) == 1:
            break

    known = {iv: last_points[iv] for iv in survivors if iv in last_points}
    if known:
        pick = min(known, key=lambda name: (known[name], name))
    else:
        pick = survivors[0]
    return SelectionResult(pick=pick, compute_spent=float(spent), rungs=tuple(records))


def monte_carlo_selection(
    scenario_fn,
    config: AuditConfig | None = None,
    n_trials: int | None = None,
    rng: np.random.Generator | None = None,
    beta: float = 1.0,
    tau: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compare all selection rules on regret, wrong-pick rate, and compute spent.

    Every rule sees the same simulated tables. Regret is measured against the
    noiseless synthetic truth when the scenario provides it, otherwise against
    measured target means.
    """

    cfg = config or AuditConfig()
    trials = cfg.counts.n_trials if n_trials is None else n_trials
    if trials <= 0:
        raise ValueError("n_trials must be positive")
    tau_value = cfg.gate.tau if tau is None else float(tau)
    root_rng = rng or np.random.default_rng(cfg.seeds.seed)
    pre_target = tuple(sorted(set((*cfg.budgets.fit, cfg.budgets.intermediate))))
    names = ("plain", "largest", "gate", "race")
    regret: dict[str, list[float]] = {name: [] for name in names}
    wrong: dict[str, list[float]] = {name: [] for name in names}
    compute: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(trials):
        data_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
        gate_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
        race_rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
        df = scenario_fn(data_rng, cfg)
        truth = evaluation_truth(df, cfg.budgets.target)
        true_best = str(truth.index[0])
        results = {
            "plain": plain_projection_result(df, cfg.budgets.fit, cfg.budgets.target),
            "largest": largest_single_result(df, cfg.budgets.fit, cfg.budgets.target),
            "gate": gate_result(
                df,
                cfg.budgets.fit,
                cfg.budgets.target,
                cfg.budgets.intermediate,
                tau_value,
                cfg.gate.n_boot,
                gate_rng,
            ),
            "race": race_pick(df, pre_target, cfg.budgets.target, cfg.gate.n_boot, race_rng, beta=beta),
        }
        for name, result in results.items():
            regret[name].append(float(truth.loc[result.pick] - truth.min()))
            wrong[name].append(float(result.pick != true_best))
            compute[name].append(float(result.compute_spent))

    return {
        name: {
            "mean_regret": float(np.mean(regret[name])),
            "wrong_pick_rate": float(np.mean(wrong[name])),
            "mean_compute": float(np.mean(compute[name])),
        }
        for name in names
    }
