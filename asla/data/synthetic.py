"""Deterministic synthetic scenarios for the audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping

import numpy as np
import pandas as pd

from asla.config import AuditConfig, BudgetLadder
from asla.data.schema import validate
from asla.models import bpb_power_law, bpb_saturating

TruthFn = Callable[[np.ndarray | float], np.ndarray | float]


@dataclass(frozen=True)
class ScenarioTruth:
    """Ground-truth metadata attached to synthetic data frames."""

    name: str
    functions: Mapping[str, TruthFn]
    target: float
    planted_pair: tuple[str, str] | None = None
    saturating_pair: tuple[str, str] | None = None

    def ranking(self, target: float | None = None) -> pd.Series:
        """Return the noiseless true ranking at ``target`` sorted by BPB."""

        t = self.target if target is None else target
        values = {name: float(fn(t)) for name, fn in self.functions.items()}
        return pd.Series(values).sort_values(kind="mergesort")


def _make_df(
    name: str,
    functions: Mapping[str, TruthFn],
    classes: Mapping[str, str],
    rng: np.random.Generator,
    noise: float,
    n_seeds: int,
    budgets: BudgetLadder,
    planted_pair: tuple[str, str] | None = None,
    saturating_pair: tuple[str, str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for intervention, fn in functions.items():
        for compute in budgets.all:
            for seed in range(n_seeds):
                rows.append(
                    {
                        "intervention": intervention,
                        "intervention_class": classes[intervention],
                        "compute": float(compute),
                        "seed": int(seed),
                        "bpb": float(fn(compute) + rng.normal(0.0, noise)),
                    }
                )
    df = pd.DataFrame(rows)
    validate(df)
    df.attrs["scenario_truth"] = ScenarioTruth(
        name=name,
        functions=dict(functions),
        target=budgets.target,
        planted_pair=planted_pair,
        saturating_pair=saturating_pair,
    )
    return df


def clean_crossover(
    rng: np.random.Generator,
    config: AuditConfig | None = None,
) -> pd.DataFrame:
    """Return a noisy power-law scenario with a planted projected decision flip."""

    cfg = config or AuditConfig()
    functions: Dict[str, TruthFn] = {
        "fast_start": lambda C: bpb_power_law(C, 0.92, 0.18, 0.75),
        "late_scaler": lambda C: bpb_power_law(C, 0.78, 0.35, 0.25),
        "steady_control": lambda C: bpb_power_law(C, 0.96, 0.20, 0.45),
        "weak_control": lambda C: bpb_power_law(C, 1.02, 0.15, 0.50),
    }
    classes = {
        "fast_start": "optimizer",
        "late_scaler": "optimizer",
        "steady_control": "control",
        "weak_control": "negative_control",
    }
    return _make_df(
        "clean_crossover",
        functions,
        classes,
        rng,
        noise=0.010,
        n_seeds=2,
        budgets=cfg.budgets,
        planted_pair=("fast_start", "late_scaler"),
    )


def saturation_crossover(
    rng: np.random.Generator,
    config: AuditConfig | None = None,
) -> pd.DataFrame:
    """Return a misspecified scenario where a saturating curve fools power-law projection."""

    cfg = config or AuditConfig()
    functions: Dict[str, TruthFn] = {
        "saturating_early": lambda C: bpb_saturating(C, 0.95, 0.10, 2.0),
        "late_power": lambda C: bpb_power_law(C, 0.916, 0.28, 0.50),
        "steady_control": lambda C: bpb_power_law(C, 0.96, 0.20, 0.45),
        "weak_control": lambda C: bpb_power_law(C, 1.02, 0.15, 0.50),
    }
    classes = {
        "saturating_early": "schedule",
        "late_power": "schedule",
        "steady_control": "control",
        "weak_control": "negative_control",
    }
    return _make_df(
        "saturation_crossover",
        functions,
        classes,
        rng,
        noise=0.002,
        n_seeds=cfg.seeds.n_seeds,
        budgets=cfg.budgets,
        planted_pair=("saturating_early", "late_power"),
        saturating_pair=("saturating_early", "late_power"),
    )


def noise_close_call(
    rng: np.random.Generator,
    config: AuditConfig | None = None,
) -> pd.DataFrame:
    """Return near-parallel power laws where noise can flip the projected winner."""

    cfg = config or AuditConfig()
    functions: Dict[str, TruthFn] = {
        "true_winner": lambda C: bpb_power_law(C, 0.880, 0.34, 0.42),
        "near_miss": lambda C: bpb_power_law(C, 0.887, 0.34, 0.42),
        "far_control": lambda C: bpb_power_law(C, 0.94, 0.25, 0.45),
        "weak_control": lambda C: bpb_power_law(C, 1.01, 0.15, 0.50),
    }
    classes = {
        "true_winner": "optimizer",
        "near_miss": "optimizer",
        "far_control": "control",
        "weak_control": "negative_control",
    }
    return _make_df(
        "noise_close_call",
        functions,
        classes,
        rng,
        noise=0.011,
        n_seeds=cfg.seeds.n_seeds,
        budgets=cfg.budgets,
        planted_pair=("true_winner", "near_miss"),
    )


def negative_controls_only(
    rng: np.random.Generator,
    config: AuditConfig | None = None,
) -> pd.DataFrame:
    """Return non-crossing interventions for false-positive measurement."""

    cfg = config or AuditConfig()
    functions: Dict[str, TruthFn] = {
        "control_a": lambda C: bpb_power_law(C, 0.88, 0.20, 0.45),
        "control_b": lambda C: bpb_power_law(C, 0.91, 0.20, 0.45),
        "control_c": lambda C: bpb_power_law(C, 0.95, 0.20, 0.45),
        "control_d": lambda C: bpb_power_law(C, 1.00, 0.20, 0.45),
    }
    classes = {name: "negative_control" for name in functions}
    return _make_df(
        "negative_controls_only",
        functions,
        classes,
        rng,
        noise=0.004,
        n_seeds=cfg.seeds.n_seeds,
        budgets=cfg.budgets,
    )


SCENARIOS: dict[str, Callable[[np.random.Generator, AuditConfig | None], pd.DataFrame]] = {
    "clean_crossover": clean_crossover,
    "saturation_crossover": saturation_crossover,
    "noise_close_call": noise_close_call,
    "negative_controls_only": negative_controls_only,
}


def true_ranking(df: pd.DataFrame, target: float | None = None) -> pd.Series:
    """Return the noiseless true ranking stored on a synthetic data frame."""

    truth = df.attrs.get("scenario_truth")
    if not isinstance(truth, ScenarioTruth):
        raise ValueError("data frame does not carry synthetic ScenarioTruth metadata")
    return truth.ranking(target)
