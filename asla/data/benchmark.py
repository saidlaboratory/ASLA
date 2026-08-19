"""ASLA-Bench: parameterized selection problems with controllable difficulty.

Every benchmark instance is a *scenario function* compatible with the Monte
Carlo harness: it maps an RNG to a runs table whose noiseless truth is known
(`ScenarioTruth` attrs). Difficulty is controlled by interpretable knobs
rather than opaque random draws:

- ``gap_to_noise``: target-budget BPB gap between the top two interventions,
  in units of the seed-noise standard deviation. Small values make every rule
  guess; large values should be easy.
- ``crossover_position``: ``log10(C_x / C_max_fit)`` of the top-2 crossing.
  Negative → the crossing is visible inside the fitting ladder; positive →
  the leaderboard flips *after* the largest fitting budget (the deceptive
  regime the audit exists for); ``None`` → curves never cross (close-call
  family).
- ``saturation_strength``: for the saturating family, the half-saturation
  budget ``c_half`` as a multiple of the ladder's geometric-mean budget.
  Values ``<= 1`` put most of the bend at or before the ladder, so in-range
  the curve masquerades as a shallow power law and one-shot projection is
  reliably fooled; larger values move the visible bend into the upper ladder,
  where a fit can catch it.

The curve parameters are sampled deterministically from ``config.seed``; the
returned scenario function consumes its RNG only for seed noise, so a config
is one reproducible *problem* and trials are noise resamples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence, cast

import numpy as np
import pandas as pd

from asla.config import AuditConfig, BudgetLadder
from asla.data.schema import validate
from asla.data.synthetic import ScenarioTruth, TruthFn
from asla.models import bpb_power_law, bpb_saturating

FAMILIES = ("close_call", "late_crossover", "saturating")


@dataclass(frozen=True)
class BenchmarkConfig:
    """One benchmark problem: a curve family plus difficulty knobs."""

    family: str = "close_call"
    gap_to_noise: float = 2.0
    crossover_position: float | None = None
    saturation_strength: float = 1.0
    n_interventions: int = 4
    noise: float = 0.01
    n_seeds: int = 4
    budgets: BudgetLadder = BudgetLadder()
    seed: int = 0

    def name(self) -> str:
        cp = "none" if self.crossover_position is None else f"{self.crossover_position:+.2f}"
        return (
            f"{self.family}__gap{self.gap_to_noise:g}__cp{cp}"
            f"__sat{self.saturation_strength:g}__k{self.n_interventions}__seed{self.seed}"
        )


def _winner_curve(rng: np.random.Generator) -> tuple[float, float, float]:
    """Sample the true winner's power-law parameters."""

    E = float(rng.uniform(0.80, 0.95))
    A = float(rng.uniform(0.15, 0.40))
    alpha = float(rng.uniform(0.25, 0.60))
    return E, A, alpha


def _rival_power_law(
    winner: tuple[float, float, float],
    delta: float,
    c_cross: float | None,
    target: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Construct the runner-up so it loses by ``delta`` at target.

    With ``c_cross`` the rival crosses the winner there (better before, worse
    after); without, it trails by ``delta`` at every budget.
    """

    E1, A1, a1 = winner
    if c_cross is None:
        return (E1 + delta, A1, a1)
    f1_cross = float(bpb_power_law(c_cross, E1, A1, a1))
    f1_target = float(bpb_power_law(target, E1, A1, a1))
    if f1_cross - f1_target <= delta:
        raise ValueError(
            "winner curve is too flat between the crossover and the target for the requested gap; "
            "lower gap_to_noise or move the crossover earlier"
        )
    for _ in range(64):
        a2 = a1 * float(rng.uniform(1.3, 2.0))
        denom = c_cross ** (-a2) - target ** (-a2)
        A2 = (f1_cross - f1_target - delta) / denom
        E2 = f1_cross - A2 * c_cross ** (-a2)
        if A2 > 0 and E2 > 0 and a2 <= 2.0:
            return (E2, A2, a2)
    raise ValueError("could not construct a valid crossing rival; adjust the knobs")


def _rival_saturating(
    winner: tuple[float, float, float],
    delta: float,
    c_cross: float,
    target: float,
    c_half: float,
) -> tuple[float, float, float]:
    """Construct a saturating rival crossing the winner at ``c_cross``."""

    E1, A1, a1 = winner
    f1_cross = float(bpb_power_law(c_cross, E1, A1, a1))
    f1_target = float(bpb_power_law(target, E1, A1, a1))
    if f1_cross - f1_target <= delta:
        raise ValueError(
            "winner curve is too flat between the crossover and the target for the requested gap; "
            "lower gap_to_noise or move the crossover earlier"
        )
    g_cross = 1.0 / (1.0 + c_cross / c_half)
    g_target = 1.0 / (1.0 + target / c_half)
    drop = (f1_cross - f1_target - delta) / (g_cross - g_target)
    floor = f1_cross - drop * g_cross
    if drop <= 0 or floor <= 0:
        raise ValueError("could not construct a valid saturating rival; adjust the knobs")
    return (floor, drop, c_half)


def _filler_curves(
    winner: tuple[float, float, float],
    delta: float,
    target: float,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, float]]:
    """Sample clearly-worse power laws that never contend for the top."""

    E1, A1, a1 = winner
    f1_target = float(bpb_power_law(target, E1, A1, a1))
    curves: list[tuple[float, float, float]] = []
    for _ in range(count):
        margin = delta * float(rng.uniform(3.0, 8.0)) + 0.02
        A = float(rng.uniform(0.10, 0.40))
        a = float(rng.uniform(0.25, 0.60))
        E = f1_target + margin - A * target ** (-a)
        if E <= 0:
            E = f1_target + margin
        curves.append((E, A, a))
    return curves


def make_scenario(config: BenchmarkConfig) -> Callable[[np.random.Generator, AuditConfig | None], pd.DataFrame]:
    """Build a deterministic scenario function from a benchmark config.

    Raises ``ValueError`` when the knobs are geometrically infeasible (e.g.
    a requested gap larger than the winner's decay between crossover and
    target).
    """

    if config.family not in FAMILIES:
        raise ValueError(f"unknown family {config.family!r}; expected one of {FAMILIES}")
    if config.n_interventions < 2:
        raise ValueError("n_interventions must be at least 2")
    if config.gap_to_noise <= 0 or config.noise <= 0:
        raise ValueError("gap_to_noise and noise must be positive")
    ladder = config.budgets
    target = float(ladder.target)
    c_max_fit = float(max(ladder.fit))
    delta = float(config.gap_to_noise * config.noise)
    param_rng = np.random.default_rng(config.seed)
    winner = _winner_curve(param_rng)

    if config.family == "close_call":
        rival_params = _rival_power_law(winner, delta, None, target, param_rng)
        rival = cast(TruthFn, lambda C, p=rival_params: bpb_power_law(C, *p))  # noqa: E731
    else:
        if config.crossover_position is None:
            raise ValueError(f"family {config.family!r} requires crossover_position")
        c_cross = float(c_max_fit * 10.0**config.crossover_position)
        if c_cross >= target:
            raise ValueError("crossover_position places the crossing at or beyond the target")
        if config.family == "late_crossover":
            rival_params = _rival_power_law(winner, delta, c_cross, target, param_rng)
            rival = cast(TruthFn, lambda C, p=rival_params: bpb_power_law(C, *p))  # noqa: E731
        else:
            c_geo = float(np.sqrt(float(min(ladder.fit)) * c_max_fit))
            c_half = c_geo * float(config.saturation_strength)
            rival_params = _rival_saturating(winner, delta, c_cross, target, c_half)
            rival = cast(TruthFn, lambda C, p=rival_params: bpb_saturating(C, *p))  # noqa: E731

    functions: Dict[str, TruthFn] = {
        "winner": cast(TruthFn, lambda C, p=winner: bpb_power_law(C, *p)),
        "rival": rival,
    }
    classes = {"winner": "candidate", "rival": "candidate"}
    for i, params in enumerate(_filler_curves(winner, delta, target, config.n_interventions - 2, param_rng)):
        name = f"filler_{i}"
        functions[name] = cast(TruthFn, lambda C, p=params: bpb_power_law(C, *p))
        classes[name] = "control"

    def scenario(rng: np.random.Generator, cfg: AuditConfig | None = None) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for intervention, fn in functions.items():
            for compute in ladder.all:
                for seed in range(config.n_seeds):
                    rows.append(
                        {
                            "intervention": intervention,
                            "intervention_class": classes[intervention],
                            "compute": float(compute),
                            "seed": int(seed),
                            "bpb": float(fn(compute) + rng.normal(0.0, config.noise)),
                        }
                    )
        df = pd.DataFrame(rows)
        validate(df)
        df.attrs["scenario_truth"] = ScenarioTruth(
            name=config.name(),
            functions=dict(functions),
            target=target,
            planted_pair=("winner", "rival"),
            saturating_pair=("winner", "rival") if config.family == "saturating" else None,
        )
        return df

    scenario.__name__ = config.name()
    return scenario


def benchmark_grid(
    families: Sequence[str] = FAMILIES,
    gaps: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    positions: Sequence[float] = (-0.30, 0.15, 0.30, 0.45),
    saturation_strengths: Sequence[float] = (1.0,),
    n_interventions: int = 4,
    noise: float = 0.01,
    n_seeds: int = 4,
    seeds: Sequence[int] = (0,),
) -> list[BenchmarkConfig]:
    """Enumerate a difficulty grid; infeasible knob combinations are skipped."""

    configs: list[BenchmarkConfig] = []
    for family in families:
        for gap in gaps:
            for seed in seeds:
                if family == "close_call":
                    candidates = [
                        BenchmarkConfig(
                            family=family,
                            gap_to_noise=gap,
                            crossover_position=None,
                            n_interventions=n_interventions,
                            noise=noise,
                            n_seeds=n_seeds,
                            seed=seed,
                        )
                    ]
                else:
                    strengths = saturation_strengths if family == "saturating" else (1.0,)
                    candidates = [
                        BenchmarkConfig(
                            family=family,
                            gap_to_noise=gap,
                            crossover_position=pos,
                            saturation_strength=strength,
                            n_interventions=n_interventions,
                            noise=noise,
                            n_seeds=n_seeds,
                            seed=seed,
                        )
                        for pos in positions
                        for strength in strengths
                    ]
                for config in candidates:
                    try:
                        make_scenario(config)
                    except ValueError:
                        continue
                    configs.append(config)
    return configs


def evaluate_configs(
    configs: Iterable[BenchmarkConfig],
    n_trials: int,
    n_boot: int,
    seed: int = 1729,
    beta: float = 1.0,
    tau: float | None = None,
) -> pd.DataFrame:
    """Run the selection-rule Monte Carlo on every config; one tidy row per rule."""

    from dataclasses import replace

    from asla.analysis.racing import monte_carlo_selection

    records: list[dict[str, object]] = []
    for config in configs:
        scenario = make_scenario(config)
        cfg = AuditConfig(budgets=config.budgets)
        cfg = replace(cfg, gate=replace(cfg.gate, n_boot=n_boot))
        results = monte_carlo_selection(
            scenario, cfg, n_trials=n_trials, rng=np.random.default_rng(seed), beta=beta, tau=tau
        )
        for rule, stats in results.items():
            records.append(
                {
                    "config": config.name(),
                    "family": config.family,
                    "gap_to_noise": float(config.gap_to_noise),
                    "crossover_position": (
                        float("nan") if config.crossover_position is None else float(config.crossover_position)
                    ),
                    "saturation_strength": float(config.saturation_strength),
                    "param_seed": int(config.seed),
                    "rule": rule,
                    **{k: float(v) for k, v in stats.items()},
                }
            )
    return pd.DataFrame.from_records(records)
