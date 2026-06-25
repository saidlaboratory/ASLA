"""Configuration dataclasses for the algorithm-selection audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class BudgetLadder:
    """Compute budgets used by synthetic demos and audits."""

    fit: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    intermediate: float = 16.0
    target: float = 64.0

    @property
    def all(self) -> Tuple[float, ...]:
        """Return fit, intermediate, and target budgets in ascending order."""

        return tuple(sorted(set((*self.fit, self.intermediate, self.target))))


@dataclass(frozen=True)
class SeedConfig:
    """Randomness configuration."""

    seed: int = 1729
    n_seeds: int = 8


@dataclass(frozen=True)
class GateConfig:
    """Uncertainty-gate configuration."""

    tau: float = 1.0
    n_boot: int = 120
    noise_band_k: float = 2.0


@dataclass(frozen=True)
class Paths:
    """Default output paths."""

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    split_path: Path = Path(".asla_split.json")


@dataclass(frozen=True)
class AuditConfig:
    """Top-level audit configuration."""

    budgets: BudgetLadder = BudgetLadder()
    seeds: SeedConfig = SeedConfig()
    gate: GateConfig = GateConfig()
    paths: Paths = Paths()
    synthetic_noise: float = 0.003

