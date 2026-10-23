"""Tests for the review-risk scripts' pure helpers."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

marin = importlib.import_module("scripts.run_marin_speedrun_gate")
cmr = importlib.import_module("scripts.run_compute_multipliers_replication")
mis = importlib.import_module("scripts.run_calibration_misspecified")
mech = importlib.import_module("scripts.run_mechanism_robustness")
lam = importlib.import_module("scripts.run_lever_arm_by_metric")

# The OLMES context reads the raw DataDecide evaluation table, which is harvested locally (data/raw is not in git).
NEEDS_RAW_EVAL = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "raw" / "datadecide" / "eval_macro_avg.parquet").exists(),
    reason="raw DataDecide eval_macro_avg.parquet not harvested",
)


def test_marin_ladders_and_shared_budgets() -> None:
    runs = [{"run_name": f"{s}/{b}", "eval_paloma_c4_en_bpb": 1.0} for s in "abc" for b in ("1m", "2m", "3m", "4m")]
    runs += [{"run_name": "d/1m", "eval_paloma_c4_en_bpb": 1.0}, {"run_name": "solo", "eval_paloma_c4_en_bpb": 1.0}]
    ladder = marin.ladders(runs)
    assert ladder["a"] == {"1m", "2m", "3m", "4m"} and "solo" not in ladder
    n, budgets, members = marin.best_shared(ladder, 4)
    assert n == 3 and members == ["a", "b", "c"] and len(budgets) == 4


def test_compute_multipliers_designs() -> None:
    designs = cmr.designs([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ([1.0, 2.0, 3.0], 4.0) in designs and ([2.0, 3.0, 4.0], 5.0) in designs
    assert all(len(fit) >= 3 and target > fit[-1] for fit, target in designs)
    assert len(designs) == 4


def test_mis_selection_rate() -> None:
    import pandas as pd

    truth = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    assert cmr.mis(truth, truth) == 0.0
    assert np.isclose(cmr.mis(pd.Series({"a": 2.0, "b": 1.0, "c": 3.0}), truth), 1 / 3)


def test_student_t_generator_has_unit_variance() -> None:
    gen = mis.StudentT(np.random.default_rng(0), 5)
    draws = gen.standard_normal(200_000)
    assert abs(draws.var() - 1) < 0.05
    assert gen.integers(0, 10) in range(10)  # other Generator methods pass through


def test_coverage_counts_and_interval() -> None:
    rows = [{"true_difference": 0.0, "estimate": e, "se": 1.0} for e in (0.1, -0.2, 3.5, 0.0)]
    out = mis.coverage(rows, 2.0, "se")
    assert out["coverage"] == 0.75 and out["n_worlds"] == 4
    assert out["ci"][0] < 0.75 < out["ci"][1]


def test_attribution_components_sum_to_one() -> None:
    a = mech.attribution({"A": 3.0, "B1": 2.5, "B2": 5.2, "B3": 5.0})
    assert np.isclose(sum(a.values()), 1.0)
    assert np.isclose(a["variance"], 0.6)


def test_lever_arm_subsets_drop_chance_rungs() -> None:
    class D:
        def __init__(self, label: str) -> None:
            self.max_fit_label = label

    designs = [D("8M"), D("60M"), D("10M"), D("300M")]
    assert lam.subset_index(designs, ["8M", "10M"]) == [1, 3]


@NEEDS_RAW_EVAL
def test_truth_perturbation_reaches_the_simulated_observations() -> None:
    from asla.analysis.calibration import simulate

    base = mis.perturbed_base("control")
    moved = mis.perturbed_base("misspec_0_5")
    a, _ = simulate(base, np.random.default_rng(1))
    b, _ = simulate(moved, np.random.default_rng(1))
    shift_obs = (b["bpb"] - a["bpb"]).to_numpy()
    shift_truth = (moved.final["mu"] - base.final["mu"]).to_numpy()
    assert np.allclose(shift_obs, shift_truth, atol=1e-9)
    assert np.abs(shift_truth).max() > 0


def test_marin_fit_helpers() -> None:
    rig = importlib.import_module("scripts.run_marin_rigour")
    pts = [(1e18, 2.0 * 1e18**-0.1), (1e19, 2.0 * 1e19**-0.1), (1e20, 2.0 * 1e20**-0.1)]
    assert np.isclose(rig.loglog(pts, 1e21), 2.0 * 1e21**-0.1)
    series = {"a": pts, "b": [(x, 1.1 * y) for x, y in pts]}
    out = rig.shared_exponent(series, {"a": 1e21, "b": 1e21})
    assert np.isclose(out["a"], 2.0 * 1e21**-0.1) and np.isclose(out["b"], 2.2 * 1e21**-0.1)
    truth = rig.gaps({"a": 1.0, "b": 2.0, "c": 3.0})
    assert rig.wrong(rig.gaps({"a": 2.0, "b": 1.0, "c": 3.0}), truth) == 1
    assert rig.converged_sse([(1e18, 1.3), (3e18, 1.2), (1e19, 1.12), (3e19, 1.07)]) < 1e-3


def test_noise_profile_summary() -> None:
    import pandas as pd

    npf = importlib.import_module("scripts.run_noise_profiles")
    out = npf.summarise(pd.Series({"a": 1.0, "b": np.e}), pd.Series({"a": 1.0, "b": 10.0}))
    assert np.isclose(out["log_range"], 1.0) and np.isclose(out["log_sd_slope_per_decade"], 1.0)


def test_reorder_rate_counts_only_determined_pairs() -> None:
    ref = importlib.import_module("scripts.run_marin_reference")
    target = {"a": 1.0, "b": 2.0, "c": 3.0}
    fit = {"a": 2.5, "b": 2.0, "c": 3.0}  # a misordered against b and c at the fit rung
    se = {("a", "b"): 0.1, ("a", "c"): 0.1, ("b", "c"): 1.0}  # b-c undetermined (gap 1 < 1.96)
    out = ref.reorder(fit, target, se)
    assert out["determined"] == 2 and out["reordered"] == 1 and out["rate"] == 0.5
    assert ref.reorder(fit, target, se, m=100.0)["determined"] == 0


def test_sign_changes_and_both_rung_reordering() -> None:
    mn = importlib.import_module("scripts.run_marin_noise")
    values = np.array([[1.0, 2.0, 1.0, 2.0], [1.5, 1.5, 1.5, 1.5], [3.0, 3.0, 3.0, 3.0]])
    changes = mn.sign_changes(values)  # pairs (0,1), (0,2), (1,2)
    assert changes.tolist() == [3, 0, 0]
    fit = {"a": 1.0, "b": 2.0}
    target = {"a": 2.0, "b": 1.0}
    se = {("a", "b"): 0.1}
    res = mn.both_rung(mn.determined(fit, se), mn.determined(target, se))
    assert res["genuine_reorderings"] == 1 and res["determined_at_both"] == 1
    loose = {("a", "b"): 10.0}
    assert mn.both_rung(mn.determined(fit, loose), mn.determined(target, se))["genuine_reorderings"] == 0
