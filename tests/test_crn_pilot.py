"""Correctness of the copied-initialisation CRN pilot at toy size, on CPU (MPS is not deterministic enough).

Required by the pilot's pre-registration before hand-off:

* rerunning an identical configuration reproduces the loss curve exactly;
* step-0 parameter hashes are identical across the four learning rates for a seed index;
* the batch sequence is identical across the four learning rates for a seed index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pilot.crn import data, launch, trainer  # noqa: E402

analysis = pytest.importorskip("scripts.analyze_crn_pilot")

TINY = trainer.ModelConfig(n_layer=1, n_head=2, d_model=16, context=16, vocab=256)
STEPS, BATCH = 6, 4


def _config(multiplier: float, seed_index: int) -> trainer.RunConfig:
    return trainer.RunConfig(
        lr_multiplier=multiplier,
        seed_index=seed_index,
        steps=STEPS,
        batch_size=BATCH,
        eval_batch_size=4,
        n_checkpoints=3,
        model=TINY,
    )


@pytest.fixture(scope="module")
def work(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("crn_pilot")
    rng = np.random.default_rng(0)
    text = data.stories_to_bytes(
        " ".join(rng.choice(["the", "cat", "sat", "on", "a", "mat", "and", "ran", "away"], 40)) for _ in range(60)
    )
    data.write_bytes(text, root / "train.bin")
    data.write_bytes(text[: 8 * (TINY.context + 1)], root / "eval.bin")
    for k in (0, 1):
        trainer.make_init(k, TINY, root / f"init_seed{k}.pt")
        trainer.make_order(k, root / "train.bin", TINY.context, STEPS * BATCH, root / f"order_seed{k}.npy")
    return root


@pytest.fixture(scope="module")
def runs(work: Path) -> dict[tuple[float, int], dict]:
    return {(m, k): trainer.train_run(_config(m, k), work, "cpu") for m in trainer.LR_MULTIPLIERS for k in (0, 1)}


def test_identical_configuration_reproduces_the_loss_curve_exactly(work: Path, runs: dict) -> None:
    again = trainer.train_run(_config(1.0, 0), work, "cpu")
    assert again["curve"] == runs[(1.0, 0)]["curve"]
    assert again["batch_sequence_sha256"] == runs[(1.0, 0)]["batch_sequence_sha256"]


def test_step0_parameters_are_identical_across_learning_rates(work: Path, runs: dict) -> None:
    for k in (0, 1):
        hashes = {runs[(m, k)]["init_param_sha256"] for m in trainer.LR_MULTIPLIERS}
        assert len(hashes) == 1
        losses = {runs[(m, k)]["curve"][0]["eval_loss"] for m in trainer.LR_MULTIPLIERS}
        assert len(losses) == 1  # same parameters, same evaluation at step 0
    assert runs[(1.0, 0)]["init_param_sha256"] != runs[(1.0, 1)]["init_param_sha256"]


def test_init_is_loaded_from_the_saved_state_not_reseeded(work: Path) -> None:
    model = trainer.GPT(TINY)
    state = torch.load(work / "init_seed0.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    torch.manual_seed(12345)
    reseeded = trainer.GPT(TINY)
    assert trainer.state_hash(model) != trainer.state_hash(reseeded)


def test_batch_sequence_is_identical_across_learning_rates(runs: dict) -> None:
    for k in (0, 1):
        assert len({runs[(m, k)]["batch_sequence_sha256"] for m in trainer.LR_MULTIPLIERS}) == 1
        assert len({runs[(m, k)]["order_file_sha256"] for m in trainer.LR_MULTIPLIERS}) == 1
    assert runs[(1.0, 0)]["batch_sequence_sha256"] != runs[(1.0, 1)]["batch_sequence_sha256"]


def test_learning_rates_actually_differ_after_training(runs: dict) -> None:
    finals = {runs[(m, 0)]["final_eval_loss"] for m in trainer.LR_MULTIPLIERS}
    assert len(finals) == len(trainer.LR_MULTIPLIERS)


def test_checkpoints_and_schedule() -> None:
    assert trainer.checkpoints(2500, 10) == [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]
    cfg = trainer.RunConfig(lr_multiplier=1.5, seed_index=0, steps=1000)
    assert np.isclose(trainer.lr_at(9, cfg), 1.5e-3)  # end of the 1% warmup
    assert np.isclose(trainer.lr_at(999, cfg), 1.5e-4, rtol=1e-3)  # cosine floor at 10%


def test_step_choice_targets_five_minutes() -> None:
    assert launch.choose_steps(0.1) == 2500  # 300 s / (0.1 s x 1.08) = 2777 -> 2500
    assert launch.choose_steps(1.0) == 2000  # clamped up
    assert launch.choose_steps(0.001) == 10000  # clamped down


def _write_synthetic_runs(root: Path, shared_sd: float, own_sd: float, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    runs_dir = root / f"runs_{shared_sd}_{own_sd}"
    runs_dir.mkdir()
    shared = rng.normal(0, shared_sd, (4, 3))
    for i, m in enumerate(trainer.LR_MULTIPLIERS):
        for k in range(4):
            curve = [{"step": 0, "eval_loss": 5.5}] + [
                {"step": 100 * (c + 1), "eval_loss": 3 - 0.1 * c + 0.02 * i + shared[k, c] + rng.normal(0, own_sd)}
                for c in range(3)
            ]
            run = {"config": {"lr_multiplier": m, "seed_index": k}, "curve": curve}
            (runs_dir / f"lr{m:g}_seed{k}.json").write_text(json.dumps(run))
    return runs_dir


def test_analysis_recommends_under_strong_crn_and_closes_without_it(tmp_path: Path) -> None:
    strong = analysis.analyse(_write_synthetic_runs(tmp_path, shared_sd=0.05, own_sd=0.005))
    assert strong["final"]["median_ratio"] < 0.2
    assert strong["decision"] == "recommend a full experiment"
    assert strong["step0_identical_across_learning_rates"] is True
    none = analysis.analyse(_write_synthetic_runs(tmp_path, shared_sd=0.0, own_sd=0.05, seed=1))
    assert none["final"]["median_ratio"] > 0.5
    assert none["decision"] == "close the CRN program"


def test_pair_ratio_excludes_same_seed_positions_in_bootstrap() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a + 0.5  # perfectly common seed effect: matched gaps are constant
    assert analysis.pair_ratio(a, b) == 0.0
    labels = np.array([0, 0, 1, 2])
    assert np.isfinite(analysis.pair_ratio(a[[0, 0, 1, 2]], b[[0, 0, 1, 2]], labels=labels))


laptop = pytest.importorskip("pilot.crn.laptop")


def test_laptop_step_rule_fits_the_budget() -> None:
    assert laptop.steps_for(0.1) == 10000  # 19800 / (16 x 0.1 x 1.1) = 11250 -> capped
    assert laptop.steps_for(1.0) == 1100  # 1125 -> 1100
    assert 16 * laptop.steps_for(0.7) * 0.7 * 1.1 <= laptop.BUDGET_SECONDS


def test_gate_statistic_separates_nondeterminism_from_seed_effect() -> None:
    exact = laptop.gate_statistic([2.0, 1.8, 1.7], [2.0, 1.8, 1.7], [2.1, 1.9, 1.75])
    assert exact["v_nd"] == 0.0 and exact["passes"] and exact["exactly_reproducible"]
    noisy = laptop.gate_statistic([2.0, 1.8, 1.7], [2.05, 1.85, 1.65], [2.06, 1.86, 1.66])
    assert not noisy["passes"]
    assert not laptop.gate_statistic([1.0], [1.0], [1.0])["passes"]  # no seed effect


def test_floor_check() -> None:
    initial = {(m, s): 5.5 for m in (0.5, 1.0) for s in (0, 1)}
    good = {(0.5, 0): 2.0, (0.5, 1): 2.01, (1.0, 0): 1.9, (1.0, 1): 1.91}
    assert laptop.floor_check(good, initial)["passes"]
    overlapping = {(0.5, 0): 2.0, (0.5, 1): 1.8, (1.0, 0): 1.9, (1.0, 1): 1.91}
    assert not laptop.floor_check(overlapping, initial)["F2_separated"]
    untrained = {k: 5.0 for k in initial}
    assert not laptop.floor_check(untrained, initial)["F1_all_well_below_initial"]


def test_stop_after_checkpoint_keeps_the_full_schedule_prefix(work: Path, runs: dict) -> None:
    short = trainer.train_run(_config(1.0, 0), work, "cpu", stop_after_checkpoint=2)
    assert short["curve"] == runs[(1.0, 0)]["curve"][:3]


def test_divergence_fractions() -> None:
    div = pytest.importorskip("scripts.run_crn_mps_divergence")
    finals = {(m, s): 1.0 + 0.1 * i + 0.01 * s * (i + 1) for i, m in enumerate((0.5, 1.0, 1.5)) for s in range(4)}
    out = div.fractions(finals, {1.0: 0.001, 1.5: 0.002}, [0.5, 1.0, 1.5], [0, 1, 2, 3])
    assert set(out) == {"0.5|1", "0.5|1.5", "1|1.5"}
    var = out["1|1.5"]["matched_gap_variance"]
    assert np.isclose(out["1|1.5"]["decision_fraction"], 0.003 / var)
    assert np.isclose(out["0.5|1"]["decision_fraction"], 0.002 / out["0.5|1"]["matched_gap_variance"])
