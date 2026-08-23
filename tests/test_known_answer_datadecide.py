"""Known-answer validation of the pipeline against DataDecide's published decision accuracy.

The synthetic tests exercise the protocol code. The real-data test runs on the
harvested table committed under ``data/`` (``asla harvest-datadecide --metric
olmes_macro_error``) and checks the 150M single-scale pairwise decision accuracy
against the published ~80% within the band documented in KNOWN_ANSWER.md. A
failure here means our pipeline does not reproduce a published number and must
be reported, not tuned away.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asla.analysis.known_answer import PUBLISHED_SINGLE_SCALE_150M, known_answer_report, pairwise_decision_accuracy
from asla.data.io import load_runs

REAL_TABLE = Path("data/datadecide_runs_olmes_macro_error.parquet")


def _synthetic(flip_pairs: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    names = [f"r{i}" for i in range(6)]
    rows = []
    for i, name in enumerate(names):
        for scale_label, compute in (("4M", 1.0), ("150M", 10.0), ("1B", 100.0)):
            for seed in range(3):
                small_value = 1.0 + 0.1 * i
                large_value = 1.0 + 0.1 * (len(names) - 1 - i) if (flip_pairs and compute == 100.0) else small_value
                rows.append(
                    {
                        "intervention": name, "intervention_class": "data", "compute": compute, "seed": seed,
                        "bpb": (large_value if compute == 100.0 else small_value) + rng.normal(0, 1e-4),
                        "metric_name": "olmes_macro_error", "scale_label": scale_label,
                    }
                )
    return pd.DataFrame(rows)


def test_protocols_agree_on_noise_free_tables():
    df = _synthetic()
    seed_mean = pairwise_decision_accuracy(df, 10.0, 100.0, protocol="seed_mean")
    per_seed = pairwise_decision_accuracy(df, 10.0, 100.0, protocol="per_seed")
    assert seed_mean["pairwise_acc"] == 1.0 and per_seed["pairwise_acc"] == 1.0
    assert seed_mean["n_pairs"] == 15 and per_seed["per_seed"] == {0: 1.0, 1: 1.0, 2: 1.0}
    reversed_df = _synthetic(flip_pairs=1)
    assert pairwise_decision_accuracy(reversed_df, 10.0, 100.0)["pairwise_acc"] == 0.0


def test_known_answer_report_checks_focal_scale_against_band():
    report = known_answer_report(_synthetic())
    assert report["focal_scale"] == "150M"
    assert report["checks"]["seed_mean"]["computed"] == 1.0
    assert report["checks"]["seed_mean"]["within_band"] is False  # 1.0 is outside 0.75-0.85 by design
    assert report["reproduced"] is False
    assert [entry["scale_label"] for entry in report["per_scale"]] == ["4M", "150M"]


def test_known_answer_report_refuses_wrong_metric():
    df = _synthetic()
    df["metric_name"] = "c4_en_bits_per_token"
    with pytest.raises(ValueError, match="defined on metric"):
        known_answer_report(df)


@pytest.mark.skipif(not REAL_TABLE.exists(), reason="harvested DataDecide table not present")
def test_pipeline_reproduces_published_datadecide_decision_accuracy():
    df = load_runs(REAL_TABLE)
    report = known_answer_report(df)
    lo, hi = PUBLISHED_SINGLE_SCALE_150M["band"]
    for protocol, check in report["checks"].items():
        assert lo <= check["computed"] <= hi, (
            f"{protocol}: computed 150M->1B pairwise decision accuracy {check['computed']:.3f} is outside the "
            f"published band [{lo}, {hi}] (published ~{check['published']}); see KNOWN_ANSWER.md"
        )
    assert report["reproduced"]
