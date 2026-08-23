import pandas as pd
import pytest

from asla.data.schema import validate
from asla.data.sources import signal_and_noise as sn


def _core():
    rows = []
    mixes = (
        ("c4", "c4"),
        ("dolma1_7", "dolma17"),
        ("dclm-baseline-qc-10p", "pos_eli5_oh_neg_dclm_refinedweb_steps_2000_lr3e4_top10p"),
    )
    for slug, mix in mixes:
        sizes = (("4M", 3.7e6, 3.75e8, ""), ("150M", 1.5e8, 1.5e10, "-2"), ("1B", 1.18e9, 1.0e11, "-2"))
        for size, n, tokens, seed in sizes:
            for step in (1000.0, 5000.0):
                score = 1.0 + 0.1 * len(slug) / 10 - 0.05 * (n > 1e8) - 0.001 * step / 1000
                rows.append(
                    {
                        "task": "paloma_c4_en", "model": f"{mix}-{size}-5xC{seed}", "model_type": "datadecide",
                        "model_path": f"allenai/DataDecide-{slug}-{size}", "model_revision": f"step{int(step)}-seed-default",
                        "primary_score": score, "primary_metric": "logits_per_byte_corr", "model_params": n,
                        "model_tokens": tokens, "flops": 6 * n * tokens, "step": step, "mix": mix, "size": size,
                        "bits_per_byte": score,
                    }
                )
    rows.append({**rows[0], "task": "hellaswag", "primary_score": 0.4, "bits_per_byte": None})
    rows.append({**rows[0], "model_type": "external", "model": "pythia", "model_path": "EleutherAI/pythia-1b"})
    return pd.DataFrame(rows)


def test_build_runs_table_maps_final_checkpoint_bpb():
    table = sn.build_runs_table(_core())
    validate(table)
    assert len(table) == 9 and set(table["metric_name"]) == {sn.METRIC_NAME}
    assert (table["step"] == 5000).all() and (table["seed"] == 0).all()
    assert set(table["intervention"]) == {"C4", "Dolma1.7", "DCLM-Baseline (QC 10%)"}
    assert set(table[table.scale_label == "1B"]["seed_label"]) == {"2"}
    assert set(table[table.scale_label == "4M"]["seed_label"]) == {"default"}
    assert table["compute"].max() == pytest.approx(6 * 1.18e9 * 1.0e11)


def test_build_runs_table_rejects_bpb_mismatch_and_unknown_slug():
    core = _core()
    bad = core.copy()
    bad.loc[bad["task"] == "paloma_c4_en", "bits_per_byte"] = 9.0
    with pytest.raises(ValueError, match="bits_per_byte"):
        sn.build_runs_table(bad)
    bad = core.copy()
    bad["mix"] = bad["mix"].str.replace("dolma17", "unknown-mix")
    with pytest.raises(ValueError, match="unmapped"):
        sn.build_runs_table(bad)


def test_coverage_summary_lists_scales():
    table = sn.build_runs_table(_core())
    text = sn.format_coverage(sn.coverage_summary(table))
    assert "no seed replicates" in text and "150M" in text
