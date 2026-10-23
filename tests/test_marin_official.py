"""Tests for the instance-11 guards on loading the Marin board."""

from __future__ import annotations

import importlib
import json

import pytest

mo = importlib.import_module("scripts.marin_official")
rig = importlib.import_module("scripts.run_marin_rigour")


def _runs(groups: dict[str, list[str]], flops: float = 4e20) -> list[dict]:
    return [
        {"run_name": f"{g}/{label}", "training_hardware_flops": flops} for g, labels in groups.items() for label in labels
    ]


def test_unrecognised_rung_label_in_an_official_group_is_a_hard_error() -> None:
    official = {"a"}
    assert mo.rung("a/1.2b", official) == "1_2b" and mo.rung("a/520m", official) == "520m"
    with pytest.raises(ValueError, match="unrecognised rung label"):
        mo.rung("a/1.3b", official)
    with pytest.raises(ValueError, match="unrecognised rung label"):
        mo.rung("b/weird", official)  # not official, not a known ladder: still fails
    assert mo.rung("parallel_llama/old_scaling/gpu/75m", official) is None
    assert mo.rung("nano_arch_ablations/00_baseline", official) is None


def test_official_groups_follow_the_display_rule() -> None:
    runs = _runs({"a": ["130m"], "b": ["130m"]}) + [{"run_name": "c/130m", "training_hardware_flops": 1e19}]
    assert mo.official_groups(runs) == ["a", "b"]


def test_member_check_fails_on_any_undocumented_difference(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _runs({"a": ["130m"], "b": ["130m"], "hackable_transformer_attn_sink/std_attn": ["1.2b"]})
    monkeypatch.setattr(mo, "OFFICIAL_COUNT_AT_PIN", 3)
    assert mo.check_members(["a", "b"], runs, allow_documented=True) == ["hackable_transformer_attn_sink/std_attn"]
    with pytest.raises(ValueError, match="missing"):
        mo.check_members(["a", "b"], runs, allow_documented=False)
    with pytest.raises(ValueError, match="missing"):
        mo.check_members(["a"], runs, allow_documented=True)
    with pytest.raises(ValueError, match="not official"):
        mo.check_members(["a", "b", "hackable_transformer_attn_sink/std_attn", "z"], runs, allow_documented=True)
    monkeypatch.setattr(mo, "OFFICIAL_COUNT_AT_PIN", 4)
    with pytest.raises(ValueError, match="expected 4"):
        mo.check_members(["a", "b"], runs, allow_documented=True)


@pytest.mark.skipif(not (rig.RAW / "runs.json").exists(), reason="board data not harvested")
def test_the_pinned_board_has_22_official_groups_and_loads_cleanly() -> None:
    runs = json.loads((rig.RAW / "runs.json").read_text())
    official = mo.official_groups(runs)
    assert len(official) == 22 and "hackable_transformer_attn_sink/std_attn" in official
    members, cells, _ = rig.load_cells()
    assert len(members) == 21 and len(cells) == 84
