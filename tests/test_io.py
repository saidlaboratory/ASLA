import pandas as pd
import pytest

from asla.data.io import save_runs_csv
from asla.data.schema import SchemaError


def test_atomic_csv_validation_preserves_existing_destination(tmp_path):
    destination = tmp_path / "runs.csv"
    destination.write_text("original\n", encoding="utf-8")
    invalid = pd.DataFrame(
        {
            "intervention": ["a"],
            "intervention_class": ["x"],
            "compute": [1.0],
            "seed": [0],
            "bpb": [0.0],
        }
    )
    with pytest.raises(SchemaError):
        save_runs_csv(invalid, destination)
    assert destination.read_text(encoding="utf-8") == "original\n"
    assert not list(tmp_path.glob(".runs.csv.*"))


def test_atomic_csv_write_failure_preserves_existing_destination(tmp_path, monkeypatch):
    destination = tmp_path / "runs.csv"
    destination.write_text("original\n", encoding="utf-8")
    valid = pd.DataFrame(
        {
            "intervention": ["a"],
            "intervention_class": ["x"],
            "compute": [1.0],
            "seed": [0],
            "bpb": [1.0],
        }
    )

    def interrupted(self, handle, index=False):
        handle.write("partial\n")
        raise OSError("simulated interruption")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        save_runs_csv(valid, destination)
    assert destination.read_text(encoding="utf-8") == "original\n"
    assert not list(tmp_path.glob(".runs.csv.*"))
