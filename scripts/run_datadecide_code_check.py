"""Task 2 of PREDICTIONS_TASK_LEVER_GAP.md: run DataDecide's own decision-accuracy code.

``allenai/DataDecide`` (``scaling_laws/utils/stats.py: compute_decision_accuracy``)
is run unmodified, with its own preprocessing from ``fit_scaling_laws.run_ladder_fits``,
on the released files already harvested here (the same bytes as the Hugging
Face release). Its output is compared with the released ``decision_acc``.

Their convention, read from the code:
* the target ranking comes from 1B rows with seed ``default`` only (their
  script keeps seed ``default`` or 6198), each model at its highest step,
  sorted by the same metric;
* the predicted ranking sorts recipes by ``stacked_pred``;
* ``compute_2_class`` compares two ordered lists, so tied values are ordered by
  the sort (pandas' default quicksort, which is not stable), not scored as ties.

Usage::

    python scripts/run_datadecide_code_check.py --code-dir <clone of allenai/DataDecide>

Writes ``results/external/datadecide_code_check.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "datadecide"
OUT = REPO / "results" / "external" / "datadecide_code_check.json"
SETUPS_TESTED = (
    "3_param-default",
    "2_param-default",
    "5_param-ai2",
    "3_param-1_step",
    "5_param-1_step-ai2",
    "3_param-default-helper_points",
    "3_param-default-step2=0.5",
    "3_param-default-helper_points-step2=0.5",
)


def _install_tqdm_stub() -> None:
    """Progress bars only; supply a pass-through if tqdm is absent rather than installing it."""

    try:
        import tqdm  # noqa: F401
    except ImportError:
        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda iterable=None, *a, **k: iterable  # type: ignore[attr-defined]
        sys.modules["tqdm"] = stub


class _Unavailable(types.ModuleType):
    """Stand-in for OLMo's ``ladder`` package, which their module imports at the top.

    Only their curve-fitting functions use it; the decision-accuracy path does
    not. Any attribute is a function that raises if called, so a silent
    substitution is impossible: if the path under test touched ``ladder``, the
    run would fail.
    """

    def __getattr__(self, name: str):  # noqa: ANN204
        def unavailable(*_args, **_kwargs):  # noqa: ANN202
            raise RuntimeError(f"ladder.{name} is not available in this check")

        return unavailable


def _install_ladder_stub() -> None:
    try:
        import ladder  # noqa: F401
    except ImportError:
        for name in (
            "ladder",
            "ladder.scaling",
            "ladder.scaling.utils",
            "ladder.scaling.fitting_functions",
            "ladder.fitting",
            "ladder.fitting.step1",
            "ladder.fitting.step2",
            "ladder.fitting.predict",
            "ladder.fitting.step1_flops",
            "ladder.fitting.predict_flops",
            "ladder.fitting.single_step",
        ):
            sys.modules[name] = _Unavailable(name)


def their_preprocessing(macro: pd.DataFrame, clean: dict[str, str]) -> pd.DataFrame:
    """The column restoration and filters of fit_scaling_laws.run_ladder_fits, applied to the released table."""

    expanded = pd.json_normalize([json.loads(m) for m in macro["metrics"]])
    df = pd.concat([macro.drop(columns=["metrics"]).reset_index(drop=True), expanded], axis=1)
    df = df[(df["seed"] == 6198) | (df["seed"] == "default")]
    df["size"] = df["params"]
    reverse = {v: k for k, v in clean.items()}
    df["group"] = df["data"].map(reverse)
    df["model"] = df["group"] + "-" + df["params"].astype(str) + "-" + df["chinchilla"].astype(str)
    return df[df["size"].isin(["150M", "300M", "530M", "750M", "1B"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", type=Path, required=True)
    args = parser.parse_args()
    code = args.code_dir / "scaling_laws"
    sys.path.insert(0, str(code))
    _install_tqdm_stub()
    _install_ladder_stub()
    from utils.constants.constants_recepies import DATA_NAME_CLEAN
    from utils.stats import compute_decision_accuracy

    commit = subprocess.run(
        ["git", "-C", str(args.code_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    df = their_preprocessing(pd.read_parquet(RAW / "eval_macro_avg.parquet"), DATA_NAME_CLEAN)
    released = pd.read_parquet(RAW / "eval_scaling_law_fit.parquet")
    results = released.drop(columns=["decision_acc"]).copy()
    recomputed = compute_decision_accuracy(df, results, "1B")
    keys = ["task", "metric", "setup"]
    released_acc = released.groupby(keys)["decision_acc"].first()
    ours = recomputed.groupby(keys)["decision_acc"].first()
    joined = pd.concat([released_acc.rename("released"), ours.rename("their_code")], axis=1).dropna()
    joined["match"] = (joined["released"] - joined["their_code"]).abs() < 1e-6
    tested = joined.loc[(slice(None), "primary_metric", list(SETUPS_TESTED)), :]
    check_file = REPO / "repro" / "datadecide_decision_acc" / "out" / "nonmatching_rows.csv"
    previously_unmatched = pd.read_csv(check_file)
    no_convention = (
        previously_unmatched[
            ~previously_unmatched.get("matched_by_any_convention", pd.Series(True, index=previously_unmatched.index)).astype(
                bool
            )
        ]
        if "matched_by_any_convention" in previously_unmatched
        else previously_unmatched.iloc[0:0]
    )
    eleven = [(r.task, "primary_metric", r.setup) for r in no_convention.itertuples()]
    eleven_rows = tested.loc[[k for k in eleven if k in tested.index]]
    target = df[(df["size"] == "1B")].copy()
    target = target.loc[target.groupby(["task", "model"])["step"].idxmax()]

    def target_tied_pairs(task: str) -> int:
        counts = target[target["task"] == task]["primary_metric"].value_counts()
        return int((counts * (counts - 1) // 2).sum())

    out = {
        "code": {
            "repository": "allenai/DataDecide",
            "commit": commit,
            "function": "scaling_laws/utils/stats.py:compute_decision_accuracy",
        },
        "convention_read_from_code": [
            "target: 1B, seed 'default' only, each model's highest step, sorted by the same metric (pandas default sort)",
            "prediction: recipes sorted by stacked_pred",
            "score: compute_2_class compares two ordered lists; ties are ordered by the sort, not scored as ties",
        ],
        "all_rows": {"n": int(len(joined)), "matched": int(joined["match"].sum())},
        "tested_88_rows": {"n": int(len(tested)), "matched": int(tested["match"].sum())},
        "previous_no_convention_rows": {
            "n": int(len(eleven_rows)),
            "matched_by_their_code": int(eleven_rows["match"].sum()),
            "rows": [
                {
                    "task": k[0],
                    "setup": k[2],
                    "released": float(v.released),
                    "their_code": float(v.their_code),
                    "target_tied_pairs": target_tied_pairs(k[0]),
                }
                for k, v in eleven_rows.iterrows()
            ],
            "explanation": (
                "every one of these rows is in a task whose 1B default-seed target has tied values; "
                "compute_2_class orders tied recipes by an unstable sort instead of scoring the tie, "
                "which no fixed tie rule reproduces"
            ),
        },
        "nonmatching_tested_rows": [
            {"task": k[0], "setup": k[2], "released": float(v.released), "their_code": float(v.their_code)}
            for k, v in tested[~tested["match"]].iterrows()
        ],
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: out[k] for k in ("all_rows", "tested_88_rows")}),
        "| previous no-convention rows:",
        out["previous_no_convention_rows"]["n"],
        "matched",
        out["previous_no_convention_rows"]["matched_by_their_code"],
    )
    for row in out["nonmatching_tested_rows"][:12]:
        print("  still unmatched:", row)


if __name__ == "__main__":
    main()
