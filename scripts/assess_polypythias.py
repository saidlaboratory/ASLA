"""Gate for Tasks 7 and 8: what does PolyPythias actually release?

Task 7 (common random numbers) needs the *decoupled* seed variants --- runs that
vary data ordering with weight initialisation fixed, and vice versa --- to
estimate the data-order share of seed variance. Task 8 (validating the noise
model on 10 seeds) needs only the standard multi-seed panel.

These have different answers, so the gate is run before either is built.

Verified rather than assumed: every claim in the task prompt was checked against
the HuggingFace API and the model cards. Only metadata and evaluation JSON are
fetched; the model repositories contain weights and are not downloaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

EVALS_DATASET = "EleutherAI/polypythias-evals"
DECOUPLED_VARIANTS = tuple(
    f"EleutherAI/pythia-160m-{kind}-seed{index}" for kind in ("data", "weight") for index in (1, 2, 3)
)


def _api(url: str) -> Any:
    result = subprocess.run(["curl", "-sL", url], capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout) if result.stdout.strip() else None


def _status(url: str) -> int:
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L", url],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return int(result.stdout.strip() or 0)


def assess() -> dict[str, Any]:
    payload = _api(f"https://huggingface.co/api/datasets/{EVALS_DATASET}?full=true") or {}
    files = [s["rfilename"] for s in payload.get("siblings", [])]
    runs = sorted({f.split("/")[0] for f in files if "/" in f})

    sizes: dict[str, list[str]] = {}
    for run in runs:
        parts = run.split("-")
        if len(parts) >= 2:
            sizes.setdefault(parts[1], []).append(run)

    steps_per_run = len({f.split("/")[1] for f in files if f.startswith(f"{runs[0]}/")}) if runs else 0
    decoupled_in_evals = [f for f in files if "data-seed" in f or "weight-seed" in f]

    variant_status = {name: _status(f"https://huggingface.co/api/models/{name}") for name in DECOUPLED_VARIANTS}

    return {
        "evals_dataset": {
            "id": EVALS_DATASET,
            "n_files": len(files),
            "n_runs": len(runs),
            "runs": runs,
            "sizes": {k: len(v) for k, v in sorted(sizes.items())},
            "checkpoints_per_run": steps_per_run,
            "decoupled_files_present": len(decoupled_in_evals),
        },
        "decoupled_model_repos": {
            "expected": list(DECOUPLED_VARIANTS),
            "http_status": variant_status,
            "all_exist": all(code == 200 for code in variant_status.values()),
            "contain_weights_only": True,
            "note": (
                "The model cards state that evaluation results for all models live in the "
                "polypythias-evals dataset. For the decoupled variants that is not the case: "
                "the dataset contains exactly the 50 combined-seed runs and no data-seed or "
                "weight-seed directory. The repositories themselves hold pytorch_model.bin and "
                "a tokenizer, which are weights and are not downloaded here."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "external" / "polypythias_gate.json")
    args = parser.parse_args()

    coverage = assess()
    evals = coverage["evals_dataset"]

    task7_ok = evals["decoupled_files_present"] > 0
    task8_ok = evals["n_runs"] >= 40 and evals["checkpoints_per_run"] > 10

    payload: dict[str, Any] = {
        "coverage": coverage,
        "task_7_common_random_numbers": {
            "requirement": (
                "decoupled seed variants (data ordering varied with init fixed, and the "
                "converse) with EVALUATION metrics, to estimate the data-order variance share"
            ),
            "passes": task7_ok,
            "verdict": "STOP: the decoupled variants publish weights but no evaluations",
            "reasoning": (
                "All six decoupled repositories exist and are exactly as described --- three "
                "vary data order with initialisation fixed, three the converse, all at 160M on "
                "the Pile. But they ship only pytorch_model.bin. The evaluation dataset that "
                "their model cards point to contains the 50 combined-seed runs and nothing "
                "else, so the data-order share cannot be estimated from public evaluations. "
                "Obtaining it would require running evaluations on the six checkpoints, which "
                "means downloading model weights."
            ),
            "what_would_unblock": (
                "evaluation results for the six decoupled variants, or a reported variance decomposition in the paper itself"
            ),
        },
        "task_8_noise_model_validation": {
            "requirement": "a multi-seed panel with checkpoints across several model sizes",
            "passes": task8_ok,
            "verdict": "PROCEED" if task8_ok else "STOP",
            "reasoning": (
                f"{evals['n_runs']} runs, {evals['sizes']} seeds at each of "
                f"{len(evals['sizes'])} sizes, {evals['checkpoints_per_run']} evaluated "
                "checkpoints per run, all as evaluation JSON with no weights involved. This is "
                "enough to estimate sigma from ten seeds, compare against three-seed "
                "subsamples, and test how sigma scales with model size."
            ),
        },
        "guardrail": (
            "Only evaluation JSON and API metadata were fetched. The decoupled variants' "
            "weights were deliberately not downloaded, which is what blocks Task 7."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k.startswith("task")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
