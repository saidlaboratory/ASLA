"""One-command launcher for the copied-initialisation CRN pilot (PREDICTIONS_TASK_CRN_PILOT.md).

    python -m pilot.crn.launch

Steps:

1. Download and convert the fixed TinyStories sample (about 260 MB, checked
   against pinned SHA-256 digests).
2. For each of the 4 seed indices, build and save the initial weights once and
   draw and save the data order once.
3. Time 50 training steps; set the step count to about 5 minutes per run
   (rounded down to a multiple of 500, clamped to [2000, 10000]); abort if the
   projected total exceeds ``--max-hours``.
4. Write ``manifest.json``, then run the 16 runs seed-major, each in a fresh
   process. Finished runs are skipped on a re-launch (resume).

Outputs: ``results/crn/pilot/manifest.json`` and ``results/crn/pilot/runs/*.json``
(small, committed); data, initial states and orders in ``data/cache/crn_pilot``.
Then run ``python scripts/analyze_crn_pilot.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TARGET_SECONDS_PER_RUN = 300.0
STEP_MULTIPLE, MIN_STEPS, MAX_STEPS = 500, 2000, 10000
OVERHEAD = 1.08  # evaluation at 11 checkpoints and process start-up, on top of the probed step time


def choose_steps(seconds_per_step: float) -> int:
    raw = TARGET_SECONDS_PER_RUN / (seconds_per_step * OVERHEAD)
    steps = int(raw // STEP_MULTIPLE) * STEP_MULTIPLE
    return max(MIN_STEPS, min(MAX_STEPS, steps))


def prepare_shared(work: Path, device: str, steps: int | None = None) -> dict[str, Any]:
    """Data, the 4 initial states and the 4 data orders; returns their digests."""

    from pilot.crn import data, trainer

    cfg = trainer.ModelConfig()
    digests: dict[str, Any] = {"data": data.prepare(work, cfg.context), "init": {}, "order": {}}
    needed = (steps or MAX_STEPS) * trainer.RunConfig(lr_multiplier=1.0, seed_index=0, steps=1).batch_size
    for k in trainer.SEED_INDICES:
        init_path, order_path = work / f"init_seed{k}.pt", work / f"order_seed{k}.npy"
        if not init_path.exists():
            trainer.make_init(k, cfg, init_path)
        if not order_path.exists():
            trainer.make_order(k, work / "train.bin", cfg.context, needed, order_path)
        digests["init"][str(k)] = trainer.file_hash(init_path)
        digests["order"][str(k)] = trainer.file_hash(order_path)
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--work", type=Path, default=REPO / "data" / "cache" / "crn_pilot")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "crn" / "pilot")
    parser.add_argument("--max-hours", type=float, default=2.5)
    args = parser.parse_args()
    import torch

    from pilot.crn import trainer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; the pilot is meant for the RTX 4080")
    manifest_path = args.out / "manifest.json"
    print("preparing data, initial states and data orders ...", flush=True)
    digests = prepare_shared(args.work, args.device)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["digests"] != digests:
            raise SystemExit("inputs changed since the manifest was written; start a fresh --out directory")
        steps = manifest["steps"]
        print(f"resuming with the recorded {steps} steps per run", flush=True)
    else:
        probe_cfg = trainer.RunConfig(lr_multiplier=1.0, seed_index=0, steps=MAX_STEPS)
        seconds = trainer.probe_step_seconds(probe_cfg, args.work, args.device)
        steps = choose_steps(seconds)
        per_run = steps * seconds * OVERHEAD
        total_hours = 16 * per_run / 3600
        print(
            f"probe: {seconds * 1000:.1f} ms/step -> {steps} steps, ~{per_run / 60:.1f} min/run, ~{total_hours:.2f} h total"
        )
        if total_hours > args.max_hours:
            raise SystemExit(f"projected {total_hours:.2f} h exceeds --max-hours {args.max_hours}")
        manifest = {
            "preregistration": "PREDICTIONS_TASK_CRN_PILOT.md",
            "steps": steps,
            "probe_seconds_per_step": seconds,
            "projected_minutes_per_run": per_run / 60,
            "projected_hours_total": total_hours,
            "lr_multipliers": list(trainer.LR_MULTIPLIERS),
            "seed_indices": list(trainer.SEED_INDICES),
            "base_lr": trainer.BASE_LR,
            "model": asdict(trainer.ModelConfig()),
            "digests": digests,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
                "platform": platform.platform(),
            },
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    runs_dir = args.out / "runs"
    for k in trainer.SEED_INDICES:
        for m in trainer.LR_MULTIPLIERS:
            name = trainer.RunConfig(lr_multiplier=m, seed_index=k, steps=steps).name()
            if (runs_dir / f"{name}.json").exists():
                print(f"{name}: done, skipping", flush=True)
                continue
            print(f"{name}: running", flush=True)
            subprocess.run(
                [
                    sys.executable, "-m", "pilot.crn.trainer",
                    "--lr-multiplier", str(m), "--seed-index", str(k), "--steps", str(steps),
                    "--work", str(args.work), "--out", str(runs_dir), "--device", args.device,
                ],
                cwd=REPO,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8"),
                check=True,
            )  # fmt: skip
    print(f"all 16 runs done. Next: python scripts/analyze_crn_pilot.py --runs {runs_dir}")


if __name__ == "__main__":
    main()
