"""The CRN pilot on the MacBook (PREDICTIONS_TASK_CRN_PILOT_LAPTOP.md).

    nice -n 19 python -m pilot.crn.laptop mps-gate      # determinism gate on MPS (3 short runs)
    nice -n 19 python -m pilot.crn.laptop cpu-select    # only if the MPS gate fails
    nice -n 19 python -m pilot.crn.laptop run --device mps|cpu   # the 16 runs; re-run to resume

Laptop safety: one training process at a time (each run is a fresh child
process, run sequentially), ``nice`` inherited from the parent, at most 2 CPU
threads, a total budget of 5.5 hours for the 16 runs. Outputs land in
``results/crn/pilot_laptop/``; data and states in ``data/cache/crn_pilot``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "data" / "cache" / "crn_pilot"
OUT = REPO / "results" / "crn" / "pilot_laptop"
THREADS = 2
BUDGET_SECONDS = 5.5 * 3600
N_RUNS = 16
OVERHEAD = 1.1
MAX_STEPS = 10_000
CPU_MIN_STEPS = 1_000
GATE_CHECKPOINTS = 3
GATE_THRESHOLD = 0.10
# label: (n_layer, d_model, n_head, context, batch), largest first
CANDIDATES: dict[str, tuple[int, int, int, int, int]] = {
    "C0": (6, 384, 6, 256, 64),
    "C1": (4, 256, 4, 256, 32),
    "C2": (4, 192, 4, 128, 32),
    "C3": (3, 128, 4, 128, 32),
    "C4": (2, 96, 4, 128, 16),
}


def steps_for(seconds_per_step: float) -> int:
    """Largest multiple of 100 (<= MAX_STEPS) with 16 x steps x step time x 1.1 within the budget."""

    raw = BUDGET_SECONDS / (N_RUNS * seconds_per_step * OVERHEAD)
    return int(min(MAX_STEPS, (raw // 100) * 100))


def gate_statistic(a: list[float], b: list[float], c: list[float]) -> dict[str, Any]:
    """Nondeterminism against seed effect over the gate checkpoints (step 0 excluded).

    a, b: two identical runs (seed index 0, 1x); c: seed index 1, 1x.
    """

    a_, b_, c_ = (np.asarray(x, dtype=float) for x in (a, b, c))
    v_nd = float(np.mean((a_ - b_) ** 2) / 2)
    v_seed = float(np.mean((a_ - c_) ** 2) / 2) - v_nd
    ratio = v_nd / v_seed if v_seed > 0 else float("inf")
    return {
        "v_nd": v_nd,
        "v_seed": v_seed,
        "ratio": ratio,
        "passes": bool(v_seed > 0 and ratio < GATE_THRESHOLD),
        "exactly_reproducible": bool(np.array_equal(a_, b_)),
    }


def floor_check(final: dict[tuple[float, int], float], initial: dict[tuple[float, int], float]) -> dict[str, Any]:
    """F1: every final <= 0.75 x its step-0 loss. F2: range of LR means > largest within-LR seed range."""

    lrs = sorted({k[0] for k in final})
    f1 = {f"{m:g}_seed{s}": final[(m, s)] <= 0.75 * initial[(m, s)] for (m, s) in final}
    means = {m: float(np.mean([v for (mm, _), v in final.items() if mm == m])) for m in lrs}
    seed_ranges = {m: float(np.ptp([v for (mm, _), v in final.items() if mm == m])) for m in lrs}
    lr_range = max(means.values()) - min(means.values())
    return {
        "F1_all_well_below_initial": all(f1.values()),
        "F1_by_run": f1,
        "F2_lr_range": lr_range,
        "F2_max_within_lr_seed_range": max(seed_ranges.values()),
        "F2_separated": bool(lr_range > max(seed_ranges.values())),
        "passes": bool(all(f1.values()) and lr_range > max(seed_ranges.values())),
    }


def model_args(label: str) -> list[str]:
    n_layer, d_model, n_head, context, batch = CANDIDATES[label]
    return [
        "--n-layer", str(n_layer), "--d-model", str(d_model), "--n-head", str(n_head),
        "--context", str(context), "--batch-size", str(batch),
    ]  # fmt: skip


def prepare(label: str) -> Path:
    """Shared data, plus initial states and orders for one model size; returns its state directory."""

    from pilot.crn import data, trainer

    n_layer, d_model, n_head, context, batch = CANDIDATES[label]
    if not (WORK / "train.bin").exists() or not (WORK / "eval.bin").exists():
        data.prepare(WORK, trainer.ModelConfig().context)
    state = WORK / f"state_{label}"
    cfg = trainer.ModelConfig(n_layer=n_layer, n_head=n_head, d_model=d_model, context=context)
    for k in trainer.SEED_INDICES:
        if not (state / f"init_seed{k}.pt").exists():
            trainer.make_init(k, cfg, state / f"init_seed{k}.pt")
        if not (state / f"order_seed{k}.npy").exists():
            trainer.make_order(k, WORK / "train.bin", context, MAX_STEPS * batch, state / f"order_seed{k}.npy")
    return state


def probe(label: str, device: str) -> float:
    import torch

    from pilot.crn import trainer

    torch.set_num_threads(THREADS)
    n_layer, d_model, n_head, context, batch = CANDIDATES[label]
    cfg = trainer.RunConfig(
        lr_multiplier=1.0,
        seed_index=0,
        steps=MAX_STEPS,
        batch_size=batch,
        model=trainer.ModelConfig(n_layer=n_layer, n_head=n_head, d_model=d_model, context=context),
    )
    return trainer.probe_step_seconds(cfg, WORK, device, n=60, skip=10, state_dir=WORK / f"state_{label}")


def run_one(
    label: str, device: str, steps: int, multiplier: float, seed_index: int, out: Path, extra: list[str] | None = None
) -> subprocess.CompletedProcess:
    command = [
        sys.executable, "-m", "pilot.crn.trainer",
        "--lr-multiplier", str(multiplier), "--seed-index", str(seed_index), "--steps", str(steps),
        "--work", str(WORK), "--state-dir", str(WORK / f"state_{label}"), "--out", str(out),
        "--device", device, "--threads", str(THREADS), *model_args(label), *(extra or []),
    ]  # fmt: skip
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True)


def _curve(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())["curve"]


def mps_gate() -> dict[str, Any]:
    import torch

    if not torch.backends.mps.is_available():
        return {"passes": False, "reason": "MPS not available"}
    prepare("C0")
    seconds = probe("C0", "mps")
    steps = steps_for(seconds)
    gate_dir = OUT / "gate"
    result: dict[str, Any] = {"device": "mps", "label": "C0", "probe_seconds_per_step": seconds, "steps": steps}
    rejected = []
    for tag, seed_index in (("gateA", 0), ("gateB", 0), ("gateC", 1)):
        if (gate_dir / f"lr1_seed{seed_index}_{tag}.json").exists():
            continue
        proc = run_one(
            "C0", "mps", steps, 1.0, seed_index, gate_dir, ["--stop-after-checkpoint", str(GATE_CHECKPOINTS), "--tag", tag]
        )
        if proc.returncode != 0:
            rejected = sorted(set(re.findall(r"(\w+(?:\.\w+)*) does not have a deterministic implementation", proc.stderr)))
            result.update(
                passes=False,
                reason="run failed under deterministic mode",
                rejected_ops=rejected,
                stderr_tail=proc.stderr[-2000:],
            )
            break
    else:
        a, b, c = (
            [p["eval_loss"] for p in _curve(gate_dir / f"lr1_seed{s}_{t}.json")[1 : GATE_CHECKPOINTS + 1]]
            for t, s in (("gateA", 0), ("gateB", 0), ("gateC", 1))
        )
        result.update(gate_statistic(a, b, c), checkpoints_used=GATE_CHECKPOINTS, a=a, b=b, c=c)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mps_gate.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def cpu_select() -> dict[str, Any]:
    probes = {}
    chosen = None
    for label in CANDIDATES:
        prepare(label)
        seconds = probe(label, "cpu")
        steps = steps_for(seconds)
        probes[label] = {"shape": CANDIDATES[label], "seconds_per_step": seconds, "steps": steps}
        if steps >= CPU_MIN_STEPS:
            chosen = label
            break
    result = {
        "device": "cpu",
        "threads": THREADS,
        "probes": probes,
        "chosen": chosen,
        "steps": probes[chosen]["steps"] if chosen else None,
        "passes_budget": chosen is not None,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cpu_select.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _table(runs_dir: Path, seeds: list[int]) -> tuple[dict, dict]:
    final, initial = {}, {}
    for path in runs_dir.glob("lr*_seed*.json"):
        run = json.loads(path.read_text())
        key = (float(run["config"]["lr_multiplier"]), int(run["config"]["seed_index"]))
        if key[1] in seeds:
            final[key], initial[key] = run["curve"][-1]["eval_loss"], run["curve"][0]["eval_loss"]
    return final, initial


def run_pilot(device: str) -> dict[str, Any]:
    from pilot.crn import trainer

    if device == "mps":
        gate = json.loads((OUT / "mps_gate.json").read_text())
        if not gate.get("passes"):
            raise SystemExit("the MPS determinism gate did not pass")
        label, steps = gate["label"], gate["steps"]
    else:
        select = json.loads((OUT / "cpu_select.json").read_text())
        if not select.get("passes_budget"):
            raise SystemExit("no CPU size fits the budget")
        label, steps = select["chosen"], select["steps"]
    runs_dir = OUT / "runs"
    manifest = {"device": device, "label": label, "shape": CANDIDATES[label], "steps": steps, "threads": THREADS}
    if (OUT / "manifest.json").exists() and json.loads((OUT / "manifest.json").read_text()) != manifest:
        raise SystemExit("manifest differs from this configuration; refusing to mix runs")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    status: dict[str, Any] = {"manifest": manifest}
    for k in trainer.SEED_INDICES:
        for m in trainer.LR_MULTIPLIERS:
            name = f"lr{m:g}_seed{k}"
            if (runs_dir / f"{name}.json").exists():
                continue
            started = time.time()
            proc = run_one(label, device, steps, m, k, runs_dir)
            if proc.returncode != 0:
                raise SystemExit(f"{name} failed:\n{proc.stderr[-2000:]}")
            print(f"{name}: {time.time() - started:.0f} s", flush=True)
        if device == "cpu" and k == 1:
            floor = floor_check(*_table(runs_dir, [0, 1]))
            status["floor_after_8_runs"] = floor
            (OUT / "floor.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            if not floor["passes"]:
                raise SystemExit("training floor failed after 8 runs; the pilot stops here")
    status["floor_all_16"] = floor_check(*_table(runs_dir, list(trainer.SEED_INDICES)))
    if device == "cpu":
        repeat_dir = OUT / "repeat"
        if not (repeat_dir / "lr1_seed0_repeat.json").exists():
            proc = run_one(label, "cpu", steps, 1.0, 0, repeat_dir, ["--stop-after-checkpoint", "3", "--tag", "repeat"])
            if proc.returncode != 0:
                raise SystemExit(proc.stderr[-2000:])
        main = [p["eval_loss"] for p in _curve(runs_dir / "lr1_seed0.json")[:4]]
        again = [p["eval_loss"] for p in _curve(repeat_dir / "lr1_seed0_repeat.json")[:4]]
        status["cpu_repeat_exact"] = main == again
        status["v_nd"] = 0.0 if main == again else float(np.mean((np.array(main[1:]) - np.array(again[1:])) ** 2) / 2)
    (OUT / "floor.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mps-gate")
    sub.add_parser("cpu-select")
    run = sub.add_parser("run")
    run.add_argument("--device", choices=["mps", "cpu"], required=True)
    args = parser.parse_args()
    if args.command == "mps-gate":
        print(json.dumps(mps_gate(), indent=2))
    elif args.command == "cpu-select":
        print(json.dumps(cpu_select(), indent=2))
    else:
        print(json.dumps(run_pilot(args.device), indent=2))


if __name__ == "__main__":
    main()
