"""Run simulator jobs in separate subprocesses with a per-job timeout, checkpointing each result as it lands.

A job is {"key", "spec", "worlds": [start, stop], "seed"}; it runs asla.analysis.sim_design.world for each
world index in the range, in its own ``nice -n 19`` Python process. At most two run at once. Each result
(or a timeout or failure record) is appended to the cache file the moment it completes, so a hang loses one
job, not the run. Re-running skips every key already in the cache with a result.

Worker mode (used internally): ``python scripts/sim_runner.py worker '<job json>'`` prints one JSON line.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = str(REPO / ".venv" / "bin" / "python")
_LOCK = threading.Lock()


def _worker(job: dict) -> dict:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import warnings

    import numpy as np

    warnings.filterwarnings("ignore")
    from asla.analysis.sim_design import world

    out = []
    for w in range(*job["worlds"]):
        rng = np.random.default_rng([int(job["seed"]), zlib.crc32(job["key"].encode()) % (2**31), w])
        out.append(world(job["spec"], rng))
    return {"key": job["key"], "results": out}


def read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if "results" in row:
                rows[row["key"]] = row
            else:
                rows.setdefault(row["key"], row)
    return rows


def _append(path: Path, row: dict) -> None:
    with _LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def _launch(job: dict, path: Path, timeout: float) -> str:
    cmd = ["nice", "-n", "19", PYTHON, "-W", "ignore", str(Path(__file__).resolve()), "worker", json.dumps(job)]
    try:
        done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        _append(path, {"key": job["key"], "timeout": True, "timeout_s": timeout})
        return "timeout"
    if done.returncode != 0:
        _append(path, {"key": job["key"], "failed": True, "stderr": done.stderr[-2000:]})
        return "failed"
    _append(path, json.loads(done.stdout.strip().splitlines()[-1]))
    return "ok"


def run_jobs(jobs: list[dict], path: Path, workers: int = 2, timeout: float = 900.0) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    have = {k for k, v in read_cache(path).items() if "results" in v}
    todo = [j for j in jobs if j["key"] not in have]
    counts = {"ok": 0, "timeout": 0, "failed": 0, "skipped": len(jobs) - len(todo)}
    with ThreadPoolExecutor(max_workers=min(workers, 2)) as pool:
        futures = [pool.submit(_launch, j, path, timeout) for j in todo]
        for i, f in enumerate(as_completed(futures), start=1):
            counts[f.result()] += 1
            if i % 10 == 0 or i == len(todo):
                print(f"{path.name}: {i}/{len(todo)} jobs {counts}", flush=True)
    return counts


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "worker":
    print(json.dumps(_worker(json.loads(sys.argv[2]))))
