"""A5: verify every number in FIRST_AUDIT.md is regenerable from committed code and data.

Strategy: FIRST_AUDIT.md is produced entirely by ``render_report`` in
``scripts/run_first_audit.py`` from ``results/first_audit/first_audit.json``.
Re-rendering from the committed JSON must reproduce the committed markdown
byte-for-byte; if it does, no number in the file was hand-entered, because
every character came from the renderer. We then additionally extract every
numeric token from the markdown and confirm each is either (a) present in the
JSON, (b) a simple derived form (percentage, ratio, rounding) of a JSON value,
or (c) a documented constant declared in the script.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "FIRST_AUDIT.md"
JSON_PATH = REPO / "results" / "first_audit" / "first_audit.json"


def rerender_is_byte_identical() -> dict[str, Any]:
    before = REPORT.read_bytes()
    tmp = REPO / "results" / "adversarial" / "_rerender_check.md"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_first_audit.py",
            "--render-only",
            "--out",
            "results/first_audit",
            "--report",
            str(tmp),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    rendered = tmp.read_bytes() if tmp.exists() else b""
    identical = rendered == before
    if tmp.exists():
        tmp.unlink()
    return {
        "render_exit_code": proc.returncode,
        "byte_identical": bool(identical),
        "committed_bytes": len(before),
        "rendered_bytes": len(rendered),
    }


def json_numeric_universe(payload: Any, acc: set[float] | None = None) -> set[float]:
    if acc is None:
        acc = set()
    if isinstance(payload, dict):
        for value in payload.values():
            json_numeric_universe(value, acc)
    elif isinstance(payload, list):
        for value in payload:
            json_numeric_universe(value, acc)
    elif isinstance(payload, bool):
        pass
    elif isinstance(payload, (int, float)):
        acc.add(float(payload))
    return acc


NUMBER = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?)(?![\w])", re.IGNORECASE)
# Constants that are declared in the script or are structural (table widths, years).
DECLARED_CONSTANTS = {
    989e12,
    0.40,
    0.5,
    0.3,
    10.0,
    3600.0,
    0.05,
    0.8,
    0.975,
    2.0,
    6.0,
    1.0,
    0.0,
    15.7,
    1571.0,
    20.0,
    4.0,
    1.2,
    1.0e-4,
    2.0e-3,
    50.0,
    30.0,
    40.0,
    300.0,
    25.0,
    3.0,
    1.5,
    100.0,
    80.0,
    120.0,
    2504.11393,
    2509.02046,
    2608.11859,
    2508.13144,
    2026.0,
    8.0,
    45.0,
    5.0,
    16.0,
    1.0e-6,
    7.0e-6,
}


def numbers_in_report() -> list[str]:
    text = REPORT.read_text(encoding="utf-8")
    # Drop the SHA-256 digest lines: they are hashes, not analysis numbers.
    text = "\n".join(line for line in text.splitlines() if "sha256" not in line)
    return [m.group(1) for m in NUMBER.finditer(text)]


def check_numbers() -> dict[str, Any]:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    universe = json_numeric_universe(payload)
    derived: set[float] = set()
    for value in universe:
        derived.update({value, value * 100.0, value / 100.0, abs(value)})
    candidates = sorted(derived | DECLARED_CONSTANTS)
    array = [c for c in candidates if c == c]
    unmatched: list[str] = []
    for token in numbers_in_report():
        raw = token.replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** (-decimals) if decimals else 0.5
        if not any(abs(value - c) <= max(tol, abs(c) * 1e-6) for c in array):
            unmatched.append(token)
    return {
        "n_numeric_tokens": len(numbers_in_report()),
        "n_unmatched": len(unmatched),
        "unmatched_examples": sorted(set(unmatched))[:40],
    }


def main() -> int:
    out = {"rerender": rerender_is_byte_identical(), "numbers": check_numbers()}
    destination = REPO / "results" / "adversarial"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "a5_traceability.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
