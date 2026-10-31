"""Analysis entry point for the pre-registered replication (PREDICTIONS_REPLICATION_PROTOCOL.md).

Input: a CSV with one row per finished run and the columns in RUN_COLUMNS (see replication_protocol/README.md).
The frozen code path is the one this repository used on DataDecide: asla.analysis.confound.window_designs
(power-law fits on seed means, single-scale ranking at the window's top, expected-error kernels against each
target's own seed evidence) and asla.analysis.fixed_target (the pooled fixed-target slope and its
leave-one-recipe-out jackknife). Before computing, the script checks the SHA-256 of every frozen module against
PINNED and refuses to run on a mismatch, so a later edit to the analysis cannot silently change the verdict.

Usage: .venv/bin/python scripts/analyze_replication.py RUNS.csv [--out verdict.json] [--allow-incomplete] [--no-stability]

Output-only numerical stability diagnostic (protocol v2.1): after the verdict is fixed, the analysis is refitted
under STABILITY_DRAWS independent relative perturbations of STABILITY_REL to every bpb value and to each size's
compute, and the spread of the slope and any verdict change are reported under "numerical_stability". The
diagnostic never feeds back into the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PINNED = {
    "asla/analysis/confound.py": "02027d977532ed90530eacde60cb15d19738e8baaa6f9965e56cb3479f2b2b1c",
    "asla/analysis/fixed_target.py": "ff675d43ac418e66df7bf19a64bc13d4d536eded19ea1782aea91c3c5f19490b",
    "asla/analysis/moderation.py": "b4f832630873c58421030a4ff9129d22e01ea1c2d3487e840d6425586dff8be1",
    "asla/analysis/target_scoring.py": "5b4edb56df30c5c1daf1e7291376dd46b7b76d15f42564705e1ed9f01f6c9dae",
    "asla/models.py": "52b913d526615728d343418fd5cad9f6ad01f309051e48466d2a73a1991f17e0",
}
# protocol v2.1 (2026-10-31): the frozen protocol files; any edit after the freeze is refused like a code change
PROTOCOL_PINS = {
    "replication_protocol/recipes.json": "4131ee41f04f7af8972c6641debebc2a8a61c0907078df4ed81b80b92f127e13",
    "replication_protocol/ladder.json": "78f07fea21b13fe2a6d02371c70d57058b52c6b66be2c2104dcef86757f0c2c0",
    "replication_protocol/preregistration.json": "2d1b027ce2e06bb04dd115c61365cd35f781bd0e18bea1e75cff5bac2e95b1fb",
}
PROTOCOL = REPO / "replication_protocol"
RUN_COLUMNS = ("recipe", "size", "seed", "params_n", "tokens_d", "c4_en_bits_per_token")
Z = 1.959963984540054
MIN_BELOW = 5
TARGET_FLOOR = "150M"
ADMIT_SHARE = 0.5
STABILITY_DRAWS = 20
STABILITY_REL = 1e-12
STABILITY_SEED = 20261031


def check_pins() -> dict[str, str]:
    bad = {}
    for rel, want in {**PINNED, **PROTOCOL_PINS}.items():
        got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        if got != want:
            bad[rel] = got
    if bad:
        raise SystemExit(f"frozen analysis code or protocol changed; refusing to run: {sorted(bad)}")
    return {**PINNED, **PROTOCOL_PINS}


def load_runs(path: Path, allow_incomplete: bool = False) -> tuple[pd.DataFrame, dict]:
    ladder = json.loads((PROTOCOL / "ladder.json").read_text(encoding="utf-8"))
    recipes = [r["id"] for r in json.loads((PROTOCOL / "recipes.json").read_text(encoding="utf-8"))["selected"]]
    sizes = [s["size"] for s in ladder["sizes"]]
    seeds = list(ladder["seeds"])
    runs = pd.read_csv(path)
    missing_cols = [c for c in RUN_COLUMNS if c not in runs.columns]
    if missing_cols:
        raise SystemExit(f"runs file lacks columns {missing_cols}")
    runs = runs[runs["recipe"].isin(recipes) & runs["size"].isin(sizes) & runs["seed"].isin(seeds)]
    have = set(zip(runs["recipe"], runs["size"], runs["seed"]))
    want = {(r, s, k) for r in recipes for s in sizes for k in seeds}
    absent = sorted(want - have)
    dropped: list[str] = []
    if absent and not allow_incomplete:
        raise SystemExit(f"{len(absent)} of {len(want)} runs missing, e.g. {absent[:3]}; see the protocol's missing-run rule")
    if absent:
        # protocol rule: a cell needs at least two seeds; a recipe missing any cell is dropped whole
        counts = runs.groupby(["recipe", "size"]).size()
        for r in recipes:
            if any(counts.get((r, s), 0) < 2 for s in sizes):
                dropped.append(r)
        runs = runs[~runs["recipe"].isin(dropped)]
    frame = pd.DataFrame(
        {
            "intervention": runs["recipe"].astype(str),
            "scale_label": runs["size"].astype(str),
            "seed": runs["seed"],
            "compute": 6.0 * runs["params_n"].astype(float) * runs["tokens_d"].astype(float),
            "bpb": runs["c4_en_bits_per_token"].astype(float),
        }
    ).reset_index(drop=True)
    # one compute per size, as on DataDecide (the ladder's nominal N and D, not per-run token counts)
    nominal = {s["size"]: 6.0 * s["params_n"] * s["tokens_d"] for s in ladder["sizes"]}
    frame["compute"] = frame["scale_label"].map(nominal)
    return frame, {"runs_used": int(len(frame)), "runs_missing": len(absent), "recipes_dropped": dropped}


def analyse(frame: pd.DataFrame) -> dict:
    from asla.analysis.confound import window_designs
    from asla.analysis.fixed_target import fixed_target_slices, jackknife, target_varied_slices

    order = list(frame.groupby("scale_label")["compute"].first().sort_values().index)
    floor = float(frame.loc[frame["scale_label"] == TARGET_FLOOR, "compute"].iloc[0])
    comp = frame.groupby("scale_label")["compute"].first()
    candidates = [i for i, s in enumerate(order) if comp[s] >= floor and i >= MIN_BELOW]
    ds = window_designs(frame)
    shares = {d.target: d.determined_share for d in ds}
    admitted = [t for t in candidates if shares.get(t, 0.0) >= ADMIT_SHARE]
    ft = {t: g for t, g in fixed_target_slices(ds, admitted).items() if g}
    out = {
        "sizes": order,
        "target_candidates": [order[t] for t in candidates],
        "determined_share": {order[t]: shares.get(t) for t in candidates},
        "targets_admitted": [order[t] for t in ft],
    }
    if not ft:
        out.update(estimate=None, se=None, interval=None, verdict="no test: no target admitted")
        return out
    jk = jackknife(ds, ft, target_varied_slices(ds), order)["fixed_target_pooled"]
    est, se = float(jk["estimate"]), float(jk["se"])
    lo, hi = est - Z * se, est + Z * se
    if lo > 0:
        verdict = "replicates: positive, interval excludes zero"
    elif hi < 0:
        verdict = "contradicts: negative, interval excludes zero"
    else:
        verdict = "inconclusive: interval includes zero"
    out.update(estimate=est, se=se, interval=[lo, hi], verdict=verdict, n_designs=len({i for g in ft.values() for gg in g for i in gg}))
    return out


def perturbed(frame: pd.DataFrame, rng: np.random.Generator, rel: float = STABILITY_REL) -> pd.DataFrame:
    """A copy with each bpb value and each size's compute multiplied by (1 + rel z), z standard normal."""

    out = frame.copy()
    out["bpb"] = out["bpb"].to_numpy(dtype=float) * (1.0 + rel * rng.standard_normal(len(out)))
    sizes = sorted(out["scale_label"].unique())
    factor = dict(zip(sizes, 1.0 + rel * rng.standard_normal(len(sizes))))
    out["compute"] = out["compute"].to_numpy(dtype=float) * out["scale_label"].map(factor).to_numpy(dtype=float)
    return out


def stability(frame: pd.DataFrame, base: dict, draws: int = STABILITY_DRAWS, rel: float = STABILITY_REL) -> dict:
    """Output-only: refit under tiny input perturbations; report the slope spread and verdict changes."""

    rng = np.random.default_rng(STABILITY_SEED)
    ests, verdicts = [], []
    for _ in range(draws):
        r = analyse(perturbed(frame, rng, rel))
        ests.append(r["estimate"])
        verdicts.append(r["verdict"].split(":")[0])
    finite = np.array([e for e in ests if e is not None], dtype=float)
    base_v = base["verdict"].split(":")[0]
    dev = finite - base["estimate"] if base["estimate"] is not None and finite.size else np.array([])
    return {
        "draws": draws,
        "relative_perturbation": rel,
        "seed": STABILITY_SEED,
        "slope_min": float(finite.min()) if finite.size else None,
        "slope_max": float(finite.max()) if finite.size else None,
        "slope_sd": float(finite.std(ddof=1)) if finite.size > 1 else None,
        "max_abs_deviation_from_reported": float(np.max(np.abs(dev))) if dev.size else None,
        "verdicts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
        "verdict_changes": int(sum(v != base_v for v in verdicts)),
        "verdict_unstable": bool(any(v != base_v for v in verdicts)),
        "note": "diagnostic only; the reported verdict is the unperturbed one",
    }


def secondary(est: float | None) -> dict:
    pre = json.loads((PROTOCOL / "preregistration.json").read_text(encoding="utf-8"))["secondary"]
    lo, hi = pre["recipe_draw_range"]
    if est is None:
        return {"within_recipe_draw_range": None}
    return {"predicted": pre["predicted_slope"], "recipe_draw_range": [lo, hi], "within_recipe_draw_range": bool(lo <= est <= hi)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    a = ap.parse_args(argv)
    pins = check_pins()
    frame, completeness = load_runs(a.runs, a.allow_incomplete)
    res = analyse(frame)
    res["secondary"] = secondary(res["estimate"])
    res["completeness"] = completeness
    if not a.no_stability:
        res["numerical_stability"] = stability(frame, res)
    res["pinned_sha256"] = pins
    res["preregistration"] = "PREDICTIONS_REPLICATION_PROTOCOL.md"
    res["protocol_version"] = "v2.1"
    text = json.dumps(res, indent=2) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)
    return res


if __name__ == "__main__":
    main()
