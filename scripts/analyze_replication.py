"""Analysis entry point for the pre-registered replication (PREDICTIONS_REPLICATION_PROTOCOL.md).

Input: a CSV with one row per finished run and the columns in RUN_COLUMNS (see replication_protocol/README.md).
The frozen code path is the one this repository used on DataDecide: asla.analysis.confound.window_designs
(power-law fits on seed means, single-scale ranking at the window's top, expected-error kernels against each
target's own seed evidence) and asla.analysis.fixed_target (the pooled fixed-target slope and its
leave-one-recipe-out jackknife). Before computing, the script checks the SHA-256 of every frozen module against
PINNED and refuses to run on a mismatch, so a later edit to the analysis cannot silently change the verdict.

Usage: .venv/bin/python scripts/analyze_replication.py RUNS.csv [--out verdict.json] [--allow-incomplete] [--no-stability]
           [--per-document-logs DIR --sensitivity-exclude EXCL.json]

Decontamination sensitivity (protocol v2.3, enforced in v2.4): with --sensitivity-exclude, every run in RUNS.csv
must have final-checkpoint per-document logs under --per-document-logs, or the script refuses; each exclusion
set is rescored by scripts/rescore_replication.py and passed through the same analyse(). The primary verdict
is always the one from RUNS.csv.

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
# protocol v2.4 (2026-10-31): the frozen protocol files; any edit after the freeze is refused like a code change
PROTOCOL_PINS = {
    "replication_protocol/recipes.json": "aec1b7ec06b89e942faa685028e4fb5101e86e02ab4822921795de8b69d62199",
    "replication_protocol/ladder.json": "3dc16ad942f997b5da698f4cca589347bba4b437332b2856740d852ba3659de4",
    "replication_protocol/preregistration.json": "0ff05826528364d8175bbca7ee62faa5022d456bb25ba75a61ed76a65f5fb661",
}
PROTOCOL = REPO / "replication_protocol"
# protocol v2.4: the rescore and the evaluator patch are frozen with the protocol
PROTOCOL_PINS.update({
    "scripts/rescore_replication.py": "5edfb7cc45ac1f2f3bc3704c5d3d212ae59605db4a68addc6961cbc5f6d956ca",
    "replication_protocol/olmo_patch/olmo_per_document.patch": "4d71d612a0834752c7cf4f9cfd9670a5a688a34cfd9b64a891a63c971a21a74f",
    "replication_protocol/olmo_patch/per_document.py": "7bd9ff72334303246267da9d7595a7b24fc0fc1dedb438719ca2f177eafd5287",
})
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
    runs = pd.read_csv(path, float_precision="round_trip")
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
    frame = to_frame(runs)
    return frame, {"runs_used": int(len(frame)), "runs_missing": len(absent), "recipes_dropped": dropped}


def to_frame(runs: pd.DataFrame) -> pd.DataFrame:
    ladder = json.loads((PROTOCOL / "ladder.json").read_text(encoding="utf-8"))
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
    return frame


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


def run_ids_of(runs: pd.DataFrame) -> list[str]:
    return [f"{r}__{s}__seed{k}" for r, s, k in zip(runs["recipe"], runs["size"], runs["seed"])]


def exclusion_sets(path: Path) -> dict[str, set[str]]:
    """A list of ids, {"exclude_eval_ids": [...]}, or the output of rescore_replication.py exclusions."""

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        return {"exclusion": set(d)}
    if "exclude_eval_ids" in d:
        return {"exclusion": set(d["exclude_eval_ids"])}
    sets = {f"corpus:{c}": set(v["exclude_eval_ids"]) for c, v in d["corpora"].items()}
    if d.get("union"):
        sets["union"] = set(d["union"])
    return sets


def sensitivity(runs_path: Path, logs: Path, exclude_path: Path) -> dict:
    """Rescore every run without each exclusion set and rerun the unchanged analysis; refuse on missing logs."""

    from scripts import rescore_replication as rs

    runs = pd.read_csv(runs_path, float_precision="round_trip")
    ids = run_ids_of(runs)
    gone = rs.missing_logs(logs, ids)
    if gone:
        raise SystemExit(f"refusing the sensitivity analysis: {len(gone)} runs lack per-document logs, e.g. {gone[:3]}")
    full = rs.build_table(logs, ids)
    out = {
        "primary_runs_match_logs_exactly": bool((full["c4_en_bits_per_token"].to_numpy() == runs["c4_en_bits_per_token"].to_numpy()).all()),
        "sets": {},
    }
    for name, ex in exclusion_sets(exclude_path).items():
        res = analyse(to_frame(rs.build_table(logs, ids, ex)))
        res["excluded_eval_ids"] = sorted(ex)
        res["secondary"] = secondary(res["estimate"])
        out["sets"][name] = res
    out["note"] = "sensitivity only; the primary verdict comes from the full evaluation set"
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    ap.add_argument("--per-document-logs", type=Path)
    ap.add_argument("--sensitivity-exclude", type=Path)
    a = ap.parse_args(argv)
    pins = check_pins()
    if a.sensitivity_exclude is not None and a.per_document_logs is None:
        raise SystemExit("--sensitivity-exclude needs --per-document-logs")
    frame, completeness = load_runs(a.runs, a.allow_incomplete)
    res = analyse(frame)
    res["secondary"] = secondary(res["estimate"])
    res["completeness"] = completeness
    if a.sensitivity_exclude is not None:
        res["decontamination_sensitivity"] = sensitivity(a.runs, a.per_document_logs, a.sensitivity_exclude)
    if not a.no_stability:
        res["numerical_stability"] = stability(frame, res)
    res["pinned_sha256"] = pins
    res["preregistration"] = "PREDICTIONS_REPLICATION_PROTOCOL.md"
    res["protocol_version"] = "v2.4"
    text = json.dumps(res, indent=2) + "\n"
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)
    return res


if __name__ == "__main__":
    main()
