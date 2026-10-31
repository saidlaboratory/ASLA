"""Dry run of per-document logging and the decontamination-sensitivity rescore (protocol v2.5). Trains nothing.

Synthetic per-document logs for all 240 protocol runs are written through the evaluator patch's own
accumulator (replication_protocol/olmo_patch/per_document.py) on the deterministic, complete loader's layout:
the 2,099 document spans of replication_protocol/eval_set.jsonl, windows of 2048 tokens with stride 2047 over
[EOS] + file (500 windows, every token predicted once), dealt to ranks as the padded rank sampler does (rank r
gets r, r + W, ...; masked dummies never logged), at a simulated world size per run. Run-level losses follow one
simulated world of the frozen design (scripts/dry_run_replication.py, world 0).

Checks:
1. rescoring with an empty exclusion list reproduces the primary bits per token exactly (bitwise, also after the
   CSV round trip), and equals an independent computation from the raw token losses to rounding;
2. a planted exclusion (two documents, which one contaminated recipe's runs find easy) changes every run, since
   every run scores every document, by the amount an independent recomputation gives;
3. the rescore refuses logs with a window missing and logs with a window on two ranks;
4. the v2.3 exclusion rule, applied to synthetic decontamination reports, recovers the planted documents;
5. the entry point's sensitivity mode runs, and refuses when one run's logs are missing.

Run: nice -n 19 .venv/bin/python scripts/dry_run_rescore.py LOG_DIR
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import rescore_replication as rs  # noqa: E402

OUT = REPO / "results" / "replication" / "rescore_dry_run.json"
CHUNK = 2048
SEED = 20261031
CONTAMINATED = "cc100_en"  # its runs find the planted documents easy


def per_document_module():
    spec = importlib.util.spec_from_file_location("per_document", REPO / "replication_protocol" / "olmo_patch" / "per_document.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_layout():
    rows = [json.loads(l) for l in (REPO / "replication_protocol" / "eval_set.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = np.array([r["token_start"] for r in rows])
    ends = np.array([r["token_end"] for r in rows])
    ids = [r["id"] for r in rows]
    return starts, ends, ids, int(ends.max())


def run_bpb_targets() -> dict[str, float]:
    d = importlib.import_module("scripts.dry_run_replication")
    cells = d.draw(d.spec(), np.random.default_rng([d.SEED, 0]))
    table = d.runs_table(cells)
    return {f"{r}__{s}__seed{k}": float(b) for r, s, k, b in zip(table["recipe"], table["size"], table["seed"], table["c4_en_bits_per_token"])}


def synth_logs(logs: Path, pd_mod, starts, ends, ids, n_tokens, planted_docs) -> dict:
    runs = rs.protocol_runs()
    target = run_bpb_targets()
    rng = np.random.default_rng(SEED)
    doc_effect = rng.normal(0.0, 0.15, len(ids))
    doc_of = np.searchsorted(starts, np.arange(n_tokens, dtype=np.int64), side="right") - 1
    layout = pd_mod.complete_layout(n_tokens, CHUNK)
    raw = {}
    for rid, cfg in runs.items():
        b = target[rid] * math.log(2)
        tok = b * (1.0 + doc_effect[doc_of]) * np.exp(0.1 * rng.standard_normal(n_tokens))
        if cfg["recipe"] == CONTAMINATED:
            tok[np.isin(doc_of, planted_docs)] *= 0.3  # memorized evaluation text
        world = int(rng.choice([1, 2, 4, 8, 16]))
        batch = int(rng.choice([1, 2, 3]))
        total = -(-len(layout) // (world * batch)) * world * batch  # the padded rank sampler's length
        step = cfg["train"]["steps"]
        for rank in range(world):
            log = pd_mod.PerDocumentLog(starts, ends, ids, CHUNK)
            for k in range(rank, total, world):
                if k >= len(layout):
                    continue  # a masked dummy instance: never logged
                off, n_pred = layout[k]
                loss = np.zeros(CHUNK - 1)
                valid = np.zeros(CHUNK - 1, dtype=bool)
                loss[:n_pred] = tok[off : off + n_pred]
                valid[:n_pred] = True
                log.add(k, off, loss, valid)
            log.write(logs / rid / f"{rs.LABEL}_step{step}_rank{rank}.jsonl.gz", rs.LABEL, step, rank)
        raw[rid] = {"tok": tok, "world": world, "batch": batch, "dummies": total - len(layout)}
    return raw


def main(log_dir: str) -> dict:
    logs = Path(log_dir)
    if logs.exists():
        shutil.rmtree(logs)
    pd_mod = per_document_module()
    starts, ends, ids, n_tokens = eval_layout()
    pdocs = [10, 11]
    planted_ids = [ids[i] for i in pdocs]
    raw = synth_logs(logs, pd_mod, starts, ends, ids, n_tokens, pdocs)
    runs = rs.protocol_runs()
    run_ids = list(runs)

    # 1. empty exclusion = primary, exactly
    primary = rs.build_table(logs, run_ids)
    again = rs.build_table(logs, run_ids, set())
    csv_path = logs / "runs.csv"
    primary.to_csv(csv_path, index=False, float_format="%.17g")
    reread = pd.read_csv(csv_path, float_precision="round_trip")
    independent_full = np.array([raw[r]["tok"].sum() / n_tokens / math.log(2) for r in run_ids])
    rel = np.abs(primary["c4_en_bits_per_token"].to_numpy() / independent_full - 1)

    # 2. planted exclusion
    excl = set(planted_ids)
    sens = rs.build_table(logs, run_ids, excl)
    changed = primary["c4_en_bits_per_token"].to_numpy() != sens["c4_en_bits_per_token"].to_numpy()
    doc_of = np.searchsorted(starts, np.arange(n_tokens, dtype=np.int64), side="right") - 1
    keep_tok = ~np.isin(doc_of, pdocs)
    indep = [raw[r]["tok"][keep_tok].sum() / keep_tok.sum() / math.log(2) for r in run_ids]
    indep_rel = np.abs(sens["c4_en_bits_per_token"].to_numpy() / np.array(indep) - 1)
    contaminated = np.array([runs[r]["recipe"] == CONTAMINATED for r in run_ids])
    shift = sens["c4_en_bits_per_token"].to_numpy() - primary["c4_en_bits_per_token"].to_numpy()

    # 3. the rescore refuses incomplete or double-scored logs
    victim = run_ids[1]
    step_v = runs[victim]["train"]["steps"]
    files_v = rs.log_files(logs, victim, step_v)
    import gzip as _gz

    def refuses(edit) -> str | None:
        tmp = logs.parent / (logs.name + "_edit")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(logs / victim, tmp)
        edit(tmp)
        try:
            rs.score_dir(tmp, step_v)
        except ValueError as e:
            return str(e)
        finally:
            shutil.rmtree(tmp)
        return None

    def drop_window(d: Path) -> None:
        f = d / files_v[0].name
        lines = _gz.open(f, "rt", encoding="utf-8").read().splitlines()
        first = json.loads(lines[1])["instance"]
        keep = [lines[0]] + [l for l in lines[1:] if json.loads(l)["instance"] != first]
        with _gz.open(f, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(keep) + "\n")

    def duplicate_window(d: Path) -> None:
        lines = _gz.open(d / files_v[0].name, "rt", encoding="utf-8").read().splitlines()
        hdr = json.loads(lines[0])
        hdr["rank"] = 99
        with _gz.open(d / f"{rs.LABEL}_step{step_v}_rank99.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write("\n".join([json.dumps(hdr), lines[1]]) + "\n")

    refuse_missing = refuses(drop_window)
    refuse_duplicate = refuses(duplicate_window)

    # 4. the v2.3 rule on synthetic decontamination reports
    reps = []
    for j, rec in enumerate(sorted({c["recipe"] for c in runs.values()})):
        if rec == CONTAMINATED:
            hits = {planted_ids[0]: 40, planted_ids[1]: 30, ids[0]: 20, ids[1]: 10}
            n13 = 100
        else:
            hits = {ids[2 + j]: 5}
            n13 = 5
        reps.append({"corpus": rec, "documents_in": 2000 if rec == CONTAMINATED else 1000, "removed_13gram": n13, "eval_ids_hit_13gram": hits})
    rule = rs.exclusions(reps)
    excl_path = logs / "exclusions.json"
    excl_path.write_text(json.dumps(rule, indent=2), encoding="utf-8")

    # 5. entry point: sensitivity mode, and refusal on missing logs
    entry = importlib.import_module("scripts.analyze_replication")
    with contextlib.redirect_stdout(io.StringIO()):
        res = entry.main([str(csv_path), "--no-stability", "--per-document-logs", str(logs), "--sensitivity-exclude", str(excl_path)])
    broken = logs.parent / (logs.name + "_missing_one")
    if broken.exists():
        shutil.rmtree(broken)
    shutil.copytree(logs, broken)
    shutil.rmtree(broken / run_ids[0])
    refused = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            entry.main([str(csv_path), "--no-stability", "--per-document-logs", str(broken), "--sensitivity-exclude", str(excl_path)])
    except SystemExit as e:
        refused = str(e)
    shutil.rmtree(broken)

    sens_set = res["decontamination_sensitivity"]["sets"]
    out = {
        "note": "synthetic per-document logs through the evaluator patch's accumulator; nothing trained",
        "runs": len(run_ids),
        "windows_per_run": len(pd_mod.complete_layout(n_tokens, CHUNK)),
        "simulated_world_sizes": sorted({raw[r]["world"] for r in run_ids}),
        "runs_with_masked_dummies": int(sum(raw[r]["dummies"] > 0 for r in run_ids)),
        "every_run_complete": True,  # build_table ran check_complete on every run without error
        "empty_exclusion_reproduces_primary_bitwise": bool((primary["c4_en_bits_per_token"].to_numpy() == again["c4_en_bits_per_token"].to_numpy()).all()),
        "csv_round_trip_bitwise": bool((reread["c4_en_bits_per_token"].to_numpy() == primary["c4_en_bits_per_token"].to_numpy()).all()),
        "max_rel_diff_vs_independent_full_set": float(rel.max()),
        "planted": {
            "eval_ids": planted_ids,
            "runs_that_scored_them": len(run_ids),
            "runs_changed": int(changed.sum()),
            "all_runs_changed_since_all_score_every_document": bool(changed.all()),
            "max_rel_diff_vs_independent_recomputation": float(indep_rel.max()),
            "mean_shift_contaminated_recipe": float(shift[contaminated].mean()),
            "mean_shift_other_runs": float(shift[~contaminated].mean()),
        },
        "rescore_refuses_missing_window": refuse_missing,
        "rescore_refuses_window_on_two_ranks": refuse_duplicate,
        "v2_3_rule": {
            "triggered": sorted(rule["corpora"]),
            "median_13gram_share": rule["median_13gram_share"],
            "recovered_planted_ids": rule["corpora"].get(CONTAMINATED, {}).get("exclude_eval_ids") == planted_ids,
        },
        "entry_point": {
            "primary_verdict": res["verdict"],
            "primary_estimate": res["estimate"],
            "primary_runs_match_logs_exactly": res["decontamination_sensitivity"]["primary_runs_match_logs_exactly"],
            "sensitivity": {k: {"estimate": v["estimate"], "verdict": v["verdict"]} for k, v in sens_set.items()},
            "refused_when_one_run_lacks_logs": refused is not None and refused.startswith("refusing the sensitivity analysis"),
            "refusal_message": refused,
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
