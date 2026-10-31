"""Compare the OLMo smoke-test logs across world sizes and against the independent reference (protocol v2.5).

Reads <out>/ws1 and <out>/ws2 (per-document logs written by olmo_eval_smoke.py run) and <out>/reference.json
(olmo_eval_smoke.py reference), and checks:

1. both world sizes scored the identical set of (instance, document, token count) rows;
2. every evaluation token was scored exactly once at each world size (rescore_replication.check_complete
   against the metadata's document spans);
3. the per-document logs of the two world sizes agree to floating-point precision;
4. scripts/rescore_replication.py's score_dir on each world size's logs reproduces the bits per token, and
   the per-document totals, computed independently from the raw per-token losses.

Run with the analysis environment: .venv/bin/python replication_protocol/smoke_test/compare_smoke.py OUT METADATA.csv.gz RESULT.json
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts import rescore_replication as rs  # noqa: E402


def spans_from(metadata: str) -> dict[str, int]:
    rows = csv.reader(io.TextIOWrapper(gzip.open(metadata), encoding="utf-8"))
    return {r[2]: int(r[1]) - int(r[0]) for r in rows}


def main(out: str, metadata: str, result: str) -> dict:
    out_p = Path(out)
    spans = spans_from(metadata)
    ref = json.loads((out_p / "reference.json").read_text(encoding="utf-8"))
    res: dict = {"reference": {k: ref[k] for k in ("tokens", "every_token_predicted_once", "bits_per_token")}}
    rows, totals = {}, {}
    for ws in (1, 2):
        d = out_p / f"ws{ws}"
        files = rs.log_files_in(d, 0)
        chunks, chunk_size = rs.read_rows(files, 0)
        complete, err = True, None
        try:
            rs.check_complete(chunks, chunk_size, spans)
        except ValueError as e:
            complete, err = False, str(e)
        bits = rs.score_dir(d, 0, spans=spans) if complete else None
        rows[ws] = {(c, doc): (loss, n) for c, rs_ in chunks.items() for doc, loss, n in rs_}
        totals[ws] = rs.doc_totals(chunks)
        infos = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("rank*_info.json"))]
        res[f"world_size_{ws}"] = {
            "rank_files": len(files),
            "chunk_size": chunk_size,
            "instances_logged": len(chunks),
            "padded_total": infos[0]["padded_total"],
            "real_instances": infos[0]["real_instances"],
            "masked_dummy_instances": infos[0]["padded_total"] - infos[0]["real_instances"],
            "batches_per_rank": [i["batches"] for i in infos],
            "every_token_scored_once": complete,
            "completeness_error": err,
            "rescore_bits_per_token": bits,
            "rescore_rel_diff_vs_reference": abs(bits / ref["bits_per_token"] - 1) if bits else None,
            "max_rel_diff_per_document_vs_reference": max(
                abs(totals[ws][doc][0] / ref["per_document"][doc][0] - 1) for doc in spans
            ),
            "per_document_token_counts_match_reference": all(totals[ws][doc][1] == ref["per_document"][doc][1] for doc in spans),
            "olmo_reported_metrics": next((i["olmo_reported_metrics"] for i in infos if "olmo_reported_metrics" in i), None),
        }
    same_keys = set(rows[1]) == set(rows[2])
    same_counts = same_keys and all(rows[1][k][1] == rows[2][k][1] for k in rows[1])
    res["identical_token_set_across_world_sizes"] = bool(same_counts)
    res["max_rel_diff_per_row_ws1_vs_ws2"] = max(abs(rows[2][k][0] / rows[1][k][0] - 1) for k in rows[1]) if same_keys else None
    res["max_abs_diff_bits_per_token_ws1_vs_ws2"] = (
        abs(res["world_size_1"]["rescore_bits_per_token"] - res["world_size_2"]["rescore_bits_per_token"])
        if res["world_size_1"]["rescore_bits_per_token"] and res["world_size_2"]["rescore_bits_per_token"]
        else None
    )
    Path(result).write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:4])
