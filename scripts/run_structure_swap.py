"""Task 2 of PREDICTIONS_TASK_TRANSFER.md: which structural component carries OLMES's projection win?

C4 and OLMES worlds at the primary design (300M -> 530M) with one component at
a time taken from the other metric: (i) noise profile shape, (ii) distribution
of true gaps at the target, (iii) curve shape and floor, (iv) misspecification
loading pattern, and (v) noise amplitude (added, not on the given list). Plus
each base with all five swapped, as a completeness check. 150 worlds per
configuration, common random numbers within a base.

Writes ``results/target_scoring/structure_swap.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import transfer_worlds as tw  # noqa: E402

OUT = REPO / "results" / "target_scoring" / "structure_swap.json"
COMPONENTS = ("i", "ii", "iii", "iv", "v")
NAMES = {
    "i": "noise profile shape",
    "ii": "true-gap distribution at target",
    "iii": "curve shape and floor",
    "iv": "misspecification loading pattern",
    "v": "noise amplitude (added)",
    "iv_common": "residual common mode (exploratory)",
    "iv_specific": "residual recipe-specific part (exploratory)",
}
N_WORLDS = 150
N_BOOT = 2000
SEED = 20260928


def configurations() -> dict[str, tuple[str, dict[str, str]]]:
    out: dict[str, tuple[str, dict[str, str]]] = {"c4": ("c4", {}), "olmes": ("olmes", {})}
    for base, other in (("c4", "olmes"), ("olmes", "c4")):
        for comp in COMPONENTS:
            out[f"{base}+{comp}"] = (base, {comp: other})
        out[f"{base}+all"] = (base, dict.fromkeys(COMPONENTS, other))
        # Exploratory, not pre-registered: (iv) split into the residual's common mode and recipe-specific part.
        out[f"{base}+iv_common"] = (base, {"iv_common": other})
        out[f"{base}+iv_specific"] = (base, {"iv_specific": other})
    return out


def run(name: str, index: int) -> float:
    base, sources = configurations()[name]
    truth, sigma = tw.hybrid(base, sources)
    rng = np.random.default_rng([SEED, ("c4", "olmes").index(base), index])
    ctx = tw.context(base)
    return tw.world_excess(truth, tw.sigma_dict(sigma), ctx, {"primary": tw.PRIMARY}, rng)["primary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    configs = configurations()
    jobs = [(name, i) for name in configs for i in range(N_WORLDS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        values = list(pool.map(run, *zip(*jobs), chunksize=25))
    by_name: dict[str, np.ndarray] = {name: np.empty(N_WORLDS) for name in configs}
    for (name, i), v in zip(jobs, values):
        by_name[name][i] = v
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, N_WORLDS, size=(N_BOOT, N_WORLDS))
    e_c4, e_olmes = float(by_name["c4"].mean()), float(by_name["olmes"].mean())
    rows: dict[str, Any] = {}
    for name, (base, sources) in configs.items():
        x = by_name[name]
        lo, hi = (float(v) for v in np.percentile(np.nanmean(x[boot], axis=1), [2.5, 97.5]))
        mean = float(np.nanmean(x))
        start, other = (e_c4, e_olmes) if base == "c4" else (e_olmes, e_c4)
        truth, sigma = tw.hybrid(base, sources)
        rows[name] = {
            "base": base,
            "swapped": {k: NAMES[k] for k in sources},
            "excess_pp": mean,
            "ci": [lo, hi],
            "sign": "projection wins" if hi < 0 else ("projection loses" if lo > 0 else "unresolved"),
            "share_of_gap_closed": (mean - start) / (other - start) if other != start else None,
            "worlds_with_undefined_excess": int(np.isnan(x).sum()),
            "physicality": tw.physical(truth, sigma, base),
        }

    def moves_sign(name: str) -> bool:
        base = configs[name][0]
        return rows[name]["sign"] == ("projection wins" if base == "c4" else "projection loses")

    identified = sorted(
        {
            comp
            for comp in COMPONENTS
            for base in ("c4", "olmes")
            if moves_sign(f"{base}+{comp}") or (rows[f"{base}+{comp}"]["share_of_gap_closed"] or 0) >= 0.5
        }
    )
    share = {c: max(rows[f"c4+{c}"]["share_of_gap_closed"], rows[f"olmes+{c}"]["share_of_gap_closed"]) for c in COMPONENTS}
    four = ("i", "ii", "iii", "iv")
    ranked = sorted(four, key=lambda c: -share[c])
    out = {
        "design": "primary 300M -> 530M; hybrids in the base metric's units (median true gap at 530M as the unit)",
        "units": {
            m: {"g": tw.context(m)["g"], "noise_amplitude_in_g": tw.noise_profile(tw.context(m))[1]} for m in ("c4", "olmes")
        },
        "noise_profiles": {m: tw.noise_profile(tw.context(m))[0].round(4).to_dict() for m in ("c4", "olmes")},
        "configurations": rows,
        "largest_share_by_component": share,
        "ranking_of_i_to_iv": ranked,
        "identified_for_task_3": identified,
    }
    out["predictions"] = {
        "P2_1_i_primary": moves_sign("c4+i") and moves_sign("olmes+i") and ranked[0] == "i",
        "P2_2_ii_secondary": ranked[1] == "ii",
        "refutation_neither_i_nor_ii_moves_sign": not any(
            moves_sign(f"{b}+{c}") for b in ("c4", "olmes") for c in ("i", "ii")
        ),
        "P2_3_iii_or_v_largest_and_i_below_half": max(share, key=share.get) in ("iii", "v")
        and all(rows[f"{b}+i"]["share_of_gap_closed"] < 0.5 for b in ("c4", "olmes")),
        "P2_4_all_swapped_reproduces": all(
            rows[f"{b}+all"]["ci"][0] <= (e_olmes if b == "c4" else e_c4) <= rows[f"{b}+all"]["ci"][1]
            for b in ("c4", "olmes")
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for name, r in rows.items():
        print(
            f"{name:10s} {r['excess_pp']:+6.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] {r['sign']:17s} "
            f"closed {r['share_of_gap_closed'] if r['share_of_gap_closed'] is None else round(r['share_of_gap_closed'], 2)} "
            f"phys {r['physicality']['tolerant_monotone']}/{r['physicality']['in_range']}"
        )
    print(out["predictions"], "identified", identified)


if __name__ == "__main__":
    main()
