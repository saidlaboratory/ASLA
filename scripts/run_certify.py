"""PREDICTIONS_TASK_CERTIFY.md: Stage 2, the certifying selector against its baselines.

One specification (``certify``) is applied unchanged to DataDecide (45 designs, leave-one-ladder-out calibration)
and to the Marin speedrun (fit 130m-520m, target 1.2b). Per-window fits are cached in ``data/cache/certify``.

Writes ``results/certify/certify.json``.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "certify" / "certify.json"
CACHE = REPO / "data" / "cache" / "certify"
ALPHA = 0.05
STEP = 1e-4
SEED = 20261026
BOOT = 2000
TRUTH_DRAWS = 4000
CHECK_DRAWS = 400
LABELS = ("4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M")  # windows exclude 750M
STARTS, ENDS = ("4M", "14M", "60M"), ("150M", "300M", "530M")
TARGETS = ("1B", "530M")
METRICS = {
    "c4_en_bits_per_token": "datadecide_runs.parquet",
    "olmes_macro_error": "datadecide_runs_olmes_macro_error.parquet",
    "olmes_macro_correct_prob_per_char_deficit": "datadecide_runs_olmes_correct_prob_per_char.parquet",
}
FORMS = ("loglog", "three")
METHODS = ("selector", "single_scale", "projection", "seed_se", "bootstrap", "conformal")


# --- forms -------------------------------------------------------------------------------------


def design_rng(name: str) -> np.random.Generator:
    """An independent generator per design, so results do not depend on run order or on what is cached."""

    import zlib

    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def fit_loglog(x: np.ndarray, y: np.ndarray, start: Any = None) -> np.ndarray:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return np.array([intercept, slope])


def eval_loglog(p: np.ndarray, x: float) -> float:
    return float(np.exp(p[0] + p[1] * np.log(x)))


def fit_three(x: np.ndarray, y: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """a x^-b + c in (log a, b, c); b in [1e-6, 2], c in [0, 0.999 min y]; minimum SSE over the fixed starts."""

    lx = np.log(x)
    lo, hi = [-60.0, 1e-6, 0.0], [80.0, 2.0, 0.999 * y.min()]
    starts = [start] if start is not None else []
    if start is None:
        for frac in (0.0, 0.5, 0.9):
            c0 = frac * y.min()
            slope, intercept = np.polyfit(lx, np.log(y - c0), 1)
            starts.append(np.array([intercept, float(np.clip(-slope, 0.01, 2.0)), c0]))
    best = None
    for s0 in starts:
        q = np.clip(s0, lo, hi)
        r = optimize.least_squares(
            lambda p: np.exp(p[0] - p[1] * lx) + p[2] - y,
            q,
            bounds=(lo, hi),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=20000,
        )
        sse = float(np.sum(r.fun**2))
        if best is None or sse < best[0]:
            best = (sse, r.x)
    assert best is not None
    return best[1]


def eval_three(p: np.ndarray, x: float) -> float:
    return float(np.exp(p[0] - p[1] * np.log(x)) + p[2])


FIT: dict[str, Callable[..., np.ndarray]] = {"loglog": fit_loglog, "three": fit_three}
EVAL: dict[str, Callable[[np.ndarray, float], float]] = {"loglog": eval_loglog, "three": eval_three}


@dataclass
class Fitted:
    """A candidate's fit on a window: base parameters and the parameters after perturbing each point."""

    x: np.ndarray
    y: np.ndarray
    base: np.ndarray
    perturbed: list[np.ndarray]

    def forecast(self, form: str, xt: float) -> tuple[float, np.ndarray]:
        f0 = EVAL[form](self.base, xt)
        grad = np.array([(EVAL[form](p, xt) - f0) / (STEP * self.y[r]) for r, p in enumerate(self.perturbed)])
        return f0, grad


def fit_candidate(form: str, x: np.ndarray, y: np.ndarray) -> Fitted:
    base = FIT[form](x, y)
    pert = []
    for r in range(len(y)):
        yp = y.copy()
        yp[r] *= 1 + STEP
        pert.append(FIT[form](x, yp, base) if form == "three" else FIT[form](x, yp))
    return Fitted(x, y, base, pert)


# --- the selector and its baselines --------------------------------------------------------------


def certify(fc: dict[str, np.ndarray], var: dict[str, np.ndarray]) -> np.ndarray:
    """CERTIFY. fc[f], var[f]: forecasts and variances (misfit included) per candidate. Returns C[i, j] = 1 iff
    'i before j' (i lower) is certified under every form."""

    n = len(next(iter(fc.values())))
    cert = np.ones((n, n), dtype=bool)
    np.fill_diagonal(cert, False)
    z = stats.norm.ppf(1 - ALPHA)
    for f in fc:
        d = fc[f][None, :] - fc[f][:, None]  # yhat_j - yhat_i
        sd = np.sqrt(var[f][:, None] + var[f][None, :])
        cert &= (d > 0) & (d >= z * sd)
    return cert


def possibly_best(cert: np.ndarray) -> np.ndarray:
    return np.asarray(~cert.any(axis=0))  # not certified worse than anyone


def deciding(score: np.ndarray) -> np.ndarray:
    d = score[None, :] - score[:, None]
    return d > 0


def calibrate_kappa(preds: list[dict[str, Any]]) -> float:
    num = den = 0.0
    for p in preds:
        n = len(p["r"])
        rc = p["r"] - p["r"].mean()
        num += float(np.sum(n / (n - 1) * rc**2 - p["u"]))
        den += float(np.sum(p["L"] ** 2))
    return float(np.sqrt(max(0.0, num / den))) if den > 0 else 0.0


def conformal_q(preds: list[dict[str, Any]]) -> float | None:
    scores = np.concatenate([np.abs(p["r"]) / p["L"] for p in preds]) if preds else np.array([])
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return float(np.sort(scores)[k - 1]) if 0 < k <= n else None


# --- evaluation ------------------------------------------------------------------------------------


def evaluate(cert: np.ndarray, m: np.ndarray, se: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = len(m)
    pairs = n * (n - 1) // 2
    charge = stats.norm.cdf((m[:, None] - m[None, :]) / np.sqrt(se[:, None] ** 2 + se[None, :] ** 2))
    c = cert & ~np.eye(n, dtype=bool)
    k = int(c.sum())
    charges = float((charge * c).sum())
    pb = possibly_best(c)
    draws = m[None, :] + se[None, :] * rng.standard_normal((TRUTH_DRAWS, n))
    best = np.argmin(draws, axis=1)
    return {
        "certified": k,
        "pairs": pairs,
        "coverage": k / pairs,
        "charges": charges,
        "fcr": charges / k if k else None,
        "possibly_best_size": int(pb.sum()),
        "contains_observed_best": bool(pb[int(np.argmin(m))]),
        "p_true_best_in_set": float(np.mean(pb[best])),
        "_kernel": (charge - ALPHA) * c,
    }


def run_design(
    name: str,
    fits: dict[str, dict[str, Fitted]],
    s_fit: np.ndarray,
    xt: np.ndarray,
    y_top: np.ndarray,
    m: np.ndarray,
    se: np.ndarray,
    kappa: dict[str, float],
    q: float | None,
    boot_delta: np.ndarray | None,
    rng: np.random.Generator,
    lever: np.ndarray,
) -> dict[str, Any]:
    cands = list(fits["three"])
    fc, grads = {}, {}
    for f in FORMS:
        out = [fits[f][c].forecast(f, xt[i]) for i, c in enumerate(cands)]
        fc[f] = np.array([o[0] for o in out])
        grads[f] = np.array([o[1] for o in out])
    seed_var = {f: np.sum(grads[f] ** 2 * s_fit**2, axis=1) for f in FORMS}
    var_sel = {f: seed_var[f] + (kappa[f] * lever * fc[f]) ** 2 for f in FORMS}
    certs: dict[str, np.ndarray] = {
        "selector": certify(fc, var_sel),
        "single_scale": deciding(y_top),
        "projection": deciding(fc["three"]),
        "seed_se": certify(fc, seed_var),
    }
    if boot_delta is not None:  # (B, n, R) resampled-mean deviations
        sims = fc["three"][None, :] + np.einsum("bnr,nr->bn", boot_delta, grads["three"])
        frac = (sims[:, None, :] - sims[:, :, None] > 0).mean(axis=0)
        certs["bootstrap"] = frac >= 1 - ALPHA
    else:  # parametric draws at the seed noise (Marin: no seeds)
        z = rng.standard_normal((BOOT, *s_fit.shape)) * s_fit[None]
        sims = fc["three"][None, :] + np.einsum("bnr,nr->bn", z, grads["three"])
        certs["bootstrap"] = (sims[:, None, :] - sims[:, :, None] > 0).mean(axis=0) >= 1 - ALPHA
    if q is not None:
        lo, hi = fc["three"] * (1 - q * lever), fc["three"] * (1 + q * lever)
        certs["conformal"] = hi[:, None] < lo[None, :]
    else:
        certs["conformal"] = np.zeros_like(certs["selector"])
    res = {k: evaluate(v, m, se, rng) for k, v in certs.items()}
    return {
        "design": name,
        "lever_arm_median": float(np.median(lever)),
        "kappa": kappa,
        "conformal_q": q,
        "methods": res,
        "_forecasts": fc,
        "_var": var_sel,
        "_seed_var": seed_var,
    }


# --- DataDecide ------------------------------------------------------------------------------------


def dd_tables(metric: str) -> dict[str, Any]:
    from asla.analysis.moderation import fit_prior, moderate

    df = pd.read_parquet(REPO / "data" / METRICS[metric])
    recipes = sorted(df["intervention"].unique())
    scales = sorted(df["scale_label"].unique(), key=lambda s: df[df.scale_label == s]["compute"].iloc[0])
    compute = {s: float(df[df.scale_label == s]["compute"].iloc[0]) for s in scales}
    g = df.groupby(["intervention", "scale_label"])["bpb"]
    mean, var, cnt = g.mean().unstack(), g.var(ddof=1).unstack(), g.count().unstack()
    se = pd.DataFrame(index=mean.index, columns=mean.columns, dtype=float)
    for s in scales:
        prior = fit_prior(var[s].tolist(), int(cnt[s].min()) - 1)
        se[s] = np.sqrt(moderate(var[s].to_numpy(), prior) / cnt[s].to_numpy())
    seeds = {(r, s): grp["bpb"].to_numpy() for (r, s), grp in df.groupby(["intervention", "scale_label"])}
    return {
        "recipes": recipes,
        "scales": scales,
        "compute": compute,
        "mean": mean.loc[recipes],
        "se": se.loc[recipes],
        "seeds": seeds,
    }


def windows() -> dict[str, list[str]]:
    return {f"{a}-{b}": list(LABELS[LABELS.index(a) : LABELS.index(b) + 1]) for a in STARTS for b in ENDS}


def dd_fits(metric: str, t: dict[str, Any]) -> dict[str, dict[str, dict[str, Fitted]]]:
    path = CACHE / f"fits_{metric}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out: dict[str, dict[str, dict[str, Fitted]]] = {}
    for w, scales in windows().items():
        x = np.array([t["compute"][s] for s in scales])
        out[w] = {f: {r: fit_candidate(f, x, t["mean"].loc[r, scales].to_numpy(float)) for r in t["recipes"]} for f in FORMS}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def dd_calibration(t: dict[str, Any], fits: dict, exclude: str, c_target: float) -> dict[str, list[dict[str, Any]]]:
    preds: dict[str, list[dict[str, Any]]] = {f: [] for f in FORMS}
    for w, scales in windows().items():
        if w == exclude:
            continue
        top = t["compute"][scales[-1]]
        s_fit = t["se"].loc[t["recipes"], scales].to_numpy(float)
        for h in t["scales"]:
            ch = t["compute"][h]
            if not top < ch < c_target:
                continue
            yh = t["mean"].loc[t["recipes"], h].to_numpy(float)
            sh = t["se"].loc[t["recipes"], h].to_numpy(float)
            for f in FORMS:
                out = [fits[w][f][r].forecast(f, ch) for r in t["recipes"]]
                fc = np.array([o[0] for o in out])
                g = np.array([o[1] for o in out])
                u = (np.sum(g**2 * s_fit**2, axis=1) + sh**2) / yh**2
                L = np.full(len(yh), np.log10(ch / top))
                preds[f].append({"r": (fc - yh) / yh, "u": u, "L": L, "window": w, "rung": h})
    return preds


def boot_deltas(t: dict[str, Any], scales: list[str], rng: np.random.Generator) -> np.ndarray:
    out = np.empty((BOOT, len(t["recipes"]), len(scales)))
    for i, r in enumerate(t["recipes"]):
        for k, s in enumerate(scales):
            v = t["seeds"][(r, s)]
            idx = rng.integers(0, len(v), size=(BOOT, len(v)))
            out[:, i, k] = v[idx].mean(axis=1) - v.mean()
    return out


def run_datadecide() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in METRICS:
        t = dd_tables(metric)
        fits = dd_fits(metric, t)
        for w, scales in windows().items():
            for target in TARGETS:
                ct, top = t["compute"][target], t["compute"][scales[-1]]
                if ct <= top:
                    continue
                name = f"{metric}|{w}|{target}"
                ck = CACHE / "designs" / f"{name.replace('|', '__')}.pkl"
                if ck.exists():
                    out[name] = pickle.loads(ck.read_bytes())
                    continue
                rng = design_rng(name)
                preds = dd_calibration(t, fits, w, ct)
                kappa = {f: calibrate_kappa(preds[f]) for f in FORMS}
                q = conformal_q(preds["three"])
                n = len(t["recipes"])
                res = run_design(
                    name,
                    fits[w],
                    t["se"].loc[t["recipes"], scales].to_numpy(float),
                    np.full(n, ct),
                    t["mean"].loc[t["recipes"], scales[-1]].to_numpy(float),
                    t["mean"].loc[t["recipes"], target].to_numpy(float),
                    t["se"].loc[t["recipes"], target].to_numpy(float),
                    kappa,
                    q,
                    boot_deltas(t, scales, rng),
                    rng,
                    np.full(n, np.log10(ct / top)),
                )
                if w == "4M-150M" and target == "1B":
                    res["linearization_check"] = linearization_check(
                        fits[w],
                        t["se"].loc[t["recipes"], scales].to_numpy(float),
                        np.full(n, ct),
                        np.full(n, np.log10(ct / top)),
                        kappa,
                        res["_forecasts"],
                        res["_var"],
                        rng,
                    )
                res.update(
                    {
                        "metric": metric,
                        "window": w,
                        "target": target,
                        "lever_arm": float(np.log10(ct / top)),
                        "calibration_predictions": {f: len(preds[f]) for f in FORMS},
                        "candidates": t["recipes"],
                    }
                )
                ck.parent.mkdir(parents=True, exist_ok=True)
                ck.write_bytes(pickle.dumps(res))
                out[name] = res
    return out


# --- Marin -----------------------------------------------------------------------------------------


def run_marin(dd_kappa: dict[str, float], low_end: bool = False) -> dict[str, Any]:
    from scripts import run_leaderboard_resolution as lr

    rng = design_rng(f"marin|{'low' if low_end else 'bound'}")
    nb = json.loads((REPO / "results" / "external" / "noise_bracket.json").read_text())
    sig = nb["e_reruns"]["sigma_by_rung"]
    if low_end:
        sig = {k: nb["c_datadecide"]["seed_sd_rel_by_size"]["1B"] for k in sig}
    _, cells = lr.listing()
    subs = sorted({k[0] for k in cells})
    fit_r = ("130m", "300m", "520m")
    x = {s: np.array([cells[(s, r)]["x"] for r in fit_r]) for s in subs}
    y = {s: np.array([cells[(s, r)]["y"] for r in fit_r]) for s in subs}
    fits = {f: {s: fit_candidate(f, x[s], y[s]) for s in subs} for f in FORMS}
    s_fit = np.array([[sig[r] * cells[(s, r)]["y"] for r in fit_r] for s in subs])
    # calibration: fit {130m, 300m}, held-out 520m. The three-parameter form is not identified on two points:
    # its kappa is transferred from DataDecide C4 (the pre-registered sensitivity), a deviation reported as such.
    cal = {s: fit_candidate("loglog", x[s][:2], y[s][:2]) for s in subs}
    out_c = [cal[s].forecast("loglog", cells[(s, "520m")]["x"]) for s in subs]
    fc_c = np.array([o[0] for o in out_c])
    g_c = np.array([o[1] for o in out_c])
    yh = np.array([cells[(s, "520m")]["y"] for s in subs])
    sh = np.array([sig["520m"] * v for v in yh])
    s_cal = s_fit[:, :2]
    L_c = np.array([np.log10(cells[(s, "520m")]["x"] / cells[(s, "300m")]["x"]) for s in subs])
    pred = {"r": (fc_c - yh) / yh, "u": (np.sum(g_c**2 * s_cal**2, axis=1) + sh**2) / yh**2, "L": L_c}
    kappa = {"loglog": calibrate_kappa([pred]), "three": dd_kappa["three"]}
    q = conformal_q([pred])
    xt = np.array([cells[(s, "1_2b")]["x"] for s in subs])
    lever = np.log10(xt / np.array([cells[(s, "520m")]["x"] for s in subs]))
    m = np.array([cells[(s, "1_2b")]["y"] for s in subs])
    se = sig["1_2b"] * m
    res = run_design(
        "marin",
        fits,
        s_fit,
        xt,
        y_top=np.array([cells[(s, "520m")]["y"] for s in subs]),
        m=m,
        se=se,
        kappa=kappa,
        q=q,
        boot_delta=None,
        rng=rng,
        lever=lever,
    )
    res["linearization_check"] = linearization_check(fits, s_fit, xt, lever, kappa, res["_forecasts"], res["_var"], rng)
    res.update(
        {
            "candidates": subs,
            "noise": "low end (DataDecide 1B)" if low_end else "Task-1 bound",
            "deviations": [
                "three-parameter kappa transferred from DataDecide C4 (not identified on two points)",
                "conformal calibrated on log-log residuals at 520m (three-parameter not identified)",
                "bootstrap: parametric draws at the seed noise (no seeds)",
            ],
            "kappa_transfer_variant": {"loglog": dd_kappa["loglog"], "three": dd_kappa["three"]},
        }
    )
    return res


def linearization_check(
    fits: dict[str, dict[str, Fitted]],
    s_fit: np.ndarray,
    xt: np.ndarray,
    lever: np.ndarray,
    kappa: dict[str, float],
    fc: dict[str, np.ndarray],
    var: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Pre-registered check: the Gaussian-limit decisions against CHECK_DRAWS full-refit draws."""

    cands = list(fits["three"])
    n = len(cands)
    frac_ok = np.ones((n, n), dtype=bool)
    for f in FORMS:
        sims = np.empty((CHECK_DRAWS, n))
        for i, c in enumerate(cands):
            fit = fits[f][c]
            for d in range(CHECK_DRAWS):
                y = fit.y + s_fit[i] * rng.standard_normal(len(fit.y))
                y = np.maximum(y, 1e-12)
                p = FIT[f](fit.x, y, fit.base) if f == "three" else FIT[f](fit.x, y)
                sims[d, i] = EVAL[f](p, xt[i])
        sims += (kappa[f] * lever * fc[f])[None, :] * rng.standard_normal((CHECK_DRAWS, n))
        frac = (sims[:, None, :] - sims[:, :, None] > 0).mean(axis=0)
        point = fc[f][None, :] - fc[f][:, None] > 0
        frac_ok &= point & (frac >= 1 - ALPHA)
    np.fill_diagonal(frac_ok, False)
    gauss = certify(fc, var)
    either = gauss | frac_ok
    return {
        "draws": CHECK_DRAWS,
        "pairs": n * (n - 1) // 2,
        "certified_gaussian": int(gauss.sum()),
        "certified_draws": int(frac_ok.sum()),
        "agree_all_ordered_pairs": float(np.mean((gauss == frac_ok)[~np.eye(n, dtype=bool)])),
        "agree_among_certified_by_either": float(np.mean((gauss == frac_ok)[either])) if either.any() else None,
    }


# --- pooling and the criterion -----------------------------------------------------------------------


def pool(designs: list[dict[str, Any]], crit: float) -> dict[str, Any]:
    from asla.analysis.target_scoring import u_statistic_test

    out = {}
    for meth in METHODS:
        rows = [d["methods"][meth] for d in designs]
        cert = sum(r["certified"] for r in rows)
        charges = sum(r["charges"] for r in rows)
        kern = np.mean([r["_kernel"] for r in rows], axis=0)
        cands = designs[0]["candidates"]
        pairs = {(cands[i], cands[j]): float(kern[i, j] + kern[j, i]) for i, j in combinations(range(len(cands)), 2)}
        test = u_statistic_test(pairs)
        out[meth] = {
            "certified": cert,
            "pairs": sum(r["pairs"] for r in rows),
            "coverage": float(np.mean([r["coverage"] for r in rows])),
            "fcr": charges / cert if cert else None,
            "excess_over_alpha_kernel": {
                "mean": float(test["mean"]),
                "se": float(test["standard_error"]),
                "upper_at_calibrated_critical": float(test["mean"] + crit * test["standard_error"]),
            },
            "possibly_best_size_mean": float(np.mean([r["possibly_best_size"] for r in rows])),
            "p_true_best_in_set_mean": float(np.mean([r["p_true_best_in_set"] for r in rows])),
        }
    return out


def by_lever(designs: list[dict[str, Any]]) -> dict[str, Any]:
    bins = {"L<0.5": (0, 0.5), "0.5<=L<1": (0.5, 1.0), "L>=1": (1.0, 99.0)}
    out = {}
    for b, (lo, hi) in bins.items():
        ds = [d for d in designs if lo <= d["lever_arm"] < hi]
        out[b] = {
            "designs": len(ds),
            **{
                m: {
                    "coverage": float(np.mean([d["methods"][m]["coverage"] for d in ds])) if ds else None,
                    "fcr": (
                        sum(d["methods"][m]["charges"] for d in ds) / max(1, sum(d["methods"][m]["certified"] for d in ds))
                    )
                    if ds and sum(d["methods"][m]["certified"] for d in ds)
                    else None,
                }
                for m in METHODS
            },
        }
    return out


def criterion(pooled: dict[str, Any]) -> dict[str, Any]:
    sel = pooled["selector"]
    s1 = sel["fcr"] is not None and sel["fcr"] <= ALPHA
    nominal = [m for m in METHODS if m != "selector" and pooled[m]["fcr"] is not None and pooled[m]["fcr"] <= ALPHA]
    s2 = all(sel["coverage"] > pooled[m]["coverage"] for m in nominal)
    return {
        "S1_fcr_at_or_below_alpha": s1,
        "S2_higher_coverage_than_nominal_baselines": s2,
        "baselines_meeting_nominal": nominal,
        "confirmed": bool(s1 and s2),
        "F3_degenerate_coverage_below_10pct": sel["coverage"] < 0.10,
    }


def strip(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, list):
        return [strip(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


def main() -> None:
    crit = float(
        json.loads((REPO / "results" / "target_scoring" / "calibration.json").read_text())["calibrated"]["random_u"][
            "critical_value"
        ]
    )
    dd = run_datadecide()
    designs = list(dd.values())
    per_metric = {m: pool([d for d in designs if d["metric"] == m], crit) for m in METRICS}
    pooled = pool(designs, crit)
    c4_cal = [d for d in designs if d["metric"] == "c4_en_bits_per_token"]
    dd_kappa = {f: float(np.median([d["kappa"][f] for d in c4_cal])) for f in FORMS}
    marin = run_marin(dd_kappa)
    marin_low = run_marin(dd_kappa, low_end=True)
    out: dict[str, Any] = {
        "preregistration": "PREDICTIONS_TASK_CERTIFY.md",
        "alpha": ALPHA,
        "calibrated_critical_value": crit,
        "datadecide": {
            "designs": [strip(d) for d in designs],
            "pooled": pooled,
            "per_metric": per_metric,
            "by_lever_arm": by_lever(designs),
            "criterion": criterion(pooled),
            "per_metric_criterion": {m: criterion(v) for m, v in per_metric.items()},
        },
        "marin": {"bound": strip(marin), "low_end": strip(marin_low)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strip(out), indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(criterion(pooled), indent=1))


if __name__ == "__main__":
    main()
