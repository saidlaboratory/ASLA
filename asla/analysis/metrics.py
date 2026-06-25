"""Decision-level metrics for projected rankings."""

from __future__ import annotations

import itertools
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _ordered_index(series: pd.Series) -> list[str]:
    frame = pd.DataFrame({"name": series.index.astype(str), "value": series.to_numpy(dtype=float)})
    frame = frame.sort_values(["value", "name"], kind="mergesort")
    return frame["name"].tolist()


def _ranks(series: pd.Series, names: list[str]) -> dict[str, int]:
    ordered = _ordered_index(series.loc[names])
    return {name: i for i, name in enumerate(ordered)}


def decision_metrics(projected: pd.Series, truth: pd.Series, k: int) -> Dict[str, float]:
    """Compute decision metrics from projected and true BPB rankings.

    Lower BPB is better. Ties are broken deterministically by intervention name
    in ascending lexical order.

    Examples:
        ``projected=[A: 1.0, B: 2.0]`` and ``truth=[A: 1.5, B: 2.5]`` gives
        ``top1_acc=1`` and ``regret=0``.

        ``projected=[B: 1.0, A: 2.0]`` and ``truth=[A: 1.5, B: 2.0]`` gives
        ``top1_acc=0`` and ``regret=0.5``.
    """

    common = sorted(set(projected.index.astype(str)) & set(truth.index.astype(str)))
    if not common:
        raise ValueError("projected and truth rankings have no interventions in common")
    projected = projected.rename(index=str).loc[common]
    truth = truth.rename(index=str).loc[common]
    p_order = _ordered_index(projected)
    t_order = _ordered_index(truth)
    kk = min(k, len(common))
    p_top = p_order[:kk]
    t_top = t_order[:kk]
    top1_acc = float(p_order[0] == t_order[0])
    topk_recall = float(len(set(p_top) & set(t_top)) / kk)

    p_rank = _ranks(projected, common)
    t_rank = _ranks(truth, common)
    pair_total = 0
    pair_correct = 0
    for a, b in itertools.combinations(common, 2):
        pair_total += 1
        pair_correct += int((p_rank[a] < p_rank[b]) == (t_rank[a] < t_rank[b]))
    pairwise_acc = float(pair_correct / pair_total) if pair_total else 1.0

    p_rank_arr = np.asarray([p_rank[name] for name in common], dtype=float)
    t_rank_arr = np.asarray([t_rank[name] for name in common], dtype=float)
    kendall = kendalltau(p_rank_arr, t_rank_arr).statistic
    spear = spearmanr(p_rank_arr, t_rank_arr).statistic
    regret = float(truth.loc[p_order[0]] - truth.min())
    return {
        "top1_acc": top1_acc,
        "topk_recall": topk_recall,
        "pairwise_acc": pairwise_acc,
        "kendall_tau": float(0.0 if np.isnan(kendall) else kendall),
        "spearman": float(0.0 if np.isnan(spear) else spear),
        "regret": regret,
    }

