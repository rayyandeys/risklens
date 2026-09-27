"""Targeted development-only subgroup and error analysis for RiskLens governance review."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .features import FeatureCleaner, NUMERIC


def top_capacity_mask(scores, row_ids, *, capacity=0.03, seed=42):
    """Return the deterministic global top-k review mask used by RiskLens metrics."""
    scores = np.asarray(scores, dtype=float)
    row_ids = np.asarray(row_ids)
    if scores.ndim != 1 or row_ids.shape != scores.shape or len(scores) == 0:
        raise ValueError("scores and row_ids must be aligned nonempty vectors")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("scores must be finite and within [0, 1]")
    if not np.issubdtype(row_ids.dtype, np.integer) or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique integers")
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")

    by_id = np.argsort(row_ids, kind="stable")
    tie_key = np.empty(len(scores), dtype=np.int64)
    tie_key[by_id] = np.random.default_rng(seed).permutation(len(scores))
    order = np.lexsort((tie_key, -scores))
    k = math.floor(len(scores) * capacity)
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:k]] = True
    return selected


def age_groups(values):
    """Coarse predeclared age grouping used only for descriptive validation analysis."""
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    labels = np.full(len(values), "missing", dtype=object)
    present = values.notna().to_numpy()
    arr = values.to_numpy(dtype=float)
    labels[present & (arr < 50)] = "age_lt_50"
    labels[present & (arr >= 50)] = "age_ge_50"
    return labels


def intended_balance_groups(frame: pd.DataFrame):
    cleaned = FeatureCleaner().transform(frame)
    missing = cleaned["intended_balcon_amount__missing"].to_numpy(dtype=float) == 1.0
    return np.where(missing, "intended_balance_missing", "intended_balance_observed")


def subgroup_metrics(y, selected, groups, *, scores=None):
    """Describe one global queue policy within each subgroup; do not retune per group."""
    y = np.asarray(y, dtype=int)
    selected = np.asarray(selected, dtype=bool)
    groups = np.asarray(groups, dtype=object)
    if not (y.shape == selected.shape == groups.shape) or len(y) == 0:
        raise ValueError("y, selected and groups must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must be binary")
    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != y.shape:
            raise ValueError("scores must align with y")

    result = []
    for name in sorted(pd.unique(groups).tolist()):
        mask = groups == name
        yg = y[mask]
        sg = selected[mask]
        rows = int(mask.sum())
        positives = int(yg.sum())
        negatives = rows - positives
        reviewed = int(sg.sum())
        tp = int((sg & (yg == 1)).sum())
        fp = int((sg & (yg == 0)).sum())
        fn = positives - tp
        tn = negatives - fp
        entry = {
            "group": str(name),
            "rows": rows,
            "fraud_rows": positives,
            "prevalence": positives / rows if rows else None,
            "reviewed": reviewed,
            "selection_rate": reviewed / rows if rows else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / reviewed if reviewed else None,
            "recall": tp / positives if positives else None,
            "false_positive_rate": fp / negatives if negatives else None,
        }
        if scores is not None:
            entry["mean_score"] = float(scores[mask].mean()) if rows else None
            entry["median_score"] = float(np.median(scores[mask])) if rows else None
        result.append(entry)
    return result


def error_partition(y, selected):
    y = np.asarray(y, dtype=int)
    selected = np.asarray(selected, dtype=bool)
    if y.shape != selected.shape:
        raise ValueError("y and selected must align")
    return np.select(
        [selected & (y == 1), selected & (y == 0), (~selected) & (y == 1)],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )


def error_summary(y, scores, selected):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if not (y.shape == scores.shape == selected.shape):
        raise ValueError("error-summary inputs must align")
    labels = error_partition(y, selected)
    rows = []
    for name in ("true_positive", "false_positive", "false_negative", "true_negative"):
        mask = labels == name
        values = scores[mask]
        rows.append({
            "error_class": name,
            "rows": int(mask.sum()),
            "score_mean": float(values.mean()) if len(values) else None,
            "score_p10": float(np.quantile(values, 0.10)) if len(values) else None,
            "score_p50": float(np.quantile(values, 0.50)) if len(values) else None,
            "score_p90": float(np.quantile(values, 0.90)) if len(values) else None,
        })
    return rows


def _standardized_mean_difference(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if var <= 0:
        return 0.0 if float(a.mean()) == float(b.mean()) else None
    return float((a.mean() - b.mean()) / math.sqrt(var))


def numeric_error_contrasts(frame: pd.DataFrame, y, selected, *, top_n=10):
    """Rank descriptive numeric shifts for FN-vs-TP and FP-vs-TN error classes."""
    cleaned = FeatureCleaner().transform(frame)
    labels = error_partition(y, selected)
    outputs = {}
    comparisons = {
        "false_negative_vs_true_positive": ("false_negative", "true_positive"),
        "false_positive_vs_true_negative": ("false_positive", "true_negative"),
    }
    for key, (left, right) in comparisons.items():
        rows = []
        for feature in NUMERIC:
            values = pd.to_numeric(cleaned[feature], errors="coerce").to_numpy(dtype=float)
            delta = _standardized_mean_difference(values[labels == left], values[labels == right])
            if delta is None:
                continue
            rows.append({"feature": feature, "standardized_mean_difference": delta,
                         "absolute_standardized_mean_difference": abs(delta)})
        rows.sort(key=lambda item: (-item["absolute_standardized_mean_difference"], item["feature"]))
        outputs[key] = rows[:top_n]
    return outputs


def compact_group_deltas(candidate_rows: Iterable[dict], reference_rows: Iterable[dict]):
    candidate = {r["group"]: r for r in candidate_rows}
    reference = {r["group"]: r for r in reference_rows}
    if set(candidate) != set(reference):
        raise ValueError("candidate/reference group names differ")
    out = []
    for group in sorted(candidate):
        c, r = candidate[group], reference[group]
        out.append({
            "group": group,
            "selection_rate_delta": c["selection_rate"] - r["selection_rate"],
            "recall_delta": None if c["recall"] is None or r["recall"] is None else c["recall"] - r["recall"],
            "false_positive_rate_delta": None if c["false_positive_rate"] is None or r["false_positive_rate"] is None else c["false_positive_rate"] - r["false_positive_rate"],
        })
    return out
