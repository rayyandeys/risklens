"""Capacity metrics with a hard floor budget and reproducible label-free ties."""
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def evaluate(y, scores, row_ids, capacities=(0.01, 0.03, 0.05), seed=42):
    y, scores, row_ids = np.asarray(y), np.asarray(scores), np.asarray(row_ids)
    if y.ndim != 1 or scores.shape != y.shape or row_ids.shape != y.shape or not len(y):
        raise ValueError("Labels, probabilities and IDs must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Evaluation requires both binary target classes")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.issubdtype(row_ids.dtype, np.integer) or len(np.unique(row_ids)) != len(y):
        raise ValueError("Row IDs must be unique integers")
    # Seeded permutation assigned in ID order, independent of input order and labels.
    by_id = np.argsort(row_ids, kind="stable")
    tie_key = np.empty(len(y), dtype=np.int64)
    tie_key[by_id] = np.random.default_rng(seed).permutation(len(y))
    order = np.lexsort((tie_key, -scores))
    positives = int(y.sum())
    result = {"rows": len(y), "fraud_rows": positives, "prevalence": float(y.mean()),
              "average_precision": float(average_precision_score(y, scores)),
              "roc_auc": float(roc_auc_score(y, scores)),
              "brier_score": float(brier_score_loss(y, scores)), "review_capacity": []}
    for capacity in capacities:
        if not 0 < capacity <= 1:
            raise ValueError("Capacity must be in (0, 1]")
        k = math.floor(len(y) * capacity)
        tp = int(y[order[:k]].sum())
        fp, fn = k - tp, positives - tp
        result["review_capacity"].append({
            "requested_fraction": capacity, "reviewed": k, "actual_fraction": k / len(y),
            "tp": tp, "fp": fp, "fn": fn, "tn": len(y) - positives - fp,
            "precision": tp / k if k else None, "recall": tp / positives,
            "random_expected_recall": k / len(y),
            "random_expected_precision": float(y.mean()) if k else None,
            "random_expected_tp": k * float(y.mean()),
            "cutoff_score": float(scores[order[k - 1]]) if k else None,
            "cutoff_tie_size": int((scores == scores[order[k - 1]]).sum()) if k else 0,
        })
    return result
