"""Bounded histogram-gradient-boosted comparison, experiment boosted-v1."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from .features import CONTRACT_VERSION, FEATURES, NUMERIC, INDICATORS, CATEGORICAL, FeatureCleaner

CONFIGURATIONS = {"histgb_15_leaves": 15, "histgb_31_leaves": 31}


def make_boosted_pipeline(max_leaf_nodes=15, max_iter=200):
    # Codes are category IDs. The explicit categorical mask prevents treating them as ordered numbers.
    categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                 encoded_missing_value=np.nan, dtype=np.float64)
    encoder = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("missing_flags", "passthrough", INDICATORS),
        ("categorical", categorical, CATEGORICAL)], sparse_threshold=0)
    mask = [False] * (len(NUMERIC) + len(INDICATORS)) + [True] * len(CATEGORICAL)
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.08, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=100, l2_regularization=1.0, max_bins=255,
        categorical_features=mask, early_stopping=False, class_weight=None, random_state=42)
    return Pipeline([("clean", FeatureCleaner()), ("encode", encoder), ("model", model)])


def validate_reference(summary, fingerprint):
    if summary.get("status") != "complete" or summary.get("test_evaluated") is not False:
        raise ValueError("Reference must be a completed development-only baseline report")
    if summary.get("dataset_sha256") != fingerprint:
        raise ValueError("Reference baseline uses a different dataset fingerprint")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("features") != FEATURES:
        raise ValueError("Reference feature contract does not match")
    if summary.get("split") != {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]}:
        raise ValueError("Reference temporal split does not match")
    for name in ("logistic_unweighted", "logistic_balanced"):
        if summary.get("models", {}).get(name, {}).get("converged") is not True:
            raise ValueError(f"Reference {name} must have converged")


def comparison_row(name, metrics):
    row = {"model": name, "average_precision": metrics["average_precision"],
           "roc_auc": metrics["roc_auc"], "brier_score": metrics["brier_score"]}
    for capacity in metrics["review_capacity"]:
        pct = round(capacity["requested_fraction"] * 100)
        row[f"recall_at_{pct}pct"] = capacity["recall"]
        row[f"precision_at_{pct}pct"] = capacity["precision"]
        row[f"fraud_caught_at_{pct}pct"] = capacity["tp"]
    return row
