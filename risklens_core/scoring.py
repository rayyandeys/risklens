"""Reference validation and model-artifact checks for review-workflow-v1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import sha256_file
from .features import CONTRACT_VERSION, FEATURES

EXPECTED_SPLIT = {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]}


def validate_robustness_reference(summary, dataset_fingerprint):
    if summary.get("status") != "complete" or summary.get("test_evaluated") is not False:
        raise ValueError("Robustness reference must be a completed development-only report")
    if summary.get("dataset_sha256") != dataset_fingerprint:
        raise ValueError("Robustness reference uses a different dataset fingerprint")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("split") != EXPECTED_SPLIT:
        raise ValueError("Robustness reference uses a different feature contract or temporal split")
    if summary.get("reference_model") != "histgb_15_leaves":
        raise ValueError("Review workflow expects histgb_15_leaves as the development reference")

    full = summary.get("ablations", {}).get("full")
    if not full or full.get("removed_features") != [] or full.get("remaining_features") != FEATURES:
        raise ValueError("Robustness reference lacks the canonical full-feature model")
    if full.get("warnings"):
        raise ValueError("Reference full-feature model has unresolved training warnings")

    comparison = summary.get("bootstrap", {}).get("histgb_15_vs_logistic_unweighted", {})
    for metric in ("recall_delta", "average_precision_delta"):
        interval = comparison.get(metric, {})
        if interval.get("ci_lower") is None or interval["ci_lower"] <= 0:
            raise ValueError(f"Reference bootstrap does not support a positive {metric} development delta")
    if comparison.get("capacity") != 0.03:
        raise ValueError("Reference bootstrap capacity differs from the workflow's 3% assumption")


def resolve_reference_artifacts(summary):
    run_dir = Path(summary["run_directory"])
    return {
        "model": run_dir / "full.joblib",
        "predictions": run_dir / "validation_predictions.csv",
    }


def verify_saved_predictions(saved_path, row_ids, labels, fresh_scores, *, atol=1e-12):
    saved = pd.read_csv(saved_path)
    required = {"row_id", "fraud_bool", "full"}
    if not required.issubset(saved.columns):
        raise ValueError("Saved robustness predictions lack required columns")
    current = pd.DataFrame({"row_id": np.asarray(row_ids, dtype=np.int64),
                            "fraud_bool_current": np.asarray(labels, dtype=int),
                            "fresh_score": np.asarray(fresh_scores, dtype=float)})
    aligned = current.merge(saved[["row_id", "fraud_bool", "full"]], on="row_id",
                            how="inner", validate="one_to_one")
    if len(aligned) != len(current):
        raise ValueError("Saved predictions do not cover the current validation population")
    if not np.array_equal(aligned["fraud_bool_current"].to_numpy(), aligned["fraud_bool"].to_numpy()):
        raise ValueError("Saved predictions have different validation labels")
    if not np.allclose(aligned["fresh_score"].to_numpy(), aligned["full"].to_numpy(), rtol=0, atol=atol):
        raise ValueError("Loaded model predictions do not reproduce the saved robustness predictions")
    return {
        "rows": int(len(aligned)),
        "max_absolute_probability_difference": float(
            np.max(np.abs(aligned["fresh_score"].to_numpy() - aligned["full"].to_numpy()))
        ),
    }


def artifact_fingerprint(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Missing model artifact: {path}")
    return sha256_file(path)
