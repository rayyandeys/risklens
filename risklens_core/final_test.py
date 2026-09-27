"""One-time final holdout protocol for frozen RiskLens model.

This module is intentionally specific to the accepted 27 September 2026 freeze.
It validates the immutable freeze before any reserved-month labels are opened and
provides deterministic monthly review-queue aggregation for the final report.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import sha256_file
from .features import CONTRACT_VERSION, FEATURES, SCHEMA
from .metrics import evaluate

FINAL_TEST_VERSION = 1
FINAL_TEST_EXPERIMENT = "final-holdout-v1"
EXPECTED_FREEZE_ID = "20260927T115644_227909Z"
EXPECTED_DATASET_SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"
EXPECTED_MODEL_SHA256 = "65e9caa079ba46af72955a024b7ae0fd7d207e80c10eba686954ff149eb5ee12"
EXPECTED_MODEL_NAME = "histgb_15_leaves/no_customer_age"
EXPECTED_REMOVED_FEATURES = ["customer_age"]
RESERVED_MONTHS = (6, 7)
CAPACITIES = (0.01, 0.03, 0.05)
PRIMARY_CAPACITY = 0.03
TIE_SEED = 42


def capacity_entry(metrics: dict, fraction: float = PRIMARY_CAPACITY) -> dict:
    for entry in metrics.get("review_capacity", []):
        if abs(float(entry.get("requested_fraction", -1)) - fraction) < 1e-12:
            return entry
    raise ValueError(f"Missing review-capacity entry for {fraction}")


def validate_freeze_manifest(manifest: dict) -> dict:
    """Require the exact reviewed freeze and return its selected-model record."""
    if manifest.get("status") != "frozen" or manifest.get("freeze_version") != 1:
        raise ValueError("Expected an immutable model-freeze v1 manifest")
    if manifest.get("freeze_id") != EXPECTED_FREEZE_ID:
        raise ValueError("Final-test package is bound to a different freeze_id")
    if manifest.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("Frozen dataset fingerprint differs from the approved final-test package")
    if manifest.get("feature_contract") != CONTRACT_VERSION:
        raise ValueError("Frozen feature contract differs from the approved contract")
    if manifest.get("test_evaluated") is not False:
        raise ValueError("Freeze manifest already indicates final-test access")

    split = manifest.get("development_split", {})
    if split != {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]}:
        raise ValueError("Frozen temporal split differs from the approved protocol")

    selected = manifest.get("selected_model", {})
    if selected.get("name") != EXPECTED_MODEL_NAME:
        raise ValueError("Unexpected frozen model name")
    if selected.get("sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("Unexpected frozen model SHA-256")
    if selected.get("removed_features") != EXPECTED_REMOVED_FEATURES:
        raise ValueError("Frozen model must remove customer_age exactly")
    expected_remaining = [c for c in FEATURES if c != "customer_age"]
    if selected.get("remaining_features") != expected_remaining:
        raise ValueError("Frozen remaining-feature list differs from the approved no-age contract")
    policy = selected.get("score_policy", {})
    if policy.get("policy") != "raw_positive_class_predict_proba" or policy.get("calibration") != "none":
        raise ValueError("Final test requires the frozen raw uncalibrated score policy")

    operating = manifest.get("operating_policy", {})
    if operating.get("requested_capacity") != PRIMARY_CAPACITY:
        raise ValueError("Frozen review capacity differs from 3%")
    if operating.get("tie_seed") != TIE_SEED:
        raise ValueError("Frozen tie seed differs from 42")
    if operating.get("budget_rule") != "floor(number_of_scored_rows * requested_capacity)":
        raise ValueError("Frozen budget rule differs from the reviewed floor rule")

    final_policy = manifest.get("final_test_policy", {})
    if final_policy.get("reserved_months") != list(RESERVED_MONTHS):
        raise ValueError("Frozen reserved months differ from 6-7")
    if final_policy.get("single_opening_after_freeze") is not True:
        raise ValueError("Freeze does not require a single final-test opening")
    if final_policy.get("no_post_test_model_or_threshold_selection") is not True:
        raise ValueError("Freeze does not forbid post-test model/threshold selection")
    return selected


def validate_audit_for_final(audit: dict, manifest: dict) -> None:
    if audit.get("sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("Audit dataset fingerprint differs from frozen dataset")
    if audit.get("columns") != SCHEMA:
        raise ValueError("Audit schema differs from the canonical 32-column contract")
    expected_counts = {str(k): int(v) for k, v in manifest.get("month_rows", {}).items()}
    actual_counts = {str(k): int(v) for k, v in audit.get("month_rows", {}).items()}
    if actual_counts != expected_counts:
        raise ValueError("Audit month counts differ from the frozen manifest")


def verify_dataset_identity(csv_path: Path) -> None:
    if sha256_file(csv_path) != EXPECTED_DATASET_SHA256:
        raise ValueError("Base.csv SHA-256 differs from the frozen dataset; do not open holdout")
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if header != SCHEMA:
        raise ValueError("Base.csv header differs from the canonical ordered schema")


def load_reserved_test(csv_path: Path, manifest: dict) -> pd.DataFrame:
    """Open the reserved rows only. Call only after the opening marker has been written."""
    pieces = []
    counts = {m: 0 for m in RESERVED_MONTHS}
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        if not chunk["month"].isin(range(8)).all():
            raise ValueError("Invalid month encountered while opening reserved holdout")
        reserved = chunk.loc[chunk["month"].isin(RESERVED_MONTHS)].copy()
        for month, count in reserved["month"].value_counts().items():
            counts[int(month)] += int(count)
        if not reserved.empty:
            pieces.append(reserved)
    if not pieces:
        raise ValueError("Reserved months 6-7 are absent")
    expected = {m: int(manifest["month_rows"][str(m)]) for m in RESERVED_MONTHS}
    if counts != expected:
        raise ValueError(f"Reserved month counts differ from freeze: observed={counts}, expected={expected}")
    frame = pd.concat(pieces).sort_index()
    if not frame["fraud_bool"].isin([0, 1]).all():
        raise ValueError("Final-test labels must be binary")
    for month in RESERVED_MONTHS:
        labels = frame.loc[frame["month"] == month, "fraud_bool"]
        if labels.nunique() != 2:
            raise ValueError(f"Reserved month {month} does not contain both target classes")
    return frame


def ranking_masks(scores, row_ids, capacities=CAPACITIES, seed=TIE_SEED) -> dict[float, np.ndarray]:
    scores = np.asarray(scores, dtype=float)
    row_ids = np.asarray(row_ids)
    if scores.ndim != 1 or row_ids.shape != scores.shape or not len(scores):
        raise ValueError("scores and row_ids must be aligned nonempty vectors")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("scores must be finite within [0,1]")
    if not np.issubdtype(row_ids.dtype, np.integer) or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_ids must be unique integers")
    by_id = np.argsort(row_ids, kind="stable")
    tie_key = np.empty(len(row_ids), dtype=np.int64)
    tie_key[by_id] = np.random.default_rng(seed).permutation(len(row_ids))
    order = np.lexsort((tie_key, -scores))
    result: dict[float, np.ndarray] = {}
    for fraction in capacities:
        k = math.floor(len(scores) * fraction)
        mask = np.zeros(len(scores), dtype=bool)
        mask[order[:k]] = True
        result[float(fraction)] = mask
    return result


def operational_aggregate(month_metrics: list[dict], capacity: float = PRIMARY_CAPACITY) -> dict:
    if not month_metrics:
        raise ValueError("Need at least one monthly metric record")
    entries = [capacity_entry(m, capacity) for m in month_metrics]
    rows = sum(int(m["rows"]) for m in month_metrics)
    fraud_rows = sum(int(m["fraud_rows"]) for m in month_metrics)
    reviewed = sum(int(e["reviewed"]) for e in entries)
    tp = sum(int(e["tp"]) for e in entries)
    fp = sum(int(e["fp"]) for e in entries)
    fn = sum(int(e["fn"]) for e in entries)
    tn = sum(int(e["tn"]) for e in entries)
    return {
        "policy": "independent monthly queues; each month receives floor(month_rows * 0.03) review slots",
        "requested_fraction": float(capacity),
        "rows": rows,
        "fraud_rows": fraud_rows,
        "reviewed": reviewed,
        "actual_fraction": reviewed / rows,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / reviewed if reviewed else None,
        "recall": tp / fraud_rows if fraud_rows else None,
        "random_expected_recall": reviewed / rows,
        "random_expected_precision": fraud_rows / rows,
        "random_expected_tp": reviewed * fraud_rows / rows,
    }


def score_distribution(scores) -> dict:
    scores = np.asarray(scores, dtype=float)
    return {
        "mean": float(np.mean(scores)),
        "p50": float(np.quantile(scores, 0.50)),
        "p95": float(np.quantile(scores, 0.95)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }


def final_test_protocol() -> dict:
    return {
        "version": FINAL_TEST_VERSION,
        "experiment": FINAL_TEST_EXPERIMENT,
        "reserved_months": list(RESERVED_MONTHS),
        "primary_operational_metric": "Recall@3% aggregated across independent month-6 and month-7 review queues",
        "monthly_queue_policy": "Each reserved month is ranked independently using raw frozen model score and frozen tie rule",
        "capacities_reported": list(CAPACITIES),
        "pooled_metrics": "AP, ROC-AUC, Brier and pooled capacity metrics are secondary descriptive holdout summaries",
        "selection": "No model, feature, calibration, capacity or threshold selection is permitted from final-test results",
        "pass_fail_threshold": "none predeclared; final results are reported as observed generalization evidence",
    }
