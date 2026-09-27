"""Open RiskLens reserved months 6-7 exactly once after the immutable model freeze.

This command is intentionally bound to freeze_id 20260927T115644_227909Z and
selected model SHA-256 65e9caa0.... It creates a durable opening marker *before*
reading reserved features/labels, then evaluates the frozen no-age HistGB without
any retraining, calibration, threshold tuning or model selection.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from risklens_core.baseline import sha256_file
from risklens_core.features import FEATURES
from risklens_core.final_test import (
    CAPACITIES,
    EXPECTED_MODEL_SHA256,
    FINAL_TEST_EXPERIMENT,
    FINAL_TEST_VERSION,
    PRIMARY_CAPACITY,
    RESERVED_MONTHS,
    TIE_SEED,
    capacity_entry,
    final_test_protocol,
    load_reserved_test,
    operational_aggregate,
    ranking_masks,
    score_distribution,
    validate_audit_for_final,
    validate_freeze_manifest,
    verify_dataset_identity,
)
from risklens_core.metrics import evaluate

CSV = Path("data/raw/Base.csv")
AUDIT = Path("reports/data_audit.json")
FREEZE = Path("reports/model_freeze.json")
OPENING_MARKER = Path("reports/final_test_opened.json")
FINAL_REPORT = Path("reports/final_test_summary.json")
ARTIFACT_ROOT = Path("artifacts/final_test")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes() -> dict:
    paths = [
        ROOT / "risklens_core/final_test.py",
        Path(__file__).resolve(),
        ROOT / "risklens_core/metrics.py",
        ROOT / "risklens_core/features.py",
        ROOT / "requirements.txt",
    ]
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def validate_model(model, selected: dict) -> None:
    if tuple(model.named_steps["clean_select"].features) != tuple(selected["remaining_features"]):
        raise ValueError("Frozen artifact feature set differs from model_freeze.json")
    hgb = model.named_steps["model"]
    for key, expected in selected.get("hyperparameters", {}).items():
        if hgb.get_params().get(key) != expected:
            raise ValueError(f"Frozen model parameter changed: {key}")


def main() -> None:
    started = time.perf_counter()
    if FINAL_REPORT.exists():
        raise ValueError(f"Final test has already been completed: {FINAL_REPORT}")
    if OPENING_MARKER.exists():
        raise ValueError(
            f"A final-test opening marker already exists: {OPENING_MARKER}. "
            "Do not rerun or delete it; inspect the previous attempt before any recovery."
        )

    print("[1/7] Verifying the exact immutable freeze and audited dataset identity (holdout still unopened)...", flush=True)
    freeze = read_json(FREEZE)
    selected = validate_freeze_manifest(freeze)
    audit = read_json(AUDIT)
    validate_audit_for_final(audit, freeze)
    verify_dataset_identity(CSV)
    freeze_sha = sha256_file(FREEZE)

    frozen_model_path = Path(selected["frozen_artifact"])
    if not frozen_model_path.is_file():
        raise ValueError(f"Frozen model artifact is missing: {frozen_model_path}")
    if sha256_file(frozen_model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("Frozen model bytes no longer match the immutable SHA-256")
    model = joblib.load(frozen_model_path)
    validate_model(model, selected)

    opening_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = ARTIFACT_ROOT / opening_id
    hashes = source_hashes()
    marker = {
        "status": "opened_incomplete",
        "experiment": FINAL_TEST_EXPERIMENT,
        "version": FINAL_TEST_VERSION,
        "opening_id": opening_id,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze_sha,
        "dataset_sha256": freeze["dataset_sha256"],
        "model_sha256": selected["sha256"],
        "protocol": final_test_protocol(),
        "source_sha256": hashes,
        "note": "This marker is written before reserved rows are loaded. Do not delete it to rerun the holdout.",
    }
    write_json(OPENING_MARKER, marker)

    print("[2/7] FINAL HOLDOUT OPENING RECORDED. Loading reserved months 6-7 for the first and only evaluation...", flush=True)
    test = load_reserved_test(CSV, freeze)

    print("[3/7] Scoring the frozen no-age HistGB artifact without retraining or calibration...", flush=True)
    with threadpool_limits(limits=4):
        scores = model.predict_proba(test[FEATURES])[:, 1]
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Frozen model produced invalid final-test scores")

    print("[4/7] Applying the frozen 1% / 3% / 5% monthly review policies...", flush=True)
    per_month = []
    prediction_parts = []
    for month in RESERVED_MONTHS:
        frame = test.loc[test["month"] == month]
        month_scores = scores[test["month"].to_numpy() == month]
        row_ids = frame.index.to_numpy(dtype=np.int64)
        labels = frame["fraud_bool"].to_numpy(dtype=int)
        metrics = evaluate(labels, month_scores, row_ids, capacities=CAPACITIES, seed=TIE_SEED)
        per_month.append({"month": int(month), "metrics": metrics, "score_distribution": score_distribution(month_scores)})
        masks = ranking_masks(month_scores, row_ids, capacities=CAPACITIES, seed=TIE_SEED)
        part = pd.DataFrame({
            "row_id": row_ids,
            "month": int(month),
            "fraud_bool": labels,
            "model_score": month_scores,
            "selected_at_1pct": masks[0.01],
            "selected_at_3pct": masks[0.03],
            "selected_at_5pct": masks[0.05],
        })
        prediction_parts.append(part)

    month_metrics = [x["metrics"] for x in per_month]
    primary = operational_aggregate(month_metrics, PRIMARY_CAPACITY)

    print("[5/7] Computing secondary pooled holdout metrics and fixed validation generalization deltas...", flush=True)
    labels_all = test["fraud_bool"].to_numpy(dtype=int)
    row_ids_all = test.index.to_numpy(dtype=np.int64)
    pooled = evaluate(labels_all, scores, row_ids_all, capacities=CAPACITIES, seed=TIE_SEED)
    validation = freeze["validation_reproduction"]["metrics"]
    val3 = capacity_entry(validation, PRIMARY_CAPACITY)
    generalization = {
        "note": "Descriptive final-minus-validation deltas only; no post-test selection or tuning is permitted.",
        "recall_at_3pct_delta": float(primary["recall"] - val3["recall"]),
        "precision_at_3pct_delta": float(primary["precision"] - val3["precision"]),
        "average_precision_delta": float(pooled["average_precision"] - validation["average_precision"]),
        "roc_auc_delta": float(pooled["roc_auc"] - validation["roc_auc"]),
        "brier_score_delta": float(pooled["brier_score"] - validation["brier_score"]),
    }

    print("[6/7] Saving immutable final-test artifacts and provenance...", flush=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["month", "row_id"])
    predictions_path = run_dir / "final_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    report = {
        "status": "complete",
        "experiment": FINAL_TEST_EXPERIMENT,
        "version": FINAL_TEST_VERSION,
        "opening_id": opening_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": True,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": freeze_sha,
        "dataset_sha256": freeze["dataset_sha256"],
        "feature_contract": freeze["feature_contract"],
        "selected_model": {
            "name": selected["name"],
            "sha256": selected["sha256"],
            "removed_features": selected["removed_features"],
            "score_policy": selected["score_policy"],
        },
        "protocol": final_test_protocol(),
        "holdout": {
            "months": list(RESERVED_MONTHS),
            "rows": int(len(test)),
            "fraud_rows": int(labels_all.sum()),
            "prevalence": float(labels_all.mean()),
        },
        "primary_operational_result": primary,
        "per_month": per_month,
        "secondary_pooled_metrics": pooled,
        "frozen_validation_metrics": validation,
        "generalization_vs_validation": generalization,
        "outputs": {
            "predictions": str(predictions_path),
            "opening_marker": str(OPENING_MARKER),
        },
        "source_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "processor": platform.processor(),
            "packages": {p: importlib.metadata.version(p) for p in [
                "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"
            ]},
        },
        "limitations": [
            "This is the one-time reserved-month evaluation; results must not be used to change the frozen model, features, calibration, review capacity or tie policy.",
            "The primary 3% capacity is a project operating assumption rather than measured analyst staffing capacity.",
            "Raw scores are model scores and are not asserted to be calibrated fraud probabilities.",
            "The Feedzai BAF benchmark is synthetic; final-holdout performance does not establish production or external validity.",
            "Per-month and pooled results describe months 6-7 of this dataset only and are not guarantees under future drift.",
            "No pass/fail threshold was predeclared; the observed holdout results are reported without post-hoc acceptance criteria.",
        ],
        "total_seconds": float(time.perf_counter() - started),
    }
    write_json(run_dir / "final_test_summary.json", report)
    write_json(FINAL_REPORT, report)

    marker["status"] = "complete"
    marker["completed_at_utc"] = report["completed_at_utc"]
    marker["final_report_sha256"] = sha256_file(FINAL_REPORT)
    marker["run_directory"] = str(run_dir)
    write_json(OPENING_MARKER, marker)

    print("[7/7] FINAL HOLDOUT COMPLETE. The frozen protocol is now permanently test-evaluated.", flush=True)
    print(f"Primary operational Recall@3%: {primary['recall']:.6f}", flush=True)
    print(f"Primary operational Precision@3%: {primary['precision']:.6f}", flush=True)
    print(f"Secondary pooled AP: {pooled['average_precision']:.6f}", flush=True)
    print(f"Saved immutable report: {FINAL_REPORT}", flush=True)
    print("Do not retrain, retune or rerun the holdout. Upload final_test_summary.json for review.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        raise SystemExit(f"Final test stopped: {exc}") from exc
