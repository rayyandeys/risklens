"""Read-only product monitoring projection for the RiskLens analyst console.

The dashboard is deliberately assembled from immutable/generated experiment reports.
It does not score rows, retrain models, open labels, or mutate workflow state.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baseline import sha256_file


class MonitoringDataError(ValueError):
    """Raised when the report chain is missing or internally inconsistent."""


REQUIRED_REPORTS = {
    "calibration": "calibration_uncertainty_summary.json",
    "temporal": "temporal_monitoring_summary.json",
    "freeze": "model_freeze.json",
    "final": "final_test_summary.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MonitoringDataError(f"Required monitoring report is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitoringDataError(f"Could not read monitoring report: {path.name}") from exc
    if not isinstance(value, dict):
        raise MonitoringDataError(f"Monitoring report must be a JSON object: {path.name}")
    return value


def _capacity(metrics: dict[str, Any], fraction: float = 0.03) -> dict[str, Any]:
    for entry in metrics.get("review_capacity", []):
        if abs(float(entry.get("requested_fraction", -1)) - fraction) < 1e-12:
            return entry
    raise MonitoringDataError(f"Missing {fraction:.0%} review-capacity entry")


def _validate_chain(reports_dir: Path, reports: dict[str, dict[str, Any]]) -> None:
    calibration = reports["calibration"]
    temporal = reports["temporal"]
    freeze = reports["freeze"]
    final = reports["final"]

    if calibration.get("status") != "complete" or calibration.get("experiment") != "calibration-uncertainty-v1":
        raise MonitoringDataError("Calibration report is not the completed calibration-uncertainty-v1 run")
    if temporal.get("status") != "complete" or temporal.get("experiment") != "temporal-monitoring-v1":
        raise MonitoringDataError("Temporal report is not the completed temporal-monitoring-v1 run")
    if freeze.get("status") != "frozen" or int(freeze.get("freeze_version", 0)) != 1:
        raise MonitoringDataError("Model freeze manifest is not frozen v1")
    if final.get("status") != "complete" or final.get("experiment") != "final-holdout-v1":
        raise MonitoringDataError("Final report is not the completed final-holdout-v1 run")
    if final.get("test_evaluated") is not True:
        raise MonitoringDataError("Final holdout report does not record completed evaluation")

    dataset_ids = {
        calibration.get("dataset_sha256"), temporal.get("dataset_sha256"),
        freeze.get("dataset_sha256"), final.get("dataset_sha256"),
    }
    if None in dataset_ids or len(dataset_ids) != 1:
        raise MonitoringDataError("Monitoring reports do not share one dataset fingerprint")
    if len({calibration.get("feature_contract"), temporal.get("feature_contract"),
            freeze.get("feature_contract"), final.get("feature_contract")}) != 1:
        raise MonitoringDataError("Monitoring reports do not share one feature contract")

    if final.get("freeze_id") != freeze.get("freeze_id"):
        raise MonitoringDataError("Final holdout is not linked to the active freeze")
    expected_freeze_hash = final.get("freeze_manifest_sha256")
    actual_freeze_hash = sha256_file(reports_dir / REQUIRED_REPORTS["freeze"])
    if expected_freeze_hash != actual_freeze_hash:
        raise MonitoringDataError("Frozen manifest bytes differ from the manifest used for the final holdout")

    frozen_model = freeze.get("selected_model", {})
    tested_model = final.get("selected_model", {})
    if frozen_model.get("name") != tested_model.get("name") or frozen_model.get("sha256") != tested_model.get("sha256"):
        raise MonitoringDataError("Final holdout model does not match the frozen model")


def _drift_projection(temporal: dict[str, Any]) -> dict[str, Any]:
    snapshot = temporal.get("latest_monitoring_snapshot", {})
    feature = snapshot.get("feature_drift", {})
    score = snapshot.get("score_drift", {})
    top_features = []
    for row in feature.get("top_features", [])[:12]:
        top_features.append({
            "feature": row.get("feature"),
            "kind": row.get("kind"),
            "psi": row.get("psi"),
            "ks_statistic": row.get("ks_statistic"),
            "reference_missing_rate": row.get("reference_missing_rate"),
            "target_missing_rate": row.get("target_missing_rate"),
            "missing_rate_delta": row.get("missing_rate_delta"),
            "unseen_category_rate": row.get("unseen_category_rate"),
            "severity": row.get("severity"),
        })
    month_over_month = [{
        "reference_month": row.get("reference_month"),
        "target_month": row.get("target_month"),
        "feature_status": row.get("feature_status"),
        "max_feature_psi": row.get("max_feature_psi"),
        "features_at_watch_or_alert": row.get("features_at_watch_or_alert"),
        "score_psi": row.get("score_psi"),
        "score_ks_statistic": row.get("score_ks_statistic"),
        "score_status": row.get("score_status"),
        "overall_status": row.get("overall_status"),
    } for row in temporal.get("month_over_month", [])]
    return {
        "status": snapshot.get("status"),
        "as_of_month": snapshot.get("as_of_month"),
        "reference_population": snapshot.get("reference_population"),
        "target_population": snapshot.get("target_population"),
        "feature": {
            "status": feature.get("status"),
            "counts_by_severity": feature.get("counts_by_severity", {}),
            "max_psi": feature.get("max_psi"),
            "features_at_watch_or_alert": feature.get("features_at_watch_or_alert"),
            "top_features": top_features,
        },
        "score": {
            "reference_model": temporal.get("reference_model"),
            "psi": score.get("psi"),
            "ks_statistic": score.get("ks_statistic"),
            "reference_mean": score.get("reference_mean"),
            "target_mean": score.get("target_mean"),
            "reference_p95": score.get("reference_p95"),
            "target_p95": score.get("target_p95"),
            "severity": score.get("severity"),
            "note": (
                "Score-drift values come from the pre-freeze full HistGB development reference. "
                "Feature drift is model-agnostic; frozen-model holdout performance is reported separately."
            ),
        },
        "month_over_month": month_over_month,
        "alert_count": len(snapshot.get("alerts", [])),
        "threshold_note": snapshot.get("thresholds", {}).get("note"),
        "interpretation": snapshot.get("interpretation"),
    }


def _stability_projection(calibration: dict[str, Any]) -> dict[str, Any]:
    summary = calibration.get("stability", {}).get("summary", {})
    queue = summary.get("queue_jaccard_vs_reference", {})
    retained = summary.get("reference_queue_retention", {})
    rank = summary.get("spearman_score_rank_correlation", {})
    methods = calibration.get("calibration", {}).get("methods", {})
    return {
        "source_model": calibration.get("reference_model"),
        "repeats": summary.get("repeats"),
        "capacity": summary.get("capacity"),
        "queue_jaccard_mean": queue.get("mean"),
        "queue_jaccard_min": queue.get("min"),
        "queue_jaccard_max": queue.get("max"),
        "reference_retention_mean": retained.get("mean"),
        "rank_correlation_mean": rank.get("mean"),
        "boundary_unstable_cases": summary.get("boundary_unstable_cases"),
        "selected_calibration": calibration.get("calibration", {}).get("selected_method"),
        "raw_brier": methods.get("raw", {}).get("brier_score"),
        "raw_ece": methods.get("raw", {}).get("ece"),
        "note": "Development sensitivity study from the pre-freeze full HistGB reference; not a confidence interval.",
    }


def _performance_projection(freeze: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    validation = freeze.get("validation_reproduction", {}).get("metrics", {})
    validation_3 = _capacity(validation, 0.03)
    primary = final.get("primary_operational_result", {})
    pooled = final.get("secondary_pooled_metrics", {})
    points = [{
        "month": 5,
        "split": "validation",
        "rows": validation.get("rows"),
        "fraud_rows": validation.get("fraud_rows"),
        "average_precision": validation.get("average_precision"),
        "roc_auc": validation.get("roc_auc"),
        "brier_score": validation.get("brier_score"),
        "recall_at_3pct": validation_3.get("recall"),
        "precision_at_3pct": validation_3.get("precision"),
        "fraud_caught_at_3pct": validation_3.get("tp"),
        "reviewed_at_3pct": validation_3.get("reviewed"),
    }]
    for month in final.get("per_month", []):
        metrics = month.get("metrics", {})
        cap = _capacity(metrics, 0.03)
        points.append({
            "month": month.get("month"),
            "split": "final_holdout",
            "rows": metrics.get("rows"),
            "fraud_rows": metrics.get("fraud_rows"),
            "average_precision": metrics.get("average_precision"),
            "roc_auc": metrics.get("roc_auc"),
            "brier_score": metrics.get("brier_score"),
            "recall_at_3pct": cap.get("recall"),
            "precision_at_3pct": cap.get("precision"),
            "fraud_caught_at_3pct": cap.get("tp"),
            "reviewed_at_3pct": cap.get("reviewed"),
        })
    return {
        "timeline": points,
        "primary_holdout": {
            "rows": primary.get("rows"),
            "fraud_rows": primary.get("fraud_rows"),
            "reviewed": primary.get("reviewed"),
            "fraud_caught": primary.get("tp"),
            "recall_at_3pct": primary.get("recall"),
            "precision_at_3pct": primary.get("precision"),
            "random_expected_tp": primary.get("random_expected_tp"),
        },
        "pooled_holdout": {
            "average_precision": pooled.get("average_precision"),
            "roc_auc": pooled.get("roc_auc"),
            "brier_score": pooled.get("brier_score"),
        },
        "generalization_delta": final.get("generalization_vs_validation", {}),
        "protocol_note": final.get("protocol", {}).get("selection"),
    }


def load_monitoring_overview(reports_dir: str | Path) -> dict[str, Any]:
    """Return the authenticated dashboard projection from the immutable report chain."""
    root = Path(reports_dir)
    reports = {key: _read_json(root / filename) for key, filename in REQUIRED_REPORTS.items()}
    _validate_chain(root, reports)
    freeze = reports["freeze"]
    final = reports["final"]
    selected = freeze.get("selected_model", {})
    score_policy = selected.get("score_policy", {})
    operating = freeze.get("operating_policy", {})

    return {
        "status": reports["temporal"].get("latest_monitoring_snapshot", {}).get("status"),
        "dataset_sha256": freeze.get("dataset_sha256"),
        "feature_contract": freeze.get("feature_contract"),
        "governance": {
            "state": "frozen_and_tested",
            "freeze_id": freeze.get("freeze_id"),
            "opening_id": final.get("opening_id"),
            "model_name": selected.get("name"),
            "model_sha256": selected.get("sha256"),
            "removed_features": selected.get("removed_features", []),
            "score_policy": score_policy.get("policy"),
            "calibration": score_policy.get("calibration"),
            "score_semantics": score_policy.get("display_semantics"),
            "review_capacity": operating.get("requested_capacity"),
            "tie_seed": operating.get("tie_seed"),
            "test_evaluated": final.get("test_evaluated"),
            "decision_basis": freeze.get("decision", {}).get("basis"),
            "caveat": freeze.get("decision", {}).get("caveat"),
        },
        "drift": _drift_projection(reports["temporal"]),
        "stability": _stability_projection(reports["calibration"]),
        "performance": _performance_projection(freeze, final),
        "limitations": list(dict.fromkeys(
            reports["temporal"].get("limitations", [])
            + reports["final"].get("limitations", [])
        )),
    }
