"""Build temporal drift evidence and a label-safe monitoring snapshot without opening test months 6-7."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from risklens_core.baseline import load_development, sha256_file
from risklens_core.features import CONTRACT_VERSION, FEATURES
from risklens_core.metrics import evaluate
from risklens_core.monitoring import (
    build_alerts,
    feature_drift,
    overall_status,
    score_drift,
    summarize_feature_drift,
    thresholds,
    validate_calibration_reference,
)
from risklens_core.robustness import make_ablation_pipeline
from risklens_core.scoring import (
    artifact_fingerprint,
    resolve_reference_artifacts,
    validate_robustness_reference,
    verify_saved_predictions,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def jsonable_records(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append({k: (None if pd.isna(v) else v) for k, v in row.items()})
    return rows


def _capacity(metrics: dict, fraction: float) -> dict:
    for row in metrics["review_capacity"]:
        if abs(row["requested_fraction"] - fraction) < 1e-12:
            return row
    raise ValueError(f"Metrics do not contain requested capacity {fraction}")


def _prospective_backtest(development: pd.DataFrame, *, capacity: float, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    months = sorted(int(m) for m in development["month"].unique())
    if months != [0, 1, 2, 3, 4, 5]:
        raise ValueError("Prospective backtest expects development months 0-5")

    for target_month in months[1:]:
        history_months = [m for m in months if m < target_month]
        fit_frame = development.loc[development["month"].isin(history_months)]
        score_frame = development.loc[development["month"] == target_month]
        if fit_frame["fraud_bool"].nunique() != 2 or score_frame["fraud_bool"].nunique() != 2:
            raise ValueError("Every prospective fold requires both fraud classes")
        pipe = make_ablation_pipeline("full")
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            pipe.fit(fit_frame[FEATURES], fit_frame["fraud_bool"].to_numpy(dtype=int))
        with threadpool_limits(limits=4):
            scores = pipe.predict_proba(score_frame[FEATURES])[:, 1]
        elapsed = time.perf_counter() - started
        warning_text = [str(w.message) for w in caught]
        if warning_text:
            raise ValueError(f"Prospective fold month {target_month} emitted warnings: {warning_text[:3]}")
        metrics = evaluate(
            score_frame["fraud_bool"].to_numpy(dtype=int),
            scores,
            score_frame.index.to_numpy(dtype=np.int64),
            capacities=(capacity,),
            seed=seed,
        )
        cap = _capacity(metrics, capacity)
        rows.append({
            "target_month": target_month,
            "fit_months": ",".join(str(m) for m in history_months),
            "fit_rows": int(len(fit_frame)),
            "score_rows": int(len(score_frame)),
            "fraud_rows": int(score_frame["fraud_bool"].sum()),
            "prevalence": metrics["prevalence"],
            "average_precision": metrics["average_precision"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
            "reviewed_at_capacity": cap["reviewed"],
            "precision_at_capacity": cap["precision"],
            "recall_at_capacity": cap["recall"],
            "fraud_caught_at_capacity": cap["tp"],
            "cutoff_score": cap["cutoff_score"],
            "fit_seconds": elapsed,
        })
        folds.append({
            "target_month": target_month,
            "fit_months": history_months,
            "fit_rows": int(len(fit_frame)),
            "fit_fraud_rows": int(fit_frame["fraud_bool"].sum()),
            "score_rows": int(len(score_frame)),
            "score_fraud_rows": int(score_frame["fraud_bool"].sum()),
            "fit_seconds": elapsed,
            "warnings": [],
            "metrics": metrics,
            "scores": scores,
            "row_ids": score_frame.index.to_numpy(dtype=np.int64),
        })
        print(
            f"  month {target_month}: train={history_months}, AP={metrics['average_precision']:.5f}, "
            f"recall@{capacity:.0%}={cap['recall']:.5f}, cutoff={cap['cutoff_score']:.5f}",
            flush=True,
        )
    return pd.DataFrame(rows), folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--robustness-summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--calibration-summary", type=Path, default=Path("reports/calibration_uncertainty_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/temporal_monitoring"))
    parser.add_argument("--summary", type=Path, default=Path("reports/temporal_monitoring_summary.json"))
    parser.add_argument("--capacity", type=float, default=0.03)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    if args.bins < 2:
        raise ValueError("bins must be >= 2")

    started = time.perf_counter()
    print("[1/8] Verifying audited development data and experiment lineage...", flush=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_summary.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    train, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_robustness_reference(robustness, fingerprint)
    artifacts = resolve_reference_artifacts(robustness)
    if not artifacts["model"].exists() or not artifacts["predictions"].exists():
        raise ValueError("Missing saved robustness model/predictions")
    model_sha256 = artifact_fingerprint(artifacts["model"])
    validate_calibration_reference(calibration, dataset_sha256=fingerprint, model_sha256=model_sha256)
    development = pd.concat([train, validation], axis=0).sort_index(kind="stable")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)

    print("[2/8] Reproducing frozen HistGB-15 month-5 scores...", flush=True)
    model = joblib.load(artifacts["model"])
    with threadpool_limits(limits=4):
        validation_scores = model.predict_proba(validation[FEATURES])[:, 1]
    reproduction = verify_saved_predictions(
        artifacts["predictions"],
        validation.index.to_numpy(dtype=np.int64),
        validation["fraud_bool"].to_numpy(dtype=int),
        validation_scores,
    )

    print("[3/8] Computing primary train(0-4) -> validation(5) feature drift...", flush=True)
    latest_feature_table = feature_drift(train, validation, bins=args.bins)
    latest_feature_table.to_csv(directory / "feature_drift_train0_4_to_month5.csv", index=False)
    latest_feature_summary = summarize_feature_drift(latest_feature_table, top_n=12)

    print("[4/8] Computing frozen-model score drift and month-by-month score distributions...", flush=True)
    with threadpool_limits(limits=4):
        training_scores = model.predict_proba(train[FEATURES])[:, 1]
    latest_score_drift = score_drift(training_scores, validation_scores, bins=args.bins)

    monthly_score_rows = []
    monthly_scores = {}
    for month in range(6):
        frame = development.loc[development["month"] == month]
        with threadpool_limits(limits=4):
            scores = model.predict_proba(frame[FEATURES])[:, 1]
        monthly_scores[month] = scores
        monthly_score_rows.append({
            "month": month,
            "rows": int(len(frame)),
            "mean_score": float(np.mean(scores)),
            "p50_score": float(np.quantile(scores, 0.50)),
            "p95_score": float(np.quantile(scores, 0.95)),
            "p99_score": float(np.quantile(scores, 0.99)),
            "prevalence": float(frame["fraud_bool"].mean()),
        })
    monthly_score_table = pd.DataFrame(monthly_score_rows)
    monthly_score_table.to_csv(directory / "monthly_score_distribution.csv", index=False)

    print("[5/8] Computing month-over-month feature and score drift through month 5...", flush=True)
    mom_rows = []
    for month in range(1, 6):
        prior = development.loc[development["month"] == month - 1]
        current = development.loc[development["month"] == month]
        table = feature_drift(prior, current, bins=args.bins)
        feature_summary = summarize_feature_drift(table, top_n=5)
        sdrift = score_drift(monthly_scores[month - 1], monthly_scores[month], bins=args.bins)
        mom_rows.append({
            "reference_month": month - 1,
            "target_month": month,
            "feature_status": feature_summary["status"],
            "max_feature_psi": feature_summary["max_psi"],
            "features_at_watch_or_alert": feature_summary["features_at_watch_or_alert"],
            "score_psi": sdrift["psi"],
            "score_ks_statistic": sdrift["ks_statistic"],
            "score_status": sdrift["severity"],
            "overall_status": overall_status(feature_summary, sdrift),
        })
        table.to_csv(directory / f"feature_drift_month{month-1}_to_month{month}.csv", index=False)
    mom_table = pd.DataFrame(mom_rows)
    mom_table.to_csv(directory / "month_over_month_drift.csv", index=False)

    print("[6/8] Running expanding-window prospective backtest for months 1-5...", flush=True)
    backtest_table, folds = _prospective_backtest(development, capacity=args.capacity, seed=args.seed)
    # Month 5 uses exactly months 0-4 and the fixed model configuration, so it should reproduce the saved reference.
    month5_fold = next(fold for fold in folds if fold["target_month"] == 5)
    max_month5_difference = float(np.max(np.abs(month5_fold["scores"] - validation_scores)))
    if max_month5_difference > 1e-12:
        raise ValueError("Prospective month-5 refit does not reproduce the frozen HistGB reference")
    backtest_table.to_csv(directory / "prospective_backtest.csv", index=False)

    print("[7/8] Building label-safe monitoring snapshot and alerts...", flush=True)
    alerts = build_alerts(latest_feature_table, latest_score_drift)
    monitoring_status = overall_status(latest_feature_summary, latest_score_drift)
    monitoring_snapshot = {
        "status": monitoring_status,
        "as_of_month": 5,
        "reference_population": "training months 0-4 pooled",
        "target_population": "validation month 5",
        "model": "histgb_15_leaves/full raw score",
        "feature_drift": latest_feature_summary,
        "score_drift": latest_score_drift,
        "alerts": alerts,
        "thresholds": thresholds(),
        "label_free": True,
        "interpretation": (
            "This snapshot is a monitoring triage artifact. Drift alerts identify distribution change; "
            "they do not establish model failure, fraud causation, or unfairness."
        ),
    }
    write_json(directory / "monitoring_snapshot.json", monitoring_snapshot)

    print("[8/8] Saving provenance and development-only summary...", flush=True)
    backtest_json = []
    for fold in folds:
        backtest_json.append({k: v for k, v in fold.items() if k not in {"scores", "row_ids"}})

    summary = {
        "status": "complete",
        "experiment": "temporal-monitoring-v1",
        "run_id": run_id,
        "run_directory": str(directory),
        "protocol": "temporal-v1",
        "feature_contract": CONTRACT_VERSION,
        "dataset_sha256": fingerprint,
        "robustness_run_id": robustness.get("run_id"),
        "calibration_uncertainty_run_id": calibration.get("run_id"),
        "reference_model": "histgb_15_leaves/full",
        "reference_model_artifact": str(artifacts["model"]),
        "reference_model_sha256": model_sha256,
        "split": {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]},
        "test_evaluated": False,
        "scored_month": 5,
        "month_rows": {str(k): int(v) for k, v in counts.items()},
        "reference_reproduction": reproduction,
        "drift_policy": {
            "alerting_is_label_free": True,
            "primary_reference": "pooled training months 0-4",
            "primary_target": "validation month 5",
            "feature_bins": args.bins,
            "thresholds": thresholds(),
            "note": "PSI/missingness thresholds are operational heuristics, not hypothesis tests.",
        },
        "latest_monitoring_snapshot": monitoring_snapshot,
        "month_over_month": jsonable_records(mom_table),
        "monthly_score_distribution": jsonable_records(monthly_score_table),
        "prospective_backtest": {
            "method": "fixed HistGB-15 configuration; each target month scored only by a model fit on earlier months",
            "capacity": args.capacity,
            "folds": backtest_json,
            "month5_max_absolute_probability_difference_vs_frozen_reference": max_month5_difference,
            "interpretation": (
                "Label-aware development backtest for delayed performance monitoring. It is separate from the "
                "label-free alert status and does not use sealed months 6-7."
            ),
        },
        "outputs": {
            "monitoring_snapshot": str(directory / "monitoring_snapshot.json"),
            "feature_drift_latest": str(directory / "feature_drift_train0_4_to_month5.csv"),
            "month_over_month_drift": str(directory / "month_over_month_drift.csv"),
            "monthly_score_distribution": str(directory / "monthly_score_distribution.csv"),
            "prospective_backtest": str(directory / "prospective_backtest.csv"),
        },
        "source_sha256": {
            "risklens_core/monitoring.py": sha256_file(ROOT / "risklens_core/monitoring.py"),
            "scripts/run_temporal_monitoring.py": sha256_file(ROOT / "scripts/run_temporal_monitoring.py"),
            "risklens_core/robustness.py": sha256_file(ROOT / "risklens_core/robustness.py"),
            "risklens_core/scoring.py": sha256_file(ROOT / "risklens_core/scoring.py"),
            "requirements.txt": sha256_file(ROOT / "requirements.txt"),
        },
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "processor": platform.processor(),
            "native_thread_limit": 4,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
            },
        },
        "limitations": [
            "Months 6-7 remain sealed and are not used for drift, backtesting, threshold choice, or alerting.",
            "PSI and missingness thresholds are engineering triage heuristics, not statistical significance tests or regulatory limits.",
            "The frozen model's score distributions on training months 0-4 are retrospective/in-sample and are used only as a label-free distribution baseline.",
            "Prospective backtest models are fit on progressively larger histories, so performance changes can reflect both population change and additional training data.",
            "Observed drift does not by itself prove degraded model quality, fraud causation, unfairness, or a need to retrain.",
            "Fraud labels may be delayed in a real deployment; label-aware performance monitoring should therefore run on a delayed cadence.",
            "The current 3% review capacity remains a project assumption rather than measured analyst staffing capacity.",
        ],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)

    print(f"  Monitoring status: {monitoring_status}", flush=True)
    print(f"  Feature watch/alerts: {latest_feature_summary['features_at_watch_or_alert']}", flush=True)
    print(f"  Score PSI: {latest_score_drift['psi']:.6f} ({latest_score_drift['severity']})", flush=True)
    print(f"  Saved: {args.summary}", flush=True)
    print("  Test months 6-7 remain sealed.", flush=True)


if __name__ == "__main__":
    main()
