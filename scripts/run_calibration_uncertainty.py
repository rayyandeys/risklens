"""Run calibration-v1 and training-sample stability analysis without opening test months 6-7."""
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
from risklens_core.calibration_uncertainty import (
    calibration_metrics,
    fit_score_calibrators,
    month_label_bootstrap_positions,
    queue_overlap,
    select_calibration_method,
    summarize_stability,
)
from risklens_core.features import CONTRACT_VERSION, FEATURES
from risklens_core.robustness import make_ablation_pipeline
from risklens_core.scoring import (
    artifact_fingerprint,
    resolve_reference_artifacts,
    validate_robustness_reference,
    verify_saved_predictions,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _temporal_oof_scores(train, *, min_history_months=2):
    months = sorted(int(m) for m in train["month"].unique())
    if months != [0, 1, 2, 3, 4]:
        raise ValueError("Calibration expects development training months 0-4")
    if min_history_months < 1 or min_history_months >= len(months):
        raise ValueError("min_history_months must leave at least one calibration target month")

    pieces = []
    folds = []
    for target_month in months[min_history_months:]:
        history_months = [m for m in months if m < target_month]
        fit_frame = train.loc[train["month"].isin(history_months)]
        score_frame = train.loc[train["month"] == target_month]
        if fit_frame["fraud_bool"].nunique() != 2 or score_frame["fraud_bool"].nunique() != 2:
            raise ValueError("Each temporal calibration fold requires both target classes")
        pipe = make_ablation_pipeline("full")
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            pipe.fit(fit_frame[FEATURES], fit_frame["fraud_bool"].to_numpy(dtype=int))
        with threadpool_limits(limits=4):
            scores = pipe.predict_proba(score_frame[FEATURES])[:, 1]
        fit_seconds = time.perf_counter() - started
        pieces.append(pd.DataFrame({
            "row_id": score_frame.index.to_numpy(dtype="int64"),
            "month": target_month,
            "fraud_bool": score_frame["fraud_bool"].to_numpy(dtype=int),
            "oof_score": scores,
        }))
        folds.append({
            "target_month": target_month,
            "fit_months": history_months,
            "fit_rows": int(len(fit_frame)),
            "fit_fraud_rows": int(fit_frame["fraud_bool"].sum()),
            "score_rows": int(len(score_frame)),
            "score_fraud_rows": int(score_frame["fraud_bool"].sum()),
            "fit_seconds": fit_seconds,
            "warnings": [str(w.message) for w in caught],
        })
    oof = pd.concat(pieces, ignore_index=True)
    if oof["row_id"].duplicated().any():
        raise ValueError("Temporal OOF rows must be unique")
    return oof, folds


def _same_reference_metrics(current, reference):
    for key in ("average_precision", "roc_auc", "brier_score"):
        if abs(current[key] - reference[key]) > 1e-12:
            raise ValueError(f"Fresh reference scoring does not reproduce robustness {key}")
    for old, new in zip(reference["review_capacity"], current["review_capacity"]):
        if old["requested_fraction"] != new["requested_fraction"] or old["tp"] != new["tp"]:
            raise ValueError("Fresh reference scoring does not reproduce robustness capacity metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--robustness-summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/calibration_uncertainty"))
    parser.add_argument("--summary", type=Path, default=Path("reports/calibration_uncertainty_summary.json"))
    parser.add_argument("--stability-repeats", type=int, default=12)
    parser.add_argument("--capacity", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()

    if args.stability_repeats < 2:
        raise ValueError("stability-repeats must be at least 2")
    if not 0 < args.capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    if args.calibration_bins < 2:
        raise ValueError("calibration-bins must be at least 2")

    started = time.perf_counter()
    print("[1/7] Verifying audited development data and frozen robustness reference...", flush=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_summary.read_text(encoding="utf-8"))
    train, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_robustness_reference(robustness, fingerprint)
    artifacts = resolve_reference_artifacts(robustness)
    if not artifacts["model"].exists() or not artifacts["predictions"].exists():
        raise ValueError("Missing saved robustness model/predictions required for exact reproduction")
    model_sha256 = artifact_fingerprint(artifacts["model"])

    print("[2/7] Reproducing the full HistGB-15 month-5 reference scores...", flush=True)
    reference_model = joblib.load(artifacts["model"])
    with threadpool_limits(limits=4):
        raw_scores = reference_model.predict_proba(validation[FEATURES])[:, 1]
    y_val = validation["fraud_bool"].to_numpy(dtype=int)
    row_ids = validation.index.to_numpy(dtype="int64")
    reproduction = verify_saved_predictions(artifacts["predictions"], row_ids, y_val, raw_scores)
    raw_report = calibration_metrics(y_val, raw_scores, row_ids, bins=args.calibration_bins)
    _same_reference_metrics(raw_report["operational"], robustness["ablations"]["full"]["validation"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)

    print("[3/7] Building temporally out-of-fold calibration scores from training months 0-4...", flush=True)
    oof, folds = _temporal_oof_scores(train, min_history_months=2)
    unresolved = [w for fold in folds for w in fold["warnings"]]
    if unresolved:
        raise ValueError(f"Temporal calibration refits emitted warnings: {unresolved[:3]}")
    calibrators = fit_score_calibrators(oof["oof_score"], oof["fraud_bool"])

    print("[4/7] Comparing raw, sigmoid and isotonic probabilities on validation month 5...", flush=True)
    validation_scores = pd.DataFrame({
        "row_id": row_ids,
        "month": 5,
        "fraud_bool": y_val,
        "raw": raw_scores,
    })
    reports = {"raw": raw_report}
    calibrator_meta = {}
    for name, calibrator in calibrators.items():
        calibrated = calibrator.predict(raw_scores)
        validation_scores[name] = calibrated
        reports[name] = calibration_metrics(y_val, calibrated, row_ids, bins=args.calibration_bins)
        calibrator_meta[name] = calibrator.parameters()
    selected_method = select_calibration_method(reports)
    for name in ("sigmoid", "isotonic"):
        reports[name]["queue_overlap_vs_raw"] = queue_overlap(
            raw_scores, validation_scores[name].to_numpy(dtype=float), row_ids,
            capacity=args.capacity, seed=args.seed,
        )

    print(f"  Selected development calibration by Brier -> ECE rule: {selected_method}", flush=True)
    print(f"  raw Brier={reports['raw']['brier_score']:.8f}; ECE={reports['raw']['ece']:.8f}", flush=True)
    print(f"  sigmoid Brier={reports['sigmoid']['brier_score']:.8f}; ECE={reports['sigmoid']['ece']:.8f}", flush=True)
    print(f"  isotonic Brier={reports['isotonic']['brier_score']:.8f}; ECE={reports['isotonic']['ece']:.8f}", flush=True)

    print(f"[5/7] Running {args.stability_repeats} month/label-stratified training bootstrap refits...", flush=True)
    ensemble = np.empty((args.stability_repeats, len(validation)), dtype=np.float64)
    stability_runs = []
    for repeat in range(args.stability_repeats):
        repeat_seed = args.seed + repeat + 1
        positions = month_label_bootstrap_positions(train, seed=repeat_seed)
        sampled = train.iloc[positions]
        pipe = make_ablation_pipeline("full")
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            pipe.fit(sampled[FEATURES], sampled["fraud_bool"].to_numpy(dtype=int))
        with threadpool_limits(limits=4):
            scores = pipe.predict_proba(validation[FEATURES])[:, 1]
        fit_seconds = time.perf_counter() - fit_started
        if caught:
            raise ValueError(f"Stability refit {repeat + 1} emitted warnings: {[str(w.message) for w in caught]}")
        ensemble[repeat] = scores
        overlap = queue_overlap(raw_scores, scores, row_ids, capacity=args.capacity, seed=args.seed)
        stability_runs.append({
            "repeat": repeat + 1,
            "seed": repeat_seed,
            "fit_seconds": fit_seconds,
            "queue_jaccard_vs_reference": overlap["jaccard"],
            "reference_retained_fraction": overlap["reference_retained_fraction"],
        })
        print(
            f"  {repeat + 1:02d}/{args.stability_repeats}: "
            f"Jaccard={overlap['jaccard']:.4f}; retention={overlap['reference_retained_fraction']:.4f}; "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )

    print("[6/7] Summarizing queue and score stability...", flush=True)
    stability_summary, stability_cases = summarize_stability(
        raw_scores, ensemble, row_ids, capacity=args.capacity, seed=args.seed
    )

    source_paths = [
        ROOT / "risklens_core" / "calibration_uncertainty.py",
        ROOT / "scripts" / "run_calibration_uncertainty.py",
        ROOT / "risklens_core" / "robustness.py",
        ROOT / "risklens_core" / "scoring.py",
        ROOT / "requirements.txt",
    ]
    summary = {
        "status": "complete",
        "run_id": run_id,
        "run_directory": str(directory),
        "experiment": "calibration-uncertainty-v1",
        "protocol": "temporal-v1",
        "feature_contract": CONTRACT_VERSION,
        "dataset_sha256": fingerprint,
        "robustness_run_id": robustness["run_id"],
        "robustness_summary_sha256": sha256_file(args.robustness_summary),
        "reference_model": "histgb_15_leaves/full",
        "reference_model_artifact": str(artifacts["model"]),
        "reference_model_sha256": model_sha256,
        "split": robustness["split"],
        "month_rows": counts,
        "test_evaluated": False,
        "scored_month": 5,
        "reference_reproduction": reproduction,
        "calibration": {
            "fit_source": "temporally out-of-fold scores from training months; target months 2,3,4 each scored only by earlier months",
            "folds": folds,
            "oof_rows": int(len(oof)),
            "oof_fraud_rows": int(oof["fraud_bool"].sum()),
            "methods": reports,
            "calibrator_parameters": calibrator_meta,
            "selection_rule": "lowest validation Brier score; then ECE; then raw, sigmoid, isotonic name preference",
            "selected_method": selected_method,
            "operational_queue_policy": "unchanged raw HistGB score ranking for review-workflow-v1; calibration is not silently backfilled into the existing queue",
        },
        "stability": {
            "method": "month/label-stratified nonparametric bootstrap of training rows; same HistGB-15 configuration refit each repeat",
            "runs": stability_runs,
            "summary": stability_summary,
        },
        "outputs": {
            "temporal_oof_scores": str(directory / "temporal_oof_calibration_scores.csv"),
            "validation_scores": str(directory / "validation_calibrated_scores.csv"),
            "calibrators": str(directory / "score_calibrators.joblib"),
            "stability_cases": str(directory / "stability_cases.csv"),
            "stability_scores": str(directory / "stability_scores.npz"),
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "processor": platform.processor(),
            "native_thread_limit": 4,
            "packages": {p: importlib.metadata.version(p) for p in
                         ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]},
        },
        "limitations": [
            "Calibration method choice uses development validation month 5; months 6-7 remain sealed for final assessment.",
            "Temporal OOF calibration uses models trained on less data than the final month-5 reference model; the mapping is transferred to the full 0-4 fit.",
            "Fixed-width ECE depends on binning and is descriptive, not a hypothesis test.",
            "Stability refits measure sensitivity to a particular stratified bootstrap scheme; they are not confidence intervals, posterior uncertainty, or guarantees under future drift.",
            "The current 3% review capacity is a project assumption rather than measured analyst staffing capacity.",
            "No analyst decision is used as ground truth and no decision is fed back into training in this experiment.",
        ],
        "total_seconds": time.perf_counter() - started,
    }

    oof.to_csv(directory / "temporal_oof_calibration_scores.csv", index=False)
    validation_scores.to_csv(directory / "validation_calibrated_scores.csv", index=False)
    joblib.dump(calibrators, directory / "score_calibrators.joblib", compress=3)
    stability_cases.to_csv(directory / "stability_cases.csv", index=False)
    np.savez_compressed(directory / "stability_scores.npz", row_ids=row_ids, scores=ensemble)
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)

    print("[7/7] Saved calibration/uncertainty artifacts. Test months 6-7 remain sealed.", flush=True)
    print(f"  Summary: {args.summary}", flush=True)
    print("Upload/paste calibration_uncertainty_summary.json; do not open the final test set yet.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Calibration/uncertainty experiment stopped: {exc}") from exc
