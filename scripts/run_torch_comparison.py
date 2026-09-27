"""Run the fixed PyTorch tabular comparison without opening the sealed test months."""
from __future__ import annotations

import argparse
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

from risklens_core.baseline import load_development, sha256_file
from risklens_core.features import CONTRACT_VERSION, FEATURES
from risklens_core.metrics import evaluate
from risklens_core.robustness import paired_bootstrap_deltas
from risklens_core.scoring import (
    EXPECTED_SPLIT,
    artifact_fingerprint,
    resolve_reference_artifacts,
    validate_robustness_reference,
    verify_saved_predictions,
)
from risklens_core.torch_comparison import (
    TorchMLPConfig,
    fit_transform_preprocessor,
    load_torch_bundle,
    parameter_count,
    predict_scores,
    require_torch,
    save_torch_bundle,
    train_mlp,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_monitoring_reference(summary: dict, *, robustness: dict, fingerprint: str) -> None:
    if summary.get("status") != "complete" or summary.get("test_evaluated") is not False:
        raise ValueError("Temporal-monitoring reference must be a completed development-only report")
    if summary.get("experiment") != "temporal-monitoring-v1":
        raise ValueError("Unexpected monitoring experiment")
    if summary.get("dataset_sha256") != fingerprint:
        raise ValueError("Monitoring reference uses a different dataset fingerprint")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("split") != EXPECTED_SPLIT:
        raise ValueError("Monitoring reference uses a different feature contract or temporal split")
    if summary.get("robustness_run_id") != robustness.get("run_id"):
        raise ValueError("Monitoring reference does not point to the current robustness run")
    if summary.get("reference_model") != "histgb_15_leaves/full":
        raise ValueError("Monitoring reference model differs from the frozen development reference")


def comparison_row(name: str, metrics: dict) -> dict:
    row = {
        "model": name,
        "average_precision": metrics["average_precision"],
        "roc_auc": metrics["roc_auc"],
        "brier_score": metrics["brier_score"],
    }
    for capacity in metrics["review_capacity"]:
        pct = int(round(capacity["requested_fraction"] * 100))
        row[f"recall_at_{pct}pct"] = capacity["recall"]
        row[f"precision_at_{pct}pct"] = capacity["precision"]
        row[f"fraud_caught_at_{pct}pct"] = capacity["tp"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--robustness-summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--monitoring-summary", type=Path, default=Path("reports/temporal_monitoring_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/torch_comparison"))
    parser.add_argument("--summary", type=Path, default=Path("reports/torch_comparison_summary.json"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    require_torch()
    input_paths = {p.resolve() for p in [args.csv, args.audit, args.robustness_summary, args.monitoring_summary]}
    if args.summary.resolve() in input_paths:
        raise ValueError("Summary output must not overwrite an input")
    if args.epochs <= 0 or args.batch_size <= 0 or args.bootstrap_repeats < 100:
        raise ValueError("Use positive epochs/batch-size and at least 100 bootstrap repeats")

    started = time.perf_counter()
    print("[1/7] Verifying audited development data and completed monitoring lineage...", flush=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_summary.read_text(encoding="utf-8"))
    monitoring = json.loads(args.monitoring_summary.read_text(encoding="utf-8"))
    train, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_robustness_reference(robustness, fingerprint)
    validate_monitoring_reference(monitoring, robustness=robustness, fingerprint=fingerprint)

    reference_artifacts = resolve_reference_artifacts(robustness)
    reference_model_sha = artifact_fingerprint(reference_artifacts["model"])
    monitoring_sha = monitoring.get("reference_model_sha256")
    if monitoring_sha and monitoring_sha != reference_model_sha:
        raise ValueError("Monitoring and current robustness model artifacts have different hashes")

    print("[2/7] Reproducing the frozen HistGB-15 month-5 reference...", flush=True)
    histgb_model = joblib.load(reference_artifacts["model"])
    with threadpool_limits(limits=args.threads):
        histgb_started = time.perf_counter()
        histgb_scores = histgb_model.predict_proba(validation[FEATURES])[:, 1]
        histgb_inference_seconds = time.perf_counter() - histgb_started
    reproduction = verify_saved_predictions(
        reference_artifacts["predictions"],
        validation.index.to_numpy(dtype=np.int64),
        validation["fraud_bool"].to_numpy(dtype=int),
        histgb_scores,
    )
    histgb_metrics = evaluate(
        validation["fraud_bool"].to_numpy(dtype=int),
        histgb_scores,
        validation.index.to_numpy(dtype=np.int64),
    )

    print("[3/7] Fitting train-only preprocessing for the fixed neural baseline...", flush=True)
    preprocess_started = time.perf_counter()
    preprocessor, x_train, x_validation = fit_transform_preprocessor(train, validation)
    preprocessing_seconds = time.perf_counter() - preprocess_started
    y_train = train["fraud_bool"].to_numpy(dtype=int)
    y_validation = validation["fraud_bool"].to_numpy(dtype=int)
    row_ids = validation.index.to_numpy(dtype=np.int64)

    config = TorchMLPConfig(
        hidden_dims=(128, 64),
        dropout=0.10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        gradient_clip_norm=5.0,
        seed=42,
        class_weighting="none",
    )
    print(
        f"[4/7] Training fixed PyTorch MLP for {config.epochs} epochs on {len(train):,} rows "
        f"({args.device}, batch={config.batch_size})...",
        flush=True,
    )
    fit_started = time.perf_counter()
    neural_model, history, training_metadata = train_mlp(
        x_train, y_train, config=config, device=args.device, threads=args.threads
    )
    fit_seconds = time.perf_counter() - fit_started
    for entry in history:
        print(f"  epoch {entry['epoch']:02d}/{config.epochs}: loss={entry['mean_training_loss']:.6f}", flush=True)

    print("[5/7] Scoring validation month 5 and verifying serialization...", flush=True)
    inference_started = time.perf_counter()
    neural_scores = predict_scores(neural_model, x_validation, device=training_metadata["device"])
    neural_inference_seconds = time.perf_counter() - inference_started
    neural_metrics = evaluate(y_validation, neural_scores, row_ids)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    preprocessor_path = directory / "preprocessor.joblib"
    model_path = directory / "torch_mlp.pt"
    joblib.dump(preprocessor, preprocessor_path, compress=3)
    save_torch_bundle(
        model_path,
        neural_model,
        input_dim=x_train.shape[1],
        config=config,
        metadata={"feature_contract": CONTRACT_VERSION, "dataset_sha256": fingerprint},
    )
    reloaded, reloaded_config, _ = load_torch_bundle(model_path, device=training_metadata["device"])
    reload_scores = predict_scores(reloaded, x_validation, device=training_metadata["device"])
    serialization_difference = float(np.max(np.abs(neural_scores - reload_scores)))
    if serialization_difference > 1e-7:
        raise ValueError("Reloaded PyTorch model does not reproduce validation scores")

    print(f"[6/7] Running {args.bootstrap_repeats} paired bootstrap replicates vs HistGB-15...", flush=True)
    bootstrap = paired_bootstrap_deltas(
        y_validation,
        neural_scores,
        histgb_scores,
        row_ids,
        repeats=args.bootstrap_repeats,
        seed=20260927,
        capacity=0.03,
        ci=0.95,
    )

    comparison = [
        comparison_row("histgb_15_leaves/full", histgb_metrics),
        comparison_row("torch_mlp_v1", neural_metrics),
    ]
    comparison.sort(key=lambda row: (-row["recall_at_3pct"], -row["average_precision"], row["model"]))
    provisional = comparison[0]["model"]
    recall_ci = bootstrap["recall_delta"]
    ap_ci = bootstrap["average_precision_delta"]
    if recall_ci["ci_lower"] > 0 and ap_ci["ci_lower"] > 0:
        evidence = "torch_positive_on_both_bootstrap_intervals"
    elif recall_ci["ci_upper"] < 0 and ap_ci["ci_upper"] < 0:
        evidence = "histgb_positive_on_both_bootstrap_intervals"
    else:
        evidence = "mixed_or_inconclusive_bootstrap_intervals"

    predictions = pd.DataFrame({
        "row_id": row_ids,
        "month": 5,
        "fraud_bool": y_validation,
        "histgb_15_leaves_full": histgb_scores,
        "torch_mlp_v1": neural_scores,
    })
    predictions.to_csv(directory / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    pd.DataFrame(comparison).to_csv(directory / "comparison.csv", index=False)

    print("[7/7] Saving development-only comparison and provenance...", flush=True)
    source_paths = [
        ROOT / "risklens_core/torch_comparison.py",
        Path(__file__).resolve(),
        ROOT / "risklens_core/features.py",
        ROOT / "risklens_core/metrics.py",
        ROOT / "requirements.txt",
    ]
    package_names = ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "torch"]
    summary = {
        "status": "complete",
        "experiment": "torch-comparison-v1",
        "run_id": run_id,
        "run_directory": str(directory),
        "protocol": "temporal-v1",
        "feature_contract": CONTRACT_VERSION,
        "dataset_sha256": fingerprint,
        "robustness_run_id": robustness.get("run_id"),
        "monitoring_run_id": monitoring.get("run_id"),
        "split": EXPECTED_SPLIT,
        "month_rows": {str(k): int(v) for k, v in counts.items()},
        "test_evaluated": False,
        "scored_month": 5,
        "reference_model": "histgb_15_leaves/full",
        "reference_model_sha256": reference_model_sha,
        "reference_reproduction": reproduction,
        "preprocessing": {
            "fit_months": [0, 1, 2, 3, 4],
            "validation_month": 5,
            "input_dimensions": int(x_train.shape[1]),
            "fit_transform_seconds": preprocessing_seconds,
            "numeric": "median imputation + StandardScaler, trained on months 0-4 only",
            "missing_flags": "canonical six indicators passed through",
            "categorical": "most-frequent imputation + one-hot encoding; unknown validation categories ignored",
        },
        "torch_model": {
            "name": "torch_mlp_v1",
            "configuration": config.to_json(),
            "training_device": training_metadata["device"],
            "positive_class_weight": training_metadata["positive_class_weight"],
            "parameter_count": parameter_count(neural_model),
            "fit_seconds": fit_seconds,
            "validation_inference_seconds": neural_inference_seconds,
            "final_training_loss": history[-1]["mean_training_loss"],
            "serialization_max_absolute_score_difference": serialization_difference,
            "artifact": str(model_path),
            "artifact_sha256": sha256_file(model_path),
            "preprocessor_artifact": str(preprocessor_path),
            "preprocessor_sha256": sha256_file(preprocessor_path),
            "validation": neural_metrics,
        },
        "histgb_reference": {
            "validation_inference_seconds": histgb_inference_seconds,
            "validation": histgb_metrics,
        },
        "paired_bootstrap_torch_minus_histgb": bootstrap,
        "comparison": comparison,
        "provisional_best_by_predeclared_rule": provisional,
        "bootstrap_evidence": evidence,
        "selection_rule": "validation recall at 3% review capacity; then average precision; then model name",
        "selection_note": (
            "This is a development comparison, not final-test certification. The assistant should review the metrics, "
            "bootstrap intervals, calibration/drift evidence and engineering tradeoffs before freezing a model."
        ),
        "outputs": {
            "predictions": str(directory / "validation_predictions.csv"),
            "training_history": str(directory / "training_history.csv"),
            "comparison": str(directory / "comparison.csv"),
            "torch_model": str(model_path),
            "preprocessor": str(preprocessor_path),
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "processor": platform.processor(),
            "native_thread_limit": args.threads,
            "packages": {name: importlib.metadata.version(name) for name in package_names},
        },
        "limitations": [
            "Months 6-7 remain sealed and are not used for neural training, validation, architecture choice or comparison.",
            "The MLP architecture and training schedule are a fixed neural baseline, not a neural architecture search.",
            "Month 5 is not used for early stopping; it is used only for this development comparison.",
            "Paired bootstrap intervals quantify validation-row sampling uncertainty only, not training-seed uncertainty or future drift.",
            "Raw neural sigmoid outputs are model scores; no claim is made that they are calibrated fraud probabilities.",
            "The current 3% review capacity remains a project assumption rather than measured analyst staffing capacity.",
            "A stronger or more heavily tuned deep tabular model could differ; this experiment answers whether a reasonable fixed MLP adds value to RiskLens.",
        ],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)

    print(f"  HistGB: AP={histgb_metrics['average_precision']:.5f}; recall@3%={histgb_metrics['review_capacity'][1]['recall']:.5f}", flush=True)
    print(f"  PyTorch: AP={neural_metrics['average_precision']:.5f}; recall@3%={neural_metrics['review_capacity'][1]['recall']:.5f}", flush=True)
    print(f"  Provisional by rule: {provisional}", flush=True)
    print(f"  Bootstrap evidence: {evidence}", flush=True)
    print(f"  Saved: {args.summary}", flush=True)
    print("  Test months 6-7 remain sealed.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"Torch comparison stopped: {exc}") from exc
