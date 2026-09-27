"""Compare two predeclared tree configurations against saved logistic results."""
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
import pandas as pd
from threadpoolctl import threadpool_limits
from risklens_core.baseline import load_development, diagnostics, sha256_file
from risklens_core.boosted import CONFIGURATIONS, make_boosted_pipeline, validate_reference, comparison_row
from risklens_core.features import CONTRACT_VERSION, FEATURES, EXCLUDED
from risklens_core.metrics import evaluate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("reports/baseline_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/boosted"))
    parser.add_argument("--summary", type=Path, default=Path("reports/boosted_summary.json"))
    args = parser.parse_args()
    if args.summary.resolve() in {args.csv.resolve(), args.audit.resolve(), args.baseline_summary.resolve()}:
        raise ValueError("Summary output must not overwrite any input")
    started = time.perf_counter()
    reference = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    print("[1/4] Verifying source data and baseline reference...", flush=True)
    train, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_reference(reference, fingerprint)
    # Tie row identity and labels to the already reviewed reference before comparing metrics.
    for entry in reference["models"].values():
        if (entry["validation"]["rows"] != len(validation) or
                entry["validation"]["fraud_rows"] != int(validation.fraud_bool.sum())):
            raise ValueError("Reference validation counts do not match current data")
    print("[2/4] Reusing the same features and temporal split; checking data...", flush=True)
    checks = diagnostics(train, validation)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((ROOT / "risklens_core").glob("*.py")) + [Path(__file__).resolve(), ROOT / "requirements.txt"]
    summary = {
        "status": "running", "run_id": run_id, "run_directory": str(directory),
        "experiment": "boosted-v1", "protocol": "temporal-v1", "feature_contract": CONTRACT_VERSION,
        "dataset_sha256": fingerprint, "baseline_run_id": reference["run_id"],
        "baseline_summary_sha256": sha256_file(args.baseline_summary),
        "split": reference["split"], "month_rows": counts, "features": FEATURES, "excluded": EXCLUDED,
        "test_evaluated": False, "seed": 42, "data_checks": checks,
        "selection_rule": "validation recall at 3% review capacity; then average precision; then model name",
        "selection_scope": "provisional development ranking, not final model certification",
        "models": {}, "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "processor": platform.processor(), "native_thread_limit": 4,
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]}},
        "interpretation": [
            "Two configurations declared before tree results. No test evaluation or threshold tuning.",
            "Native numeric missing-value and categorical splits; training-only category vocabulary.",
            "200 boosting iterations, early stopping disabled to avoid a random internal validation split.",
            "3% is an assumed review budget chosen for the experiment, not measured staffing capacity.",
            "No confidence interval or significance claim yet. Feature inputs are shared; preprocessing differs by model family.",
            "Raw tree probabilities are not guaranteed calibrated. Calibration remains pending.",
        ],
    }
    predictions = pd.DataFrame({"row_id": validation.index, "month": 5,
                                "fraud_bool": validation.fraud_bool.to_numpy(dtype=int)})
    for name, leaves in CONFIGURATIONS.items():
        print(f"[3/4] Fitting {name}: 200 boosting iterations...", flush=True)
        pipe = make_boosted_pipeline(max_leaf_nodes=leaves)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            pipe.fit(train[FEATURES], train.fraud_bool.to_numpy(dtype=int))
        fit_seconds = time.perf_counter() - fit_started
        with threadpool_limits(limits=4):
            probability = pipe.predict_proba(validation[FEATURES])[:, 1]
        model = pipe.named_steps["model"]
        metrics = evaluate(validation.fraud_bool.to_numpy(dtype=int), probability,
                           validation.index.to_numpy(dtype="int64"))
        entry = {"fit_seconds": fit_seconds, "iterations": int(model.n_iter_),
                 "native_categorical_columns": int(model.is_categorical_.sum()),
                 "parameters": {k: model.get_params()[k] for k in
                                ["loss", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf",
                                 "l2_regularization", "max_bins", "early_stopping", "class_weight", "random_state"]},
                 "warnings": [str(w.message) for w in caught], "validation": metrics}
        summary["models"][name] = entry
        predictions[name] = probability
        joblib.dump(pipe, directory / f"{name}.joblib", compress=3)
        write_json(directory / "summary.json", summary)
        print(f"  AP={metrics['average_precision']:.5f}; ROC-AUC={metrics['roc_auc']:.5f}", flush=True)
        for row in metrics["review_capacity"]:
            print(f"  Review {row['requested_fraction']:.0%}: caught={row['tp']}; recall={row['recall']:.4f}; precision={row['precision']:.4f}", flush=True)
    comparison = [comparison_row(name, entry["validation"]) for name, entry in reference["models"].items()]
    comparison.extend(comparison_row(name, entry["validation"]) for name, entry in summary["models"].items())
    comparison.sort(key=lambda row: (-row["recall_at_3pct"], -row["average_precision"], row["model"]))
    summary["comparison"] = comparison
    summary["provisional_best_model"] = comparison[0]["model"]
    summary["status"] = "needs_warning_review" if any(e["warnings"] for e in summary["models"].values()) else "complete"
    summary["total_seconds"] = time.perf_counter() - started
    predictions.to_csv(directory / "validation_predictions.csv", index=False)
    pd.DataFrame(comparison).to_csv(directory / "comparison.csv", index=False)
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)
    print(f"[4/4] Saved {args.summary}. Provisional best: {summary['provisional_best_model']}", flush=True)
    print("Upload boosted_summary.json. Keep all timestamped model/prediction files for later analysis.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Boosted comparison stopped: {exc}") from exc
