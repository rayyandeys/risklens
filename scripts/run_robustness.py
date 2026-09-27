"""Run predeclared feature ablations and paired validation bootstrap for RiskLens."""
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

from risklens_core.baseline import load_development, sha256_file
from risklens_core.boosted import validate_reference as validate_baseline_reference, comparison_row
from risklens_core.features import CONTRACT_VERSION, FEATURES, EXCLUDED
from risklens_core.metrics import evaluate
from risklens_core.robustness import ABLATIONS, ablated_features, make_ablation_pipeline, paired_bootstrap_deltas


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_boosted_reference(summary, baseline, fingerprint):
    if summary.get("status") != "complete" or summary.get("test_evaluated") is not False:
        raise ValueError("Boosted reference must be a completed development-only report")
    if summary.get("dataset_sha256") != fingerprint:
        raise ValueError("Boosted reference uses a different dataset fingerprint")
    if summary.get("baseline_run_id") != baseline.get("run_id"):
        raise ValueError("Boosted reference points to a different baseline run")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("features") != FEATURES:
        raise ValueError("Boosted reference feature contract does not match")
    if summary.get("split") != baseline.get("split"):
        raise ValueError("Boosted reference temporal split does not match")
    if summary.get("provisional_best_model") != "histgb_15_leaves":
        raise ValueError("robustness-v1 is predeclared for histgb_15_leaves as the development reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("reports/baseline_summary.json"))
    parser.add_argument("--boosted-summary", type=Path, default=Path("reports/boosted_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/robustness"))
    parser.add_argument("--summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    inputs = {p.resolve() for p in [args.csv, args.audit, args.baseline_summary, args.boosted_summary]}
    if args.summary.resolve() in inputs:
        raise ValueError("Summary output must not overwrite an input")

    started = time.perf_counter()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    boosted = json.loads(args.boosted_summary.read_text(encoding="utf-8"))
    print("[1/5] Verifying audited data and development references...", flush=True)
    train, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_baseline_reference(baseline, fingerprint)
    validate_boosted_reference(boosted, baseline, fingerprint)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((ROOT / "risklens_core").glob("*.py")) + [Path(__file__).resolve(), ROOT / "requirements.txt"]

    summary = {
        "status": "running", "run_id": run_id, "run_directory": str(directory),
        "experiment": "robustness-v1", "protocol": "temporal-v1", "feature_contract": CONTRACT_VERSION,
        "dataset_sha256": fingerprint, "baseline_run_id": baseline["run_id"],
        "boosted_run_id": boosted["run_id"], "boosted_summary_sha256": sha256_file(args.boosted_summary),
        "split": baseline["split"], "month_rows": counts, "test_evaluated": False, "seed": 42,
        "reference_model": "histgb_15_leaves", "reference_validation": boosted["models"]["histgb_15_leaves"]["validation"],
        "ablations": {}, "bootstrap": {}, "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "processor": platform.processor(), "native_thread_limit": 4,
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]}},
        "interpretation": [
            "Ablations were declared before their BAF results: full, no credit-risk score, no proposed limit, neither, and no age.",
            "All ablations retain the canonical field cleaning, temporal split, HistGB-15 hyperparameters and training-only category vocabulary.",
            "Months 6-7 remain sealed. No threshold or feature choice is tuned on the final test set.",
            "Bootstrap intervals quantify validation-sample uncertainty only; they are not guarantees of production performance.",
            "The 3% review budget remains an explicit project assumption rather than measured staffing capacity.",
        ],
    }

    predictions = pd.DataFrame({"row_id": validation.index, "month": 5,
                                "fraud_bool": validation.fraud_bool.to_numpy(dtype=int)})
    y_train = train.fraud_bool.to_numpy(dtype=int)
    y_val = validation.fraud_bool.to_numpy(dtype=int)
    row_ids = validation.index.to_numpy(dtype="int64")

    print("[2/5] Fitting five predeclared HistGB-15 feature ablations...", flush=True)
    for name, removed in ABLATIONS.items():
        pipe = make_ablation_pipeline(name)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            pipe.fit(train[FEATURES], y_train)
        fit_seconds = time.perf_counter() - fit_started
        with threadpool_limits(limits=4):
            probability = pipe.predict_proba(validation[FEATURES])[:, 1]
        metrics = evaluate(y_val, probability, row_ids)
        model = pipe.named_steps["model"]
        summary["ablations"][name] = {
            "removed_features": list(removed), "remaining_features": ablated_features(name),
            "fit_seconds": fit_seconds, "iterations": int(model.n_iter_),
            "native_categorical_columns": int(model.is_categorical_.sum()),
            "warnings": [str(w.message) for w in caught], "validation": metrics,
        }
        predictions[name] = probability
        joblib.dump(pipe, directory / f"{name}.joblib", compress=3)
        print(f"  {name}: AP={metrics['average_precision']:.5f}; recall@3%={metrics['review_capacity'][1]['recall']:.4f}; caught={metrics['review_capacity'][1]['tp']}", flush=True)

    print("[3/5] Checking full-ablation reproducibility against the prior HistGB-15 run...", flush=True)
    prior = boosted["models"]["histgb_15_leaves"]["validation"]
    current = summary["ablations"]["full"]["validation"]
    for key in ["average_precision", "roc_auc", "brier_score"]:
        if abs(current[key] - prior[key]) > 1e-12:
            raise ValueError(f"Full ablation does not reproduce prior HistGB-15 {key}")
    for old, new in zip(prior["review_capacity"], current["review_capacity"]):
        if old["tp"] != new["tp"] or abs(old["recall"] - new["recall"]) > 1e-12:
            raise ValueError("Full ablation does not reproduce prior HistGB-15 capacity metrics")

    print(f"[4/5] Running {args.bootstrap_repeats} paired bootstrap replicates versus logistic_unweighted...", flush=True)
    baseline_pred_path = Path(baseline["run_directory"]) / "validation_predictions.csv"
    if not baseline_pred_path.exists():
        raise ValueError(f"Missing saved baseline predictions: {baseline_pred_path}")
    baseline_predictions = pd.read_csv(baseline_pred_path)
    needed = {"row_id", "fraud_bool", "logistic_unweighted"}
    if not needed.issubset(baseline_predictions.columns):
        raise ValueError("Baseline prediction file lacks required columns")
    aligned = predictions[["row_id", "fraud_bool", "full"]].merge(
        baseline_predictions[["row_id", "fraud_bool", "logistic_unweighted"]], on="row_id",
        suffixes=("_current", "_baseline"), validate="one_to_one")
    if len(aligned) != len(validation) or not (aligned["fraud_bool_current"] == aligned["fraud_bool_baseline"]).all():
        raise ValueError("Baseline predictions do not align with current validation rows and labels")
    summary["bootstrap"]["histgb_15_vs_logistic_unweighted"] = paired_bootstrap_deltas(
        aligned["fraud_bool_current"].to_numpy(dtype=int), aligned["full"].to_numpy(dtype=float),
        aligned["logistic_unweighted"].to_numpy(dtype=float), aligned["row_id"].to_numpy(dtype="int64"),
        repeats=args.bootstrap_repeats, seed=42, capacity=0.03, ci=0.95)

    rows = []
    for name, entry in summary["ablations"].items():
        rows.append({"ablation": name, "removed": ",".join(entry["removed_features"]), **comparison_row(name, entry["validation"])})
    pd.DataFrame(rows).to_csv(directory / "ablation_comparison.csv", index=False)
    predictions.to_csv(directory / "validation_predictions.csv", index=False)
    summary["status"] = "needs_warning_review" if any(e["warnings"] for e in summary["ablations"].values()) else "complete"
    summary["total_seconds"] = time.perf_counter() - started
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)
    print(f"[5/5] Saved {args.summary}. Test months remain sealed.", flush=True)
    print("Upload robustness_summary.json. Keep the timestamped models and prediction files.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Robustness experiment stopped: {exc}") from exc
