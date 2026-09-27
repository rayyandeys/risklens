"""Train two development baselines; never evaluate final-test months."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from risklens_core.baseline import load_development, diagnostics, make_pipeline, sha256_file
from risklens_core.features import CONTRACT_VERSION, FEATURES, EXCLUDED
from risklens_core.metrics import evaluate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--summary", type=Path, default=Path("reports/baseline_summary.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    print("[1/4] Checking dataset hash and loading months 0-5...", flush=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    train, validation, fingerprint, counts = load_development(args.csv, audit)
    print(f"Training: {len(train):,}; validation: {len(validation):,}. Final test is excluded.", flush=True)
    print("[2/4] Checking features and cross-split duplicates...", flush=True)
    checks = diagnostics(train, validation)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source_paths = sorted((ROOT / "risklens_core").glob("*.py")) + [Path(__file__).resolve(), ROOT / "requirements.txt"]
    summary = {
        "status": "running", "run_id": run_id, "run_directory": str(run_dir),
        "dataset_sha256": fingerprint, "protocol": "temporal-v1", "feature_contract": CONTRACT_VERSION,
        "split": {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]},
        "month_rows": counts, "features": FEATURES, "excluded": EXCLUDED, "seed": 42,
        "test_evaluated": False, "git_commit": commit,
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {"python": platform.python_version(), "os": platform.platform(),
                        "processor": platform.processor(), "native_thread_limit": 4,
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]}},
        "data_checks": checks, "models": {},
        "interpretation": ["Validation results only; no final performance claims.",
                           "Weighted probabilities are uncalibrated; compare Brier scores, not just ranking.",
                           "Top-k uses floor(N*capacity), with seeded label-free ties based on original CSV row IDs.",
                           "Preprocessing was fitted only on months 0-4; no oversampling.",
                           "Calibration, uncertainty, subgroup and boosted-tree experiments remain pending."],
    }
    y_train = train.fraud_bool.to_numpy(dtype=int)
    y_val = validation.fraud_bool.to_numpy(dtype=int)
    row_ids = validation.index.to_numpy(dtype=np.int64)
    # No model fitting for this baseline: predicts the observed training prevalence.
    prior = np.full(len(validation), y_train.mean())
    summary["constant_training_prior"] = evaluate(y_val, prior, row_ids)
    prediction_frame = pd.DataFrame({"row_id": row_ids, "month": 5, "fraud_bool": y_val})
    convergence_ok = True
    for name, weight in [("logistic_unweighted", None), ("logistic_balanced", "balanced")]:
        print(f"[3/4] Fitting {name} (up to 1,000 iterations)...", flush=True)
        pipe = make_pipeline(class_weight=weight)
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter("always")
            pipe.fit(train[FEATURES], y_train)
        fit_seconds = time.perf_counter() - fit_start
        with threadpool_limits(limits=4):
            probs = pipe.predict_proba(validation[FEATURES])[:, 1]
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        convergence_ok &= converged
        entry = {"converged": converged, "fit_seconds": fit_seconds,
                 "iterations": pipe.named_steps["model"].n_iter_.tolist(),
                 "parameters": {"solver": "lbfgs", "C": 1.0, "class_weight": weight, "max_iter": 1000, "tol": 1e-4},
                 "warnings": [str(w.message) for w in caught], "validation": evaluate(y_val, probs, row_ids)}
        prediction_frame[name] = probs
        joblib.dump(pipe, run_dir / f"{name}.joblib", compress=3)
        summary["models"][name] = entry
        write_json(run_dir / "summary.json", summary)
        metrics = entry["validation"]
        print(f"  AP={metrics['average_precision']:.5f}; ROC-AUC={metrics['roc_auc']:.5f}; converged={converged}", flush=True)
        for row in metrics["review_capacity"]:
            print(f"  Review {row['requested_fraction']:.0%}: precision={row['precision']:.4f}; recall={row['recall']:.4f}", flush=True)
    prediction_frame.to_csv(run_dir / "validation_predictions.csv", index=False)
    summary["status"] = "complete" if convergence_ok else "needs_convergence_review"
    summary["total_seconds"] = time.perf_counter() - started
    write_json(run_dir / "summary.json", summary)
    write_json(args.summary, summary)
    print(f"[4/4] Saved {args.summary}", flush=True)
    print("Send baseline_summary.json back for review. Keep model files and predictions locally.", flush=True)
    if not convergence_ok:
        print("WARNING: A model did not converge; do not treat those metrics as a finished baseline.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Baseline stopped: {exc}") from exc
