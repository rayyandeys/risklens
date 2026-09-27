"""Build the first label-safe RiskLens analyst queue from the validated development reference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from risklens_core.baseline import load_development, sha256_file
from risklens_core.features import FEATURES
from risklens_core.metrics import evaluate
from risklens_core.review import build_review_population, selected_queue
from risklens_core.scoring import (
    artifact_fingerprint,
    resolve_reference_artifacts,
    validate_robustness_reference,
    verify_saved_predictions,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _same_metrics(current, reference):
    for key in ("average_precision", "roc_auc", "brier_score"):
        if abs(current[key] - reference[key]) > 1e-12:
            raise ValueError(f"Fresh scoring does not reproduce robustness {key}")
    for old, new in zip(reference["review_capacity"], current["review_capacity"]):
        if old["requested_fraction"] != new["requested_fraction"]:
            raise ValueError("Review-capacity definitions changed")
        for key in ("tp", "fp", "fn", "tn"):
            if old[key] != new[key]:
                raise ValueError("Fresh scoring does not reproduce robustness capacity counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--robustness-summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/review_workflow"))
    parser.add_argument("--summary", type=Path, default=Path("reports/review_workflow_summary.json"))
    parser.add_argument("--capacity", type=float, default=0.03)
    args = parser.parse_args()

    if not 0 < args.capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    started = time.perf_counter()

    print("[1/4] Verifying audited data and robustness reference...", flush=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_summary.read_text(encoding="utf-8"))
    _, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_robustness_reference(robustness, fingerprint)
    artifacts = resolve_reference_artifacts(robustness)
    if not artifacts["predictions"].exists():
        raise ValueError(f"Missing robustness predictions: {artifacts['predictions']}")
    model_sha256 = artifact_fingerprint(artifacts["model"])

    print("[2/4] Loading the full HistGB-15 artifact and reproducing validation scores...", flush=True)
    model = joblib.load(artifacts["model"])
    with threadpool_limits(limits=4):
        scores = model.predict_proba(validation[FEATURES])[:, 1]
    y = validation.fraud_bool.to_numpy(dtype=int)
    row_ids = validation.index.to_numpy(dtype="int64")
    reproduction = verify_saved_predictions(artifacts["predictions"], row_ids, y, scores)
    metrics = evaluate(y, scores, row_ids)
    _same_metrics(metrics, robustness["ablations"]["full"]["validation"])

    print(f"[3/4] Building label-safe analyst queue at {args.capacity:.1%} capacity...", flush=True)
    population = build_review_population(
        validation[FEATURES], scores, row_ids, capacity=args.capacity, seed=42,
        model_name="histgb_15_leaves/full",
    )
    queue = selected_queue(population)
    truth = pd.DataFrame({
        "case_id": [f"RL-{int(row_id):09d}" for row_id in row_ids],
        "source_row_id": row_ids,
        "fraud_bool": y,
    }).sort_values("source_row_id", kind="stable")
    decision_template = queue.loc[:, ["case_id"]].copy()
    decision_template["decision"] = ""
    decision_template["analyst_note"] = ""

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    population_path = directory / "scored_population.csv"
    queue_path = directory / "analyst_queue.csv"
    truth_path = directory / "offline_ground_truth.csv"
    decisions_path = directory / "analyst_decisions_template.csv"
    population.to_csv(population_path, index=False)
    queue.to_csv(queue_path, index=False)
    truth.to_csv(truth_path, index=False)
    decision_template.to_csv(decisions_path, index=False)

    cutoff = queue.iloc[-1]["risk_score"] if len(queue) else None
    summary = {
        "status": "complete",
        "run_id": run_id,
        "run_directory": str(directory),
        "experiment": "review-workflow-v1",
        "protocol": "temporal-v1",
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
        "review_policy": {
            "capacity": args.capacity,
            "selection": "top risk scores with the same deterministic label-free tie policy as evaluation",
            "selected_cases": int(len(queue)),
            "population_cases": int(len(population)),
            "cutoff_score": None if cutoff is None else float(cutoff),
            "seed": 42,
            "capacity_is_assumption": True,
        },
        "reproduction_check": reproduction,
        "offline_validation_metrics": metrics,
        "label_separation": {
            "analyst_queue_contains_target": False,
            "scored_population_contains_target": False,
            "ground_truth_file": str(truth_path),
            "ground_truth_purpose": "offline evaluation only; do not expose in analyst UI",
        },
        "outputs": {
            "scored_population": str(population_path),
            "analyst_queue": str(queue_path),
            "offline_ground_truth": str(truth_path),
            "analyst_decisions_template": str(decisions_path),
        },
        "interpretation": [
            "This is a development-only workflow slice over validation month 5, not a production deployment.",
            "Months 6-7 remain sealed and are not read for model evaluation or queue construction.",
            "The analyst-facing queue intentionally withholds fraud_bool; labels are isolated for offline evaluation.",
            "The 3% review capacity is a project assumption rather than measured staffing capacity.",
            "Explanations, persistence, authentication, API/UI and monitoring remain later milestones.",
        ],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)

    print(f"  Population: {len(population):,}; selected for review: {len(queue):,}", flush=True)
    print(f"  Cutoff score: {float(cutoff):.8f}" if cutoff is not None else "  Cutoff score: n/a", flush=True)
    print(f"[4/4] Saved {args.summary}. Test months remain sealed.", flush=True)
    print("Upload review_workflow_summary.json. Keep the timestamped queue files for the API/UI milestone.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Review workflow stopped: {exc}") from exc
