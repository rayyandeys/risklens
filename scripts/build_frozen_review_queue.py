"""Rebuild the month-5 analyst queue from the immutable frozen no-age HistGB model.

This is a post-final-test product-alignment step, not model selection. It never reads months 6-7
for scoring, ranking, labels, thresholds, or explanation inputs. The old full-model workflow run is
left untouched so its review history remains auditable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from risklens_core.baseline import load_development, sha256_file
from risklens_core.features import FEATURES
from risklens_core.frozen_review import (
    EXPECTED_CAPACITY,
    EXPECTED_MODEL,
    assert_validation_reproduction,
    verify_final_test_for_review,
    verify_freeze_for_review,
)
from risklens_core.metrics import evaluate
from risklens_core.review import build_review_population, selected_queue


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--freeze", type=Path, default=Path("reports/model_freeze.json"))
    parser.add_argument("--final-test", type=Path, default=Path("reports/final_test_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/frozen_review_workflow"))
    parser.add_argument("--summary", type=Path, default=Path("reports/frozen_review_workflow_summary.json"))
    args = parser.parse_args()
    started = time.perf_counter()

    print("[1/5] Verifying audit, immutable freeze, and completed one-time holdout lineage...", flush=True)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    final_test = json.loads(args.final_test.read_text(encoding="utf-8"))
    _, validation, fingerprint, counts = load_development(args.csv, audit)
    verify_freeze_for_review(freeze, fingerprint)
    verify_final_test_for_review(final_test, freeze, fingerprint)

    selected = freeze["selected_model"]
    model_path = Path(selected["frozen_artifact"])
    if not model_path.exists():
        raise ValueError(f"Missing frozen model artifact: {model_path}")
    if sha256_file(model_path) != selected["sha256"]:
        raise ValueError("Frozen model artifact hash differs from the immutable manifest")

    print("[2/5] Reproducing frozen no-age model scores on development month 5 only...", flush=True)
    model = joblib.load(model_path)
    with threadpool_limits(limits=4):
        scores = model.predict_proba(validation[FEATURES])[:, 1]
    y = validation.fraud_bool.to_numpy(dtype=int)
    row_ids = validation.index.to_numpy(dtype="int64")
    metrics = evaluate(y, scores, row_ids)
    assert_validation_reproduction(metrics, freeze["validation_reproduction"]["metrics"])

    print("[3/5] Building the frozen-model analyst queue at the immutable 3% capacity...", flush=True)
    population = build_review_population(
        validation[FEATURES], scores, row_ids,
        capacity=EXPECTED_CAPACITY, seed=42, model_name=EXPECTED_MODEL,
    )
    queue = selected_queue(population)
    if len(queue) != int(len(validation) * EXPECTED_CAPACITY):
        raise ValueError("Frozen review budget no longer matches floor(N * 3%)")

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

    cutoff = float(queue.iloc[-1]["risk_score"])
    summary = {
        "status": "complete",
        "run_id": run_id,
        "run_directory": str(directory),
        "experiment": "review-workflow-v2",
        "protocol": "temporal-v1-frozen-product",
        "dataset_sha256": fingerprint,
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": sha256_file(args.freeze),
        "final_test_opening_id": final_test["opening_id"],
        "final_test_summary_sha256": sha256_file(args.final_test),
        "reference_model": EXPECTED_MODEL,
        "reference_model_artifact": str(model_path),
        "reference_model_sha256": selected["sha256"],
        "removed_features": selected["removed_features"],
        "split": {
            "train": [0, 1, 2, 3, 4],
            "validation": [5],
            "test_reserved": [6, 7],
        },
        "month_rows": counts,
        "test_evaluated": True,
        "scored_month": 5,
        "queue_data_scope": {
            "development_month_only": True,
            "reserved_test_rows_used": False,
            "reserved_test_labels_used": False,
            "post_test_model_selection": False,
        },
        "review_policy": {
            "capacity": EXPECTED_CAPACITY,
            "selection": "top raw frozen-model scores with the immutable deterministic label-free tie policy",
            "selected_cases": int(len(queue)),
            "population_cases": int(len(population)),
            "cutoff_score": cutoff,
            "seed": 42,
            "capacity_is_assumption": True,
        },
        "validation_reproduction": metrics,
        "label_separation": {
            "analyst_queue_contains_target": False,
            "scored_population_contains_target": False,
            "ground_truth_file": str(truth_path),
            "ground_truth_purpose": "offline development evaluation only; never expose in analyst UI",
        },
        "outputs": {
            "scored_population": str(population_path),
            "analyst_queue": str(queue_path),
            "offline_ground_truth": str(truth_path),
            "analyst_decisions_template": str(decisions_path),
        },
        "interpretation": [
            "This run realigns the demo analyst product to the already-frozen no-age model after final evaluation.",
            "It does not select or tune a model from final-test results and it never scores reserved months 6-7 for this queue.",
            "The previous full-model workflow run and its review history remain intact for auditability.",
            "The analyst-facing queue withholds fraud_bool; labels stay isolated in an offline development artifact.",
            "The 3% review capacity remains a project assumption rather than measured staffing capacity.",
        ],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)

    print(f"  Population: {len(population):,}; selected: {len(queue):,}; cutoff={cutoff:.8f}", flush=True)
    print("[4/5] Verified no reserved-test rows or labels participate in the rebuilt queue.", flush=True)
    print(f"[5/5] Saved {args.summary}. Old workflow run remains untouched.", flush=True)
    print("Next: import this summary as a NEW workflow run, then generate explanations for that run.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Frozen review workflow stopped: {exc}") from exc
