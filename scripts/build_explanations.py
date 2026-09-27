"""Cache model-score explanations for a bounded slice of the existing analyst queue."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sqlalchemy import select, func
from threadpoolctl import threadpool_limits
from risklens_api.database import build_database
from risklens_core.baseline import sha256_file
from risklens_core import explanations as explainer_module
from risklens_core.explanations import METHOD, canonical_hash, explain_case, load_training_background
from risklens_core.explanation_store import CaseExplanation, case_frame, find_snapshot, save_snapshot, validate_snapshot
from risklens_core.features import CONTRACT_VERSION
from risklens_core.persistence import Case, QueueEntry, WorkflowRun, _validate_review_reference


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--review-summary", type=Path, default=Path("reports/review_workflow_summary.json"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--background-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/explanations"))
    parser.add_argument("--summary", type=Path, default=Path("reports/explanations_summary.json"))
    args = parser.parse_args()
    if not 1 <= args.limit <= 5000 or args.offset < 0 or not 2 <= args.background_size <= 128 or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid limit, offset, background size or seed")
    started = time.perf_counter()
    summary = json.loads(args.review_summary.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    _validate_review_reference(summary)
    engine, sessions = build_database()
    try:
        with sessions() as session:
            run = session.get(WorkflowRun, summary["run_id"])
            if run is None or run.summary_sha256 != sha256_file(args.review_summary):
                raise ValueError("Review summary does not match the imported database run")
            if run.dataset_sha256 != audit["sha256"] or run.scored_month != 5:
                raise ValueError("Database run differs from audited development data")
            if run.model_sha256 != summary["reference_model_sha256"]:
                raise ValueError("Stored model provenance differs from review summary")
            if run.model_name != summary["reference_model"]:
                raise ValueError("Stored model name differs from the review summary")
            model_path = Path(summary["reference_model_artifact"])
            if sha256_file(model_path) != run.model_sha256:
                raise ValueError("Model artifact hash differs from the stored scoring run")
            print("[1/3] Verifying data and sampling label-free training reference rows...", flush=True)
            background, metadata = load_training_background(args.csv, audit, size=args.background_size, seed=args.seed)
            configuration = {"method": METHOD, "feature_contract": CONTRACT_VERSION,
                             "background_sha256": metadata["background_sha256"],
                             "model_sha256": run.model_sha256, "seed": args.seed,
                             "explainer_source_sha256": sha256_file(explainer_module.__file__)}
            config_hash = canonical_hash(configuration)
            entries = session.execute(select(QueueEntry, Case).join(Case, Case.case_id == QueueEntry.case_id).where(
                QueueEntry.run_id == run.run_id).order_by(QueueEntry.risk_rank).offset(args.offset).limit(args.limit)).all()
            if not entries:
                raise ValueError("No queued cases in this rank range")
            # Only deserialize the user's own locally generated model after hash verification.
            model = joblib.load(model_path)
            build_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            directory = args.out_dir / build_id
            directory.mkdir(parents=True, exist_ok=False)
            background.to_csv(directory / "training_background.csv", index_label="source_row_id")
            write_json(directory / "background.json", metadata)
            write_json(directory / "configuration.json", configuration)
            inserted = reused = 0
            errors, case_ids, score_errors = [], [], []
            max_excluded_age_contribution = 0.0
            print(f"[2/3] Explaining {len(entries)} cases with {len(background) * 2} permutation paths each...", flush=True)
            with threadpool_limits(limits=4):
                for number, (entry, case) in enumerate(entries, start=1):
                    record = find_snapshot(session, entry.id, config_hash)
                    if record is not None:
                        if record.background_sha256 != metadata["background_sha256"]:
                            raise ValueError("Cached explanation has different background provenance")
                        payload = validate_snapshot(record, entry, case, run).model_dump()
                        reused += 1
                    else:
                        frame = case_frame(case)
                        fresh_score = float(model.predict_proba(frame)[0, 1])
                        if not np.isfinite(fresh_score) or abs(fresh_score - entry.risk_score) > 1e-12:
                            raise ValueError(f"Case {case.case_id} does not reproduce the persisted queue score")
                        case_seed = int(canonical_hash({"seed": args.seed, "case_id": case.case_id})[:8], 16)
                        payload = explain_case(model, frame, background, seed=case_seed)
                        _, created = save_snapshot(session, entry, case, run, payload,
                                                   config_sha256=config_hash,
                                                   background_sha256=metadata["background_sha256"])
                        inserted += int(created)
                        reused += int(not created)
                    if run.model_name == "histgb_15_leaves/no_customer_age":
                        age_value = next((item["contribution"] for item in payload["features"] if item["feature"] == "customer_age"), None)
                        if age_value is None:
                            raise ValueError("Frozen no-age explanation is missing customer_age audit contribution")
                        max_excluded_age_contribution = max(max_excluded_age_contribution, abs(float(age_value)))
                        if abs(float(age_value)) > 1e-12:
                            raise ValueError("Frozen no-age model explanation unexpectedly depends on customer_age")
                    errors.append(payload["reconstruction_error"])
                    score_errors.append(abs(payload["model_score"] - entry.risk_score))
                    case_ids.append(case.case_id)
                    if number == 1 or number % 10 == 0 or number == len(entries):
                        print(f"  {number}/{len(entries)} cases ready", flush=True)
            coverage = session.scalar(select(func.count()).select_from(CaseExplanation).join(
                QueueEntry, QueueEntry.id == CaseExplanation.queue_entry_id).where(
                QueueEntry.run_id == run.run_id, CaseExplanation.config_sha256 == config_hash))
            report = {"status": "complete", "experiment": "case-explanations-v1", "build_id": build_id,
                      "run_id": run.run_id, "model_sha256": run.model_sha256,
                      "model_name": run.model_name, "removed_features": summary.get("removed_features", []),
                      "dataset_sha256": run.dataset_sha256, "feature_contract": CONTRACT_VERSION,
                      "configuration": configuration, "config_sha256": config_hash,
                      "background": metadata, "scored_month": 5, "test_evaluated": bool(summary.get("test_evaluated", False)),
                      "reserved_test_rows_used": False, "max_excluded_age_contribution": max_excluded_age_contribution,
                      "requested_cases": len(entries), "inserted": inserted, "reused": reused,
                      "covered_cases_for_configuration": int(coverage), "queue_cases": run.selected_cases,
                      "case_ids": case_ids, "max_reconstruction_error": max(errors),
                      "max_queue_score_difference": max(score_errors), "artifacts_directory": str(directory),
                      "total_seconds": time.perf_counter() - started,
                      "limitations": explainer_module.LIMITATIONS}
            write_json(directory / "summary.json", report)
            write_json(args.summary, report)
            print(f"[3/3] Saved {args.summary}; coverage={coverage}/{run.selected_cases}; new={inserted}; reused={reused}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise SystemExit(f"Explanation build stopped: {exc}") from exc
