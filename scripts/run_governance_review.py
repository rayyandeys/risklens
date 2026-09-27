"""Run the final development-only subgroup/error governance review before model freeze."""
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

import numpy as np
import pandas as pd

from risklens_core.baseline import load_development, sha256_file
from risklens_core.features import CONTRACT_VERSION
from risklens_core.governance import (
    age_groups,
    compact_group_deltas,
    error_partition,
    error_summary,
    intended_balance_groups,
    numeric_error_contrasts,
    subgroup_metrics,
    top_capacity_mask,
)
from risklens_core.metrics import evaluate
from risklens_core.robustness import paired_bootstrap_deltas
from risklens_core.scoring import EXPECTED_SPLIT, resolve_reference_artifacts, validate_robustness_reference


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_chain(summary: dict, *, experiment: str, fingerprint: str) -> None:
    if summary.get("status") != "complete" or summary.get("test_evaluated") is not False:
        raise ValueError(f"{experiment} must be a completed development-only report")
    if summary.get("experiment") != experiment:
        raise ValueError(f"Unexpected experiment in {experiment} reference")
    if summary.get("dataset_sha256") != fingerprint:
        raise ValueError(f"{experiment} uses a different dataset fingerprint")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("split") != EXPECTED_SPLIT:
        raise ValueError(f"{experiment} uses a different feature contract or split")


def validate_lineage(robustness, calibration, monitoring, torch):
    if calibration.get("robustness_run_id") != robustness.get("run_id"):
        raise ValueError("Calibration report does not point to the current robustness run")
    if calibration.get("calibration", {}).get("selected_method") != "raw":
        raise ValueError("Governance review expects raw score calibration decision")
    if monitoring.get("robustness_run_id") != robustness.get("run_id"):
        raise ValueError("Monitoring report does not point to the current robustness run")
    if monitoring.get("calibration_uncertainty_run_id") != calibration.get("run_id"):
        raise ValueError("Monitoring report does not point to the current calibration run")
    if torch.get("robustness_run_id") != robustness.get("run_id"):
        raise ValueError("PyTorch report does not point to the current robustness run")
    if torch.get("monitoring_run_id") != monitoring.get("run_id"):
        raise ValueError("PyTorch report does not point to the current monitoring run")
    if torch.get("provisional_best_by_predeclared_rule") != "histgb_15_leaves/full":
        raise ValueError("Unexpected development model-family result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--robustness-summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--calibration-summary", type=Path, default=Path("reports/calibration_uncertainty_summary.json"))
    parser.add_argument("--monitoring-summary", type=Path, default=Path("reports/temporal_monitoring_summary.json"))
    parser.add_argument("--torch-summary", type=Path, default=Path("reports/torch_comparison_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/governance_review"))
    parser.add_argument("--summary", type=Path, default=Path("reports/governance_review_summary.json"))
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args()

    if args.bootstrap_repeats < 100:
        raise ValueError("Use at least 100 bootstrap repeats")
    inputs = {p.resolve() for p in [args.csv, args.audit, args.robustness_summary, args.calibration_summary,
                                    args.monitoring_summary, args.torch_summary]}
    if args.summary.resolve() in inputs:
        raise ValueError("Summary output must not overwrite an input")

    started = time.perf_counter()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_summary.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_summary.read_text(encoding="utf-8"))
    monitoring = json.loads(args.monitoring_summary.read_text(encoding="utf-8"))
    torch = json.loads(args.torch_summary.read_text(encoding="utf-8"))

    print("[1/6] Verifying audited development data and experiment lineage...", flush=True)
    _, validation, fingerprint, counts = load_development(args.csv, audit)
    validate_robustness_reference(robustness, fingerprint)
    for summary, experiment in [
        (calibration, "calibration-uncertainty-v1"),
        (monitoring, "temporal-monitoring-v1"),
        (torch, "torch-comparison-v1"),
    ]:
        validate_chain(summary, experiment=experiment, fingerprint=fingerprint)
    validate_lineage(robustness, calibration, monitoring, torch)

    print("[2/6] Loading frozen full/no-age validation predictions and reproducing metrics...", flush=True)
    prediction_path = resolve_reference_artifacts(robustness)["predictions"]
    predictions = pd.read_csv(prediction_path)
    required = {"row_id", "fraud_bool", "full", "no_customer_age"}
    if not required.issubset(predictions.columns):
        raise ValueError("Robustness validation predictions lack full/no-age columns")
    current = pd.DataFrame({
        "row_id": validation.index.to_numpy(dtype=np.int64),
        "fraud_bool_current": validation["fraud_bool"].to_numpy(dtype=int),
    })
    aligned = current.merge(predictions[list(required)], on="row_id", how="inner", validate="one_to_one")
    if len(aligned) != len(validation):
        raise ValueError("Robustness predictions do not cover current validation rows")
    if not np.array_equal(aligned["fraud_bool_current"].to_numpy(), aligned["fraud_bool"].to_numpy()):
        raise ValueError("Robustness predictions contain different validation labels")
    aligned = aligned.sort_values("row_id", kind="stable").reset_index(drop=True)
    validation_by_id = validation.loc[aligned["row_id"].to_numpy()].copy()
    y = aligned["fraud_bool"].to_numpy(dtype=int)
    row_ids = aligned["row_id"].to_numpy(dtype=np.int64)
    full_scores = aligned["full"].to_numpy(dtype=float)
    no_age_scores = aligned["no_customer_age"].to_numpy(dtype=float)
    full_metrics = evaluate(y, full_scores, row_ids)
    no_age_metrics = evaluate(y, no_age_scores, row_ids)
    for name, observed in [("full", full_metrics), ("no_customer_age", no_age_metrics)]:
        expected = robustness["ablations"][name]["validation"]
        for key in ("average_precision", "roc_auc", "brier_score"):
            if abs(observed[key] - expected[key]) > 1e-12:
                raise ValueError(f"{name} predictions do not reproduce {key}")
        if observed["review_capacity"][1]["tp"] != expected["review_capacity"][1]["tp"]:
            raise ValueError(f"{name} predictions do not reproduce recall@3%")

    print(f"[3/6] Running {args.bootstrap_repeats} paired bootstrap replicates: no-age minus full...", flush=True)
    no_age_bootstrap = paired_bootstrap_deltas(
        y, no_age_scores, full_scores, row_ids,
        repeats=args.bootstrap_repeats, seed=20260927, capacity=0.03, ci=0.95,
    )

    print("[4/6] Computing descriptive subgroup metrics under one global 3% queue policy...", flush=True)
    full_selected = top_capacity_mask(full_scores, row_ids, capacity=0.03, seed=42)
    no_age_selected = top_capacity_mask(no_age_scores, row_ids, capacity=0.03, seed=42)
    age = age_groups(validation_by_id["customer_age"])
    balance = intended_balance_groups(validation_by_id)
    age_full = subgroup_metrics(y, full_selected, age, scores=full_scores)
    age_no_age = subgroup_metrics(y, no_age_selected, age, scores=no_age_scores)
    balance_full = subgroup_metrics(y, full_selected, balance, scores=full_scores)
    balance_no_age = subgroup_metrics(y, no_age_selected, balance, scores=no_age_scores)

    print("[5/6] Building local error-analysis artifacts...", flush=True)
    errors = error_partition(y, full_selected)
    error_rows = pd.DataFrame({
        "row_id": row_ids,
        "score": full_scores,
        "error_class": errors,
        "age_group": age,
        "intended_balance_group": balance,
    })
    near_boundary_missed = error_rows.loc[error_rows.error_class == "false_negative"].sort_values(
        ["score", "row_id"], ascending=[False, True], kind="stable").head(25)
    high_score_false_alarms = error_rows.loc[error_rows.error_class == "false_positive"].sort_values(
        ["score", "row_id"], ascending=[False, True], kind="stable").head(25)
    contrasts = numeric_error_contrasts(validation_by_id, y, full_selected, top_n=10)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / run_id
    directory.mkdir(parents=True, exist_ok=False)
    near_boundary_missed.to_csv(directory / "near_boundary_missed_fraud.csv", index=False)
    high_score_false_alarms.to_csv(directory / "high_score_false_alarms.csv", index=False)
    pd.DataFrame(age_full).to_csv(directory / "age_groups_full.csv", index=False)
    pd.DataFrame(age_no_age).to_csv(directory / "age_groups_no_age.csv", index=False)
    pd.DataFrame(balance_full).to_csv(directory / "intended_balance_groups_full.csv", index=False)

    print("[6/6] Saving development-only governance summary; test months remain sealed...", flush=True)
    source_paths = [ROOT / "risklens_core/governance.py", Path(__file__).resolve(),
                    ROOT / "risklens_core/robustness.py", ROOT / "risklens_core/metrics.py",
                    ROOT / "requirements.txt"]
    summary = {
        "status": "complete",
        "experiment": "governance-review-v1",
        "run_id": run_id,
        "run_directory": str(directory),
        "protocol": "temporal-v1",
        "feature_contract": CONTRACT_VERSION,
        "dataset_sha256": fingerprint,
        "split": EXPECTED_SPLIT,
        "month_rows": {str(k): int(v) for k, v in counts.items()},
        "test_evaluated": False,
        "scored_month": 5,
        "lineage": {
            "robustness_run_id": robustness["run_id"],
            "calibration_uncertainty_run_id": calibration["run_id"],
            "temporal_monitoring_run_id": monitoring["run_id"],
            "torch_comparison_run_id": torch["run_id"],
        },
        "reference_model": "histgb_15_leaves/full",
        "challenger": "histgb_15_leaves/no_customer_age",
        "overall_metrics": {"full": full_metrics, "no_customer_age": no_age_metrics},
        "no_age_minus_full_bootstrap": no_age_bootstrap,
        "global_queue_policy": {
            "capacity": 0.03,
            "tie_seed": 42,
            "full_selected": int(full_selected.sum()),
            "no_age_selected": int(no_age_selected.sum()),
            "intersection": int((full_selected & no_age_selected).sum()),
            "jaccard": float((full_selected & no_age_selected).sum() / (full_selected | no_age_selected).sum()),
            "note": "Subgroup metrics use one global queue per model; no subgroup-specific threshold is tuned.",
        },
        "subgroups": {
            "customer_age": {
                "definition": "age_lt_50, age_ge_50, missing; coarse descriptive groups predeclared before this BAF run",
                "full": age_full,
                "no_customer_age": age_no_age,
                "no_age_minus_full": compact_group_deltas(age_no_age, age_full),
            },
            "intended_balance_missingness": {
                "definition": "canonical cleaned intended_balcon_amount missing flag versus observed",
                "full": balance_full,
                "no_customer_age": balance_no_age,
                "no_age_minus_full": compact_group_deltas(balance_no_age, balance_full),
            },
        },
        "error_analysis": {
            "full_model_error_classes": error_summary(y, full_scores, full_selected),
            "top_numeric_contrasts": contrasts,
            "near_boundary_missed_fraud": str(directory / "near_boundary_missed_fraud.csv"),
            "high_score_false_alarms": str(directory / "high_score_false_alarms.csv"),
        },
        "model_freeze_readiness": {
            "status": "awaiting_review",
            "note": "This command does not declare a fairness conclusion or switch models. Review the direct no-age bootstrap and subgroup/error evidence before freezing the development choice.",
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {
            "python": platform.python_version(), "os": platform.platform(), "processor": platform.processor(),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"]},
        },
        "limitations": [
            "Months 6-7 remain sealed; all subgroup and error findings are development-month descriptions.",
            "Age groups are coarse benchmark diagnostics and do not establish legal or regulatory fairness.",
            "The BAF dataset is synthetic and lacks many protected attributes and real decision outcomes; fairness certification is out of scope.",
            "Group metrics use the global 3% review policy; differences may reflect prevalence, score distributions, sampling variation and other factors.",
            "The no-age comparison removes only customer_age; correlated proxy features may remain.",
            "Error contrasts are descriptive associations, not causal explanations of fraud or model errors.",
            "The 3% review capacity remains a project assumption rather than measured staffing capacity.",
        ],
        "total_seconds": time.perf_counter() - started,
    }
    write_json(directory / "summary.json", summary)
    write_json(args.summary, summary)
    print(f"Saved {args.summary}. Upload it for the model-freeze decision.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Governance review stopped: {exc}") from exc
