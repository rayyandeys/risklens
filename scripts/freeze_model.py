"""Freeze the reviewed RiskLens model/protocol before opening months 6-7.

This command scores *development month 5 only* to verify the exact selected model
artifact.  It does not evaluate, preprocess, diagnose, or inspect labels from months
6-7.  Final-test evaluation is intentionally a separate package/checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from risklens_core.baseline import load_development, sha256_file
from risklens_core.features import CONTRACT_VERSION, FEATURES
from risklens_core.metrics import evaluate
from risklens_core.model_freeze import (
    CAPACITY,
    FREEZE_VERSION,
    SELECTED_ABLATION,
    SELECTED_MODEL_NAME,
    TIE_SEED,
    frozen_policy,
    validate_governance_for_freeze,
)
from risklens_core.scoring import EXPECTED_SPLIT, validate_robustness_reference


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_linked_report(summary: dict, *, experiment: str, fingerprint: str) -> None:
    if summary.get("status") != "complete" or summary.get("experiment") != experiment:
        raise ValueError(f"Expected completed {experiment} report")
    if summary.get("test_evaluated") is not False:
        raise ValueError(f"{experiment} indicates final-test access")
    if summary.get("dataset_sha256") != fingerprint:
        raise ValueError(f"{experiment} uses a different dataset fingerprint")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("split") != EXPECTED_SPLIT:
        raise ValueError(f"{experiment} uses a different feature contract or split")


def compare_metrics(observed: dict, expected: dict) -> None:
    for key in ("average_precision", "roc_auc", "brier_score"):
        if abs(float(observed[key]) - float(expected[key])) > 1e-12:
            raise ValueError(f"Selected model does not reproduce governance {key}")
    observed_3 = next(x for x in observed["review_capacity"] if abs(x["requested_fraction"] - CAPACITY) < 1e-12)
    expected_3 = next(x for x in expected["review_capacity"] if abs(x["requested_fraction"] - CAPACITY) < 1e-12)
    if observed_3["tp"] != expected_3["tp"] or abs(observed_3["recall"] - expected_3["recall"]) > 1e-12:
        raise ValueError("Selected model does not reproduce governance Recall@3%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/raw/Base.csv"))
    parser.add_argument("--audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--robustness-summary", type=Path, default=Path("reports/robustness_summary.json"))
    parser.add_argument("--calibration-summary", type=Path, default=Path("reports/calibration_uncertainty_summary.json"))
    parser.add_argument("--monitoring-summary", type=Path, default=Path("reports/temporal_monitoring_summary.json"))
    parser.add_argument("--torch-summary", type=Path, default=Path("reports/torch_comparison_summary.json"))
    parser.add_argument("--governance-summary", type=Path, default=Path("reports/governance_review_summary.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/model_freeze"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/model_freeze.json"))
    args = parser.parse_args()

    if args.manifest.exists():
        raise ValueError(f"Freeze manifest already exists and is immutable: {args.manifest}")

    print("[1/5] Verifying audited development data and completed evidence chain...", flush=True)
    audit = load_json(args.audit)
    _, validation, fingerprint, counts = load_development(args.csv, audit)
    robustness = load_json(args.robustness_summary)
    calibration = load_json(args.calibration_summary)
    monitoring = load_json(args.monitoring_summary)
    torch = load_json(args.torch_summary)
    governance = load_json(args.governance_summary)

    validate_robustness_reference(robustness, fingerprint)
    for summary, experiment in [
        (calibration, "calibration-uncertainty-v1"),
        (monitoring, "temporal-monitoring-v1"),
        (torch, "torch-comparison-v1"),
        (governance, "governance-review-v1"),
    ]:
        validate_linked_report(summary, experiment=experiment, fingerprint=fingerprint)
    if governance.get("lineage", {}).get("robustness_run_id") != robustness.get("run_id"):
        raise ValueError("Governance report points to a different robustness run")
    if governance.get("lineage", {}).get("calibration_uncertainty_run_id") != calibration.get("run_id"):
        raise ValueError("Governance report points to a different calibration run")
    if governance.get("lineage", {}).get("temporal_monitoring_run_id") != monitoring.get("run_id"):
        raise ValueError("Governance report points to a different monitoring run")
    if governance.get("lineage", {}).get("torch_comparison_run_id") != torch.get("run_id"):
        raise ValueError("Governance report points to a different PyTorch run")
    if calibration.get("calibration", {}).get("selected_method") != "raw":
        raise ValueError("Freeze expects the reviewed raw-score calibration decision")

    decision = validate_governance_for_freeze(governance)

    print("[2/5] Loading the exact no-age HistGB artifact from robustness-v1...", flush=True)
    source_model = Path(robustness["run_directory"]) / f"{SELECTED_ABLATION}.joblib"
    prediction_path = Path(robustness["run_directory"]) / "validation_predictions.csv"
    if not source_model.is_file() or not prediction_path.is_file():
        raise ValueError("Required robustness model/prediction artifact is missing")
    source_model_sha = sha256_file(source_model)
    model = joblib.load(source_model)
    if tuple(model.named_steps["clean_select"].features) != tuple(decision["remaining_features"]):
        raise ValueError("Selected artifact feature set does not match the frozen no-age contract")
    hgb = model.named_steps["model"]
    expected_params = {
        "learning_rate": 0.08,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "max_bins": 255,
        "early_stopping": False,
        "class_weight": None,
        "random_state": 42,
    }
    actual_params = hgb.get_params()
    for key, expected in expected_params.items():
        if actual_params.get(key) != expected:
            raise ValueError(f"Selected artifact has unexpected HistGB parameter {key}")

    print("[3/5] Reproducing selected month-5 validation scores and metrics...", flush=True)
    with threadpool_limits(limits=4):
        fresh_scores = model.predict_proba(validation[FEATURES])[:, 1]
    saved = pd.read_csv(prediction_path)
    required = {"row_id", "fraud_bool", SELECTED_ABLATION}
    if not required.issubset(saved.columns):
        raise ValueError("Robustness predictions lack the selected no-age score column")
    current = pd.DataFrame({
        "row_id": validation.index.to_numpy(dtype=np.int64),
        "fraud_bool_current": validation["fraud_bool"].to_numpy(dtype=int),
        "fresh_score": fresh_scores,
    })
    aligned = current.merge(saved[["row_id", "fraud_bool", SELECTED_ABLATION]], on="row_id", validate="one_to_one")
    if len(aligned) != len(validation):
        raise ValueError("Saved selected-model predictions do not cover validation month 5")
    if not np.array_equal(aligned["fraud_bool_current"].to_numpy(), aligned["fraud_bool"].to_numpy()):
        raise ValueError("Saved selected-model predictions contain different labels")
    max_diff = float(np.max(np.abs(aligned["fresh_score"].to_numpy() - aligned[SELECTED_ABLATION].to_numpy())))
    if max_diff > 1e-12:
        raise ValueError("Selected model artifact does not reproduce saved validation predictions")
    observed = evaluate(
        aligned["fraud_bool"].to_numpy(dtype=int),
        aligned["fresh_score"].to_numpy(dtype=float),
        aligned["row_id"].to_numpy(dtype=np.int64),
        capacities=(0.01, 0.03, 0.05),
        seed=TIE_SEED,
    )
    compare_metrics(observed, decision["selected_validation"])

    print("[4/5] Copying the exact selected artifact into an immutable freeze directory...", flush=True)
    freeze_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.out_dir / freeze_id
    directory.mkdir(parents=True, exist_ok=False)
    frozen_model = directory / "selected_model.joblib"
    shutil.copy2(source_model, frozen_model)
    if sha256_file(frozen_model) != source_model_sha:
        raise ValueError("Frozen model copy hash differs from source artifact")

    source_paths = [
        ROOT / "risklens_core/model_freeze.py",
        Path(__file__).resolve(),
        ROOT / "risklens_core/robustness.py",
        ROOT / "risklens_core/metrics.py",
        ROOT / "risklens_core/features.py",
        ROOT / "requirements.txt",
    ]
    manifest = {
        "status": "frozen",
        "freeze_version": FREEZE_VERSION,
        "freeze_id": freeze_id,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "temporal-v1",
        "dataset_sha256": fingerprint,
        "feature_contract": CONTRACT_VERSION,
        "development_split": EXPECTED_SPLIT,
        "month_rows": {str(k): int(v) for k, v in counts.items()},
        "test_evaluated": False,
        "selected_model": {
            "name": SELECTED_MODEL_NAME,
            "family": "HistGradientBoostingClassifier",
            "source_artifact": str(source_model),
            "frozen_artifact": str(frozen_model),
            "sha256": source_model_sha,
            "removed_features": decision["removed_features"],
            "remaining_features": decision["remaining_features"],
            "hyperparameters": expected_params,
            "score_policy": frozen_policy()["score"],
        },
        "operating_policy": frozen_policy()["review"],
        "final_test_policy": frozen_policy()["test_policy"],
        "validation_reproduction": {
            "rows": int(len(aligned)),
            "max_absolute_score_difference": max_diff,
            "metrics": observed,
        },
        "decision": {
            "basis": decision["governance_basis"],
            "reference_full_validation": decision["full_validation"],
            "selected_no_age_validation": decision["selected_validation"],
            "no_age_minus_full_bootstrap": decision["no_age_minus_full_bootstrap"],
            "queue_overlap": decision["queue_overlap"],
            "caveat": (
                "Removing customer_age reduces direct reliance on age but does not establish fairness; correlated proxies, "
                "prevalence differences and other subgroup disparities remain possible."
            ),
        },
        "lineage": {
            "robustness_run_id": robustness["run_id"],
            "calibration_uncertainty_run_id": calibration["run_id"],
            "temporal_monitoring_run_id": monitoring["run_id"],
            "torch_comparison_run_id": torch["run_id"],
            "governance_review_run_id": governance["run_id"],
            "report_sha256": {
                "robustness": sha256_file(args.robustness_summary),
                "calibration_uncertainty": sha256_file(args.calibration_summary),
                "temporal_monitoring": sha256_file(args.monitoring_summary),
                "torch_comparison": sha256_file(args.torch_summary),
                "governance_review": sha256_file(args.governance_summary),
            },
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "processor": platform.processor(),
            "packages": {p: importlib.metadata.version(p) for p in [
                "numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"
            ]},
        },
        "limitations": [
            "Model selection and all diagnostics before this manifest used development months 0-5 only; months 6-7 remain sealed.",
            "The no-age choice is a governance/engineering decision, not evidence of fairness certification or statistical superiority.",
            "Raw model scores are not asserted to be calibrated fraud probabilities.",
            "The 3% review capacity is a project operating assumption rather than measured analyst staffing capacity.",
            "After final-test opening, this model, feature set, score policy, capacity and tie rule must not be changed in response to test results.",
        ],
    }
    write_json(directory / "model_freeze.json", manifest)
    write_json(args.manifest, manifest)

    print("[5/5] MODEL/PROTOCOL FROZEN. Months 6-7 remain unopened by this command.", flush=True)
    print(f"Saved immutable manifest: {args.manifest}", flush=True)
    print(f"Frozen model SHA-256: {source_model_sha}", flush=True)
    print("Upload model_freeze.json before running any final-test command.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        raise SystemExit(f"Model freeze stopped: {exc}") from exc
