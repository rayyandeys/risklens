"""Fail-fast checks for the deployable RiskLens analyst service.

This command is read-only. It does not retrain, rescore, open the holdout, mutate
review state, or connect to a production database.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risklens_core.product_monitoring import load_monitoring_overview

REQUIRED_REPORTS = (
    "calibration_uncertainty_summary.json",
    "temporal_monitoring_summary.json",
    "model_freeze.json",
    "final_test_summary.json",
)


def _version(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.strip().lstrip("^~>=< ").split("."))
    except ValueError as exc:
        raise ValueError(f"Could not parse version: {value!r}") from exc


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    dist_index = ROOT / "frontend" / "dist" / "index.html"
    if not dist_index.is_file():
        errors.append("frontend/dist/index.html is missing; run `cd frontend && npm run build` first")

    reports = ROOT / "reports"
    for name in REQUIRED_REPORTS:
        if not (reports / name).is_file():
            errors.append(f"required runtime monitoring evidence is missing: reports/{name}")

    overview = None
    if not any("runtime monitoring evidence" in e for e in errors):
        try:
            overview = load_monitoring_overview(reports)
        except Exception as exc:  # fail-fast CLI surface
            errors.append(f"monitoring evidence chain failed validation: {exc}")

    review_summary = reports / "frozen_review_workflow_summary.json"
    if not review_summary.is_file():
        warnings.append(
            "reports/frozen_review_workflow_summary.json is missing. The web service can deploy, "
            "but you will not be able to seed the frozen analyst queue into remote PostgreSQL from this checkout."
        )
    else:
        try:
            summary = json.loads(review_summary.read_text(encoding="utf-8"))
            if summary.get("reference_model") != "histgb_15_leaves/no_customer_age":
                errors.append("frozen review summary is not the frozen no-age workflow")
            if summary.get("review_policy", {}).get("selected_cases") != 3579:
                errors.append("frozen review summary does not contain the expected 3,579-case queue")
            queue_path = ROOT / Path(summary.get("outputs", {}).get("analyst_queue", ""))
            if not queue_path.is_file():
                warnings.append(
                    "the local analyst_queue.csv referenced by frozen_review_workflow_summary.json is missing; "
                    "keep local artifacts if you plan to seed remote PostgreSQL with import_review_queue.py"
                )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"could not validate frozen review workflow summary: {exc}")

    package = ROOT / "frontend" / "package.json"
    try:
        payload = json.loads(package.read_text(encoding="utf-8"))
        vite = payload.get("devDependencies", {}).get("vite")
        if not vite or _version(vite) < (7, 3, 6):
            errors.append(f"Vite must be at least 7.3.6 for the reviewed dependency fix; found {vite!r}")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"could not verify frontend/package.json: {exc}")

    blueprint = ROOT / "render.yaml"
    if not blueprint.is_file():
        errors.append("render.yaml is missing")
    else:
        text = blueprint.read_text(encoding="utf-8")
        for needle in (
            "runtime: python",
            "preDeployCommand: python scripts/migrate_database.py",
            "healthCheckPath: /ready",
            "RISKLENS_ENV",
            "RISKLENS_ALLOWED_HOSTS",
            "DATABASE_URL",
        ):
            if needle not in text:
                errors.append(f"render.yaml is missing required deployment setting: {needle}")

    print("RiskLens deployment readiness")
    print("=" * 72)
    if overview:
        governance = overview["governance"]
        primary = overview["performance"]["primary_holdout"]
        print(f"Frozen model       : {governance['model_name']}")
        print(f"Frozen model SHA   : {governance['model_sha256']}")
        print(f"Test evaluated     : {governance['test_evaluated']}")
        print(f"Final Recall@3%    : {primary['recall_at_3pct']:.6f}")
        print(f"Final Precision@3% : {primary['precision_at_3pct']:.6f}")
    print(f"Built SPA present  : {dist_index.is_file()}")
    print(f"Patched Vite gate  : checked")
    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")
    if errors:
        print("\nFAILED:")
        for item in errors:
            print(f"- {item}")
        raise SystemExit(1)
    print("\nREADY: source/evidence assets pass the deployment preflight.")
    print("This command did not connect to or mutate any production resource.")


if __name__ == "__main__":
    main()
