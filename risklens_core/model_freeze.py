"""Immutable development model-freeze policy for RiskLens.

This module records the reviewed governance decision without touching the sealed
months 6-7.  It deliberately separates model selection from final-test scoring.
"""
from __future__ import annotations

from copy import deepcopy

from .features import CONTRACT_VERSION
from .robustness import ablated_features
from .scoring import EXPECTED_SPLIT

FREEZE_VERSION = 1
SELECTED_ABLATION = "no_customer_age"
SELECTED_MODEL_NAME = "histgb_15_leaves/no_customer_age"
REMOVED_FEATURES = ("customer_age",)
CAPACITY = 0.03
TIE_SEED = 42
SCORE_POLICY = "raw_positive_class_predict_proba"
CALIBRATION_POLICY = "none"


def capacity_entry(metrics: dict, fraction: float = CAPACITY) -> dict:
    for entry in metrics.get("review_capacity", []):
        if abs(float(entry.get("requested_fraction", -1)) - fraction) < 1e-12:
            return entry
    raise ValueError(f"Missing review-capacity entry for {fraction}")


def validate_governance_for_freeze(summary: dict) -> dict:
    """Validate the reviewed development evidence and return a compact decision record.

    The no-age variant is selected as a governance choice: it removes direct use of
    customer_age while showing no observed aggregate development loss at the frozen
    3% operating point.  The bootstrap is *not* interpreted as proving superiority.
    """
    if summary.get("status") != "complete" or summary.get("experiment") != "governance-review-v1":
        raise ValueError("Expected a completed governance-review-v1 report")
    if summary.get("test_evaluated") is not False:
        raise ValueError("Cannot freeze after final-test evaluation has been opened")
    if summary.get("feature_contract") != CONTRACT_VERSION or summary.get("split") != EXPECTED_SPLIT:
        raise ValueError("Governance report uses a different feature contract or temporal split")
    if summary.get("reference_model") != "histgb_15_leaves/full":
        raise ValueError("Unexpected governance reference model")
    if summary.get("challenger") != SELECTED_MODEL_NAME:
        raise ValueError("Governance report lacks the reviewed no-age challenger")

    overall = summary.get("overall_metrics", {})
    full = overall.get("full")
    no_age = overall.get("no_customer_age")
    if not full or not no_age:
        raise ValueError("Governance report lacks full/no-age metrics")
    full_3 = capacity_entry(full)
    no_age_3 = capacity_entry(no_age)

    # We are not claiming statistical superiority.  But the governance switch must
    # not knowingly trade away the predeclared operational point estimate or AP.
    if float(no_age_3["recall"]) + 1e-12 < float(full_3["recall"]):
        raise ValueError("No-age model has lower development Recall@3%; freeze requires explicit re-review")
    if float(no_age["average_precision"]) + 1e-12 < float(full["average_precision"]):
        raise ValueError("No-age model has lower development AP; freeze requires explicit re-review")

    bootstrap = summary.get("no_age_minus_full_bootstrap", {})
    if bootstrap.get("capacity") != CAPACITY or int(bootstrap.get("repeats", 0)) < 100:
        raise ValueError("No-age/full paired bootstrap evidence is missing or incompatible")

    policy = summary.get("global_queue_policy", {})
    if policy.get("capacity") != CAPACITY or policy.get("tie_seed") != TIE_SEED:
        raise ValueError("Governance queue policy differs from the intended frozen operating policy")

    return {
        "selected_model": SELECTED_MODEL_NAME,
        "removed_features": list(REMOVED_FEATURES),
        "remaining_features": ablated_features(SELECTED_ABLATION),
        "governance_basis": (
            "Remove customer_age as a direct model input before final-test opening because the reviewed "
            "no-age challenger has no observed aggregate development loss at the frozen 3% operating point "
            "and has slightly higher AP/Brier point estimates. This is a governance choice, not a claim of "
            "statistical superiority or fairness certification."
        ),
        "full_validation": deepcopy(full),
        "selected_validation": deepcopy(no_age),
        "no_age_minus_full_bootstrap": deepcopy(bootstrap),
        "queue_overlap": deepcopy(policy),
        "subgroup_diagnostics": deepcopy(summary.get("subgroups", {})),
    }


def frozen_policy() -> dict:
    return {
        "feature_contract": CONTRACT_VERSION,
        "selected_ablation": SELECTED_ABLATION,
        "selected_model": SELECTED_MODEL_NAME,
        "removed_features": list(REMOVED_FEATURES),
        "remaining_features": ablated_features(SELECTED_ABLATION),
        "score": {
            "policy": SCORE_POLICY,
            "calibration": CALIBRATION_POLICY,
            "display_semantics": "model score; not asserted to be a calibrated fraud probability",
        },
        "review": {
            "requested_capacity": CAPACITY,
            "budget_rule": "floor(number_of_scored_rows * requested_capacity)",
            "tie_breaking": "seeded label-free permutation assigned in ascending original row_id order",
            "tie_seed": TIE_SEED,
        },
        "test_policy": {
            "reserved_months": [6, 7],
            "single_opening_after_freeze": True,
            "no_post_test_model_or_threshold_selection": True,
        },
    }
