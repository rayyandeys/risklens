"""Guards for realigning the analyst workflow to the frozen post-governance model.

This module deliberately separates two facts:
- the immutable freeze manifest was created while months 6-7 were still sealed;
- the final-test report now records that those months were opened exactly once.

The analyst queue rebuilt here still scores development month 5 only. Final-test rows or labels
must never participate in queue construction, explanation generation, or analyst decisions.
"""
from __future__ import annotations

from typing import Any

EXPECTED_MODEL = "histgb_15_leaves/no_customer_age"
EXPECTED_CAPACITY = 0.03
EXPECTED_RESERVED = [6, 7]
EXPECTED_SCORED_MONTH = 5


def verify_freeze_for_review(freeze: dict[str, Any], dataset_sha256: str) -> None:
    if freeze.get("status") != "frozen":
        raise ValueError("Expected an immutable frozen model manifest")
    if freeze.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Freeze dataset fingerprint differs from the audited dataset")
    if freeze.get("development_split", {}).get("test_reserved") != EXPECTED_RESERVED:
        raise ValueError("Freeze reserved-month policy changed")
    if freeze.get("test_evaluated") is not False:
        raise ValueError("Freeze manifest must record the pre-opening state")
    selected = freeze.get("selected_model", {})
    if selected.get("name") != EXPECTED_MODEL:
        raise ValueError("Unexpected frozen model for analyst workflow")
    if selected.get("removed_features") != ["customer_age"]:
        raise ValueError("Frozen model feature-removal policy changed")
    score_policy = selected.get("score_policy", {})
    if score_policy.get("policy") != "raw_positive_class_predict_proba" or score_policy.get("calibration") != "none":
        raise ValueError("Frozen score policy changed")
    operating = freeze.get("operating_policy", {})
    if abs(float(operating.get("requested_capacity", -1)) - EXPECTED_CAPACITY) > 1e-15:
        raise ValueError("Frozen analyst capacity changed")
    if int(operating.get("tie_seed", -1)) != 42:
        raise ValueError("Frozen tie seed changed")
    final_policy = freeze.get("final_test_policy", {})
    if final_policy.get("single_opening_after_freeze") is not True or final_policy.get("no_post_test_model_or_threshold_selection") is not True:
        raise ValueError("Freeze post-test governance rule changed")


def verify_final_test_for_review(final_test: dict[str, Any], freeze: dict[str, Any], dataset_sha256: str) -> None:
    if final_test.get("status") != "complete" or final_test.get("experiment") != "final-holdout-v1":
        raise ValueError("Expected the completed one-time final holdout report")
    if final_test.get("test_evaluated") is not True:
        raise ValueError("Final-test report must record the opened holdout")
    if final_test.get("freeze_id") != freeze.get("freeze_id"):
        raise ValueError("Final-test report does not descend from the immutable freeze")
    if final_test.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Final-test dataset fingerprint differs from the audited dataset")
    selected = final_test.get("selected_model", {})
    frozen = freeze.get("selected_model", {})
    if selected.get("name") != frozen.get("name") or selected.get("sha256") != frozen.get("sha256"):
        raise ValueError("Final-test model provenance differs from the frozen model")
    if final_test.get("protocol", {}).get("reserved_months") != EXPECTED_RESERVED:
        raise ValueError("Final-test reserved-month policy changed")
    if final_test.get("protocol", {}).get("selection") != "No model, feature, calibration, capacity or threshold selection is permitted from final-test results":
        raise ValueError("Final-test no-selection rule changed")


def assert_validation_reproduction(current: dict[str, Any], frozen: dict[str, Any]) -> None:
    """Require exact capacity counts and effectively exact continuous metrics on month 5."""
    for key in ("average_precision", "roc_auc", "brier_score"):
        if abs(float(current[key]) - float(frozen[key])) > 1e-12:
            raise ValueError(f"Fresh frozen-model scoring does not reproduce validation {key}")
    old_caps = frozen.get("review_capacity", [])
    new_caps = current.get("review_capacity", [])
    if len(old_caps) != len(new_caps):
        raise ValueError("Validation review-capacity definitions changed")
    for old, new in zip(old_caps, new_caps):
        if old.get("requested_fraction") != new.get("requested_fraction"):
            raise ValueError("Validation review-capacity fractions changed")
        for key in ("tp", "fp", "fn", "tn"):
            if int(old[key]) != int(new[key]):
                raise ValueError("Frozen validation capacity counts do not reproduce")
