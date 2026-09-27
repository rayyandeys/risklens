import copy
import unittest

import numpy as np

from risklens_core.features import CONTRACT_VERSION, FEATURES
from risklens_core.final_test import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_FREEZE_ID,
    EXPECTED_MODEL_SHA256,
    operational_aggregate,
    ranking_masks,
    validate_freeze_manifest,
)


def freeze_manifest():
    return {
        "status": "frozen",
        "freeze_version": 1,
        "freeze_id": EXPECTED_FREEZE_ID,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "feature_contract": CONTRACT_VERSION,
        "test_evaluated": False,
        "development_split": {"train": [0, 1, 2, 3, 4], "validation": [5], "test_reserved": [6, 7]},
        "selected_model": {
            "name": "histgb_15_leaves/no_customer_age",
            "sha256": EXPECTED_MODEL_SHA256,
            "removed_features": ["customer_age"],
            "remaining_features": [c for c in FEATURES if c != "customer_age"],
            "score_policy": {"policy": "raw_positive_class_predict_proba", "calibration": "none"},
        },
        "operating_policy": {
            "requested_capacity": 0.03,
            "budget_rule": "floor(number_of_scored_rows * requested_capacity)",
            "tie_seed": 42,
        },
        "final_test_policy": {
            "reserved_months": [6, 7],
            "single_opening_after_freeze": True,
            "no_post_test_model_or_threshold_selection": True,
        },
    }


def month_metric(rows, positives, reviewed, tp):
    return {
        "rows": rows,
        "fraud_rows": positives,
        "review_capacity": [{
            "requested_fraction": 0.03,
            "reviewed": reviewed,
            "tp": tp,
            "fp": reviewed - tp,
            "fn": positives - tp,
            "tn": rows - positives - (reviewed - tp),
        }],
    }


class FinalTestProtocolTests(unittest.TestCase):
    def test_exact_frozen_manifest_is_accepted(self):
        selected = validate_freeze_manifest(freeze_manifest())
        self.assertEqual(selected["sha256"], EXPECTED_MODEL_SHA256)
        self.assertNotIn("customer_age", selected["remaining_features"])

    def test_any_opened_or_changed_freeze_is_rejected(self):
        for key, value in [
            ("test_evaluated", True),
            ("freeze_id", "other"),
        ]:
            manifest = freeze_manifest()
            manifest[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_freeze_manifest(manifest)
        manifest = freeze_manifest()
        manifest["selected_model"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_freeze_manifest(manifest)
        manifest = freeze_manifest()
        manifest["operating_policy"]["requested_capacity"] = 0.05
        with self.assertRaises(ValueError):
            validate_freeze_manifest(manifest)

    def test_operational_aggregate_uses_independent_monthly_budgets(self):
        result = operational_aggregate([
            month_metric(100, 10, 3, 2),
            month_metric(200, 20, 6, 3),
        ])
        self.assertEqual(result["rows"], 300)
        self.assertEqual(result["reviewed"], 9)
        self.assertEqual(result["tp"], 5)
        self.assertAlmostEqual(result["recall"], 5 / 30)
        self.assertAlmostEqual(result["precision"], 5 / 9)

    def test_ranking_masks_are_input_order_invariant_with_ties(self):
        scores = np.array([0.9, 0.5, 0.5, 0.1])
        ids = np.array([40, 10, 30, 20], dtype=np.int64)
        first = ranking_masks(scores, ids, capacities=(0.5,), seed=42)[0.5]
        perm = np.array([2, 0, 3, 1])
        second = ranking_masks(scores[perm], ids[perm], capacities=(0.5,), seed=42)[0.5]
        selected_first = set(ids[first].tolist())
        selected_second = set(ids[perm][second].tolist())
        self.assertEqual(selected_first, selected_second)

    def test_calibration_or_reserved_month_change_is_rejected(self):
        manifest = freeze_manifest()
        manifest["selected_model"]["score_policy"]["calibration"] = "sigmoid"
        with self.assertRaises(ValueError):
            validate_freeze_manifest(manifest)
        manifest = freeze_manifest()
        manifest["final_test_policy"]["reserved_months"] = [7]
        with self.assertRaises(ValueError):
            validate_freeze_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
