import copy
import unittest

from risklens_core.features import CONTRACT_VERSION
from risklens_core.model_freeze import (
    CAPACITY,
    SELECTED_MODEL_NAME,
    frozen_policy,
    validate_governance_for_freeze,
)
from risklens_core.scoring import EXPECTED_SPLIT


def metric(ap, brier, recall, tp):
    return {
        "average_precision": ap,
        "roc_auc": 0.89,
        "brier_score": brier,
        "review_capacity": [
            {"requested_fraction": 0.01, "recall": 0.2, "tp": 10},
            {"requested_fraction": 0.03, "recall": recall, "tp": tp},
            {"requested_fraction": 0.05, "recall": 0.5, "tp": 20},
        ],
    }


def governance():
    return {
        "status": "complete",
        "experiment": "governance-review-v1",
        "test_evaluated": False,
        "feature_contract": CONTRACT_VERSION,
        "split": copy.deepcopy(EXPECTED_SPLIT),
        "reference_model": "histgb_15_leaves/full",
        "challenger": SELECTED_MODEL_NAME,
        "overall_metrics": {
            "full": metric(0.18, 0.011, 0.420, 594),
            "no_customer_age": metric(0.19, 0.0109, 0.421, 595),
        },
        "no_age_minus_full_bootstrap": {
            "repeats": 1000,
            "capacity": CAPACITY,
            "recall_delta": {"ci_lower": -0.01, "ci_upper": 0.01},
            "average_precision_delta": {"ci_lower": -0.001, "ci_upper": 0.008},
        },
        "global_queue_policy": {"capacity": CAPACITY, "tie_seed": 42, "jaccard": 0.77},
        "subgroups": {},
    }


class ModelFreezeTests(unittest.TestCase):
    def test_review_selects_no_age_and_policy_is_explicit(self):
        decision = validate_governance_for_freeze(governance())
        self.assertEqual(decision["selected_model"], SELECTED_MODEL_NAME)
        self.assertNotIn("customer_age", decision["remaining_features"])
        policy = frozen_policy()
        self.assertEqual(policy["review"]["requested_capacity"], 0.03)
        self.assertEqual(policy["review"]["tie_seed"], 42)
        self.assertEqual(policy["test_policy"]["reserved_months"], [6, 7])
        self.assertEqual(policy["score"]["calibration"], "none")

    def test_final_test_access_in_governance_is_rejected(self):
        report = governance()
        report["test_evaluated"] = True
        with self.assertRaises(ValueError):
            validate_governance_for_freeze(report)

    def test_aggregate_regression_requires_rereview(self):
        report = governance()
        report["overall_metrics"]["no_customer_age"]["review_capacity"][1]["recall"] = 0.40
        with self.assertRaises(ValueError):
            validate_governance_for_freeze(report)

    def test_wrong_queue_policy_is_rejected(self):
        report = governance()
        report["global_queue_policy"]["capacity"] = 0.05
        with self.assertRaises(ValueError):
            validate_governance_for_freeze(report)


if __name__ == "__main__":
    unittest.main()
