import unittest

import numpy as np
import pandas as pd

from test_baseline import fixture
from risklens_core.features import FEATURES
from risklens_core.metrics import evaluate
from risklens_core.review import (
    REVIEW_DECISIONS,
    apply_review_decisions,
    build_review_population,
    deterministic_ranking,
    selected_queue,
)
from risklens_core.scoring import validate_robustness_reference, verify_saved_predictions


class ReviewWorkflowTests(unittest.TestCase):
    def test_ranking_matches_capacity_metric_selection(self):
        frame = fixture(200)
        ids = np.arange(5000, 5200, dtype=np.int64)
        scores = np.linspace(0.01, 0.99, 200)
        # Introduce a tied block to exercise the seeded tie policy.
        scores[50:70] = 0.7
        population = build_review_population(frame[FEATURES], scores, ids, capacity=0.03, seed=42)
        selected = selected_queue(population)
        metrics = evaluate(frame.fraud_bool.to_numpy(), scores, ids, capacities=(0.03,), seed=42)
        self.assertEqual(len(selected), metrics["review_capacity"][0]["reviewed"])
        ranking = deterministic_ranking(scores, ids, seed=42)
        expected_ids = set(ids[ranking[:len(selected)]])
        self.assertEqual(set(selected.source_row_id), expected_ids)

    def test_ranking_is_invariant_to_input_order_for_ties(self):
        ids = np.arange(100, 120, dtype=np.int64)
        scores = np.full(20, 0.5)
        first = ids[deterministic_ranking(scores, ids, seed=7)]
        permutation = np.array([7, 2, 13, 0, 19, 4, 8, 3, 16, 5, 9, 1, 10, 11, 12, 6, 14, 15, 17, 18])
        second_ids = ids[permutation]
        second_scores = scores[permutation]
        second = second_ids[deterministic_ranking(second_scores, second_ids, seed=7)]
        np.testing.assert_array_equal(first, second)

    def test_queue_withholds_target_and_decisions_are_validated(self):
        frame = fixture(100)
        ids = np.arange(1000, 1100, dtype=np.int64)
        scores = np.linspace(0.001, 0.999, 100)
        population = build_review_population(frame, scores, ids, capacity=0.05)
        self.assertNotIn("fraud_bool", population.columns)
        queue = selected_queue(population)
        self.assertEqual(len(queue), 5)
        decisions = pd.DataFrame({
            "case_id": [queue.loc[0, "case_id"], queue.loc[1, "case_id"]],
            "decision": ["confirmed_fraud", "escalate"],
            "analyst_note": ["synthetic test", "second look"],
        })
        updated = apply_review_decisions(queue, decisions)
        self.assertEqual(updated.loc[0, "review_status"], "resolved")
        self.assertEqual(updated.loc[1, "review_status"], "escalated")
        self.assertEqual(set(REVIEW_DECISIONS), {"confirmed_fraud", "legitimate", "escalate"})
        bad = decisions.copy()
        bad.loc[0, "decision"] = "approve_loan"
        with self.assertRaises(ValueError):
            apply_review_decisions(queue, bad)

    def test_invalid_scores_ids_and_capacity_rejected(self):
        frame = fixture(20)
        ids = np.arange(20, dtype=np.int64)
        scores = np.linspace(0.1, 0.9, 20)
        with self.assertRaises(ValueError):
            build_review_population(frame, scores, ids, capacity=0)
        bad_scores = scores.copy(); bad_scores[0] = 2
        with self.assertRaises(ValueError):
            build_review_population(frame, bad_scores, ids)
        bad_ids = ids.copy(); bad_ids[1] = bad_ids[0]
        with self.assertRaises(ValueError):
            build_review_population(frame, scores, bad_ids)

    def test_reference_and_saved_prediction_guards(self):
        summary = {
            "status": "complete", "test_evaluated": False, "dataset_sha256": "abc",
            "feature_contract": "baf-base-v1",
            "split": {"train": [0,1,2,3,4], "validation": [5], "test_reserved": [6,7]},
            "reference_model": "histgb_15_leaves",
            "ablations": {"full": {"removed_features": [], "remaining_features": FEATURES, "warnings": []}},
            "bootstrap": {"histgb_15_vs_logistic_unweighted": {
                "capacity": 0.03,
                "recall_delta": {"ci_lower": 0.01},
                "average_precision_delta": {"ci_lower": 0.02},
            }},
        }
        validate_robustness_reference(summary, "abc")
        summary["bootstrap"]["histgb_15_vs_logistic_unweighted"]["recall_delta"]["ci_lower"] = -0.01
        with self.assertRaises(ValueError):
            validate_robustness_reference(summary, "abc")

        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            pd.DataFrame({"row_id": [1,2], "fraud_bool": [0,1], "full": [0.2,0.8]}).to_csv(path, index=False)
            result = verify_saved_predictions(path, [1,2], [0,1], [0.2,0.8])
            self.assertEqual(result["rows"], 2)
            with self.assertRaises(ValueError):
                verify_saved_predictions(path, [1,2], [0,1], [0.2,0.7])


if __name__ == "__main__":
    unittest.main()
