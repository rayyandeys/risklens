import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import select

from test_baseline import fixture
from risklens_core.features import FEATURES
from risklens_core.frozen_review import (
    assert_validation_reproduction,
    verify_final_test_for_review,
    verify_freeze_for_review,
)
from risklens_core.persistence import WorkflowRun, create_database_engine, create_session_factory, import_review_queue, initialize_database
from risklens_core.review import build_review_population, selected_queue


DATASET = "d" * 64
MODEL = "m" * 64


def freeze_manifest():
    return {
        "status": "frozen",
        "freeze_id": "freeze-1",
        "dataset_sha256": DATASET,
        "test_evaluated": False,
        "development_split": {"test_reserved": [6, 7]},
        "selected_model": {
            "name": "histgb_15_leaves/no_customer_age",
            "sha256": MODEL,
            "removed_features": ["customer_age"],
            "score_policy": {"policy": "raw_positive_class_predict_proba", "calibration": "none"},
        },
        "operating_policy": {"requested_capacity": 0.03, "tie_seed": 42},
        "final_test_policy": {"single_opening_after_freeze": True, "no_post_test_model_or_threshold_selection": True},
    }


def final_report():
    return {
        "status": "complete",
        "experiment": "final-holdout-v1",
        "test_evaluated": True,
        "freeze_id": "freeze-1",
        "dataset_sha256": DATASET,
        "selected_model": {"name": "histgb_15_leaves/no_customer_age", "sha256": MODEL},
        "protocol": {
            "reserved_months": [6, 7],
            "selection": "No model, feature, calibration, capacity or threshold selection is permitted from final-test results",
        },
    }


class FrozenReviewGuardTests(unittest.TestCase):
    def test_freeze_and_final_lineage_are_accepted(self):
        freeze = freeze_manifest()
        verify_freeze_for_review(freeze, DATASET)
        verify_final_test_for_review(final_report(), freeze, DATASET)

    def test_tampered_model_or_post_test_rule_is_rejected(self):
        freeze = freeze_manifest()
        bad = final_report()
        bad["selected_model"]["sha256"] = "x" * 64
        with self.assertRaises(ValueError):
            verify_final_test_for_review(bad, freeze, DATASET)
        freeze["final_test_policy"]["no_post_test_model_or_threshold_selection"] = False
        with self.assertRaises(ValueError):
            verify_freeze_for_review(freeze, DATASET)

    def test_validation_reproduction_checks_capacity_counts(self):
        reference = {
            "average_precision": 0.2, "roc_auc": 0.9, "brier_score": 0.01,
            "review_capacity": [{"requested_fraction": 0.03, "tp": 10, "fp": 20, "fn": 30, "tn": 40}],
        }
        assert_validation_reproduction(dict(reference), reference)
        changed = json.loads(json.dumps(reference))
        changed["review_capacity"][0]["tp"] = 11
        with self.assertRaises(ValueError):
            assert_validation_reproduction(changed, reference)


class FrozenReviewPersistenceTests(unittest.TestCase):
    def test_v2_import_is_new_run_and_preserves_full_feature_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = fixture(100)
            ids = np.arange(5000, 5100, dtype=np.int64)
            scores = np.linspace(0.99, 0.01, 100)
            pop = build_review_population(frame[FEATURES], scores, ids, capacity=0.03,
                                          model_name="histgb_15_leaves/no_customer_age")
            queue = selected_queue(pop)
            queue_path = root / "queue.csv"
            queue.to_csv(queue_path, index=False)
            summary = {
                "status": "complete", "run_id": "frozen-run", "experiment": "review-workflow-v2",
                "protocol": "temporal-v1-frozen-product", "dataset_sha256": DATASET,
                "reference_model": "histgb_15_leaves/no_customer_age", "reference_model_sha256": MODEL,
                "removed_features": ["customer_age"], "freeze_id": "freeze-1", "final_test_opening_id": "open-1",
                "split": {"train": [0,1,2,3,4], "validation": [5], "test_reserved": [6,7]},
                "test_evaluated": True, "scored_month": 5,
                "queue_data_scope": {"development_month_only": True, "reserved_test_rows_used": False,
                                     "reserved_test_labels_used": False, "post_test_model_selection": False},
                "review_policy": {"capacity": 0.03, "selected_cases": len(queue), "population_cases": len(pop)},
                "label_separation": {"analyst_queue_contains_target": False},
                "outputs": {"analyst_queue": str(queue_path)},
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            engine = create_database_engine(f"sqlite:///{root / 'db.sqlite'}")
            initialize_database(engine)
            Session = create_session_factory(engine)
            try:
                with Session() as session:
                    result = import_review_queue(session, summary_path)
                    self.assertFalse(result["idempotent"])
                    run = session.get(WorkflowRun, "frozen-run")
                    self.assertEqual(run.model_name, "histgb_15_leaves/no_customer_age")
            finally:
                engine.dispose()

    def test_v2_rejects_reserved_test_use(self):
        # Persistence-level v2 guard must refuse a summary that admits test-row use.
        from risklens_core.persistence import _validate_review_reference
        summary = {
            "status": "complete", "experiment": "review-workflow-v2", "reference_model": "histgb_15_leaves/no_customer_age",
            "removed_features": ["customer_age"], "freeze_id": "f", "final_test_opening_id": "o",
            "split": {"test_reserved": [6, 7]}, "test_evaluated": True, "scored_month": 5,
            "queue_data_scope": {"development_month_only": True, "reserved_test_rows_used": True,
                                 "reserved_test_labels_used": False, "post_test_model_selection": False},
            "review_policy": {"capacity": 0.03, "selected_cases": 1},
            "label_separation": {"analyst_queue_contains_target": False},
        }
        with self.assertRaises(ValueError):
            _validate_review_reference(summary)


if __name__ == "__main__":
    unittest.main()
