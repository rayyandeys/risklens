import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import func, select

from test_baseline import fixture
from risklens_core.features import FEATURES
from risklens_core.persistence import (
    Case,
    QueueEntry,
    ReviewEvent,
    WorkflowRun,
    create_database_engine,
    create_session_factory,
    import_review_queue,
    initialize_database,
)
from risklens_core.review import build_review_population, selected_queue


def make_review_files(root: Path, *, run_id="run-test"):
    frame = fixture(100)
    ids = np.arange(1000, 1100, dtype=np.int64)
    scores = np.linspace(0.99, 0.01, 100)
    population = build_review_population(frame[FEATURES], scores, ids, capacity=0.05,
                                         model_name="histgb_15_leaves/full")
    queue = selected_queue(population)
    queue_path = root / "analyst_queue.csv"
    queue.to_csv(queue_path, index=False)
    summary = {
        "status": "complete",
        "run_id": run_id,
        "experiment": "review-workflow-v1",
        "protocol": "temporal-v1",
        "dataset_sha256": "a" * 64,
        "reference_model": "histgb_15_leaves/full",
        "reference_model_sha256": "b" * 64,
        "split": {"train": [0,1,2,3,4], "validation": [5], "test_reserved": [6,7]},
        "test_evaluated": False,
        "scored_month": 5,
        "review_policy": {
            "capacity": 0.05, "selected_cases": len(queue), "population_cases": len(population),
        },
        "label_separation": {"analyst_queue_contains_target": False},
        "outputs": {"analyst_queue": str(queue_path)},
    }
    summary_path = root / "review_workflow_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path, queue_path, queue


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = create_database_engine(f"sqlite:///{self.root / 'test.db'}")
        initialize_database(self.engine)
        self.Session = create_session_factory(self.engine)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_import_is_label_safe_and_idempotent(self):
        summary_path, _, queue = make_review_files(self.root)
        with self.Session() as session:
            result = import_review_queue(session, summary_path)
            self.assertFalse(result["idempotent"])
            self.assertEqual(result["selected_cases"], len(queue))
            self.assertEqual(session.scalar(select(func.count()).select_from(WorkflowRun)), 1)
            self.assertEqual(session.scalar(select(func.count()).select_from(Case)), len(queue))
            self.assertEqual(session.scalar(select(func.count()).select_from(QueueEntry)), len(queue))
            self.assertEqual(session.scalar(select(func.count()).select_from(ReviewEvent)), len(queue))
            case = session.scalars(select(Case).limit(1)).one()
            self.assertNotIn("fraud_bool", case.feature_payload)

        with self.Session() as session:
            again = import_review_queue(session, summary_path)
            self.assertTrue(again["idempotent"])
            self.assertEqual(session.scalar(select(func.count()).select_from(QueueEntry)), len(queue))

    def test_target_column_is_rejected(self):
        summary_path, queue_path, _ = make_review_files(self.root)
        queue = pd.read_csv(queue_path)
        queue["fraud_bool"] = 0
        queue.to_csv(queue_path, index=False)
        with self.Session() as session:
            with self.assertRaises(ValueError):
                import_review_queue(session, summary_path)

    def test_noncontiguous_rank_is_rejected(self):
        summary_path, queue_path, _ = make_review_files(self.root)
        queue = pd.read_csv(queue_path)
        queue.loc[0, "risk_rank"] = 99
        queue.to_csv(queue_path, index=False)
        with self.Session() as session:
            with self.assertRaises(ValueError):
                import_review_queue(session, summary_path)


if __name__ == "__main__":
    unittest.main()
