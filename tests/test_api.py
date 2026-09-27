from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from test_persistence import make_review_files
from risklens_core.persistence import import_review_queue, create_database_engine, initialize_database
from risklens_api.auth import issue_credential
from risklens_api.factory import create_app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db_path = self.root / "api.db"
        engine = create_database_engine(f"sqlite:///{self.db_path}")
        initialize_database(engine)
        engine.dispose()
        self.app = create_app(f"sqlite:///{self.db_path}")
        with self.app.state.SessionLocal() as session:
            summary_path, _, self.queue = make_review_files(self.root, run_id="api-run")
            import_review_queue(session, summary_path)
            _, token = issue_credential(session, "analyst-test", "analyst")
        self.client = TestClient(self.app, headers={"Authorization": f"Bearer {token}"})

    def tearDown(self):
        self.app.state.engine.dispose()
        self.temp.cleanup()

    def test_health_summary_listing_and_detail_withhold_label(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.get("/api/v1/runs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["run_id"], "api-run")
        self.assertEqual(response.json()[0]["status_counts"], {"pending": len(self.queue)})

        cases = self.client.get("/api/v1/runs/api-run/cases?limit=2")
        self.assertEqual(cases.status_code, 200)
        self.assertEqual(len(cases.json()), 2)
        self.assertLess(cases.json()[0]["risk_rank"], cases.json()[1]["risk_rank"])

        case_id = cases.json()[0]["case_id"]
        detail = self.client.get(f"/api/v1/runs/api-run/cases/{case_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn("fraud_bool", detail.json()["features"])
        self.assertEqual(detail.json()["version"], 1)

    def test_decision_creates_history_and_stale_write_conflicts(self):
        case = self.client.get("/api/v1/runs/api-run/cases?limit=1").json()[0]
        case_id = case["case_id"]
        payload = {
            "decision": "confirmed_fraud",
            "analyst_note": "synthetic review",
            "expected_version": 1,
        }
        response = self.client.post(f"/api/v1/runs/api-run/cases/{case_id}/decision", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_status"], "resolved")
        self.assertEqual(response.json()["version"], 2)

        history = self.client.get(f"/api/v1/runs/api-run/cases/{case_id}/events")
        self.assertEqual(history.status_code, 200)
        self.assertEqual([e["event_type"] for e in history.json()], ["imported", "decision"])

        stale = self.client.post(f"/api/v1/runs/api-run/cases/{case_id}/decision", json=payload)
        self.assertEqual(stale.status_code, 409)

    def test_escalation_filter_and_validation(self):
        case = self.client.get("/api/v1/runs/api-run/cases?limit=1").json()[0]
        response = self.client.post(
            f"/api/v1/runs/api-run/cases/{case['case_id']}/decision",
            json={"decision": "escalate", "analyst_note": "second look", "expected_version": 1},
        )
        self.assertEqual(response.status_code, 200)
        escalated = self.client.get("/api/v1/runs/api-run/cases?status=escalated")
        self.assertEqual(escalated.status_code, 200)
        self.assertEqual(len(escalated.json()), 1)
        bad = self.client.post(
            f"/api/v1/runs/api-run/cases/{case['case_id']}/decision",
            json={"decision": "approve_loan", "expected_version": 2},
        )
        self.assertEqual(bad.status_code, 422)


if __name__ == "__main__":
    unittest.main()
