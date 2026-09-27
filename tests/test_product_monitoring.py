import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from risklens_api.auth import issue_credential
from risklens_api.factory import create_app
from risklens_core.baseline import sha256_file
from risklens_core.persistence import create_database_engine, initialize_database
from risklens_core.product_monitoring import MonitoringDataError, load_monitoring_overview


def metric(rows=100, fraud=10, recall=0.4, precision=0.2, ap=0.18, auc=0.89, brier=0.01):
    reviewed = 3
    tp = 4 if fraud == 10 else max(1, int(fraud * recall))
    return {
        "rows": rows,
        "fraud_rows": fraud,
        "average_precision": ap,
        "roc_auc": auc,
        "brier_score": brier,
        "review_capacity": [{
            "requested_fraction": 0.03,
            "reviewed": reviewed,
            "tp": tp,
            "precision": precision,
            "recall": recall,
        }],
    }


def write_reports(root: Path):
    dataset = "d" * 64
    contract = "baf-base-v1"
    model_hash = "m" * 64
    calibration = {
        "status": "complete", "experiment": "calibration-uncertainty-v1",
        "dataset_sha256": dataset, "feature_contract": contract,
        "reference_model": "histgb_15_leaves/full",
        "calibration": {"selected_method": "raw", "methods": {"raw": {"brier_score": 0.01, "ece": 0.003}}},
        "stability": {"summary": {
            "repeats": 12, "capacity": 0.03,
            "queue_jaccard_vs_reference": {"mean": 0.68, "min": 0.66, "max": 0.70},
            "reference_queue_retention": {"mean": 0.81},
            "spearman_score_rank_correlation": {"mean": 0.96},
            "boundary_unstable_cases": 100,
        }},
    }
    temporal = {
        "status": "complete", "experiment": "temporal-monitoring-v1",
        "dataset_sha256": dataset, "feature_contract": contract,
        "reference_model": "histgb_15_leaves/full",
        "latest_monitoring_snapshot": {
            "status": "alert", "as_of_month": 5,
            "reference_population": "training months 0-4 pooled", "target_population": "validation month 5",
            "feature_drift": {
                "status": "alert", "counts_by_severity": {"normal": 24, "watch": 6, "alert": 4},
                "max_psi": 3.7, "features_at_watch_or_alert": 10,
                "top_features": [{"feature": "velocity_4w", "kind": "numeric", "psi": 3.7,
                                  "ks_statistic": 0.8, "reference_missing_rate": 0.0,
                                  "target_missing_rate": 0.0, "missing_rate_delta": 0.0,
                                  "unseen_category_rate": None, "severity": "alert"}],
            },
            "score_drift": {"psi": 0.006, "ks_statistic": 0.03, "reference_mean": 0.01,
                            "target_mean": 0.009, "reference_p95": 0.04, "target_p95": 0.03,
                            "severity": "normal"},
            "alerts": [{"type": "feature_drift"}],
            "thresholds": {"note": "heuristic"}, "interpretation": "triage only",
        },
        "month_over_month": [{"reference_month": 4, "target_month": 5, "feature_status": "alert",
                               "max_feature_psi": 1.6, "features_at_watch_or_alert": 4,
                               "score_psi": 0.015, "score_ks_statistic": 0.04,
                               "score_status": "normal", "overall_status": "alert"}],
        "limitations": ["drift limitation"],
    }
    freeze = {
        "status": "frozen", "freeze_version": 1, "freeze_id": "freeze-1",
        "dataset_sha256": dataset, "feature_contract": contract,
        "selected_model": {
            "name": "histgb_15_leaves/no_customer_age", "sha256": model_hash,
            "removed_features": ["customer_age"],
            "score_policy": {"policy": "raw_positive_class_predict_proba", "calibration": "none",
                             "display_semantics": "model score"},
        },
        "operating_policy": {"requested_capacity": 0.03, "tie_seed": 42},
        "validation_reproduction": {"metrics": metric()},
        "decision": {"basis": "remove age", "caveat": "not fairness certification"},
    }
    (root / "calibration_uncertainty_summary.json").write_text(json.dumps(calibration), encoding="utf-8")
    (root / "temporal_monitoring_summary.json").write_text(json.dumps(temporal), encoding="utf-8")
    freeze_path = root / "model_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    final = {
        "status": "complete", "experiment": "final-holdout-v1", "test_evaluated": True,
        "opening_id": "open-1", "freeze_id": "freeze-1", "freeze_manifest_sha256": sha256_file(freeze_path),
        "dataset_sha256": dataset, "feature_contract": contract,
        "selected_model": {"name": "histgb_15_leaves/no_customer_age", "sha256": model_hash},
        "primary_operational_result": {"rows": 200, "fraud_rows": 20, "reviewed": 6, "tp": 8,
                                       "recall": 0.4, "precision": 0.2, "random_expected_tp": 0.6},
        "secondary_pooled_metrics": {"average_precision": 0.18, "roc_auc": 0.88, "brier_score": 0.012},
        "per_month": [{"month": 6, "metrics": metric()}, {"month": 7, "metrics": metric()}],
        "generalization_vs_validation": {"note": "descriptive", "recall_at_3pct_delta": -0.02,
                                          "precision_at_3pct_delta": 0.01, "average_precision_delta": -0.01,
                                          "roc_auc_delta": -0.01, "brier_score_delta": 0.002},
        "protocol": {"selection": "no post-test selection"},
        "limitations": ["holdout limitation"],
    }
    (root / "final_test_summary.json").write_text(json.dumps(final), encoding="utf-8")


class ProductMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reports = self.root / "reports"
        self.reports.mkdir()
        write_reports(self.reports)

    def tearDown(self):
        self.temp.cleanup()

    def test_projection_keeps_sources_distinct(self):
        result = load_monitoring_overview(self.reports)
        self.assertEqual(result["status"], "alert")
        self.assertEqual(result["governance"]["model_name"], "histgb_15_leaves/no_customer_age")
        self.assertEqual(result["drift"]["score"]["reference_model"], "histgb_15_leaves/full")
        self.assertEqual([p["month"] for p in result["performance"]["timeline"]], [5, 6, 7])
        self.assertEqual(result["performance"]["timeline"][0]["split"], "validation")
        self.assertEqual(result["performance"]["timeline"][1]["split"], "final_holdout")
        self.assertEqual(result["limitations"], ["drift limitation", "holdout limitation"])

    def test_freeze_tampering_is_rejected(self):
        path = self.reports / "model_freeze.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["selected_model"]["sha256"] = "x" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(MonitoringDataError):
            load_monitoring_overview(self.reports)

    def test_missing_report_is_rejected(self):
        (self.reports / "temporal_monitoring_summary.json").unlink()
        with self.assertRaises(MonitoringDataError):
            load_monitoring_overview(self.reports)

    def test_authenticated_api_and_missing_evidence_status(self):
        db = self.root / "api.db"
        engine = create_database_engine(f"sqlite:///{db}")
        initialize_database(engine)
        engine.dispose()
        app = create_app(f"sqlite:///{db}", reports_dir=self.reports)
        with app.state.SessionLocal() as session:
            _, token = issue_credential(session, "monitor-test", "viewer")
        client = TestClient(app, headers={"Authorization": f"Bearer {token}"})
        try:
            response = client.get("/api/v1/monitoring/overview")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["governance"]["test_evaluated"])
            self.assertEqual(client.get("/api/v1/monitoring/overview", headers={"Authorization": "Bearer bad"}).status_code, 401)
            (self.reports / "final_test_summary.json").unlink()
            self.assertEqual(client.get("/api/v1/monitoring/overview").status_code, 503)
        finally:
            app.state.engine.dispose()


if __name__ == "__main__":
    unittest.main()
