from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from threadpoolctl import threadpool_limits
from alembic import command

from risklens_core.baseline import sha256_file
from risklens_core.features import FEATURES, CATEGORICAL, FeatureCleaner
from risklens_core.explanations import explain_case, feature_hash, load_training_background
from risklens_core.explanation_store import CaseExplanation, ExplanationPayload, case_frame, save_snapshot
from risklens_core.migrations import ROOT, config_for, upgrade_database
from risklens_core.persistence import Case, QueueEntry, WorkflowRun, create_database_engine, create_session_factory, import_review_queue
from risklens_core.review import build_review_population, selected_queue
from risklens_core.robustness import make_ablation_pipeline
from risklens_api.factory import create_app
from risklens_api.auth import issue_credential, authenticate
from test_baseline import fixture


class AdditiveModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, X):
        cleaned = FeatureCleaner().transform(X)
        score = (0.1 + 0.2 * cleaned.income + 0.1 * cleaned.email_is_free +
                 0.3 * cleaned.prev_address_months_count__missing).to_numpy()
        return np.c_[1-score, score]


class InteractionModel:
    classes_ = np.array([0, 1])
    def predict_proba(self, X):
        score = (0.1 + 0.4 * X.income * X.email_is_free).to_numpy(dtype=float)
        return np.c_[1-score, score]


def zeros(n=1):
    return pd.DataFrame({name: ["A" if name in CATEGORICAL else 0] * n for name in FEATURES})


def make_project(root):
    frame = fixture(800)
    frame["fraud_bool"] = (frame.income > 0.5).astype(int)
    csv = root / "Base.csv"
    frame.to_csv(csv, index=False)
    audit = {"sha256": sha256_file(csv), "month_rows": {str(m): int((frame.month == m).sum()) for m in range(8)},
             "development_target_summary": {"train": {"rows": int((frame.month <= 4).sum())}}}
    audit_path = root / "audit.json"
    audit_path.write_text(json.dumps(audit))
    train = frame[frame.month <= 4]
    validation = frame[frame.month == 5]
    model = make_ablation_pipeline("full", max_iter=8)
    with threadpool_limits(limits=1):
        model.fit(train[FEATURES], train.fraud_bool)
        scores = model.predict_proba(validation[FEATURES])[:, 1]
    model_path = root / "full.joblib"
    joblib.dump(model, model_path)
    queue = selected_queue(build_review_population(validation[FEATURES], scores, validation.index.to_numpy(),
                                                   capacity=.05, model_name="histgb_15_leaves/full"))
    queue_path = root / "queue.csv"
    queue.to_csv(queue_path, index=False)
    summary = {"status": "complete", "experiment": "review-workflow-v1", "protocol": "temporal-v1",
               "run_id": "explain-run", "dataset_sha256": audit["sha256"], "reference_model": "histgb_15_leaves/full",
               "reference_model_sha256": sha256_file(model_path), "reference_model_artifact": str(model_path),
               "test_evaluated": False, "scored_month": 5, "split": {"test_reserved": [6, 7]},
               "label_separation": {"analyst_queue_contains_target": False},
               "review_policy": {"capacity": .05, "selected_cases": len(queue), "population_cases": len(validation)},
               "outputs": {"analyst_queue": str(queue_path)}}
    summary_path = root / "review.json"
    summary_path.write_text(json.dumps(summary))
    return csv, audit_path, summary_path, queue, model, train[FEATURES].iloc[:4]


class ExplanationMathTests(unittest.TestCase):
    def test_additive_contributions_missing_flag_and_repeatability(self):
        case, background = zeros(), zeros(4)
        case.loc[0, ["income", "email_is_free", "prev_address_months_count"]] = [1, 1, -1]
        result = explain_case(AdditiveModel(), case, background)
        self.assertEqual(result, explain_case(AdditiveModel(), case, background))
        contributions = {r["feature"]: r for r in result["features"]}
        for name, expected in [("income", .2), ("email_is_free", .1), ("prev_address_months_count", .3)]:
            self.assertAlmostEqual(contributions[name]["contribution"], expected)
        self.assertTrue(contributions["prev_address_months_count"]["missing_after_cleaning"])
        self.assertAlmostEqual(result["reference_score"], .1)
        self.assertAlmostEqual(result["model_score"], .7)
        self.assertLess(result["reconstruction_error"], 1e-12)
        self.assertEqual(ExplanationPayload.model_validate(result).method, "paired-permutation-v1")

    def test_forward_reverse_splits_two_feature_interaction(self):
        case, background = zeros(), zeros(4)
        case.loc[0, ["income", "email_is_free"]] = 1
        result = explain_case(InteractionModel(), case, background, seed=19)
        contributions = {r["feature"]: r["contribution"] for r in result["features"]}
        self.assertAlmostEqual(contributions["income"], .2)
        self.assertAlmostEqual(contributions["email_is_free"], .2)
        self.assertEqual(sum(abs(v) for k,v in contributions.items() if k not in ("income", "email_is_free")), 0)

    def test_labels_and_invalid_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            explain_case(AdditiveModel(), zeros().assign(fraud_bool=1), zeros(2))
        with self.assertRaises(ValueError):
            explain_case(AdditiveModel(), zeros().assign(income=100), zeros(2))
        class NaNModel(AdditiveModel):
            def predict_proba(self, X):
                return np.full((len(X), 2), np.nan)
        with self.assertRaises(ValueError):
            explain_case(NaNModel(), zeros(), zeros(2))
        with self.assertRaises(ValueError):
            explain_case(AdditiveModel(), zeros(), zeros(1))

    def test_training_background_excludes_validation_holdout_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = fixture(160)
            path = root / 'data.csv'
            frame.to_csv(path, index=False)
            audit = {"sha256": sha256_file(path), "month_rows": {str(m):20 for m in range(8)},
                     "development_target_summary": {"train": {"rows": 100}}}
            with patch('risklens_core.explanations.pd.read_csv', wraps=pd.read_csv) as reader:
                background, info = load_training_background(path, audit, size=8)
                self.assertNotIn("fraud_bool", reader.call_args.kwargs["usecols"])
            self.assertTrue(frame.loc[background.index, "month"].le(4).all())
            self.assertEqual(set(background.columns), set(FEATURES))
            again, info2 = load_training_background(path, audit, size=8)
            pd.testing.assert_frame_equal(background, again)
            self.assertEqual(info, info2)
            frame.loc[frame.month >= 5, "income"] = 999
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_training_background(path, audit, size=8)
            audit["sha256"] = sha256_file(path)
            after, _ = load_training_background(path, audit, size=8)
            pd.testing.assert_frame_equal(background, after)


class ExplanationStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv, self.audit, self.summary, self.queue, self.model, self.background = make_project(self.root)
        self.url = f"sqlite:///{self.root / 'app.db'}"
        self.engine = create_database_engine(self.url)
        upgrade_database(self.engine)
        self.sessions = create_session_factory(self.engine)
        with self.sessions() as session:
            import_review_queue(session, self.summary)
            _, self.token = issue_credential(session, "viewer-test", "viewer")
        self.app = create_app(self.url)
        self.client = TestClient(self.app)
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.case_id = self.queue.iloc[0].case_id
        self.path = f"/api/v1/runs/explain-run/cases/{self.case_id}/explanation"

    def tearDown(self):
        self.client.close()
        self.app.state.engine.dispose()
        self.engine.dispose()
        self.temp.cleanup()

    def sample(self, session):
        entry = session.scalar(select(QueueEntry).where(QueueEntry.case_id == self.case_id))
        case = session.get(Case, self.case_id)
        run = session.get(WorkflowRun, 'explain-run')
        with threadpool_limits(limits=1):
            payload = explain_case(self.model, case_frame(case), self.background)
        return entry, case, run, payload

    def test_snapshot_api_auth_coverage_and_idempotency(self):
        self.assertEqual(self.client.get(self.path).status_code, 401)
        self.assertEqual(self.client.get('/api/v1/runs/explain-run/explanations/summary').status_code, 401)
        self.assertEqual(self.client.get(self.path, headers=self.headers).status_code, 404)
        with self.sessions() as session:
            entry, case, run, payload = self.sample(session)
            first, created = save_snapshot(session, entry, case, run, payload, config_sha256='a'*64, background_sha256='b'*64)
            second, inserted = save_snapshot(session, entry, case, run, payload, config_sha256='a'*64, background_sha256='b'*64)
            self.assertTrue(created)
            self.assertFalse(inserted)
            self.assertEqual(first.id, second.id)
        result = self.client.get(self.path, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertNotIn('fraud_bool', result.text)
        self.assertEqual(len(result.json()['explanation']['features']), len(FEATURES))
        coverage = self.client.get('/api/v1/runs/explain-run/explanations/summary', headers=self.headers).json()
        self.assertEqual(coverage['explained_cases'], 1)
        self.assertEqual(coverage['remaining_cases'], len(self.queue)-1)

    def test_mismatch_and_tampered_payload_are_rejected(self):
        with self.sessions() as session:
            entry, case, run, payload = self.sample(session)
            wrong = deepcopy(payload)
            wrong['features'][0]['contribution'] += .1
            with self.assertRaises(ValueError):
                save_snapshot(session, entry, case, run, wrong, config_sha256='a'*64, background_sha256='b'*64)
            entry.risk_score += .01
            with self.assertRaises(ValueError):
                save_snapshot(session, entry, case, run, payload, config_sha256='a'*64, background_sha256='b'*64)
            session.rollback()
        with self.sessions() as session:
            entry, case, run, payload = self.sample(session)
            record, _ = save_snapshot(session, entry, case, run, payload, config_sha256='a'*64, background_sha256='b'*64)
            record.input_sha256 = '0'*64
            session.commit()
        self.assertEqual(self.client.get(self.path, headers=self.headers).status_code, 409)

    def test_cli_end_to_end_and_resume(self):
        report = self.root / 'explanations.json'
        cmd = [sys.executable, str(ROOT/'scripts/build_explanations.py'), '--csv', str(self.csv),
               '--audit', str(self.audit), '--review-summary', str(self.summary), '--limit', '2',
               '--background-size', '4', '--out-dir', str(self.root/'out'), '--summary', str(report)]
        for expected_new, expected_reused in [(2,0),(0,2)]:
            result = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, DATABASE_URL=self.url), text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data['inserted'], expected_new)
            self.assertEqual(data['reused'], expected_reused)
            self.assertEqual(data['covered_cases_for_configuration'], 2)
            self.assertFalse(data['test_evaluated'])
            self.assertLess(data['max_reconstruction_error'], 1e-10)
        self.assertEqual(self.client.get(self.path, headers=self.headers).status_code, 200)
        with self.sessions() as session:
            run = session.get(WorkflowRun, 'explain-run')
            model_path = Path(json.loads(self.summary.read_text())['reference_model_artifact'])
            model_path.write_bytes(b'changed')
        result = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, DATABASE_URL=self.url), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Model artifact hash differs', result.stderr)

    def test_v2_upgrade_preserves_credentials_and_cases(self):
        url = f"sqlite:///{self.root / 'v2.db'}"
        engine = create_database_engine(url)
        try:
            with engine.begin() as conn:
                command.upgrade(config_for(conn), '0002_auth')
            sessions = create_session_factory(engine)
            with sessions() as session:
                import_review_queue(session, self.summary)
                _, token = issue_credential(session, 'prior-identity', 'analyst')
            with engine.connect() as conn:
                before = {name:conn.execute(text(f'SELECT * FROM {name}')).all() for name in
                          ('api_credentials','cases','workflow_runs','queue_entries','review_events')}
            upgrade_database(engine)
            with engine.connect() as conn:
                after = {name:conn.execute(text(f'SELECT * FROM {name}')).all() for name in before}
            self.assertEqual(before, after)
            with sessions() as session:
                self.assertIsNotNone(authenticate(session, token))
        finally:
            engine.dispose()
