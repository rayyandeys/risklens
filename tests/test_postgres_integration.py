"""Opt-in real PostgreSQL gate; uses a unique temporary schema, never public tables."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
import uuid
import numpy as np
from risklens_core.features import FEATURES
from risklens_core.explanations import explain_case
from risklens_core.explanation_store import case_frame, save_snapshot
from risklens_core.persistence import QueueEntry, Case, WorkflowRun
from sqlalchemy import select
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from risklens_core.database_config import database_url
from risklens_core.persistence import Base, create_database_engine, create_session_factory, import_review_queue
from risklens_core.migrations import upgrade_database
from risklens_api.auth import issue_credential, ApiCredential
from risklens_api.factory import create_app
from test_persistence import make_review_files


@unittest.skipUnless(os.getenv("RISKLENS_TEST_POSTGRES_URL"), "Set RISKLENS_TEST_POSTGRES_URL for live PostgreSQL gate")
class PostgresIntegrationTests(unittest.TestCase):
    def test_migrate_import_auth_concurrency_and_revocation(self):
        url = make_url(database_url(os.environ["RISKLENS_TEST_POSTGRES_URL"]))
        if url.drivername != "postgresql+psycopg":
            self.fail("RISKLENS_TEST_POSTGRES_URL must point to PostgreSQL")
        # The caller provides a dedicated test database whose role may create schemas.
        schema = "risklens_test_" + uuid.uuid4().hex
        admin = create_database_engine(url.render_as_string(hide_password=False))
        engine = None
        app = None
        try:
            with admin.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            isolated = url.update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
            engine = create_database_engine(isolated)
            upgrade_database(engine)
            with engine.connect() as conn:
                self.assertEqual(compare_metadata(MigrationContext.configure(conn), Base.metadata), [])
            with tempfile.TemporaryDirectory() as tmp:
                summary, _, queue = make_review_files(Path(tmp), run_id="pg-run")
                with create_session_factory(engine)() as session:
                    self.assertFalse(import_review_queue(session, summary)["idempotent"])
                    self.assertTrue(import_review_queue(session, summary)["idempotent"])
                    credential, token = issue_credential(session, "pg-analyst", "analyst")
                    credential_id = credential.credential_id
                    entry = session.scalar(select(QueueEntry).order_by(QueueEntry.risk_rank))
                    case = session.get(Case, entry.case_id)
                    run = session.get(WorkflowRun, "pg-run")
                    class ConstantTestModel:
                        classes_ = np.array([0, 1])
                        def predict_proba(self, frame):
                            return np.tile([1-entry.risk_score, entry.risk_score], (len(frame), 1))
                    payload = explain_case(ConstantTestModel(), case_frame(case), queue[FEATURES].iloc[:2])
                    save_snapshot(session, entry, case, run, payload,
                                  config_sha256="c"*64, background_sha256="d"*64)
                app = create_app(isolated)
                headers = {"Authorization": f"Bearer {token}"}
                path = f"/api/v1/runs/pg-run/cases/{queue.iloc[0].case_id}"
                with TestClient(app) as client:
                    self.assertEqual(client.get(path).status_code, 401)
                    detail = client.get(path, headers=headers).json()
                    self.assertNotIn("fraud_bool", detail["features"])
                    explanation = client.get(path + "/explanation", headers=headers)
                    self.assertEqual(explanation.status_code, 200)
                    self.assertLess(explanation.json()["explanation"]["reconstruction_error"], 1e-10)
                def decide(_):
                    with TestClient(app) as client:
                        return client.post(path + "/decision", headers=headers,
                                           json={"decision": "legitimate", "expected_version": 1}).status_code
                with ThreadPoolExecutor(max_workers=2) as pool:
                    self.assertEqual(sorted(pool.map(decide, range(2))), [200, 409])
                with TestClient(app) as client:
                    history = client.get(path + "/events", headers=headers).json()
                    self.assertEqual([e["case_version"] for e in history], [1, 2])
                    self.assertEqual(history[-1]["analyst_id"], "pg-analyst")
                    with create_session_factory(engine)() as session:
                        session.get(ApiCredential, credential_id).revoked = True
                        session.commit()
                    self.assertEqual(client.get(path, headers=headers).status_code, 401)
        finally:
            if app:
                app.state.engine.dispose()
            if engine:
                engine.dispose()
            with admin.begin() as conn:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            admin.dispose()
