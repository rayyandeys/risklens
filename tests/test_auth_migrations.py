from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from scripts import migrate_database
from datetime import timedelta
import io
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import event, inspect, select, text
from sqlalchemy.exc import IntegrityError

from risklens_api.auth import ApiCredential, authenticate, issue_credential
from risklens_api.factory import create_app
from risklens_api.service import apply_decision
from risklens_core.database_config import database_url
from risklens_core.migrations import config_for, upgrade_database, require_current_schema, ROOT
from risklens_core.persistence import Base, QueueEntry, ReviewEvent, create_database_engine, create_session_factory, import_review_queue, utcnow
from test_persistence import make_review_files


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.url = f"sqlite:///{self.root / 'auth.db'}"
        engine = create_database_engine(self.url)
        upgrade_database(engine)
        engine.dispose()
        self.app = create_app(self.url)
        self.sessions = self.app.state.SessionLocal
        with self.sessions() as session:
            summary, _, queue = make_review_files(self.root, run_id="auth-run")
            import_review_queue(session, summary)
            self.case_id = queue.iloc[0].case_id
            self.tokens = {}
            self.ids = {}
            for role in ("viewer", "analyst", "admin"):
                record, token = issue_credential(session, f"{role}-identity", role)
                self.tokens[role] = token
                self.ids[role] = record.credential_id
        self.client = TestClient(self.app)
        self.path = f"/api/v1/runs/auth-run/cases/{self.case_id}"
        self.payload = {"decision": "escalate", "analyst_note": "review", "expected_version": 1}

    def tearDown(self):
        self.client.close()
        self.app.state.engine.dispose()
        self.temp.cleanup()

    def headers(self, role):
        return {"Authorization": f"Bearer {self.tokens[role]}"}

    def test_all_business_routes_require_credentials(self):
        paths = ["/api/v1/auth/me", "/api/v1/runs", "/api/v1/runs/auth-run/summary",
                 "/api/v1/runs/auth-run/cases", self.path, self.path + "/events"]
        for path in paths:
            with self.subTest(path=path):
                result = self.client.get(path)
                self.assertEqual(result.status_code, 401)
                self.assertEqual(result.headers["www-authenticate"], "Bearer")
        self.assertEqual(self.client.post(self.path + "/decision", json=self.payload).status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)
        for token in ("bad", "rl_" + "a" * 43):
            self.assertEqual(self.client.get("/api/v1/runs", headers={"Authorization": f"Bearer {token}"}).status_code, 401)

    def test_viewer_can_read_but_cannot_decide(self):
        self.assertEqual(self.client.get(self.path, headers=self.headers("viewer")).status_code, 200)
        result = self.client.post(self.path + "/decision", json=self.payload, headers=self.headers("viewer"))
        self.assertEqual(result.status_code, 403)
        history = self.client.get(self.path + "/events", headers=self.headers("viewer")).json()
        self.assertEqual(len(history), 1)

    def test_identity_cannot_be_spoofed_and_is_recorded(self):
        bad = dict(self.payload, analyst_id="someone-else")
        self.assertEqual(self.client.post(self.path + "/decision", json=bad, headers=self.headers("analyst")).status_code, 422)
        self.assertEqual(self.client.post(self.path + "/decision", json=self.payload, headers=self.headers("analyst")).status_code, 200)
        history = self.client.get(self.path + "/events", headers=self.headers("viewer")).json()
        self.assertEqual(history[-1]["analyst_id"], "analyst-identity")
        self.assertEqual(self.client.post(self.path + "/decision", json=self.payload, headers=self.headers("analyst")).status_code, 409)

    def test_admin_can_decide_and_me_exposes_no_secret(self):
        self.assertEqual(self.client.post(self.path + "/decision", json=self.payload, headers=self.headers("admin")).status_code, 200)
        response = self.client.get("/api/v1/auth/me", headers=self.headers("admin"))
        self.assertEqual(set(response.json()), {"analyst_id", "role", "expires_at"})
        self.assertNotIn(self.tokens["admin"], response.text)

    def test_expiry_revocation_and_only_hashed_tokens_persist(self):
        with self.sessions() as session:
            self.assertIsNotNone(authenticate(session, self.tokens["analyst"]))
            credential = session.get(ApiCredential, self.ids["analyst"])
            self.assertNotEqual(credential.token_sha256, self.tokens["analyst"])
            credential.expires_at = utcnow() - timedelta(seconds=1)
            session.get(ApiCredential, self.ids["viewer"]).revoked = True
            session.commit()
        for role in ("analyst", "viewer"):
            self.assertEqual(self.client.get(self.path, headers=self.headers(role)).status_code, 401)
        raw = (self.root / 'auth.db').read_bytes()
        for token in self.tokens.values():
            self.assertNotIn(token.encode(), raw)

    def test_two_writers_only_one_decision_event(self):
        def send(_):
            with TestClient(self.app) as client:
                return client.post(self.path + "/decision", json=self.payload, headers=self.headers("analyst")).status_code
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(send, range(2)))
        self.assertEqual(sorted(codes), [200, 409])
        history = self.client.get(self.path + "/events", headers=self.headers("viewer")).json()
        self.assertEqual([e["case_version"] for e in history], [1, 2])

    def test_failed_history_insert_rolls_back_case_change(self):
        def reject_event(mapper, connection, target):
            raise RuntimeError("synthetic history failure")
        event.listen(ReviewEvent, "before_insert", reject_event)
        try:
            with self.sessions() as session:
                with self.assertRaises(RuntimeError):
                    apply_decision(session, "auth-run", self.case_id, decision="legitimate",
                                   analyst_note="", analyst_id="test", expected_version=1)
                session.rollback()
        finally:
            event.remove(ReviewEvent, "before_insert", reject_event)
        detail = self.client.get(self.path, headers=self.headers("viewer")).json()
        self.assertEqual(detail["version"], 1)
        self.assertEqual(detail["review_status"], "pending")


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.url = f"sqlite:///{self.root / 'legacy.db'}"
        self.engine = create_database_engine(self.url)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def legacy(self):
        with self.engine.begin() as conn:
            command.upgrade(config_for(conn), "0001_review")
            conn.execute(text("DROP TABLE alembic_version"))

    def test_upgrade_preserves_populated_legacy_and_decisions(self):
        self.legacy()
        sessions = create_session_factory(self.engine)
        with sessions() as session:
            summary, _, queue = make_review_files(self.root)
            import_review_queue(session, summary)
            apply_decision(session, "run-test", queue.iloc[0].case_id, decision="legitimate",
                           analyst_note="keep me", analyst_id="legacy-reviewer", expected_version=1)
        def snapshot():
            with self.engine.connect() as conn:
                return {name: conn.execute(text(f"SELECT * FROM {name}")).all()
                        for name in ("cases", "workflow_runs", "queue_entries", "review_events")}
        before = snapshot()
        with self.assertRaises(RuntimeError):
            upgrade_database(self.engine)
        upgrade_database(self.engine, adopt_existing=True)
        upgrade_database(self.engine, adopt_existing=True)
        self.assertEqual(before, snapshot())
        require_current_schema(self.engine)

    def test_unknown_legacy_schema_is_not_stamped(self):
        self.legacy()
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE cases ADD COLUMN fraud_bool INTEGER"))
        with self.assertRaises(RuntimeError):
            upgrade_database(self.engine, adopt_existing=True)
        self.assertNotIn("alembic_version", inspect(self.engine).get_table_names())

    def test_fresh_upgrade_matches_models_and_enforces_foreign_keys(self):
        upgrade_database(self.engine)
        with self.engine.connect() as conn:
            self.assertEqual(compare_metadata(MigrationContext.configure(conn), Base.metadata), [])
            self.assertEqual(conn.execute(text("PRAGMA foreign_keys")).scalar(), 1)
        with create_session_factory(self.engine)() as session:
            session.add(ReviewEvent(queue_entry_id=999, event_type="imported", to_status="pending",
                                    analyst_note="", case_version=1, created_at=utcnow()))
            with self.assertRaises(IntegrityError):
                session.commit()

    def test_cli_creates_readable_backup_before_adoption(self):
        self.legacy()
        environment = dict(os.environ, DATABASE_URL=self.url)
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/migrate_database.py'), '--adopt-existing'],
                                env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        backups = list((self.root / "backups").glob("*.db"))
        self.assertEqual(len(backups), 1)
        with closing(sqlite3.connect(backups[0])) as conn:
            self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("api_credentials", names)
        require_current_schema(self.engine)

    def test_backup_connections_close_on_success_and_failure(self):
        self.legacy()
        original_connect = sqlite3.connect
        for fail_integrity in (False, True):
            with self.subTest(fail_integrity=fail_integrity):
                connections = []

                class TrackedConnection(sqlite3.Connection):
                    def execute(connection, sql, *args, **kwargs):
                        if fail_integrity and sql == "PRAGMA integrity_check":
                            raise RuntimeError("synthetic integrity failure")
                        return super().execute(sql, *args, **kwargs)

                def connect(*args, **kwargs):
                    conn = original_connect(*args, factory=TrackedConnection, **kwargs)
                    connections.append(conn)
                    return conn

                with patch.dict(os.environ, DATABASE_URL=self.url), \
                     patch.object(sys, "argv", ["migrate_database.py", "--adopt-existing"]), \
                     patch.object(migrate_database.sqlite3, "connect", side_effect=connect), \
                     redirect_stdout(io.StringIO()):
                    try:
                        if fail_integrity:
                            with self.assertRaisesRegex(RuntimeError, "synthetic integrity failure"):
                                migrate_database.main()
                        else:
                            migrate_database.main()
                        self.assertEqual(len(connections), 2)
                        for conn in connections:
                            with self.assertRaises(sqlite3.ProgrammingError):
                                conn.execute("SELECT 1")
                    finally:
                        # Even if the regression fails, release handles so cleanup
                        # cannot hide the real assertion on Windows.
                        for conn in connections:
                            conn.close()

    def test_api_refuses_unmigrated_database(self):
        with self.assertRaises(RuntimeError):
            create_app(self.url)
        self.assertNotIn("api_credentials", inspect(self.engine).get_table_names())

    def test_postgres_migrations_compile_and_url_normalizes(self):
        stream = io.StringIO()
        config = Config(str(ROOT / "alembic.ini"), output_buffer=stream)
        with patch.dict(os.environ, DATABASE_URL="postgresql://test:secret%25@localhost/risklens"):
            command.upgrade(config, "head", sql=True)
        sql = stream.getvalue()
        self.assertIn("CREATE TABLE api_credentials", sql)
        self.assertIn("TIMESTAMP WITH TIME ZONE", sql)
        self.assertIn("SERIAL", sql)
        self.assertEqual(database_url("postgresql://u:p@localhost/db"), "postgresql+psycopg://u:p@localhost/db")
