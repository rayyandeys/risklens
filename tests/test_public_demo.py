"""Public access boundaries: reads only, frozen run allowlist, no note/identity leaks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import func, select
from test_persistence import make_review_files
from risklens_api.auth import ApiCredential, issue_credential
from risklens_api.factory import create_app
from risklens_api.settings import RuntimeSettings
from risklens_core.persistence import (
    Case, QueueEntry, ReviewEvent, WorkflowRun, create_database_engine,
    initialize_database, import_review_queue,
)
from risklens_api.service import apply_decision


class PublicDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.url = f"sqlite:///{root / 'demo.db'}"
        engine = create_database_engine(self.url)
        initialize_database(engine)
        engine.dispose()
        settings = RuntimeSettings('test', ('testserver',), False, False, 32768, False, True)
        self.app = create_app(self.url, runtime_settings=settings)
        freeze = json.loads((Path(__file__).resolve().parents[1] / 'reports/model_freeze.json').read_text())
        with self.app.state.SessionLocal() as session:
            summary, _, _ = make_review_files(root, run_id='demo-run')
            import_review_queue(session, summary)
            run = session.get(WorkflowRun, 'demo-run')
            run.model_name = freeze['selected_model']['name']
            run.model_sha256 = freeze['selected_model']['sha256']
            run.dataset_sha256 = freeze['dataset_sha256']
            run.capacity = .03
            session.commit()
            self.case_id = session.scalar(select(Case.case_id))
            apply_decision(session, 'demo-run', self.case_id, decision='escalate',
                           analyst_note='PRIVATE NOTE', analyst_id='private-person', expected_version=1)
            _, self.token = issue_credential(session, 'real-analyst', 'analyst')
        self.client = TestClient(self.app)
        self.path = f'/api/demo/runs/demo-run/cases/{self.case_id}'

    def tearDown(self):
        self.client.close()
        self.app.state.engine.dispose()
        self.temp.cleanup()

    def test_anonymous_reads_and_redaction(self):
        self.assertEqual(self.client.get('/api/demo/config').json(), {'enabled': True})
        me = self.client.get('/api/demo/auth/me').json()
        self.assertEqual(me['role'], 'viewer')
        self.assertIsNone(me['expires_at'])
        for path in ['/api/demo/runs', '/api/demo/runs/demo-run/summary',
                     '/api/demo/runs/demo-run/cases?limit=2', self.path, self.path + '/events',
                     '/api/demo/runs/demo-run/explanations/summary', '/api/demo/monitoring/overview']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, (path, response.text))
            self.assertEqual(response.headers['cache-control'], 'no-store')
            self.assertNotIn('PRIVATE NOTE', response.text)
            self.assertNotIn('private-person', response.text)
            self.assertNotIn('fraud_bool', response.json().get('features', {}) if isinstance(response.json(), dict) else {})
        self.assertEqual(self.client.get(self.path).json()['analyst_note'], '')
        self.assertTrue(all(e['analyst_id'] is None for e in self.client.get(self.path + '/events').json()))
        self.assertEqual(self.client.get(self.path + '/explanation').status_code, 404)
        self.assertEqual(self.client.get('/api/demo/runs/demo-run/cases?limit=101').status_code, 422)

    def test_demo_cannot_write_or_authenticate_and_reads_do_not_issue_credentials(self):
        with self.app.state.SessionLocal() as session:
            before = session.scalar(select(func.count()).select_from(ApiCredential))
            events = session.scalar(select(func.count()).select_from(ReviewEvent))
        payload = {'decision': 'legitimate', 'analyst_note': '', 'expected_version': 2}
        for method in ['POST', 'PUT', 'PATCH', 'DELETE']:
            response = self.client.request(method, self.path + '/decision', json=payload)
            self.assertIn(response.status_code, (404, 405))
        for headers in [{}, {'Authorization': 'Bearer public-demo'}]:
            self.assertEqual(self.client.get('/api/v1/runs', headers=headers).status_code, 401)
            self.assertEqual(self.client.post(self.path.replace('/api/demo/', '/api/v1/') + '/decision', json=payload, headers=headers).status_code, 401)
        self.client.get('/api/demo/auth/me')
        with self.app.state.SessionLocal() as session:
            self.assertEqual(session.scalar(select(func.count()).select_from(ApiCredential)), before)
            self.assertEqual(session.scalar(select(func.count()).select_from(ReviewEvent)), events)
            self.assertEqual(session.scalar(select(QueueEntry.review_status)), 'escalated')
        private = self.client.get(self.path.replace('/api/demo/', '/api/v1/'), headers={'Authorization': f'Bearer {self.token}'})
        self.assertEqual(private.status_code, 200)
        self.assertEqual(private.json()['analyst_note'], 'PRIVATE NOTE')

    def test_other_runs_are_not_public(self):
        with self.app.state.SessionLocal() as session:
            run = session.get(WorkflowRun, 'demo-run')
            run.dataset_sha256 = 'a' * 64
            session.commit()
        self.assertEqual(self.client.get('/api/demo/runs').json(), [])
        for suffix in ['/summary', '/cases', f'/cases/{self.case_id}', f'/cases/{self.case_id}/events',
                       '/explanations/summary', f'/cases/{self.case_id}/explanation']:
            self.assertEqual(self.client.get('/api/demo/runs/demo-run' + suffix).status_code, 404)

    def test_disabled_demo_fails_closed(self):
        app = create_app(self.url, runtime_settings=replace(self.app.state.runtime_settings, public_demo_enabled=False))
        try:
            with TestClient(app) as client:
                self.assertEqual(client.get('/api/demo/config').json(), {'enabled': False})
                for path in ['/api/demo/auth/me', '/api/demo/runs', self.path, '/api/demo/monitoring/overview']:
                    self.assertEqual(client.get(path).status_code, 404)
        finally:
            app.state.engine.dispose()
