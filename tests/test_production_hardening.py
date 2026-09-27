import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from risklens_api.factory import create_app
from risklens_api.settings import RuntimeSettings
from risklens_core.migrations import upgrade_database
from risklens_core.persistence import create_database_engine


class RuntimeSettingsTests(unittest.TestCase):
    def test_development_defaults_are_local_and_docs_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = RuntimeSettings.from_env()
        self.assertEqual(settings.environment, "development")
        self.assertTrue(settings.docs_enabled)
        self.assertFalse(settings.require_postgres)
        self.assertIn("localhost", settings.allowed_hosts)

    def test_production_requires_explicit_nonwildcard_hosts_and_disables_docs(self):
        with patch.dict(os.environ, {"RISKLENS_ENV": "production"}, clear=True):
            with self.assertRaises(RuntimeError):
                RuntimeSettings.from_env()
        with patch.dict(os.environ, {
            "RISKLENS_ENV": "production",
            "RISKLENS_ALLOWED_HOSTS": "*",
        }, clear=True):
            with self.assertRaises(RuntimeError):
                RuntimeSettings.from_env()
        with patch.dict(os.environ, {
            "RISKLENS_ENV": "production",
            "RISKLENS_ALLOWED_HOSTS": "risklens.example.com",
        }, clear=True):
            settings = RuntimeSettings.from_env()
        self.assertFalse(settings.docs_enabled)
        self.assertTrue(settings.hsts_enabled)
        self.assertTrue(settings.require_postgres)

    def test_production_refuses_sqlite(self):
        settings = RuntimeSettings(
            environment="production", allowed_hosts=("testserver",), docs_enabled=False,
            hsts_enabled=True, max_request_body_bytes=32768, require_postgres=True,
        )
        with self.assertRaises(RuntimeError):
            create_app("sqlite:///:memory:", runtime_settings=settings)


class HardeningMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.url = f"sqlite:///{Path(self.temp.name) / 'hardening.db'}"
        engine = create_database_engine(self.url)
        upgrade_database(engine)
        engine.dispose()
        self.settings = RuntimeSettings(
            environment="production", allowed_hosts=("testserver", "risklens.example.com"),
            docs_enabled=False, hsts_enabled=True, max_request_body_bytes=4096,
            require_postgres=False,  # dependency-injected only for this middleware unit test
        )
        self.app = create_app(self.url, runtime_settings=self.settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.app.state.engine.dispose()
        self.temp.cleanup()

    def test_security_headers_docs_and_readiness(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertIn("max-age=31536000", response.headers["strict-transport-security"])
        self.assertRegex(response.headers["x-request-id"], r"^[0-9a-f]{32}$")
        self.assertEqual(self.client.get("/ready").status_code, 200)
        self.assertEqual(self.client.get("/docs").status_code, 404)
        self.assertEqual(self.client.get("/openapi.json").status_code, 404)

    def test_api_responses_are_not_cacheable(self):
        response = self.client.get("/api/v1/runs")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_untrusted_host_and_large_write_are_rejected(self):
        with TestClient(self.app, base_url="http://evil.example") as bad_client:
            self.assertEqual(bad_client.get("/health").status_code, 400)
        response = self.client.post(
            "/api/v1/runs/x/cases/y/decision",
            content=b"x" * 4097,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
