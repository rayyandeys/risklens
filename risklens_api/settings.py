"""Runtime settings and production guardrails for the RiskLens API.

Research/evaluation scripts deliberately remain independent of this module. These
settings only control the analyst-facing service so post-test product hardening
cannot silently change the frozen model or evaluation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import os


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _hosts(value: str | None, *, production: bool) -> tuple[str, ...]:
    if value is None or not value.strip():
        if production:
            raise RuntimeError("RISKLENS_ALLOWED_HOSTS is required in production")
        return ("127.0.0.1", "localhost", "testserver")
    hosts = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not hosts:
        raise ValueError("RISKLENS_ALLOWED_HOSTS must contain at least one hostname")
    if production and "*" in hosts:
        raise RuntimeError("Wildcard trusted hosts are refused in production")
    return hosts


@dataclass(frozen=True)
class RuntimeSettings:
    environment: str
    allowed_hosts: tuple[str, ...]
    docs_enabled: bool
    hsts_enabled: bool
    max_request_body_bytes: int
    require_postgres: bool

    @property
    def production(self) -> bool:
        return self.environment == "production"

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        environment = os.getenv("RISKLENS_ENV", "development").strip().lower()
        if environment not in {"development", "test", "production"}:
            raise ValueError("RISKLENS_ENV must be development, test or production")
        production = environment == "production"
        allowed_hosts = _hosts(os.getenv("RISKLENS_ALLOWED_HOSTS"), production=production)
        docs_enabled = _truthy(os.getenv("RISKLENS_ENABLE_DOCS"), default=not production)
        if production and docs_enabled:
            raise RuntimeError("Interactive API docs are disabled in production")
        raw_limit = os.getenv("RISKLENS_MAX_REQUEST_BODY_BYTES", "32768")
        try:
            max_request_body_bytes = int(raw_limit)
        except ValueError as exc:
            raise ValueError("RISKLENS_MAX_REQUEST_BODY_BYTES must be an integer") from exc
        if not 4096 <= max_request_body_bytes <= 1_048_576:
            raise ValueError("RISKLENS_MAX_REQUEST_BODY_BYTES must be between 4096 and 1048576")
        return cls(
            environment=environment,
            allowed_hosts=allowed_hosts,
            docs_enabled=docs_enabled,
            hsts_enabled=production,
            max_request_body_bytes=max_request_body_bytes,
            require_postgres=production,
        )
