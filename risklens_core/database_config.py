"""One database URL policy for API, imports, migrations and credential CLI."""
import os
from sqlalchemy.engine import make_url


def database_url(value: str | None = None) -> str:
    raw = value or os.getenv("DATABASE_URL") or "sqlite:///./risklens.db"
    parsed = make_url(raw)
    if parsed.drivername in ("postgres", "postgresql"):
        parsed = parsed.set(drivername="postgresql+psycopg")
    if parsed.drivername not in ("sqlite", "postgresql+psycopg"):
        raise ValueError("Use sqlite or postgresql+psycopg for DATABASE_URL")
    return parsed.render_as_string(hide_password=False)
