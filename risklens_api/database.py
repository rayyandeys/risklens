from risklens_core.persistence import create_database_engine, create_session_factory
from risklens_core.database_config import database_url as resolve_url
from risklens_core.migrations import require_current_schema


def build_database(database_url: str | None = None):
    engine = create_database_engine(resolve_url(database_url))
    try:
        require_current_schema(engine)
    except Exception:
        engine.dispose()
        raise
    return engine, create_session_factory(engine)
