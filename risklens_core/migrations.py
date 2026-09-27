"""Explicit migrations; legacy adoption refuses unknown schemas instead of blind stamping."""
from pathlib import Path
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import MetaData, create_engine, inspect, text

HEAD = "0003_explanations"
BASELINE = "0001_review"
ROOT = Path(__file__).resolve().parents[1]


def config_for(connection):
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["connection"] = connection
    return config


def require_current_schema(engine):
    with engine.connect() as connection:
        revision = MigrationContext.configure(connection).get_current_revision()
        if revision != HEAD:
            raise RuntimeError("Database needs migration. Stop API, then run python scripts/migrate_database.py --adopt-existing")


def verify_legacy_schema(connection):
    reference = create_engine("sqlite://")
    try:
        with reference.begin() as scratch:
            command.upgrade(config_for(scratch), BASELINE)
        expected = MetaData()
        expected.reflect(reference, only=["cases", "workflow_runs", "queue_entries", "review_events"])
        actual = inspect(connection)
        if set(actual.get_table_names()) != set(expected.tables):
            raise RuntimeError("Existing database is not the exact v1 schema; adoption refused")
        differences = compare_metadata(MigrationContext.configure(connection), expected)
        if differences:
            raise RuntimeError("Existing v1 schema differs from the known revision; adoption refused")
        for name, table in expected.tables.items():
            if actual.get_pk_constraint(name)["constrained_columns"] != [c.name for c in table.primary_key]:
                raise RuntimeError("Existing primary key differs; adoption refused")
        if connection.dialect.name == "sqlite" and connection.execute(text("PRAGMA foreign_key_check")).first():
            raise RuntimeError("Existing database contains orphaned references; adoption refused")
    finally:
        reference.dispose()


def upgrade_database(engine, *, adopt_existing=False):
    with engine.begin() as connection:
        revision = MigrationContext.configure(connection).get_current_revision()
        tables = set(inspect(connection).get_table_names())
        if revision is None and tables:
            if not adopt_existing:
                raise RuntimeError("Unversioned database: use migrate_database.py --adopt-existing")
            if connection.dialect.name != "sqlite":
                raise RuntimeError("Automatic legacy adoption is supported only for the v1 SQLite database")
            verify_legacy_schema(connection)
            command.stamp(config_for(connection), BASELINE)
        command.upgrade(config_for(connection), "head")
    require_current_schema(engine)
