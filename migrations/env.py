from alembic import context
from risklens_core.persistence import Base, create_database_engine
from risklens_api.auth import ApiCredential  # register auth metadata
from risklens_core.explanation_store import CaseExplanation
from risklens_core.database_config import database_url

config = context.config

def run(connection):
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

if context.is_offline_mode():
    context.configure(url=database_url(), target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
elif config.attributes.get("connection") is not None:
    run(config.attributes["connection"])
else:
    engine = create_database_engine(database_url())
    with engine.connect() as connection:
        run(connection)
    engine.dispose()
