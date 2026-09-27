"""Upgrade schema explicitly. Stop the API first. SQLite backups precede all upgrades."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy.engine import make_url
from risklens_core.database_config import database_url
from risklens_core.persistence import create_database_engine
from risklens_core.migrations import HEAD, upgrade_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    url = database_url()
    parsed = make_url(url)
    if parsed.drivername == "sqlite" and parsed.database and parsed.database != ":memory:":
        source = Path(parsed.database).resolve()
        if source.exists():
            backup_root = source.parent / "backups"
            backup_root.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            destination = backup_root / f"{source.stem}_before_auth_{stamp}.db"
            with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(destination)) as backup:
                original.backup(backup)
                if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("Backup integrity check failed; migration not started")
            print(f"Backup: {destination}")
    engine = create_database_engine(url)
    try:
        upgrade_database(engine, adopt_existing=args.adopt_existing)
    finally:
        engine.dispose()
    print(f"Schema ready: {HEAD}. Existing cases and review history preserved.")

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
