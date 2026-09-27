"""Import the verified RiskLens analyst queue into persistent local storage."""
from __future__ import annotations

import argparse
from os import getenv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risklens_core.persistence import create_database_engine, create_session_factory, import_review_queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/review_workflow_summary.json"))
    parser.add_argument("--database-url", default=getenv("DATABASE_URL", "sqlite:///./risklens.db"))
    args = parser.parse_args()

    engine = create_database_engine(args.database_url)
    from risklens_core.migrations import require_current_schema
    require_current_schema(engine)
    SessionLocal = create_session_factory(engine)
    with SessionLocal() as session:
        result = import_review_queue(session, args.summary)
    print(
        f"Imported run {result['run_id']}: selected={result['selected_cases']:,}; "
        f"new applications={result['inserted_cases']:,}; idempotent={result['idempotent']}"
    )
    print("The analyst database contains no fraud_bool ground-truth labels.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Import stopped: {exc}") from exc
