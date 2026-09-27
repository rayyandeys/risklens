# Windows cleanup fix v1.1

The user's first Windows run failed in temporary-backup cleanup with WinError 32.
SQLite connections used as transaction contexts remained open. Both the backup
script and test now use contextlib.closing. The added regression explicitly checked
live handles and failed before the production fix, on success and exception paths.
Final local result: 56 tests; 55 passed; 1 live PostgreSQL test skipped.
Windows rerun is pending. The user's operational migration and authenticated API
already succeeded, so no operational migration rerun is necessary for this fix.

Python documentation: https://docs.python.org/3.12/library/sqlite3.html#how-to-use-the-connection-context-manager

---

Historical initial verification (before the Windows report):

# Validation record: auth and migrations v1

Date: 27 September 2026. Python 3.12 on Linux; exact requirements installed in an
isolated environment. Windows execution remains for the user's local verification.

- Full suite: 55 tests run; 54 passed; 1 live PostgreSQL integration test skipped.
- SQLAlchemy 2.0.50, FastAPI 0.128.2, Pydantic 2.13.4, HTTPX 0.28.1,
  Alembic 1.20.0, Psycopg 3.3.6; existing ML requirements retained.
- Migration SQL compiled for PostgreSQL including SERIAL keys, JSON, timezone-aware
  timestamps, unique constraints and foreign keys. This is not a live server test.
- Additional end-to-end smoke: loaded the ORIGINAL uploaded v1 ORM definition into
  an isolated module, created its unversioned SQLite schema, imported a synthetic
  5-case queue, recorded a decision, invoked the actual migration CLI, and compared
  all columns/rows of all four existing tables before and after. Exact match.
  Credential issuance CLI, authenticated API reads, and revocation CLI also passed.
- Unit tests additionally verify backup integrity, schema drift rejection, auth
  on all business routes, viewer denial, spoofed identity rejection, expired/revoked
  credentials, digest-only token storage, version conflicts, concurrent writers,
  rollback if event insertion fails, and enabled SQLite foreign keys.
- All 11 existing training/evaluation/scoring/queue source files compared
  byte-for-byte with the uploaded ZIP: unchanged.
- No BAF dataset, user database or model artifacts were available in the workspace.
  Synthetic tests establish software behavior, not new fraud-model performance.
- PostgreSQL server unavailable in this workspace. Run the opt-in gate described
  in AUTH_MIGRATIONS.md before claiming live PostgreSQL compatibility.
- User's actual SQLite upgrade is pending. Backups are created by the upgrade CLI.
