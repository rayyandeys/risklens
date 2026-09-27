# Auth and database migrations v1 — 27 September 2026

This release upgrades the existing review API. It does not train models, regenerate
queues or read Base.csv. Months 6–7 remain outside this milestone.

## Upgrade your existing Windows project

1. Stop Uvicorn with Ctrl+C **before copying files**.
2. Extract the downloaded ZIP in Downloads. Copy the contents of its `risklens`
   directory into `C:\Users\smray\Documents\Projects\risklens`. Replace matching
   files. Do not delete `.venv`, `data`, `reports`, `artifacts`, or `risklens.db`.
3. Use the existing activated CMD virtual environment, from the project root:

```cmd
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts\migrate_database.py --adopt-existing
python scripts\manage_credentials.py create --analyst-id rayyan --role analyst --hours 8
python -m uvicorn risklens_api.app:app --reload
```

The test suite uses temporary databases. The live PostgreSQL test is skipped unless
explicitly configured; see below. Expected local total: 56 tests, 55 pass, 1 skipped.

The migration script backs up an existing SQLite database using the SQLite backup
API, checks backup integrity, verifies the legacy schema, stamps only the known v1
revision, then adds the credential table. Backup path prints before migration.
Existing cases, provenance, ranks, decisions, versions and review events survive.
To restore after a failed upgrade: stop the API, preserve the failed DB separately,
and copy the printed backup back to `risklens.db`. Restore the v1 source package too
if you intend to run the old API; the new API requires the migrated schema. Do not
restore over a running database.

Unexpected schemas are refused. Do not manually stamp head to bypass this check.
Repeated upgrades are safe; each CLI invocation backs up an existing SQLite file.
New empty databases work without `--adopt-existing`.

All tools use DATABASE_URL, falling back to `sqlite:///./risklens.db`. Do not change
DATABASE_URL during the existing-database upgrade. The API and import CLI now refuse
an unmigrated database instead of silently creating tables on startup.

## Authorize in Swagger

Open http://127.0.0.1:8000/docs. Click **Authorize**, paste only the `rl_...` token
printed by the credential CLI (no extra `Bearer` prefix), click Authorize and Close.
Run `GET /api/v1/auth/me`, then `GET /api/v1/runs`. The existing run should still
have 3,579 selected cases. Status counts reflect any decisions already made.

The token is a secret: do not paste it into chat, screenshots, source code or Git.
It is shown once and expires after 8 hours in the command above. If lost or expired,
issue a new credential. Revoke a credential by its nonsecret ID:

```cmd
python scripts\manage_credentials.py revoke --credential-id YOUR_CREDENTIAL_ID
```

To verify read-only access, create another credential with `--role viewer`, then
use that token in Swagger. A valid viewer may read cases but a decision returns 403.

| Role | Read runs, cases, events | Write decisions | Provision/revoke credentials |
|---|---|---|---|
| viewer | Yes | No | No API permission |
| analyst | Yes | Yes | No API permission |
| admin | Yes | Yes | Trusted operator CLI only |

Admin currently inherits analyst permissions; no remote administration endpoints
exist. Credential creation/revocation requires local operator/database access, not
an API token. There is no self-registration, password login, SSO, or tenant isolation.
The three roles apply to a single trusted team; multi-team isolation is future work.

Decision bodies now contain only:

```json
{"decision":"escalate","analyst_note":"Needs a second review","expected_version":1}
```

Use the current case version. `analyst_id` is deliberately rejected (422); it is
recorded from the authenticated credential instead. Valid decisions remain
`confirmed_fraud`, `legitimate`, `escalate`. Stale versions return 409. A successful
update and its history entry commit in one transaction. History is append-only
through the API; a privileged database operator can still edit it. This is not
cryptographically tamper-proof auditing. Legacy event identities retain their
original values and were not authenticated retroactively.

401 means missing/invalid/expired/revoked credential; 403 means wrong role;
404 means missing run/case; 409 means stale version; 422 means invalid request.
`/health`, `/docs`, and `/openapi.json` remain public and do not return case data.
Use TLS before exposing bearer authentication beyond local development.

## PostgreSQL schema path — live server verification still required

Alembic revisions are frozen operations, not live ORM `create_all` calls.
The Psycopg 3 driver and shared URL handling are included. `postgresql://` and
`postgres://` normalize to `postgresql+psycopg://`. PostgreSQL SQL compilation has
been tested, but this build environment had no running PostgreSQL server.

Keep SQLite for the immediate upgrade. For a **new** PostgreSQL database later:

```cmd
set DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/risklens
python scripts\migrate_database.py
python scripts\import_review_queue.py
python scripts\manage_credentials.py create --analyst-id rayyan --role analyst --hours 8
python -m uvicorn risklens_api.app:app --reload
```

URL-encode special password characters and use the provider's required TLS options
for a remote server. Do not put actual credentials into tracked files.
This initializes a fresh PostgreSQL store and imports the saved queue; it does NOT
copy SQLite decisions/history/credentials. Preserve the SQLite DB and backup.
If it contains reviews you want to retain, complete a data-transfer milestone before
switching the operational store. No cloud resources are provisioned by this ZIP.

Run the optional integration gate against a **dedicated test database**:

```cmd
set RISKLENS_TEST_POSTGRES_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/risklens_test
python -m unittest discover -s tests -p test_postgres_integration.py -v
set RISKLENS_TEST_POSTGRES_URL=
```

The test creates a unique temporary schema and drops only that schema in cleanup;
the test role needs schema creation permission. It exercises real migrations,
schema matching, idempotent imports, JSON feature reads, authentication, two
concurrent decision writers, history timestamps, and credential revocation.
Never claim PostgreSQL is verified until this gate passes against an actual server.

## Design and verification

Opaque bearer tokens use 256 bits of random entropy and expire at a fixed UTC time.
Only SHA-256 digests are stored; this is high-entropy token storage, not password
hashing. Every request checks current expiry/revocation/role in the database.
This makes revocation immediate for subsequent requests; already in-flight requests
may finish. A future browser login should use an established identity provider or a
separately designed session flow rather than treating these development credentials
as a complete identity platform.

Tests cover regression behavior, no-credential denial for every data route, viewer
restrictions, identity spoofing rejection, expiry/revocation, token storage, stale
and concurrent writes, atomic rollback if a history insert fails, SQLite foreign
keys, old database preservation, backup integrity, schema drift rejection and
PostgreSQL migration SQL compilation.

References used for implementation:
- https://alembic.sqlalchemy.org/en/latest/cookbook.html
- https://alembic.sqlalchemy.org/en/latest/tutorial.html
- https://fastapi.tiangolo.com/reference/security/
