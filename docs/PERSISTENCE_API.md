> Historical v1 notes. For the current authenticated API and migration commands,
> use [AUTH_MIGRATIONS.md](AUTH_MIGRATIONS.md). Caller-supplied analyst_id is no longer accepted.

# RiskLens persistence + review API milestone

This milestone turns the validated `review-workflow-v1` CSV queue into an auditable application backend.
It does **not** retrain a model and does **not** evaluate months 6-7.

## Storage design

The schema separates four concerns:

- `workflow_runs`: immutable queue/model/dataset provenance for a scoring run.
- `cases`: source application identity and feature payload. No `fraud_bool` is stored.
- `queue_entries`: run-specific risk score/rank, current review state, decision, note and version.
- `review_events`: append-only import/decision history with analyst ID and case version.

Separating `cases` from `queue_entries` means a later model/run can score the same application without overwriting its earlier ranking provenance.

## Concurrency rule

Every queue entry begins at `version=1`. Decision writes require `expected_version` and perform a compare-and-swap update. A stale analyst write receives HTTP 409 rather than silently overwriting another decision. Every successful decision increments the version and appends an immutable event.

## Local database

For this milestone the zero-setup default is SQLite:

```bat
python scripts\import_review_queue.py
```

This creates `risklens.db`, validates `reports/review_workflow_summary.json` plus its queue, imports the 3% queue idempotently, and refuses an analyst queue containing `fraud_bool`.

The ORM is SQLAlchemy 2.x and the application accepts `DATABASE_URL`, so PostgreSQL can replace SQLite without changing the domain/API code. PostgreSQL deployment/migrations are a later milestone.

## API

Run locally with:

```bat
python -m uvicorn risklens_api.app:app --reload
```

Open `/docs` for Swagger UI.

Key endpoints:

- `GET /health`
- `GET /api/v1/runs`
- `GET /api/v1/runs/{run_id}/summary`
- `GET /api/v1/runs/{run_id}/cases`
- `GET /api/v1/runs/{run_id}/cases/{case_id}`
- `GET /api/v1/runs/{run_id}/cases/{case_id}/events`
- `POST /api/v1/runs/{run_id}/cases/{case_id}/decision`

Authentication is intentionally not claimed yet. The next application milestones add auth/roles, PostgreSQL migrations, explanation payloads, frontend and monitoring.
