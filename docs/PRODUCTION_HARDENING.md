# Production hardening checkpoint

This checkpoint hardens the analyst-facing service **after** model freeze and final-test opening. It does not change the frozen model, features, score policy, 3% review capacity, evaluation results, or review data.

## Runtime controls

`RISKLENS_ENV=production` enables production guardrails. Production refuses SQLite and requires PostgreSQL. `RISKLENS_ALLOWED_HOSTS` is mandatory and wildcard hosts are refused. Interactive OpenAPI/Swagger/ReDoc endpoints are disabled. The application emits HSTS plus CSP, clickjacking, MIME-sniffing, referrer, permissions, COOP and CORP headers. Authenticated API responses are marked `Cache-Control: no-store` and every response receives a server-generated request ID.

Decision/write bodies have a small configurable `Content-Length` gate (`RISKLENS_MAX_REQUEST_BODY_BYTES`, default 32 KiB). This is a defense-in-depth application limit, not a replacement for ingress/proxy request limits.

`/health` is a liveness endpoint and intentionally does not touch the database. `/ready` executes a database probe. The API already refuses startup when the Alembic schema is not current.

## Authentication threat model

RiskLens uses random opaque bearer credentials. Only SHA-256 token digests are persisted, credentials expire and can be revoked, and the authenticated identity—not a client-supplied identity—is written to review history. The browser stores the token in `sessionStorage`, so it disappears when the tab/session is closed. The production CSP reduces script injection exposure, but a bearer token is still a secret and should never be pasted into logs, Git, screenshots, issue trackers or chat.

This is appropriate for a portfolio/single-team analyst system. A real multi-tenant deployment would normally integrate enterprise identity (OIDC/SAML), centralized secrets, network controls, SIEM/audit export, managed rate limiting/WAF and organization-specific authorization policy.

## Dependency separation

`requirements.txt` is intentionally left byte-for-byte unchanged because historical experiment/final-test reports recorded its SHA-256 as provenance. `requirements-api.txt` is a smaller production runtime set and excludes PyTorch, which is needed for the completed challenger experiment but not for serving the analyst application.

For the frontend, do **not** run `npm audit fix --force`. Inspect both production and complete dependency trees:

```bat
cd frontend
npm audit --omit=dev
npm audit
```

A vulnerability that exists only in build tooling has a different deployment exposure from a vulnerable browser runtime dependency, but it should still be documented and upgraded deliberately.

## Local development

No environment variables are required for the existing local SQLite workflow. Development defaults trust `127.0.0.1`, `localhost` and `testserver`, and API docs remain available.

## Production environment example

See `deploy/env.production.example`. Do not place actual credentials in that file.
