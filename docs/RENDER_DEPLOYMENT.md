# Render deployment: frozen RiskLens analyst service

This deployment is a product-serving step **after** the model/protocol freeze and the one-time months 6–7 holdout evaluation. It must not retrain, recalibrate, rescore the holdout, change the 3% operating policy, or regenerate final-test metrics.

## Architecture

One Render Python web service serves both FastAPI and the prebuilt React/Vite `frontend/dist` bundle. PostgreSQL is external to the process and is provided through `DATABASE_URL`. This keeps browser/API requests same-origin and avoids adding a deployment-only CORS policy. The service runs Alembic migrations in Render's pre-deploy phase and uses `/ready` as the health check.

The production web image installs `requirements-api.txt`, not the research `requirements.txt`. The latter is intentionally unchanged because completed experiment reports recorded its hash as provenance.

## Files intentionally committed for runtime

The repository normally ignores generated files. Deployment makes two narrow exceptions:

- `frontend/dist/**` — the reviewed production SPA build. The Render Python runtime does not need Node.
- `reports/calibration_uncertainty_summary.json`, `reports/temporal_monitoring_summary.json`, `reports/model_freeze.json`, and `reports/final_test_summary.json` — exact immutable evidence required by the Monitoring page.

Do not rewrite or reformat `model_freeze.json`: `final_test_summary.json` records its exact SHA-256 and the monitoring endpoint verifies the bytes.

Raw BAF data, `artifacts/`, offline ground-truth files, local SQLite databases, `.env` files, and bearer credentials remain excluded from Git.

## Before pushing

From the repository root:

```bat
cd frontend
npm run build
npm audit --omit=dev
npm audit
cd ..
python -m unittest discover -s tests -v
python scripts\check_deployment_readiness.py
```

The full npm audit should be clean after the deliberate Vite 7.3.6 update. The Python suite may retain the one optional PostgreSQL integration skip unless `RISKLENS_TEST_POSTGRES_URL` is configured.

Then inspect what Git will publish:

```bat
git status
git diff -- .gitignore render.yaml frontend\package.json docs\RENDER_DEPLOYMENT.md scripts\check_deployment_readiness.py
```

Never commit `.env`, database URLs, bearer tokens, `risklens.db`, raw `Base.csv`, `artifacts/`, or the offline fraud-label files.

## Render Blueprint

`render.yaml` defines the web service. `DATABASE_URL` is intentionally `sync: false` so the database credential is supplied in Render instead of committed. `RISKLENS_ALLOWED_HOSTS` self-references the deployed Render host, production docs are disabled, migrations run before rollout, and `/ready` gates deployment health.

A managed PostgreSQL URL from Neon or Render is acceptable. Use a persistent database suitable for the length of time you want the portfolio demo to remain online.

## Seed the production PostgreSQL database

The deployed service schema will be created by the pre-deploy migration, but a brand-new database has no analyst queue. Seed it **from the local development checkout**, where the frozen queue artifacts, exact model artifact, and BAF development data already exist.

In a temporary CMD window, set the remote PostgreSQL URL. Do not paste it into chat, Git, screenshots, or shell history you plan to publish.

```bat
set DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE
python scripts\migrate_database.py
python scripts\import_review_queue.py --summary reports\frozen_review_workflow_summary.json
python scripts\build_explanations.py --review-summary reports\frozen_review_workflow_summary.json --summary reports\frozen_explanations_summary.json --limit 50
```

The import guard rejects ground-truth labels. Explanation generation uses the local frozen model and development-only reference data, then persists explanation snapshots to PostgreSQL. It does not open or rescore months 6–7.

After seeding, create a short-lived analyst credential against the same remote database:

```bat
python scripts\manage_credentials.py create --analyst-id rayyan --role analyst --hours 8
```

Copy only the returned `rl_...` token into the deployed browser tab. Treat it as a secret. You can create a separate viewer credential for read-only demos instead of sharing analyst access.

After seeding, clear the local shell variable:

```bat
set DATABASE_URL=
```

## Production verification

Verify:

- `/health` returns service liveness.
- `/ready` returns database readiness and `environment=production`.
- `/docs`, `/redoc`, and `/openapi.json` are unavailable.
- the Analyst Console loads the frozen `histgb_15_leaves/no_customer_age` workflow after authentication.
- Monitoring shows Frozen/Tested once and the sealed holdout Recall@3% result.
- a test analyst decision persists after a browser refresh and the History tab records the authenticated analyst ID.

Do not use the public production service to rerun research commands or the final holdout.

## Public portfolio viewing

Enable `RISKLENS_PUBLIC_DEMO=true` in the service environment after deploying the
public-demo update. Visitors can then choose **Explore demo** without receiving a
token. Analyst actions still require authenticated access. See
[PUBLIC_DEMO.md](PUBLIC_DEMO.md) for installation, boundaries and checks.
