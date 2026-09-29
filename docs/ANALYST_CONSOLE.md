# Analyst Console v1

RiskLens now includes a React/TypeScript analyst console over the authenticated review API.
This is an operational interface for the synthetic BAF development workflow; it is not a
production fraud-decision product and must not be represented as one.

## What it exposes

- Opaque bearer-token sign-in against `/api/v1/auth/me`; the browser keeps the token in
  `sessionStorage` only, so closing the tab clears it.
- Workflow-run selection and queue summary cards.
- Ranked queue browsing with status, decision and minimum-score filters plus pagination.
- Case detail with raw feature snapshot. The target label is never returned by the API.
- Immutable paired-permutation explanation snapshots, including score decomposition,
  top positive/negative contributions, provenance, reconstruction residual and limitations.
- Append-only review history.
- Analyst/admin review writes using the API's optimistic-lock version. HTTP 409 reloads the
  latest case before the analyst can retry. Viewer credentials remain read-only.

The UI deliberately labels the score as a model score and repeats that attribution is an
explanation of model behaviour, not a cause of fraud. It does not rename the current uncalibrated
score a probability. Cases without an explanation snapshot show a bounded-coverage message rather
than inventing an explanation.

## Local development

Start the API first:

```bat
python -m uvicorn risklens_api.app:app --reload
```

In a second CMD window:

```bat
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` and `/health` to the API at
`http://127.0.0.1:8000`, so no development CORS relaxation is required.

Create a temporary credential with the existing trusted CLI, then paste the token into the
console. Do not commit or screenshot bearer credentials.

## Single-process build

```bat
cd frontend
npm run build
cd ..
python -m uvicorn risklens_api.app:app
```

When `frontend/dist/index.html` exists, FastAPI serves the Vite build at `/` after registering
all API, health and documentation routes. This keeps local demo/deployment topology simple while
preserving a separately buildable frontend source tree.

## Security / product boundaries

The current credential scheme is intentionally single-team development auth. There is no password
login, SSO, organization boundary, CSRF cookie session or tenant isolation. A future public or
multi-user deployment should terminate TLS, use a real identity provider/session design, restrict
origins at the edge, rotate credentials, add rate limits and define a proper authorization model.

The interface must not be used to automate adverse decisions about real people. BAF is synthetic;
review decisions here are workflow demonstrations. Calibration analysis, governance review, temporal monitoring, the PyTorch comparison and
the one-time final evaluation were completed in later milestones. See the root README and
committed reports for the current results; this document originally described console v1.

## Historical verification status at console v1

This section records the original console milestone, not the current deployment status.
The repository now includes the built frontend and completed final-evaluation evidence.
Use the current deployment guide for release checks.

Backend regression suite after adding optional static serving: 64 tests run, 63 passed and the
optional live PostgreSQL gate skipped because `RISKLENS_TEST_POSTGRES_URL` was not set.

The sandbox used to prepare this package could not reach the npm registry (`EAI_AGAIN`), so the
frontend dependency install/build could not be executed there. The TypeScript/TSX source was
syntax-checked with the available compiler using local module stubs; the real `npm run build`
remains a required user-side verification gate before this UI milestone is called complete.
