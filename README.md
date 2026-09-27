# Current release: authenticated monitoring dashboard v1

The frozen no-age HistGB model has completed its one-time months 6-7 holdout evaluation and the analyst queue has been realigned to that exact frozen artifact. This release turns the existing drift, stability, freeze and final-test evidence into a read-only **Monitoring** page inside the authenticated React console. It does not retrain, rescore, reopen the holdout or change any decision policy.

The new `/api/v1/monitoring/overview` endpoint verifies report lineage and the exact frozen-manifest hash before projecting aggregate evidence to the UI. The dashboard separates model-agnostic feature drift, the pre-freeze development score-drift study, frozen-model month-5/6/7 performance, queue-boundary sensitivity and immutable governance lineage. Missing or tampered evidence fails closed with HTTP 503.

After merging while preserving `.venv`, `data`, `reports`, `artifacts`, `risklens.db`, and `frontend/node_modules`, run:

```bat
python -m unittest discover -s tests -v
cd frontend
npm run build
cd ..
python -m uvicorn risklens_api.app:app --reload
```

This source package verifies **107 Python tests** locally: 106 pass and the same optional live PostgreSQL integration gate is skipped when `RISKLENS_TEST_POSTGRES_URL` is unset. No Python or JavaScript dependency changed. See [MONITORING_DASHBOARD.md](docs/MONITORING_DASHBOARD.md).

---

# Current release: temporal monitoring v1

The analyst console is operational and `calibration-uncertainty-v1` is complete. The observed
development result kept the raw HistGB-15 score: neither sigmoid nor isotonic calibration improved
month-5 Brier/ECE, while the bootstrap stability analysis showed substantial sensitivity around the
3% queue boundary. The next milestone therefore measures temporal population/score drift and builds
a label-safe monitoring payload before the PyTorch comparison or final holdout evaluation.

Merge this package while preserving `.venv`, `data`, `reports`, `artifacts`, `risklens.db`, and the
locally built frontend. No dependencies changed. Then run:

```bat
python -m unittest discover -s tests -v
python scripts\run_temporal_monitoring.py
```

Expect **79 tests** with the same one optional PostgreSQL integration skip when
`RISKLENS_TEST_POSTGRES_URL` is unset. The monitoring command writes
`reports/temporal_monitoring_summary.json` plus a timestamped `artifacts/temporal_monitoring/`
directory. It intentionally performs expanding-window HistGB refits for months 1-5, so it can take
several minutes. Months 6-7 remain sealed. See [TEMPORAL_MONITORING.md](docs/TEMPORAL_MONITORING.md).

---

# Current release: case explanations v1

Start with [CASE_EXPLANATIONS.md](docs/CASE_EXPLANATIONS.md). Existing users should
stop the API, merge this update, run tests, migrate to 0003_explanations, build the
first 50 explanations, then restart the API. Dependencies and trained models are unchanged.
Local suite: 64 tests, 63 pass, one PostgreSQL integration test skipped.
Prior auth/concurrency PostgreSQL gate passed on the user's Windows/Docker setup;
the new explanation storage extension still needs its live gate before deployment.

Historical milestone notes follow; the linked current guide supersedes their commands.

---

> v1.1 fixes the Windows temporary-backup file lock. Existing v1 users should stop
> Uvicorn, merge this update, rerun tests and restart the API. No migration rerun
> or dependency change is required. See PROJECT_HANDOFF.md for the current checkpoint.

# Current checkpoint: authenticated review API + migrations

Read [the upgrade guide](docs/AUTH_MIGRATIONS.md) first. This supersedes older API
startup commands below: migrate your database, issue a credential, then start the API.
No ML reruns are needed. Local test result: 55 passing, one optional PostgreSQL test skipped.

# RiskLens — Account Fraud Detection and Review System

## Current milestone: persistent analyst backend

The validated month-5 review queue can now be imported into an auditable SQLAlchemy database and served through FastAPI. This milestone does **not** retrain the model or evaluate reserved months 6-7.

After merging this package into your existing project, install the newly added backend dependencies, run the full test suite, import the existing verified queue, then start the API:

```bat
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts\import_review_queue.py
python -m uvicorn risklens_api.app:app --reload
```

Expect **41 tests**. The default database is `risklens.db` (SQLite) and is ignored by Git. Open `http://127.0.0.1:8000/docs` for the local Swagger UI. The importer is idempotent and refuses target leakage into the analyst store. Analyst decisions use versioned compare-and-swap writes and append immutable review events. See `docs/PERSISTENCE_API.md`.

Important: preserve `.venv`, `data/raw/Base.csv`, `reports/`, `artifacts/`, and the existing review-workflow outputs when merging. Do not rerun training, robustness or queue construction for this milestone.

## Current step: review workflow v1

`robustness-v1` is complete on the real BAF development split. The full HistGB-15 model remains the current development reference; the no-age variant is retained as a challenger. See `docs/ROBUSTNESS_RESULTS.md`.

This package adds the first label-safe scoring and analyst-review slice. Merge it into the existing project while preserving `.venv`, `data/raw/Base.csv`, `reports/` and `artifacts/`, then run:

```bat
python -m unittest discover -s tests -v
python scripts\build_review_queue.py
```

Expect 35 tests. The queue command verifies the audit, robustness report, saved model artifact and saved validation predictions before writing a 3% ranked review queue. `fraud_bool` is kept out of analyst-facing files. Months 6-7 remain sealed. Upload `reports/review_workflow_summary.json` after it completes.


## Current step: boosted comparison

The user has completed the logistic run; results are recorded in `docs/BASELINE_RESULTS.md`.
Both converged. Unweighted logistic is the current reference. The next bounded experiment
is specified in `docs/BOOSTED_EXPERIMENT.md`, before any tree results have been observed.

Merge this ZIP's project files into the existing risklens directory, replacing matching files.
Keep `.venv`, raw data, reports and artifacts; they are not included in the ZIP.
No dependency changes or logistic rerun are needed. In the active venv:

```bat
python -m unittest discover -s tests -v
python scripts\train_boosted.py
```

Expect 26 tests. Upload `reports/boosted_summary.json` after training. Keep both generations of
model/prediction artifacts locally. If a command fails, share the exact error before continuing.

## Previous step: logistic baseline

Milestone 2: reproducible logistic baselines with development-only evaluation.
The package contains a working audit and training pipeline. The user's real dataset run is recorded
in BASELINE_RESULTS.md; these are month-5 validation metrics. There is no web app yet.

## Continue after the audit (your current step)

Merge the contents of this ZIP's `risklens` folder into your existing project. Replace the matching
project files, but keep `.venv`, `data/raw/Base.csv` and `reports/data_audit.json`. These local items
are not included in the ZIP. Do not place a second risklens directory inside the first one.

With your existing environment active in CMD:

```bat
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts\train_baseline.py
```

Expect 21 tests. The training command prints four stages; individual fits can take several minutes
depending on your machine. It trains unweighted and balanced logistic regression and writes
`reports/baseline_summary.json`. Upload that JSON for review. Keep model files and validation
predictions locally in the timestamped `artifacts/baselines` directory.

The command verifies the existing dataset audit and refuses a checksum mismatch. Never edit a
checksum to bypass that check. If it reports duplicates or convergence problems, share the message.
Months 6–7 are never evaluated by this command. A baseline result is not the finished project.

## Start on Windows (CMD)

Extract the `risklens` folder into `C:\Users\smray\Documents\Projects`.
If that folder already exists, extract elsewhere and compare before merging.

```bat
cd /d C:\Users\smray\Documents\Projects\risklens
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The original audit uses only the standard library. For the complete current test suite, install
`requirements.txt` first. Tests use synthetic fixtures and do not establish fraud performance.

Download the Bank Account Fraud Dataset Suite from the link in the official Feedzai repository:

- Repository: https://github.com/feedzai/bank-account-fraud
- Dataset: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

Extract the base CSV into `data/raw/Base.csv`. Use the base dataset first, keep variants separate,
and retain the downloaded dataset version and its usage terms. Do not commit raw data or credentials.
If the download uses a different filename, pass its actual path to `--csv`.

```bat
python scripts\audit_data.py --csv data\raw\Base.csv --out reports\data_audit.json
```

Share `reports/data_audit.json` in the next chat message. If the command fails, share the error.
The audit streams rows, computes the file SHA-256, checks structural integrity and label/month
values, screens training-data missingness, and reports development class balance. It does not
remove columns or change the dataset. Full feature-schema and leakage reviews remain necessary.

## Files

- `scripts/audit_data.py`: auditable, standard-library-only data check and CLI.
- `tests/test_audit.py`: malformed data, temporal coverage, hash, and holdout-report checks.
- `docs/EVALUATION_PROTOCOL.md`: proposed split and evaluation gates.
- `PROJECT_HANDOFF.md`: current state, decisions, and exact continuation instructions.
- `data/raw/`: local dataset; ignored by Git except its placeholder.
- `reports/`: generated audit report; raw generated reports are ignored by default.

## After baseline review

Review convergence, missingness and validation metrics, then add the gradient-boosted comparison
and an early scoring/review workflow. The full scope still includes PyTorch comparison, uncertainty,
calibration, drift/subgroup/error analysis, persistence, API, UI, operations and deployment.

Suggested later application stack: FastAPI, PostgreSQL, React/TypeScript, and a small shared
component system. This is a proposed choice, not a recovered prior agreement or implemented code.

## Robustness milestone

After `boosted-v1`, run the predeclared feature-sensitivity and paired-bootstrap experiment:

```cmd
python -m unittest discover -s tests -v
python scripts\run_robustness.py
```

See `docs/ROBUSTNESS_EXPERIMENT.md`. Months 6-7 remain sealed.


## Analyst console

The current package includes a React/TypeScript/Vite review console in `frontend/`. It consumes
the authenticated `/api/v1` workflow directly: ranked queue, case details, bounded explanation
snapshots, raw features, append-only history and role-gated review decisions. The interface never
loads the fraud target and does not present the uncalibrated model score as a fraud probability.

```bat
python -m uvicorn risklens_api.app:app --reload
cd frontend
npm install
npm run dev
```

For a single-process demo, run `npm run build`; on the next API start, FastAPI serves
`frontend/dist` at `http://127.0.0.1:8000/`. See `docs/ANALYST_CONSOLE.md` for the security and
verification boundaries.


## Governance review (development-only)

After the fixed PyTorch comparison, `scripts/run_governance_review.py` performs the final targeted
month-5 governance/error analysis before model freeze. It compares the no-age HistGB challenger to
the full reference with a paired bootstrap, reports coarse age and intended-balance-missingness
metrics under the same global 3% queue, and saves local false-negative/false-positive diagnostics.
It does not evaluate months 6-7 or alter the product queue. See `docs/GOVERNANCE_REVIEW.md`.

## Model freeze checkpoint

After the development-only governance review, `scripts/freeze_model.py` freezes the reviewed
`histgb_15_leaves/no_customer_age` artifact and operating policy before any reserved months 6-7
final evaluation. See `docs/MODEL_FREEZE.md`. The final-test runner is intentionally a separate
post-freeze checkpoint.

## One-time final holdout

After `reports/model_freeze.json` has been independently reviewed, `scripts/run_final_test.py`
opens reserved months 6-7 exactly once under the frozen no-age HistGB model and 3% monthly review
policy. It writes a durable opening marker before reading reserved rows and refuses a second run.
See `docs/FINAL_TEST.md`. Do not run this command until the freeze manifest has been approved.

## Production hardening

The analyst service now has an explicit production mode (`RISKLENS_ENV=production`) with PostgreSQL-only startup, trusted-host enforcement, disabled interactive API docs, security headers, no-store API caching, request IDs, request-size defense in depth, and database readiness probing. The original research dependency lock remains unchanged for provenance; deploy the API with `requirements-api.txt` so the completed PyTorch experiment does not inflate the serving image. See `docs/PRODUCTION_HARDENING.md`.

## Deployment

RiskLens includes a Render-ready production configuration in `render.yaml`. The deployed Python service serves the prebuilt React console, requires PostgreSQL in production, runs schema migrations before rollout, and gates health on `/ready`. Run `python scripts/check_deployment_readiness.py` before pushing deployment assets. See [`docs/RENDER_DEPLOYMENT.md`](docs/RENDER_DEPLOYMENT.md) for the full production and database-seeding procedure.
