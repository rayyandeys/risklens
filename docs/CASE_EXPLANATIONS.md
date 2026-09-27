# Case explanations v1 — 27 September 2026

This release adds explanations to the existing analyst queue. It does not train a
model, change review capacity, evaluate labels, or make final-test claims.

## Upgrade the current Windows project

Stop Uvicorn with Ctrl+C, extract the ZIP in Downloads, then merge the contents of
its `risklens` folder into your existing project. Replace matching files and retain
`.venv`, `data`, `reports`, `artifacts`, `backups`, and `risklens.db`.
Dependencies have not changed. From the existing activated CMD terminal:

```cmd
python -m unittest discover -s tests -v
python scripts\migrate_database.py
python scripts\build_explanations.py --limit 50
python -m uvicorn risklens_api.app:app --reload
```

Expected local suite: 64 tests, 63 pass, one opt-in PostgreSQL test skipped.
Migration `0003_explanations` adds a table and preserves cases, reviews and credentials.
The migration CLI creates a SQLite backup first. Stop if it reports an error.
Do not rerun training, robustness, review-queue generation, or queue import.

The explanation builder verifies the review-summary hash against the database,
the dataset against the audit, and the saved model against the run's model hash.
Each new explanation must reproduce the stored queue score to absolute tolerance
1e-12; failure stops the build. Never edit hashes to bypass a failure.
Only load your own locally generated joblib artifact; joblib is not an untrusted
model interchange format.

The command reads `data/raw/Base.csv`, `reports/data_audit.json`,
`reports/review_workflow_summary.json`, and the saved `full.joblib` referenced there.
It reads only month/feature columns from the CSV, never fraud labels. Months 5–7
are excluded from the background sample; months 6–7 are never model-evaluated.
Selected queue cases come from the existing month-5 analyst database.

Default workload: top 50 ranks, 16 training background rows, 32 permutation paths
per case, 928 hybrid model rows per case (plus the standalone reproduction check).
Contributions are calculated offline and persisted per case; API reads are cheap.
The database commits each completed case so interrupted runs can resume safely.
Re-running identical arguments verifies and reuses existing snapshots. The final
complete report is written only after all requested cases succeed; if a run stops,
do not mistake an older reports/explanations_summary.json for that run's result.

Upload **reports/explanations_summary.json** after a successful build. It contains
coverage, reconstruction checks, reference provenance and duration, without tokens.
Artifacts under `artifacts/explanations/<build-id>/` retain the sampled label-free
training rows, background metadata, configuration and report. Keep them with the DB.

## Read explanations through the authenticated API

Authorize in http://127.0.0.1:8000/docs with your valid token. Existing credentials
survive the migration. If expired, issue another using manage_credentials.py.

- `GET /api/v1/runs/{run_id}/explanations/summary`: count queued cases with any
  explanation snapshot, plus remaining cases. Coverage is across configurations.
- `GET /api/v1/runs/{run_id}/cases/{case_id}/explanation`: newest snapshot for this
  case, with model/input/background/configuration fingerprints and attribution data.

Use `GET /api/v1/runs/{run_id}/cases?limit=1` to get the first case ID. Case IDs are
source-row identifiers, not queue ranks. Viewer, analyst and admin may read.
Unauthenticated requests return 401. A case without an explanation returns 404,
not a made-up explanation. Provenance/reconstruction mismatches return 409.
There is no public endpoint that starts expensive explanation work.

After reviewing the first batch, expand in bounded batches, for example:

```cmd
python scripts\build_explanations.py --offset 50 --limit 100
```

`offset` is zero-based queue rank offset. The initial 50-case report is not a claim
that all 3,579 cases are explained. Use the API coverage summary for overall status.
Changing `--background-size` or `--seed` creates a distinct configuration and keeps
old snapshots; the API serves the latest stored snapshot for each case.

## What the numbers mean

Method: `paired-permutation-v1`, a direct sampled implementation of permutation
Shapley estimation. For each of 16 training rows, begin at that row's feature values,
replace features with the case's values in a random order, and record each change
in model score. Repeat with the reverse feature order. Average the contributions.
This is our implementation, not a call to SHAP's TreeExplainer or an exact SHAP claim.

The 28 raw features are the players. Every hybrid row goes through the saved full
preprocessing/model pipeline. Missing-value indicators are recomputed from their
raw features, so a feature and its missing flag cannot be explained independently
using contradictory values. Category handling is the model's original encoding.
The order seed derives from the case ID and configured seed, so batching does not
change results. Reference sampling uses only training rows and is reproducible.

`reference_score + sum(feature contributions) = model_score` to tolerance 1e-10.
This identity checks arithmetic consistency; it does NOT prove attribution accuracy.
Positive contributions raise the score relative to the sampled training reference;
negative contributions lower it. Scores are raw, uncalibrated model outputs.
For example +0.02 means +0.02 model-score units, not a proven 2-point increase in
real-world fraud probability. Reference score is an average prediction on sampled
training rows, not the dataset fraud rate and not an individual's baseline risk.

`pair_std_dev` measures variation across the per-background paired contributions.
It is a diagnostic, not a confidence interval, not predictive uncertainty and not
an explanation-stability certification. Contributions can change with the sampled
background and feature dependencies. More samples may improve approximation but do
not solve the underlying interpretation limitations.

## Limitations that remain visible in API responses

- Attribution describes model behaviour, not fraud causality or evidence of intent.
- Hybrid feature combinations may be unrealistic when features are correlated.
- A small training reference sample is a development approximation.
- The current model is uncalibrated. Calibration remains a later milestone.
- Age/proxy attribution is not a fairness assessment. Subgroup/error analysis is pending.
- None of this authorizes automatic adverse decisions about real people.

## Verification and remaining work

Local tests cover hand-computed additive effects, equal sharing of a two-feature
interaction under reverse permutations, grouped missing flags, repeatability,
invalid scores and label rejection, training-only reference sampling, authenticated
reads, cache idempotency, provenance mismatch rejection, current-schema migration
preservation, and a complete CLI build/resume with a real saved HistGB pipeline.

The user already passed the real PostgreSQL authentication/import/concurrency gate
on PostgreSQL 17 before this release (1 test, 2.258 seconds). That establishes the
previous release's path; it is not evidence that this new table has been tested on
PostgreSQL. The integration gate now also writes and serves explanation JSON.
Rerun it against the same disposable PostgreSQL setup before deploying this release
on PostgreSQL. Offline PostgreSQL migration SQL compilation is tested locally.

Next: inspect actual BAF explanation timing, reproduction and coverage; review
approximation stability; implement the analyst UI. PyTorch comparison, calibration,
subgroup/error analysis, drift/monitoring, deployment and sealed test evaluation
remain required. The existing operational store is still SQLite; no data transfer
or cloud database provisioning occurs here.

Primary references:
- https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html
- https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html
