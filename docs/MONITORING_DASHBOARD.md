# Monitoring dashboard v1

RiskLens now exposes the completed experiment lineage through an authenticated, read-only monitoring page in the analyst console. The page is an evidence projection, not a new experiment: it reads existing generated reports and does not score rows, retrain models, open labels, modify decisions, or change the frozen protocol.

## Data sources

`GET /api/v1/monitoring/overview` requires a valid RiskLens credential and validates these local reports before returning data:

- `reports/calibration_uncertainty_summary.json`
- `reports/temporal_monitoring_summary.json`
- `reports/model_freeze.json`
- `reports/final_test_summary.json`

The endpoint verifies a shared dataset fingerprint and feature contract, verifies the final holdout is linked to the active freeze, hashes `model_freeze.json` and checks that exact hash against the final-test report, and checks that the frozen and final-tested model names/hashes match. Missing or inconsistent evidence returns HTTP 503 rather than rendering a misleading dashboard.

## What the page shows

The page separates four concepts that should not be conflated:

1. **Population drift.** Month-5 feature drift from the existing temporal-monitoring study, including severity counts, PSI, KS statistics, missingness changes and month-over-month summaries.
2. **Development score drift.** The score-drift signal from the pre-freeze full HistGB development reference. The UI labels this source explicitly; it is not presented as a frozen no-age score-drift study.
3. **Frozen-model generalization.** Month-5 frozen no-age validation metrics followed by month-6 and month-7 sealed final-holdout metrics at the immutable 3% operating point.
4. **Governance and stability.** Immutable freeze/final-opening lineage plus the earlier development bootstrap sensitivity study, clearly labelled as pre-freeze development evidence.

The page keeps the final result descriptive. It does not introduce a post-hoc pass/fail threshold and does not use final-test results to alter the model, feature set, calibration, capacity or tie rule.

## Local verification

After merging this release, preserve your local `reports/`, `artifacts/`, database, raw data and frontend dependencies. Run:

```bat
python -m unittest discover -s tests -v
cd frontend
npm run build
cd ..
python -m uvicorn risklens_api.app:app --reload
```

Open `http://127.0.0.1:8000/`, authenticate, then select **Monitoring** in the sidebar. No `npm install` is required if the existing frontend dependencies from the analyst-console milestone are still present; this release adds no JavaScript dependencies.

## Evidence boundaries

- Feature drift is not proof of model failure, unfairness or fraud causation.
- The score-drift section is explicitly sourced from the pre-freeze full HistGB study.
- Bootstrap queue stability is a development sensitivity analysis, not a confidence interval or posterior uncertainty estimate.
- Final holdout results are from the synthetic Feedzai BAF benchmark and do not establish production or external validity.
- The 3% review budget remains an engineering project assumption rather than measured analyst staffing capacity.
