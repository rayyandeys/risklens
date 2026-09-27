# Temporal drift and monitoring — `temporal-monitoring-v1`

## Goal

Turn RiskLens' offline model into a system that can answer a production-style question: **has the
population or model-score distribution changed enough to deserve investigation?** This milestone
adds a label-safe monitoring snapshot and a separate delayed-label performance backtest. It does
not open the sealed test months and it does not automatically retrain the model.

## Primary monitoring comparison

The primary snapshot compares **validation month 5** with the pooled **training months 0–4**.
Feature drift and score drift are label-free. The frozen `histgb_15_leaves/full` artifact is used
for score-distribution monitoring; the operational score remains the raw score because the prior
calibration experiment selected `raw` on development data.

For each transformed model feature, the monitor records population stability index (PSI),
missingness change, and—for numeric features—the two-sample KS statistic. Categorical features
also record the fraction of target values that were not present in the reference population.
Reference-defined bins/categories are reused for the target; target data never defines its own
comparison bins.

The default dashboard triage heuristics are deliberately explicit:

- PSI `< 0.10`: normal; `0.10–0.25`: watch; `>= 0.25`: alert.
- Absolute missing-rate change `< 2 percentage points`: normal; `2–5 pp`: watch; `>= 5 pp`: alert.

These values are **engineering heuristics**, not hypothesis tests, legal/regulatory limits, or
claims that a particular amount of drift necessarily harms the model. Production thresholds should
be revisited once real monitoring history and operating costs exist.

## Month-over-month history

The command also compares month `0→1`, `1→2`, ..., `4→5` and saves the feature tables plus a compact
trend CSV. This makes a one-off month-5 alert interpretable in the context of whether change was
sudden or accumulated gradually.

## Prospective delayed-label backtest

Drift is not the same as performance. A separate expanding-window backtest evaluates the fixed
HistGB-15 configuration as if time were moving forward:

- train month 0 → score month 1
- train months 0–1 → score month 2
- train months 0–2 → score month 3
- train months 0–3 → score month 4
- train months 0–4 → score month 5

Each target month is therefore scored only by earlier months. The report tracks prevalence, AP,
ROC-AUC, Brier score, precision/recall at the assumed 3% review capacity and the queue cutoff. The
month-5 fold must reproduce the frozen full HistGB model to numerical tolerance.

This backtest is label-aware and is **not** used to determine the real-time monitoring status. In a
real fraud system labels arrive later, so feature/score monitoring can run immediately while
performance monitoring runs on a delayed cadence.

## Outputs

`python scripts/run_temporal_monitoring.py` writes a timestamped directory under
`artifacts/temporal_monitoring/` containing:

- `monitoring_snapshot.json` — compact, label-safe payload intended for the later Monitoring UI.
- `feature_drift_train0_4_to_month5.csv` — primary feature-level comparison.
- `feature_drift_monthX_to_monthY.csv` — month-over-month feature evidence.
- `month_over_month_drift.csv` — compact drift trend.
- `monthly_score_distribution.csv` — frozen-model score distribution by month.
- `prospective_backtest.csv` — expanding-window delayed-label performance trend.
- `summary.json` — complete provenance and limitations.

A convenience copy is written to `reports/temporal_monitoring_summary.json`.

## Non-claims

- A PSI alert is not proof that the model is wrong.
- A stable PSI is not proof that the model is safe.
- KS values are descriptive here; no repeated hypothesis-test significance claims are made.
- The final model has not been certified for production.
- Analyst decisions are not ground-truth labels and are not used to train this monitor.
- Months 6–7 remain sealed for the later one-time final evaluation.
