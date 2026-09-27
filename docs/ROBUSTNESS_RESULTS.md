# robustness-v1 — observed development results

Run: `20260927T070126_112700Z` on the audited Base.csv fingerprint already recorded by the project.
These are month-5 validation results only. Months 6-7 remained sealed.

## Feature ablations

| Variant | AP | ROC-AUC | Brier | Recall@3% | Fraud caught@3% |
|---|---:|---:|---:|---:|---:|
| full | 0.18833 | 0.89499 | 0.01056 | 0.42098 | 594 |
| no credit_risk_score | 0.18304 | 0.89332 | 0.01059 | 0.40822 | 576 |
| no proposed_credit_limit | 0.17150 | 0.89313 | 0.01067 | 0.41176 | 581 |
| neither risk score nor limit | 0.16500 | 0.89157 | 0.01071 | 0.40609 | 573 |
| no customer_age | 0.19180 | 0.89415 | 0.01054 | 0.42169 | 595 |

The no-age model gained one caught case at the 3% budget and higher AP, but performed worse than the
full model at 1% and 5% capacity. This experiment did not bootstrap full versus no-age, so it is kept
as a challenger rather than declared superior.

## Paired validation bootstrap: full HistGB-15 vs logistic_unweighted

1,000 paired nonparametric row-bootstrap replicates, seed 42, 95% percentile intervals:

- Delta Recall@3% mean: +0.02515; 95% interval: [+0.00935, +0.04209]; 99.8% of replicates > 0.
- Delta Average Precision mean: +0.02943; 95% interval: [+0.01945, +0.04019]; 100% of replicates > 0.

These intervals quantify validation-sample uncertainty only. They do not establish production
performance, causal superiority, fairness, or final-test performance.

## Development decision

Retain `histgb_15_leaves` with the full feature contract as the current development reference for the
first scoring/review workflow. Retain `no_customer_age` as a challenger for later governance and
subgroup analysis. Do not open months 6-7 yet.
