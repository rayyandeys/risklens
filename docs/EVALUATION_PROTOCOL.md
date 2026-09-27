# RiskLens evaluation protocol — temporal v1

Status: chronology and baseline feature handling frozen on 27 September 2026 after reading the
official datasheet and user-reported data audit, before baseline results. See FEATURE_CONTRACT.md.
This is RiskLens's protocol, not reproduction of Feedzai's published split or results. Subsequent
model/calibration experiments require their own recorded configuration before using final test.

## Question

How much account-opening fraud can a model identify when analysts can review only 1%, 3%, or 5%
of applications, and how does that performance change over time?

BAF is a generated benchmark derived from anonymized real-world bank account opening data.
Results on it support benchmark conclusions, not claims of performance on a live bank.
Source: https://github.com/feedzai/bank-account-fraud

## Frozen chronology

| Partition | Months | Role |
| --- | --- | --- |
| Training | 0–4 | Fit model and all learned preprocessing |
| Validation | 5 | Select model, hyperparameters, and operating policy |
| Test | 6–7 | Final locked evaluation, including separate monthly results |

The reported audit confirms month encoding and coverage. The starter requires integer
months 0–7 and both target classes in training and validation. It rejects other input instead
of silently changing the protocol. It reports row counts for all months but withholds test
target distributions and performance. Structural validation is not feature selection.

Exclude `fraud_bool` from predictors. Initially use `month` only for splitting/reporting.
Investigate exported index columns before excluding them. Review every feature for meaning,
sentinel values, and availability at the time an account application is scored. Check duplicate
and repeated-entity risks where identifiers exist. No arbitrary random split or oversampling
before splitting. Never concatenate dataset variants without studying their overlap.

## Comparisons and metrics

1. Establish random-ranking expectations and a simple logistic regression baseline.
2. Compare gradient-boosted trees under the same split and features.
3. Add the planned PyTorch comparison with a bounded training/tuning budget; report if it loses.

Primary operational metric: recall within review capacity, with precision and actual reviewed
counts at 1%, 3%, and 5%. Report average precision (explicitly named, not ambiguously labelled
as a different PR-AUC definition), ROC-AUC as secondary, confusion counts, and prevalence.
Define rounding and deterministic tie-breaking for top-k ranking; ties must not use labels.
Measure capacity within each test month as well as any pooled summary.

A top-k simulation selects by score within a batch and enforces a budget. A fixed score threshold
does not guarantee the same review fraction under drift. Evaluate and label these policies separately.
Do not optimize test thresholds to obtain an attractive result.

Use training-only transformations and keep calibration fitting separate from final assessment.
Plan calibration partitions or temporal cross-validation before fitting a calibrator; do not
reuse final-test data for it. Report Brier score and reliability plots when probabilities exist.
Class weighting may alter calibration: evaluate it rather than assume probabilities are calibrated.

Add temporal error analysis, subgroup analysis justified by the data, drift diagnostics, and
uncertainty intervals with the sampling unit and dependence assumptions stated. A drift signal
does not automatically prove a drop in predictive performance.

## Reproducibility

Every reported run should record dataset hash and version, split version, feature list,
preprocessing, model/configuration, seed, dependency versions, code commit, runtime, hardware,
and raw metric outputs. Only measured results may appear in the README or resume.

## Product evidence to add later

Ranked review queue, case detail with model explanations, analyst decisions and audit history,
model-version-aware scoring API, persistence, and monitoring. Explanations describe model behaviour,
not proof of causation. Analyst decisions are simulated in the demo and are not automatically
ground truth or permission to retrain. Benchmark truth must not leak into scoring inputs.

The UI should make review actions fast, legible, keyboard-accessible, and easy to audit. Use
motion to explain state changes and make the public introduction polished; preserve reduced-motion
support and a responsive fallback for any 3D scene. No component library has been integrated yet.
