# RiskLens calibration + uncertainty experiment — v1

Status: implementation prepared on 27 September 2026. Actual BAF results are **not** recorded in
this document until the user runs the command against the audited local artifacts.

## Why this milestone exists

The current HistGB-15 score is useful for ranking, but a value such as `0.8425` must not be called an
84.25% fraud probability merely because it lies in `[0,1]`. Calibration asks whether predicted
probabilities agree with observed frequencies. Stability asks a different question: how sensitive
scores and the 3% analyst queue are to perturbations of the observed training sample.

Neither analysis proves causality, fairness, future performance, or production readiness.

## Leakage-safe calibration design

The sealed test months 6-7 are never read. The final development reference remains the saved
HistGB-15/full model trained on months 0-4 and evaluated on month 5.

Calibration fitting does **not** reuse month-5 labels. Instead, the script creates temporal
out-of-fold predictions inside the training period:

- train months 0-1 -> score month 2;
- train months 0-2 -> score month 3;
- train months 0-3 -> score month 4.

Those out-of-fold scores and labels fit two predeclared mappings:

1. sigmoid / Platt-style calibration on the raw score logit;
2. isotonic regression with monotonic clipping outside the observed score range.

The saved full model then produces the unchanged raw month-5 scores. Raw, sigmoid and isotonic
probabilities are compared on month 5 using Brier score, log loss, fixed-width 10-bin ECE, maximum
calibration gap and reliability-bin details. The development selection rule is declared before the
run: lowest Brier score, then ECE, then a conservative deterministic method-name preference.
`raw` is allowed to win; calibration is not assumed to help.

The existing review queue is **not silently rewritten**. Review-workflow-v1 remains ranked by the
raw HistGB score until the model/protocol is frozen. Sigmoid is strictly monotone and should retain
ranking; isotonic may introduce ties. Queue overlap versus raw is therefore recorded explicitly.

## Stability / uncertainty design

This is a bounded training-sample sensitivity analysis, not a posterior probability model.
For each repeat, training rows are resampled with replacement **inside each month x target-label
cell**, preserving that cell's observed size. The same full HistGB-15 configuration is refitted and
month 5 is rescored.

Default: 12 refits. The script records:

- empirical score mean/std and 10th/90th percentiles per validation case;
- frequency with which each case enters the 3% queue;
- queue Jaccard overlap against the saved reference queue;
- fraction of the reference queue retained by each refit;
- Spearman score-rank correlation versus the reference;
- number of boundary-unstable cases whose selection frequency is between 10% and 90%.

The 10th/90th empirical ranges are **not confidence intervals**. Results are conditional on this
bootstrap scheme, model class, feature contract and the observed development period.

## Command

After merging the package into the current project, preserve `.venv`, `data`, `reports`,
`artifacts`, `risklens.db` and the current frontend build. No dependencies changed.

```bat
python -m unittest discover -s tests -v
python scripts\run_calibration_uncertainty.py
```

The second command is intentionally heavier than the earlier analysis steps because it performs
three temporal calibration refits plus 12 stability refits. Do not interrupt it merely because a
single fit takes tens of seconds. It prints progress after every stability fit.

Outputs:

- `reports/calibration_uncertainty_summary.json`
- timestamped `artifacts/calibration_uncertainty/<run_id>/summary.json`
- `temporal_oof_calibration_scores.csv`
- `validation_calibrated_scores.csv`
- `score_calibrators.joblib`
- `stability_cases.csv`
- `stability_scores.npz`

Upload or paste `reports/calibration_uncertainty_summary.json` after completion. Do not evaluate
months 6-7 yet.
