# Boosted comparison v1 — declared before observing tree results

Purpose: compare nonlinear models against the measured logistic reference under temporal-v1
and baf-base-v1. No new dependencies or data are needed. This is a bounded two-configuration
experiment, not an exhaustive search or a claim that trees must beat logistic regression.

Model: scikit-learn 1.8 HistGradientBoostingClassifier. Fixed settings: learning_rate=0.08,
200 boosting iterations, min_samples_leaf=100, l2_regularization=1, max_bins=255,
class_weight=None, seed=42. Compare max_leaf_nodes=15 and 31. Early stopping is disabled:
fit only months 0–4 and avoid the estimator's default random internal validation partition.
The first run will use all 200 iterations; keep the fixed budget even if results are disappointing.

Trees use the identical 28 raw input features and sentinel cleaning, with six missing indicators.
They handle numeric missingness natively; unlike logistic, they need no scaling or median imputation.
Training-fitted OrdinalEncoder maps the five categorical fields to codes. An explicit mask tells
the estimator they are categories rather than ordered numeric values. Unseen/missing categories
become NaN and use the tree's missing-value handling. Save this whole preprocessing/model pipeline.
Documenting these family-specific transformations is part of a fair comparison.

Before comparing stored baseline metrics, check completion/convergence, dataset fingerprint,
feature contract, features, split, holdout flag, and validation counts. Save the reference report's
hash and run ID. The original baseline report and model files are not overwritten or retrained.

## Provisional ranking rule

Primary: validation recall at 3% review capacity. Secondary: average precision. Final exact tie:
model name, for deterministic output. The 3% budget is a project operating assumption, selected
after reviewing logistic baselines but before observing any boosted results. It is not measured
bank staffing capacity. Continue reporting 1% and 5%, precision, Brier score and ROC-AUC.
Selection ranks four models: both logistic references and both new tree configurations.

Do not call this final model selection: uncertainty, calibration, subgroup/error analysis, PyTorch,
feature ablations and temporal stability work remain. Do not inspect months 6–7 to choose settings.
Warnings make the run require review even if metrics were generated. No probabilities are assumed
calibrated merely because ranking improved.

## Run

```bat
python -m unittest discover -s tests -v
python scripts\train_boosted.py
```

Upload `reports/boosted_summary.json`. Keep timestamped `artifacts/boosted` outputs:
two models, validation predictions, comparison.csv, summary.json. Timings, actual iterations,
warnings, source hashes, environment, and baseline reference are recorded.

## Sources

- https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
