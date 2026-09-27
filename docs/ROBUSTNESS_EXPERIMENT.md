# RiskLens robustness-v1

This development-only experiment is declared after `boosted-v1` and before any robustness results are inspected.

## Question

Does the provisional HistGradientBoosting model's validation performance survive removal of inputs whose deployment provenance or fairness implications deserve explicit scrutiny, and how uncertain is its measured advantage over the unweighted logistic reference?

## Fixed temporal protocol

- Train: months 0–4.
- Validation: month 5.
- Final test: months 6–7, still sealed.
- Same audited `Base.csv`, field-specific cleaning rules, review capacities, and label-free tie handling as earlier experiments.

## Predeclared ablations

1. `full`: current 28 raw model inputs.
2. `no_credit_risk_score`: remove `credit_risk_score`.
3. `no_proposed_credit_limit`: remove `proposed_credit_limit`.
4. `no_credit_risk_or_limit`: remove both fields.
5. `no_customer_age`: remove `customer_age`.

Every ablation uses the already selected development configuration: HistGradientBoosting with 15 leaves, 200 iterations, learning rate 0.08, minimum leaf size 100, L2=1, max bins 255, no class weighting, no random internal early-stopping split, and seed 42. This is a robustness test, not a new hyperparameter search.

## Why these fields

`credit_risk_score` and `proposed_credit_limit` were retained in the benchmark under explicit scoring-time assumptions rather than proven production provenance. Their removal tests dependence on those assumptions. `customer_age` is retained in the benchmark but receives a no-age ablation because subgroup/fairness analysis is planned later.

The experiment does **not** claim these fields are leakage, unfair, or unusable. It tests sensitivity to their inclusion.

## Uncertainty

A paired nonparametric bootstrap resamples month-5 validation rows jointly for HistGB-15 and the saved unweighted logistic predictions. It estimates 95% percentile intervals for:

- difference in recall at the assumed 3% review capacity;
- difference in average precision.

The bootstrap describes uncertainty from this validation sample. It is not a production-performance confidence guarantee and is not used to open the sealed test set.

## Command

```cmd
python -m unittest discover -s tests -v
python scripts\run_robustness.py
```

Default bootstrap repeats: 1,000. The run verifies the audited dataset, baseline reference, boosted reference, and exact reproducibility of the prior full HistGB-15 validation metrics before accepting the result.
