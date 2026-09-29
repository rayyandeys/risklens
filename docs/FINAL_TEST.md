# Final holdout evaluation

`final-holdout-v1` is the one-time reserved-month evaluation for the immutable RiskLens freeze
`20260927T115644_227909Z`. It is deliberately separated from development experiments.

## Frozen inputs

- Model: `histgb_15_leaves/no_customer_age`
- Frozen model SHA-256: `65e9caa079ba46af72955a024b7ae0fd7d207e80c10eba686954ff149eb5ee12`
- Score: raw positive-class `predict_proba`; no calibration
- Reserved months: 6 and 7
- Primary capacity: 3%
- Budget: `floor(month_rows * 0.03)` independently for each reserved month
- Tie rule: label-free seeded permutation assigned in ascending original row ID order, seed 42
- No post-test model, feature, calibration, threshold or capacity selection

The **primary operational result** is Recall@3% aggregated across two independent monthly queues.
Month-level 1/3/5% metrics are also reported. Pooled AP, ROC-AUC, Brier and pooled-capacity metrics
are secondary descriptive summaries and do not replace the monthly operating policy.

No pass/fail threshold is invented after seeing the test. The point of the holdout is to report how
the previously frozen system generalized, even if the result is worse than development performance.

## One-time opening guard

Before reading any reserved rows, `scripts/run_final_test.py` verifies the exact freeze, dataset hash,
model hash and policy. It then writes `reports/final_test_opened.json`. If that marker or the final
report already exists, the runner refuses to execute again. Do not delete the marker to obtain a
second test result.

If the process is interrupted after the marker is written, stop and inspect the attempt rather than
rerunning or changing code. The marker is evidence that the holdout has already been opened.

## Historical one-time execution command

**Completed on 27 September 2026. Do not run this again for the current freeze.**
The commands below document the original procedure. Read
[`final_test_summary.json`](../reports/final_test_summary.json) for the saved result.

```bat
python -m unittest discover -s tests -v
python scripts\run_final_test.py
```

After completion, upload `reports/final_test_summary.json`. The output includes per-month metrics,
the primary monthly-capacity aggregate, secondary pooled metrics, validation-to-test deltas,
prediction provenance and limitations.
