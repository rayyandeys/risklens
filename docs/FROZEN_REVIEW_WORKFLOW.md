# Frozen analyst workflow realignment

The original Analyst Console milestone was intentionally built before final model governance. Its
`review-workflow-v1` run therefore preserves scores, ranks, explanations and the sample analyst
decision made with `histgb_15_leaves/full`.

The final immutable model is `histgb_15_leaves/no_customer_age`. After the one-time final holdout
was completed, the product must not silently continue presenting the older full-model queue as if
it came from the frozen model.

`build_frozen_review_queue.py` creates a **new** `review-workflow-v2` run over development month 5:

- loads the exact model artifact copied by `model_freeze.json`;
- verifies its SHA-256 and the completed final-holdout lineage;
- reproduces the frozen month-5 metrics;
- applies the frozen 3% capacity and seed-42 label-free tie rule;
- never reads reserved months 6-7 for the queue;
- keeps `fraud_bool` outside the analyst-facing CSV/database;
- leaves the earlier workflow run and its review history untouched.

After building the new queue, import it as a new database run and generate explanation snapshots
against that new run. The console orders runs newest-first, so the newly imported frozen-model run
becomes the default without destroying the historical run.

```bat
python scripts\build_frozen_review_queue.py
python scripts\import_review_queue.py --summary reports\frozen_review_workflow_summary.json
python scripts\build_explanations.py --review-summary reports\frozen_review_workflow_summary.json --summary reports\frozen_explanations_summary.json --limit 50
```

For the no-age model, `customer_age` remains part of the stored synthetic application record but is
not used by the model. The explanation builder verifies that its contribution is numerically zero.
Removing a direct age input does not establish fairness or remove possible proxy effects.
