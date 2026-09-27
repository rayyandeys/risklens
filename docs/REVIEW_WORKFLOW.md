# review-workflow-v1

## Goal

Turn the validated model into the first end-to-end RiskLens product slice without pretending that a
validation benchmark is production. The workflow scores month 5 with the saved full HistGB-15 model,
reproduces the saved robustness probabilities, ranks cases with the same deterministic label-free tie
policy used in evaluation, and selects the top 3% for analyst review.

## Label separation

`fraud_bool` is never written to `analyst_queue.csv` or `scored_population.csv`. Ground truth is
written to `offline_ground_truth.csv` only for later offline evaluation. This prevents a demo analyst
workflow from leaking the answer into the decision surface.

## Outputs

- `scored_population.csv`: all validation applications, risk scores/ranks, queue state, raw model inputs.
- `analyst_queue.csv`: only applications inside the assumed review budget.
- `offline_ground_truth.csv`: isolated case labels for evaluation, not for analyst display.
- `analyst_decisions_template.csv`: case IDs plus blank decision/note fields.
- `summary.json` and `reports/review_workflow_summary.json`: provenance, hashes, reproduction checks,
  policy, metrics and output paths.

Allowed review decisions in the current domain layer are `confirmed_fraud`, `legitimate`, and
`escalate`. Persistence, identities/authentication, concurrency, API endpoints, explanations and the
React analyst interface are intentionally not faked in this milestone; they follow after this slice
is verified.

## Run

```bat
python -m unittest discover -s tests -v
python scripts\build_review_queue.py
```

The command must reproduce the saved robustness predictions and metrics before it creates a queue.
Months 6-7 remain sealed.
