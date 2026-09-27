# RiskLens governance review — development-only v1

This is the last targeted development analysis before the model/protocol freeze. It does **not**
open months 6-7 and it does not alter the analyst queue, database or product model.

The review exists because the full HistGB-15 model retained `customer_age`, while the earlier
predeclared no-age ablation was almost tied on Recall@3% and slightly higher on AP. Rather than
silently choosing either variant, this step directly compares no-age versus full with a paired
month-5 bootstrap and reports coarse age-group behavior under the same global 3% review policy.

It also reports intended-balance-missingness groups because missingness is material in this dataset,
and creates local error-analysis artifacts for false negatives and false positives. These are
benchmark diagnostics, not fairness certification or causal conclusions.

## Run

```bat
python -m unittest discover -s tests -v
python scripts\run_governance_review.py
```

Upload `reports\governance_review_summary.json`. Do not open the final test months after the command;
model/protocol freeze is a separate explicit checkpoint.

## Interpretation rules

- One global 3% queue is formed per model. No group receives its own threshold or budget.
- Age groups are deliberately coarse (`<50`, `>=50`, missing) to avoid post-hoc slicing.
- The no-age bootstrap is validation-row sampling uncertainty only; it is not a fairness test.
- Group recall/selection/FPR differences are descriptive and can reflect prevalence and other
  correlated features. The synthetic BAF benchmark cannot certify production fairness.
- Error CSVs are local analysis artifacts and must not be fed back into the product as labels or
  analyst decisions.
