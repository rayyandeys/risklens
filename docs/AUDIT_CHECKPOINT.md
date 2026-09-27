# User-reported Base.csv audit — 27 September 2026

Input hash: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`.
Source: user pasted the output from their local run of scripts/audit_data.py.
The assistant has not received or independently run the one-million-row CSV.

| Partition | Months | Rows | Fraud rows | Fraud rate |
| --- | --- | ---: | ---: | ---: |
| Training | 0–4 | 675,666 | 6,740 | 0.997534% |
| Validation | 5 | 119,323 | 1,411 | 1.182505% |
| Reserved test | 6–7 | 205,011 | Not inspected | Not inspected |

Month counts: 0=132440, 1=127620, 2=136979, 3=150936, 4=127691,
5=119323, 6=108168, 7=96843. Total: 1,000,000, 32 columns, no exported index columns.
All screened training columns had zero literal-null tokens.

Nonzero exact -1 counts in training: previous address 469907; current address 2827;
credit risk score 438; bank months 172287; session length 1278; device distinct emails 276.
These are raw audit counts, not a uniform missing-value definition. Credit-score -1 is valid.
All negative intended balances must additionally be counted by the new field-specific cleaner.

The prevalence change is a descriptive observation, not evidence by itself of model failure,
concept drift, or statistically established distribution shift. No trained-model metrics exist yet.
