# Logistic baseline checkpoint — user-run validation results

Run: `20260927T061624_500821Z`, supplied 27 September 2026 as Pasted markdown(1).md.
Source is the user's baseline summary, not an independent rerun by the assistant.
Dataset SHA-256: `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809`.
Temporal-v1, feature contract baf-base-v1. Months 0–4 training, 5 validation, 6–7 reserved.

## Measured results

| Metric | Unweighted logistic | Balanced logistic |
| --- | ---: | ---: |
| Average precision | 0.15884631484401698 | 0.1548420389370944 |
| ROC-AUC | 0.8837108350067936 | 0.88290628540671 |
| Brier score (lower is better) | 0.010841437129148755 | 0.10568091456650877 |
| Fraud caught at 1% / 1,193 reviews | 306 | 295 |
| Fraud caught at 3% / 3,579 reviews | 558 | 550 |
| Fraud caught at 5% / 5,966 reviews | 695 | 695 |
| Fit time (seconds) | 11.650200600037351 | 15.526913000037894 |
| Iterations | 28 | 68 |

Validation has 119,323 applications and 1,411 fraud cases. Both models converged with no warnings.
Total reported run time: 40.55337839992717 seconds. All dependencies matched the pinned versions.
Git commit was null; source hashes were recorded. No final-test evaluation occurred.

Unweighted precision at 1/3/5% review: 25.6496%, 15.5909%, 11.6493%.
Unweighted recall at 1/3/5% review: 21.6867%, 39.5464%, 49.2558%.
At 5%, 5,271 reviewed cases were non-fraud and 716 fraud cases remained unreviewed.
Thus the run shows useful ranking signal, but substantial missed fraud and analyst workload.

The constant training-prior predictor had AP 0.01182504630289215, ROC-AUC 0.5,
Brier 0.011688635985701681. Its observed tied-ranking results are a particular seeded ordering;
use the separately reported random expectations for theoretical comparisons.

## Interpretation and diagnostics

Unweighted logistic is the current reference: slightly higher AP/ROC-AUC/recall at 1 and 3%,
equal recall at 5%, and much lower probability error. No uncertainty intervals or significance
claim establish whether the small ranking differences generalize. Brier score is an overall
probability-error measure and does not isolate calibration. Neither model is certified calibrated.
Class weighting is not automatically helpful merely because the dataset is imbalanced.

No feature-hash duplicates within training/validation or across those splits were reported.
No constant training features or unseen validation categories were reported. Hash screening
does not establish point-in-time availability of features or entity independence.
Correct cleaning found 493,751 missing intended balances among 675,666 training records,
in addition to the already recorded address/bank/session/device sentinels.

The UI, model comparisons, calibration, uncertainty, drift, subgroup/error analysis, operational
workflow and final-test assessment are still required. No production or final-test claim follows.
