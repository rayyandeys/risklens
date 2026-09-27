# Review workflow result — 27 September 2026

Run `20260927T073240_094094Z` completed successfully over validation month 5.

- Population scored: 119,323 applications.
- Assumed analyst capacity: 3%; 3,579 cases selected.
- Saved HistGB-15/full model reproduced the prior probabilities to a maximum absolute difference of approximately `1.0e-16`.
- At 3% capacity: 594 of 1,411 fraud cases were inside the review queue; recall 0.420978 and precision 0.165968.
- `fraud_bool` was absent from both the scored population and analyst queue; ground truth was isolated in a separate offline-evaluation file.
- Months 6-7 remained sealed (`test_evaluated=false`).

This establishes a reproducible development queue, not a production deployment. Persistence, API, explanations, authentication/UI and monitoring follow separately.
