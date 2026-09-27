# RiskLens model/protocol freeze v1

This checkpoint is intentionally separate from final-test evaluation. It consumes only the already
reviewed development evidence and re-scores validation month 5 to verify the exact artifact that will
be frozen. It does **not** evaluate or inspect target labels from reserved months 6-7.

## Reviewed decision

Freeze `histgb_15_leaves/no_customer_age` rather than the full HistGB model. The governance review
found essentially tied aggregate development performance, with the no-age variant catching one more
fraud case at the 3% budget and slightly higher AP / slightly lower Brier point estimates. The paired
bootstrap did not establish statistical superiority. The reason to remove `customer_age` is therefore
a governance/engineering choice to avoid direct use of age when no aggregate development loss was
observed — **not** a fairness certification. Correlated proxies and subgroup differences can remain.

The frozen score remains the raw positive-class `predict_proba` output and is displayed as a **model
score**, not a calibrated fraud probability. The frozen review budget is 3%, using `floor(N * 0.03)`
and the existing seeded, label-free row-ID tie rule with seed 42.

## Run

```bat
python -m unittest discover -s tests -v
python scripts\freeze_model.py
```

The command writes `reports/model_freeze.json` and copies the selected model byte-for-byte into a
new timestamped `artifacts/model_freeze/<freeze_id>/selected_model.joblib` directory. The manifest is
immutable: the script refuses to overwrite an existing `reports/model_freeze.json`.

Upload the manifest for review. Do **not** run a final-test command until the freeze manifest has been
reviewed. The final-test runner is intentionally not included in this package.
