# torch-comparison-v1

## Question

Does a reasonable fixed neural tabular baseline add measurable development value beyond the current
HistGB-15 reference, without touching the sealed months 6-7?

This experiment exists because RiskLens deliberately includes a PyTorch comparison, not because a
neural network is assumed to be superior for structured tabular data.

## Protocol

- Training: months 0-4 only.
- Development evaluation: month 5 only.
- Final test: months 6-7 remain sealed.
- Same `baf-base-v1` raw feature contract as the current model.
- Canonical sentinel cleaning runs before neural preprocessing.
- Numeric inputs: training-only median imputation and standardization.
- Six canonical missing indicators are retained.
- Categorical inputs: training-only most-frequent imputation + one-hot encoding; unseen month-5
  categories are ignored rather than learned from validation.
- MLP: 128 -> 64 hidden units, GELU activations, 0.10 dropout, one output logit.
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4.
- Loss: unweighted binary cross entropy with logits.
- Fixed schedule: 15 epochs, batch size 8192, seed 42. No month-5 early stopping.
- CPU is the default device for repeatability. `--device auto` is optional but hardware may change
  exact floating-point behavior.

The fixed schedule prevents the neural experiment from becoming an open-ended hyperparameter hunt.
A stronger tuned deep-tabular model could exist; that is not the question this baseline answers.

## Comparison

Month-5 metrics are AP, ROC-AUC, Brier score and the existing 1/3/5% capacity metrics. The original
predeclared development ordering remains recall@3%, then AP. In addition, 1,000 paired validation-row
bootstrap replicates estimate PyTorch-minus-HistGB deltas for recall@3% and AP.

The bootstrap is conditional on these already-trained models and on month 5. It does not capture
training-seed uncertainty, future drift or final-test uncertainty.

## Artifacts

`reports/torch_comparison_summary.json` is the compact handoff report. Timestamped artifacts retain:

- train-fitted neural preprocessor;
- PyTorch state bundle;
- training-loss history;
- month-5 scores for both models;
- comparison table and full provenance.

Do not call raw sigmoid output a calibrated fraud probability. It is a model score until separate
calibration evidence supports probability language.
