# RiskLens

**Production-oriented fraud detection, model governance, monitoring, and analyst review system**

[Live Application](https://rayyan-risklens.onrender.com)

RiskLens is an end-to-end fraud-risk platform built around a strict temporal evaluation protocol rather than a conventional random train/test split. It combines machine-learning experimentation, frozen-model governance, one-time future holdout evaluation, drift monitoring, model explanations, authentication, and a persistent analyst review workflow backed by PostgreSQL.

The project uses the **Feedzai Bank Account Fraud benchmark** containing **1,000,000 applications**. Models are developed only on historical months, selected using a predefined operational metric, frozen before the final test window is opened, and evaluated once on future data.

Rather than treating the model score as an automatic fraud decision, RiskLens uses the model as a **ranking system for a capacity-constrained analyst review queue**.

---

## Tech Stack

**Machine Learning**

- Python
- scikit-learn
- HistGradientBoosting
- Logistic Regression
- PyTorch
- NumPy
- pandas
- bootstrap evaluation
- permutation-based local explanations

**Backend**

- FastAPI
- SQLAlchemy
- Alembic
- PostgreSQL
- Pydantic
- credential-based authentication

**Frontend**

- React
- TypeScript
- Vite

**Infrastructure**

- Neon PostgreSQL
- Render
- GitHub

---

# Final Holdout Results

The final model, feature set, score semantics, and operating capacity were frozen before the untouched future test window was opened.

| Metric | Final Holdout |
|---|---:|
| Test population | 205,011 |
| Fraud cases | 2,878 |
| Fraud prevalence | 1.40% |
| PR-AUC | 0.1851 |
| ROC-AUC | 0.8882 |
| Brier score | 0.01253 |
| Analyst review capacity | 3% per month |
| Applications reviewed | 6,150 |
| Fraud cases caught | 1,141 |
| Recall @ 3% | 39.65% |
| Precision @ 3% | 18.55% |
| Expected fraud caught by random review | ~86 |
| TP yield vs random review | ~13.2× |

At the same 3% review capacity, random review would be expected to capture approximately 3% of fraud cases. The frozen RiskLens ranking captures approximately **39.65%**.

The model output is used as a **ranking score** and is **not presented as a calibrated probability of fraud**.

---

# System Architecture

```mermaid
flowchart LR
    A[Feedzai BAF Dataset] --> B[Data Integrity + Feature Contract]

    B --> C[Temporal Split]

    C --> D[Logistic Regression]
    C --> E[HistGradientBoosting]
    C --> F[PyTorch MLP]

    D --> G[Validation Evaluation]
    E --> G
    F --> G

    G --> H[Bootstrap + Calibration + Stability Analysis]

    H --> I[Governance Review]

    I --> J[Frozen HistGradientBoosting Model]

    J --> K[One-Time Future Holdout]
    J --> L[3% Analyst Review Queue]
    J --> M[Permutation Explanations]

    L --> N[FastAPI]
    M --> N

    N --> O[(PostgreSQL / Neon)]
    N --> P[React Analyst Console]

    Q[Feature Drift Monitoring] --> N
    R[Score Drift Monitoring] --> N
    K --> S[Generalization Evidence]
    S --> N

    P --> T[Review / Escalate / Audit History]