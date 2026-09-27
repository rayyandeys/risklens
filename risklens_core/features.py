"""Feature contract v1; see docs/FEATURE_CONTRACT.md for rationale."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CONTRACT_VERSION = "baf-base-v1"
SCHEMA = """fraud_bool income name_email_similarity prev_address_months_count
current_address_months_count customer_age days_since_request intended_balcon_amount
payment_type zip_count_4w velocity_6h velocity_24h velocity_4w bank_branch_count_8w
date_of_birth_distinct_emails_4w employment_status credit_risk_score email_is_free
housing_status phone_home_valid phone_mobile_valid bank_months_count has_other_cards
proposed_credit_limit foreign_request source session_length_in_minutes device_os
keep_alive_session device_distinct_emails_8w device_fraud_count month""".split()
CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
EXCLUDED = ["fraud_bool", "month", "days_since_request", "device_fraud_count"]
FEATURES = [c for c in SCHEMA if c not in EXCLUDED]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
MINUS_ONE_MISSING = ["prev_address_months_count", "current_address_months_count",
                     "bank_months_count", "session_length_in_minutes", "device_distinct_emails_8w"]
MISSING_FIELDS = MINUS_ONE_MISSING + ["intended_balcon_amount"]
INDICATORS = [f"{c}__missing" for c in MISSING_FIELDS]
BINARY = ["email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
          "foreign_request", "keep_alive_session"]


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Stateless field-specific cleaning, also retained inside saved pipelines."""
    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError("Expected a DataFrame with unique feature columns")
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        result = X.loc[:, FEATURES].copy()
        for column in NUMERIC:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
            if np.isinf(result[column]).any():
                raise ValueError(f"Infinite numeric values in {column}")
        for column in MINUS_ONE_MISSING:
            result[column] = result[column].mask(result[column].eq(-1))
        result["intended_balcon_amount"] = result["intended_balcon_amount"].mask(
            result["intended_balcon_amount"].lt(0))
        for column in MISSING_FIELDS:
            result[f"{column}__missing"] = result[column].isna().astype(float)
        for column in BINARY:
            if not result[column].dropna().isin([0, 1]).all():
                raise ValueError(f"Non-binary values in {column}")
        for column in CATEGORICAL:
            # Keep absent categories as NaN for the training-fitted imputer.
            result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
        return result
