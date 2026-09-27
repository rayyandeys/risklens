"""Feature-ablation and paired bootstrap utilities for robustness-v1."""
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score

from .features import FEATURES, NUMERIC, INDICATORS, CATEGORICAL, FeatureCleaner
from .metrics import evaluate

ABLATIONS = {
    "full": (),
    "no_credit_risk_score": ("credit_risk_score",),
    "no_proposed_credit_limit": ("proposed_credit_limit",),
    "no_credit_risk_or_limit": ("credit_risk_score", "proposed_credit_limit"),
    "no_customer_age": ("customer_age",),
}


class CleanThenSelect(TransformerMixin, BaseEstimator):
    """Apply the canonical field cleaning, then expose only a declared raw-feature subset."""
    def __init__(self, features):
        self.features = tuple(features)

    def fit(self, X, y=None):
        self.transform(X)
        return self

    def transform(self, X):
        cleaned = FeatureCleaner().transform(X)
        selected = list(self.features)
        indicators = [f"{c}__missing" for c in self.features if f"{c}__missing" in INDICATORS]
        return cleaned.loc[:, selected + indicators]


def ablated_features(name):
    if name not in ABLATIONS:
        raise KeyError(f"Unknown ablation: {name}")
    removed = set(ABLATIONS[name])
    return [c for c in FEATURES if c not in removed]


def make_ablation_pipeline(name, max_iter=200):
    features = ablated_features(name)
    numeric = [c for c in NUMERIC if c in features]
    indicators = [f"{c}__missing" for c in features if f"{c}__missing" in INDICATORS]
    categorical_cols = [c for c in CATEGORICAL if c in features]
    categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                 encoded_missing_value=np.nan, dtype=np.float64)
    encoder = ColumnTransformer([
        ("numeric", "passthrough", numeric),
        ("missing_flags", "passthrough", indicators),
        ("categorical", categorical, categorical_cols),
    ], sparse_threshold=0)
    mask = [False] * (len(numeric) + len(indicators)) + [True] * len(categorical_cols)
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.08, max_iter=max_iter, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=1.0, max_bins=255,
        categorical_features=mask, early_stopping=False, class_weight=None, random_state=42)
    return Pipeline([
        ("clean_select", CleanThenSelect(features)),
        ("encode", encoder),
        ("model", model),
    ])


def _recall_at_fraction(y, scores, row_ids, fraction=0.03, seed=42):
    return evaluate(y, scores, row_ids, capacities=(fraction,), seed=seed)["review_capacity"][0]["recall"]


def paired_bootstrap_deltas(y, candidate_scores, reference_scores, row_ids, *,
                            repeats=1000, seed=42, capacity=0.03, ci=0.95):
    """Bootstrap validation rows jointly and estimate candidate-reference metric deltas.

    Duplicate sampled rows receive bootstrap-position IDs because evaluate() requires unique IDs.
    Tie-breaking therefore remains label-free and deterministic within each replicate.
    """
    y = np.asarray(y, dtype=int)
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)
    row_ids = np.asarray(row_ids)
    if not (y.shape == candidate_scores.shape == reference_scores.shape == row_ids.shape):
        raise ValueError("Bootstrap inputs must be aligned vectors")
    if len(y) == 0 or len(np.unique(y)) != 2:
        raise ValueError("Bootstrap requires a nonempty binary validation sample")
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap repeats")
    if not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")

    rng = np.random.default_rng(seed)
    recall_delta = np.empty(repeats, dtype=float)
    ap_delta = np.empty(repeats, dtype=float)
    completed = 0
    attempts = 0
    max_attempts = repeats * 20
    n = len(y)
    while completed < repeats and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if len(np.unique(ys)) != 2:
            continue
        candidate = candidate_scores[idx]
        reference = reference_scores[idx]
        bootstrap_ids = np.arange(n, dtype=np.int64)
        recall_delta[completed] = (
            _recall_at_fraction(ys, candidate, bootstrap_ids, capacity, seed)
            - _recall_at_fraction(ys, reference, bootstrap_ids, capacity, seed)
        )
        ap_delta[completed] = average_precision_score(ys, candidate) - average_precision_score(ys, reference)
        completed += 1
    if completed != repeats:
        raise RuntimeError("Could not generate enough valid bootstrap replicates")

    alpha = (1 - ci) / 2
    def summarize(values):
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "ci_lower": float(np.quantile(values, alpha)),
            "ci_upper": float(np.quantile(values, 1 - alpha)),
            "fraction_gt_zero": float(np.mean(values > 0)),
            "fraction_ge_zero": float(np.mean(values >= 0)),
        }
    return {
        "method": "paired nonparametric row bootstrap on validation month",
        "repeats": repeats,
        "seed": seed,
        "confidence_level": ci,
        "capacity": capacity,
        "recall_delta": summarize(recall_delta),
        "average_precision_delta": summarize(ap_delta),
    }
