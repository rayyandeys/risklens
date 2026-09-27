"""Calibration and training-sample stability utilities for RiskLens.

The utilities in this module are development-only. They never read the sealed test months.
Calibration is learned from temporally out-of-fold training scores and evaluated on month 5.
Stability is a sensitivity analysis over month/label-stratified bootstrap refits of the same
HistGB-15 configuration; it is not a Bayesian posterior or a confidence interval for fraud risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

from .metrics import evaluate
from .review import deterministic_ranking

EPS = 1e-6


def _validated_binary_scores(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or len(y) == 0:
        raise ValueError("Labels and scores must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Calibration requires both binary classes")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    return y, scores


def clipped_logit(scores, eps=EPS):
    scores = np.asarray(scores, dtype=float)
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    clipped = np.clip(scores, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


@dataclass
class SigmoidScoreCalibrator:
    """Platt-style calibration over a model's log-odds scores."""

    eps: float = EPS

    def fit(self, scores, y):
        y, scores = _validated_binary_scores(y, scores)
        x = clipped_logit(scores, self.eps)

        # Parameterize slope as exp(log_slope) so the fitted mapping is strictly
        # increasing and therefore cannot silently reverse the review ranking.
        prevalence = np.clip(float(y.mean()), self.eps, 1.0 - self.eps)
        initial = np.array([np.log(prevalence / (1.0 - prevalence)), 0.0], dtype=float)

        def objective(theta):
            intercept, log_slope = theta
            slope = np.exp(log_slope)
            z = intercept + slope * x
            # Stable Bernoulli negative log-likelihood.
            loss = np.logaddexp(0.0, z) - y * z
            p = expit(z)
            residual = p - y
            grad_intercept = residual.sum()
            grad_log_slope = (residual * x * slope).sum()
            return float(loss.sum()), np.array([grad_intercept, grad_log_slope], dtype=float)

        result = minimize(
            fun=lambda theta: objective(theta)[0],
            x0=initial,
            jac=lambda theta: objective(theta)[1],
            method="BFGS",
            options={"maxiter": 1000, "gtol": 1e-8},
        )
        if not result.success and np.linalg.norm(result.jac) > 1e-5:
            raise ValueError(f"Sigmoid calibration failed to converge: {result.message}")
        self.intercept_ = float(result.x[0])
        self.slope_ = float(np.exp(result.x[1]))
        self.optimizer_message_ = str(result.message)
        return self

    def predict(self, scores):
        if not hasattr(self, "intercept_"):
            raise ValueError("SigmoidScoreCalibrator must be fitted before predict")
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Scores must be finite probabilities in [0, 1]")
        z = self.intercept_ + self.slope_ * clipped_logit(scores, self.eps)
        return expit(z)

    def parameters(self):
        if not hasattr(self, "intercept_"):
            raise ValueError("SigmoidScoreCalibrator must be fitted before parameters")
        return {
            "intercept": float(self.intercept_),
            "slope_on_raw_logit": float(self.slope_),
            "clip_epsilon": float(self.eps),
            "optimizer_message": self.optimizer_message_,
        }


@dataclass
class IsotonicScoreCalibrator:
    """Monotone nonparametric score calibrator."""

    def fit(self, scores, y):
        y, scores = _validated_binary_scores(y, scores)
        if len(np.unique(scores)) < 2:
            raise ValueError("Isotonic calibration requires at least two distinct scores")
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model_.fit(scores, y)
        return self

    def predict(self, scores):
        if not hasattr(self, "model_"):
            raise ValueError("IsotonicScoreCalibrator must be fitted before predict")
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Scores must be finite probabilities in [0, 1]")
        return np.asarray(self.model_.predict(scores), dtype=float)

    def parameters(self):
        if not hasattr(self, "model_"):
            raise ValueError("IsotonicScoreCalibrator must be fitted before parameters")
        return {
            "threshold_count": int(len(self.model_.X_thresholds_)),
            "x_min": float(self.model_.X_thresholds_[0]),
            "x_max": float(self.model_.X_thresholds_[-1]),
        }


def calibration_bins(y, scores, *, bins=10):
    """Return fixed-width reliability bins and ECE diagnostics."""
    y, scores = _validated_binary_scores(y, scores)
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    # score==1 belongs to the final bin.
    bucket = np.minimum(np.searchsorted(edges, scores, side="right") - 1, bins - 1)
    bucket = np.maximum(bucket, 0)
    rows = []
    weighted_gap = 0.0
    max_gap = 0.0
    n = len(y)
    for i in range(bins):
        mask = bucket == i
        count = int(mask.sum())
        if count:
            mean_score = float(scores[mask].mean())
            observed = float(y[mask].mean())
            gap = abs(mean_score - observed)
            weighted_gap += (count / n) * gap
            max_gap = max(max_gap, gap)
        else:
            mean_score = None
            observed = None
            gap = None
        rows.append({
            "bin": i + 1,
            "lower": float(edges[i]),
            "upper": float(edges[i + 1]),
            "count": count,
            "mean_score": mean_score,
            "observed_rate": observed,
            "absolute_gap": None if gap is None else float(gap),
        })
    return {
        "bins": rows,
        "ece": float(weighted_gap),
        "max_calibration_gap": float(max_gap),
    }


def calibration_metrics(y, scores, row_ids, *, bins=10):
    """Operational and probabilistic diagnostics for one score vector."""
    y, scores = _validated_binary_scores(y, scores)
    row_ids = np.asarray(row_ids)
    if row_ids.shape != y.shape:
        raise ValueError("row_ids must align with labels")
    reliability = calibration_bins(y, scores, bins=bins)
    bounded = np.clip(scores, EPS, 1.0 - EPS)
    return {
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, bounded, labels=[0, 1])),
        "ece": reliability["ece"],
        "max_calibration_gap": reliability["max_calibration_gap"],
        "reliability_bins": reliability["bins"],
        "operational": evaluate(y, scores, row_ids),
    }


def fit_score_calibrators(oof_scores, y):
    """Fit the two predeclared calibrators from temporally out-of-fold training scores."""
    y, oof_scores = _validated_binary_scores(y, oof_scores)
    sigmoid = SigmoidScoreCalibrator().fit(oof_scores, y)
    isotonic = IsotonicScoreCalibrator().fit(oof_scores, y)
    return {"sigmoid": sigmoid, "isotonic": isotonic}


def select_calibration_method(reports):
    """Predeclared development rule: lowest Brier, then ECE, then conservative name order."""
    required = {"raw", "sigmoid", "isotonic"}
    if set(reports) != required:
        raise ValueError(f"Calibration reports must be exactly {sorted(required)}")
    preference = {"raw": 0, "sigmoid": 1, "isotonic": 2}
    for name, report in reports.items():
        for key in ("brier_score", "ece"):
            value = report.get(key)
            if value is None or not np.isfinite(value):
                raise ValueError(f"Calibration report {name} lacks finite {key}")
    return min(reports, key=lambda name: (
        reports[name]["brier_score"], reports[name]["ece"], preference[name]
    ))


def queue_selection(scores, row_ids, *, capacity=0.03, seed=42):
    scores = np.asarray(scores, dtype=float)
    row_ids = np.asarray(row_ids)
    if not 0 < capacity <= 1:
        raise ValueError("capacity must be in (0, 1]")
    order = deterministic_ranking(scores, row_ids, seed=seed)
    budget = math.floor(len(scores) * capacity)
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:budget]] = True
    return selected


def queue_overlap(reference_scores, candidate_scores, row_ids, *, capacity=0.03, seed=42):
    ref = queue_selection(reference_scores, row_ids, capacity=capacity, seed=seed)
    cand = queue_selection(candidate_scores, row_ids, capacity=capacity, seed=seed)
    intersection = int(np.logical_and(ref, cand).sum())
    union = int(np.logical_or(ref, cand).sum())
    selected = int(ref.sum())
    return {
        "selected_cases": selected,
        "intersection": intersection,
        "jaccard": float(intersection / union) if union else 1.0,
        "reference_retained_fraction": float(intersection / selected) if selected else 1.0,
    }


def month_label_bootstrap_positions(frame: pd.DataFrame, *, seed: int):
    """Resample within each observed month/label cell, preserving each cell's row count."""
    required = {"month", "fraud_bool"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns) or len(frame) == 0:
        raise ValueError("frame must contain nonempty month and fraud_bool columns")
    if not frame["fraud_bool"].isin([0, 1]).all():
        raise ValueError("fraud_bool must be binary")
    rng = np.random.default_rng(seed)
    positions = []
    months = sorted(int(x) for x in frame["month"].unique())
    for month in months:
        for label in (0, 1):
            group = np.flatnonzero(
                (frame["month"].to_numpy() == month) & (frame["fraud_bool"].to_numpy() == label)
            )
            if len(group) == 0:
                raise ValueError(f"Month {month} lacks class {label}; stratified bootstrap undefined")
            positions.append(rng.choice(group, size=len(group), replace=True))
    sampled = np.concatenate(positions)
    # Shuffle only after each cell has been sampled; the model does not use input order.
    return sampled[rng.permutation(len(sampled))]


def summarize_stability(reference_scores, ensemble_scores, row_ids, *, capacity=0.03, seed=42):
    """Summarize training-sample perturbation sensitivity of scores and queue membership."""
    reference_scores = np.asarray(reference_scores, dtype=float)
    ensemble_scores = np.asarray(ensemble_scores, dtype=float)
    row_ids = np.asarray(row_ids)
    if reference_scores.ndim != 1 or row_ids.shape != reference_scores.shape:
        raise ValueError("reference scores and row_ids must be aligned vectors")
    if ensemble_scores.ndim != 2 or ensemble_scores.shape[1] != len(reference_scores):
        raise ValueError("ensemble_scores must have shape [repeats, cases]")
    if ensemble_scores.shape[0] < 2:
        raise ValueError("At least two perturbation refits are required")
    if not np.isfinite(ensemble_scores).all() or (ensemble_scores < 0).any() or (ensemble_scores > 1).any():
        raise ValueError("ensemble scores must be probabilities in [0, 1]")

    ref_selected = queue_selection(reference_scores, row_ids, capacity=capacity, seed=seed)
    selections = []
    jaccards = []
    retentions = []
    rank_correlations = []
    for scores in ensemble_scores:
        selected = queue_selection(scores, row_ids, capacity=capacity, seed=seed)
        selections.append(selected)
        intersection = int(np.logical_and(ref_selected, selected).sum())
        union = int(np.logical_or(ref_selected, selected).sum())
        jaccards.append(intersection / union if union else 1.0)
        retentions.append(intersection / int(ref_selected.sum()) if ref_selected.sum() else 1.0)
        rho = spearmanr(reference_scores, scores).statistic
        rank_correlations.append(float(rho))
    selections = np.asarray(selections, dtype=float)
    selection_frequency = selections.mean(axis=0)
    score_std = ensemble_scores.std(axis=0, ddof=1)
    q10 = np.quantile(ensemble_scores, 0.10, axis=0)
    q90 = np.quantile(ensemble_scores, 0.90, axis=0)
    unstable = (selection_frequency > 0.10) & (selection_frequency < 0.90)

    order = deterministic_ranking(reference_scores, row_ids, seed=seed)
    rank = np.empty(len(row_ids), dtype=np.int64)
    rank[order] = np.arange(1, len(row_ids) + 1, dtype=np.int64)
    case_table = pd.DataFrame({
        "row_id": row_ids.astype(np.int64, copy=False),
        "reference_score": reference_scores,
        "reference_rank": rank,
        "reference_selected": ref_selected,
        "ensemble_mean_score": ensemble_scores.mean(axis=0),
        "ensemble_score_std": score_std,
        "ensemble_q10": q10,
        "ensemble_q90": q90,
        "selection_frequency": selection_frequency,
        "boundary_unstable": unstable,
    }).sort_values(["reference_rank", "row_id"], kind="stable")

    def stats(values):
        values = np.asarray(values, dtype=float)
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }

    summary = {
        "repeats": int(ensemble_scores.shape[0]),
        "capacity": float(capacity),
        "reference_selected_cases": int(ref_selected.sum()),
        "queue_jaccard_vs_reference": stats(jaccards),
        "reference_queue_retention": stats(retentions),
        "spearman_score_rank_correlation": stats(rank_correlations),
        "score_std": {
            "median": float(np.median(score_std)),
            "p90": float(np.quantile(score_std, 0.90)),
            "p95": float(np.quantile(score_std, 0.95)),
            "max": float(score_std.max()),
        },
        "boundary_unstable_cases": int(unstable.sum()),
        "reference_queue_mean_selection_frequency": (
            float(selection_frequency[ref_selected].mean()) if ref_selected.any() else 1.0
        ),
        "interpretation": (
            "Empirical sensitivity to month/label-stratified training-sample bootstrap refits. "
            "These are not confidence intervals or posterior fraud probabilities."
        ),
    }
    return summary, case_table
