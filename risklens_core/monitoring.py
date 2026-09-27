"""Temporal drift and monitoring utilities for RiskLens development monitoring.

This module is deliberately label-free for drift alerting. Labels are used only by the
separate prospective backtest in scripts/run_temporal_monitoring.py. The sealed test months
6-7 are never loaded by these helpers.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .features import CATEGORICAL, FEATURES, NUMERIC, INDICATORS, FeatureCleaner

PSI_WATCH = 0.10
PSI_ALERT = 0.25
MISSING_WATCH = 0.02
MISSING_ALERT = 0.05
EPS = 1e-6


SEVERITY_ORDER = {"normal": 0, "watch": 1, "alert": 2}


def _worst(*levels: str) -> str:
    for level in levels:
        if level not in SEVERITY_ORDER:
            raise ValueError(f"Unknown severity: {level}")
    return max(levels, key=SEVERITY_ORDER.get) if levels else "normal"


def _severity_from_psi(value: float) -> str:
    if not np.isfinite(value) or value < 0:
        raise ValueError("PSI must be finite and nonnegative")
    if value >= PSI_ALERT:
        return "alert"
    if value >= PSI_WATCH:
        return "watch"
    return "normal"


def _severity_from_missing_delta(delta: float) -> str:
    value = abs(float(delta))
    if value >= MISSING_ALERT:
        return "alert"
    if value >= MISSING_WATCH:
        return "watch"
    return "normal"


def _psi(reference_counts: np.ndarray, target_counts: np.ndarray) -> float:
    reference_counts = np.asarray(reference_counts, dtype=float)
    target_counts = np.asarray(target_counts, dtype=float)
    if reference_counts.ndim != 1 or target_counts.shape != reference_counts.shape:
        raise ValueError("PSI count vectors must be aligned one-dimensional arrays")
    if (reference_counts < 0).any() or (target_counts < 0).any():
        raise ValueError("PSI counts cannot be negative")
    if reference_counts.sum() <= 0 or target_counts.sum() <= 0:
        raise ValueError("PSI requires nonempty reference and target populations")
    ref = reference_counts / reference_counts.sum()
    tar = target_counts / target_counts.sum()
    ref = np.clip(ref, EPS, None)
    tar = np.clip(tar, EPS, None)
    ref = ref / ref.sum()
    tar = tar / tar.sum()
    return float(np.sum((tar - ref) * np.log(tar / ref)))


def _numeric_edges(reference: pd.Series, bins: int) -> np.ndarray:
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    finite = pd.to_numeric(reference, errors="raise").dropna().to_numpy(dtype=float)
    if len(finite) == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    if not np.isfinite(finite).all():
        raise ValueError("Reference numeric feature contains infinite values")
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))
    inner = np.unique(quantiles[1:-1])
    if len(inner) == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    return np.concatenate(([-np.inf], inner, [np.inf])).astype(float)


def numeric_drift(reference: pd.Series, target: pd.Series, *, bins: int = 10) -> dict:
    ref = pd.to_numeric(reference, errors="raise")
    tar = pd.to_numeric(target, errors="raise")
    if np.isinf(ref.dropna()).any() or np.isinf(tar.dropna()).any():
        raise ValueError("Numeric drift does not accept infinite values")
    edges = _numeric_edges(ref, bins)
    ref_nonmissing = ref.dropna().to_numpy(dtype=float)
    tar_nonmissing = tar.dropna().to_numpy(dtype=float)
    ref_hist = np.histogram(ref_nonmissing, bins=edges)[0]
    tar_hist = np.histogram(tar_nonmissing, bins=edges)[0]
    # Missingness is a dedicated PSI bucket so a feature can drift even if its observed values do not.
    ref_counts = np.concatenate((ref_hist, [int(ref.isna().sum())]))
    tar_counts = np.concatenate((tar_hist, [int(tar.isna().sum())]))
    psi = _psi(ref_counts, tar_counts)
    ref_missing = float(ref.isna().mean())
    tar_missing = float(tar.isna().mean())
    delta = tar_missing - ref_missing
    if len(ref_nonmissing) and len(tar_nonmissing):
        ks = float(ks_2samp(ref_nonmissing, tar_nonmissing, method="asymp").statistic)
    else:
        ks = None
    severity = _worst(_severity_from_psi(psi), _severity_from_missing_delta(delta))
    return {
        "kind": "numeric",
        "psi": psi,
        "ks_statistic": ks,
        "reference_missing_rate": ref_missing,
        "target_missing_rate": tar_missing,
        "missing_rate_delta": float(delta),
        "bin_count_including_missing": int(len(ref_counts)),
        "severity": severity,
    }


def categorical_drift(reference: pd.Series, target: pd.Series) -> dict:
    missing = "__RISKLENS_MISSING__"
    other = "__RISKLENS_OTHER__"
    ref = reference.astype(object).where(reference.notna(), missing).astype(str)
    tar = target.astype(object).where(target.notna(), missing).astype(str)
    reference_categories = sorted(set(ref.tolist()))
    allowed = set(reference_categories)
    tar_bucketed = tar.where(tar.isin(allowed), other)
    categories = reference_categories + ([] if other in allowed else [other])
    ref_counts = np.array([(ref == category).sum() for category in categories], dtype=float)
    tar_counts = np.array([(tar_bucketed == category).sum() for category in categories], dtype=float)
    psi = _psi(ref_counts, tar_counts)
    ref_missing = float((ref == missing).mean())
    tar_missing = float((tar == missing).mean())
    delta = tar_missing - ref_missing
    unseen_rate = float((tar_bucketed == other).mean()) if other in categories else 0.0
    severity = _worst(_severity_from_psi(psi), _severity_from_missing_delta(delta))
    return {
        "kind": "categorical",
        "psi": psi,
        "reference_missing_rate": ref_missing,
        "target_missing_rate": tar_missing,
        "missing_rate_delta": float(delta),
        "unseen_category_rate": unseen_rate,
        "reference_category_count": int(len(reference_categories)),
        "severity": severity,
    }


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
        raise ValueError("Monitoring requires a nonempty DataFrame")
    return FeatureCleaner().transform(frame)


def feature_drift(reference: pd.DataFrame, target: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    """Compare cleaned feature distributions with fixed reference-defined bins/categories."""
    ref = clean_features(reference)
    tar = clean_features(target)
    rows = []
    for feature in NUMERIC + INDICATORS:
        result = numeric_drift(ref[feature], tar[feature], bins=bins)
        rows.append({"feature": feature, **result})
    for feature in CATEGORICAL:
        result = categorical_drift(ref[feature], tar[feature])
        rows.append({"feature": feature, **result})
    table = pd.DataFrame(rows)
    if set(table["feature"]) != set(NUMERIC + INDICATORS + CATEGORICAL):
        raise RuntimeError("Feature drift report did not cover the full transformed contract")
    severity_rank = table["severity"].map(SEVERITY_ORDER)
    table = table.assign(_severity_rank=severity_rank).sort_values(
        ["_severity_rank", "psi", "feature"], ascending=[False, False, True], kind="stable"
    ).drop(columns=["_severity_rank"]).reset_index(drop=True)
    return table


def score_drift(reference_scores, target_scores, *, bins: int = 10) -> dict:
    ref = pd.Series(np.asarray(reference_scores, dtype=float))
    tar = pd.Series(np.asarray(target_scores, dtype=float))
    if len(ref) == 0 or len(tar) == 0:
        raise ValueError("Score drift requires nonempty populations")
    if not np.isfinite(ref).all() or not np.isfinite(tar).all():
        raise ValueError("Scores must be finite")
    if (ref < 0).any() or (ref > 1).any() or (tar < 0).any() or (tar > 1).any():
        raise ValueError("Scores must be within [0, 1]")
    base = numeric_drift(ref, tar, bins=bins)
    return {
        "psi": base["psi"],
        "ks_statistic": base["ks_statistic"],
        "reference_mean": float(ref.mean()),
        "target_mean": float(tar.mean()),
        "reference_p50": float(ref.quantile(0.50)),
        "target_p50": float(tar.quantile(0.50)),
        "reference_p95": float(ref.quantile(0.95)),
        "target_p95": float(tar.quantile(0.95)),
        "severity": _severity_from_psi(base["psi"]),
    }


def summarize_feature_drift(table: pd.DataFrame, *, top_n: int = 10) -> dict:
    required = {"feature", "psi", "severity"}
    if not isinstance(table, pd.DataFrame) or not required.issubset(table.columns) or len(table) == 0:
        raise ValueError("feature drift table lacks required columns")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    counts = {level: int((table["severity"] == level).sum()) for level in SEVERITY_ORDER}
    status = _worst(*[level for level, count in counts.items() if count] or ["normal"])
    top = table.sort_values(["psi", "feature"], ascending=[False, True], kind="stable").head(top_n)
    columns = [
        "feature", "kind", "psi", "ks_statistic", "reference_missing_rate",
        "target_missing_rate", "missing_rate_delta", "unseen_category_rate", "severity",
    ]
    records = []
    for row in top.reindex(columns=columns).to_dict(orient="records"):
        records.append({k: (None if pd.isna(v) else v) for k, v in row.items()})
    return {
        "status": status,
        "counts_by_severity": counts,
        "max_psi": float(table["psi"].max()),
        "features_at_watch_or_alert": int(table["severity"].isin(["watch", "alert"]).sum()),
        "top_features": records,
    }


def build_alerts(feature_table: pd.DataFrame, score_report: dict, *, max_feature_alerts: int = 12) -> list[dict]:
    alerts: list[dict] = []
    candidates = feature_table.loc[feature_table["severity"].isin(["watch", "alert"])].copy()
    candidates["_rank"] = candidates["severity"].map(SEVERITY_ORDER)
    candidates = candidates.sort_values(["_rank", "psi", "feature"], ascending=[False, False, True])
    for _, row in candidates.head(max_feature_alerts).iterrows():
        alerts.append({
            "type": "feature_drift",
            "severity": row["severity"],
            "feature": row["feature"],
            "psi": float(row["psi"]),
            "missing_rate_delta": None if pd.isna(row.get("missing_rate_delta")) else float(row["missing_rate_delta"]),
        })
    if score_report["severity"] in {"watch", "alert"}:
        alerts.append({
            "type": "score_drift",
            "severity": score_report["severity"],
            "psi": float(score_report["psi"]),
            "ks_statistic": float(score_report["ks_statistic"]),
        })
    return alerts


def overall_status(feature_summary: dict, score_report: dict) -> str:
    return _worst(feature_summary["status"], score_report["severity"])


def validate_calibration_reference(summary: dict, *, dataset_sha256: str, model_sha256: str) -> None:
    """Require the immediately previous development experiment before monitoring is generated."""
    if summary.get("status") != "complete" or summary.get("experiment") != "calibration-uncertainty-v1":
        raise ValueError("Monitoring requires a completed calibration-uncertainty-v1 report")
    if summary.get("dataset_sha256") != dataset_sha256:
        raise ValueError("Calibration/uncertainty report uses a different dataset")
    if summary.get("reference_model_sha256") != model_sha256:
        raise ValueError("Calibration/uncertainty report uses a different reference model")
    if summary.get("test_evaluated") is not False or summary.get("scored_month") != 5:
        raise ValueError("Calibration/uncertainty report must remain development-only on month 5")
    if summary.get("calibration", {}).get("selected_method") != "raw":
        raise ValueError("Current monitoring package expects the observed development decision to keep raw scores")


def thresholds() -> dict:
    return {
        "psi": {
            "normal_below": PSI_WATCH,
            "watch_from": PSI_WATCH,
            "alert_from": PSI_ALERT,
        },
        "absolute_missing_rate_delta": {
            "normal_below": MISSING_WATCH,
            "watch_from": MISSING_WATCH,
            "alert_from": MISSING_ALERT,
        },
        "note": (
            "These are predeclared engineering heuristics for dashboard triage, not statistical tests or "
            "regulatory thresholds. They must be revisited with production history."
        ),
    }
