"""Deterministic risk ranking and analyst-review queue primitives for review-workflow-v1."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .features import FEATURES

REVIEW_DECISIONS = ("confirmed_fraud", "legitimate", "escalate")


def _validated_vectors(scores, row_ids):
    scores = np.asarray(scores, dtype=float)
    row_ids = np.asarray(row_ids)
    if scores.ndim != 1 or row_ids.shape != scores.shape or len(scores) == 0:
        raise ValueError("Scores and row IDs must be aligned nonempty vectors")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Risk scores must be finite probabilities within [0, 1]")
    if not np.issubdtype(row_ids.dtype, np.integer) or len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Row IDs must be unique integers")
    return scores, row_ids.astype(np.int64, copy=False)


def deterministic_ranking(scores, row_ids, *, seed=42):
    """Return zero-based row positions ordered high-risk first with label-free seeded tie breaks.

    This intentionally mirrors the tie policy used by risklens_core.metrics.evaluate().
    """
    scores, row_ids = _validated_vectors(scores, row_ids)
    by_id = np.argsort(row_ids, kind="stable")
    tie_key = np.empty(len(row_ids), dtype=np.int64)
    tie_key[by_id] = np.random.default_rng(seed).permutation(len(row_ids))
    return np.lexsort((tie_key, -scores))


def build_review_population(frame: pd.DataFrame, scores, row_ids, *, capacity=0.03, seed=42,
                            model_name="histgb_15_leaves"):
    """Build the scored population while withholding the target from analyst-facing output."""
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("Expected a DataFrame with unique columns")
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    if not 0 < capacity <= 1:
        raise ValueError("Review capacity must be in (0, 1]")
    scores, row_ids = _validated_vectors(scores, row_ids)
    if len(frame) != len(scores):
        raise ValueError("Frame, scores and row IDs must have the same number of rows")

    order = deterministic_ranking(scores, row_ids, seed=seed)
    rank = np.empty(len(frame), dtype=np.int64)
    rank[order] = np.arange(1, len(frame) + 1, dtype=np.int64)
    budget = math.floor(len(frame) * capacity)

    output = frame.loc[:, FEATURES].reset_index(drop=True).copy()
    output.insert(0, "case_id", [f"RL-{int(row_id):09d}" for row_id in row_ids])
    output.insert(1, "source_row_id", row_ids)
    output.insert(2, "model_name", model_name)
    output.insert(3, "risk_score", scores)
    output.insert(4, "risk_rank", rank)
    # Highest-risk row has percentile 1.0; lowest-risk row approaches 0.
    output.insert(5, "risk_percentile", (len(frame) - rank + 1) / len(frame))
    output.insert(6, "review_selected", rank <= budget)
    output.insert(7, "review_status", np.where(rank <= budget, "pending", "not_selected"))
    output.insert(8, "decision", "")
    output.insert(9, "analyst_note", "")
    return output.sort_values(["risk_rank", "source_row_id"], kind="stable").reset_index(drop=True)


def selected_queue(scored_population: pd.DataFrame):
    required = {"case_id", "source_row_id", "risk_score", "risk_rank", "review_selected", "review_status"}
    if not isinstance(scored_population, pd.DataFrame) or not required.issubset(scored_population.columns):
        raise ValueError("Scored population does not match the review workflow schema")
    result = scored_population.loc[scored_population["review_selected"]].copy()
    return result.sort_values(["risk_rank", "source_row_id"], kind="stable").reset_index(drop=True)


def apply_review_decisions(queue: pd.DataFrame, decisions: pd.DataFrame):
    """Apply validated analyst decisions to selected cases without using ground-truth labels."""
    required_queue = {"case_id", "review_selected", "review_status", "decision", "analyst_note"}
    required_decisions = {"case_id", "decision"}
    if not required_queue.issubset(queue.columns):
        raise ValueError("Queue does not match the review workflow schema")
    if not isinstance(decisions, pd.DataFrame) or not required_decisions.issubset(decisions.columns):
        raise ValueError("Decisions require case_id and decision columns")
    if decisions["case_id"].duplicated().any():
        raise ValueError("Each case may appear at most once in a decision batch")
    invalid = sorted(set(decisions["decision"]) - set(REVIEW_DECISIONS))
    if invalid:
        raise ValueError(f"Unsupported review decisions: {invalid}")

    result = queue.copy()
    selected_ids = set(result.loc[result["review_selected"], "case_id"])
    unknown = sorted(set(decisions["case_id"]) - selected_ids)
    if unknown:
        raise ValueError("Decisions may only target cases selected for analyst review")

    note_lookup = decisions.set_index("case_id").get("analyst_note")
    for row in decisions.itertuples(index=False):
        mask = result["case_id"].eq(row.case_id)
        result.loc[mask, "decision"] = row.decision
        result.loc[mask, "review_status"] = "escalated" if row.decision == "escalate" else "resolved"
        if note_lookup is not None:
            note = note_lookup.get(row.case_id)
            result.loc[mask, "analyst_note"] = "" if pd.isna(note) else str(note)
    return result
