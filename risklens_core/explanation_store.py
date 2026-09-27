"""Immutable explanation snapshots; analyst-facing payloads have an explicit schema."""
from __future__ import annotations
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint, select
from sqlalchemy.orm import Mapped, mapped_column

from .features import FEATURES
from .explanations import METHOD, feature_hash
from .persistence import Base, QueueEntry, Case, WorkflowRun, utcnow


class FeatureContribution(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    feature: str
    value: str | float | int | None
    missing_after_cleaning: bool
    contribution: float
    pair_std_dev: float = Field(ge=0)


class ExplanationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    method: Literal["paired-permutation-v1"]
    feature_contract: Literal["baf-base-v1"]
    score_scale: Literal["uncalibrated_model_score"]
    reference_score: float = Field(ge=0, le=1)
    model_score: float = Field(ge=0, le=1)
    reconstruction_error: float = Field(ge=0, le=1e-10)
    background_size: int = Field(ge=2, le=128)
    permutation_paths: int
    model_rows_evaluated: int
    seed: int = Field(ge=0)
    features: list[FeatureContribution]
    limitations: list[str]

    @model_validator(mode="after")
    def consistent(self):
        if len(self.features) != len(FEATURES) or {f.feature for f in self.features} != set(FEATURES):
            raise ValueError("Explanation must contain each model feature exactly once")
        residual = abs(self.reference_score + sum(f.contribution for f in self.features) - self.model_score)
        if residual > 1e-10 or abs(residual - self.reconstruction_error) > 1e-10:
            raise ValueError("Explanation reconstruction failed")
        if self.permutation_paths != self.background_size * 2 or self.model_rows_evaluated != self.permutation_paths * (len(FEATURES) + 1):
            raise ValueError("Explanation sampling metadata is inconsistent")
        return self


class CaseExplanation(Base):
    __tablename__ = "case_explanations"
    __table_args__ = (UniqueConstraint("queue_entry_id", "config_sha256", name="uq_explanation_entry_config"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    queue_entry_id: Mapped[int] = mapped_column(ForeignKey("queue_entries.id", ondelete="CASCADE"), nullable=False, index=True)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    background_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def case_frame(case):
    return pd.DataFrame([case.feature_payload], columns=FEATURES)


def validate_snapshot(record, entry, case, run):
    if record.model_sha256 != run.model_sha256 or record.input_sha256 != feature_hash(case_frame(case)):
        raise ValueError("Cached explanation provenance differs from the current case/model")
    payload = ExplanationPayload.model_validate(record.payload)
    if abs(payload.model_score - entry.risk_score) > 1e-12:
        raise ValueError("Explanation score differs from the queued score")
    return payload


def find_snapshot(session, entry_id, config_sha256):
    return session.scalar(select(CaseExplanation).where(
        CaseExplanation.queue_entry_id == entry_id, CaseExplanation.config_sha256 == config_sha256))


def save_snapshot(session, entry, case, run, payload, *, config_sha256, background_sha256):
    parsed = ExplanationPayload.model_validate(payload)
    if abs(parsed.model_score - entry.risk_score) > 1e-12:
        raise ValueError("Explanation does not reproduce stored queue score")
    record = find_snapshot(session, entry.id, config_sha256)
    if record:
        validate_snapshot(record, entry, case, run)
        if record.background_sha256 != background_sha256 or record.payload != parsed.model_dump():
            raise ValueError("Existing explanation differs for the same configuration")
        return record, False
    record = CaseExplanation(queue_entry_id=entry.id, config_sha256=config_sha256,
                             model_sha256=run.model_sha256, input_sha256=feature_hash(case_frame(case)),
                             background_sha256=background_sha256, payload=parsed.model_dump(), created_at=utcnow())
    session.add(record)
    session.commit()
    return record, True
