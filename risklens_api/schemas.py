from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

Decision = Literal["confirmed_fraud", "legitimate", "escalate"]


class RunSummary(BaseModel):
    run_id: str
    experiment: str
    model_name: str
    model_sha256: str
    scored_month: int
    capacity: float
    population_cases: int
    selected_cases: int
    imported_at: datetime
    status_counts: dict[str, int]
    decision_counts: dict[str, int]


class CaseListItem(BaseModel):
    case_id: str
    source_row_id: int
    risk_score: float
    risk_rank: int
    risk_percentile: float
    review_status: str
    decision: str | None
    version: int


class CaseDetail(CaseListItem):
    run_id: str
    model_name: str
    analyst_note: str
    features: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Decision
    analyst_note: str = Field(default="", max_length=2000)
    expected_version: int = Field(ge=1)


class ReviewEventResponse(BaseModel):
    id: int
    event_type: str
    from_status: str | None
    to_status: str
    decision: str | None
    analyst_note: str
    analyst_id: str | None
    case_version: int
    created_at: datetime


from risklens_core.explanation_store import ExplanationPayload


class ExplanationResponse(BaseModel):
    explanation_id: int
    run_id: str
    case_id: str
    model_sha256: str
    input_sha256: str
    config_sha256: str
    background_sha256: str
    created_at: datetime
    explanation: ExplanationPayload
