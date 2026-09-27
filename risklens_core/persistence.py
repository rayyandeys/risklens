"""Persistent analyst-review storage for RiskLens.

The schema deliberately separates source applications from queue memberships so the same
application can appear in later scoring runs without losing historical rank/model provenance.
Ground-truth fraud labels are never persisted in this analyst-facing database.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, create_engine, select, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .baseline import sha256_file
from .features import FEATURES
from .review import REVIEW_DECISIONS


class Base(DeclarativeBase):
    pass


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment: Mapped[str] = mapped_column(String(64), nullable=False)
    protocol: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    scored_month: Mapped[int] = mapped_column(Integer, nullable=False)
    capacity: Mapped[float] = mapped_column(Float, nullable=False)
    population_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    selected_cases: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    queue_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    entries: Mapped[list["QueueEntry"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Case(Base):
    __tablename__ = "cases"

    case_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_row_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    feature_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    entries: Mapped[list["QueueEntry"]] = relationship(back_populates="case")


class QueueEntry(Base):
    __tablename__ = "queue_entries"
    __table_args__ = (
        UniqueConstraint("run_id", "case_id", name="uq_queue_run_case"),
        UniqueConstraint("run_id", "risk_rank", name="uq_queue_run_rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workflow_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.case_id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_percentile: Mapped[float] = mapped_column(Float, nullable=False)
    review_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    analyst_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    run: Mapped[WorkflowRun] = relationship(back_populates="entries")
    case: Mapped[Case] = relationship(back_populates="entries")
    events: Mapped[list["ReviewEvent"]] = relationship(back_populates="entry", cascade="all, delete-orphan")


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    queue_entry_id: Mapped[int] = mapped_column(ForeignKey("queue_entries.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str] = mapped_column(String(32), nullable=False)
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    analyst_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    analyst_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    case_version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    entry: Mapped[QueueEntry] = relationship(back_populates="events")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_database_engine(database_url: str = "sqlite:///./risklens.db"):
    from .database_config import database_url as resolve_url
    database_url = resolve_url(database_url)
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    engine = create_engine(database_url, **kwargs)
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
    return engine


def create_session_factory(engine):
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def initialize_database(engine) -> None:
    from .migrations import upgrade_database
    upgrade_database(engine)


def _validate_review_reference(summary: dict[str, Any]) -> None:
    if summary.get("status") != "complete":
        raise ValueError("Expected a completed review workflow summary")
    experiment = summary.get("experiment")
    if experiment not in {"review-workflow-v1", "review-workflow-v2"}:
        raise ValueError("Expected a supported review-workflow-v1/v2 summary")
    if summary.get("split", {}).get("test_reserved") != [6, 7] or summary.get("scored_month") != 5:
        raise ValueError("Unexpected development/test split in review workflow summary")
    if experiment == "review-workflow-v1":
        if summary.get("test_evaluated") is not False:
            raise ValueError("Legacy review workflow must precede final-test opening")
    else:
        if summary.get("test_evaluated") is not True:
            raise ValueError("Frozen product workflow must record the completed final-test state")
        scope = summary.get("queue_data_scope", {})
        if scope.get("development_month_only") is not True or scope.get("reserved_test_rows_used") is not False or scope.get("reserved_test_labels_used") is not False:
            raise ValueError("Frozen product queue must be built from development month 5 only")
        if scope.get("post_test_model_selection") is not False:
            raise ValueError("Frozen product workflow cannot perform post-test model selection")
        if summary.get("reference_model") != "histgb_15_leaves/no_customer_age" or summary.get("removed_features") != ["customer_age"]:
            raise ValueError("Frozen product workflow model provenance is unexpected")
        if not summary.get("freeze_id") or not summary.get("final_test_opening_id"):
            raise ValueError("Frozen product workflow must record freeze/final-test lineage")
    separation = summary.get("label_separation", {})
    if separation.get("analyst_queue_contains_target") is not False:
        raise ValueError("Analyst queue must explicitly exclude the fraud target")
    policy = summary.get("review_policy", {})
    if not 0 < float(policy.get("capacity", 0)) <= 1:
        raise ValueError("Review workflow has an invalid capacity")
    if int(policy.get("selected_cases", -1)) < 1:
        raise ValueError("Review workflow selected no analyst cases")


def _validate_queue(queue: pd.DataFrame, summary: dict[str, Any]) -> None:
    required = {
        "case_id", "source_row_id", "model_name", "risk_score", "risk_rank", "risk_percentile",
        "review_selected", "review_status", "decision", "analyst_note", *FEATURES,
    }
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"Analyst queue is missing columns: {sorted(missing)}")
    if "fraud_bool" in queue.columns:
        raise ValueError("Analyst queue must never contain fraud_bool")
    if len(queue) != int(summary["review_policy"]["selected_cases"]):
        raise ValueError("Analyst queue row count differs from the workflow summary")
    if queue["case_id"].duplicated().any() or queue["source_row_id"].duplicated().any():
        raise ValueError("Analyst queue contains duplicate cases")
    if not queue["review_selected"].astype(bool).all():
        raise ValueError("Analyst queue contains non-selected rows")
    if set(queue["review_status"].astype(str)) != {"pending"}:
        raise ValueError("Fresh analyst queue must contain only pending cases")
    if queue["risk_rank"].astype(int).tolist() != list(range(1, len(queue) + 1)):
        raise ValueError("Analyst queue ranks must be contiguous from 1 through the review budget")
    scores = queue["risk_score"].astype(float)
    if ((scores < 0) | (scores > 1)).any() or not scores.is_monotonic_decreasing:
        raise ValueError("Analyst queue risk scores must be descending probabilities")
    if queue["risk_percentile"].astype(float).lt(0).any() or queue["risk_percentile"].astype(float).gt(1).any():
        raise ValueError("Risk percentile must be within [0, 1]")
    if queue["model_name"].nunique() != 1 or queue["model_name"].iloc[0] != summary["reference_model"]:
        raise ValueError("Analyst queue model provenance differs from the workflow summary")


def import_review_queue(session: Session, summary_path: str | Path) -> dict[str, Any]:
    """Validate and idempotently import an analyst queue into persistent storage."""
    import json

    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_review_reference(summary)
    queue_path = Path(summary["outputs"]["analyst_queue"])
    if not queue_path.exists():
        raise ValueError(f"Missing analyst queue: {queue_path}")
    queue = pd.read_csv(queue_path)
    _validate_queue(queue, summary)

    summary_hash = sha256_file(summary_path)
    queue_hash = sha256_file(queue_path)
    run_id = str(summary["run_id"])
    existing = session.get(WorkflowRun, run_id)
    if existing is not None:
        if existing.summary_sha256 != summary_hash or existing.queue_sha256 != queue_hash:
            raise ValueError("Run ID already exists with different source content")
        return {"run_id": run_id, "inserted_cases": 0, "selected_cases": existing.selected_cases, "idempotent": True}

    now = utcnow()
    run = WorkflowRun(
        run_id=run_id,
        experiment=str(summary["experiment"]),
        protocol=str(summary["protocol"]),
        dataset_sha256=str(summary["dataset_sha256"]),
        model_name=str(summary["reference_model"]),
        model_sha256=str(summary["reference_model_sha256"]),
        scored_month=int(summary["scored_month"]),
        capacity=float(summary["review_policy"]["capacity"]),
        population_cases=int(summary["review_policy"]["population_cases"]),
        selected_cases=int(summary["review_policy"]["selected_cases"]),
        summary_sha256=summary_hash,
        queue_sha256=queue_hash,
        imported_at=now,
    )
    session.add(run)
    session.flush()

    inserted_cases = 0
    for record in queue.to_dict(orient="records"):
        case_id = str(record["case_id"])
        source_row_id = int(record["source_row_id"])
        feature_payload = {name: (None if pd.isna(record[name]) else record[name]) for name in FEATURES}
        case = session.get(Case, case_id)
        if case is None:
            case = Case(case_id=case_id, source_row_id=source_row_id, feature_payload=feature_payload, created_at=now)
            session.add(case)
            inserted_cases += 1
        else:
            if case.source_row_id != source_row_id or case.feature_payload != feature_payload:
                raise ValueError(f"Case {case_id} conflicts with an existing persisted application")

        entry = QueueEntry(
            run_id=run_id,
            case_id=case_id,
            model_name=str(record["model_name"]),
            risk_score=float(record["risk_score"]),
            risk_rank=int(record["risk_rank"]),
            risk_percentile=float(record["risk_percentile"]),
            review_status="pending",
            decision=None,
            analyst_note="",
            version=1,
            created_at=now,
            updated_at=now,
        )
        session.add(entry)
        session.flush()
        session.add(ReviewEvent(
            queue_entry_id=entry.id,
            event_type="imported",
            from_status=None,
            to_status="pending",
            decision=None,
            analyst_note="",
            analyst_id=None,
            case_version=1,
            created_at=now,
        ))

    session.commit()
    return {"run_id": run_id, "inserted_cases": inserted_cases, "selected_cases": len(queue), "idempotent": False}
