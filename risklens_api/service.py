from __future__ import annotations

from collections import Counter

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from risklens_core.persistence import Case, QueueEntry, ReviewEvent, WorkflowRun, utcnow
from risklens_core.review import REVIEW_DECISIONS


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    pass


def get_run(session: Session, run_id: str) -> WorkflowRun:
    run = session.get(WorkflowRun, run_id)
    if run is None:
        raise NotFoundError("Workflow run not found")
    return run


def list_runs(session: Session) -> list[WorkflowRun]:
    return list(session.scalars(select(WorkflowRun).order_by(WorkflowRun.imported_at.desc())))


def run_summary(session: Session, run_id: str) -> dict:
    run = get_run(session, run_id)
    statuses = Counter(dict(session.execute(
        select(QueueEntry.review_status, func.count()).where(QueueEntry.run_id == run_id).group_by(QueueEntry.review_status)
    ).all()))
    decisions = Counter(dict(session.execute(
        select(QueueEntry.decision, func.count()).where(
            QueueEntry.run_id == run_id, QueueEntry.decision.is_not(None)
        ).group_by(QueueEntry.decision)
    ).all()))
    return {
        "run_id": run.run_id,
        "experiment": run.experiment,
        "model_name": run.model_name,
        "model_sha256": run.model_sha256,
        "scored_month": run.scored_month,
        "capacity": run.capacity,
        "population_cases": run.population_cases,
        "selected_cases": run.selected_cases,
        "imported_at": run.imported_at,
        "status_counts": dict(statuses),
        "decision_counts": {str(k): v for k, v in decisions.items()},
    }


def list_cases(session: Session, run_id: str, *, status: str | None = None,
               decision: str | None = None, min_score: float | None = None,
               limit: int = 50, offset: int = 0) -> list[tuple[QueueEntry, Case]]:
    get_run(session, run_id)
    statement = select(QueueEntry, Case).join(Case, Case.case_id == QueueEntry.case_id).where(QueueEntry.run_id == run_id)
    if status is not None:
        statement = statement.where(QueueEntry.review_status == status)
    if decision is not None:
        statement = statement.where(QueueEntry.decision == decision)
    if min_score is not None:
        statement = statement.where(QueueEntry.risk_score >= min_score)
    statement = statement.order_by(QueueEntry.risk_rank.asc()).offset(offset).limit(limit)
    return list(session.execute(statement).all())


def get_case(session: Session, run_id: str, case_id: str) -> tuple[QueueEntry, Case]:
    result = session.execute(
        select(QueueEntry, Case).join(Case, Case.case_id == QueueEntry.case_id).where(
            QueueEntry.run_id == run_id, QueueEntry.case_id == case_id
        )
    ).first()
    if result is None:
        raise NotFoundError("Case not found in workflow run")
    return result


def list_events(session: Session, run_id: str, case_id: str) -> list[ReviewEvent]:
    entry, _ = get_case(session, run_id, case_id)
    return list(session.scalars(
        select(ReviewEvent).where(ReviewEvent.queue_entry_id == entry.id).order_by(ReviewEvent.id.asc())
    ))


def apply_decision(session: Session, run_id: str, case_id: str, *, decision: str,
                   analyst_note: str, analyst_id: str, expected_version: int) -> tuple[QueueEntry, Case]:
    if decision not in REVIEW_DECISIONS:
        raise ValueError("Unsupported analyst decision")
    entry, case = get_case(session, run_id, case_id)
    if entry.version != expected_version:
        raise ConflictError(f"Stale case version: expected {expected_version}, current {entry.version}")

    from_status = entry.review_status
    to_status = "escalated" if decision == "escalate" else "resolved"
    now = utcnow()
    result = session.execute(
        update(QueueEntry)
        .where(QueueEntry.id == entry.id, QueueEntry.version == expected_version)
        .values(
            review_status=to_status,
            decision=decision,
            analyst_note=analyst_note,
            version=expected_version + 1,
            updated_at=now,
        )
    )
    if result.rowcount != 1:
        session.rollback()
        raise ConflictError("Case changed while the decision was being written")

    session.add(ReviewEvent(
        queue_entry_id=entry.id,
        event_type="decision",
        from_status=from_status,
        to_status=to_status,
        decision=decision,
        analyst_note=analyst_note,
        analyst_id=analyst_id,
        case_version=expected_version + 1,
        created_at=now,
    ))
    session.commit()
    return get_case(session, run_id, case_id)
