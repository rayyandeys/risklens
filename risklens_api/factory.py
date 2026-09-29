from __future__ import annotations

from pathlib import Path
import secrets

from fastapi import Depends, FastAPI, HTTPException, Query, APIRouter, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .auth import authenticate
from sqlalchemy.orm import Session
from sqlalchemy import select, func, text
from sqlalchemy.engine import make_url
from risklens_core.explanation_store import CaseExplanation, validate_snapshot
from risklens_core.persistence import QueueEntry
from risklens_core.product_monitoring import MonitoringDataError, load_monitoring_overview
from .service import get_run
from .schemas import ExplanationResponse

from .database import build_database
from .schemas import CaseDetail, CaseListItem, DecisionRequest, ReviewEventResponse, RunSummary
from .service import ConflictError, NotFoundError, apply_decision, get_case, list_cases, list_events, list_runs, run_summary
from .settings import RuntimeSettings


def _case_list_item(entry, case):
    return CaseListItem(
        case_id=case.case_id,
        source_row_id=case.source_row_id,
        risk_score=entry.risk_score,
        risk_rank=entry.risk_rank,
        risk_percentile=entry.risk_percentile,
        review_status=entry.review_status,
        decision=entry.decision,
        version=entry.version,
    )


def _case_detail(entry, case):
    return CaseDetail(
        **_case_list_item(entry, case).model_dump(),
        run_id=entry.run_id,
        model_name=entry.model_name,
        analyst_note=entry.analyst_note,
        features=case.feature_payload,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


def create_app(
    database_url: str | None = None,
    reports_dir: str | Path | None = None,
    runtime_settings: RuntimeSettings | None = None,
) -> FastAPI:
    settings = runtime_settings or RuntimeSettings.from_env()
    from risklens_core.database_config import database_url as resolve_database_url
    resolved_database_url = resolve_database_url(database_url)
    driver = make_url(resolved_database_url).drivername
    if settings.require_postgres and driver != "postgresql+psycopg":
        raise RuntimeError("Production RiskLens requires PostgreSQL; SQLite is development-only")

    engine, SessionLocal = build_database(resolved_database_url)
    app = FastAPI(
        title="RiskLens Review API",
        version="0.3.0",
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.state.engine = engine
    app.state.SessionLocal = SessionLocal
    app.state.runtime_settings = settings
    app.state.reports_dir = Path(reports_dir) if reports_dir is not None else Path(__file__).resolve().parents[1] / "reports"

    @app.middleware("http")
    async def production_guardrails(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if request.method in {"POST", "PUT", "PATCH"} and content_length:
            try:
                too_large = int(content_length) > settings.max_request_body_bytes
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
            if too_large:
                return JSONResponse(status_code=413, content={"detail": "Request body too large"})

        response = await call_next(request)
        response.headers["X-Request-ID"] = secrets.token_hex(16)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; "
            "img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        if settings.hsts_enabled:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    def get_session():
        with SessionLocal() as session:
            yield session

    bearer = HTTPBearer(auto_error=False)

    def current_identity(credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                         session: Session = Depends(get_session)):
        identity = authenticate(session, credentials.credentials) if credentials else None
        if identity is None:
            raise HTTPException(401, "Missing, invalid, expired or revoked credential",
                                headers={"WWW-Authenticate": "Bearer"})
        return identity

    def reviewer(identity=Depends(current_identity)):
        if identity.role not in ("analyst", "admin"):
            raise HTTPException(403, "Analyst role required")
        return identity

    router = APIRouter(dependencies=[Depends(current_identity)])

    @router.get("/api/v1/auth/me")
    def me(identity=Depends(current_identity)):
        return {"analyst_id": identity.analyst_id, "role": identity.role,
                "expires_at": identity.expires_at}

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "risklens-review-api"}

    @app.get("/ready")
    def ready(session: Session = Depends(get_session)):
        try:
            session.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Database is not ready") from exc
        return {"status": "ready", "database": "ok", "environment": settings.environment}

    @router.get("/api/v1/runs", response_model=list[RunSummary])
    def runs(session: Session = Depends(get_session)):
        return [run_summary(session, run.run_id) for run in list_runs(session)]

    @router.get("/api/v1/runs/{run_id}/summary", response_model=RunSummary)
    def summary(run_id: str, session: Session = Depends(get_session)):
        try:
            return run_summary(session, run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/v1/runs/{run_id}/cases", response_model=list[CaseListItem])
    def cases(
        run_id: str,
        status: str | None = None,
        decision: str | None = None,
        min_score: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        session: Session = Depends(get_session),
    ):
        try:
            return [_case_list_item(entry, case) for entry, case in list_cases(
                session, run_id, status=status, decision=decision, min_score=min_score,
                limit=limit, offset=offset,
            )]
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/v1/runs/{run_id}/cases/{case_id}", response_model=CaseDetail)
    def case_detail(run_id: str, case_id: str, session: Session = Depends(get_session)):
        try:
            return _case_detail(*get_case(session, run_id, case_id))
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/api/v1/runs/{run_id}/cases/{case_id}/events", response_model=list[ReviewEventResponse])
    def events(run_id: str, case_id: str, session: Session = Depends(get_session)):
        try:
            return [ReviewEventResponse(
                id=e.id, event_type=e.event_type, from_status=e.from_status, to_status=e.to_status,
                decision=e.decision, analyst_note=e.analyst_note, analyst_id=e.analyst_id,
                case_version=e.case_version, created_at=e.created_at,
            ) for e in list_events(session, run_id, case_id)]
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post("/api/v1/runs/{run_id}/cases/{case_id}/decision", response_model=CaseDetail)
    def decision(run_id: str, case_id: str, request: DecisionRequest,
                 identity=Depends(reviewer), session: Session = Depends(get_session)):
        try:
            return _case_detail(*apply_decision(
                session, run_id, case_id, decision=request.decision, analyst_note=request.analyst_note,
                analyst_id=identity.analyst_id, expected_version=request.expected_version,
            ))
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/v1/runs/{run_id}/explanations/summary")
    def explanation_summary(run_id: str, session: Session = Depends(get_session)):
        try:
            run = get_run(session, run_id)
        except NotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        covered = session.scalar(select(func.count(func.distinct(CaseExplanation.queue_entry_id)))
                                 .join(QueueEntry, QueueEntry.id == CaseExplanation.queue_entry_id)
                                 .where(QueueEntry.run_id == run_id))
        return {"run_id": run_id, "queue_cases": run.selected_cases, "explained_cases": covered,
                "remaining_cases": run.selected_cases - covered}

    @router.get("/api/v1/monitoring/overview")
    def monitoring_overview():
        try:
            return load_monitoring_overview(app.state.reports_dir)
        except MonitoringDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.get("/api/v1/runs/{run_id}/cases/{case_id}/explanation", response_model=ExplanationResponse)
    def case_explanation(run_id: str, case_id: str, session: Session = Depends(get_session)):
        try:
            entry, case = get_case(session, run_id, case_id)
            run = get_run(session, run_id)
        except NotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        record = session.scalar(select(CaseExplanation).where(CaseExplanation.queue_entry_id == entry.id)
                                .order_by(CaseExplanation.id.desc()).limit(1))
        if record is None:
            raise HTTPException(404, "Explanation not generated for this case; run the explanation builder")
        try:
            payload = validate_snapshot(record, entry, case, run)
        except ValueError as exc:
            raise HTTPException(409, "Explanation provenance or reconstruction check failed") from exc
        return ExplanationResponse(explanation_id=record.id, run_id=run_id, case_id=case_id,
                                   model_sha256=record.model_sha256, input_sha256=record.input_sha256,
                                   config_sha256=record.config_sha256, background_sha256=record.background_sha256,
                                   created_at=record.created_at, explanation=payload)

    app.include_router(router)

    # Separate anonymous GET surface: never grants an identity to /api/v1.
    # Only the frozen synthetic BAF month-5 workflow is eligible for publication.
    @app.get("/api/demo/config")
    def demo_config():
        return {"enabled": settings.public_demo_enabled}

    def require_demo():
        if not settings.public_demo_enabled:
            raise HTTPException(404, "Public demo is disabled")

    def public_run(run_id: str, session: Session = Depends(get_session)):
        try:
            run = get_run(session, run_id)
        except NotFoundError as exc:
            raise HTTPException(404, "Demo run not found") from exc
        if not demo_eligible(run):
            raise HTTPException(404, "Demo run not found")
        return run

    def demo_eligible(run):
        return (
            run.dataset_sha256 == "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"
            and run.model_sha256 == "65e9caa079ba46af72955a024b7ae0fd7d207e80c10eba686954ff149eb5ee12"
            and run.model_name == "histgb_15_leaves/no_customer_age"
            and run.scored_month == 5 and run.capacity == 0.03
        )

    demo = APIRouter(prefix="/api/demo", dependencies=[Depends(require_demo)])

    @demo.get("/auth/me")
    def demo_identity():
        return {"analyst_id": "Public demo", "role": "viewer", "expires_at": None}

    @demo.get("/runs", response_model=list[RunSummary])
    def demo_runs(session: Session = Depends(get_session)):
        return [run_summary(session, run.run_id) for run in list_runs(session) if demo_eligible(run)]

    @demo.get("/monitoring/overview")
    def demo_monitoring():
        return monitoring_overview()

    @demo.get("/runs/{run_id}/summary", response_model=RunSummary)
    def demo_summary(run=Depends(public_run), session: Session = Depends(get_session)):
        return run_summary(session, run.run_id)

    @demo.get("/runs/{run_id}/cases", response_model=list[CaseListItem])
    def demo_cases(
        status: str | None = None, decision: str | None = None,
        min_score: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
        run=Depends(public_run), session: Session = Depends(get_session),
    ):
        return cases(run.run_id, status, decision, min_score, limit, offset, session)

    @demo.get("/runs/{run_id}/cases/{case_id}", response_model=CaseDetail)
    def demo_detail(case_id: str, run=Depends(public_run), session: Session = Depends(get_session)):
        detail = case_detail(run.run_id, case_id, session)
        # Never publish free-text analyst notes, including notes on synthetic cases.
        detail.analyst_note = ""
        return detail

    @demo.get("/runs/{run_id}/cases/{case_id}/events", response_model=list[ReviewEventResponse])
    def demo_events(case_id: str, run=Depends(public_run), session: Session = Depends(get_session)):
        history = events(run.run_id, case_id, session)
        for item in history:
            item.analyst_note = ""
            item.analyst_id = None
        return history

    @demo.get("/runs/{run_id}/explanations/summary")
    def demo_explanation_summary(run=Depends(public_run), session: Session = Depends(get_session)):
        return explanation_summary(run.run_id, session)

    @demo.get("/runs/{run_id}/cases/{case_id}/explanation", response_model=ExplanationResponse)
    def demo_explanation(case_id: str, run=Depends(public_run), session: Session = Depends(get_session)):
        return case_explanation(run.run_id, case_id, session)

    app.include_router(demo)

    # A production Vite build can be served by the same process. API/docs routes
    # are registered first, so mounting the SPA at / does not shadow them.
    frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if (frontend_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="analyst-console")

    return app
