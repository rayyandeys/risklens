"""Expiring, revocable opaque bearer credentials for the single-team review API.

Only a SHA-256 digest is stored. Tokens contain 256 bits of random entropy;
this is not password hashing. Provision/revoke via the trusted operator CLI.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import re
import secrets

from sqlalchemy import Boolean, CheckConstraint, DateTime, String, select
from sqlalchemy.orm import Mapped, mapped_column

from risklens_core.persistence import Base, utcnow

ROLES = ("viewer", "analyst", "admin")


class ApiCredential(Base):
    __tablename__ = "api_credentials"
    __table_args__ = (CheckConstraint("role IN ('viewer', 'analyst', 'admin')", name="ck_credential_role"),)
    credential_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    analyst_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    token_sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


def issue_credential(session, analyst_id: str, role: str, hours: int = 8):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,127}", analyst_id):
        raise ValueError("Identity must be 1-128 letters/digits or . _ @ -, starting with a letter/digit")
    if role not in ROLES or not 1 <= hours <= 720:
        raise ValueError("Invalid role or expiry (1-720 hours)")
    token = "rl_" + secrets.token_urlsafe(32)
    now = utcnow()
    credential = ApiCredential(
        credential_id=secrets.token_hex(16), analyst_id=analyst_id, role=role,
        token_sha256=hashlib.sha256(token.encode("ascii")).hexdigest(),
        created_at=now, expires_at=now + timedelta(hours=hours), revoked=False,
    )
    session.add(credential)
    session.commit()
    return credential, token


def authenticate(session, token: str):
    if not re.fullmatch(r"rl_[A-Za-z0-9_-]{43}", token):
        return None
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    credential = session.scalar(select(ApiCredential).where(ApiCredential.token_sha256 == digest))
    if credential is None or credential.revoked or credential.role not in ROLES:
        return None
    expires = credential.expires_at
    # SQLite returns naive timestamps; all writes use UTC.
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires <= utcnow():
        return None
    return credential
