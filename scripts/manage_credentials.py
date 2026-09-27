"""Trusted operator CLI: issue/revoke short-lived credentials; never commit tokens."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risklens_api.database import build_database
from risklens_api.auth import ApiCredential, ROLES, issue_credential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--analyst-id", required=True)
    create.add_argument("--role", choices=ROLES, required=True)
    create.add_argument("--hours", type=int, default=8)
    revoke = sub.add_parser("revoke")
    revoke.add_argument("--credential-id", required=True)
    args = parser.parse_args()
    engine, sessions = build_database()
    try:
        with sessions() as session:
            if args.action == "create":
                credential, token = issue_credential(session, args.analyst_id, args.role, args.hours)
                print(f"Credential ID: {credential.credential_id}")
                print(f"Identity: {credential.analyst_id}; role: {credential.role}; expires: {credential.expires_at.isoformat()}")
                print("Copy this token into Swagger Authorize; it is shown only once. Do not share it:")
                print(token)
            else:
                credential = session.get(ApiCredential, args.credential_id)
                if credential is None:
                    raise ValueError("Credential not found")
                credential.revoked = True
                session.commit()
                print("Credential revoked.")
    finally:
        engine.dispose()

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
