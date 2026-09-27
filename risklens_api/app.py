"""ASGI entry point for `python -m uvicorn risklens_api.app:app`."""
from .factory import create_app

app = create_app()
