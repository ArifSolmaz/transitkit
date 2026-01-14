"""Minimal TestClient stub for FastAPI."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Response


class TestClient:
    """Test client that calls FastAPI stub handlers."""

    __test__ = False

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def get(self, path: str) -> Response:
        return self.app._handle("GET", path)

    def post(self, path: str, json: Any = None) -> Response:
        return self.app._handle("POST", path, payload=json)
