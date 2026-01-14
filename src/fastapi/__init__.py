"""Minimal FastAPI stub for test environments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class Response:
    """Simple response object with JSON payload."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


class FastAPI:
    """Minimal FastAPI-compatible app for tests."""

    def __init__(self) -> None:
        self._routes: Dict[Tuple[str, str], Callable[..., Any]] = {}

    def get(self, path: str):
        def decorator(func: Callable[..., Any]):
            self._routes[("GET", path)] = func
            return func

        return decorator

    def post(self, path: str):
        def decorator(func: Callable[..., Any]):
            self._routes[("POST", path)] = func
            return func

        return decorator

    def _handle(self, method: str, path: str, payload: Any = None) -> Response:
        handler = self._routes.get((method, path))
        if handler is None:
            return Response({"detail": "Not Found"}, status_code=404)
        if payload is None:
            result = handler()
        else:
            result = handler(payload)
        return Response(result)
