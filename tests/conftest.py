"""Shared test helpers: a mock httpx client driven by recorded/synthetic routes.

All adapter tests are hermetic: they never touch a live server. Each adapter is
handed an ``httpx.Client`` backed by :class:`httpx.MockTransport`, which returns
canned responses keyed by URL path.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import httpx
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
LIVE = FIXTURES / "live"


def load_text(name: str) -> str:
    """Read a fixture file's raw text."""
    path = LIVE / name
    if not path.exists():
        path = FIXTURES / name
    return path.read_text()


def _response(status: int, body: Any) -> httpx.Response:
    """Build an httpx.Response. dict/list bodies are JSON-encoded; str bodies are
    sent verbatim with a content-type inferred from whether they parse as JSON."""
    if isinstance(body, (dict, list)):
        text = json.dumps(body)
        ct = "application/json"
    else:
        text = str(body)
        try:
            json.loads(text)
            ct = "application/json"
        except Exception:
            ct = "text/html; charset=utf-8"
    return httpx.Response(status, content=text.encode(), headers={"content-type": ct})


def make_client(routes: dict[str, Any], *, default_status: int = 404) -> httpx.Client:
    """Return an httpx.Client whose responses are dictated by ``routes``.

    ``routes`` maps a URL path (e.g. ``"/v1/models"``) to either:
      - ``(status, body)`` tuple, or
      - ``body`` (implies HTTP 200), or
      - an ``httpx.Response`` factory taking the request.

    Unmatched paths return ``default_status``.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        spec = routes.get(request.url.path)
        if spec is None:
            return httpx.Response(default_status, text="not found")
        if callable(spec):
            return spec(request)
        if isinstance(spec, tuple):
            status, body = spec
            return _response(status, body)
        return _response(200, spec)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


@pytest.fixture
def vllm_routes() -> dict[str, Any]:
    """Routes that mimic a real vLLM 0.20 server (from recorded fixtures)."""
    return {
        "/v1/models": load_text("vllm_8088_models.json"),
        "/metrics": (200, load_text("vllm_8088_metrics.txt")),
        "/version": {"version": "0.20.1+test"},
    }


@pytest.fixture
def ollama_routes() -> dict[str, Any]:
    """Routes that mimic a real Ollama server (recorded fixtures; /api/ps empty)."""
    return {
        "/api/tags": load_text("ollama_tags.json"),
        "/api/ps": load_text("ollama_ps.json"),
        "/api/version": load_text("ollama_version.json"),
    }


@pytest.fixture
def auth_router_routes() -> dict[str, Any]:
    """An auth-gated OpenAI router: 401 on everything except an open /health."""
    err = {"error": {"message": "missing or invalid API key",
                     "type": "authentication_error", "code": "invalid_api_key"}}
    return {
        "/health": (200, {"status": "ok"}),
        "/v1/models": (401, err),
        "/api/tags": (401, err),
        "/api/version": (401, err),
        "/get_model_info": (401, err),
        "/info": (401, err),
        "/props": (401, err),
    }


@pytest.fixture
def open_webui_routes() -> dict[str, Any]:
    """Open WebUI: HTML everywhere, but /api/version returns JSON (the trap)."""
    html = "<!doctype html><html><head><title>Open WebUI</title></head></html>"
    return {
        "/": (200, html),
        "/v1/models": (200, html),
        "/api/tags": (200, html),
        "/api/version": (200, {"version": "0.9.5", "deployment_id": ""}),
    }
