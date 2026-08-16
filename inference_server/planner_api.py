"""HTTP gateway for VLM functional decomposition."""

from __future__ import annotations

import hmac
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

try:
    from .functional_planner import (
        FunctionalPlanner,
        FunctionalPlanningError,
        PlannerConfig,
    )
    from .server import load_profiles
except ImportError:  # Direct execution from this standalone workspace.
    from functional_planner import (
        FunctionalPlanner,
        FunctionalPlanningError,
        PlannerConfig,
    )
    from server import load_profiles


MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _is_loopback(host: str) -> bool:
    return host.strip().strip("[]").lower() in {"127.0.0.1", "::1", "localhost"}


def _boolean_env(values: dict[str, str], name: str, default: bool) -> bool:
    value = values.get(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def config_from_env(environ: dict[str, str] | None = None) -> tuple[PlannerConfig, str, str, int]:
    values = os.environ if environ is None else environ
    upstream_key = values.get("INFERENCE_API_KEY", "").strip()
    incoming_key = values.get("PLANNER_API_KEY", "").strip() or upstream_key
    profile_name = values.get("INFERENCE_MODEL", "").strip()
    model = values.get("PLANNER_MODEL", "").strip()
    if not model and profile_name:
        profile = load_profiles().get(profile_name)
        if profile and profile.get("available", True):
            model = str(profile["served_name"])
    if not model:
        raise ValueError("PLANNER_MODEL or INFERENCE_MODEL is required")
    timeout = float(values.get("PLANNER_MODEL_TIMEOUT_SECONDS", "300"))
    max_tokens = int(values.get("PLANNER_MAX_TOKENS", "8192"))
    enable_thinking = _boolean_env(values, "PLANNER_ENABLE_THINKING", True)
    port = int(values.get("PLANNER_PORT", "8080"))
    host = values.get("PLANNER_HOST", "127.0.0.1").strip()
    if timeout <= 0 or max_tokens <= 0 or not 1 <= port <= 65535:
        raise ValueError("Planner timeout, token limit, and port must be positive")
    if not incoming_key and not _is_loopback(host):
        raise ValueError("PLANNER_API_KEY is required for a non-loopback host")
    config = PlannerConfig(
        base_url=values.get("PLANNER_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
        model=model,
        api_key=upstream_key,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    return config, incoming_key, host, port


class PlanningHTTPServer(ThreadingHTTPServer):
    planner: FunctionalPlanner
    api_key: str


class Handler(BaseHTTPRequestHandler):
    server: PlanningHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if not self.server.api_key:
            return True
        authorization = self.headers.get("Authorization", "")
        if hmac.compare_digest(authorization, f"Bearer {self.server.api_key}"):
            return True
        self._json(HTTPStatus.UNAUTHORIZED, {"error": "Invalid API key"})
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "functional-planner",
                    "model": self.server.planner.config.model,
                },
            )
            return
        if self.path == "/v1/functions":
            if not self._authorized():
                return
            self._json(
                HTTPStatus.OK,
                {
                    "functions": self.server.planner.catalog["functions"],
                    "min_ranked_candidates": self.server.planner.catalog[
                        "min_ranked_candidates"
                    ],
                    "max_ranked_candidates": self.server.planner.catalog[
                        "max_ranked_candidates"
                    ],
                },
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/decompose":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        if not self._authorized():
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return
        if not 1 <= content_length <= MAX_REQUEST_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request too large"})
            return
        try:
            request = json.loads(self.rfile.read(content_length))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object")
            result = self.server.planner.decompose(request)
        except (json.JSONDecodeError, ValueError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except FunctionalPlanningError as error:
            self._json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
            return
        self._json(HTTPStatus.OK, result)


def build_server(
    host: str,
    port: int,
    planner: FunctionalPlanner,
    api_key: str,
) -> PlanningHTTPServer:
    server = PlanningHTTPServer((host, port), Handler)
    server.planner = planner
    server.api_key = api_key
    return server


def main() -> None:
    try:
        config, api_key, host, port = config_from_env()
    except ValueError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    server = build_server(host, port, FunctionalPlanner(config), api_key)
    print(
        f"Functional planning API listening on http://{host}:{port} "
        f"for {config.model}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
