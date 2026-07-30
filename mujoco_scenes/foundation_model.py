"""Remote foundation-model ranking over visible candidates only."""

from __future__ import annotations

import http.client
import json
import os
import ssl
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit


SYSTEM_PROMPT = """\
Rank the visible candidates for the requested robot function.
Use general functional suitability and safe, conventional robot use.
Rank every supplied candidate exactly once. Do not invent candidates.
Return only the requested JSON object."""

ASSESSMENT_PROMPT = """\
Assess which visible candidates can perform the requested robot function.
Rank every functional candidate by safe, conventional suitability.
Do not invent candidates. Return only the requested JSON object."""

RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["ranked_candidate_ids"],
    "additionalProperties": False,
}

ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "functional_candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "ranked_candidate_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["functional_candidate_ids", "ranked_candidate_ids"],
    "additionalProperties": False,
}


class FoundationModelError(RuntimeError):
    """Base error for remote inference and response validation."""


class FoundationModelResponseError(FoundationModelError):
    """The server or model returned an unusable response."""


@dataclass(frozen=True)
class Candidate:
    """One observed object or region that may satisfy a function."""

    candidate_id: str
    category: str
    facts: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.category.strip():
            raise ValueError("candidate category must not be empty")
        return {
            "id": self.candidate_id,
            "category": self.category,
            "facts": _json_value(self.facts, "candidate facts"),
        }


@dataclass(frozen=True)
class RankingRequest:
    """The complete, observation-bounded input to candidate ranking."""

    required_function: str
    candidates: Sequence[Candidate]
    target: Mapping[str, object] | None = None


@dataclass(frozen=True)
class RankingResult:
    """A validated ordering of the supplied candidate IDs."""

    candidate_ids: tuple[str, ...]
    model: str
    latency_ms: float | None = None


@dataclass(frozen=True)
class FunctionalAssessmentResult:
    """Boolean functional assessment followed by preference ordering."""

    functional_candidate_ids: tuple[str, ...]
    ranked_candidate_ids: tuple[str, ...]
    model: str
    latency_ms: float | None = None


@dataclass(frozen=True)
class ServerConfig:
    """Connection and generation settings for an OpenAI-compatible server."""

    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "tamp-ranker"
    api_key: str | None = None
    timeout_seconds: float = 15.0
    max_tokens: int = 96

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not self.model.strip():
            raise ValueError("model must not be empty")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> ServerConfig:
        values = os.environ if environ is None else environ
        api_key = values.get("TAMP_FM_API_KEY") or None
        try:
            timeout = float(values.get("TAMP_FM_TIMEOUT_SECONDS", "15"))
            max_tokens = int(values.get("TAMP_FM_MAX_TOKENS", "96"))
        except ValueError as error:
            raise ValueError(
                "TAMP_FM_TIMEOUT_SECONDS and TAMP_FM_MAX_TOKENS "
                "must be numeric"
            ) from error
        return cls(
            base_url=values.get(
                "TAMP_FM_BASE_URL", "http://127.0.0.1:8000/v1"
            ),
            model=values.get("TAMP_FM_MODEL", "tamp-ranker"),
            api_key=api_key,
            timeout_seconds=timeout,
            max_tokens=max_tokens,
        )


class RankingBackend(Protocol):
    """Common interface used by the future task executive."""

    def rank(self, request: RankingRequest) -> RankingResult:
        """Return a complete validated candidate ordering."""


class AssessmentBackend(Protocol):
    def assess(
        self, request: RankingRequest
    ) -> FunctionalAssessmentResult:
        """Return the functional subset and its validated ordering."""


class JsonTransport(Protocol):
    """Small injectable transport used by the server ranker."""

    def post(self, path: str, payload: Mapping[str, object]) -> object:
        """POST JSON and return the decoded response."""

    def close(self) -> None:
        """Release transport resources."""


class OpenAICompatibleRanker:
    """Rank candidates through a vLLM or SGLang chat-completions server."""

    def __init__(
        self,
        config: ServerConfig,
        transport: JsonTransport | None = None,
    ):
        self.config = config
        self._transport = transport or _PersistentJsonTransport(config)

    @classmethod
    def from_env(cls) -> OpenAICompatibleRanker:
        return cls(ServerConfig.from_env())

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> OpenAICompatibleRanker:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _complete(
        self,
        request: RankingRequest,
        prompt: str,
        schema_name: str,
        schema: Mapping[str, object],
    ) -> tuple[Mapping[str, object], tuple[str, ...], float]:
        candidate_payload, candidate_ids = _validate_request(request)
        user_payload: dict[str, object] = {
            "required_function": request.required_function,
            "candidates": candidate_payload,
        }
        if request.target is not None:
            user_payload["target"] = _json_value(
                request.target, "ranking target"
            )

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload, separators=(",", ":"), sort_keys=True
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        started = time.perf_counter()
        response = self._transport.post("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _parse_response(response), candidate_ids, latency_ms

    def rank(self, request: RankingRequest) -> RankingResult:
        result, candidate_ids, latency_ms = self._complete(
            request,
            SYSTEM_PROMPT,
            "candidate_ranking",
            RANKING_SCHEMA,
        )
        ranked_ids = _string_list(result, "ranked_candidate_ids")
        _validate_ranking(ranked_ids, candidate_ids)
        return RankingResult(
            tuple(ranked_ids), self.config.model, latency_ms
        )

    def assess(
        self, request: RankingRequest
    ) -> FunctionalAssessmentResult:
        result, candidate_ids, latency_ms = self._complete(
            request,
            ASSESSMENT_PROMPT,
            "functional_assessment",
            ASSESSMENT_SCHEMA,
        )
        functional_ids = _string_list(
            result, "functional_candidate_ids"
        )
        ranked_ids = _string_list(result, "ranked_candidate_ids")
        _validate_subset(functional_ids, candidate_ids, "functional")
        _validate_ranking(ranked_ids, functional_ids)
        return FunctionalAssessmentResult(
            tuple(functional_ids),
            tuple(ranked_ids),
            self.config.model,
            latency_ms,
        )


class FixedRankingBackend:
    """Small deterministic backend for tests and offline development."""

    def __init__(self, candidate_ids: Sequence[str]):
        self._candidate_ids = tuple(candidate_ids)

    def rank(self, request: RankingRequest) -> RankingResult:
        _, visible_ids = _validate_request(request)
        _validate_ranking(self._candidate_ids, visible_ids)
        return RankingResult(self._candidate_ids, "fixed")


class FixedAssessmentBackend:
    """Deterministic functional assessment for tests and offline runs."""

    def __init__(
        self,
        functional_candidate_ids: Sequence[str],
        ranked_candidate_ids: Sequence[str] | None = None,
    ):
        self._functional_ids = tuple(functional_candidate_ids)
        self._ranked_ids = tuple(
            functional_candidate_ids
            if ranked_candidate_ids is None
            else ranked_candidate_ids
        )

    def assess(
        self, request: RankingRequest
    ) -> FunctionalAssessmentResult:
        _, visible_ids = _validate_request(request)
        _validate_subset(self._functional_ids, visible_ids, "functional")
        _validate_ranking(self._ranked_ids, self._functional_ids)
        return FunctionalAssessmentResult(
            self._functional_ids,
            self._ranked_ids,
            "fixed",
        )


def _json_value(value: object, label: str) -> object:
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only JSON values") from error
    return json.loads(encoded)


def _validate_request(
    request: RankingRequest,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    if not request.required_function.strip():
        raise ValueError("required_function must not be empty")
    candidates = tuple(request.candidates)
    if not candidates:
        raise ValueError("at least one visible candidate is required")
    payload = [candidate.as_dict() for candidate in candidates]
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    return payload, candidate_ids


def _parse_response(response: object) -> Mapping[str, object]:
    try:
        choices = response["choices"]  # type: ignore[index]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise FoundationModelResponseError(
            "response has no chat-completion content"
        ) from error
    if not isinstance(content, str):
        raise FoundationModelResponseError(
            "chat-completion content must be text"
        )
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError) as error:
        raise FoundationModelResponseError(
            "model did not return a JSON object"
        ) from error
    if not isinstance(result, dict):
        raise FoundationModelResponseError("model response must be an object")
    return result


def _string_list(
    result: Mapping[str, object], field_name: str
) -> list[str]:
    values = result.get(field_name)
    if (
        not isinstance(values, list)
        or any(not isinstance(item, str) for item in values)
    ):
        raise FoundationModelResponseError(
            f"{field_name} must be a list of strings"
        )
    return values


def _validate_subset(
    candidate_ids: Sequence[str],
    visible_ids: Sequence[str],
    label: str,
) -> None:
    if len(candidate_ids) != len(set(candidate_ids)):
        raise FoundationModelResponseError(
            f"{label} candidates contain duplicate IDs"
        )
    unknown = set(candidate_ids) - set(visible_ids)
    if unknown:
        raise FoundationModelResponseError(
            f"{label} candidates include unobserved IDs: {sorted(unknown)}"
        )


def _validate_ranking(
    ranked_ids: Sequence[str], visible_ids: Sequence[str]
) -> None:
    expected = set(visible_ids)
    actual = set(ranked_ids)
    if len(ranked_ids) != len(actual):
        raise FoundationModelResponseError(
            "ranking contains duplicate candidate IDs"
        )
    unknown = actual - expected
    if unknown:
        raise FoundationModelResponseError(
            f"ranking contains unobserved candidates: {sorted(unknown)}"
        )
    missing = expected - actual
    if missing:
        raise FoundationModelResponseError(
            f"ranking omitted visible candidates: {sorted(missing)}"
        )


class _PersistentJsonTransport:
    """One persistent HTTP/1.1 connection with stale-connection recovery."""

    def __init__(self, config: ServerConfig):
        parsed = urlsplit(config.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an http(s) URL without credentials, "
                "query, or fragment"
            )
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        path = parsed.path.rstrip("/")
        self._base_path = "/v1" if not path else path
        self._api_key = config.api_key
        self._timeout = config.timeout_seconds
        self._connection: (
            http.client.HTTPConnection | http.client.HTTPSConnection | None
        ) = None
        self._lock = threading.Lock()

    def _connect(
        self,
    ) -> http.client.HTTPConnection | http.client.HTTPSConnection:
        if self._connection is None:
            if self._scheme == "https":
                self._connection = http.client.HTTPSConnection(
                    self._host,
                    self._port,
                    timeout=self._timeout,
                    context=ssl.create_default_context(),
                )
            else:
                self._connection = http.client.HTTPConnection(
                    self._host, self._port, timeout=self._timeout
                )
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _drop_connection(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def post(self, path: str, payload: Mapping[str, object]) -> object:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        endpoint = f"{self._base_path}{path}"

        with self._lock:
            for attempt in range(2):
                connection = self._connect()
                try:
                    connection.request("POST", endpoint, body, headers)
                    response = connection.getresponse()
                    response_body = response.read()
                except (
                    ConnectionError,
                    http.client.HTTPException,
                    OSError,
                    TimeoutError,
                ) as error:
                    self._drop_connection()
                    if attempt == 0:
                        continue
                    raise FoundationModelError(
                        f"inference request failed: {error}"
                    ) from error

                if response.will_close:
                    self._drop_connection()
                text = response_body.decode("utf-8", errors="replace")
                if not 200 <= response.status < 300:
                    detail = text[:500].strip()
                    raise FoundationModelError(
                        f"inference server returned HTTP {response.status}: "
                        f"{detail or response.reason}"
                    )
                try:
                    return json.loads(text)
                except json.JSONDecodeError as error:
                    raise FoundationModelResponseError(
                        "inference server returned invalid JSON"
                    ) from error
        raise AssertionError("unreachable")
