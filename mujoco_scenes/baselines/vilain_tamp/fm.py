"""Independent, injectable model-call boundary with reproducible artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from typing import Any, Mapping, Protocol

from .artifacts import atomic_write_json, atomic_write_text, redact_secrets


class FMCallType(str, Enum):
    OBJECT_ESTIMATION = "OBJECT_ESTIMATION"
    INITIAL_STATE = "INITIAL_STATE"
    GOAL_STATE = "GOAL_STATE"
    CORRECTIVE_PLANNING = "CORRECTIVE_PLANNING"


class FMTransportError(RuntimeError):
    """Raised when an injected model transport cannot complete a request."""


@dataclass(frozen=True)
class FMRequest:
    call_type: FMCallType
    model: str
    revision: str | None
    messages: tuple[Mapping[str, Any], ...]
    image_artifacts: tuple[str, ...] = ()
    response_format: str = "text"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def sanitized_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "call_type": self.call_type.value,
                "model": self.model,
                "revision": self.revision,
                "messages": [dict(message) for message in self.messages],
                "image_artifacts": list(self.image_artifacts),
                "response_format": self.response_format,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class FMTransportResponse:
    raw_text: str
    call_id: str
    model: str
    revision: str | None
    usage: Mapping[str, int | float]
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)


class QwenObjectEstimatorTransport(Protocol):
    """Injected transport for the object-estimation call."""

    def complete(self, request: FMRequest) -> FMTransportResponse: ...


class GPTReasoningTransport(Protocol):
    """Injected text/multimodal transport for interpretation and correction."""

    def complete(self, request: FMRequest) -> FMTransportResponse: ...


@dataclass(frozen=True)
class FMCallRecord:
    call_type: FMCallType
    call_id: str
    model: str
    revision: str | None
    request_artifact: str
    raw_response_artifact: str
    metadata_artifact: str
    usage: Mapping[str, int | float]
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_type": self.call_type.value,
            "call_id": self.call_id,
            "model": self.model,
            "revision": self.revision,
            "request_artifact": self.request_artifact,
            "raw_response_artifact": self.raw_response_artifact,
            "metadata_artifact": self.metadata_artifact,
            "usage": dict(self.usage),
            "latency_seconds": self.latency_seconds,
        }


class RecordedFMClient:
    """Record one call made through a caller-supplied transport."""

    def __init__(self, transport: QwenObjectEstimatorTransport | GPTReasoningTransport):
        self.transport = transport

    def invoke(
        self,
        request: FMRequest,
        artifact_dir: str | Path,
    ) -> tuple[FMTransportResponse, FMCallRecord]:
        destination = Path(artifact_dir)
        request_path = atomic_write_json(
            destination / "request.json", request.sanitized_dict()
        )
        started = time.perf_counter()
        try:
            response = self.transport.complete(request)
        except Exception as error:
            raise FMTransportError(
                f"{request.call_type.value} transport failed: {error}"
            ) from error
        latency = time.perf_counter() - started
        _validate_response(request, response)

        raw_path = atomic_write_text(destination / "raw_response.txt", response.raw_text)
        metadata_path = atomic_write_json(
            destination / "model_metadata.json",
            {
                "call_type": request.call_type.value,
                "call_id": response.call_id,
                "model": response.model,
                "revision": response.revision,
                "usage": dict(response.usage),
                "provider_metadata": redact_secrets(dict(response.provider_metadata)),
                "latency_seconds": latency,
            },
        )
        record = FMCallRecord(
            call_type=request.call_type,
            call_id=response.call_id,
            model=response.model,
            revision=response.revision,
            request_artifact=str(request_path),
            raw_response_artifact=str(raw_path),
            metadata_artifact=str(metadata_path),
            usage=dict(response.usage),
            latency_seconds=latency,
        )
        return response, record


def _validate_response(request: FMRequest, response: FMTransportResponse) -> None:
    if not response.raw_text.strip():
        raise FMTransportError("model response is empty")
    if not response.call_id.strip():
        raise FMTransportError("model response has no call ID")
    if response.model != request.model:
        raise FMTransportError(
            f"model mismatch: requested {request.model!r}, received {response.model!r}"
        )
    if request.revision is not None and response.revision != request.revision:
        raise FMTransportError(
            f"revision mismatch: requested {request.revision!r}, received {response.revision!r}"
        )
    if any(value < 0 for value in response.usage.values()):
        raise FMTransportError("usage values must be non-negative")
