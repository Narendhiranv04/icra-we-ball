"""Lazy, baseline-owned transports for the paper-faithful model condition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Any, Mapping, Protocol, Sequence

from .config import BaselineConfig, Domain, ModelCondition
from .contracts import CameraFrameArtifacts, ViLaInObservation
from .domains import load_domain
from .fm import (
    FMCallRecord,
    FMCallType,
    FMRequest,
    FMTransportError,
    FMTransportResponse,
    RecordedFMClient,
)
from .prompts import build_object_estimation_prompt


PAPER_QWEN_MODEL = "Qwen2.5-VL-7B-Instruct"
PAPER_QWEN_SOURCE = "Qwen/Qwen2.5-VL-7B-Instruct"
PAPER_REASONING_MODEL = "gpt-4o-2024-08-06"
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class QwenGeneration:
    text: str
    input_tokens: int
    output_tokens: int
    device: str
    dtype: str


class QwenBackend(Protocol):
    def generate(
        self,
        *,
        model_source: str,
        revision: str,
        messages: Sequence[Mapping[str, Any]],
        image_paths: Sequence[Path],
        max_new_tokens: int,
    ) -> QwenGeneration: ...


@dataclass(frozen=True)
class LiveModelClients:
    """Concrete clients and exact model identities for interpreter/CP wiring."""

    object_client: RecordedFMClient
    reasoning_client: RecordedFMClient
    object_estimator_model: str
    object_estimator_revision: str
    reasoning_model: str
    reasoning_model_revision: str


def build_paper_faithful_clients(
    *,
    config: BaselineConfig,
    image_root: str | Path,
    qwen_revision: str,
    qwen_model_source: str = PAPER_QWEN_SOURCE,
    qwen_backend: QwenBackend | None = None,
    openai_client: Any | None = None,
) -> LiveModelClients:
    """Resolve the paper condition without initializing either model backend."""
    if config.model_condition is not ModelCondition.PAPER_FAITHFUL:
        raise ValueError("live paper-faithful clients require paper_faithful config")
    if config.object_estimator_model != PAPER_QWEN_MODEL:
        raise ValueError("configuration does not select the paper Qwen model")
    if config.reasoning_model != PAPER_REASONING_MODEL:
        raise ValueError("configuration does not select the pinned GPT-4o snapshot")
    if not _FULL_COMMIT.fullmatch(qwen_revision):
        raise ValueError("qwen_revision must be an exact 40-character commit")
    return LiveModelClients(
        object_client=RecordedFMClient(
            QwenVLTransport(
                model_source=qwen_model_source,
                image_root=image_root,
                backend=qwen_backend,
            )
        ),
        reasoning_client=RecordedFMClient(
            OpenAIReasoningTransport(
                timeout_seconds=config.timeouts.model_seconds,
                client=openai_client,
            )
        ),
        object_estimator_model=PAPER_QWEN_MODEL,
        object_estimator_revision=qwen_revision,
        reasoning_model=PAPER_REASONING_MODEL,
        reasoning_model_revision=PAPER_REASONING_MODEL,
    )


class QwenVLTransport:
    """Local Qwen2.5-VL transport loaded only when ``complete`` is called."""

    def __init__(
        self,
        *,
        model_source: str = PAPER_QWEN_SOURCE,
        image_root: str | Path,
        max_new_tokens: int = 2048,
        backend: QwenBackend | None = None,
        require_dedicated_venv: bool = True,
    ) -> None:
        if not model_source.strip():
            raise ValueError("model_source must not be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        self.model_source = model_source
        self.image_root = Path(image_root).resolve()
        self.max_new_tokens = max_new_tokens
        self._backend = backend
        self.require_dedicated_venv = require_dedicated_venv

    def complete(self, request: FMRequest) -> FMTransportResponse:
        if request.call_type is not FMCallType.OBJECT_ESTIMATION:
            raise FMTransportError("Qwen transport accepts only object-estimation calls")
        if request.model != PAPER_QWEN_MODEL:
            raise FMTransportError(
                f"paper-faithful object model must be {PAPER_QWEN_MODEL!r}"
            )
        if request.revision is None or not _FULL_COMMIT.fullmatch(request.revision):
            raise FMTransportError("Qwen revision must be an exact 40-character commit")
        image_paths = tuple(self._resolve_image(item) for item in request.image_artifacts)
        if not image_paths:
            raise FMTransportError("object estimation requires at least one RGB image")

        backend = self._backend
        if backend is None:
            if self.require_dedicated_venv:
                require_vilain_environment()
            backend = _TransformersQwenBackend()
            self._backend = backend
        generated = backend.generate(
            model_source=self.model_source,
            revision=request.revision,
            messages=request.messages,
            image_paths=image_paths,
            max_new_tokens=self.max_new_tokens,
        )
        return FMTransportResponse(
            raw_text=generated.text,
            call_id=f"local-qwen-{uuid.uuid4()}",
            model=request.model,
            revision=request.revision,
            usage={
                "input_tokens": generated.input_tokens,
                "output_tokens": generated.output_tokens,
            },
            provider_metadata={
                "provider": "local_transformers",
                "model_source": self.model_source,
                "resolved_revision": request.revision,
                "device": generated.device,
                "dtype": generated.dtype,
                "max_new_tokens": self.max_new_tokens,
            },
        )

    def _resolve_image(self, value: str) -> Path:
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.image_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.image_root)
        except ValueError as error:
            raise FMTransportError(f"image escapes observation root: {value}") from error
        if not resolved.is_file():
            raise FMTransportError(f"object-estimation image is missing: {value}")
        return resolved


class _TransformersQwenBackend:
    """Actual Transformers backend; heavyweight imports remain call-local."""

    def __init__(self) -> None:
        self._loaded_key: tuple[str, str] | None = None
        self._model: Any | None = None
        self._processor: Any | None = None

    def generate(
        self,
        *,
        model_source: str,
        revision: str,
        messages: Sequence[Mapping[str, Any]],
        image_paths: Sequence[Path],
        max_new_tokens: int,
    ) -> QwenGeneration:
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise FMTransportError(
                "Qwen runtime dependencies are unavailable in .venv-vilain-tamp"
            ) from error

        key = (model_source, revision)
        if self._loaded_key != key:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_source,
                revision=revision,
                torch_dtype="auto",
            )
            self._model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoProcessor.from_pretrained(
                model_source, revision=revision
            )
            self._loaded_key = key
        model = self._model
        processor = self._processor
        assert model is not None and processor is not None
        rendered_messages = _qwen_messages(messages, image_paths)
        prompt = processor.apply_chat_template(
            rendered_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(rendered_messages)
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        parameter = next(model.parameters())
        return QwenGeneration(
            text=text,
            input_tokens=int(inputs.input_ids.shape[-1]),
            output_tokens=int(trimmed[0].shape[-1]),
            device=str(parameter.device),
            dtype=str(parameter.dtype),
        )


class OpenAIReasoningTransport:
    """Pinned GPT-4o transport with environment-owned credentials."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        client: Any | None = None,
        require_dedicated_venv: bool = True,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.timeout_seconds = timeout_seconds
        self._client = client
        self.require_dedicated_venv = require_dedicated_venv

    def complete(self, request: FMRequest) -> FMTransportResponse:
        if request.call_type is FMCallType.OBJECT_ESTIMATION:
            raise FMTransportError("GPT reasoning transport does not estimate objects")
        if request.model != PAPER_REASONING_MODEL:
            raise FMTransportError(
                f"paper-faithful reasoning model must be {PAPER_REASONING_MODEL!r}"
            )
        if request.revision not in (None, PAPER_REASONING_MODEL):
            raise FMTransportError("GPT revision must equal the immutable model snapshot")
        if request.image_artifacts:
            raise FMTransportError("reasoning calls must not receive observation images")

        client = self._client
        if client is None:
            if self.require_dedicated_venv:
                require_vilain_environment()
            try:
                from openai import OpenAI
            except ImportError as error:
                raise FMTransportError(
                    "OpenAI runtime dependency is unavailable in .venv-vilain-tamp"
                ) from error
            client = OpenAI(timeout=self.timeout_seconds)
            self._client = client

        response = client.chat.completions.create(
            model=request.model,
            messages=[dict(message) for message in request.messages],
            temperature=0,
        )
        provider_model = str(response.model)
        if provider_model != PAPER_REASONING_MODEL:
            raise FMTransportError(
                f"provider returned unexpected model snapshot {provider_model!r}"
            )
        raw_text = _openai_text(response)
        usage = getattr(response, "usage", None)
        call_id = getattr(response, "id", None)
        if not isinstance(call_id, str) or not call_id.strip():
            raise FMTransportError("OpenAI response has no call ID")
        return FMTransportResponse(
            raw_text=raw_text,
            call_id=call_id,
            model=provider_model,
            revision=PAPER_REASONING_MODEL,
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
            provider_metadata={
                "provider": "openai",
                "response_model": provider_model,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "temperature": 0,
                "timeout_seconds": self.timeout_seconds,
            },
        )


def require_vilain_environment() -> None:
    """Require live model initialization to occur in the dedicated venv."""
    active = os.environ.get("VIRTUAL_ENV")
    environment = Path(active).name if active else Path(sys.prefix).name
    if environment != ".venv-vilain-tamp":
        raise FMTransportError(
            "live model calls require the dedicated .venv-vilain-tamp environment"
        )


def _qwen_messages(
    messages: Sequence[Mapping[str, Any]], image_paths: Sequence[Path]
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    attached = False
    for message in messages:
        role = str(message.get("role", ""))
        content: list[dict[str, str]] = []
        if role == "user" and not attached:
            content.extend({"type": "image", "image": str(path)} for path in image_paths)
            attached = True
        content.append({"type": "text", "text": str(message.get("content", ""))})
        rendered.append({"role": role, "content": content})
    if not attached:
        raise FMTransportError("Qwen request has no user message for image attachment")
    return rendered


def _openai_text(response: Any) -> str:
    choices = getattr(response, "choices", ())
    if not choices:
        raise FMTransportError("OpenAI response has no choices")
    content = getattr(choices[0].message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise FMTransportError("OpenAI response has no text content")
    return content


def _load_observations(path: Path) -> tuple[ViLaInObservation, ...]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    rows = loaded.get("observations") if isinstance(loaded, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("observation manifest must contain observations")
    observations = []
    for row in rows:
        frames = tuple(CameraFrameArtifacts(**frame) for frame in row["camera_frames"])
        observations.append(ViLaInObservation(**{**row, "camera_frames": frames}))
    return tuple(observations)


def validate_standalone_object_estimation(
    *,
    observation_manifest: str | Path,
    task_instruction: str,
    domain: Domain | str,
    model_source: str,
    revision: str,
    output_directory: str | Path,
    transport: QwenVLTransport | None = None,
) -> FMCallRecord:
    """Make exactly one recorded object-estimation call from captured views."""
    if not task_instruction.strip():
        raise ValueError("task_instruction must not be empty")
    manifest = Path(observation_manifest).resolve()
    observations = _load_observations(manifest)
    domain_key = domain.value if isinstance(domain, Domain) else Domain(domain).value
    if any(item.domain != domain_key for item in observations):
        raise ValueError("observation manifest domain does not match --domain")
    bundle = build_object_estimation_prompt(
        task_instruction=task_instruction,
        domain=load_domain(domain_key),
        observations=observations,
    )
    live_transport = transport or QwenVLTransport(
        model_source=model_source, image_root=manifest.parent
    )
    _, record = RecordedFMClient(live_transport).invoke(
        FMRequest(
            call_type=FMCallType.OBJECT_ESTIMATION,
            model=PAPER_QWEN_MODEL,
            revision=revision,
            messages=bundle.messages(),
            image_artifacts=bundle.image_artifacts,
            response_format="json",
            metadata={
                "model_source": model_source,
                "observation_manifest": str(manifest),
            },
        ),
        output_directory,
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one standalone, recorded ViLaIn object-estimation call."
    )
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--domain", choices=[item.value for item in Domain], required=True)
    parser.add_argument("--model-source", default=PAPER_QWEN_SOURCE)
    parser.add_argument("--revision", required=True, help="Exact 40-character HF commit")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    record = validate_standalone_object_estimation(
        observation_manifest=args.observation_manifest,
        task_instruction=args.task,
        domain=args.domain,
        model_source=args.model_source,
        revision=args.revision,
        output_directory=args.output_directory,
    )
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
