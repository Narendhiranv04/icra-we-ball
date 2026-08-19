"""OpenAI-compatible client for the training-free LLM3-style planner."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .catalog import load_catalog, scene_actions
from .models import Failure, Observation, Plan, ValidationError, parse_plan
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, response_schema, task_payload


MODEL_REGISTRY = Path(__file__).parents[1] / "inference_server" / "models.json"
QWEN_THINKING_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


class PlanningError(RuntimeError):
    """The model request or completion could not produce a valid plan."""


class CompletionTransport(Protocol):
    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """Return one decoded OpenAI-compatible completion."""


@dataclass(frozen=True)
class PlannerConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen35-9b"
    api_key: str = ""
    timeout_seconds: float = 300.0
    max_tokens: int = 12288
    max_actions: int = 20
    enable_thinking: bool = True
    sampling: Mapping[str, object] = field(
        default_factory=lambda: dict(QWEN_THINKING_SAMPLING)
    )
    structured_output: bool = True
    reasoning_markers: tuple[str, str] | None = None
    system_prompt_prefix: str = ""
    toggle_thinking: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError("timeout and max_tokens must be positive")
        if self.max_actions <= 0:
            raise ValueError("max_actions must be positive")
        if not isinstance(self.sampling, Mapping):
            raise ValueError("sampling must be an object")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "PlannerConfig":
        values = os.environ if environ is None else environ
        thinking = values.get("LLM3_ENABLE_THINKING", "true").strip().lower()
        if thinking not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("LLM3_ENABLE_THINKING must be true or false")
        profile_name = values.get(
            "LLM3_PROFILE", values.get("INFERENCE_MODEL", "qwen35-9b")
        ).strip()
        profile = _load_profile(profile_name)
        planner = profile.get("planner", {})
        thinking_mode = str(planner.get("thinking_mode", "toggle"))
        enable_thinking = thinking in {"true", "1", "yes"}
        if thinking_mode == "always" and not enable_thinking:
            raise ValueError(f"{profile_name} is a fixed-thinking checkpoint")
        if thinking_mode == "none" and enable_thinking:
            raise ValueError(f"{profile_name} has no configured thinking mode")
        sampling_name = "thinking" if enable_thinking else "direct"
        sampling = dict(
            planner.get("sampling", {}).get(
                sampling_name, QWEN_THINKING_SAMPLING
            )
        )
        if "LLM3_TEMPERATURE" in values:
            sampling["temperature"] = float(values["LLM3_TEMPERATURE"])
        if "LLM3_TOP_P" in values:
            sampling["top_p"] = float(values["LLM3_TOP_P"])
        markers = planner.get("reasoning_markers")
        if markers is not None and (
            not isinstance(markers, list) or len(markers) != 2
        ):
            raise ValueError(f"Invalid reasoning markers for {profile_name}")
        return cls(
            base_url=values.get(
                "LLM3_MODEL_BASE_URL",
                values.get("PLANNER_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
            ),
            model=values.get(
                "LLM3_MODEL",
                values.get("PLANNER_MODEL", str(profile.get("served_name", profile_name))),
            ),
            api_key=values.get(
                "LLM3_API_KEY", values.get("INFERENCE_API_KEY", "")
            ),
            timeout_seconds=float(values.get("LLM3_TIMEOUT_SECONDS", "300")),
            max_tokens=int(
                values.get("LLM3_MAX_TOKENS", str(planner.get("max_tokens", 12288)))
            ),
            max_actions=int(values.get("LLM3_MAX_ACTIONS", "20")),
            enable_thinking=enable_thinking,
            sampling=sampling,
            structured_output=bool(planner.get("structured_output", True)),
            reasoning_markers=(tuple(markers) if markers else None),
            system_prompt_prefix=str(planner.get("system_prompt_prefix", "")),
            toggle_thinking=thinking_mode == "toggle",
        )


@dataclass(frozen=True)
class PlanResult:
    plan: Plan
    model: str
    latency_ms: float
    prompt_version: int = PROMPT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_version": self.prompt_version,
            "execution_started": False,
        }


class OpenAITransport:
    def __init__(self, config: PlannerConfig):
        self.config = config

    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise PlanningError(
                f"Model server returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise PlanningError(f"Cannot reach model server: {error.reason}") from error
        if not isinstance(result, Mapping):
            raise PlanningError("Model server response must be a JSON object")
        return result

    def _headers(self) -> dict[str, str]:
        result = {"Content-Type": "application/json"}
        if self.config.api_key:
            result["Authorization"] = f"Bearer {self.config.api_key}"
        return result


class LLM3Planner:
    def __init__(
        self,
        config: PlannerConfig,
        *,
        transport: CompletionTransport | None = None,
        catalog: Mapping[str, Any] | None = None,
    ):
        self.config = config
        self.transport = transport or OpenAITransport(config)
        self.catalog = catalog or load_catalog()

    @classmethod
    def from_env(cls) -> "LLM3Planner":
        return cls(PlannerConfig.from_env())

    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        *,
        history: Sequence[Mapping[str, Any]] = (),
        failure: Failure | None = None,
    ) -> PlanResult:
        normalized_goal = goal.strip()
        if not normalized_goal or len(normalized_goal) > 4000:
            raise ValidationError("goal must contain between 1 and 4000 characters")
        normalized_images = _validate_images(images)
        actions = scene_actions(self.catalog, observation.scene)
        payload = self._completion_payload(
            normalized_goal,
            observation,
            normalized_images,
            history,
            failure,
            actions,
        )
        started = time.perf_counter()
        response = self.transport.complete(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = _response_content(response, self.config.reasoning_markers)
        try:
            plan = parse_plan(
                content,
                actions,
                observation,
                max_actions=self.config.max_actions,
            )
        except ValidationError as error:
            raise PlanningError(f"Invalid model plan: {error}") from error
        return PlanResult(plan, self.config.model, latency_ms)

    def _completion_payload(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        history: Sequence[Mapping[str, Any]],
        failure: Failure | None,
        actions: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, object]:
        schema = response_schema(sorted(actions), self.config.max_actions)
        task = task_payload(goal, observation, actions, history, failure)
        if not self.config.structured_output:
            task["output_schema"] = schema
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(task, separators=(",", ":"), sort_keys=True),
            }
        ]
        for image in images:
            content.append(
                {"type": "text", "text": f"Camera view: {image['camera']}"}
            )
            content.append(
                {"type": "image_url", "image_url": {"url": image["data_url"]}}
            )
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{self.config.system_prompt_prefix}\n\n{SYSTEM_PROMPT}"
                        if self.config.system_prompt_prefix
                        else SYSTEM_PROMPT
                    ),
                },
                {"role": "user", "content": content},
            ],
            **dict(self.config.sampling),
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if self.config.toggle_thinking:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.config.enable_thinking
            }
        if self.config.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm3_action_plan",
                    "strict": True,
                    "schema": schema,
                },
            }
        return payload


def _validate_images(
    images: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], ...]:
    if not 1 <= len(images) <= 8:
        raise ValidationError("images must contain between 1 and 8 views")
    result = []
    cameras: set[str] = set()
    for image in images:
        camera = image.get("camera", "").strip()
        data_url = image.get("data_url", "")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", camera):
            raise ValidationError(f"Invalid camera name {camera!r}")
        if camera in cameras:
            raise ValidationError(f"Duplicate camera name {camera!r}")
        if not data_url.startswith("data:image/") or ";base64," not in data_url[:128]:
            raise ValidationError(f"Camera {camera!r} needs a base64 image data URL")
        cameras.add(camera)
        result.append({"camera": camera, "data_url": data_url})
    return tuple(result)


def _response_content(
    response: Mapping[str, object],
    reasoning_markers: tuple[str, str] | None = None,
) -> Mapping[str, Any]:
    try:
        choice = response["choices"][0]  # type: ignore[index]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise PlanningError("Completion response has no message content") from error
    if isinstance(content, Mapping):
        return content
    if content is None:
        reason = choice.get("finish_reason", "unknown")
        raise PlanningError(
            "Completion has no final JSON content "
            f"(finish_reason={reason}); increase LLM3_MAX_TOKENS"
        )
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        )
    if not isinstance(content, str):
        raise PlanningError("Completion content must be JSON text")
    content = content.strip()
    if reasoning_markers and reasoning_markers[1] in content:
        content = content.split(reasoning_markers[1], 1)[1].strip()
    if content.startswith("```") and content.endswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise PlanningError("Completion content is not valid JSON") from error
    if not isinstance(result, Mapping):
        raise PlanningError("Completion JSON must be an object")
    return result


def _load_profile(name: str) -> Mapping[str, Any]:
    if not MODEL_REGISTRY.is_file():
        if name == "qwen35-9b":
            return {
                "served_name": name,
                "planner": {
                    "thinking_mode": "toggle",
                    "max_tokens": 12288,
                    "sampling": {"thinking": QWEN_THINKING_SAMPLING},
                },
            }
        raise ValueError(f"Model registry not found: {MODEL_REGISTRY}")
    document = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
    profiles = document.get("models", {})
    if name not in profiles or profiles[name].get("available", True) is not True:
        raise ValueError(f"Unknown or unavailable inference profile {name!r}")
    return profiles[name]
