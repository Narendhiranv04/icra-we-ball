"""OpenAI-compatible client for the training-free LLM3-style planner."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from baseline_common.inference import (
    OpenAITransport,
    PlanningError,
    QWEN_THINKING_SAMPLING,
    load_model_profile,
    response_content,
    validate_images,
)

from .catalog import (
    load_catalog,
    load_parameter_catalog,
    scene_actions,
    scene_parameters,
)
from .models import (
    Failure,
    LLM3Plan,
    Observation,
    ValidationError,
    parse_llm3_plan,
)
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, response_schema, task_payload


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
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must not be empty")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.timeout_seconds <= 0
            or self.max_tokens <= 0
        ):
            raise ValueError("timeout and max_tokens must be positive")
        if (
            isinstance(self.max_actions, bool)
            or not isinstance(self.max_actions, int)
            or self.max_actions <= 0
        ):
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
        profile = load_model_profile(profile_name)
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
            not isinstance(markers, list)
            or len(markers) != 2
            or not all(isinstance(marker, str) and marker for marker in markers)
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
    plan: LLM3Plan
    model: str
    latency_ms: float
    prompt_version: int = PROMPT_VERSION
    token_usage: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_version": self.prompt_version,
            "token_usage": dict(self.token_usage),
            "execution_started": False,
        }


class LLM3Planner:
    def __init__(
        self,
        config: PlannerConfig,
        *,
        transport: CompletionTransport | None = None,
        catalog: Mapping[str, Any] | None = None,
        parameter_catalog: Mapping[str, Any] | None = None,
    ):
        self.config = config
        self.transport = OpenAITransport(config) if transport is None else transport
        self.catalog = load_catalog() if catalog is None else catalog
        self.parameter_catalog = (
            load_parameter_catalog()
            if parameter_catalog is None
            else parameter_catalog
        )

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
        if observation.goal_satisfied:
            return PlanResult(
                LLM3Plan(
                    "GOAL_COMPLETE",
                    (),
                    "The independent goal verifier is satisfied.",
                ),
                self.config.model,
                0.0,
            )
        normalized_images = validate_images(images)
        actions = scene_actions(self.catalog, observation.scene)
        parameters = scene_parameters(self.parameter_catalog, observation.scene)
        if set(actions) != set(parameters):
            raise PlanningError(
                "LLM3 parameter catalogue does not match the action catalogue"
            )
        payload = self._completion_payload(
            normalized_goal,
            observation,
            normalized_images,
            history,
            failure,
            actions,
            parameters,
        )
        started = time.perf_counter()
        response = self.transport.complete(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = response_content(response, self.config.reasoning_markers)
        try:
            plan = parse_llm3_plan(
                content,
                actions,
                parameters,
                observation,
                max_actions=self.config.max_actions,
            )
        except ValidationError as error:
            raise PlanningError(f"Invalid model plan: {error}") from error
        usage = response.get("usage", {})
        token_usage = {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        } if isinstance(usage, Mapping) else {}
        return PlanResult(
            plan, self.config.model, latency_ms, token_usage=token_usage
        )

    def _completion_payload(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        history: Sequence[Mapping[str, Any]],
        failure: Failure | None,
        actions: Mapping[str, Mapping[str, Any]],
        parameters: Mapping[str, Mapping[str, Mapping[str, float]]],
    ) -> dict[str, object]:
        schema = response_schema(
            actions, parameters, self.config.max_actions
        )
        task = task_payload(
            goal, observation, actions, parameters, history, failure
        )
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
