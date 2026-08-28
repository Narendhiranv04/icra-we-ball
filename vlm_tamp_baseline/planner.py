"""OpenAI-compatible VLM-TAMP subgoal proposer."""

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
from baseline_common.models import Observation, ValidationError

from .catalog import load_catalog, scene_subgoals
from .models import (
    EnglishPlan,
    ObjectUniverse,
    RefinementFailure,
    Subgoal,
    SubgoalPlan,
    parse_english_plan,
    parse_subgoal_plan,
)
from .prompt import (
    ENGLISH_SYSTEM_PROMPT,
    GROUNDING_SYSTEM_PROMPT,
    PROMPT_VERSION,
    english_response_schema,
    grounding_payload,
    grounding_response_schema,
    scene_payload,
)


class CompletionTransport(Protocol):
    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class VLMTAMPPlannerConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen35-9b"
    api_key: str = ""
    timeout_seconds: float = 600.0
    max_tokens: int = 24576
    max_subgoals: int = 20
    seed: int = 0
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
            isinstance(self.max_subgoals, bool)
            or not isinstance(self.max_subgoals, int)
            or self.max_subgoals <= 0
        ):
            raise ValueError("max_subgoals must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.sampling, Mapping):
            raise ValueError("sampling must be an object")

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "VLMTAMPPlannerConfig":
        values = os.environ if environ is None else environ
        profile_name = values.get(
            "VLM_TAMP_PROFILE", values.get("INFERENCE_MODEL", "qwen35-9b")
        ).strip()
        profile = load_model_profile(profile_name)
        planner = profile.get("planner", {})
        thinking_mode = str(planner.get("thinking_mode", "toggle"))
        raw_thinking = values.get(
            "VLM_TAMP_ENABLE_THINKING", "true"
        ).strip().lower()
        if raw_thinking not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("VLM_TAMP_ENABLE_THINKING must be true or false")
        enable_thinking = raw_thinking in {"true", "1", "yes"}
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
        markers = planner.get("reasoning_markers")
        if markers is not None and (
            not isinstance(markers, list)
            or len(markers) != 2
            or not all(isinstance(marker, str) and marker for marker in markers)
        ):
            raise ValueError(f"Invalid reasoning markers for {profile_name}")
        return cls(
            base_url=values.get(
                "VLM_TAMP_MODEL_BASE_URL",
                values.get("PLANNER_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
            ),
            model=values.get(
                "VLM_TAMP_MODEL", str(profile.get("served_name", profile_name))
            ),
            api_key=values.get(
                "VLM_TAMP_API_KEY", values.get("INFERENCE_API_KEY", "")
            ),
            timeout_seconds=float(values.get("VLM_TAMP_TIMEOUT_SECONDS", "600")),
            max_tokens=int(
                values.get(
                    "VLM_TAMP_MAX_TOKENS", str(planner.get("max_tokens", 24576))
                )
            ),
            max_subgoals=int(values.get("VLM_TAMP_MAX_SUBGOALS", "20")),
            seed=int(values.get("VLM_TAMP_SEED", "0")),
            enable_thinking=enable_thinking,
            sampling=sampling,
            structured_output=bool(planner.get("structured_output", True)),
            reasoning_markers=tuple(markers) if markers else None,
            system_prompt_prefix=str(planner.get("system_prompt_prefix", "")),
            toggle_thinking=thinking_mode == "toggle",
        )


@dataclass(frozen=True)
class SubgoalPlanResult:
    plan: SubgoalPlan
    english_plan: EnglishPlan
    model: str
    latency_ms: float
    uses_privileged_object_universe: bool
    prompt_version: int = PROMPT_VERSION
    token_usage: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "english_plan": self.english_plan.as_dict(),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_version": self.prompt_version,
            "token_usage": dict(self.token_usage),
            "uses_privileged_object_universe": self.uses_privileged_object_universe,
            "refinement_started": False,
            "execution_started": False,
        }


class VLMTAMPPlanner:
    def __init__(
        self,
        config: VLMTAMPPlannerConfig,
        *,
        transport: CompletionTransport | None = None,
        catalog: Mapping[str, Any] | None = None,
    ):
        self.config = config
        self.transport = OpenAITransport(config) if transport is None else transport
        self.catalog = load_catalog() if catalog is None else catalog
        self.request_trace: list[dict[str, Any]] = []
        self.response_trace: list[dict[str, Any]] = []

    @classmethod
    def from_env(cls) -> "VLMTAMPPlanner":
        return cls(VLMTAMPPlannerConfig.from_env())

    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        *,
        universe: ObjectUniverse | None = None,
        succeeded_subgoals: Sequence[Subgoal] = (),
        action_history: Sequence[Mapping[str, Any]] = (),
        failure: RefinementFailure | None = None,
    ) -> SubgoalPlanResult:
        self.request_trace = []
        self.response_trace = []
        normalized_goal = goal.strip()
        if not normalized_goal or len(normalized_goal) > 4000:
            raise ValidationError("goal must contain between 1 and 4000 characters")
        if observation.goal_satisfied:
            return SubgoalPlanResult(
                SubgoalPlan("GOAL_COMPLETE", ()),
                EnglishPlan("GOAL_COMPLETE", ()),
                self.config.model,
                0.0,
                bool(universe and universe.privileged),
            )
        normalized_images = validate_images(images)
        universe = universe or ObjectUniverse.observed(observation)
        if not observation.object_ids <= universe.object_ids:
            missing = sorted(observation.object_ids - universe.object_ids)
            raise ValidationError(
                "Object universe omits visible objects: " + ", ".join(missing)
            )
        predicates = scene_subgoals(self.catalog, observation.scene)
        english_payload = self._english_completion_payload(
            normalized_goal,
            observation,
            normalized_images,
            universe,
            predicates,
            succeeded_subgoals,
            action_history,
            failure,
        )
        self.request_trace.append(self._auditable_request(english_payload))
        started = time.perf_counter()
        response = self.transport.complete(english_payload)
        self.response_trace.append(self._auditable_request(response))
        english_usage = self._usage(response)
        english_content = response_content(response, self.config.reasoning_markers)
        try:
            english_plan = parse_english_plan(
                english_content, max_steps=self.config.max_subgoals
            )
        except ValidationError as error:
            raise PlanningError(f"Invalid VLM-TAMP English plan: {error}") from error
        if english_plan.status == "GOAL_COMPLETE":
            raise PlanningError(
                "VLM declared GOAL_COMPLETE although the independent goal "
                "verifier is false"
            )
        if english_plan.status != "STEPS":
            plan = SubgoalPlan("NO_VALID_SUBGOALS", ())
            latency_ms = (time.perf_counter() - started) * 1000.0
            return SubgoalPlanResult(
                plan,
                english_plan,
                self.config.model,
                latency_ms,
                universe.privileged,
                token_usage=english_usage,
            )

        grounding_request = self._grounding_completion_payload(
            english_plan, observation, normalized_images, universe, predicates
        )
        self.request_trace.append(self._auditable_request(grounding_request))
        response = self.transport.complete(grounding_request)
        self.response_trace.append(self._auditable_request(response))
        grounding_usage = self._usage(response)
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = response_content(response, self.config.reasoning_markers)
        try:
            plan = parse_subgoal_plan(
                content,
                predicates,
                observation,
                universe,
                max_subgoals=self.config.max_subgoals,
            )
        except ValidationError as error:
            raise PlanningError(f"Invalid VLM-TAMP subgoal plan: {error}") from error
        if plan.status == "GOAL_COMPLETE":
            raise PlanningError(
                "VLM grounding declared GOAL_COMPLETE although the independent "
                "goal verifier is false"
            )
        return SubgoalPlanResult(
            plan,
            english_plan,
            self.config.model,
            latency_ms,
            universe.privileged,
            token_usage=self._sum_usage(english_usage, grounding_usage),
        )

    @classmethod
    def _auditable_request(cls, value: object) -> Any:
        """Preserve exact prompt text/schema without duplicating image bytes."""
        if isinstance(value, Mapping):
            return {
                str(key): cls._auditable_request(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._auditable_request(item) for item in value]
        if isinstance(value, str) and value.startswith("data:image/"):
            return "<embedded image omitted; see saved camera PNG>"
        return value

    @staticmethod
    def _usage(response: Mapping[str, object]) -> dict[str, int]:
        usage = response.get("usage", {})
        if not isinstance(usage, Mapping):
            return {}
        return {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }

    @staticmethod
    def _sum_usage(*rows: Mapping[str, int]) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            for key, value in row.items():
                result[key] = result.get(key, 0) + int(value)
        return result

    def _english_completion_payload(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        universe: ObjectUniverse,
        predicates: Mapping[str, Mapping[str, Any]],
        succeeded_subgoals: Sequence[Subgoal],
        action_history: Sequence[Mapping[str, Any]],
        failure: RefinementFailure | None,
    ) -> dict[str, object]:
        schema = english_response_schema(self.config.max_subgoals)
        task = scene_payload(
            goal,
            observation,
            universe,
            succeeded_subgoals,
            action_history,
            failure,
        )
        if not self.config.structured_output:
            task["output_schema"] = schema
        content: list[dict[str, object]] = [
            {"type": "text", "text": json.dumps(task, separators=(",", ":"), sort_keys=True)}
        ]
        for image in images:
            content.extend(
                [
                    {"type": "text", "text": f"Camera view: {image['camera']}"},
                    {"type": "image_url", "image_url": {"url": image["data_url"]}},
                ]
            )
        system = ENGLISH_SYSTEM_PROMPT
        if self.config.system_prompt_prefix:
            system = f"{self.config.system_prompt_prefix}\n\n{system}"
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **dict(self.config.sampling),
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "seed": self.config.seed,
        }
        if self.config.toggle_thinking:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.config.enable_thinking
            }
        if self.config.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vlm_tamp_english_goals",
                    "strict": True,
                    "schema": schema,
                },
            }
        return payload

    def _grounding_completion_payload(
        self,
        english_plan: EnglishPlan,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        universe: ObjectUniverse,
        predicates: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, object]:
        schema = grounding_response_schema(
            predicates,
            self.config.max_subgoals,
            object_ids=sorted(universe.object_ids),
            region_ids=sorted(observation.region_ids),
        )
        task = grounding_payload(
            english_plan.steps, observation, universe, predicates
        )
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": json.dumps(task, separators=(",", ":"), sort_keys=True),
            }
        ]
        for image in images:
            content.extend(
                [
                    {"type": "text", "text": f"Camera view: {image['camera']}"},
                    {"type": "image_url", "image_url": {"url": image["data_url"]}},
                ]
            )
        system = GROUNDING_SYSTEM_PROMPT
        if self.config.system_prompt_prefix:
            system = f"{self.config.system_prompt_prefix}\n\n{system}"
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            **dict(self.config.sampling),
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "seed": self.config.seed,
        }
        if self.config.toggle_thinking:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.config.enable_thinking
            }
        if self.config.structured_output:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "vlm_tamp_grounded_subgoals",
                    "strict": True,
                    "schema": schema,
                },
            }
        return payload
