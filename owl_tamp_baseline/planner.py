"""OpenAI-compatible two-stage OWL-TAMP planner."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from baseline_common.inference import (
    QWEN_THINKING_SAMPLING,
    OpenAITransport,
    PlanningError,
    load_model_profile,
    response_content,
    validate_images,
)
from baseline_common.models import Observation

from .constraints import validate_constraints
from .domain import executed_encoding, relaxed_ground, validate_sketch
from .models import Constraint, PlanSketch, PlanningResult, ValidationError
from .prompt import (
    constraint_prompt,
    constraint_response_schema,
    discrete_prompt,
    discrete_response_schema,
)
from .refinement import SampleOracle, search_then_sample


SINGLE_CALL_MAX_TOKENS = 2048


def registry_sampling(profile_name: str, enable_thinking: bool) -> dict[str, object]:
    """Published sampling for a checkpoint in thinking or non-thinking mode."""
    planner_profile = load_model_profile(profile_name).get("planner", {})
    return dict(
        planner_profile.get("sampling", {}).get(
            "thinking" if enable_thinking else "direct", QWEN_THINKING_SAMPLING
        )
    )


def protocol_max_tokens(requested: int, protocol: str) -> int:
    """Reserve context for the large relaxed-grounding prompt."""
    if protocol not in {"native", "single_call"}:
        raise ValueError("protocol must be 'native' or 'single_call'")
    return min(requested, SINGLE_CALL_MAX_TOKENS) if protocol == "single_call" else requested


class CompletionTransport(Protocol):
    def complete(self, payload: Mapping[str, object]) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class OWLTAMPPlannerConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen35-9b"
    api_key: str = ""
    timeout_seconds: float = 600.0
    max_tokens: int = 8192
    seed: int = 0
    sampling: Mapping[str, object] = field(
        default_factory=lambda: {"temperature": 0.2, "top_p": 1.0}
    )
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not self.base_url.strip() or not self.model.strip():
            raise ValueError("base_url and model must not be empty")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError("timeout_seconds and max_tokens must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @classmethod
    def from_env(cls) -> "OWLTAMPPlannerConfig":
        profile_name = os.environ.get("OWL_TAMP_PROFILE", "qwen35-9b")
        profile = load_model_profile(profile_name)
        enable_thinking = os.environ.get(
            "OWL_TAMP_ENABLE_THINKING", "false"
        ).lower() in {"true", "1", "yes"}
        # Resolve decoding from the model registry, the way VLM-TAMP does.
        # This loader previously read only served_name and left `sampling` at
        # the {temperature 0.2, top_p 1.0} placeholder, which happened to match
        # what the VLM-TAMP runner's default "paper" decoding overrides to, so
        # the two agreed only by coincidence.  Reading the registry makes the
        # agreement hold under --decoding model-native too.
        sampling = registry_sampling(profile_name, enable_thinking)
        return cls(
            base_url=os.environ.get("OWL_TAMP_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("OWL_TAMP_MODEL", str(profile.get("served_name", profile_name))),
            api_key=os.environ.get("OWL_TAMP_API_KEY", ""),
            timeout_seconds=float(os.environ.get("OWL_TAMP_TIMEOUT_SECONDS", "600")),
            max_tokens=int(os.environ.get("OWL_TAMP_MAX_TOKENS", "8192")),
            seed=int(os.environ.get("OWL_TAMP_SEED", "0")),
            enable_thinking=enable_thinking,
            sampling=sampling,
        )


class OWLTAMPPlanner:
    def __init__(
        self,
        config: OWLTAMPPlannerConfig,
        *,
        transport: CompletionTransport | None = None,
    ):
        self.config = config
        self.transport = OpenAITransport(config) if transport is None else transport
        self.trace: dict[str, Any] = {}
        self.response_trace: list[Mapping[str, object]] = []

    def plan(
        self,
        goal: str,
        observation: Observation,
        images: Sequence[Mapping[str, str]],
        oracle: SampleOracle,
        *,
        movable_object_ids: Sequence[str] | None = None,
        max_vlm_requests: int | None = None,
    ) -> PlanningResult:
        goal = goal.strip()
        if not goal or len(goal) > 4000:
            raise ValidationError("goal must contain between 1 and 4000 characters")
        normalized_images = validate_images(images)
        if (
            max_vlm_requests is not None
            and (
                isinstance(max_vlm_requests, bool)
                or not isinstance(max_vlm_requests, int)
                or max_vlm_requests <= 0
            )
        ):
            raise ValueError("max_vlm_requests must be a positive integer or None")
        self.response_trace = []
        inspectable = (
            region.region_id
            for region in observation.regions
            if region.state == "closed"
        )
        grounded = relaxed_ground(
            observation.scene,
            observation.object_ids
            if movable_object_ids is None
            else movable_object_ids,
            observation.region_ids,
            inspectable,
        )
        started = time.perf_counter()
        sketch_prompt = discrete_prompt(
            observation.scene, goal, observation, grounded
        )
        sketch_response = self.transport.complete(
            self._payload(
                sketch_prompt,
                normalized_images,
                schema=discrete_response_schema(),
                schema_name="owl_tamp_discrete_sketch",
            )
        )
        self.response_trace.append(sketch_response)
        try:
            sketch = PlanSketch.parse(response_content(sketch_response), max_actions=64)
            validate_sketch(observation.scene, sketch.actions, grounded)
        except (PlanningError, ValidationError) as error:
            failure = f"Invalid OWL-TAMP discrete sketch: {error}"
            sketch = PlanSketch("NO_PLAN", (), ())
            result = PlanningResult(
                "INVALID_MODEL_OUTPUT", sketch, (), (), 0, 0, failure
            )
            self.trace = {
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "grounded_action_count": len(grounded),
                "executed_encoding": [],
                "sketch": sketch.as_dict(),
                "constraints": [],
                "result": result.as_dict(),
                "model_visible_state": observation.as_annotated_prompt_dict(),
                "camera_ids": [row["camera"] for row in normalized_images],
                "model_prompts": {
                    "discrete_sketch": sketch_prompt,
                    "continuous_constraints": [],
                },
                "model_responses": self.response_trace,
            }
            return result
        constraints: tuple[Constraint, ...] = ()
        constraint_prompts = []
        constraint_budget_available = (
            max_vlm_requests is None or max_vlm_requests > len(self.response_trace)
        )
        if sketch.status == "PLAN" and constraint_budget_available:
            generated = []
            for action_index in range(len(sketch.actions)):
                if (
                    max_vlm_requests is not None
                    and len(self.response_trace) >= max_vlm_requests
                ):
                    break
                generated_prompt = constraint_prompt(
                    observation.scene,
                    goal,
                    observation,
                    sketch,
                    action_index,
                )
                constraint_prompts.append(generated_prompt)
                constraint_response = self.transport.complete(
                    self._payload(
                        generated_prompt,
                        normalized_images,
                        schema=constraint_response_schema(action_index),
                        schema_name="owl_tamp_action_constraint",
                    )
                )
                self.response_trace.append(constraint_response)
                try:
                    rows = Constraint.parse_many(
                        response_content(constraint_response), len(sketch.actions)
                    )
                    if len(rows) != 1 or rows[0].action_index != action_index:
                        raise ValidationError(
                            f"constraint call must return only action index {action_index}"
                        )
                    generated.extend(rows)
                except (PlanningError, ValidationError) as error:
                    result = PlanningResult(
                        "INVALID_MODEL_OUTPUT", sketch, (), (), 0, 0,
                        f"Invalid OWL-TAMP constraint for action "
                        f"{action_index}: {error}",
                    )
                    self.trace = {
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000.0, 3
                        ),
                        "grounded_action_count": len(grounded),
                        "executed_encoding": list(executed_encoding(sketch.actions)),
                        "sketch": sketch.as_dict(),
                        "constraints": [row.as_dict() for row in generated],
                        "result": result.as_dict(),
                        "model_visible_state": observation.as_annotated_prompt_dict(),
                        "camera_ids": [row["camera"] for row in normalized_images],
                        "model_prompts": {
                            "discrete_sketch": sketch_prompt,
                            "continuous_constraints": constraint_prompts,
                        },
                        "model_responses": self.response_trace,
                    }
                    return result
            try:
                constraints = validate_constraints(generated)
            except ValidationError as error:
                result = PlanningResult(
                    "INVALID_MODEL_OUTPUT", sketch, (), (), 0, 0,
                    f"Invalid OWL-TAMP constraints: {error}",
                )
                self.trace = {
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000.0, 3
                    ),
                    "grounded_action_count": len(grounded),
                    "executed_encoding": list(executed_encoding(sketch.actions)),
                    "sketch": sketch.as_dict(),
                    "constraints": [row.as_dict() for row in generated],
                    "result": result.as_dict(),
                    "model_visible_state": observation.as_annotated_prompt_dict(),
                    "camera_ids": [row["camera"] for row in normalized_images],
                    "model_prompts": {
                        "discrete_sketch": sketch_prompt,
                        "continuous_constraints": constraint_prompts,
                    },
                    "model_responses": self.response_trace,
                }
                return result
        result = search_then_sample(observation, grounded, sketch, constraints, oracle)
        self.trace = {
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "grounded_action_count": len(grounded),
            "executed_encoding": list(executed_encoding(sketch.actions)),
            "sketch": sketch.as_dict(),
            "constraints": [row.as_dict() for row in constraints],
            "result": result.as_dict(),
            "model_visible_state": observation.as_annotated_prompt_dict(),
            "camera_ids": [row["camera"] for row in normalized_images],
            "model_prompts": {
                "discrete_sketch": sketch_prompt,
                "continuous_constraints": constraint_prompts,
            },
            "model_responses": self.response_trace,
            "max_vlm_requests": max_vlm_requests,
            "constraint_generation_complete": (
                sketch.status != "PLAN"
                or len(constraints) == len(sketch.actions)
            ),
        }
        return result

    def _payload(
        self,
        prompt: str,
        images: Sequence[Mapping[str, str]],
        *,
        schema: Mapping[str, object],
        schema_name: str,
    ) -> dict[str, object]:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": row["data_url"]},
            }
            for row in images
        )
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            },
            "seed": self.config.seed,
            **dict(self.config.sampling),
        }
        if self.config.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload
