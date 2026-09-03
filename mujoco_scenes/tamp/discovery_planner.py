"""OpenAI-compatible VLM planner for discovery-based MuJoCo execution."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baseline_common.inference import OpenAITransport, PlanningError, response_content

from .discovery_replanning import (
    PlanStatus,
    PlannerRequest,
    PlannerResult,
    RecoverablePlanningError,
)
from .skills import SkillAction


DEFAULT_ACTION_CATALOG = Path(__file__).resolve().parents[2] / "baseline_common" / "action_catalog.json"


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    base_url: str
    model: str
    scene: str
    api_key: str = ""
    timeout_seconds: float = 600.0
    max_tokens: int = 4096
    temperature: float = 0.0
    seed: int = 0
    enable_thinking: bool | None = None
    trace_dir: str | Path | None = None


class OpenAIDiscoveryPlanner:
    """Ask a remote VLM for a direct plan over the current visible state.

    This intentionally exposes neither hidden object inventories nor MuJoCo
    backend names.  The action catalogue includes INSPECT, but the prompt does
    not tell the model to inspect before planning.
    """

    def __init__(
        self, config: OpenAIPlannerConfig, *, action_catalog: str | Path = DEFAULT_ACTION_CATALOG
    ):
        if not config.base_url.strip() or not config.model.strip() or not config.scene.strip():
            raise ValueError("base_url, model, and scene must be non-empty")
        if config.timeout_seconds <= 0 or config.max_tokens <= 0:
            raise ValueError("timeout_seconds and max_tokens must be positive")
        if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.config = config
        document = json.loads(Path(action_catalog).read_text(encoding="utf-8"))
        try:
            self.actions = document["scenes"][config.scene]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Action catalogue has no scene {config.scene!r}") from error
        if not isinstance(self.actions, Mapping):
            raise ValueError("Action catalogue scene must be an object")
        self.transport = OpenAITransport(config)
        self.trace_dir = Path(config.trace_dir).resolve() if config.trace_dir else None
        self.call_count = 0

    def plan(self, request: PlannerRequest) -> PlannerResult:
        if request.scene != self.config.scene:
            raise ValueError(
                f"Planner configured for {self.config.scene!r}, got {request.scene!r}"
            )
        payload = self._payload(request)
        self.call_count += 1
        started = time.monotonic()
        try:
            response = self.transport.complete(payload)
            content = response_content(response)
            elapsed = time.monotonic() - started
            result = self._parse_result(content, request=request, latency_s=elapsed)
            self._write_trace(request, content, latency_s=elapsed)
            return result
        except PlanningError as error:
            self._write_trace(
                request,
                None,
                latency_s=time.monotonic() - started,
                error=str(error),
            )
            raise RecoverablePlanningError(str(error)) from error

    def _write_trace(
        self,
        request: PlannerRequest,
        response: Mapping[str, Any] | None,
        *,
        latency_s: float,
        error: str = "",
    ) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "call": self.call_count,
            "model": self.config.model,
            "system_prompt": self._system_prompt(),
            "user_prompt": self._user_prompt(request),
            "camera_names": [image.camera for image in request.snapshot.images],
            "response": response,
            "latency_s": latency_s,
            "error": error or None,
            "private_goal_evaluator_exposed": False,
        }
        path = self.trace_dir / f"call_{self.call_count:03d}.json"
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _payload(self, request: PlannerRequest) -> dict[str, object]:
        user_text = self._user_prompt(request)
        content: list[dict[str, object]] = [{"type": "text", "text": user_text}]
        for image in request.snapshot.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image.data_url},
                }
            )
        payload: dict[str, object] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.config.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.config.enable_thinking
            }
        return payload

    def _system_prompt(self) -> str:
        return (
            "You are a robot task planner. Produce a task-level action sequence from "
            "the current observation and goal. Use only visible object IDs, known region "
            "IDs, the robot's held_object when present, and the action schema. Do not infer "
            "hidden objects or simulator "
            "state. Do not output trajectories, PDDL, coordinates, explanations, or markdown."
        )

    def _user_prompt(self, request: PlannerRequest) -> str:
        state = _visible_state(request.snapshot.state.as_dict())
        event = request.replan_event.as_dict() if request.replan_event else None
        contract = {
            name: {
                "description": str(spec.get("description", "")),
                "arguments": dict(spec.get("arguments", {})),
            }
            for name, spec in self.actions.items()
            if isinstance(spec, Mapping)
        }
        body = {
            "goal": request.goal,
            "observation": state,
            "completed_actions": [_action_dict(action) for action in request.completed_actions],
            "replanning_event": event,
            "available_actions": contract,
            "output_schema": {
                "status": "PLAN | GOAL_COMPLETE | NO_VALID_PLAN",
                "actions": [
                    {"name": "ACTION_NAME", "arguments": {"argument_name": "observed_id"}}
                ],
            },
        }
        return json.dumps(body, separators=(",", ":"), ensure_ascii=False)

    def _parse_result(
        self,
        payload: Mapping[str, Any],
        *,
        request: PlannerRequest,
        latency_s: float,
    ) -> PlannerResult:
        if set(payload) != {"status", "actions"}:
            raise PlanningError("Planner output must contain exactly status and actions")
        status_text = payload.get("status")
        try:
            status = PlanStatus(str(status_text))
        except ValueError as error:
            raise PlanningError(f"Unsupported planner status {status_text!r}") from error
        rows = payload.get("actions")
        if not isinstance(rows, list):
            raise PlanningError("Planner actions must be an array")
        if status is not PlanStatus.PLAN:
            if rows:
                raise PlanningError(f"{status.value} must not include actions")
            return PlannerResult(status, latency_s=latency_s)
        if not rows:
            raise PlanningError("PLAN must contain at least one action")
        actions = self._parse_actions(rows, request.snapshot.state.as_dict())
        return PlannerResult(status, actions, raw_output=json.dumps(payload), latency_s=latency_s)

    def _parse_actions(
        self, rows: list[object], raw_state: Mapping[str, object]
    ) -> tuple[SkillAction, ...]:
        robot = raw_state.get("robot", {})
        held_object = robot.get("held_object") if isinstance(robot, Mapping) else None
        holding = held_object if isinstance(held_object, str) else None
        actions: list[SkillAction] = []
        for row in rows:
            action = self._parse_action(row, raw_state)
            self._validate_action_sequence(action, holding)
            if action.name == "PICK":
                holding = str(action.arguments["object_id"])
            elif action.name == "PLACE":
                holding = None
            actions.append(action)
        return tuple(actions)

    @staticmethod
    def _validate_action_sequence(action: SkillAction, holding: str | None) -> None:
        values = action.arguments
        if action.name == "PICK":
            if holding is not None:
                raise PlanningError(
                    f"PICK requires an empty gripper, but {holding!r} is held"
                )
            return
        if action.name == "PLACE":
            object_id = str(values["object_id"])
            if holding != object_id:
                raise PlanningError(
                    f"PLACE requires holding {object_id!r}, but holding {holding!r}"
                )
            return
        if action.name == "POUR":
            source_id = str(values["source_id"])
            target_id = str(values["target_id"])
            if source_id == target_id:
                raise PlanningError("POUR source_id and target_id must differ")
            if holding != source_id:
                raise PlanningError(
                    f"POUR requires holding {source_id!r}, but holding {holding!r}"
                )
            return
        if action.name == "STIR":
            tool_id = str(values["tool_id"])
            target_id = str(values["target_id"])
            if tool_id == target_id:
                raise PlanningError("STIR tool_id and target_id must differ")
            if holding != tool_id:
                raise PlanningError(
                    f"STIR requires holding {tool_id!r}, but holding {holding!r}"
                )

    def _parse_action(self, row: object, raw_state: Mapping[str, object]) -> SkillAction:
        if not isinstance(row, Mapping) or set(row) != {"name", "arguments"}:
            raise PlanningError("Each action must contain exactly name and arguments")
        name = str(row.get("name") or "").upper()
        spec = self.actions.get(name)
        if not isinstance(spec, Mapping):
            raise PlanningError(f"Unsupported action {name!r}")
        arguments = row.get("arguments")
        if not isinstance(arguments, Mapping):
            raise PlanningError(f"{name} arguments must be an object")
        expected = spec.get("arguments", {})
        if not isinstance(expected, Mapping) or set(arguments) != set(expected):
            raise PlanningError(f"{name} arguments must be exactly {sorted(expected)}")
        normalized: dict[str, str] = {}
        for key, value in arguments.items():
            if not isinstance(value, str) or not value.strip():
                raise PlanningError(f"{name}.{key} must be a non-empty string")
            normalized[str(key)] = value.strip()
        self._validate_visible_references(name, normalized, raw_state)
        return SkillAction(name, normalized)

    @staticmethod
    def _validate_visible_references(
        name: str,
        arguments: Mapping[str, str],
        raw_state: Mapping[str, object],
    ) -> None:
        objects = raw_state.get("objects", {})
        regions = raw_state.get("regions", {})
        visible_objects = {
            object_id
            for object_id, item in objects.items()
            if isinstance(item, Mapping) and item.get("visible") is True
        } if isinstance(objects, Mapping) else set()
        visible_regions = {
            region_id
            for region_id, item in regions.items()
            if isinstance(item, Mapping) and item.get("visible") is True
        } if isinstance(regions, Mapping) else set()
        robot = raw_state.get("robot", {})
        held_object = robot.get("held_object") if isinstance(robot, Mapping) else None
        if isinstance(held_object, str):
            visible_objects.add(held_object)
        for key, value in arguments.items():
            if key in {"object_id", "source_id", "target_id", "tool_id"} and value not in visible_objects:
                raise PlanningError(
                    f"{name}.{key} references object {value!r}, which is not visible"
                )
            if key == "region_id" and value not in visible_objects | visible_regions:
                raise PlanningError(
                    f"{name}.region_id references unknown visible destination {value!r}"
                )


def _action_dict(action: SkillAction) -> dict[str, object]:
    return {"name": action.name, "arguments": dict(action.arguments)}


def _visible_state(raw: Mapping[str, object]) -> dict[str, object]:
    """Strip invisible objects before any state reaches the model."""
    objects = raw.get("objects", {})
    visible_objects = {
        object_id: row
        for object_id, row in objects.items()
        if isinstance(row, Mapping) and row.get("visible") is True
    } if isinstance(objects, Mapping) else {}
    regions = raw.get("regions", {})
    visible_regions = {
        region_id: row
        for region_id, row in regions.items()
        if isinstance(row, Mapping) and row.get("visible") is True
    } if isinstance(regions, Mapping) else {}
    visible_ids = set(visible_objects) | set(visible_regions)
    relations = [
        relation
        for relation in raw.get("relations", []) if isinstance(relation, Mapping)
        and relation.get("subject") in visible_ids
        and relation.get("object") in visible_ids
    ] if isinstance(raw.get("relations", []), list) else []
    return {
        "revision": raw.get("revision", 0),
        "robot": raw.get("robot", {}),
        "visible_objects": visible_objects,
        "known_regions": visible_regions,
        "relations": relations,
    }
