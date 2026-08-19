"""Validated data contracts for the LLM3-style baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


PLAN_STATUSES = {"PLAN", "GOAL_COMPLETE", "NO_VALID_PLAN"}


class ValidationError(ValueError):
    """Input or model output violates the observation-bounded contract."""


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be an object")
    return value


@dataclass(frozen=True)
class Entity:
    entity_id: str
    kind: str
    label: str
    facts: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: object) -> "Entity":
        raw = _mapping(value, "entity")
        if set(raw) - {"id", "kind", "label", "facts"}:
            raise ValidationError("Entity contains unsupported fields")
        kind = _string(raw.get("kind"), "entity.kind")
        if kind != "object":
            raise ValidationError("Visible entities must have kind 'object'")
        facts = _mapping(raw.get("facts", {}), "entity.facts")
        return cls(
            _string(raw.get("id"), "entity.id"),
            kind,
            _string(raw.get("label"), "entity.label"),
            dict(facts),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.entity_id,
            "kind": self.kind,
            "label": self.label,
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class Region:
    region_id: str
    label: str
    state: str = "unknown"
    inspected: bool = False

    @classmethod
    def from_dict(cls, value: object) -> "Region":
        raw = _mapping(value, "region")
        if set(raw) - {"id", "label", "state", "inspected"}:
            raise ValidationError("Region contains unsupported fields")
        state = _string(raw.get("state", "unknown"), "region.state")
        if state not in {"open", "closed", "unknown"}:
            raise ValidationError("region.state must be open, closed, or unknown")
        inspected = raw.get("inspected", False)
        if not isinstance(inspected, bool):
            raise ValidationError("region.inspected must be boolean")
        return cls(
            _string(raw.get("id"), "region.id"),
            _string(raw.get("label"), "region.label"),
            state,
            inspected,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.region_id,
            "label": self.label,
            "state": self.state,
            "inspected": self.inspected,
        }


@dataclass(frozen=True)
class Observation:
    scene: str
    revision: int
    entities: tuple[Entity, ...]
    regions: tuple[Region, ...]
    robot: Mapping[str, Any]
    goal_satisfied: bool = False

    @classmethod
    def from_dict(cls, value: object) -> "Observation":
        raw = _mapping(value, "observation")
        allowed = {
            "scene",
            "revision",
            "visible_entities",
            "known_regions",
            "robot",
            "goal_satisfied",
        }
        if set(raw) - allowed:
            raise ValidationError("Observation contains unsupported fields")
        revision = raw.get("revision", 0)
        if not isinstance(revision, int) or revision < 0:
            raise ValidationError("observation.revision must be non-negative")
        entities_raw = raw.get("visible_entities", [])
        regions_raw = raw.get("known_regions", [])
        if not isinstance(entities_raw, Sequence) or isinstance(entities_raw, str):
            raise ValidationError("visible_entities must be an array")
        if not isinstance(regions_raw, Sequence) or isinstance(regions_raw, str):
            raise ValidationError("known_regions must be an array")
        entities = tuple(Entity.from_dict(item) for item in entities_raw)
        regions = tuple(Region.from_dict(item) for item in regions_raw)
        entity_ids = [item.entity_id for item in entities]
        region_ids = [item.region_id for item in regions]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValidationError("Visible entity IDs must be unique")
        if len(region_ids) != len(set(region_ids)):
            raise ValidationError("Known region IDs must be unique")
        if set(entity_ids) & set(region_ids):
            raise ValidationError("Object and region IDs must not overlap")
        goal_satisfied = raw.get("goal_satisfied", False)
        if not isinstance(goal_satisfied, bool):
            raise ValidationError("goal_satisfied must be boolean")
        return cls(
            _string(raw.get("scene"), "observation.scene"),
            revision,
            entities,
            regions,
            dict(_mapping(raw.get("robot", {}), "observation.robot")),
            goal_satisfied,
        )

    @property
    def object_ids(self) -> frozenset[str]:
        return frozenset(item.entity_id for item in self.entities)

    @property
    def region_ids(self) -> frozenset[str]:
        return frozenset(item.region_id for item in self.regions)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "revision": self.revision,
            "visible_entities": [item.as_dict() for item in self.entities],
            "known_regions": [item.as_dict() for item in self.regions],
            "robot": dict(self.robot),
            "goal_satisfied": self.goal_satisfied,
        }


@dataclass(frozen=True)
class Action:
    skill: str
    arguments: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {"skill": self.skill, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class Plan:
    status: str
    actions: tuple[Action, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "actions": [action.as_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    action: Action | None = None
    action_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.action is not None:
            result["action"] = self.action.as_dict()
        if self.action_index is not None:
            result["action_index"] = self.action_index
        return result


@dataclass(frozen=True)
class ActionResult:
    success: bool
    failure_code: str | None = None
    message: str = ""
    recoverable: bool = True

    @classmethod
    def succeeded(cls) -> "ActionResult":
        return cls(True)

    @classmethod
    def failed(
        cls,
        code: str,
        message: str,
        *,
        recoverable: bool = True,
    ) -> "ActionResult":
        return cls(False, code, message, recoverable)


def parse_plan(
    value: object,
    actions: Mapping[str, Mapping[str, Any]],
    observation: Observation,
    *,
    max_actions: int,
) -> Plan:
    raw = _mapping(value, "plan")
    if set(raw) != {"status", "actions"}:
        raise ValidationError("Plan must contain only status and actions")
    status = _string(raw["status"], "plan.status")
    if status not in PLAN_STATUSES:
        raise ValidationError(f"Unsupported plan status {status!r}")
    action_values = raw["actions"]
    if not isinstance(action_values, list):
        raise ValidationError("plan.actions must be an array")
    if len(action_values) > max_actions:
        raise ValidationError(f"Plan exceeds {max_actions} actions")
    parsed = tuple(
        _parse_action(item, actions, observation) for item in action_values
    )
    if status == "PLAN" and not parsed:
        raise ValidationError("PLAN requires at least one action")
    if status != "PLAN" and parsed:
        raise ValidationError(f"{status} must not contain actions")
    return Plan(status, parsed)


def _parse_action(
    value: object,
    actions: Mapping[str, Mapping[str, Any]],
    observation: Observation,
) -> Action:
    raw = _mapping(value, "action")
    if set(raw) != {"skill", "arguments"}:
        raise ValidationError("Action must contain only skill and arguments")
    skill = _string(raw["skill"], "action.skill")
    if skill not in actions:
        raise ValidationError(f"Action {skill!r} is not available in this scene")
    arguments = _mapping(raw["arguments"], "action.arguments")
    expected = actions[skill]["arguments"]
    if set(arguments) != set(expected):
        raise ValidationError(
            f"Action {skill} requires arguments {sorted(expected)}"
        )
    parsed: dict[str, str] = {}
    for name, reference_kind in expected.items():
        reference = _string(arguments[name], f"action.arguments.{name}")
        allowed = (
            observation.object_ids
            if reference_kind == "object"
            else observation.region_ids
        )
        if reference not in allowed:
            raise ValidationError(
                f"Action {skill} references unobserved {reference_kind} "
                f"{reference!r}"
            )
        parsed[name] = reference
    return Action(skill, parsed)
