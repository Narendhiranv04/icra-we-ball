"""Validated VLM-TAMP subgoal and failure contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baseline_common.models import Observation, ValidationError


PLAN_STATUSES = {"SUBGOALS", "GOAL_COMPLETE", "NO_VALID_SUBGOALS"}
ENGLISH_PLAN_STATUSES = {"STEPS", "GOAL_COMPLETE", "NO_VALID_STEPS"}


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ObjectReference:
    object_id: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.object_id}


@dataclass(frozen=True)
class ObjectUniverse:
    objects: tuple[ObjectReference, ...]
    privileged: bool = False

    @classmethod
    def observed(cls, observation: Observation) -> "ObjectUniverse":
        return cls(
            tuple(ObjectReference(item.entity_id) for item in observation.entities),
            False,
        )

    @classmethod
    def from_dict(cls, value: object) -> "ObjectUniverse":
        if not isinstance(value, Mapping) or set(value) != {"objects"}:
            raise ValidationError("Object universe must contain only objects")
        rows = value["objects"]
        if not isinstance(rows, list):
            raise ValidationError("Object universe objects must be an array")
        objects = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"id"}:
                raise ValidationError("Universe objects require only an id")
            objects.append(ObjectReference(_text(row["id"], "object.id")))
        ids = [item.object_id for item in objects]
        if len(ids) != len(set(ids)):
            raise ValidationError("Object universe IDs must be unique")
        return cls(tuple(objects), True)

    @property
    def object_ids(self) -> frozenset[str]:
        return frozenset(item.object_id for item in self.objects)

    def as_dict(self) -> dict[str, Any]:
        return {
            "objects": [item.as_dict() for item in self.objects],
            "uses_privileged_object_universe": self.privileged,
        }


@dataclass(frozen=True)
class Subgoal:
    predicate: str
    arguments: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class SubgoalPlan:
    status: str
    subgoals: tuple[Subgoal, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "subgoals": [item.as_dict() for item in self.subgoals],
        }


@dataclass(frozen=True)
class EnglishPlan:
    """First-stage, ungrounded intermediate goals from VLM-TAMP."""

    status: str
    steps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "steps": list(self.steps)}


def parse_english_plan(value: object, *, max_steps: int) -> EnglishPlan:
    if not isinstance(value, Mapping) or set(value) != {"status", "steps"}:
        raise ValidationError("English plan must contain only status and steps")
    status = _text(value["status"], "english_plan.status")
    if status not in ENGLISH_PLAN_STATUSES:
        raise ValidationError(f"Unsupported English plan status {status!r}")
    rows = value["steps"]
    if not isinstance(rows, list):
        raise ValidationError("english_plan.steps must be an array")
    if len(rows) > max_steps:
        raise ValidationError(f"English plan exceeds {max_steps} steps")
    steps = tuple(_text(row, "english_plan.step") for row in rows)
    if status == "STEPS" and not steps:
        raise ValidationError("STEPS requires at least one intermediate goal")
    if status != "STEPS" and steps:
        raise ValidationError(f"{status} must not contain steps")
    return EnglishPlan(status, steps)


@dataclass(frozen=True)
class RefinementFailure:
    code: str
    message: str
    subgoal: Subgoal | None = None
    collided_objects: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.subgoal is not None:
            result["subgoal"] = self.subgoal.as_dict()
        if self.collided_objects:
            result["collided_objects"] = list(self.collided_objects)
        return result


def parse_subgoal_plan(
    value: object,
    predicates: Mapping[str, Mapping[str, Any]],
    observation: Observation,
    universe: ObjectUniverse,
    *,
    max_subgoals: int,
) -> SubgoalPlan:
    if not isinstance(value, Mapping) or set(value) != {"status", "subgoals"}:
        raise ValidationError("Plan must contain only status and subgoals")
    status = _text(value["status"], "plan.status")
    if status not in PLAN_STATUSES:
        raise ValidationError(f"Unsupported plan status {status!r}")
    rows = value["subgoals"]
    if not isinstance(rows, list):
        raise ValidationError("plan.subgoals must be an array")
    if len(rows) > max_subgoals:
        raise ValidationError(f"Plan exceeds {max_subgoals} subgoals")
    parsed = tuple(
        _parse_subgoal(row, predicates, observation, universe) for row in rows
    )
    if status == "SUBGOALS" and not parsed:
        raise ValidationError("SUBGOALS requires at least one subgoal")
    if status != "SUBGOALS" and parsed:
        raise ValidationError(f"{status} must not contain subgoals")
    return SubgoalPlan(status, parsed)


def _parse_subgoal(
    value: object,
    predicates: Mapping[str, Mapping[str, Any]],
    observation: Observation,
    universe: ObjectUniverse,
) -> Subgoal:
    if not isinstance(value, Mapping) or set(value) != {"predicate", "arguments"}:
        raise ValidationError("Subgoal requires predicate and arguments")
    predicate = _text(value["predicate"], "subgoal.predicate")
    if predicate not in predicates:
        raise ValidationError(f"Unknown subgoal predicate {predicate!r}")
    arguments = value["arguments"]
    if not isinstance(arguments, Mapping):
        raise ValidationError("subgoal.arguments must be an object")
    expected = predicates[predicate]["arguments"]
    if set(arguments) != set(expected):
        raise ValidationError(
            f"Subgoal {predicate} requires arguments {sorted(expected)}"
        )
    parsed: dict[str, str] = {}
    for name, kind in expected.items():
        reference = _text(arguments[name], f"subgoal.arguments.{name}")
        if kind == "object":
            allowed = universe.object_ids
        elif kind == "region":
            allowed = observation.region_ids
        elif kind == "destination":
            allowed = universe.object_ids | observation.region_ids
        else:
            raise ValidationError(
                f"Subgoal {predicate} has unsupported reference kind {kind!r}"
            )
        if reference not in allowed:
            raise ValidationError(
                f"Subgoal {predicate} references unknown {kind} {reference!r}"
            )
        parsed[name] = reference
    return Subgoal(predicate, parsed)
