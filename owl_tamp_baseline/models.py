"""Strict model-facing and result contracts for OWL-TAMP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ValidationError(ValueError):
    pass


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Action:
    operator: str
    arguments: tuple[str, ...]

    @classmethod
    def parse(cls, value: object) -> "Action":
        if not isinstance(value, Mapping) or set(value) != {"operator", "arguments"}:
            raise ValidationError("action requires only operator and arguments")
        operator = _text(value["operator"], "action.operator").upper()
        arguments = value["arguments"]
        if not isinstance(arguments, list):
            raise ValidationError("action.arguments must be an array")
        return cls(operator, tuple(_text(item, "action.argument") for item in arguments))

    def as_dict(self) -> dict[str, Any]:
        return {"operator": self.operator, "arguments": list(self.arguments)}


@dataclass(frozen=True)
class PlanSketch:
    status: str
    actions: tuple[Action, ...]
    goal_literals: tuple[str, ...]

    @classmethod
    def parse(cls, value: object, *, max_actions: int = 64) -> "PlanSketch":
        if not isinstance(value, Mapping) or set(value) != {
            "status", "actions", "goal_literals"
        }:
            raise ValidationError(
                "plan sketch requires only status, actions, and goal_literals"
            )
        status = _text(value["status"], "status").upper()
        if status not in {"PLAN", "NO_PLAN"}:
            raise ValidationError("status must be PLAN or NO_PLAN")
        rows = value["actions"]
        goals = value["goal_literals"]
        if not isinstance(rows, list) or not isinstance(goals, list):
            raise ValidationError("actions and goal_literals must be arrays")
        if len(rows) > max_actions:
            raise ValidationError(f"plan sketch exceeds {max_actions} actions")
        actions = tuple(Action.parse(row) for row in rows)
        literals = tuple(_text(row, "goal_literal") for row in goals)
        if status == "PLAN" and not actions:
            raise ValidationError("PLAN requires at least one action")
        if status == "NO_PLAN" and (actions or literals):
            raise ValidationError("NO_PLAN must not contain actions or goals")
        return cls(status, actions, literals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "actions": [action.as_dict() for action in self.actions],
            "goal_literals": list(self.goal_literals),
        }


@dataclass(frozen=True)
class Constraint:
    action_index: int
    description: str
    expression: str

    @classmethod
    def parse_many(cls, value: object, action_count: int) -> tuple["Constraint", ...]:
        if not isinstance(value, Mapping) or set(value) != {"constraints"}:
            raise ValidationError("constraint response requires only constraints")
        rows = value["constraints"]
        if not isinstance(rows, list):
            raise ValidationError("constraints must be an array")
        result = []
        seen = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {
                "action_index", "description", "expression"
            }:
                raise ValidationError("invalid constraint row")
            index = row["action_index"]
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValidationError("constraint action_index must be an integer")
            if index < 0 or index >= action_count or index in seen:
                raise ValidationError("constraint action_index is invalid or repeated")
            seen.add(index)
            result.append(
                cls(index, _text(row["description"], "description"), _text(row["expression"], "expression"))
            )
        return tuple(result)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "description": self.description,
            "expression": self.expression,
        }


@dataclass(frozen=True)
class PlanningResult:
    status: str
    sketch: PlanSketch
    actions: tuple[Action, ...]
    constraints: tuple[Constraint, ...]
    skeletons_tested: int
    samples_tested: int
    failure: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sketch": self.sketch.as_dict(),
            "actions": [action.as_dict() for action in self.actions],
            "constraints": [item.as_dict() for item in self.constraints],
            "skeletons_tested": self.skeletons_tested,
            "samples_tested": self.samples_tested,
            "failure": self.failure,
        }
