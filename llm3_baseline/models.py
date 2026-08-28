"""LLM3 full-plan and continuous-parameter data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from baseline_common.models import (
    Action,
    ActionResult,
    Entity,
    Failure,
    Observation,
    PLAN_STATUSES,
    Plan,
    Region,
    ValidationError,
    parse_plan,
)


@dataclass(frozen=True)
class LLM3Plan:
    status: str
    actions: tuple[Action, ...]
    reasoning: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasoning": self.reasoning,
            "actions": [item.as_dict() for item in self.actions],
        }


def parse_llm3_plan(
    value: object,
    actions: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Mapping[str, Mapping[str, float]]],
    observation: Observation,
    *,
    max_actions: int,
) -> LLM3Plan:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "reasoning",
        "actions",
    }:
        raise ValidationError(
            "LLM3 plan must contain only status, reasoning, and actions"
        )
    reasoning = value["reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValidationError("LLM3 reasoning must be a non-empty string")
    raw_actions = value["actions"]
    if not isinstance(raw_actions, list):
        raise ValidationError("LLM3 actions must be an array")
    if value["status"] not in {"PLAN", "NO_VALID_PLAN"}:
        raise ValidationError(
            "LLM3 model status must be PLAN or NO_VALID_PLAN; goal completion "
            "is decided by the independent verifier"
        )
    discrete_rows = []
    parameter_rows = []
    for row in raw_actions:
        if not isinstance(row, Mapping) or set(row) != {
            "skill",
            "arguments",
            "parameters",
        }:
            raise ValidationError(
                "Each LLM3 action requires skill, arguments, and parameters"
            )
        discrete_rows.append(
            {"skill": row["skill"], "arguments": row["arguments"]}
        )
        parameter_rows.append(row["parameters"])
    discrete = parse_plan(
        {"status": value["status"], "actions": discrete_rows},
        actions,
        observation,
        max_actions=max_actions,
    )
    parsed = []
    for action, raw in zip(discrete.actions, parameter_rows):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"{action.skill}.parameters must be an object")
        specification = parameters.get(action.skill)
        if specification is None:
            raise ValidationError(
                f"No continuous parameter specification for {action.skill}"
            )
        if set(raw) != set(specification):
            raise ValidationError(
                f"{action.skill} requires continuous parameters "
                f"{sorted(specification)}"
            )
        values: dict[str, float] = {}
        for name, bounds in specification.items():
            value_at_name = raw[name]
            if isinstance(value_at_name, bool) or not isinstance(
                value_at_name, (int, float)
            ):
                raise ValidationError(
                    f"{action.skill}.{name} must be a finite number"
                )
            number = float(value_at_name)
            if not float("-inf") < number < float("inf"):
                raise ValidationError(
                    f"{action.skill}.{name} must be a finite number"
                )
            lower = float(bounds["minimum"])
            upper = float(bounds["maximum"])
            if not lower <= number <= upper:
                raise ValidationError(
                    f"{action.skill}.{name}={number} is outside [{lower}, {upper}]"
                )
            values[name] = number
        parsed.append(Action(action.skill, action.arguments, values))
    return LLM3Plan(discrete.status, tuple(parsed), reasoning.strip())

__all__ = [
    "Action",
    "ActionResult",
    "Entity",
    "Failure",
    "Observation",
    "PLAN_STATUSES",
    "Plan",
    "LLM3Plan",
    "Region",
    "ValidationError",
    "parse_plan",
    "parse_llm3_plan",
]
