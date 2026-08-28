"""Prompt and structured-output schema for direct action planning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import Failure, Observation


PROMPT_VERSION = 4
SYSTEM_PROMPT = """\
You are the task-and-motion planner in the LLM3 baseline.
Return only the requested JSON object.

Rules:
- Use only the supplied action catalogue, visible object IDs, and known region
  IDs. Never invent an object, region, action, or argument.
- The five RGB views annotate persistent object and region IDs but provide no
  semantic class names. Infer object meaning from pixels. The accompanying
  textualized state contains identity and observable relations only.
- Do not assume unobserved objects, hidden contents, semantic labels, or
  functional roles.
- Produce a full action plan from the current state to the task goal. Every
  action includes both grounded discrete arguments and the complete continuous
  parameter dictionary specified for that skill.
- The reasoning field must briefly diagnose the previous motion failure, when
  present, and state whether new continuous parameters or symbolic backtracking
  are needed. It is not an execution result.
- Action arguments must satisfy the reference kinds in the supplied catalogue.
- Motion planning checks geometry, IK, collisions, action preconditions, and
  goal completion. Use its feedback to resample parameters or backtrack.
- When failure feedback is supplied, revise the remaining plan using only that
  feedback and the latest observation. Do not repeat a failed action unchanged
  unless the new observation provides a concrete reason it can now succeed.
- Goal completion is decided by an independent verifier, not by this model.
  Return NO_VALID_PLAN when no catalogue action can make progress from the
  supplied observation.
- Do not include markdown or extra fields.
"""


def response_schema(
    actions: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Mapping[str, Mapping[str, float]]],
    max_actions: int,
) -> dict[str, Any]:
    action_variants = []
    for skill, action in sorted(actions.items()):
        parameter_properties = {
            name: {
                "type": "number",
                "minimum": float(bounds["minimum"]),
                "maximum": float(bounds["maximum"]),
            }
            for name, bounds in parameters[skill].items()
        }
        action_variants.append(
            {
                "type": "object",
                "properties": {
                    "skill": {"const": skill},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            name: {"type": "string"}
                            for name in action["arguments"]
                        },
                        "required": list(action["arguments"]),
                        "additionalProperties": False,
                    },
                    "parameters": {
                        "type": "object",
                        "properties": parameter_properties,
                        "required": list(parameter_properties),
                        "additionalProperties": False,
                    },
                },
                "required": ["skill", "arguments", "parameters"],
                "additionalProperties": False,
            }
        )
    common = {
        "reasoning": {"type": "string", "minLength": 1},
    }

    def variant(status: str, minimum: int, maximum: int) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"const": status},
                **common,
                "actions": {
                    "type": "array",
                    "minItems": minimum,
                    "maxItems": maximum,
                    "items": {"oneOf": action_variants},
                },
            },
            "required": ["status", "reasoning", "actions"],
            "additionalProperties": False,
        }

    return {
        "oneOf": [
            variant("PLAN", 1, max_actions),
            variant("NO_VALID_PLAN", 0, 0),
        ]
    }


def task_payload(
    goal: str,
    observation: Observation,
    actions: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Mapping[str, Mapping[str, float]]],
    history: Sequence[Mapping[str, Any]],
    failure: Failure | None,
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "goal": goal,
        "textualized_state": observation.as_semantic_neutral_prompt_dict(),
        "action_catalog": actions,
        "continuous_parameter_catalog": parameters,
        "previous_plan_trace": list(history),
        "last_failure": failure.as_dict() if failure else None,
    }
