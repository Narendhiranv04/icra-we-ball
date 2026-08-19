"""Prompt and structured-output schema for direct action planning."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import Failure, Observation


PROMPT_VERSION = 1
SYSTEM_PROMPT = """\
You are the task planner in an LLM3-style robot planning baseline.
Return only the requested JSON object.

Rules:
- Use only the supplied action catalogue, visible object IDs, and known region
  IDs. Never invent an object, region, action, or argument.
- The observation contains visible image-derived state only. A known but
  uninspected region has unknown contents. Do not assume what is inside it.
- Produce a short executable action sequence toward the goal. Do not output
  functional requirements or candidate-type rankings.
- INSPECT may name a known region even when it has not been inspected. Other
  actions may name only currently visible objects and known regions.
- A separate deterministic system checks geometry, IK, collisions, action
  preconditions, and goal completion. Do not claim that an unchecked motion
  will succeed.
- When failure feedback is supplied, revise the remaining plan using only that
  feedback and the latest observation. Do not repeat a failed action unchanged
  unless the new observation provides a concrete reason it can now succeed.
- Return GOAL_COMPLETE only when the supplied observation explicitly has
  goal_satisfied=true. Return NO_VALID_PLAN only when no catalogue action can
  make progress from the observed state.
- Do not include chain-of-thought, commentary, markdown, or extra fields.
"""


def response_schema(action_names: Sequence[str], max_actions: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["PLAN", "GOAL_COMPLETE", "NO_VALID_PLAN"],
            },
            "actions": {
                "type": "array",
                "maxItems": max_actions,
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "enum": list(action_names)},
                        "arguments": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                    },
                    "required": ["skill", "arguments"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status", "actions"],
        "additionalProperties": False,
    }


def task_payload(
    goal: str,
    observation: Observation,
    actions: Mapping[str, Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    failure: Failure | None,
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "goal": goal,
        "observation": observation.as_prompt_dict(),
        "action_catalog": actions,
        "completed_action_history": list(history),
        "last_failure": failure.as_dict() if failure else None,
    }
