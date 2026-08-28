"""Paper-protocol VLM-TAMP English decomposition and grounding prompts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from baseline_common.models import Observation

from .models import ObjectUniverse, RefinementFailure, Subgoal


PROMPT_VERSION = 8
ENGLISH_SYSTEM_PROMPT = """\
You are an intermediate-goal proposer for a robot planning system.
Return only the requested JSON object.

Given the task goal, current observation, and images, propose a short ordered
sequence of intermediate goal states in plain English.
A second model query grounds these states into formal predicates, and a separate
task-and-motion planner chooses actions and continuous motion parameters.

Rules:
- Use only facts established by the supplied scene state and images. Do not
  assume an unobserved object's location or other unprovided scene facts.
- The RGB views show semantic instance aliases without internal IDs. The
  semantic_annotations mapping links every visible alias to the exact planning
  ID used by the symbolic state. Use only this observable mapping.
- Decide which intermediate goals are necessary and how they should be ordered.
- Include only objects supported by the images as relevant to the task. Do not
  enumerate every visible object or propose moving every object by default.
- Treat the action history and last planning failure as part of the current state.
- The independent simulator verifier has already established that the goal is
  incomplete before this request. Do not declare goal completion.
- Do not output formal predicates, primitive actions, coordinates, explanations,
  markdown, chain-of-thought, or extra fields.
"""

GROUNDING_SYSTEM_PROMPT = """\
You are the second-stage formal translator in a VLM-TAMP baseline.
Return only the requested JSON object.

Translate the supplied English intermediate goals, in the same order, into the
provided formal subgoal predicates. Skip an English step only when no supplied
predicate represents it. Resolve image aliases through semantic_annotations,
then output only exact IDs from the supplied planning objects and regions. Do
not add primitive actions, motion parameters, explanations,
markdown, chain-of-thought, or extra fields.
"""


def english_response_schema(max_steps: int) -> dict[str, Any]:
    step = {"type": "string", "minLength": 1, "maxLength": 240}
    return {
        "oneOf": [
            _status_array_variant(
                "STEPS", "steps", step, min_items=1, max_items=max_steps
            ),
            _status_array_variant(
                "NO_VALID_STEPS", "steps", step, min_items=0, max_items=0
            ),
        ]
    }


def grounding_response_schema(
    predicates: Mapping[str, Mapping[str, Any]] | Sequence[str],
    max_subgoals: int,
    *,
    object_ids: Sequence[str] = (),
    region_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if isinstance(predicates, Mapping) and object_ids and region_ids:
        values = {
            "object": sorted(set(object_ids)),
            "region": sorted(set(region_ids)),
            "destination": sorted(set(object_ids) | set(region_ids)),
        }
        variants = []
        for predicate, definition in predicates.items():
            arguments = definition["arguments"]
            variants.append(
                {
                    "type": "object",
                    "properties": {
                        "predicate": {"const": predicate},
                        "arguments": {
                            "type": "object",
                            "properties": {
                                name: {"type": "string", "enum": values[kind]}
                                for name, kind in arguments.items()
                            },
                            "required": list(arguments),
                            "additionalProperties": False,
                        },
                    },
                    "required": ["predicate", "arguments"],
                    "additionalProperties": False,
                }
            )
        subgoal: dict[str, Any] = {"oneOf": variants}
    else:
        subgoal = {
            "type": "object",
            "properties": {
                "predicate": {"type": "string", "enum": list(predicates)},
                "arguments": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                },
            },
            "required": ["predicate", "arguments"],
            "additionalProperties": False,
        }
    return {
        "oneOf": [
            _status_array_variant(
                "SUBGOALS",
                "subgoals",
                subgoal,
                min_items=1,
                max_items=max_subgoals,
            ),
            _status_array_variant(
                "NO_VALID_SUBGOALS",
                "subgoals",
                subgoal,
                min_items=0,
                max_items=0,
            ),
        ]
    }


def _status_array_variant(
    status: str,
    field: str,
    item_schema: Mapping[str, Any],
    *,
    min_items: int,
    max_items: int,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"const": status},
            field: {
                "type": "array",
                "minItems": min_items,
                "maxItems": max_items,
                "items": item_schema,
            },
        },
        "required": ["status", field],
        "additionalProperties": False,
    }


def scene_payload(
    goal: str,
    observation: Observation,
    universe: ObjectUniverse,
    succeeded_subgoals: Sequence[Subgoal],
    action_history: Sequence[Mapping[str, Any]],
    failure: RefinementFailure | None,
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "goal": goal,
        "textualized_state": observation.as_annotated_prompt_dict(),
        "object_universe": universe.as_dict(),
        "succeeded_subgoals": [item.as_dict() for item in succeeded_subgoals],
        "executed_action_history": list(action_history),
        "last_refinement_failure": failure.as_dict() if failure else None,
    }


def grounding_payload(
    english_steps: Sequence[str],
    observation: Observation,
    universe: ObjectUniverse,
    predicates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "english_intermediate_goals": list(english_steps),
        "planning_objects": universe.as_dict(),
        "known_regions": [
            {
                "id": item.region_id,
                "alias": item.label,
                "state": item.state,
                "inspected": item.inspected,
            }
            for item in observation.regions
        ],
        "semantic_annotations": observation.as_annotated_prompt_dict()[
            "semantic_annotations"
        ],
        "formal_subgoal_catalog": predicates,
    }
