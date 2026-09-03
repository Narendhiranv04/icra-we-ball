"""Prompts following OWL-TAMP's discrete-sketch and constraint stages."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from baseline_common.models import Observation

from .models import Action, PlanSketch


PROMPT_VERSION = 3
MAX_SKETCH_ACTIONS = 64


def discrete_response_schema(max_actions: int = MAX_SKETCH_ACTIONS) -> dict[str, object]:
    action = {
        "type": "object",
        "properties": {
            "operator": {"type": "string", "minLength": 1, "maxLength": 32},
            "arguments": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {"type": "string", "minLength": 1, "maxLength": 64},
            },
        },
        "required": ["operator", "arguments"],
        "additionalProperties": False,
    }
    properties = {
        "status": {"type": "string"},
        "actions": {"type": "array", "items": action},
        "goal_literals": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 160},
        },
    }
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    **properties,
                    "status": {"const": "PLAN"},
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_actions,
                        "items": action,
                    },
                },
                "required": ["status", "actions", "goal_literals"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    **properties,
                    "status": {"const": "NO_PLAN"},
                    "actions": {"type": "array", "maxItems": 0},
                    "goal_literals": {"type": "array", "maxItems": 0},
                },
                "required": ["status", "actions", "goal_literals"],
                "additionalProperties": False,
            },
        ]
    }


def constraint_response_schema(action_index: int) -> dict[str, object]:
    helper_expression = (
        r"^(?:within_distance\([^(),]+,[^(),]+,[0-9]+(?:\.[0-9]+)?\)"
        r"|inside\([^(),]+,[^(),]+\)"
        r"|supported_by\([^(),]+,[^(),]+\)"
        r"|collision_free\([0-9]+\)"
        r"|reachable\([0-9]+\)"
        r"|upright\([^(),]+\))$"
    )
    return {
        "type": "object",
        "properties": {
            "constraints": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "action_index": {"const": action_index},
                        "description": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 160,
                        },
                        "expression": {
                            "type": "string",
                            "maxLength": 256,
                            "pattern": helper_expression,
                        },
                    },
                    "required": ["action_index", "description", "expression"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["constraints"],
        "additionalProperties": False,
    }


def discrete_prompt(
    scene: str,
    goal: str,
    observation: Observation,
    grounded_actions: Sequence[Action],
) -> str:
    actions = [row.as_dict() for row in grounded_actions]
    state = observation.as_annotated_prompt_dict()
    if scene == "living_room":
        goal_predicates = ["at(object,region)", "holding(object)"]
    elif scene == "workshop":
        goal_predicates = [
            "at(object,destination)", "holding(object)", "open(region)",
            "inserted(fastener,target)", "fastened(tool,fastener,target)",
        ]
    else:
        goal_predicates = [
            "at(object,destination)",
            "holding(object)",
            "open(region)",
            "poured(source,target)",
            "stirred(tool,target)",
            "served_with(target,tool)",
        ]
    return (
        "You are the discrete planning stage of OWL-TAMP. Infer a partial action "
        "sketch that is a required subsequence of a valid plan. Continuous poses, "
        "grasps, paths, and collision checks are deliberately deferred. Use only "
        "the exact grounded actions supplied below. RGB labels are semantic "
        "aliases; semantic_annotations maps them to planning IDs. Do not invent "
        "IDs, aliases, hidden contents, actions, or achieved effects. Reason "
        "internally, then return JSON only. Return NO_PLAN only when the observable "
        "state and grounded actions prove that no sketch can address the goal. "
        "Return a concise partial skeleton containing only task-essential action "
        "choices. Do not enumerate, copy, or summarize the grounded-action list. "
        f"Return no more than {MAX_SKETCH_ACTIONS} actions.\n\n"
        f"Scene: {scene}\nTask: {goal}\n"
        f"Observable initial state:\n{json.dumps(state, sort_keys=True, separators=(',', ':'))}\n"
        f"Relaxed grounded actions:\n{json.dumps(actions, sort_keys=True, separators=(',', ':'))}\n"
        f"Allowed grounded goal-literal forms: {json.dumps(goal_predicates)}\n"
        "Output schema: {\"status\":\"PLAN|NO_PLAN\",\"actions\":["
        "{\"operator\":string,\"arguments\":[string]}],\"goal_literals\":[string]}"
    )


def constraint_prompt(
    scene: str,
    goal: str,
    observation: Observation,
    sketch: PlanSketch,
    action_index: int,
) -> str:
    helper_codebook = {
        "within_distance(a,b,meters)": "center distance is at most meters",
        "inside(a,b)": "a is geometrically contained by b",
        "supported_by(a,b)": "a has stable support on b",
        "collision_free(action_index)": "sampled motion for the action is collision-free",
        "reachable(action_index)": "sampled manipulation pose is robot-reachable",
        "upright(a)": "a preserves an upright liquid-safe orientation",
    }
    return (
        "You are the continuous-constraint stage of OWL-TAMP. Translate only "
        "geometric or physical requirements implied by the task and action sketch "
        "into boolean expressions using the helper codebook. Do not add semantic "
        "facts, hidden objects, new actions, preferences, or efficiency criteria. "
        "Every expression must be a single helper call. Return JSON only.\n\n"
        f"Scene: {scene}\nTask: {goal}\n"
        f"Observable state: {json.dumps(observation.as_annotated_prompt_dict(), sort_keys=True, separators=(',', ':'))}\n"
        f"Action sketch: {json.dumps(sketch.as_dict(), sort_keys=True, separators=(',', ':'))}\n"
        f"Generate the constraint for action index {action_index} only.\n"
        f"Helper codebook: {json.dumps(helper_codebook, sort_keys=True)}\n"
        "Output schema: {\"constraints\":[{\"action_index\":integer,"
        "\"description\":string,\"expression\":string}]}"
    )
