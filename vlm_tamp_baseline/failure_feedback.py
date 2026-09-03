"""Compact, model-visible feedback for failed VLM-TAMP refinements."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import RefinementFailure


_MESSAGES = {
    "collision": "The requested action could not be completed safely from the current observed state.",
    "path_blocked": "The robot could not find a collision-free path for the requested action.",
    "ik_failed": "The robot could not reach a valid pose for the requested action.",
    "grasp_failed": "The requested object could not be grasped from its current observed pose.",
    "placement_failed": "The object could not be placed at the requested destination.",
    "precondition_failed": "The requested action's preconditions are not satisfied in the current observed state.",
    "hand_not_empty": "The robot is already holding an object and cannot perform the requested action.",
    "object_not_visible": "The requested object is not currently visible.",
    "ungrounded_object": "The requested object or destination is not currently visible.",
    "target_occupied": "The requested destination is occupied.",
    "function_unsatisfied": "The requested object cannot satisfy the required function.",
    "no_candidate": "No feasible candidate was found for the requested subgoal.",
    "tamp_refinement_failed": "No feasible task-and-motion refinement was found for the requested subgoal.",
    "symbolic_refinement_failed": "The requested subgoal cannot be refined from the current observed state.",
    "unsupported_subgoal": "The requested subgoal is not supported by this task domain.",
    "inference_failed": "The model request failed before a valid response was received. Retry from the current observation.",
    "invalid_vlm_output": "The previous response was not a valid plan. Propose valid intermediate goals from the current observation.",
    "no_valid_subgoals": "The previous response contained no valid formal subgoals. Propose a new plan from the current observation.",
    "effect_not_observed": "The requested effect was not observed after execution.",
    "goal_not_satisfied": "The executed subgoals did not satisfy the task goal.",
    "execution_failed": "The requested action could not be completed from the current observed state.",
    "internal_error": "The requested action could not be completed because the execution backend failed.",
}


def _code(value: object) -> str:
    text = str(value or "execution_failed").strip().lower()
    return text if text in _MESSAGES else "execution_failed"


def model_failure_feedback(failure: RefinementFailure | None) -> dict[str, Any] | None:
    """Return bounded failure information for the next VLM request.

    Raw controller messages can contain hundreds of calibration candidates.
    They remain in the episode artifact but are never included in a prompt.
    """
    if failure is None:
        return None
    code = _code(failure.code)
    result: dict[str, Any] = {"code": code, "message": _MESSAGES[code]}
    if failure.subgoal is not None:
        result["subgoal"] = failure.subgoal.as_dict()
    return result


def model_action_history(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove private physical diagnostics from VLM-visible action history."""
    result: list[dict[str, Any]] = []
    for row in rows:
        action = row.get("action")
        success = row.get("success")
        if not isinstance(action, Mapping) or not isinstance(success, bool):
            continue
        item: dict[str, Any] = {"action": dict(action), "success": success}
        if success:
            effects = row.get("effects", ())
            if isinstance(effects, Sequence) and not isinstance(effects, str):
                item["effects"] = [str(effect) for effect in effects]
        else:
            code = _code(row.get("failure_code"))
            item["failure"] = {"code": code, "message": _MESSAGES[code]}
        result.append(item)
    return result
