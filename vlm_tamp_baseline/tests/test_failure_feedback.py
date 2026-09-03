from __future__ import annotations

from baseline_common.models import Entity, Observation, Region

from vlm_tamp_baseline.failure_feedback import (
    model_action_history,
    model_failure_feedback,
)
from vlm_tamp_baseline.models import ObjectUniverse, RefinementFailure, Subgoal
from vlm_tamp_baseline.prompt import scene_payload


def _observation() -> Observation:
    return Observation(
        "kitchen",
        1,
        (Entity("object_0001", "object", "coffee_source"),),
        (Region("countertop", "countertop", "open", True),),
        {"held_object": None},
    )


def test_model_feedback_hides_raw_controller_diagnostics() -> None:
    raw = "jar_yaw0_offset: 34.0 cm; jar_yaw30_offset: 35.8 cm"
    failure = RefinementFailure(
        "collision",
        raw,
        Subgoal("HOLDING", {"object_id": "object_0001"}),
    )

    feedback = model_failure_feedback(failure)

    assert feedback == {
        "code": "collision",
        "message": "The requested action could not be completed safely from the current observed state.",
        "subgoal": {
            "predicate": "HOLDING",
            "arguments": {"object_id": "object_0001"},
        },
    }
    assert raw not in str(feedback)


def test_inference_failure_has_a_distinct_model_visible_code() -> None:
    feedback = model_failure_feedback(
        RefinementFailure("inference_failed", "Model server returned HTTP 503")
    )

    assert feedback == {
        "code": "inference_failed",
        "message": "The model request failed before a valid response was received. Retry from the current observation.",
    }


def test_model_action_history_hides_raw_execution_message_and_details() -> None:
    raw = "No collision-free grasp candidate; jar_yaw0_offset: 34.0 cm"
    history = model_action_history(
        (
            {
                "action": {"skill": "PICK", "arguments": {"object_id": "object_0001"}},
                "success": False,
                "failure_code": "grasp_failed",
                "message": raw,
                "details": {"private": raw},
            },
        )
    )

    assert history == [
        {
            "action": {"skill": "PICK", "arguments": {"object_id": "object_0001"}},
            "success": False,
            "failure": {
                "code": "grasp_failed",
                "message": "The requested object could not be grasped from its current observed pose.",
            },
        }
    ]
    assert raw not in str(history)


def test_scene_payload_uses_only_compact_failure_feedback() -> None:
    raw = "all rejected grasp candidates and controller telemetry"
    observation = _observation()
    payload = scene_payload(
        "Prepare coffee.",
        observation,
        ObjectUniverse.observed(observation),
        (),
        (
            {
                "action": {"skill": "PICK", "arguments": {"object_id": "object_0001"}},
                "success": False,
                "failure_code": "collision",
                "message": raw,
                "details": {"raw": raw},
            },
        ),
        RefinementFailure("collision", raw),
    )

    assert payload["last_refinement_failure"]["code"] == "collision"
    assert payload["executed_action_history"][0]["failure"]["code"] == "collision"
    assert raw not in str(payload)
