from __future__ import annotations

import json

import pytest
from mujoco_scenes.tamp.discovery_planner import (
    OpenAIDiscoveryPlanner,
    OpenAIPlannerConfig,
)
from mujoco_scenes.tamp.discovery_replanning import PlannerRequest, PlanningSnapshot
from mujoco_scenes.tamp.discovery_replanning import RecoverablePlanningError
from mujoco_scenes.tamp.state import ObjectObservation, ObservedState, RegionObservation, RobotObservation


class FakeTransport:
    def __init__(self, content):
        self.content = content
        self.payload = None

    def complete(self, payload):
        self.payload = payload
        return {"choices": [{"message": {"content": self.content}}]}


def _request() -> PlannerRequest:
    state = ObservedState(
        {
            "mug": ObjectObservation("mug", "mug", True, "countertop"),
            "hidden_spoon": ObjectObservation("hidden_spoon", "spoon", False, "D1"),
        },
        {
            "countertop": RegionObservation("countertop", "countertop", True),
            "D1": RegionObservation("D1", "drawer", False),
        },
        RobotObservation("home"),
    )
    return PlannerRequest("kitchen", "Place the visible mug on the counter.", PlanningSnapshot(state), (), ())


def _planner(content):
    planner = OpenAIDiscoveryPlanner(
        OpenAIPlannerConfig("http://127.0.0.1:1/v1", "test-model", "kitchen")
    )
    transport = FakeTransport(content)
    planner.transport = transport
    return planner, transport


def test_prompt_exposes_only_visible_state_and_no_old_plan():
    planner, transport = _planner(
        {"status": "PLAN", "actions": [{"name": "PICK", "arguments": {"object_id": "mug"}}]}
    )

    result = planner.plan(_request())

    assert result.actions[0].arguments["object_id"] == "mug"
    user_text = transport.payload["messages"][1]["content"][0]["text"]
    prompt = json.loads(user_text)
    assert set(prompt["observation"]["visible_objects"]) == {"mug"}
    assert set(prompt["observation"]["known_regions"]) == {"countertop"}
    assert "hidden_spoon" not in user_text
    assert "unexecuted_previous_actions" not in prompt
    assert prompt["available_actions"]["PICK"]["description"] == "Pick one currently visible object."
    assert "held_object" in transport.payload["messages"][0]["content"]
    assert transport.payload["seed"] == 0


def test_unknown_object_reference_is_rejected_before_execution():
    planner, _transport = _planner(
        {
            "status": "PLAN",
            "actions": [{"name": "PICK", "arguments": {"object_id": "hidden_spoon"}}],
        }
    )

    with pytest.raises(RecoverablePlanningError, match="not visible"):
        planner.plan(_request())


def test_model_call_trace_omits_image_payloads_and_private_gt(tmp_path):
    planner = OpenAIDiscoveryPlanner(
        OpenAIPlannerConfig(
            "http://127.0.0.1:1/v1",
            "test-model",
            "kitchen",
            trace_dir=tmp_path,
        )
    )
    planner.transport = FakeTransport({
        "status": "PLAN",
        "actions": [{"name": "PICK", "arguments": {"object_id": "mug"}}],
    })

    planner.plan(_request())
    trace = json.loads((tmp_path / "call_001.json").read_text())

    assert trace["response"]["status"] == "PLAN"
    assert trace["camera_names"] == []
    assert trace["private_goal_evaluator_exposed"] is False
    assert "base64" not in json.dumps(trace)


def test_invalid_action_order_is_rejected_before_execution():
    planner, _transport = _planner(
        {
            "status": "PLAN",
            "actions": [
                {
                    "name": "PLACE",
                    "arguments": {"object_id": "mug", "region_id": "countertop"},
                }
            ],
        }
    )

    with pytest.raises(RecoverablePlanningError, match="requires holding"):
        planner.plan(_request())
