from __future__ import annotations

import base64
from collections import deque

from baseline_common.models import ActionResult, Entity, Observation, Region

from owl_tamp_baseline.domain import executed_encoding, relaxed_ground
from owl_tamp_baseline.models import Action, PlanSketch, PlanningResult
from owl_tamp_baseline.planner import (
    OWLTAMPPlanner,
    OWLTAMPPlannerConfig,
    protocol_max_tokens,
)
from owl_tamp_baseline.receding_horizon import OWLTAMPRecedingHorizon
from owl_tamp_baseline.refinement import constrained_breadth_first_search


def observation() -> Observation:
    return Observation(
        "living_room",
        0,
        (Entity("object_0001", "object", "object_0001", {"region_id": "staging_area"}),),
        (
            Region("staging_area", "staging_area", "open", True),
            Region("region_0001", "region_0001", "open", True),
        ),
        {"holding": None, "workspace": "home"},
        False,
    )


def test_relaxed_grounding_and_executed_encoding() -> None:
    grounded = relaxed_ground(
        "living_room", {"object_0001"}, {"staging_area", "region_0001"}
    )
    pick = Action("PICK", ("object_0001",))
    place = Action("PLACE", ("object_0001", "region_0001"))
    assert pick in grounded and place in grounded
    rows = executed_encoding((pick, place))
    assert rows[0]["extra_precondition"] == "Executed(0)"
    assert rows[1]["extra_effect"] == "Executed(2)"


def test_workshop_relaxed_grounding_supports_inspection_and_fastening() -> None:
    grounded = relaxed_ground(
        "workshop",
        {"object_0001", "object_0002", "object_0003"},
        {"region_0001", "region_0004"},
        {"region_0001"},
    )
    assert Action("INSPECT", ("region_0001",)) in grounded
    assert Action("FASTEN", ("object_0001", "object_0002", "object_0003")) in grounded


def test_single_call_protocol_reserves_prompt_context() -> None:
    assert protocol_max_tokens(8192, "single_call") == 2048
    assert protocol_max_tokens(8192, "native") == 8192


def test_symbolic_search_requires_goal_literals() -> None:
    state = observation()
    grounded = relaxed_ground("living_room", state.object_ids, state.region_ids)
    plan = constrained_breadth_first_search(
        state,
        grounded,
        (Action("PLACE", ("object_0001", "region_0001")),),
        ("at(object_0001,region_0001)",),
    )
    assert plan == (
        Action("PICK", ("object_0001",)),
        Action("PLACE", ("object_0001", "region_0001")),
    )


def test_symbolic_search_handles_full_living_room_plan_with_default_budget() -> None:
    objects = tuple(f"object_{index:04d}" for index in range(1, 6))
    regions = ("staging_area", "region_0001", "region_0002", "region_0003")
    state = Observation(
        "living_room",
        0,
        tuple(
            Entity(item, "object", item, {"region_id": "staging_area"})
            for item in objects
        ),
        tuple(Region(item, item, "open", True) for item in regions),
        {"holding": None, "workspace": "home"},
        False,
    )
    targets = ("region_0001", "region_0001", "region_0002", "region_0003", "region_0003")
    sketch = tuple(
        action
        for object_id, target in zip(objects, targets)
        for action in (
            Action("PICK", (object_id,)),
            Action("PLACE", (object_id, target)),
        )
    )
    grounded = relaxed_ground("living_room", state.object_ids, state.region_ids)
    goals = tuple(
        f"at({object_id},{target})"
        for object_id, target in zip(objects, targets)
    )

    assert constrained_breadth_first_search(state, grounded, sketch, goals) == sketch


class FakeTransport:
    def __init__(self):
        self.payloads = []
        self.responses = deque(
            [
                {
                    "status": "PLAN",
                    "actions": [
                        {"operator": "PICK", "arguments": ["object_0001"]},
                        {"operator": "PLACE", "arguments": ["object_0001", "region_0001"]},
                    ],
                    "goal_literals": ["at(object_0001,region_0001)"],
                },
                {
                    "constraints": [
                        {"action_index": 0, "description": "reachable grasp", "expression": "reachable(0)"}
                    ]
                },
                {
                    "constraints": [
                        {"action_index": 1, "description": "stable support", "expression": "supported_by(object_0001, region_0001)"}
                    ]
                },
            ]
        )

    def complete(self, payload):
        self.payloads.append(payload)
        content = self.responses.popleft()
        return {"choices": [{"message": {"content": content}}]}


def test_two_stage_planner_uses_images_and_refines() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
    transport = FakeTransport()
    planner = OWLTAMPPlanner(
        OWLTAMPPlannerConfig(), transport=transport
    )
    result = planner.plan(
        "Put the visible object on the target support",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
    )
    assert result.status == "PLAN"
    assert len(result.actions) == 2
    assert result.samples_tested == 2
    assert not transport.responses
    assert len(transport.payloads) == 3
    sketch_payload = transport.payloads[0]
    sketch_format = sketch_payload["response_format"]
    assert sketch_format["type"] == "json_schema"
    assert sketch_format["json_schema"]["strict"] is True
    sketch_text = sketch_payload["messages"][0]["content"][0]["text"]
    assert "Return no more than 64 actions" in sketch_text
    assert "only task-essential action choices" in sketch_text
    assert "Do not enumerate" in sketch_text
    assert "Format example only" not in sketch_text
    assert "example_object" not in sketch_text
    constraint_format = transport.payloads[1]["response_format"]
    constraint = constraint_format["json_schema"]["schema"]
    assert constraint["properties"]["constraints"]["maxItems"] == 1
    assert constraint["properties"]["constraints"]["items"]["properties"][
        "action_index"
    ]["const"] == 0
    assert "pattern" in constraint["properties"]["constraints"]["items"][
        "properties"
    ]["expression"]
    assert len(planner.response_trace) == 3


def test_planner_can_exclude_fixed_scene_objects_from_relaxed_grounding() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")

    class NoPlanTransport:
        def complete(self, _payload):
            return {
                "choices": [
                    {
                        "message": {
                            "content": {
                                "status": "NO_PLAN",
                                "actions": [],
                                "goal_literals": [],
                            }
                        }
                    }
                ]
            }

    planner = OWLTAMPPlanner(OWLTAMPPlannerConfig(), transport=NoPlanTransport())
    planner.plan(
        "Inspect the available storage",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
        movable_object_ids=(),
    )

    prompt = planner.trace["model_prompts"]["discrete_sketch"]
    assert '"operator":"PICK"' not in prompt
    assert '"operator":"PLACE"' not in prompt


def test_single_call_protocol_skips_auxiliary_constraint_queries() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")

    class SketchOnlyTransport:
        def __init__(self):
            self.calls = 0

        def complete(self, _payload):
            self.calls += 1
            return {
                "choices": [{"message": {"content": {
                    "status": "PLAN",
                    "actions": [
                        {"operator": "PICK", "arguments": ["object_0001"]},
                        {"operator": "PLACE", "arguments": ["object_0001", "region_0001"]},
                    ],
                    "goal_literals": ["at(object_0001,region_0001)"],
                }}}]
            }

    transport = SketchOnlyTransport()
    planner = OWLTAMPPlanner(OWLTAMPPlannerConfig(), transport=transport)
    result = planner.plan(
        "Put the object on the target support",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
        max_vlm_requests=1,
    )

    assert transport.calls == 1
    assert len(planner.response_trace) == 1
    assert planner.trace["constraint_generation_complete"] is False
    assert result.status == "PLAN"


class InvalidConstraintTransport:
    def __init__(self):
        self.responses = deque(
            [
                {
                    "status": "PLAN",
                    "actions": [
                        {"operator": "PICK", "arguments": ["object_0001"]},
                    ],
                    "goal_literals": ["holding(object_0001)"],
                },
                {
                    "constraints": [
                        {
                            "action_index": 0,
                            "description": "unsupported model helper",
                            "expression": "graspable(object_0001)",
                        }
                    ]
                },
            ]
        )

    def complete(self, _payload):
        content = self.responses.popleft()
        return {"choices": [{"message": {"content": content}}]}


def test_invalid_model_constraint_is_a_scored_failure_not_a_crash() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
    planner = OWLTAMPPlanner(
        OWLTAMPPlannerConfig(), transport=InvalidConstraintTransport()
    )
    result = planner.plan(
        "Pick the visible object",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
    )
    assert result.status == "INVALID_MODEL_OUTPUT"
    assert not result.actions
    assert "unknown helper" in result.failure


class InvalidJSONTransport:
    def complete(self, _payload):
        return {"choices": [{"message": {"content": '{"status": "PLAN"'}}]}


def test_invalid_json_is_a_scored_failure_not_a_crash() -> None:
    image = "data:image/png;base64," + base64.b64encode(b"x").decode("ascii")
    planner = OWLTAMPPlanner(
        OWLTAMPPlannerConfig(), transport=InvalidJSONTransport()
    )
    result = planner.plan(
        "Pick the visible object",
        observation(),
        ({"camera": "camera_1", "data_url": image},),
        lambda _action, _constraints, _trial: True,
    )
    assert result.status == "INVALID_MODEL_OUTPUT"
    assert result.failure == (
        "Invalid OWL-TAMP discrete sketch: Completion content is not valid JSON"
    )


def test_receding_horizon_reobserves_after_each_successful_action() -> None:
    state = {"revision": 0, "held": False, "done": False}

    class Planner:
        response_trace = [{"request": 1}, {"request": 2}]
        trace = {"stage": "initial"}

        def plan(self, _goal, observation, _images, _oracle, **_kwargs):
            if observation.revision == 0:
                action = Action("PICK", ("object_0001",))
            else:
                action = Action("PLACE", ("object_0001", "region_0001"))
            return PlanningResult("PLAN", PlanSketch("PLAN", (action,), ()), (action,), (), 1, 1)

    def observe():
        return (
            Observation(
                "living_room",
                state["revision"],
                (Entity("object_0001", "object", "object", {}),),
                (Region("region_0001", "region", "open", True),),
                {"holding": state["held"]},
                state["done"],
            ),
            (),
        )

    def execute(action):
        if action.operator == "PICK":
            state["held"] = True
        else:
            state["held"] = False
            state["done"] = True
        state["revision"] += 1
        return ActionResult.succeeded(f"completed({action.operator})")

    result = OWLTAMPRecedingHorizon(
        Planner(), observe, execute, lambda row: row.goal_satisfied,
        lambda _action, _constraints, _trial: True,
    ).run("Place the object")

    assert result.success
    assert result.planning_rounds == 2
    assert result.replans == 1
    assert result.executed_actions == 2
    assert result.raw_vlm_requests == 4
