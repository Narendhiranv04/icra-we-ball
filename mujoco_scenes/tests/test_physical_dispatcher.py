from mujoco_scenes.tamp.physical_dispatcher import (
    MuJoCoSkillDispatcher,
    canonical_action,
)
from mujoco_scenes.tamp.skills import FailureCode, SkillAction, SkillStartError


class FakePhysical:
    def __init__(self, result=None):
        self.result = result or {"success": True, "status": "OK"}
        self.requests = []

    def execute_phase2_action(self, action):
        self.requests.append(action)
        return self.result


class PlanAwarePhysical(FakePhysical):
    def authorize_plan(self, plan):
        self.plan = plan


def test_public_actions_translate_to_phase_c_contract():
    assert canonical_action(
        SkillAction("pour", {"source_id": "kettle", "target_id": "mug"})
    ) == {"action": "POUR", "arguments": ["kettle", "mug"]}
    assert canonical_action(
        SkillAction("place", {"object": "mug", "region": "serving"})
    ) == {"action": "PLACE", "arguments": ["mug", "serving"]}


def test_dispatcher_preserves_physical_failure_for_replanning():
    physical = FakePhysical(
        {"success": False, "failure_code": "APPROACH_COLLISION"}
    )
    dispatcher = MuJoCoSkillDispatcher(physical)
    dispatcher.start(SkillAction("pick", {"object_id": "spoon"}))
    result = dispatcher.update()
    assert result is not None and not result.success
    assert result.failure_code is FailureCode.COLLISION
    assert result.recoverable


def test_dispatcher_preserves_json_safe_physical_failure_telemetry():
    physical = FakePhysical(
        {
            "success": False,
            "failure_code": "PLACEMENT_FAILED",
            "post_place": {"final_position": (0.1, 0.2, 0.3)},
        }
    )
    dispatcher = MuJoCoSkillDispatcher(physical)
    dispatcher.start(
        SkillAction(
            "PLACE", {"object_id": "mug", "region_id": "counter"}
        )
    )
    result = dispatcher.update()

    assert result.failure_code is FailureCode.PLACEMENT_FAILED
    assert result.details["post_place"]["final_position"] == [0.1, 0.2, 0.3]


def test_occupied_gripper_exception_is_a_recoverable_precondition_failure():
    class OccupiedGripperPhysical:
        def execute_phase2_action(self, _action):
            raise RuntimeError("The gripper is not available for another pick")

    dispatcher = MuJoCoSkillDispatcher(OccupiedGripperPhysical())
    dispatcher.start(SkillAction("PICK", {"object_id": "spoon"}))
    result = dispatcher.update()

    assert result.failure_code is FailureCode.PRECONDITION_FAILED
    assert result.recoverable
    assert result.details["exception_type"] == "RuntimeError"


def test_inspection_uses_observation_refresh_handler_not_physical_action():
    physical = FakePhysical()
    inspected = []
    dispatcher = MuJoCoSkillDispatcher(
        physical,
        inspect=lambda region: inspected.append(region) or {"success": True},
    )
    dispatcher.start(SkillAction("INSPECT", {"region_id": "C1"}))
    result = dispatcher.update()
    assert result.success
    assert result.effects == ("inspected(C1)",)
    assert inspected == ["C1"]
    assert physical.requests == []


def test_missing_required_argument_fails_before_execution():
    try:
        canonical_action(SkillAction("PICK", {}))
    except SkillStartError as error:
        assert error.code is FailureCode.PRECONDITION_FAILED
    else:
        raise AssertionError("missing PICK argument was accepted")


def test_selected_plan_is_frozen_before_physical_execution():
    physical = PlanAwarePhysical()
    dispatcher = MuJoCoSkillDispatcher(physical)
    dispatcher.prepare(
        (
            SkillAction("PICK", {"object_id": "spoon"}),
            SkillAction("STIR", {"tool_id": "spoon", "target_id": "mug"}),
        )
    )
    assert physical.plan == [
        {"action": "PICK", "arguments": ["spoon"]},
        {"action": "STIR", "arguments": ["spoon", "mug"]},
    ]


def test_plan_preparation_returns_structured_failure_instead_of_raising():
    class RejectingPhysical(FakePhysical):
        def authorize_plan(self, _plan):
            raise RuntimeError("RRT collision while authorizing plan")

    result = MuJoCoSkillDispatcher(RejectingPhysical()).prepare(
        (SkillAction("PICK", {"object_id": "spoon"}),)
    )

    assert not result.success
    assert result.failure_code is FailureCode.COLLISION
    assert result.recoverable


def test_invalid_action_during_preparation_is_a_precondition_failure():
    result = MuJoCoSkillDispatcher(PlanAwarePhysical()).prepare(
        (SkillAction("PICK", {}),)
    )

    assert not result.success
    assert result.failure_code is FailureCode.PRECONDITION_FAILED


def test_successful_physical_motion_emits_verified_symbolic_effect():
    dispatcher = MuJoCoSkillDispatcher(FakePhysical())
    dispatcher.start(
        SkillAction("PLACE", {"object_id": "mug", "region_id": "serving"})
    )
    assert dispatcher.update().effects == ("placed(mug,serving)",)


def test_malformed_physical_success_is_a_terminal_internal_failure():
    dispatcher = MuJoCoSkillDispatcher(
        FakePhysical({"success": "yes", "effects": "holding(spoon)"})
    )
    dispatcher.start(SkillAction("PICK", {"object_id": "spoon"}))

    result = dispatcher.update()

    assert not result.success
    assert result.failure_code is FailureCode.INTERNAL_ERROR
    assert not result.recoverable


def test_physical_effects_must_not_be_a_single_string():
    dispatcher = MuJoCoSkillDispatcher(
        FakePhysical({"success": True, "effects": "holding(spoon)"})
    )
    dispatcher.start(SkillAction("PICK", {"object_id": "spoon"}))

    result = dispatcher.update()

    assert not result.success
    assert result.failure_code is FailureCode.INTERNAL_ERROR
