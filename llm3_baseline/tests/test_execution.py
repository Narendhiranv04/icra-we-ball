from llm3_baseline.execution import LLM3MuJoCoExecutor
from llm3_baseline.execution import observation_from_state
from llm3_baseline.models import Action
from mujoco_scenes.tamp.physical_dispatcher import MuJoCoSkillDispatcher
from mujoco_scenes.tamp.state import (
    ObjectObservation,
    ObservedState,
    RegionObservation,
    RobotObservation,
)


class FakePhysical:
    def __init__(self, result):
        self.result = result

    def execute_phase2_action(self, _action):
        return self.result


def test_llm3_and_functional_planner_share_failure_mapping():
    adapter = LLM3MuJoCoExecutor(
        MuJoCoSkillDispatcher(
            FakePhysical({"success": False, "status": "IK_TOLERANCE"})
        )
    )
    result = adapter.execute(Action("PICK", {"object_id": "spoon"}))
    assert not result.success
    assert result.failure_code == "ik_failed"
    assert result.recoverable


def test_llm3_receives_only_visible_entities_from_shared_state():
    state = ObservedState(
        {
            "mug": ObjectObservation("mug", "mug", True),
            "spoon": ObjectObservation("spoon", "spoon", False),
        },
        {"D1": RegionObservation("D1", "drawer", True, open=False)},
        RobotObservation("home"),
        revision=3,
    )
    observation = observation_from_state("kitchen", state)
    assert observation.object_ids == {"mug"}
    assert observation.region_ids == {"D1"}


def test_successful_physical_effects_reach_the_baseline_ledger():
    observed = []
    adapter = LLM3MuJoCoExecutor(
        MuJoCoSkillDispatcher(FakePhysical({"success": True})),
        effect_sink=observed.append,
    )
    result = adapter.execute(Action("PICK", {"object_id": "spoon"}))
    assert result.success
    assert result.effects == ("holding(spoon)",)
    assert observed == [("holding(spoon)",)]


def test_failure_telemetry_reaches_episode_action_result():
    adapter = LLM3MuJoCoExecutor(
        MuJoCoSkillDispatcher(
            FakePhysical(
                {
                    "success": False,
                    "failure_code": "PLACEMENT_FAILED",
                    "post_place": {"support_contact": False},
                }
            )
        )
    )
    result = adapter.execute(
        Action("PLACE", {"object_id": "spoon", "region_id": "countertop"})
    )

    assert result.details["post_place"]["support_contact"] is False


def test_invalid_action_returns_failure_instead_of_escaping_executor():
    adapter = LLM3MuJoCoExecutor(
        MuJoCoSkillDispatcher(FakePhysical({"success": True}))
    )

    result = adapter.execute(Action("PICK", {}))

    assert not result.success
    assert result.failure_code == "precondition_failed"


def test_unexpected_dispatch_exception_returns_terminal_failure():
    class BrokenDispatcher:
        def start(self, _action):
            raise RuntimeError("controller unavailable")

    result = LLM3MuJoCoExecutor(BrokenDispatcher()).execute(
        Action("PICK", {"object_id": "spoon"})
    )

    assert not result.success
    assert result.failure_code == "internal_error"
    assert not result.recoverable
