from types import SimpleNamespace

from mujoco_scenes.kitchen_tamp_execution import (
    KitchenExecutionObserver,
    effects_goal_verifier,
)
from mujoco_scenes.tamp.grounded_execution import GroundedTask
from mujoco_scenes.tamp.state import ObservedState, RobotObservation


def test_effect_goal_verifier_requires_all_physical_postconditions():
    verify = effects_goal_verifier(("stirred(spoon,mug)", "placed(mug,serving)"))
    task = GroundedTask("coffee", "serve", {"status": "COMPLETE"})
    state = ObservedState({}, {}, RobotObservation("home"))
    assert not verify(
        task,
        state,
        [{"success": True, "effects": ["stirred(spoon,mug)"]}],
    )
    assert verify(
        task,
        state,
        [
            {"success": True, "effects": ["stirred(spoon,mug)"]},
            {"success": True, "effects": ["placed(mug,serving)"]},
        ],
    )


def _phase_b_with_held(held):
    inventory = {
        "objects": [
            {
                "generic_object_id": "object_0001",
                "semantic_label": "coffee jar",
                "source_context": {"observed_source_region": "countertop"},
            }
        ]
    }
    return SimpleNamespace(
        resolution={
            "accepted": [
                {
                    "generic_object_id": "object_0001",
                    "physical_backend_body": "s1i_compact_coffee_jar",
                }
            ]
        },
        inventory=inventory,
        current_workspace=SimpleNamespace(value="HOME"),
        manipulation=SimpleNamespace(
            executor=SimpleNamespace(held_object=held)
        ),
        scene=SimpleNamespace(get_region_observation_states=lambda: {}),
    )


def test_execution_observer_maps_backend_held_name_to_generic_id():
    state = KitchenExecutionObserver(
        _phase_b_with_held("s1i_compact_coffee_jar")
    )()

    assert state.robot.held_object == "object_0001"
    assert "s1i_compact_coffee_jar" not in str(state.as_dict())


def test_execution_observer_rejects_unbound_backend_held_name():
    observer = KitchenExecutionObserver(_phase_b_with_held("private_body_name"))

    try:
        observer()
    except RuntimeError as error:
        assert "no generic execution binding" in str(error)
    else:
        raise AssertionError("unbound backend identity crossed the observer")
