from unittest.mock import Mock

from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from mujoco_scenes.kitchen_object_manipulation import KitchenPlacementResolver


def test_phase_c_operators_fail_without_symbolic_effects():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    for operator in ("pour", "stir"):
        result = dispatcher.execute_phase2_action(
            {"action": operator, "arguments": ["object_1", "object_2"]}
        )
        assert result["status"] == "UNSUPPORTED_PHASE_C_OPERATOR"
        assert result["symbolic_effects_applied"] is False


def test_carried_move_validates_before_and_after_and_uses_payload_collision():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.phase_a = Mock()
    dispatcher.phase_a.current_workspace = KitchenWorkspace.HOME
    dispatcher.phase_a._move.return_value = {"success": True, "status": "OK"}
    dispatcher.manipulation = Mock()
    dispatcher._held_state = Mock(
        side_effect=[
            {"validation_status": "TRUE"},
            {"validation_status": "TRUE"},
        ]
    )
    result = dispatcher.move(KitchenWorkspace.LEFT_SIDE, carrying_object_id="object_1")
    assert result["success"] is True
    assert result["held_object_included_in_collision_check"] is True
    assert result["held_state_before"]["validation_status"] == "TRUE"
    assert result["held_state_after"]["validation_status"] == "TRUE"


def test_redundant_move_is_omitted():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.phase_a = Mock()
    dispatcher.phase_a.current_workspace = KitchenWorkspace.HOME
    result = dispatcher.move(KitchenWorkspace.HOME)
    assert result["status"] == "REDUNDANT_MOVE_OMITTED"
    dispatcher.phase_a._move.assert_not_called()


def test_serving_allocator_is_deterministic_and_role_separated():
    inventory = {
        "objects": [
            {"generic_object_id": "coffee_b", "selected_functions": ["coffee_vessel"], "observed_centroid_world_m": [0.2, 0, 0.63]},
            {"generic_object_id": "coffee_a", "selected_functions": ["coffee_vessel"], "observed_centroid_world_m": [-0.2, 0, 0.63]},
            {"generic_object_id": "soup_a", "selected_functions": ["soup_bowl"], "observed_centroid_world_m": [-0.1, 0, 0.63]},
            {"generic_object_id": "soup_b", "selected_functions": ["soup_bowl"], "observed_centroid_world_m": [0.1, 0, 0.63]},
        ]
    }
    resolution = {
        "accepted": [
            {"generic_object_id": row["generic_object_id"], "physical_backend_body": row["generic_object_id"]}
            for row in inventory["objects"]
        ]
    }
    resolver = KitchenPlacementResolver(Mock(), inventory, resolution)
    assert resolver.resolve("coffee_a", "serving_area").target_position_world_m[:2] == (-0.16, -0.48)
    assert resolver.resolve("soup_a", "serving_area").target_position_world_m[:2] == (-0.16, -0.64)


def test_low_level_pick_rejects_wrong_workspace():
    # The policy is checked before any IK/contact execution. This guard does
    # not depend on simulator metadata or a Python held-object flag.
    from mujoco_scenes.kitchen_object_manipulation import KitchenObjectManipulationExecutor

    executor = object.__new__(KitchenObjectManipulationExecutor)
    executor.by_id = {"object_1": {"physical_backend_body": "body", "grasp_family": "VESSEL"}}
    executor.inventory_by_id = {
        "object_1": {
            "source_context": {
                "required_workspace": "left_side",
                "source_container": "C1",
            }
        }
    }
    result = executor.pick("object_1", KitchenWorkspace.HOME, {"C1"})
    assert result.status == "WORKSPACE_PRECONDITION_UNSATISFIED"
    assert result.physics_steps == 0
