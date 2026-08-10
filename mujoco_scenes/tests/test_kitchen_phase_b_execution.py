from unittest.mock import Mock

from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from mujoco_scenes.kitchen_object_manipulation import (
    KitchenPlacementResolver,
    ServingPlacementState,
)


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


def _serving_resolver():
    objects = [
        {
            "generic_object_id": object_id,
            "selected_functions": ["soup_bowl"],
            "observed_centroid_world_m": [x, 0.0, 0.63],
            "observed_dimensions_m": {"length": 0.10, "width": 0.10},
        }
        for object_id, x in (("bowl_a", -0.2), ("bowl_b", 0.0), ("bowl_c", 0.2))
    ]
    resolution = {
        "accepted": [
            {"generic_object_id": row["generic_object_id"], "physical_backend_body": row["generic_object_id"]}
            for row in objects
        ]
    }
    return KitchenPlacementResolver(Mock(), {"objects": objects}, resolution)


def test_serving_allocator_uses_persistent_occupied_state_for_second_and_third():
    resolver = _serving_resolver()
    targets = []
    for object_id in ("bowl_a", "bowl_b", "bowl_c"):
        target = resolver.resolve(object_id, "serving_area")
        resolver.record_successful_serving_placement(object_id, target)
        targets.append(target)

    centres = [target.target_position_world_m[:2] for target in targets]
    assert len(set(centres)) == 3
    for index, centre in enumerate(centres):
        for other in centres[index + 1:]:
            assert not resolver.footprints_overlap(centre, (0.10, 0.10), other, (0.10, 0.10))


def test_serving_allocator_rejects_overlapping_nominal_candidate():
    resolver = _serving_resolver()
    nominal = resolver.resolve("bowl_b", "serving_area")
    resolver.serving_placements["occupied"] = ServingPlacementState(
        object_id="occupied",
        backend_body="occupied",
        centre_xy_m=nominal.target_position_world_m[:2],
        footprint_xy_m=(0.10, 0.10),
        yaw_world_rad=0.0,
        support_backend="serving_surface",
    )
    replacement = resolver.resolve("bowl_b", "serving_area")
    assert replacement.target_position_world_m[:2] != nominal.target_position_world_m[:2]
    assert not resolver.footprints_overlap(
        replacement.target_position_world_m[:2],
        resolver.footprint("bowl_b"),
        nominal.target_position_world_m[:2],
        (0.10, 0.10),
    )


def test_serving_allocator_sequence_is_deterministic():
    def allocate():
        resolver = _serving_resolver()
        result = []
        for object_id in ("bowl_a", "bowl_b", "bowl_c"):
            target = resolver.resolve(object_id, "serving_area")
            resolver.record_successful_serving_placement(object_id, target)
            result.append(target.target_position_world_m)
        return result

    assert allocate() == allocate()


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
