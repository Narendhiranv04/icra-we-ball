from unittest.mock import Mock

import numpy as np

from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from mujoco_scenes.kitchen_object_manipulation import (
    aligned_payload_gripper_yaw,
    KitchenPlacementResolver,
    KitchenObjectManipulationExecutor,
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


def test_serving_bowl_alignment_preserves_live_grasp_yaw_offset():
    # A body currently at 0 degrees while its gripper is at -60 degrees must
    # keep the gripper at -60 degrees to release body-aligned on a 0-degree
    # support.  This avoids transferring the storage grasp yaw to the bowl.
    top_down = np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)))
    angle = np.deg2rad(-60.0)
    grasp_yaw = np.array((
        (np.cos(angle), -np.sin(angle), 0.0),
        (np.sin(angle), np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    ))
    yaw = aligned_payload_gripper_yaw(
        grasp_yaw @ top_down, np.eye(3), top_down, np.deg2rad(0.0)
    )
    assert np.isclose(yaw, np.deg2rad(-60.0))

    assert -np.pi <= yaw <= np.pi


def test_home_place_stance_candidates_are_bounded_and_target_centred():
    candidates = KitchenObjectManipulationExecutor._home_place_candidates(
        np.array((-0.14, -0.24, 0.598))
    )
    assert len(candidates) == 12
    assert candidates[0] == (0.20, 0.14, 0.0)
    assert {row[0] for row in candidates} == {0.20, 0.23, 0.25, 0.28}
    assert {round(row[1], 2) for row in candidates} == {0.11, 0.14, 0.17}
    assert all(-0.18 <= row[1] <= 0.18 for row in candidates)


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


def test_carried_move_rejects_gripper_relative_transform_drift():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.phase_a = Mock()
    dispatcher.phase_a.current_workspace = KitchenWorkspace.HOME
    dispatcher.phase_a._move.return_value = {"success": True, "status": "OK"}
    dispatcher.manipulation = Mock()
    dispatcher._held_state = Mock(
        side_effect=[
            {
                "validation_status": "TRUE",
                "relative_position_m": [0.0, 0.0, 0.1],
                "relative_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "validation_status": "TRUE",
                "relative_position_m": [0.02, 0.0, 0.1],
                "relative_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        ]
    )

    result = dispatcher.move(
        KitchenWorkspace.LEFT_SIDE, carrying_object_id="object_1"
    )

    assert result["success"] is False
    assert result["status"] == "OBJECT_DROPPED"
    assert result["relative_position_drift_m"] == 0.02
    assert result["relative_transform_drift_valid"] is False


def test_carried_move_folds_held_payload_when_rotation_requires_compact_pose():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.phase_a = Mock()
    dispatcher.phase_a.current_workspace = KitchenWorkspace.HOME
    dispatcher.phase_a._move.side_effect = [
        RuntimeError("Final base rotation is in collision; move the arm to its compact navigation pose before moving"),
        {"success": True, "status": "OK"},
    ]
    dispatcher.manipulation = Mock()
    dispatcher.manipulation.executor.fold_held_payload_for_navigation.return_value = {
        "performed": True,
        "direct_object_qpos_write": False,
        "grasp_weld_retained": True,
    }
    dispatcher._held_state = Mock(
        side_effect=[
            {"validation_status": "TRUE"},
            {"validation_status": "TRUE"},
        ]
    )
    result = dispatcher.move(
        KitchenWorkspace.RIGHT_SIDE, carrying_object_id="object_1"
    )
    assert result["success"] is True
    assert result["held_navigation_preparation"]["performed"] is True
    dispatcher.manipulation.executor.fold_held_payload_for_navigation.assert_called_once()
    assert dispatcher.phase_a._move.call_count == 2


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
    coffee_target = resolver.resolve("coffee_a", "serving_area")
    soup_target = resolver.resolve("soup_a", "serving_area")
    assert coffee_target.target_position_world_m[0] == -0.15
    assert soup_target.target_position_world_m[0] == -0.15
    assert abs(coffee_target.target_position_world_m[1] + 0.56) < 0.08
    assert abs(soup_target.target_position_world_m[1] + 0.56) < 0.08
    for target, object_id in (
        (coffee_target, "coffee_a"),
        (soup_target, "soup_a"),
    ):
        radius = resolver.rotation_safe_half_extent(object_id)
        x, y = target.target_position_world_m[:2]
        assert 0.25 - abs(x) - radius >= target.edge_margin_m - 1e-9
        assert 0.15 - abs(y + 0.56) - radius >= target.edge_margin_m - 1e-9


def test_serving_allocator_contains_rotated_asymmetric_payload():
    inventory = {
        "objects": [{
            "generic_object_id": "wide_bowl",
            "selected_functions": ["soup_bowl"],
            "observed_centroid_world_m": [0.2, 0.0, 0.63],
            "observed_dimensions_m": {"length": 0.13, "width": 0.12},
        }]
    }
    resolution = {
        "accepted": [{
            "generic_object_id": "wide_bowl",
            "physical_backend_body": "wide_bowl",
        }]
    }
    resolver = KitchenPlacementResolver(Mock(), inventory, resolution)
    target = resolver.resolve("wide_bowl", "serving_area")
    radius = resolver.rotation_safe_half_extent("wide_bowl")
    x, y = target.target_position_world_m[:2]

    assert 0.25 - abs(x) - radius >= 0.012 - 1e-9
    assert 0.15 - abs(y + 0.56) - radius >= 0.012 - 1e-9
    assert target.provenance == "ROTATION_SAFE_OBSERVED_FOOTPRINT_SERVING_ALLOCATOR_V3"


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
