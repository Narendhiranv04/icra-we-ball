from unittest.mock import Mock, patch
from dataclasses import dataclass

import pytest

from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from mujoco_scenes.kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher
from mujoco_scenes.kitchen_object_manipulation import (
    KitchenObjectManipulationExecutor,
    PhysicalPickResult,
)


def test_prepare_open_storage_container_releases_fixture_for_box():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.scene = Mock()
    dispatcher.scene.release_storage_fixture.return_value = True
    dispatcher.scene.model = Mock()
    dispatcher.scene.data = Mock()

    record = {}
    with patch("mujoco.mj_step") as mock_mj_step:
        dispatcher._prepare_open_storage_container("B1", record)

    assert record["storage_fixture_release_deferred_to_manipulation_stance"] is False
    assert record["storage_fixture_released"] is True
    assert record["storage_fixture_active_before_grasp_planning"] is False
    dispatcher.scene.release_storage_fixture.assert_called_once_with("B1")
    assert mock_mj_step.call_count == 120


@pytest.mark.parametrize("container", ["C2", "D1", "D2"])
def test_prepare_open_storage_container_defers_fixture_release_for_c2_and_drawers(container: str):
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.scene = Mock()
    dispatcher.scene.release_storage_fixture.return_value = True

    record = {}
    with patch("mujoco.mj_step") as mock_mj_step:
        dispatcher._prepare_open_storage_container(container, record)

    assert record["storage_fixture_release_deferred_to_manipulation_stance"] is True
    assert record["storage_fixture_released"] is False
    assert record["storage_fixture_active_before_grasp_planning"] is True
    dispatcher.scene.release_storage_fixture.assert_not_called()
    assert mock_mj_step.call_count == 0


def test_phase_b_pick_prepares_storage_container_when_newly_opened():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.inventory_by_id = {
        "object_0009": {
            "source_context": {
                "source_container": "B1",
                "required_workspace": KitchenWorkspace.RIGHT_SIDE.value,
            }
        }
    }
    dispatcher.binding_by_id = {"object_0009": {"physical_backend_body": "ab3_deep_bowl"}}
    dispatcher.phase_a = Mock()
    dispatcher.phase_a.current_workspace = KitchenWorkspace.RIGHT_SIDE
    dispatcher.physically_open_containers = Mock(return_value=set())
    dispatcher.phase_a.request.return_value = {"success": True, "action": "OPEN", "arguments": ["B1"]}

    dispatcher.scene = Mock()
    dispatcher.scene.release_storage_fixture.return_value = True
    dispatcher.scene.model = Mock()
    dispatcher.scene.data = Mock()

    pick_result = PhysicalPickResult(
        "object_0009", "ab3_deep_bowl", {}, KitchenWorkspace.RIGHT_SIDE.value, "BOWL",
        True, "PICK_COMPLETED", "PICK_COMPLETED", "ok", 100, 1.0, True, (), (), None, None, False, None
    )
    dispatcher.manipulation = Mock()
    dispatcher.manipulation.pick.return_value = pick_result
    dispatcher.manipulation.executor = Mock()
    dispatcher.manipulation.executor.storage_fixture_release_telemetry = None

    with patch("mujoco.mj_step") as mock_mj_step:
        record = dispatcher.pick("object_0009")

    assert record["success"] is True
    assert record["storage_fixture_released"] is True
    assert record["storage_fixture_release_deferred_to_manipulation_stance"] is False
    assert record["storage_fixture_active_before_grasp_planning"] is False
    # Called during _prepare_open_storage_container, and optionally post-pick cleanup
    assert dispatcher.scene.release_storage_fixture.call_count >= 1
    dispatcher.phase_a.request.assert_called_once_with("OPEN", "B1", execute=True)


def test_phase_b_pick_prepares_storage_container_when_already_opened():
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.inventory_by_id = {
        "object_0009": {
            "source_context": {
                "source_container": "B1",
                "required_workspace": KitchenWorkspace.RIGHT_SIDE.value,
            }
        }
    }
    dispatcher.binding_by_id = {"object_0009": {"physical_backend_body": "ab3_deep_bowl"}}
    dispatcher.phase_a = Mock()
    dispatcher.phase_a.current_workspace = KitchenWorkspace.RIGHT_SIDE
    # Container B1 is ALREADY open (e.g. opened during inspection)
    dispatcher.physically_open_containers = Mock(return_value={"B1"})

    dispatcher.scene = Mock()
    dispatcher.scene.release_storage_fixture.return_value = True
    dispatcher.scene.model = Mock()
    dispatcher.scene.data = Mock()

    pick_result = PhysicalPickResult(
        "object_0009", "ab3_deep_bowl", {}, KitchenWorkspace.RIGHT_SIDE.value, "BOWL",
        True, "PICK_COMPLETED", "PICK_COMPLETED", "ok", 100, 1.0, True, (), (), None, None, False, None
    )
    dispatcher.manipulation = Mock()
    dispatcher.manipulation.pick.return_value = pick_result
    dispatcher.manipulation.executor = Mock()
    dispatcher.manipulation.executor.storage_fixture_release_telemetry = None

    with patch("mujoco.mj_step") as mock_mj_step:
        record = dispatcher.pick("object_0009")

    assert record["success"] is True
    assert record.get("redundant_open_omitted") is True
    # Phase A OPEN should NOT have been requested
    dispatcher.phase_a.request.assert_not_called()
    # But storage fixture MUST be released and settled via _prepare_open_storage_container
    assert record["storage_fixture_released"] is True
    assert record["storage_fixture_release_deferred_to_manipulation_stance"] is False
    assert record["storage_fixture_active_before_grasp_planning"] is False
    assert dispatcher.scene.release_storage_fixture.call_count >= 1


def test_gt_dispatcher_preserves_underlying_pick_exception():
    dispatcher = object.__new__(KitchenGroundTruthExecutionDispatcher)
    dispatcher._settle_navigation_posture = Mock()
    dispatcher._allow_served_payloads_for_next_motion = Mock()
    dispatcher.assisted_suite = False
    dispatcher.allow_assisted_pick_recovery = True

    dispatcher.phase_b = Mock()
    dispatcher.phase_b.pick.side_effect = RuntimeError("Local manipulation base positioning timed out")

    dispatcher._benchmark_pick_recovery_evidence = Mock(return_value={
        "accepted": False,
        "reason": "PRECLOSE_EXCEEDED",
    })

    result = dispatcher.pick("object_0009")

    assert result["success"] is False
    assert result["status"] == "PICK_EXCEPTION"
    assert result["failure_code"] == "ACCESS_BLOCKED"
    assert result["controller_status"] == "PICK_EXCEPTION"
    assert result["controller_message"] == "Local manipulation base positioning timed out"
    assert result["exception_type"] == "RuntimeError"
    assert result["stage"] == "PHASE_B_PICK"


def test_gt_dispatcher_preserves_controller_status_and_message_on_recovery_rejection():
    dispatcher = object.__new__(KitchenGroundTruthExecutionDispatcher)
    dispatcher._settle_navigation_posture = Mock()
    dispatcher._allow_served_payloads_for_next_motion = Mock()
    dispatcher.assisted_suite = False
    dispatcher.allow_assisted_pick_recovery = True

    dispatcher.phase_b = Mock()
    dispatcher.phase_b.pick.return_value = {
        "success": False,
        "status": "DIRECT_GRASP_INFEASIBLE",
        "failure_code": "DIRECT_GRASP_INFEASIBLE",
        "message": "IK candidate rejected due to collision",
    }

    dispatcher._benchmark_pick_recovery_evidence = Mock(return_value={
        "accepted": False,
        "reason": "PRECLOSE_EXCEEDED",
    })

    result = dispatcher.pick("object_0009")

    assert result["success"] is False
    assert result["controller_status"] == "DIRECT_GRASP_INFEASIBLE"
    assert result["controller_message"] == "IK candidate rejected due to collision"
    assert result["failure_code"] == "ACCESS_BLOCKED"


def test_source_aware_pick_spec_retains_default_tolerance_for_box_bowl():
    @dataclass
    class DummyPickSpec:
        grasp_z_offset: float = 0.05
        final_tracking_tolerance: float = 0.02

    spec = DummyPickSpec()

    # Reverted condition: only CUPBOARD UTENSIL gets 0.500; BOX BOWL gets default spec tolerance
    def compute_final_tracking_tolerance(source_kind: str, family: str) -> float:
        return (
            0.500
            if source_kind == "CUPBOARD" and family == "UTENSIL"
            else spec.final_tracking_tolerance
        )

    assert compute_final_tracking_tolerance("BOX", "BOWL") == 0.02
    assert compute_final_tracking_tolerance("CUPBOARD", "UTENSIL") == 0.500
    assert compute_final_tracking_tolerance("DRAWER", "UTENSIL") == 0.02
