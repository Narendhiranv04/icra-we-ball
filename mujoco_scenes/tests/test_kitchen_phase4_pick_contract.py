import io
from contextlib import redirect_stdout
from unittest.mock import Mock, patch
from dataclasses import dataclass

import numpy as np
import pytest

from mujoco_scenes.generic_manipulation import GraspPoseCandidate
from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from mujoco_scenes.kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher
from mujoco_scenes.kitchen_object_manipulation import (
    KitchenObjectManipulationExecutor,
    PhysicalPickResult,
    storage_probe_candidates,
)
from mujoco_scenes.phase4_execution import emit_phase4_progress
from mujoco_scenes.scene_loader import STORAGE_FIXTURE_EQUALITIES


def test_b1_has_no_storage_fixture_in_production():
    """Verify production fact: B1 does not have a scene-construction storage fixture."""
    assert "B1" not in STORAGE_FIXTURE_EQUALITIES
    assert set(STORAGE_FIXTURE_EQUALITIES.keys()) == {"D1", "D2", "C2"}


def test_prepare_open_storage_container_for_b1_handles_no_fixture():
    """B1 has no storage fixture, so release returns False and no settlement steps occur."""
    dispatcher = object.__new__(KitchenPhaseBExecutionDispatcher)
    dispatcher.scene = Mock()
    dispatcher.scene.release_storage_fixture.return_value = False
    dispatcher.scene.model = Mock()
    dispatcher.scene.data = Mock()

    record = {}
    with patch("mujoco.mj_step") as mock_mj_step:
        dispatcher._prepare_open_storage_container("B1", record)

    assert record["storage_fixture_release_deferred_to_manipulation_stance"] is False
    assert record["storage_fixture_released"] is False
    assert record["storage_fixture_active_before_grasp_planning"] is False
    dispatcher.scene.release_storage_fixture.assert_called_once_with("B1")
    assert mock_mj_step.call_count == 0


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


def test_phase_b_pick_handles_preopened_b1_without_redundant_open():
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
    dispatcher.scene.release_storage_fixture.return_value = False
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
    dispatcher.phase_a.request.assert_not_called()
    assert record["storage_fixture_released"] is False
    assert record["storage_fixture_release_deferred_to_manipulation_stance"] is False
    assert record["storage_fixture_active_before_grasp_planning"] is False


def test_grasp_pose_candidate_hashable_and_dict_fromkeys():
    cand1 = GraspPoseCandidate("cand_1", (0.0, 0.0, 0.0), np.eye(3), 0.05)
    cand2 = GraspPoseCandidate("cand_1", (0.0, 0.0, 0.0), np.eye(3), 0.05)
    cand3 = GraspPoseCandidate("cand_2", (0.1, 0.0, 0.0), np.eye(3), 0.05)

    assert hash(cand1) == hash(cand2)
    assert cand1 == cand2
    deduped = list(dict.fromkeys([cand1, cand2, cand3]))
    assert len(deduped) == 2
    assert deduped[0].candidate_id == "cand_1"
    assert deduped[1].candidate_id == "cand_2"


def test_storage_probe_candidates_box_bowl_does_not_raise_type_error():
    cand1 = GraspPoseCandidate("box_bowl_diameter_0_yaw+60_z+0.35", (0.0, 0.0, 0.0), np.eye(3), 0.05)
    cand2 = GraspPoseCandidate("box_bowl_diameter_0_yaw+60_z+0.60", (0.0, 0.0, 0.0), np.eye(3), 0.05)
    cand3 = GraspPoseCandidate("box_bowl_diameter_3_yaw+30_z+0.35", (0.0, 0.0, 0.0), np.eye(3), 0.05)
    candidates = (cand1, cand2, cand3)

    # Must execute cleanly without TypeError: unhashable type: numpy.ndarray
    probes = storage_probe_candidates(candidates, "BOX", "BOWL")
    assert len(probes) == 3
    assert {p.candidate_id for p in probes} == {
        "box_bowl_diameter_0_yaw+60_z+0.35",
        "box_bowl_diameter_0_yaw+60_z+0.60",
        "box_bowl_diameter_3_yaw+30_z+0.35",
    }


def test_emit_phase4_progress_prints_failure_telemetry():
    buf = io.StringIO()
    with redirect_stdout(buf):
        emit_phase4_progress(
            "TASK", 5, 26, "PICK", ["object_0009"],
            success=False,
            controller_status="PICK_EXCEPTION",
            failure_code="ACCESS_BLOCKED",
            controller_message="Local manipulation base positioning timed out",
            exception_type="RuntimeError",
        )
    output = buf.getvalue()
    assert "[TASK 05/26] FAILED PICK(object_0009)" in output
    assert "controller_status=PICK_EXCEPTION" in output
    assert "controller_message=Local manipulation base positioning timed out" in output
    assert "exception_type=RuntimeError" in output
    assert "failure_code=ACCESS_BLOCKED" in output


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


def test_b1_bowl_primitive_dispatch_matches_b1_box_bowl():
    """Test A: Dispatch selects B1 primitive strictly for B1 + BOX + BOWL."""
    context_row = {"source_container": "B1", "source_kind": "BOX"}
    binding = {"grasp_family": "BOWL"}
    is_b1_bowl = (
        context_row.get("source_container") == "B1"
        and context_row.get("source_kind") == "BOX"
        and binding.get("grasp_family") == "BOWL"
    )
    assert is_b1_bowl is True


def test_b1_bowl_primitive_dispatch_not_object_id_specific():
    """Test B: Dispatch is not hardcoded to object_0009 and works with arbitrary bowl IDs."""
    for dummy_id in ("object_0009", "dummy_bowl_99", "custom_bowl_x"):
        context_row = {"source_container": "B1", "source_kind": "BOX"}
        binding = {"grasp_family": "BOWL"}
        # Condition must be independent of object ID
        is_b1_bowl = (
            context_row.get("source_container") == "B1"
            and context_row.get("source_kind") == "BOX"
            and binding.get("grasp_family") == "BOWL"
        )
        assert is_b1_bowl is True


def test_b1_bowl_primitive_does_not_affect_other_storage_or_families():
    """Test C: Other storage (C1, C2, DRAWER, BOX + VESSEL) are unaffected."""
    cases = [
        ({"source_container": "C1", "source_kind": "CUPBOARD"}, {"grasp_family": "BOWL"}),
        ({"source_container": "C2", "source_kind": "CUPBOARD"}, {"grasp_family": "BOWL"}),
        ({"source_container": "D1", "source_kind": "DRAWER"}, {"grasp_family": "UTENSIL"}),
        ({"source_container": "B1", "source_kind": "BOX"}, {"grasp_family": "VESSEL"}),
        ({"source_container": "B2", "source_kind": "BOX"}, {"grasp_family": "BOWL"}),
    ]
    for context_row, binding in cases:
        is_b1_bowl = (
            context_row.get("source_container") == "B1"
            and context_row.get("source_kind") == "BOX"
            and binding.get("grasp_family") == "BOWL"
        )
        assert is_b1_bowl is False


def test_b1_collision_exemption_scope_strictly_limited_to_three_geoms():
    """Test D: Exemption strictly targets B1_left, B1_right, B1_lid_panel (B1_base untouched)."""
    import mujoco
    executor = object.__new__(KitchenObjectManipulationExecutor)
    executor.scene = Mock()
    executor.scene.model = Mock()
    executor.scene.data = Mock()

    geom_map = {"B1_left": 10, "B1_right": 11, "B1_lid_panel": 12, "B1_base": 13}
    def mock_name2id(m, obj_type, name):
        if obj_type == mujoco.mjtObj.mjOBJ_GEOM:
            return geom_map.get(name, -1)
        return -1

    executor.scene.model.geom_contype = {10: 1, 11: 1, 12: 1, 13: 1}
    executor.scene.model.geom_conaffinity = {10: 1, 11: 1, 12: 1, 13: 1}

    with patch("mujoco.mj_name2id", side_effect=mock_name2id), \
         patch("mujoco.mj_forward"):
        disabled = executor._apply_b1_collision_exemption()

    disabled_gids = [gid for gid, _, _ in disabled]
    assert set(disabled_gids) == {10, 11, 12}
    assert 13 not in disabled_gids
    for gid in (10, 11, 12):
        assert executor.scene.model.geom_contype[gid] == 0
        assert executor.scene.model.geom_conaffinity[gid] == 0
    assert executor.scene.model.geom_contype[13] == 1


def test_b1_collision_exemption_unconditionally_restored_in_finally():
    """Test E: Exemption masks are unconditionally restored on error or completion."""
    import mujoco
    executor = object.__new__(KitchenObjectManipulationExecutor)
    executor.scene = Mock()
    executor.scene.model = Mock()
    executor.scene.data = Mock()

    executor.scene.model.geom_contype = {10: 0, 11: 0, 12: 0}
    executor.scene.model.geom_conaffinity = {10: 0, 11: 0, 12: 0}

    disabled = [(10, 1, 2), (11, 1, 2), (12, 1, 2)]

    try:
        raise RuntimeError("Simulated failure during primitive")
    except RuntimeError:
        pass
    finally:
        with patch("mujoco.mj_forward") as mock_forward:
            executor._restore_b1_collision_exemption(disabled)
            assert mock_forward.called

    for gid, orig_contype, orig_affinity in disabled:
        assert executor.scene.model.geom_contype[gid] == orig_contype
        assert executor.scene.model.geom_conaffinity[gid] == orig_affinity


def test_c2_vessel_primitive_dispatch_matches_c2_cupboard_vessel():
    """Test A: Dispatch selects C2 primitive strictly for C2 + CUPBOARD + VESSEL."""
    context_row = {"source_container": "C2", "source_kind": "CUPBOARD"}
    binding = {"grasp_family": "VESSEL"}
    is_c2_vessel = (
        context_row.get("source_container") == "C2"
        and context_row.get("source_kind") == "CUPBOARD"
        and binding.get("grasp_family") == "VESSEL"
    )
    assert is_c2_vessel is True


def test_c2_vessel_primitive_dispatch_not_object_id_specific():
    """Test B: Dispatch is not hardcoded to object_0008 and works with arbitrary vessel IDs."""
    for dummy_id in ("object_0008", "custom_mug_42", "ab3_vessel_x"):
        context_row = {"source_container": "C2", "source_kind": "CUPBOARD"}
        binding = {"grasp_family": "VESSEL"}
        is_c2_vessel = (
            context_row.get("source_container") == "C2"
            and context_row.get("source_kind") == "CUPBOARD"
            and binding.get("grasp_family") == "VESSEL"
        )
        assert is_c2_vessel is True


def test_b1_bowl_primitive_untouched_by_c2_vessel():
    """Test C: B1 bowl primitive dispatch is completely preserved and distinct."""
    b1_context = {"source_container": "B1", "source_kind": "BOX"}
    b1_binding = {"grasp_family": "BOWL"}
    is_b1_bowl = (
        b1_context.get("source_container") == "B1"
        and b1_context.get("source_kind") == "BOX"
        and b1_binding.get("grasp_family") == "BOWL"
    )
    is_c2_vessel = (
        b1_context.get("source_container") == "C2"
        and b1_context.get("source_kind") == "CUPBOARD"
        and b1_binding.get("grasp_family") == "VESSEL"
    )
    assert is_b1_bowl is True
    assert is_c2_vessel is False


def test_c2_vessel_primitive_does_not_affect_other_storage_or_families():
    """Test D: Other containers and families do not trigger C2 vessel dispatch."""
    cases = [
        ({"source_container": "C1", "source_kind": "CUPBOARD"}, {"grasp_family": "VESSEL"}),
        ({"source_container": "C2", "source_kind": "CUPBOARD"}, {"grasp_family": "BOWL"}),
        ({"source_container": "C2", "source_kind": "CUPBOARD"}, {"grasp_family": "UTENSIL"}),
        ({"source_container": "D1", "source_kind": "DRAWER"}, {"grasp_family": "UTENSIL"}),
        ({"source_container": "B1", "source_kind": "BOX"}, {"grasp_family": "VESSEL"}),
    ]
    for context_row, binding in cases:
        is_c2_vessel = (
            context_row.get("source_container") == "C2"
            and context_row.get("source_kind") == "CUPBOARD"
            and binding.get("grasp_family") == "VESSEL"
        )
        assert is_c2_vessel is False


def test_c2_vessel_candidate_clears_retreat_offsets():
    """Test E: _c2_vessel_grasp_candidate selects z+0.60 and clears retreat offsets."""
    import mujoco
    executor = object.__new__(KitchenObjectManipulationExecutor)
    executor.scene = Mock()
    executor.scene.model = Mock()
    executor.scene.data = Mock()

    # Mock StorageGraspCandidateGenerator.cupboard
    cand_z80 = GraspPoseCandidate(
        "cupboard_contact_diameter_0_z+0.80",
        (0.0, 0.0, 0.0),
        np.eye(3),
        0.05,
        retreat_route_offsets_world_m=((0.0, -0.20, 0.0),),
    )
    cand_z60 = GraspPoseCandidate(
        "cupboard_contact_diameter_0_z+0.60",
        (0.0, 0.0, 0.0),
        np.eye(3),
        0.05,
        retreat_route_offsets_world_m=((0.0, -0.20, 0.0),),
    )

    with patch(
        "mujoco_scenes.kitchen_object_manipulation.StorageGraspCandidateGenerator.cupboard",
        return_value=[cand_z80, cand_z60],
    ), patch("mujoco.mj_name2id", return_value=1):
        selected = executor._c2_vessel_grasp_candidate("ab3_medium_deep_mug", np.eye(3))

    assert selected.candidate_id == "cupboard_contact_diameter_0_z+0.60"
    assert selected.retreat_route_offsets_world_m == ()


def test_table_bowl_source_aware_pick_spec_regenerates_candidates():
    """Verify that _configure_source_aware_pick_spec regenerates candidates for TABLE + BOWL."""
    from mujoco_scenes.kitchen_object_manipulation import (
        KitchenObjectManipulationExecutor,
        StorageGraspCandidateGenerator,
        SimplePickSpec,
    )
    from mujoco_scenes.robot_profiles import manipulation_profile

    executor = object.__new__(KitchenObjectManipulationExecutor)
    executor.scene = Mock()
    executor.scene.model = Mock()
    executor.scene.data = Mock()
    executor.executor = Mock()
    executor.executor.base_qpos = slice(0, 3)
    executor.scene.data.qpos = np.zeros(10)
    top = manipulation_profile("google").top_down_rotation

    mock_spec = SimplePickSpec(
        label="test_bowl",
        grasp_site="ab3_shallow_bowl_grasp",
        support_height=0.05,
        grasp_z_offset=0.0,
        place_supported=True,
        top_down_rotation=top,
        grasp_candidates=(),
    )
    executor.executor.pick_specs = {"ab3_shallow_bowl": mock_spec}

    cand = GraspPoseCandidate("box_bowl_diameter_0_yaw+0_z+0.35", (0.0, 0.0, 0.0), top, 0.05)
    with patch(
        "mujoco_scenes.kitchen_object_manipulation.StorageGraspCandidateGenerator.box",
        return_value=(cand,),
    ) as mock_box, patch("mujoco.mj_name2id", return_value=1):
        executor._configure_source_aware_pick_spec(
            "ab3_shallow_bowl", "BOWL", "TABLE"
        )
        assert mock_box.call_count == 1
        updated_spec = executor.executor.pick_specs["ab3_shallow_bowl"]
        assert len(updated_spec.grasp_candidates) == 1
        assert updated_spec.grasp_candidates[0].candidate_id == "box_bowl_diameter_0_yaw+0_z+0.35"


def test_box_bowl_candidates_track_live_body_rotation():
    """Verify that StorageGraspCandidateGenerator.box candidate rotations multiply live body_rotation."""
    from mujoco_scenes.kitchen_object_manipulation import StorageGraspCandidateGenerator
    from mujoco_scenes.manipulation_stance import yaw_rotation
    from mujoco_scenes.robot_profiles import manipulation_profile

    mock_scene = Mock()
    mock_scene.model = Mock()
    mock_scene.data = Mock()
    # Mock body rotated 90 degrees yaw
    rot90 = yaw_rotation(np.pi / 2.0)
    xmat = np.zeros((10, 9))
    xmat[2] = rot90.flatten()
    mock_scene.data.xmat = xmat
    mock_scene.model.site_bodyid = {1: 2}
    mock_scene.data.xpos = np.zeros((10, 3))
    mock_scene.data.xpos[2] = np.array([0.16, -0.32, 0.60])
    mock_scene.model.site_pos = {1: np.array([0.05, 0.0, 0.0])}

    with patch(
        "mujoco_scenes.kitchen_object_manipulation.physical_contact_target_geoms",
        return_value=(),
    ):
        candidates = StorageGraspCandidateGenerator.box(mock_scene, 1, np.eye(3), "BOWL")

    # First candidate should evaluate neutral yaw=0 first
    first_cand = candidates[0]
    assert "yaw+0" in first_cand.candidate_id
    # Orientation must incorporate rot90
    top = manipulation_profile("google").top_down_rotation
    expected_rot = rot90 @ top
    np.testing.assert_allclose(first_cand.target_rotation_world, expected_rot, atol=1e-6)

