import json
from pathlib import Path

import mujoco

from mujoco_scenes.run_workshop_ground_truth_execution import run_variant
from mujoco_scenes.workshop_execution_handoff import validate_frozen_handoff_suite
from mujoco_scenes.workshop_ground_truth_planner import (
    generate_gt_plan,
    load_action_vocabulary,
    load_variant_specs,
    solve_gt_assignment,
)
from mujoco_scenes.workshop_ground_truth_state import initial_workshop_state, symbolic_preflight
from mujoco_scenes.workshop_scene import WORKSHOP_CONTAINER_JOINTS, WorkshopScene
from mujoco_scenes import workshop_actions


def test_target_joint_api_uses_actual_recessed_hole_not_fixture_origin():
    scene = WorkshopScene(robot="google", variant="F0_MANUAL_FIRST_ONE_REGION")
    target = scene.privileged_get_target_joint_specification()
    assert target["hole_entry_center_world_m"] == [-0.44000000000000006, 0.32, 0.7190000000000001]
    assert target["seated_fastener_tip_world_m"] == [-0.44000000000000006, 0.32, 0.6890000000000001]
    assert target["hole_axis_world"] == [0.0, 0.0, 1.0]


def test_action_vocabulary_is_complete():
    assert set(load_action_vocabulary()["operators"]) == {"OPEN", "PICK", "PLACE", "SCREW"}


def test_drawers_start_closed_and_only_open_through_open_action():
    for variant_id in load_variant_specs():
        scene = WorkshopScene(robot="none", variant=variant_id)
        for region_id in ("LEFT_DRAWER", "RIGHT_DRAWER"):
            mechanism = WORKSHOP_CONTAINER_JOINTS[region_id]
            joint_id = mujoco.mj_name2id(
                scene.model, mujoco.mjtObj.mjOBJ_JOINT, mechanism["joint"]
            )
            qpos = scene.data.qpos[scene.model.jnt_qposadr[joint_id]]
            assert abs(qpos) < 1e-5, (variant_id, region_id, qpos)
            assert not scene.state.container_open_state[region_id]

        opened = scene.open_container("LEFT_DRAWER")
        assert opened["newly_opened"]
        left_joint = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            WORKSHOP_CONTAINER_JOINTS["LEFT_DRAWER"]["joint"],
        )
        right_joint = mujoco.mj_name2id(
            scene.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            WORKSHOP_CONTAINER_JOINTS["RIGHT_DRAWER"]["joint"],
        )
        assert scene.data.qpos[scene.model.jnt_qposadr[left_joint]] >= 0.20
        assert abs(scene.data.qpos[scene.model.jnt_qposadr[right_joint]]) < 1e-5

        scene.reset()
        assert abs(scene.data.qpos[scene.model.jnt_qposadr[left_joint]]) < 1e-5
        assert not scene.state.container_open_state["LEFT_DRAWER"]


def test_workshop_viewer_delegates_to_actions_panel(monkeypatch):
    scene = WorkshopScene(robot="google", variant="F0_MANUAL_FIRST_ONE_REGION")
    called = {}

    def fake_panel(received_scene, camera):
        called.update(scene=received_scene, camera=camera)

    monkeypatch.setattr(workshop_actions, "launch_workshop_action_viewer", fake_panel)
    scene.launch_viewer("free", actions_panel=True)
    assert called == {"scene": scene, "camera": "free"}


def test_all_redesigned_plans_pass_symbolic_preflight():
    for variant_id, spec in load_variant_specs().items():
        assignment = solve_gt_assignment(variant_id)
        plan = generate_gt_plan(assignment)
        result = symbolic_preflight(initial_workshop_state(spec["storage_contents"]), plan, assignment)
        assert result["success"], (variant_id, result)


def test_plans_use_one_open_per_inspected_region_and_direct_repair_flow():
    for variant_id, spec in load_variant_specs().items():
        plan = generate_gt_plan(solve_gt_assignment(variant_id))
        operators = [action["operator"] for action in plan]
        assert set(operators) <= {"OPEN", "PICK", "PLACE", "SCREW"}
        assert [action["arguments"][0] for action in plan if action["operator"] == "OPEN"] == spec["expected_inspection_regions"]
        if spec["intended_outcome"] == "INFEASIBLE":
            assert operators == ["OPEN"] * len(spec["expected_inspection_regions"])
            continue
        screw_place = next(action for action in plan if action["operator"] == "PLACE" and action["arguments"][0] == "workshop_medium_phillips_screw")
        assert screw_place["arguments"] == ["workshop_medium_phillips_screw", "workshop_frame_joint"]
        screw = next(action for action in plan if action["operator"] == "SCREW")
        assert screw["arguments"][1:] == ["workshop_medium_phillips_screw", "workshop_frame_joint"]


def test_frozen_yoloworld_l_handoff_is_exact_for_all_variants():
    result = validate_frozen_handoff_suite()
    assert result["passed"]
    assert result["exact_matches"] == result["total_variants"] == 10


def test_feasible_assisted_execution_reaches_measured_terminal_state(tmp_path: Path):
    result = run_variant("F0_MANUAL_FIRST_ONE_REGION", output_root=tmp_path)
    assert result["success"]
    assert result["outcome"] == "SUCCESS"
    assert result["actions_completed"] == result["total_actions"]
    trace = json.loads(
        (tmp_path / "F0_MANUAL_FIRST_ONE_REGION" / "execution_trace.json")
        .read_text(encoding="utf-8")
    )
    open_result = trace["actions"][0]["physical_result"]
    assert open_result["operator"] == "OPEN"
    assert open_result["robot_actuated_motion"]
    assert open_result["storage_motion_source"] == "GOOGLE_ROBOT_HANDLE_MANIPULATION"
    assert open_result["articulation"]["physical_handle_grasp_constraint_used"]
    assert not open_result["articulation"]["direct_container_actuator_used"]
    assert open_result["articulation"]["joint_position_during_handle_tracking"] >= 0.20
    place_result = next(
        step["physical_result"]
        for step in trace["actions"]
        if step["action"]["operator"] == "PLACE"
        and step["action"]["arguments"][1] == "workshop_frame_joint"
    )
    assert place_result["head_above_tip_m"] >= 0.040
    assert place_result["vertical_axis_error_rad"] <= 0.03
    assert 0.012 <= place_result["insertion_depth_m"] <= 0.018
    assert 0.012 <= place_result["remaining_drive_depth_m"] <= 0.018
    assert len(place_result["reorientation"]["continuous_reorientation_steps"]) == 16
    assert place_result["reorientation"]["gripper_remained_closed_during_reorientation"]
    assert place_result["reorientation"]["vertical_axis_world"] == [0.0, 0.0, -1.0]
    screw_result = next(
        step["physical_result"]
        for step in trace["actions"]
        if step["action"]["operator"] == "SCREW"
    )
    assert screw_result["reorientation"][
        "gripper_remained_closed_during_reorientation"
    ]
    assert len(screw_result["reorientation"][
        "continuous_reorientation_steps"
    ]) == 16
    assert screw_result["gripper_closed_during_contact_alignment"]
    assert len(screw_result["continuous_contact_alignment_steps"]) == 8
    assert all(
        step["grasp_constraint_active"]
        for step in screw_result["continuous_contact_alignment_steps"]
    )
    assert screw_result["driver_tip_to_head_error_m"] <= 0.008
    assert screw_result["continuous_drive_duration_s"] == 4.0
    assert screw_result["robot_followed_axial_descent"]
    assert 0.012 <= screw_result["axial_advance_m"] <= 0.018
    assert screw_result["driver_grasp_retained_during_retraction"]
    assert screw_result["driver_post_drive_lift_m"] >= 0.10
    returned_driver = trace["actions"][-1]["physical_result"]
    assert returned_driver["placement_reorientation"][
        "gripper_remained_closed_during_reorientation"
    ]
    pick_results = [
        step["physical_result"]
        for step in trace["actions"]
        if step["action"]["operator"] == "PICK"
    ]
    assert pick_results
    for physical in pick_results:
        assert physical["contact_grasp"]["bilateral_contact_confirmed"]
        assert physical["attachment"]["bilateral_contact_confirmed"]
        assert physical["attachment"]["attachment_translation_snap_m"] <= 0.004


def test_infeasible_execution_inspects_all_regions_without_termination(tmp_path: Path):
    result = run_variant("I1_NO_SCREW", output_root=tmp_path)
    assert result["success"]
    assert result["outcome"] == "INFEASIBLE_CONFIRMED"
