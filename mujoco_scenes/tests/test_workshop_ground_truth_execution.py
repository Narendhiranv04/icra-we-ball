import json
from pathlib import Path

from mujoco_scenes.run_workshop_ground_truth_execution import run_variant
from mujoco_scenes.workshop_execution_handoff import validate_frozen_handoff_suite
from mujoco_scenes.workshop_ground_truth_planner import (
    generate_gt_plan,
    load_action_vocabulary,
    load_variant_specs,
    solve_gt_assignment,
)
from mujoco_scenes.workshop_ground_truth_state import initial_workshop_state, symbolic_preflight
from mujoco_scenes.workshop_scene import WorkshopScene
from mujoco_scenes import workshop_actions


def test_target_joint_api_uses_actual_recessed_hole_not_fixture_origin():
    scene = WorkshopScene(robot="google", variant="F0_MANUAL_FIRST_ONE_REGION")
    target = scene.privileged_get_target_joint_specification()
    assert target["hole_entry_center_world_m"] == [-0.25, 0.48, 0.7190000000000001]
    assert target["seated_fastener_tip_world_m"] == [-0.25, 0.48, 0.6890000000000001]
    assert target["hole_axis_world"] == [0.0, 0.0, 1.0]


def test_action_vocabulary_is_complete():
    assert len(load_action_vocabulary()["operators"]) == 10


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


def test_infeasible_execution_inspects_then_terminates(tmp_path: Path):
    result = run_variant("I1_NO_SCREW", output_root=tmp_path)
    assert result["success"]
    assert result["outcome"] == "INFEASIBLE_CONFIRMED"
