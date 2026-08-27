"""Unit and integration tests for Kitchen Ground-Truth Oracle execution.

Tests cover:
- Oracle world-state model and state transitions
- Tool reusability (sequential STIRs, PLACE, repick)
- Storage container access and relocation to countertop
- Infeasibility confirmation logic
- Deterministic assignment solver (coffee cover, soup bipartite matching, distributed tools)
- Five-camera discovery, rendering consistency, and mosaic composition
- Video recording smoke test
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest
import numpy as np

from mujoco_scenes.kitchen_ground_truth_planner import (
    generate_ground_truth_plan,
    solve_ground_truth_assignment,
)
from mujoco_scenes.kitchen_ground_truth_recorder import (
    FIVE_PROJECT_CAMERAS,
    KitchenGroundTruthRecorder,
    create_camera_manifest,
)
from mujoco_scenes.kitchen_ground_truth_state import (
    OracleWorldState,
    StatePreconditionError,
    initialize_oracle_world_state,
    run_symbolic_preflight,
)
from mujoco_scenes.run_kitchen_ground_truth_execution import (
    discover_variant_names,
    load_variants_config,
)
from mujoco_scenes.kitchen_ground_truth_execution import (
    KitchenGroundTruthExecutionDispatcher,
)
from mujoco_scenes.scene_loader import KitchenScene


def test_assisted_pick_recovery_activates_matching_payload_weld(monkeypatch):
    """A declared fallback PICK must be a live, exclusive held state.

    This is the regression for F2's old POUR_SOURCE_NOT_HELD failure: the old
    fallback changed the executor's Python state but did not select and enable
    the object-specific MuJoCo equality.
    """
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F2",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(
        scene, "F2_DISTRIBUTED_COFFEE_TWO", "FEASIBLE"
    )
    dispatcher = KitchenGroundTruthExecutionDispatcher(scene, assignment)
    source = assignment.sources["water_source"]
    monkeypatch.setattr(
        dispatcher.phase_b,
        "pick",
        lambda _object_id: {"success": False, "status": "FORCED_TEST_MISS"},
    )

    result = dispatcher.pick(source)

    assert result["success"]
    assert result["status"] == "ASSISTED_PICK_WELD_VERIFIED"
    assert result["assisted_execution"] is True
    assert result["assistance_reason"] == "FORCED_TEST_MISS"
    assert result["held_state"]["validation_status"] == "TRUE"
    assert result["held_state"]["exclusive_payload_weld"] is True


def test_strict_pick_does_not_teleport_payload_after_physical_miss(monkeypatch):
    """Strict final-paper execution must expose a grasp miss, not hide it."""
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F2",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(
        scene, "F2_DISTRIBUTED_COFFEE_TWO", "FEASIBLE"
    )
    dispatcher = KitchenGroundTruthExecutionDispatcher(
        scene, assignment, allow_assisted_pick_recovery=False
    )
    monkeypatch.setattr(
        dispatcher.phase_b,
        "pick",
        lambda _object_id: {"success": False, "status": "FORCED_TEST_MISS"},
    )

    result = dispatcher.pick(assignment.sources["water_source"])

    assert not result["success"]
    assert result["status"] == "FORCED_TEST_MISS"
    assert result.get("assisted_execution", False) is False
    assert result.get("direct_payload_pose_write", False) is False


def test_strict_stirrer_pick_does_not_teleport_payload_after_physical_miss(monkeypatch):
    """The reusable spoon must obey the same strict PICK contract."""
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F4",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(
        scene, "F4_TOOLS_IN_DRAWERS", "FEASIBLE"
    )
    dispatcher = KitchenGroundTruthExecutionDispatcher(
        scene, assignment, allow_assisted_pick_recovery=False
    )
    monkeypatch.setattr(
        dispatcher.phase_b,
        "pick",
        lambda _object_id: {"success": False, "status": "FORCED_TEST_MISS"},
    )

    result = dispatcher.pick("s1i_final_long_narrow_spoon")

    assert not result["success"]
    assert result["status"] == "FORCED_TEST_MISS"
    assert result.get("assisted_execution", False) is False
    assert result.get("direct_payload_pose_write", False) is False


def test_failed_physical_stir_is_not_promoted_to_success():
    source = Path("mujoco_scenes/kitchen_ground_truth_execution.py").read_text()
    stir = source[
        source.index("def stir(self, tool_id: str"):
        source.index("def place_serving_utensil", source.index("def stir(self, tool_id: str"))
    ]

    assert 'record["success"] = True' not in stir
    assert 'record["status"] = "STIR_MOTION_VERIFIED"' not in stir


def test_normal_runner_disables_pose_write_recovery():
    source = Path("mujoco_scenes/run_kitchen_ground_truth_execution.py").read_text()
    construction = source[
        source.index("dispatcher = KitchenGroundTruthExecutionDispatcher("):
        source.index("# Initial frame capture")
    ]

    assert "allow_assisted_pick_recovery=assisted_suite" in construction


def test_hidden_targets_use_physically_validated_role_slots():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F0",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(scene, "F0_ALL_VISIBLE", "FEASIBLE")
    dispatcher = KitchenGroundTruthExecutionDispatcher(scene, assignment)

    assert dispatcher._get_candidate_staging_spots("ab3_medium_deep_mug")[0] == (-0.35, -0.34)
    assert dispatcher._get_candidate_staging_spots("ab3_deep_bowl")[0] == (0.25, -0.10)


def test_hidden_soup_first_serving_order_reserves_space_for_all_targets():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F2",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(
        scene, "F2_HIDDEN_SOUP_BOWL", "FEASIBLE"
    )
    dispatcher = KitchenGroundTruthExecutionDispatcher(scene, assignment)
    resolver = dispatcher.phase_b.manipulation.placement_resolver

    order = (
        "ab3_deep_bowl",
        "ab3_narrow_deep_cup",
        "ab3_medium_deep_mug",
        "ab3_shallow_bowl",
    )
    targets = []
    for object_id in order:
        target = resolver.resolve(object_id, "serving_area")
        resolver.record_successful_serving_placement(object_id, target)
        targets.append(target)

    assert len(targets) == 4
    # Hidden storage must not swap the canonical K1/K2 serving layout.  The
    # large deep bowl always uses the left soup slot, away from the mug.
    assert targets[0].target_position_world_m[0] == pytest.approx(-0.15)


@pytest.mark.parametrize(
    ("variant", "code"),
    [
        ("F0_ALL_VISIBLE", "F0"),
        ("F1_HIDDEN_COFFEE_VESSEL", "F1"),
        ("F2_HIDDEN_SOUP_BOWL", "F2"),
        ("F3_HIDDEN_VESSELS_MIXED", "F3"),
        ("F4_TOOLS_IN_DRAWERS", "F4"),
        ("F5_FULL_DISTRIBUTED_SEARCH", "F5"),
    ],
)
def test_every_feasible_variant_uses_the_same_serving_layout(variant, code):
    scene = KitchenScene(
        f"S1_integrated_kitchen_object_function_feasibility_{code}",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(scene, variant, "FEASIBLE")
    dispatcher = KitchenGroundTruthExecutionDispatcher(scene, assignment)

    assert dispatcher.phase_b.manipulation.placement_resolver.serving_slot_by_id == {
        "ab3_narrow_deep_cup": (-0.15, -0.48),
        "ab3_medium_deep_mug": (0.15, -0.48),
        "ab3_deep_bowl": (-0.15, -0.64),
        "ab3_shallow_bowl": (0.15, -0.64),
    }


def test_kettle_and_jar_pours_preserve_pick_pose_and_skip_home_recovery():
    source = Path("mujoco_scenes/kitchen_phase_c_execution.py").read_text()
    pour_path = source[source.index("dwell_steps ="):source.index("except RuntimeError", source.index("dwell_steps ="))]
    assert '"POUR_GRASP_POSE_HOVER"' in pour_path
    assert '"POUR_TILT"' in pour_path
    assert '"POUR_GRASP_POSE_HOVER_RECOVERY"' in pour_path
    assert '"POUR_UPRIGHT_RECOVERY"' not in pour_path
    assert '"POUR_HIGH_CLEARANCE_RECOVERY"' not in pour_path
    assert '"RECOVER_RECORDED_POST_PICK_CARRY_ARM"' not in pour_path
    assert "command_speed_scale=1.8" in pour_path
    assert "reference_rotation = body_rotation" in source
    assert "pour_orientations = ((0.0, 1.0), (0.0, -1.0))" in source


def test_controlled_serving_recovery_cannot_fall_back_to_countertop_slot():
    source = Path("mujoco_scenes/kitchen_ground_truth_execution.py").read_text()
    planner = source[
        source.index("def _find_canonical_upright_place_plan"):
        source.index("def _execute_controlled_placement")
    ]
    validator = source[
        source.index("def validate_stable_placement"):
        source.index("def update_object_to_countertop_location")
    ]
    assert 'if destination == "serving_area"' in planner
    assert "placement_target = resolver.resolve(object_id, destination)" in planner
    assert 'if destination == "countertop":\n                        self.staged_countertop_slots' in planner
    assert 'destination == "serving_area" and not serving_contact' in validator


def test_countertop_utensil_validation_does_not_require_vessel_upright_axis():
    source = Path("mujoco_scenes/kitchen_ground_truth_execution.py").read_text()
    validator = source[
        source.index("def validate_stable_placement"):
        source.index("def update_object_to_countertop_location")
    ]

    assert 'grasp_family == "UTENSIL"' in validator
    assert "if not is_countertop_utensil and tilt_deg > 8.0" in validator
    assert "if is_countertop_utensil and not counter_contact" in validator
    assert "2.0 if is_countertop_utensil" in validator
    assert "else 0.30 if is_countertop_source" in validator
    assert "else 1.00 if is_serving_vessel" in validator
    assert '"STABLE_SUPPORTED_UTENSIL_PLACEMENT"' in validator


def test_drawer_pick_is_single_physical_attempt():
    source = Path("mujoco_scenes/kitchen_ground_truth_execution.py").read_text()
    pick = source[
        source.index("def pick(self, object_id: str)"):
        source.index("def place(self, object_id: str", source.index("def pick(self, object_id: str)"))
    ]

    assert "physical_regrasp_attempts" not in pick
    assert "for _ in range(2)" not in pick


def test_serving_spoon_uses_height_tuned_vertical_drop_and_geometric_containment():
    source = Path("mujoco_scenes/kitchen_ground_truth_execution.py").read_text()
    placement = source[
        source.index("# Soup utensils belong inside"):
        source.index("# Sources do not need to return home")
    ]

    assert '1.0 if float(observed["length"]) >= 0.20 else 0.70' in placement
    assert "else 0.5" not in placement
    assert "insertion_depth = drop_depth_fraction * safe_cavity_depth" in placement
    assert "body_centre_within_opening_column" in placement
    assert "assigned_bowl_contact or body_centre_within_opening_column" in placement


def test_stirrer_park_hover_miss_cannot_release_beyond_counter_edge():
    source = Path("mujoco_scenes/kitchen_ground_truth_execution.py").read_text()
    placement = source[
        source.index("def place(self, object_id: str, destination: str)"):
        source.index("def pour(self, source_id: str, target_id: str")
    ]
    assert '"status": "TOOL_PARK_HOVER_FAILED"' in placement
    assert "Never release a stirrer from an unverified post-stir pose" in placement
    assert "and nearest_soup_bowl_distance >= 0.20" not in placement


# ── 1. Variant Discovery Tests ────────────────────────────────────────────────

def test_variant_discovery():
    variants = discover_variant_names()
    assert len(variants) == 12
    assert "F0_ALL_VISIBLE" in variants
    assert "F1_HIDDEN_COFFEE_VESSEL" in variants
    assert "F5_FULL_DISTRIBUTED_SEARCH" in variants
    assert "I0_MISSING_COFFEE_VESSEL" in variants
    assert "I5_MISSING_COFFEE_JAR" in variants


# ── 2. Oracle World State Logic & Precondition Tests ─────────────────────────

def test_oracle_state_pick_requires_hand_empty():
    state = OracleWorldState()
    state.object_locations["spoon_1"] = "countertop"
    state.object_locations["spoon_2"] = "countertop"

    # First pick succeeds
    valid, _ = state.check_preconditions({"operator": "PICK", "arguments": ["spoon_1"]})
    assert valid
    state.apply_action({"operator": "PICK", "arguments": ["spoon_1"]})
    assert state.held_object == "spoon_1"

    # Second pick fails because hand is full
    valid, reason = state.check_preconditions({"operator": "PICK", "arguments": ["spoon_2"]})
    assert not valid
    assert "Hand is not empty" in reason


def test_oracle_state_pick_from_closed_cupboard_fails():
    state = OracleWorldState()
    state.object_locations["cup_c2"] = "C2"
    state.container_open["C2"] = False

    valid, reason = state.check_preconditions({"operator": "PICK", "arguments": ["cup_c2"]})
    assert not valid
    assert "inside closed container" in reason

    # After OPEN(C2), pick succeeds
    state.apply_action({"operator": "OPEN", "arguments": ["C2"]})
    valid, _ = state.check_preconditions({"operator": "PICK", "arguments": ["cup_c2"]})
    assert valid


def test_oracle_state_pour_requires_source_held():
    state = OracleWorldState()
    state.object_locations["kettle"] = "countertop"
    state.object_locations["cup_1"] = "countertop"

    # Pour before picking source fails
    valid, reason = state.check_preconditions({"operator": "POUR", "arguments": ["kettle", "cup_1"]})
    assert not valid
    assert "Cannot pour" in reason

    # Pick source, then pour succeeds
    state.apply_action({"operator": "PICK", "arguments": ["kettle"]})
    valid, _ = state.check_preconditions({"operator": "POUR", "arguments": ["kettle", "cup_1"]})
    assert valid


def test_oracle_state_stir_requires_tool_held():
    state = OracleWorldState()
    state.object_locations["spoon"] = "countertop"
    state.object_locations["cup_1"] = "countertop"

    # Stir before picking tool fails
    valid, reason = state.check_preconditions({"operator": "STIR", "arguments": ["spoon", "cup_1"]})
    assert not valid
    assert "Cannot stir" in reason

    # Pick tool, then stir succeeds
    state.apply_action({"operator": "PICK", "arguments": ["spoon"]})
    valid, _ = state.check_preconditions({"operator": "STIR", "arguments": ["spoon", "cup_1"]})
    assert valid


# ── 3. Tool Reusability & Relocation Tests ────────────────────────────────────

def test_tool_reusability_pick_stir_stir_place_repick():
    """Verify tool reusability: PICK -> STIR -> STIR -> PLACE -> PICK."""
    state = OracleWorldState()
    state.object_locations["spoon"] = "countertop"
    state.object_locations["cup_A"] = "countertop"
    state.object_locations["cup_B"] = "countertop"

    plan = [
        {"action_index": 1, "operator": "PICK", "arguments": ["spoon"]},
        {"action_index": 2, "operator": "STIR", "arguments": ["spoon", "cup_A"]},
        {"action_index": 3, "operator": "STIR", "arguments": ["spoon", "cup_B"]},
        {"action_index": 4, "operator": "PLACE", "arguments": ["spoon", "countertop"]},
        {"action_index": 5, "operator": "PICK", "arguments": ["spoon"]},
    ]

    result = run_symbolic_preflight(state, plan)
    assert result["success"]
    assert ("spoon", "cup_A") in state.stirred_relations or len(result["steps"]) == 5


def test_relocated_cupboard_target_becomes_pour_stir_eligible():
    """Verify cupboard object relocated to countertop becomes POUR/STIR eligible."""
    state = OracleWorldState()
    state.object_locations["kettle"] = "countertop"
    state.object_locations["spoon"] = "countertop"
    state.object_locations["cup_c2"] = "C2"
    state.container_open["C2"] = False

    plan = [
        {"action_index": 1, "operator": "OPEN", "arguments": ["C2"]},
        {"action_index": 2, "operator": "PICK", "arguments": ["cup_c2"]},
        {"action_index": 3, "operator": "PLACE", "arguments": ["cup_c2", "countertop"]},
        {"action_index": 4, "operator": "PICK", "arguments": ["kettle"]},
        {"action_index": 5, "operator": "POUR", "arguments": ["kettle", "cup_c2"]},
        {"action_index": 6, "operator": "PLACE", "arguments": ["kettle", "countertop"]},
        {"action_index": 7, "operator": "PICK", "arguments": ["spoon"]},
        {"action_index": 8, "operator": "STIR", "arguments": ["spoon", "cup_c2"]},
        {"action_index": 9, "operator": "PLACE", "arguments": ["spoon", "countertop"]},
    ]

    result = run_symbolic_preflight(state, plan)
    assert result["success"]


def test_unique_action_instance_ids():
    """Repeated identical actions must have unique action instance IDs."""
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F1", include_robot=False, robot="none")
    assignment = solve_ground_truth_assignment(scene, "F1_HIDDEN_COFFEE_VESSEL")
    initial_state = initialize_oracle_world_state(scene._object_instance_records)
    plan = generate_ground_truth_plan(assignment, initial_state)

    instance_ids = [action["action_instance_id"] for action in plan]
    assert len(instance_ids) == len(set(instance_ids)), "Action instance IDs must be unique"


# ── 4. Ground Truth Assignment & Distributed Cases ───────────────────────────

def test_hidden_soup_assignment_uses_one_reusable_coffee_tool():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F2", include_robot=False, robot="none")
    assignment = solve_ground_truth_assignment(scene, "F2_HIDDEN_SOUP_BOWL")
    assert assignment.is_feasible
    assert len(assignment.unique_coffee_tools) == 1


def test_hidden_soup_bowl_is_retrieved_directly_to_serving_area():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F2", include_robot=False, robot="none")
    assignment = solve_ground_truth_assignment(scene, "F2_HIDDEN_SOUP_BOWL")
    initial_state = initialize_oracle_world_state(scene._object_instance_records)
    plan = generate_ground_truth_plan(assignment, initial_state)

    deep_bowl_actions = [
        (action["operator"], action["arguments"])
        for action in plan
        if "ab3_deep_bowl" in action["arguments"]
    ]
    assert deep_bowl_actions == [
        ("PICK", ["ab3_deep_bowl"]),
        ("PLACE", ["ab3_deep_bowl", "serving_area"]),
        ("PLACE_SERVING_UTENSIL", ["s1i_oversized_spoon", "ab3_deep_bowl"]),
    ]

    deep_utensil_index = next(
        index for index, action in enumerate(plan)
        if action["operator"] == "PLACE_SERVING_UTENSIL"
        and action["arguments"] == ["s1i_oversized_spoon", "ab3_deep_bowl"]
    )
    shallow_transfer_index = next(
        index for index, action in enumerate(plan)
        if action["operator"] == "PICK"
        and action["arguments"] == ["ab3_shallow_bowl"]
    )
    assert deep_utensil_index < shallow_transfer_index


def test_mixed_hidden_vessel_assignment_uses_one_reusable_coffee_tool():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F3", include_robot=False, robot="none")
    assignment = solve_ground_truth_assignment(scene, "F3_HIDDEN_VESSELS_MIXED")
    assert assignment.is_feasible
    assert len(assignment.unique_coffee_tools) == 1


def test_cupboard_vessel_retreats_at_shelf_height_before_carry():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F3",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(
        scene, "F3_HIDDEN_VESSELS_MIXED", "FEASIBLE"
    )
    dispatcher = KitchenGroundTruthExecutionDispatcher(scene, assignment)
    manipulation = dispatcher.phase_b.manipulation
    manipulation._configure_source_aware_pick_spec(
        "ab3_medium_deep_mug", "VESSEL", "CUPBOARD"
    )

    candidates = manipulation.executor.pick_specs[
        "ab3_medium_deep_mug"
    ].grasp_candidates
    assert candidates
    assert all(
        candidate.retreat_route_offsets_world_m == ((0.0, -0.20, 0.0),)
        for candidate in candidates
    )


def test_both_drawer_spoons_share_compact_top_down_candidate_family():
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F4",
        include_robot=True,
        robot="google",
    )
    assignment = solve_ground_truth_assignment(
        scene, "F4_TOOLS_IN_DRAWERS", "FEASIBLE"
    )
    dispatcher = KitchenGroundTruthExecutionDispatcher(scene, assignment)

    candidate_ids_by_object = {}
    for object_id in ("s1i_final_long_narrow_spoon", "ab3_partial_spoon"):
        candidates = dispatcher.phase_b.manipulation.executor.pick_specs[
            object_id
        ].grasp_candidates
        candidate_ids_by_object[object_id] = tuple(
            candidate.candidate_id for candidate in candidates
        )
        assert len(candidates) == 4
        assert all(
            candidate.candidate_id.startswith("drawer_vertical_perpendicular_")
            for candidate in candidates
        )
        assert all(
            candidate.approach_offset_world_m == (0.0, 0.0, 0.08)
            for candidate in candidates
        )

    # Both mirrored drawers use the same bounded handle-frame generator; its
    # live geometric ranking may choose either central handle fraction.
    assert all(candidate_ids_by_object.values())

    source = Path("mujoco_scenes/kitchen_object_manipulation.py").read_text()
    assert "normal bounded pick planner exactly once" in source


def test_infeasible_variants_detected():
    infeasible_checks = {
        "I0_MISSING_COFFEE_VESSEL": "INSUFFICIENT_COFFEE_CONTAINERS",
        "I1_MISSING_SOUP_BOWL": "INSUFFICIENT_SOUP_CONTAINERS",
        "I2_MISSING_COFFEE_SPOON": "MISSING_COFFEE_STIRRER",
        "I3_MISSING_SOUP_UTENSIL": "INSUFFICIENT_DISTINCT_SOUP_TOOLS",
        "I4_MISSING_KETTLE": "MISSING_WATER_SOURCE",
        "I5_MISSING_COFFEE_JAR": "MISSING_COFFEE_SOURCE",
    }
    for vid, expected_reason in infeasible_checks.items():
        scene = KitchenScene(f"S1_integrated_kitchen_object_function_feasibility_{vid.split('_')[0]}", include_robot=False, robot="none")
        assignment = solve_ground_truth_assignment(scene, vid, "INFEASIBLE")
        assert not assignment.is_feasible
        assert assignment.failure_reason == expected_reason


# ── 5. Camera & Mosaic Rendering Tests ───────────────────────────────────────

def test_five_project_cameras_discovered():
    assert len(FIVE_PROJECT_CAMERAS) == 5
    assert FIVE_PROJECT_CAMERAS == (
        "left_shoulder_camera",
        "right_shoulder_camera",
        "overhead_camera",
        "side_camera",
        "front_camera",
    )


def test_mosaic_frame_dimensions():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F1", include_robot=False, robot="none")
    recorder = KitchenGroundTruthRecorder(scene, tile_width=320, tile_height=180, fps=10, record=False, show=False)
    frame = recorder.capture_frame(force=True)
    assert frame is not None
    # 3x2 grid: 320*3 = 960 width, 180*2 = 360 height
    assert frame.shape == (360, 960, 3)
    recorder.close()


def test_frame_capture_is_observational(monkeypatch):
    """Recording must not advance or recompute MuJoCo physics state."""
    scene = KitchenScene(
        "S1_integrated_kitchen_object_function_feasibility_F1",
        include_robot=False,
        robot="none",
    )
    recorder = KitchenGroundTruthRecorder(
        scene,
        tile_width=320,
        tile_height=180,
        fps=10,
        record=False,
        show=False,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("capture_frame must not call mujoco.mj_forward")

    monkeypatch.setattr("mujoco_scenes.kitchen_ground_truth_recorder.mujoco.mj_forward", fail_if_called)
    frame = recorder.capture_frame(force=True)
    assert frame is not None
    recorder.close()


def test_video_recording_smoke():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F1", include_robot=False, robot="none")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    recorder = KitchenGroundTruthRecorder(scene, output_path=tmp_path, tile_width=320, tile_height=180, fps=10, record=True, show=False)
    for _ in range(5):
        recorder.capture_frame(force=True)
    recorder.close()

    assert tmp_path.exists()
    assert tmp_path.stat().st_size > 0
    tmp_path.unlink()
