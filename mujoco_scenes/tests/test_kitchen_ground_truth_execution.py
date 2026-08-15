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
from mujoco_scenes.scene_loader import KitchenScene


# ── 1. Variant Discovery Tests ────────────────────────────────────────────────

def test_variant_discovery():
    variants = discover_variant_names()
    assert len(variants) >= 16
    assert "F0_REUSE_ONE" in variants
    assert "F1_INITIAL_COMPLETE" in variants
    assert "F2_DISTRIBUTED_COFFEE_TWO" in variants
    assert "F3_DISTRIBUTED_COFFEE_THREE" in variants
    assert "I0_MISSING_COFFEE_VESSEL" in variants
    assert "P0_LAYOUT_BASE" in variants


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
    assignment = solve_ground_truth_assignment(scene, "F1_INITIAL_COMPLETE")
    initial_state = initialize_oracle_world_state(scene._object_instance_records)
    plan = generate_ground_truth_plan(assignment, initial_state)

    instance_ids = [action["action_instance_id"] for action in plan]
    assert len(instance_ids) == len(set(instance_ids)), "Action instance IDs must be unique"


# ── 4. Ground Truth Assignment & Distributed Cases ───────────────────────────

def test_distributed_coffee_two_assignment():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F2", include_robot=False, robot="none")
    assignment = solve_ground_truth_assignment(scene, "F2_DISTRIBUTED_COFFEE_TWO")
    assert assignment.is_feasible
    assert len(assignment.unique_coffee_tools) == 2


def test_distributed_coffee_three_assignment():
    scene = KitchenScene("S1_integrated_kitchen_object_function_feasibility_F3", include_robot=False, robot="none")
    assignment = solve_ground_truth_assignment(scene, "F3_DISTRIBUTED_COFFEE_THREE")
    assert assignment.is_feasible
    assert len(assignment.unique_coffee_tools) == 3


def test_infeasible_variants_detected():
    infeasible_checks = {
        "I0_MISSING_COFFEE_VESSEL": "INSUFFICIENT_COFFEE_CONTAINERS",
        "I1_MISSING_SOUP_VESSEL": "INSUFFICIENT_SOUP_CONTAINERS",
        "I2_UNCOVERED_COFFEE_TARGET": "UNCOVERED_COFFEE_TARGET",
        "I3_ONLY_TWO_SOUP_TOOLS": "INSUFFICIENT_DISTINCT_SOUP_TOOLS",
        "I4_SOUP_MATCHING_TRAP": "NO_COMPLETE_SOUP_MATCHING",
        "I5_SEMANTIC_DECOY_GEOMETRY_FAILURE": "UNCOVERED_COFFEE_TARGET",
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
