"""Focused scientific guards for Living-Room mobile execution Phase 3."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_scenes.generic_manipulation import CalibratedPickPlaceExecutor
from mujoco_scenes.living_room_mobile_execution import (
    BasePose,
    SCENE,
    _joint_base_to_world,
    _world_to_joint_base,
    allocate_observed_placements,
    candidate_stances,
)
from mujoco_scenes.living_room_region_scene import build_l2_region_xml


ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "runs/living_room_region_phase1/living_room_region_phase1_final_closure_v3_20260809/F0_BASE"
PHASE2 = ROOT / "mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/variants/F0_BASE"


def _payloads():
    if not (PHASE1 / "payload_registry.json").exists():
        pytest.skip("authoritative untracked Phase-1 run is unavailable")
    return json.loads((PHASE1 / "payload_registry.json").read_text())


def _regions():
    return json.loads((PHASE1 / "region_registry.json").read_text())


def _plan():
    return json.loads((PHASE2 / "plan.json").read_text())


def test_execution_annotations_exist_only_with_google_robot():
    robot_xml = build_l2_region_xml(SCENE, "google")
    observation_xml = build_l2_region_xml(SCENE, "none")
    assert "phase3_grasp_a2_drink_left" in robot_xml
    assert "google:pick_weld_a2_drink_left" in robot_xml
    assert "phase3_grasp_a2_drink_left" not in observation_xml


def test_all_six_payloads_have_execution_grasp_frames_and_welds():
    xml = build_l2_region_xml(SCENE, "google")
    for name in (
        "a2_drink_left", "a2_drink_right", "a2_snack_left",
        "a2_snack_right", "a2_remote_payload", "a2_controller_payload",
    ):
        assert f'phase3_grasp_{name}' in xml
        assert f'google:pick_weld_{name}' in xml


def test_execution_sites_are_invisible():
    xml = build_l2_region_xml(SCENE, "google")
    start = xml.index('name="phase3_grasp_a2_drink_left"')
    assert 'rgba="0 0 0 0"' in xml[start:start + 220]


def test_welds_start_inactive():
    xml = build_l2_region_xml(SCENE, "google")
    start = xml.index('name="google:pick_weld_a2_drink_left"')
    assert 'active="false"' in xml[start:start + 260]


def test_dynamic_place_rejects_nonfinite_target():
    executor = object.__new__(CalibratedPickPlaceExecutor)
    executor.mode = "holding"
    executor.held_object = "payload"
    executor.pick_specs = {"payload": type("S", (), {"place_supported": True})()}
    executor.failure = None
    executor.calibration_attempt_ticks = 0
    executor._near_navigation_home = lambda: True
    with pytest.raises(ValueError, match="finite xyz"):
        executor.request_place_world(np.array((0.0, np.nan, 0.5)))


def test_dynamic_place_records_world_target():
    executor = object.__new__(CalibratedPickPlaceExecutor)
    executor.mode = "holding"
    executor.held_object = "payload"
    executor.pick_specs = {"payload": type("S", (), {"place_supported": True})()}
    executor.failure = None
    executor.calibration_attempt_ticks = 0
    executor._near_navigation_home = lambda: True
    target = np.array((1.0, 2.0, 0.5))
    executor.request_place_world(target)
    assert np.allclose(executor.pending_place_world, target)
    assert executor.pending_place_site == "<dynamic_world_target>"


def test_base_coordinate_round_trip():
    pose = BasePose(-1.2, 0.8, 1.1)
    recovered = _joint_base_to_world(_world_to_joint_base(pose))
    assert np.allclose(
        (recovered.x, recovered.y, recovered.yaw),
        (pose.x, pose.y, pose.yaw),
    )


def test_current_pose_is_always_first_stance_candidate():
    current = BasePose(0.1, -0.2, 0.3)
    assert candidate_stances(np.array((1.0, 1.0, 0.5)), current)[0] == current


def test_stance_generation_is_deterministic():
    target = np.array((1.0, 1.0, 0.5))
    current = BasePose(0.0, 0.0, 0.0)
    assert candidate_stances(target, current) == candidate_stances(target, current)


def test_stance_generation_is_dynamic_not_named_destination():
    current = BasePose(0.0, 0.0, 0.0)
    left = candidate_stances(np.array((-1.0, 1.0, 0.5)), current)[1]
    right = candidate_stances(np.array((1.0, 1.0, 0.5)), current)[1]
    assert left.x != right.x


def test_all_phase2_objects_receive_one_dynamic_placement():
    result = allocate_observed_placements(_payloads(), _regions(), _plan())
    assert {row["object_id"] for row in result["placements"]} == {
        f"object_{index:04d}" for index in range(1, 7)
    }


def test_placements_use_only_observed_geometry_provenance():
    result = allocate_observed_placements(_payloads(), _regions(), _plan())
    assert all(
        row["source"] == "PHASE1_OBSERVED_REGION_AND_PAYLOAD_GEOMETRY"
        for row in result["placements"]
    )


def test_every_placement_is_inside_measured_support():
    result = allocate_observed_placements(_payloads(), _regions(), _plan())
    assert all(row["within_measured_support"] for row in result["placements"])


def test_shared_targets_are_distinct():
    rows = allocate_observed_placements(_payloads(), _regions(), _plan())["placements"]
    for region in {row["region_id"] for row in rows}:
        points = [tuple(row["desired_body_world_m"][:2]) for row in rows if row["region_id"] == region]
        assert len(points) == len(set(points))


def test_phase2_plan_has_six_pick_place_pairs_and_no_move():
    operators = [row["operator"] for row in _plan()["actions"]]
    assert operators == ["PICK", "PLACE"] * 6


def test_scene_name_is_frozen_authoritative_f0():
    assert SCENE == "L2_integrated_living_room_region_function_F0_BASE"
