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
    inspect_held_object_state,
    oriented_rectangle_corners,
    oriented_rectangles_clearance,
    rectangle_inside_observed_support,
    verify_physical_on_relation,
)
from mujoco_scenes.living_room_region_scene import build_l2_region_xml


ROOT = Path(__file__).resolve().parents[2]
PHASE1 = ROOT / "mujoco_scenes/benchmark_reports/living_room_region_feasibility_phase1/variants/F0_ALL_OBJECTS_IN_STAGING"
PHASE2 = ROOT / "mujoco_scenes/benchmark_reports/living_room_symbolic_phase2/variants/F0_ALL_OBJECTS_IN_STAGING"


def _payloads():
    if not (PHASE1 / "payload_registry.json").exists():
        pytest.skip("authoritative untracked Phase-1 run is unavailable")
    return json.loads((PHASE1 / "payload_registry.json").read_text())


def _regions():
    return json.loads((PHASE1 / "region_registry.json").read_text())


def _plan():
    return json.loads((PHASE2 / "plan.json").read_text())


def _assignments():
    return json.loads((PHASE1 / "region_assignments.json").read_text())


def _placements():
    return allocate_observed_placements(_payloads(), _regions(), _plan(), _assignments())


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
        "a2_snack_right", "a2_remote_payload",
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
    result = _placements()
    assert {row["object_id"] for row in result["placements"]} == {
        f"object_{index:04d}" for index in range(1, 6)
    }


def test_placements_use_only_observed_geometry_provenance():
    result = _placements()
    assert all(
        row["source"] == "PHASE1_SELECTED_MEASURED_PACKING"
        for row in result["placements"]
    )


def test_every_placement_is_inside_measured_support():
    result = _placements()
    assert all(row["within_measured_support"] for row in result["placements"])


def test_shared_targets_are_distinct():
    rows = _placements()["placements"]
    for region in {row["region_id"] for row in rows}:
        points = [tuple(row["desired_body_world_m"][:2]) for row in rows if row["region_id"] == region]
        assert len(points) == len(set(points))


def test_phase2_plan_has_five_pick_place_pairs_and_no_move():
    operators = [row["operator"] for row in _plan()["actions"]]
    assert operators == ["PICK", "PLACE"] * 5


def test_scene_name_is_frozen_authoritative_f0():
    assert SCENE == "L2_integrated_living_room_region_function_F0_ALL_OBJECTS_IN_STAGING"


@pytest.mark.parametrize("yaw", (0.0, np.pi / 2))
def test_oriented_rectangle_inside_axis_aligned_support(yaw):
    corners = oriented_rectangle_corners(np.zeros(2), 0.4, 0.2, yaw)
    result = rectangle_inside_observed_support(corners, np.zeros(2), np.array((1.0, 0.0)), 1.0, 1.0)
    assert result["inside"]
    assert result["minimum_edge_margin_m"] == pytest.approx(0.3)


def test_rectangle_center_inside_but_corner_outside_is_rejected():
    corners = oriented_rectangle_corners(np.array((0.43, 0.0)), 0.2, 0.2, 0.0)
    result = rectangle_inside_observed_support(corners, np.zeros(2), np.array((1.0, 0.0)), 1.0, 1.0)
    assert not result["inside"]
    assert result["minimum_edge_margin_m"] < 0


def test_oriented_rectangle_overlap_and_clearance():
    left = oriented_rectangle_corners(np.array((-0.15, 0.0)), 0.2, 0.2, 0.0)
    right = oriented_rectangle_corners(np.array((0.15, 0.0)), 0.2, 0.2, np.pi / 2)
    separated = oriented_rectangles_clearance(left, right)
    assert not separated["overlap"]
    assert separated["signed_clearance_m"] == pytest.approx(0.1)
    overlap = oriented_rectangles_clearance(left, oriented_rectangle_corners(np.array((-0.10, 0.0)), 0.2, 0.2, 0.0))
    assert overlap["overlap"]
    assert overlap["signed_clearance_m"] < 0


def test_phase1_selected_packing_and_orientation_are_consumed():
    result = _placements()
    assert result["phase1_selected_packing_consumed"]
    assert all(row["packing_arrangement"] in {"ALONG_LENGTH", "ALONG_WIDTH", "SINGLE_CENTERED"} for row in result["placements"])
    assert all(row["phase1_orientation_deg"] in {0, 90} for row in result["placements"])
    assert all(row["predicted_minimum_margin_m"] >= row["edge_clearance_m"] for row in result["placements"])
    assert all(row["valid"] for row in result["pairwise_rectangle_checks"])


def _held_state_model():
    xml = """<mujoco><worldbody>
      <geom name="a2_floor" type="plane" size="3 3 .1"/>
      <body name="google:link_gripper" pos="0 0 .5">
        <joint name="google:joint_finger_right" type="slide" axis="1 0 0" range="0 2"/>
        <joint name="google:joint_finger_left" type="slide" axis="0 1 0" range="0 2"/>
        <geom name="gripper" type="sphere" size=".01" contype="0" conaffinity="0"/>
      </body>
      <body name="a2_drink_left" pos="0 0 .32"><freejoint/><geom name="payload" type="box" size=".03 .03 .05"/></body>
      <body name="a2_drink_right" pos="1 0 .32"><freejoint/><geom name="payload2" type="box" size=".03 .03 .05"/></body>
    </worldbody><equality>
      <weld name="google:pick_weld_a2_drink_left" body1="google:link_gripper" body2="a2_drink_left" active="false"/>
      <weld name="google:pick_weld_a2_drink_right" body1="google:link_gripper" body2="a2_drink_right" active="false"/>
    </equality></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name in ("google:joint_finger_right", "google:joint_finger_left"):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = 0.05
    mujoco.mj_forward(model, data)
    return model, data


def test_inactive_weld_is_rejected_even_if_python_would_claim_holding():
    model, data = _held_state_model()
    state = inspect_held_object_state(model, data, "object_0001", "a2_drink_left")
    assert state.validation_status == "FALSE"
    assert "GRASP_WELD_INACTIVE" in state.rejection_reasons


def test_active_correct_payload_weld_is_accepted():
    model, data = _held_state_model()
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "google:pick_weld_a2_drink_left")
    data.eq_active[weld] = 1
    state = inspect_held_object_state(model, data, "object_0001", "a2_drink_left")
    assert state.validation_status == "TRUE"


def test_wrong_or_multiple_payload_weld_is_rejected():
    model, data = _held_state_model()
    wrong = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "google:pick_weld_a2_drink_right")
    data.eq_active[wrong] = 1
    wrong_state = inspect_held_object_state(model, data, "object_0001", "a2_drink_left")
    assert wrong_state.validation_status == "FALSE"
    correct = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "google:pick_weld_a2_drink_left")
    data.eq_active[correct] = 1
    multiple = inspect_held_object_state(model, data, "object_0001", "a2_drink_left")
    assert "UNEXPECTED_ACTIVE_PAYLOAD_WELD" in multiple.rejection_reasons


def _on_verifier_fixture(*, second_payload=False):
    second = ('<body name="a2_drink_right" pos=".30 0 .59"><freejoint/>'
              '<geom name="payload2" type="cylinder" size=".03 .07" mass=".2"/></body>') if second_payload else ""
    second_weld = ('<weld name="google:pick_weld_a2_drink_right" body1="google:link_gripper" '
                   'body2="a2_drink_right" active="false"/>') if second_payload else ""
    xml = f"""<mujoco><option timestep=".002" gravity="0 0 -9.81"/><worldbody>
      <geom name="a2_floor" type="plane" size="3 3 .1"/>
      <geom name="a2_personal_left_top" type="box" pos="0 0 .5" size=".5 .5 .02"/>
      <body name="a2_drink_left" pos="0 0 .59"><freejoint/><geom name="payload" type="cylinder" size=".03 .07" mass=".2"/></body>
      {second}<body name="google:link_gripper" pos="0 0 1"><geom type="sphere" size=".01" contype="0" conaffinity="0"/></body>
    </worldbody><equality><weld name="google:pick_weld_a2_drink_left" body1="google:link_gripper" body2="a2_drink_left" active="false"/>{second_weld}</equality></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    region = {"geometry": {
        "centroid_world_m": {"value": [0, 0, .52]},
        "principal_axis_world": {"value": [1, 0, 0]},
        "support_length_m": {"value": 1.0},
        "support_width_m": {"value": 1.0},
    }}
    placement = {"region_id": "region_0001", "footprint_length_m": .06,
                 "footprint_width_m": .06, "yaw_world_rad": 0.0,
                 "inter_payload_clearance_m": .025}
    placements = {"object_0001": placement}
    backends = {"object_0001": "a2_drink_left"}
    if second_payload:
        placements["object_0002"] = dict(placement)
        backends["object_0002"] = "a2_drink_right"
    return model, data, region, placement, placements, backends


def _verify_fixture(model, data, region, placement, placements, backends):
    return verify_physical_on_relation(
        model, data, "object_0001", "a2_drink_left", "region_0001",
        "a2_personal_left_top", region, placement, placements, backends,
    )


def test_physical_on_requires_actual_support_contact_and_settled_state():
    fixture = _on_verifier_fixture()
    valid = _verify_fixture(*fixture)
    assert valid["verified"]
    model, data, region, placement, placements, backends = fixture
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a2_drink_left")
    joint = int(model.body_jntadr[body])
    qpos = int(model.jnt_qposadr[joint])
    data.qpos[qpos + 2] = .85
    mujoco.mj_forward(model, data)
    floating = _verify_fixture(model, data, region, placement, placements, backends)
    assert not floating["verified"]
    assert not floating["support_contact"]["support_contact_found"]


def test_physical_on_rejects_floor_edge_overhang_and_motion():
    model, data, region, placement, placements, backends = _on_verifier_fixture()
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a2_drink_left")
    joint = int(model.body_jntadr[body])
    qpos = int(model.jnt_qposadr[joint])
    dof = int(model.jnt_dofadr[joint])
    data.qpos[qpos:qpos + 3] = (0, 0, .07)
    for _ in range(400):
        mujoco.mj_step(model, data)
    floor = _verify_fixture(model, data, region, placement, placements, backends)
    assert not floor["verified"] and floor["floor_contact_found"]
    data.qpos[qpos:qpos + 3] = (.49, 0, .59)
    data.qvel[dof:dof + 6] = 0
    for _ in range(400):
        mujoco.mj_step(model, data)
    edge = _verify_fixture(model, data, region, placement, placements, backends)
    assert not edge["verified"] and not edge["footprint_inside_observed_support"]["inside"]
    data.qpos[qpos:qpos + 3] = (0, 0, .59)
    data.qvel[dof:dof + 6] = (0, 0, 0, 1, 0, 0)
    mujoco.mj_forward(model, data)
    moving = _verify_fixture(model, data, region, placement, placements, backends)
    assert not moving["verified"] and not moving["settling"]["stable"]
    assisted = verify_physical_on_relation(
        model, data, "object_0001", "a2_drink_left", "region_0001",
        "a2_personal_left_top", region, placement, placements, backends,
        assisted_validation=True,
    )
    assert assisted["verified"]
    assert not assisted["strict_verified"]
    assert assisted["assisted_postcondition_accepted"]


def test_physical_on_rejects_oriented_payload_overlap():
    model, data, region, placement, placements, backends = _on_verifier_fixture(second_payload=True)
    other = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a2_drink_right")
    joint = int(model.body_jntadr[other])
    qpos = int(model.jnt_qposadr[joint])
    data.qpos[qpos:qpos + 3] = (.04, 0, .59)
    for _ in range(400):
        mujoco.mj_step(model, data)
    result = _verify_fixture(model, data, region, placement, placements, backends)
    assert not result["verified"]
    assert not result["payload_nonoverlap"]["valid_nonoverlap"]
    assisted = verify_physical_on_relation(
        model, data, "object_0001", "a2_drink_left", "region_0001",
        "a2_personal_left_top", region, placement, placements, backends,
        assisted_validation=True,
    )
    assert not assisted["verified"]
    assert not assisted["assisted_postcondition_accepted"]
