from __future__ import annotations

import inspect
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import mujoco
import pytest

from mujoco_scenes.living_room_region_scene import (
    L2_ABLATION1_SCENES,
    L2_ABLATION2_SCENES,
    L2LivingRoomRegionScene,
    build_l2_region_xml,
)
from mujoco_scenes.region_ablation2 import (
    DEFAULT_CONTROLS_TASK_CONFIG,
    DEFAULT_DRINKS_TASK_CONFIG,
    DEFAULT_TASK_CONFIG,
    RegionAllocationSolver,
    RegionAblation2Run,
    evaluate_fits_set_on,
    load_ablation2_task,
)


TASK = load_ablation2_task(DEFAULT_TASK_CONFIG)


def _measurement(value):
    return {"value": value, "status": "MEASURED"}


def _payload(length, width):
    return {
        "footprint_length_m": _measurement(length),
        "footprint_width_m": _measurement(width),
    }


def _region(length, width):
    return {
        "support_length_m": _measurement(length),
        "support_width_m": _measurement(width),
    }


def _drink_row(slot, payload, seat, region, rank):
    return {
        "slot_id": slot,
        "payload_id": payload,
        "seating_target_id": seat,
        "region_id": region,
        "candidate_rank": rank,
        "compatibility_status": "TRUE",
        "fit_margin_m": 0.20,
        "near_seat_margin_m": 0.15,
    }


def _solver(task=None):
    drink_rows = [
        _drink_row("drink_slot_1", "object_a", "seat_a", "personal_a", 1),
        _drink_row("drink_slot_1", "object_a", "seat_a", "shared_trap", 3),
        _drink_row("drink_slot_2", "object_b", "seat_b", "personal_b", 2),
        _drink_row("drink_slot_2", "object_b", "seat_b", "shared_trap", 3),
    ]
    control_rows = [
        {
            "payload_ids": ["remote", "controller"],
            "region_id": "control_table",
            "candidate_rank": 4,
            "packing_margin_m": 0.31,
            "accessibility_margin_m": 0.12,
            "compatibility_status": "TRUE",
        }
    ]
    individual = [
        {
            "payload_id": payload,
            "region_id": "control_table",
            "candidate_rank": 4,
            "fit_margin_m": 0.35,
            "accessibility_margin_m": 0.12,
            "compatibility_status": "TRUE",
        }
        for payload in ("remote", "controller")
    ]
    return RegionAllocationSolver(
        drink_rows=drink_rows,
        control_rows=control_rows,
        control_individual_rows=individual,
        task_config=deepcopy(task or TASK),
    )


def test_ablation1_scene_names_are_preserved():
    assert L2_ABLATION1_SCENES == (
        "L2_living_room_region_ablation1_primary",
        "L2_living_room_region_ablation1_initial_complete",
        "L2_living_room_region_ablation1_exhaustion",
    )
    assert not set(L2_ABLATION1_SCENES) & set(L2_ABLATION2_SCENES)


@pytest.mark.parametrize("scene_name", L2_ABLATION2_SCENES)
def test_ablation2_scenes_compile_without_robot(scene_name):
    model = mujoco.MjModel.from_xml_string(
        build_l2_region_xml(scene_name, robot="none")
    )
    assert model.ncam == 5
    free_joints = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    assert len(free_joints) == 6


@pytest.mark.parametrize("scene_name", L2_ABLATION2_SCENES)
def test_ablation2_scenes_compile_with_google_robot(scene_name):
    scene = L2LivingRoomRegionScene(scene_name, robot="google")
    assert scene.has_robot is True
    assert scene.model.ncam >= 5


def test_scene_runtime_does_not_publish_simulator_payload_names():
    scene = L2LivingRoomRegionScene(
        L2_ABLATION2_SCENES[0], robot="none"
    )
    assert scene.get_visible_object_instances() == []


def test_function_policies_are_declarative_and_cross_group_sharing_disabled():
    assert (
        TASK["function_groups"]["personal_drinks"]["usage_policy"]
        == "DEDICATED_REGION_PER_TARGET"
    )
    assert (
        TASK["function_groups"]["shared_controls"]["usage_policy"]
        == "SHARED_REGION_REQUIRED"
    )
    assert TASK["allow_cross_function_region_sharing"] is False


def test_primary_policy_counterexamples_and_derived_count():
    solver = _solver()
    shared = solver.solve("always_shared")
    distinct = solver.solve("always_distinct")
    aware = solver.solve("function_aware")
    assert shared["status"] == "COMPLETE"
    assert shared["distinct_physical_region_count"] == 2
    drink_regions = {
        item["region_id"]
        for item in shared["assignments"]
        if item["function_group"] == "personal_drinks"
    }
    assert drink_regions == {"shared_trap"}
    assert distinct["status"] == "EXHAUSTED"
    assert aware["status"] == "COMPLETE"
    assert aware["distinct_physical_region_count"] == 3


def test_function_aware_drinks_cover_two_targets_with_distinct_regions():
    solver = _solver(load_ablation2_task(DEFAULT_DRINKS_TASK_CONFIG))
    result = solver.solve("function_aware")
    assert result["status"] == "COMPLETE"
    assignments = result["assignments"]
    assert {item["target_id"] for item in assignments} == {"seat_a", "seat_b"}
    assert len({item["region_id"] for item in assignments}) == 2


def test_one_region_near_both_seats_cannot_satisfy_dedicated_policy():
    solver = _solver(load_ablation2_task(DEFAULT_DRINKS_TASK_CONFIG))
    solver.drink_rows = [
        row for row in solver.drink_rows if row["region_id"] == "shared_trap"
    ]
    assert solver.solve("always_shared")["status"] == "COMPLETE"
    assert solver.solve("function_aware")["status"] == "EXHAUSTED"


def test_two_regions_for_only_one_target_do_not_cover_both_slots():
    solver = _solver(load_ablation2_task(DEFAULT_DRINKS_TASK_CONFIG))
    solver.drink_rows = [
        row
        for row in solver.drink_rows
        if row["slot_id"] == "drink_slot_1"
    ]
    assert solver.solve("function_aware")["status"] == "EXHAUSTED"


def test_function_aware_controls_share_one_region():
    solver = _solver(load_ablation2_task(DEFAULT_CONTROLS_TASK_CONFIG))
    aware = solver.solve("function_aware")
    distinct = solver.solve("always_distinct")
    assert aware["status"] == "COMPLETE"
    assert aware["distinct_physical_region_count"] == 1
    assert {item["region_id"] for item in aware["assignments"]} == {
        "control_table"
    }
    assert distinct["status"] == "EXHAUSTED"


def test_cross_function_region_conflict_is_rejected():
    solver = _solver()
    solver.drink_rows = [
        row
        for row in solver.drink_rows
        if row["region_id"] != "shared_trap"
    ]
    solver.control_rows[0]["region_id"] = "personal_a"
    for row in solver.control_individual_rows:
        row["region_id"] = "personal_a"
    assert solver.solve("function_aware")["status"] == "EXHAUSTED"


def test_solver_has_no_persistent_region_id_literal():
    assert "region_000" not in inspect.getsource(RegionAllocationSolver)


def test_fits_set_on_tests_rotations_axes_clearances_and_margins():
    relation = evaluate_fits_set_on(
        [_payload(0.21, 0.08), _payload(0.23, 0.14)],
        _region(0.96, 0.66),
        task_config=TASK,
    )
    assert relation["status"] == "TRUE"
    assert len(relation["tested_packings"]) == 8
    assert {
        item["arrangement"] for item in relation["tested_packings"]
    } == {"ALONG_LENGTH", "ALONG_WIDTH"}
    assert relation["edge_clearance_margin_m"] > 0
    assert relation["inter_payload_clearance_m"] > 0
    assert relation["signed_clearance_margin_m"] > 0
    assert relation["selected_packing"]["non_overlapping"] is True


def test_fits_set_on_rejects_area_only_shape_false_positive():
    relation = evaluate_fits_set_on(
        [_payload(0.40, 0.10), _payload(0.40, 0.10)],
        _region(0.50, 0.25),
        task_config=TASK,
    )
    assert 0.50 * 0.25 > 2 * 0.40 * 0.10
    assert relation["status"] == "FALSE"
    assert relation["signed_clearance_margin_m"] < 0


@pytest.mark.parametrize(
    "payloads,region",
    [
        ([_payload(0.2, 0.1)], _region(0.6, 0.4)),
        (
            [_payload(0.2, 0.1), {"footprint_length_m": _measurement(None)}],
            _region(0.6, 0.4),
        ),
    ],
)
def test_fits_set_on_inadequate_evidence_is_unknown(payloads, region):
    relation = evaluate_fits_set_on(
        payloads, region, task_config=TASK
    )
    assert relation["status"] == "UNKNOWN"
    assert relation["value"] is None


def test_all_policy_summaries_reference_one_hashed_evidence_manifest(tmp_path):
    path = tmp_path / "run"
    run = RegionAblation2Run(path, scene_name="synthetic_test_scene")
    evidence = path / "observation" / "cameras" / "view_1" / "rgb.png"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"one immutable rendered observation")
    run.policy_evaluations = {
        policy: {
            "status": "EXHAUSTED",
            "assignments": [],
            "classification": "TEST",
            "distinct_physical_region_count": None,
        }
        for policy in ("always_shared", "always_distinct", "function_aware")
    }
    run._persist({})
    summary = json.loads(
        (path / "region_ablation2_summary.json").read_text()
    )
    assert summary["single_initial_observation"] is True
    assert summary["rerendered_for_policies"] is False
    assert summary["semantic_inference_repeated_for_policies"] is False
    assert summary["same_evidence_manifest"] == [
        {
            "path": "observation/cameras/view_1/rgb.png",
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    ]
    assert set(summary["policies"]) == {
        "always_shared",
        "always_distinct",
        "function_aware",
    }
