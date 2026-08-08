from __future__ import annotations

import inspect
from pathlib import Path

import mujoco
import pytest

from mujoco_scenes.living_room_region_function import (
    DEFAULT_TASK_CONFIG,
    EXPECTED_VARIANTS,
    GlobalRegionAllocationSolver,
    IntegratedLivingRoomRegionRun,
    load_integrated_task,
    variant_code,
    write_resolved_integrated_rig,
)
from mujoco_scenes.living_room_region_oracle import (
    ORACLE_MARKER,
    evaluate_privileged_oracle,
)
from mujoco_scenes.living_room_region_scene import (
    L2_INTEGRATED_GOAL,
    L2_INTEGRATED_SCENES,
    L2LivingRoomRegionScene,
    build_l2_region_xml,
)
from mujoco_scenes.region_ablation2 import InitialEvidenceCapture, evaluate_fits_set_on


TASK = load_integrated_task()


def _personal(slot, target, region, rank=1, status="TRUE"):
    return {
        "slot_id": slot,
        "seating_target_id": target,
        "payload_ids": [f"{slot}_drink", f"{slot}_snack"],
        "region_id": region,
        "candidate_rank": rank,
        "semantic_role_status": status,
        "PLANAR_SUPPORT": status,
        "FITS_SET_ON": status,
        "NEAR_SEAT": status,
        "compatibility_status": status,
        "fit_margin_m": 0.1,
        "context_margin_m": 0.1,
    }


def _shared(region, rank=3, status="TRUE"):
    return {
        "slot_id": "shared_controls_slot",
        "payload_ids": ["remote", "controller"],
        "seating_target_ids": ["seat_a", "seat_b"],
        "region_id": region,
        "candidate_rank": rank,
        "semantic_role_status": status,
        "PLANAR_SUPPORT": status,
        "FITS_SET_ON": status,
        "ACCESSIBLE_FROM_BOTH_SEATS": status,
        "compatibility_status": status,
        "fit_margin_m": 0.1,
        "context_margin_m": 0.1,
    }


def test_fixed_task_contract_is_region_only_and_has_exact_goal():
    assert TASK["natural_language_goal"] == L2_INTEGRATED_GOAL
    assert TASK["requirement_entity_kind"] == "REGION"
    assert "object_functions" not in TASK
    assert all(
        group["candidate_entity_kind"] == "REGION"
        for group in TASK["function_groups"].values()
    )


def test_usage_and_region_sharing_policies_are_declarative():
    assert TASK["function_groups"]["personal_refreshment"]["usage_policy"] == "DEDICATED_REGION_PER_TARGET"
    assert TASK["function_groups"]["shared_controls"]["usage_policy"] == "SHARED_REGION_REQUIRED"
    assert TASK["allow_cross_function_region_sharing"] is False


@pytest.mark.parametrize("scene_name", L2_INTEGRATED_SCENES)
def test_every_integrated_scene_compiles_with_five_cameras_and_six_payloads(scene_name):
    model = mujoco.MjModel.from_xml_string(build_l2_region_xml(scene_name, "none"))
    assert model.ncam == 5
    assert sum(
        model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE
        for index in range(model.njnt)
    ) == 6


@pytest.mark.parametrize("scene_name", L2_INTEGRATED_SCENES)
def test_every_integrated_scene_uses_identical_goal(scene_name):
    scene = L2LivingRoomRegionScene(scene_name, robot="none")
    assert scene.goal == L2_INTEGRATED_GOAL
    assert scene.get_visible_object_instances() == []


def test_global_solver_recovers_matching_that_greedy_order_can_strand():
    rows = [
        _personal("slot_a", "seat_a", "flexible", 1),
        _personal("slot_a", "seat_a", "a_only", 2),
        _personal("slot_b", "seat_b", "flexible", 1),
    ]
    result = GlobalRegionAllocationSolver(
        rows, [_shared("shared")],
        allow_cross_function_region_sharing=False,
    ).solve()
    assert result["status"] == "COMPLETE"
    assignment = {item["slot_id"]: item["region_id"] for item in result["assignments"]}
    assert assignment["slot_a"] == "a_only"
    assert assignment["slot_b"] == "flexible"


def test_greedy_diagnostic_fails_where_global_solver_reassigns():
    rows = [
        _personal("slot_a", "seat_a", "flexible", 1),
        _personal("slot_a", "seat_a", "a_only", 2),
        _personal("slot_b", "seat_b", "flexible", 1),
    ]
    solver = GlobalRegionAllocationSolver(
        rows, [_shared("shared")],
        allow_cross_function_region_sharing=False,
    )
    assert solver.greedy()["status"] == "INFEASIBLE"
    assert solver.solve()["status"] == "COMPLETE"


def test_personal_regions_are_distinct_and_controls_share_one_region():
    result = GlobalRegionAllocationSolver(
        [_personal("a", "seat_a", "left"), _personal("b", "seat_b", "right")],
        [_shared("center")], allow_cross_function_region_sharing=False,
    ).solve()
    assert result["status"] == "COMPLETE"
    assert result["distinct_selected_region_count"] == 3
    assert len([row for row in result["assignments"] if row["function_id"] == "SHARED_CONTROLS_REGION"]) == 1


def test_cross_function_region_conflict_is_infeasible():
    result = GlobalRegionAllocationSolver(
        [_personal("a", "seat_a", "left"), _personal("b", "seat_b", "shared")],
        [_shared("shared")], allow_cross_function_region_sharing=False,
    ).solve()
    assert result["status"] == "INFEASIBLE"


def test_unknown_edges_never_enter_global_allocation():
    result = GlobalRegionAllocationSolver(
        [_personal("a", "seat_a", "left"), _personal("b", "seat_b", "right", status="UNKNOWN")],
        [_shared("center")], allow_cross_function_region_sharing=False,
    ).solve()
    assert result["status"] == "INFEASIBLE"


def test_same_evidence_modes_change_only_acceptance_logic():
    marker = _personal("a", "seat_a", "left")
    marker["semantic_role_status"] = "FALSE"
    marker["compatibility_status"] = "FALSE"
    second = _personal("b", "seat_b", "right")
    solver = GlobalRegionAllocationSolver(
        [marker, second], [_shared("center")],
        allow_cross_function_region_sharing=False,
    )
    assert solver.solve("geometry_only")["status"] == "COMPLETE"
    assert solver.solve("joint")["status"] == "INFEASIBLE"


def test_set_fit_rejects_area_only_false_positive():
    measured = lambda value: {"value": value, "status": "MEASURED"}
    payloads = [
        {"footprint_length_m": measured(0.28), "footprint_width_m": measured(0.12)},
        {"footprint_length_m": measured(0.28), "footprint_width_m": measured(0.12)},
    ]
    region = {
        "support_length_m": measured(0.31),
        "support_width_m": measured(0.31),
    }
    relation = evaluate_fits_set_on(payloads, region, task_config=TASK)
    assert relation["status"] == "FALSE"
    assert relation["signed_clearance_margin_m"] < 0


def test_resolved_proposals_are_opaque_and_do_not_encode_validity(tmp_path):
    path = write_resolved_integrated_rig(L2_INTEGRATED_SCENES[0], tmp_path / "rig.yaml")
    text = path.read_text(encoding="utf-8")
    assert "region_selectors" in text
    assert "expected_valid" not in text
    assert "PERSONAL_REFRESHMENT_REGION" not in text
    assert "SHARED_CONTROLS_REGION" not in text


def test_production_module_does_not_import_oracle_or_planning():
    source = inspect.getsource(__import__("mujoco_scenes.living_room_region_function", fromlist=["x"]))
    assert "living_room_region_oracle" not in source
    assert "problem.pddl" not in source
    assert "PICK(" not in source
    assert "PLACE(" not in source


def test_oracle_is_marked_privileged_and_expected_variant_family_is_balanced():
    assert ORACLE_MARKER == "PRIVILEGED_ORACLE_EVALUATION_ONLY"
    assert set(EXPECTED_VARIANTS) == {variant_code(name) for name in L2_INTEGRATED_SCENES}
    assert set(EXPECTED_VARIANTS.values()) == {"COMPLETE", "INFEASIBLE"}


def test_integrated_runtime_declares_single_initial_stage():
    source = inspect.getsource(IntegratedLivingRoomRegionRun)
    assert '"perception_stage_count": 1' in source
    assert "inspect_sequence" not in source


def test_one_initial_rgbd_capture_observes_all_entities(tmp_path, monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    scene = L2LivingRoomRegionScene(L2_INTEGRATED_SCENES[0], robot="none")
    rig = write_resolved_integrated_rig(scene.scene_name, tmp_path / "rig.yaml")
    observation = InitialEvidenceCapture(
        scene,
        rig_config=rig,
        task_config=TASK,
        width=320,
        height=240,
    ).capture(tmp_path / "observation")
    assert len(observation.cameras) == 5
    assert len(observation.payloads) == 6
    assert len(observation.seats) == 2
    assert sorted(observation.seats) == ["seat_0001", "seat_0002"]
    assert len(observation.regions) == 5
    metadata = (tmp_path / "observation" / "inspection_metadata.json").read_text()
    assert "INITIAL_SINGLE_MULTI_VIEW_CAPTURE" in metadata


@pytest.mark.parametrize("scene_name", L2_INTEGRATED_SCENES)
def test_privileged_oracle_matches_curated_physical_outcome(scene_name):
    scene = L2LivingRoomRegionScene(scene_name, robot="none")
    oracle = evaluate_privileged_oracle(scene, TASK)
    assert oracle["artifact_classification"] == ORACLE_MARKER
    assert oracle["production_consumed_this_artifact"] is False
    assert oracle["status"] == EXPECTED_VARIANTS[variant_code(scene_name)]
