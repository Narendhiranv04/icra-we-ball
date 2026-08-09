from __future__ import annotations

import inspect
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from mujoco_scenes.living_room_region_function import (
    DEFAULT_TASK_CONFIG,
    EXPECTED_VARIANTS,
    GlobalRegionAllocationSolver,
    IntegratedLivingRoomRegionRun,
    REGION_PROPOSAL_PROVENANCE,
    load_integrated_task,
    variant_code,
    write_resolved_integrated_rig,
)
from mujoco_scenes.living_room_robot_spawn_validation import (
    validate_google_robot_spawn,
)
from mujoco_scenes.living_room_region_oracle import (
    ORACLE_MARKER,
    evaluate_privileged_oracle,
)
from mujoco_scenes.living_room_region_scene import (
    INTEGRATED_ROOM_LAYOUT,
    L2_INTEGRATED_GOAL,
    L2_INTEGRATED_SCENES,
    L2LivingRoomRegionScene,
    build_l2_region_xml,
)
from mujoco_scenes.region_ablation2 import InitialEvidenceCapture, evaluate_fits_set_on


TASK = load_integrated_task()
ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "living_room_realistic"


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
    assert "expected_valid:" not in text
    assert "PERSONAL_REFRESHMENT_REGION" not in text
    assert "SHARED_CONTROLS_REGION" not in text
    config = yaml.safe_load(text)
    assert config["region_proposal_provenance"] == REGION_PROPOSAL_PROVENANCE


def test_neutral_proposal_provenance_explicitly_disclaims_privileged_meaning():
    assert REGION_PROPOSAL_PROVENANCE == {
        "region_proposal_source": "SIMULATOR_DERIVED_NEUTRAL_SPATIAL_GATE",
        "region_proposal_encodes_function": False,
        "region_proposal_encodes_semantic_class": False,
        "region_proposal_encodes_expected_validity": False,
        "region_dimensions_for_functional_reasoning": "OBSERVED_RGBD_POINT_CLOUD",
    }


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
    metadata_record = json.loads(metadata)
    assert (
        metadata_record["region_proposal_provenance"]
        == REGION_PROPOSAL_PROVENANCE
    )


@pytest.mark.parametrize("scene_name", L2_INTEGRATED_SCENES)
def test_privileged_oracle_matches_curated_physical_outcome(scene_name):
    scene = L2LivingRoomRegionScene(scene_name, robot="none")
    oracle = evaluate_privileged_oracle(scene, TASK)
    assert oracle["artifact_classification"] == ORACLE_MARKER
    assert oracle["production_consumed_this_artifact"] is False
    assert oracle["status"] == EXPECTED_VARIANTS[variant_code(scene_name)]


def test_external_asset_manifest_is_complete_cc0_and_hash_verified():
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert len(manifest["assets"]) == 5
    assert {item["source"] for item in manifest["assets"]} == {"Poly Haven"}
    assert {item["license"] for item in manifest["assets"]} == {"CC0-1.0"}
    assert all(item["source_author"] for item in manifest["assets"])
    for asset in manifest["assets"]:
        assert asset["download_date"] == "2026-08-09"
        assert asset["scene_instances"]
        for part in asset["processed_parts"]:
            mesh = Path(__file__).resolve().parents[1] / part["processed_filename"]
            texture = Path(__file__).resolve().parents[1] / part["texture_file"]
            assert hashlib.sha256(mesh.read_bytes()).hexdigest() == part["processed_sha256"]
            assert hashlib.sha256(texture.read_bytes()).hexdigest() == part["texture_sha256"]


def test_integrated_room_is_sparse_scaled_and_has_canonical_spawn():
    root = build_l2_region_xml(L2_INTEGRATED_SCENES[0], "none")
    assert "integrated_realistic_visual" in root
    assert "l2_canonical_robot_spawn" in root
    assert INTEGRATED_ROOM_LAYOUT["chair_left"][0] < -1.0
    assert INTEGRATED_ROOM_LAYOUT["chair_right"][0] > 1.0
    assert INTEGRATED_ROOM_LAYOUT["staging_table"][1] < -1.5
    assert INTEGRATED_ROOM_LAYOUT["media_console"][1] > 2.0


def test_f0_accent_and_i3_control_use_natural_isotropic_wooden_table():
    for scene_name, body_name, top_name in (
        (L2_INTEGRATED_SCENES[0], "a2_shared_drink_trap", "a2_shared_drink_top"),
        (
            next(name for name in L2_INTEGRATED_SCENES if "I3_SHARED_FIT_FAILURE" in name),
            "a2_control_table",
            "a2_control_table_top",
        ),
    ):
        root = ET.fromstring(build_l2_region_xml(scene_name, "none"))
        top = root.find(f".//geom[@name='{top_name}']")
        half = np.fromstring(top.get("size"), sep=" ")
        assert half[0] == pytest.approx(0.150)
        assert half[1] == pytest.approx(0.150)
        body = root.find(f".//body[@name='{body_name}']")
        visuals = [
            geom for geom in body.findall("geom")
            if "integrated_realistic_visual" in geom.get("name", "")
        ]
        assert visuals
        for visual in visuals:
            mesh = root.find(f".//mesh[@name='{visual.get('mesh')}']")
            scale = np.fromstring(mesh.get("scale"), sep=" ")
            assert np.max(scale) / np.min(scale) <= 1.15
            assert np.allclose(scale, [1.0, 1.0, 1.0])


def test_google_robot_f0_spawn_has_clearance_and_faces_workspace():
    validation = validate_google_robot_spawn(L2_INTEGRATED_SCENES[0])
    assert validation["all_passed"] is True
    assert validation["workspace_facing_alignment"] == pytest.approx(1.0)
    assert validation["minimum_static_clearance_m"] >= 0.10
    assert validation["invalid_robot_furniture_contacts"] == []


def test_integrated_payloads_are_spaced_without_xy_overlap():
    scene = L2LivingRoomRegionScene(L2_INTEGRATED_SCENES[0], robot="none")
    body_names = (
        "a2_drink_left", "a2_snack_left", "a2_drink_right",
        "a2_snack_right", "a2_remote_payload", "a2_controller_payload",
    )
    centers = []
    for name in body_names:
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
        centers.append(scene.data.xpos[body_id, :2].copy())
    distances = [
        float(np.linalg.norm(first - second))
        for index, first in enumerate(centers)
        for second in centers[index + 1 :]
    ]
    assert min(distances) >= 0.20


@pytest.mark.parametrize("scene_name", L2_INTEGRATED_SCENES)
def test_candidate_supports_do_not_intersect_each_other_or_seating(scene_name):
    root = ET.fromstring(build_l2_region_xml(scene_name, "none"))
    support_names = (
        "a2_personal_left_top", "a2_personal_right_top",
        "a2_shared_drink_top", "a2_control_table_top",
    )
    seat_names = tuple(
        f"a2_seat_{side}_{part}"
        for side in ("left", "right")
        for part in ("base", "cushion", "back", "arm_l", "arm_r")
    )

    def rectangle(name):
        geom = root.find(f".//geom[@name='{name}']")
        center = np.fromstring(geom.get("pos"), sep=" ")[:2]
        half = np.fromstring(geom.get("size"), sep=" ")[:2]
        return center - half, center + half

    supports = {name: rectangle(name) for name in support_names}
    seats = {name: rectangle(name) for name in seat_names}

    def overlaps(first, second):
        return bool(
            np.all(first[0] < second[1] - 1e-6)
            and np.all(second[0] < first[1] - 1e-6)
        )

    support_items = list(supports.items())
    for index, (first_name, first) in enumerate(support_items):
        for second_name, second in support_items[index + 1 :]:
            assert not overlaps(first, second), (first_name, second_name)
        for seat_name, seat in seats.items():
            assert not overlaps(first, seat), (first_name, seat_name)


def test_integrated_camera_rig_has_five_distinct_calibrated_views(tmp_path):
    rig_paths = []
    ranks = []
    for scene_name in L2_INTEGRATED_SCENES:
        path = write_resolved_integrated_rig(scene_name, tmp_path / f"{variant_code(scene_name)}.yaml")
        config = yaml.safe_load(path.read_text())
        rig_paths.append(tuple(config["camera_slots"].values()))
        ranks.append(tuple(row["candidate_rank"] for row in config["region_selectors"].values()))
        assert len(config["capture"]["cameras"]) == 5
        positions = {
            tuple(row["position_world_m"])
            for row in config["capture"]["cameras"].values()
        }
        assert len(positions) == 5
        assert all(row["volume"] for row in config["region_selectors"].values())
    assert len(set(rig_paths)) == 1
    assert len(set(ranks)) == 1


def test_f6_surplus_is_physically_distinct_and_has_more_oracle_solutions():
    by_code = {
        variant_code(name): evaluate_privileged_oracle(
            L2LivingRoomRegionScene(name, robot="none"), TASK
        )
        for name in L2_INTEGRATED_SCENES
        if variant_code(name) in {"F0_BASE", "F6_DECOY_SURPLUS"}
    }
    assert by_code["F6_DECOY_SURPLUS"]["complete_solution_count"] > by_code["F0_BASE"]["complete_solution_count"]


def test_oracle_payload_footprints_come_from_instantiated_geometry():
    source = inspect.getsource(
        __import__("mujoco_scenes.living_room_region_oracle", fromlist=["x"])
    )
    assert "_body_collision_footprint" in source
    assert "instantiated_collision_geom_world_aabb_evaluation_only" in source
    assert "PAYLOAD_FOOTPRINT" not in source


def test_benchmark_runs_production_before_privileged_oracle():
    source = inspect.getsource(
        __import__("mujoco_scenes.run_living_room_region_benchmark", fromlist=["x"]).main
    )
    production_position = source.index(").run(scene)")
    oracle_position = source.index("evaluate_privileged_oracle(scene, task)")
    assert production_position < oracle_position
    assert "evaluation_order.json" in source


def test_task_payload_grouping_is_observation_based_not_id_order():
    source = inspect.getsource(IntegratedLivingRoomRegionRun._build_compatibility)
    assert "observed_centroid_world_m" in source
    assert "itertools.permutations" in source
    assert "MINIMUM_TOTAL_OBSERVED_CENTROID_DISTANCE" == TASK["payload_groups"]["personal_refreshment_sets"]["grouping_policy"]
