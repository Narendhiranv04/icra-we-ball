from __future__ import annotations

import inspect
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_scenes.living_room_region_scene import (
    L2_ABLATION1_SCENES,
    L2_ABLATION3_SCENES,
    L2_SCENES,
    L2LivingRoomRegionScene,
    build_l2_region_xml,
)
from mujoco_scenes.region_ablation import (
    DEFAULT_TASK_CONFIG,
    PersistentRegionRegistry,
    RegionAblationRun,
)
from mujoco_scenes.region_grounding import (
    REGION_MEASUREMENT_PURPOSE,
    REGION_VISUALIZATION_PURPOSE,
    PayloadMeasurementEvidence,
    L2RegionEvidenceCapture,
    RegionCameraCapture,
    RegionMeasurementEvidence,
    RegionStageCapture,
    _single_free_rigid_instance_geom_ids,
    evaluate_fits_on,
    evaluate_near_seating_area,
    extract_payload_properties,
    extract_region_properties,
    load_region_task,
    require_region_measurement_evidence,
    run_region_semantics,
    semantic_region_role_status,
)
from mujoco_scenes.semantic_grounding import Detection, load_semantic_config
from mujoco_scenes.generate_region_ablation_report import _data_uri, _write_html


TASK = load_region_task(DEFAULT_TASK_CONFIG)


def _horizontal_points(length=0.7, width=0.5, z=0.5, count=1600):
    rng = np.random.default_rng(22)
    return np.column_stack(
        (
            rng.uniform(-length / 2, length / 2, count),
            rng.uniform(-width / 2, width / 2, count),
            rng.normal(z, 0.0006, count),
        )
    ).astype(np.float32)


def _region_evidence(
    points,
    *,
    purpose=REGION_MEASUREMENT_PURPOSE,
    quality=True,
    path="stages/000/region_evidence/fused.ply",
):
    points = np.asarray(points, np.float32)
    colors = np.full((len(points), 3), (170, 130, 70), np.uint8)
    cameras = ("inspection_left", "inspection_right", "inspection_top")
    return RegionMeasurementEvidence(
        measurement_points=points,
        measurement_colors=colors,
        points_by_camera={camera: points.copy() for camera in cameras},
        source_stage=0,
        inspection_label="CONFIGURED_LABEL_NOT_A_SEMANTIC",
        measurement_cloud_path=path,
        contributing_camera_ids=cameras,
        measurement_quality={
            "quality_is_valid": quality,
            "point_count": len(points),
            "contributing_camera_count": len(cameras),
        },
        cloud_purpose=purpose,
    )


def _payload_evidence(length=0.42, width=0.28):
    points = _horizontal_points(length, width, z=0.65, count=1200)
    colors = np.full((len(points), 3), (220, 100, 30), np.uint8)
    cameras = ("inspection_left", "inspection_right", "inspection_top")
    return PayloadMeasurementEvidence(
        measurement_points=points,
        measurement_colors=colors,
        points_by_camera={camera: points.copy() for camera in cameras},
        source_stage=0,
        measurement_cloud_path="stages/000/payload_evidence/fused.ply",
        contributing_camera_ids=cameras,
        measurement_quality={
            "quality_is_valid": True,
            "point_count": len(points),
            "contributing_camera_count": len(cameras),
        },
    )


def test_region_extractor_requires_fresh_typed_evidence():
    with pytest.raises(TypeError):
        require_region_measurement_evidence(np.zeros((20, 3)))
    with pytest.raises(ValueError):
        extract_region_properties(
            _region_evidence(
                _horizontal_points(),
                purpose=REGION_VISUALIZATION_PURPOSE,
            ),
            task_config=TASK,
        )


@pytest.mark.parametrize(
    "path",
    (
        "objects/region_0001/cumulative.ply",
        "room/combined_cloud.ply",
    ),
)
def test_region_measurement_api_does_not_accept_historical_cloud_purpose(path):
    evidence = _region_evidence(
        _horizontal_points(), purpose=REGION_VISUALIZATION_PURPOSE, path=path
    )
    with pytest.raises(ValueError):
        extract_region_properties(evidence, task_config=TASK)


@pytest.mark.parametrize(
    "path",
    (
        "regions/region_0001/cumulative.ply",
        "stages/000/combined_cloud.ply",
        "stages/000/full_room_cloud.ply",
    ),
)
def test_measurement_marker_cannot_disguise_historical_region_path(path):
    with pytest.raises(ValueError):
        extract_region_properties(
            _region_evidence(_horizontal_points(), path=path),
            task_config=TASK,
        )


def test_planar_support_is_measured_from_points_not_inspection_aabb():
    properties = extract_region_properties(
        _region_evidence(_horizontal_points(0.72, 0.48)),
        task_config=TASK,
    )
    assert properties["PLANAR_SUPPORT"]["value"] is True
    assert properties["support_length_m"]["value"] == pytest.approx(
        0.69, abs=0.035
    )
    assert properties["support_width_m"]["value"] == pytest.approx(
        0.46, abs=0.035
    )
    assert "inspection_volume" not in properties


def test_nonhorizontal_support_fails_planarity_predicate():
    points = _horizontal_points()
    points[:, 2] += 0.8 * points[:, 0]
    properties = extract_region_properties(
        _region_evidence(points), task_config=TASK
    )
    assert properties["PLANAR_SUPPORT"]["value"] is False


def test_insufficient_region_evidence_returns_unknown():
    properties = extract_region_properties(
        _region_evidence(_horizontal_points(), quality=False),
        task_config=TASK,
    )
    assert properties["property_status"] == "UNKNOWN"
    assert properties["PLANAR_SUPPORT"]["status"] == "UNKNOWN"


def test_payload_footprint_and_fit_orientations_are_measured():
    payload = extract_payload_properties(_payload_evidence())
    region = extract_region_properties(
        _region_evidence(_horizontal_points(0.60, 0.46)),
        task_config=TASK,
    )
    relation = evaluate_fits_on(payload, region, task_config=TASK)
    assert relation["status"] == "TRUE"
    assert {item["orientation_degrees"] for item in relation["tested_orientations"]} == {
        0,
        90,
    }
    assert relation["signed_fit_margin_m"] > 0.0


def test_small_support_has_robust_negative_fit_margin():
    payload = extract_payload_properties(_payload_evidence())
    region = extract_region_properties(
        _region_evidence(_horizontal_points(0.30, 0.26)),
        task_config=TASK,
    )
    relation = evaluate_fits_on(payload, region, task_config=TASK)
    assert relation["status"] == "FALSE"
    assert relation["signed_fit_margin_m"] < -0.08


def test_missing_fit_operand_is_unknown():
    relation = evaluate_fits_on({}, {}, task_config=TASK)
    assert relation["status"] == "UNKNOWN"
    assert relation["value"] is None


def test_near_seating_uses_observed_points_and_preserves_margin():
    region = extract_region_properties(
        _region_evidence(_horizontal_points()), task_config=TASK
    )
    sofa = _horizontal_points(0.8, 0.3, z=0.55)
    sofa[:, 0] += 0.5
    relation = evaluate_near_seating_area(region, sofa, task_config=TASK)
    assert relation["status"] == "TRUE"
    assert relation["signed_margin_m"] == pytest.approx(
        relation["maximum_distance_m"] - relation["measured_distance_m"]
    )
    assert "simulator_pose" not in relation


def test_region_registry_assigns_generic_persistent_ids():
    registry = PersistentRegionRegistry()
    properties = extract_region_properties(
        _region_evidence(_horizontal_points()), task_config=TASK
    )
    arguments = dict(
        inspection_label="SOFA_SEAT_PATCH",
        properties=properties,
        semantic_context={"parent_furniture": {"canonical_label": "rug"}},
        functional_evaluation={},
        evidence_path="stages/000/region_evidence/fused.ply",
        contributing_cameras=("left", "right", "top"),
        point_count=1600,
    )
    first, discovered = registry.update(stage=0, **arguments)
    second, rediscovered = registry.update(stage=1, **arguments)
    assert first == second == "region_0001"
    assert discovered is True
    assert rediscovered is False
    assert registry.records[first]["identity"]["observation_count"] == 2
    assert registry.records[first]["identity"]["entity_type"] == "region"
    assert registry.records[first]["semantic_context"]["parent_furniture"][
        "canonical_label"
    ] == "rug"


def test_configured_label_is_provenance_not_semantic_evidence():
    registry = PersistentRegionRegistry()
    properties = extract_region_properties(
        _region_evidence(_horizontal_points()), task_config=TASK
    )
    region_id, _ = registry.update(
        stage=0,
        inspection_label="COFFEE_TABLE",
        properties=properties,
        semantic_context={"parent_furniture": {"status": "UNKNOWN"}},
        functional_evaluation={},
        evidence_path="fused.ply",
        contributing_cameras=("a", "b", "c"),
        point_count=1600,
    )
    record = registry.records[region_id]
    assert record["semantic_context"]["parent_furniture"]["status"] == "UNKNOWN"
    assert (
        record["provenance"]["configured_inspection_label"]
        == "COFFEE_TABLE"
    )


def test_region_semantic_role_is_tri_state():
    assert semantic_region_role_status({}, TASK)["status"] == "UNKNOWN"
    assert semantic_region_role_status(
        {"status": "SUPPORTED", "canonical_label": "rug"}, TASK
    )["status"] == "FALSE"
    assert semantic_region_role_status(
        {"status": "SUPPORTED", "canonical_label": "coffee_table"}, TASK
    )["status"] == "TRUE"


class _RgbOnlyDetector:
    name = "mock_rgb_detector"
    checkpoint = "mock.pt"
    version = "1"

    def __init__(self):
        self.inputs = []

    def detect(self, image, vocabulary):
        self.inputs.append((np.asarray(image).copy(), tuple(vocabulary)))
        return (
            Detection("coffee table", "coffee table", 0.9, (5, 5, 35, 35)),
            Detection("serving tray", "serving tray", 0.9, (40, 5, 65, 30)),
            Detection("sofa", "sofa", 0.9, (5, 40, 55, 62)),
        )


def test_region_semantics_associate_rgb_boxes_to_projected_masks(tmp_path):
    masks = []
    for bounds in ((5, 5, 35, 35), (40, 5, 65, 30), (5, 40, 55, 62)):
        mask = np.zeros((70, 70), bool)
        mask[bounds[1] : bounds[3], bounds[0] : bounds[2]] = True
        masks.append(mask)
    camera = RegionCameraCapture(
        camera_id="inspection_front",
        model_camera_name="camera",
        rgb=np.zeros((70, 70, 3), np.uint8),
        depth_m=np.ones((70, 70), np.float32),
        segmentation=np.zeros((70, 70, 2), np.int32),
        intrinsics=np.eye(3),
        position_world_m=np.zeros(3),
        rotation_world_from_camera=np.eye(3),
        validation={"usable": True},
        region_mask=masks[0],
        payload_mask=masks[1],
        sofa_mask=masks[2],
        region_points=np.zeros((20, 3)),
        region_colors=np.zeros((20, 3), np.uint8),
        payload_points=np.zeros((20, 3)),
        payload_colors=np.zeros((20, 3), np.uint8),
        sofa_points=np.zeros((20, 3)),
    )
    evidence = _region_evidence(_horizontal_points())
    payload = _payload_evidence()
    stage = RegionStageCapture(
        stage=0,
        inspection_label="DO_NOT_USE_AS_LABEL",
        cameras={"inspection_front": camera},
        region_evidence=evidence,
        payload_evidence=payload,
        sofa_points=np.zeros((20, 3)),
    )
    config = load_semantic_config(
        vocabulary_path=(
            Path(__file__).parents[1]
            / "configs"
            / "l2_region_semantic_vocabulary.yaml"
        )
    )
    task = {**TASK, "semantic_requirements": {**TASK["semantic_requirements"]}}
    task["semantic_requirements"]["minimum_supporting_views"] = 1
    task["semantic_requirements"]["minimum_winning_score_margin"] = 0.0
    detector = _RgbOnlyDetector()
    records = run_region_semantics(
        stage,
        detector=detector,
        semantic_config=config,
        task_config=task,
        stage_dir=tmp_path,
    )
    assert records["region_parent"]["canonical_label"] == "coffee_table"
    assert records["payload"]["canonical_label"] == "serving_tray"
    assert records["seating"]["canonical_label"] == "sofa"
    assert detector.inputs[0][0].shape == (70, 70, 3)
    assert "DO_NOT_USE_AS_LABEL" not in detector.inputs[0][1]


@pytest.mark.parametrize("scene_name", L2_SCENES)
def test_l2_variants_compile_without_robot(scene_name):
    model = mujoco.MjModel.from_xml_string(
        build_l2_region_xml(scene_name, "none")
    )
    assert model.ncam >= 5
    if scene_name in L2_ABLATION1_SCENES:
        assert (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "l2_refreshment_tray"
            )
            >= 0
        )
    elif scene_name not in L2_ABLATION3_SCENES:
        assert sum(
            model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            for joint_id in range(model.njnt)
        ) in {5, 6}
    else:
        assert sum(
            model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
            for joint_id in range(model.njnt)
        ) == 4


@pytest.mark.parametrize("robot", ("none", "google"))
def test_payload_segmentation_instance_is_selected_without_a_body_name(robot):
    scene = L2LivingRoomRegionScene(L2_SCENES[0], robot=robot)
    geom_ids = _single_free_rigid_instance_geom_ids(scene.model)
    # The realistic loaded-tray payload retains five analytic collision geoms
    # and adds scanned tray, mug, and bowl visual geoms to the same free body.
    assert len(geom_ids) == 8
    owning_bodies = {
        int(scene.model.geom_bodyid[geom_id]) for geom_id in geom_ids
    }
    assert len(owning_bodies) == 1


def test_runtime_payload_capture_does_not_consume_simulator_names():
    source = inspect.getsource(L2RegionEvidenceCapture.capture)
    assert "payload_instance_name" not in source
    assert "mjOBJ_BODY" not in source


def test_l2_primary_compiles_with_google_robot():
    scene = L2LivingRoomRegionScene(L2_SCENES[0], robot="google")
    assert scene.has_robot is True
    assert (
        mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, "google:base_link"
        )
        >= 0
    )


def _functional(semantic, fits, joint):
    return {
        "semantic_role": {"status": semantic},
        "seating_semantics": {"status": "TRUE"},
        "PLANAR_SUPPORT": {"tri_state": "TRUE"},
        "FITS_ON": {
            "status": fits,
            "payload_length_m": 0.40,
            "payload_width_m": 0.28,
            "tested_orientations": [],
            "selected_orientation_degrees": 0,
            "signed_fit_margin_m": 0.1 if fits == "TRUE" else -0.1,
        },
        "NEAR_SEATING_AREA": {
            "status": "TRUE",
            "measured_distance_m": 1.0,
            "signed_margin_m": 0.4,
        },
        "geometry_only_status": "TRUE" if fits == "TRUE" else "FALSE",
        "semantic_only_status": semantic,
        "joint_status": joint,
        "rejection_reason": None if joint == "TRUE" else "rejected",
    }


def _stage(stage, region_id, label, semantic, fits, joint):
    properties = extract_region_properties(
        _region_evidence(_horizontal_points()), task_config=TASK
    )
    return {
        "stage": stage,
        "stage_name": f"{stage:03d}_{label}",
        "configured_inspection_label": label,
        "region_id": region_id,
        "geometric_properties": properties,
        "semantic_context": {
            "parent_furniture": {
                "canonical_label": {
                    "SOFA_SEAT_PATCH": "sofa",
                    "SMALL_SIDE_TABLE": "side_table",
                    "COFFEE_TABLE": "coffee_table",
                }[label],
                "confidence": 0.8,
                "supporting_view_count": 3,
            }
        },
        "functional_evaluations": _functional(
            semantic, fits, joint
        ),
        "evidence_path": f"stages/{stage:03d}/region_evidence/fused.ply",
        "semantic_overview_path": f"stages/{stage:03d}/semantic_overview.png",
    }


def test_same_evidence_modes_select_expected_counterexamples(tmp_path):
    run = RegionAblationRun(
        tmp_path / "run",
        scene_name=L2_SCENES[0],
        width=64,
        height=48,
    )
    run.stage_records = [
        _stage(
            0, "region_0001", "SOFA_SEAT_PATCH", "FALSE", "TRUE", "FALSE"
        ),
        _stage(
            1,
            "region_0002",
            "SMALL_SIDE_TABLE",
            "TRUE",
            "FALSE",
            "FALSE",
        ),
        _stage(2, "region_0003", "COFFEE_TABLE", "TRUE", "TRUE", "TRUE"),
    ]
    run.production_status = "COMPLETE"
    summary = run.evaluate_same_evidence()
    assert summary["modes"]["geometry_only"]["selected_region_id"] == "region_0001"
    assert summary["modes"]["semantic_only"]["selected_region_id"] == "region_0002"
    assert summary["modes"]["joint"]["selected_region_id"] == "region_0003"
    assert summary["rerendered_for_diagnostics"] is False


def test_living_room_presentation_report_exposes_complete_component_audit(
    tmp_path,
):
    row = {
        "region_id": "region_0001",
        "parent_semantic_label": "coffee_table",
        "semantic_confidence": 0.8,
        "semantic_supporting_views": 5,
        "support_length_m": 0.7,
        "support_width_m": 0.5,
        "fit_margin_m": 0.1,
        "semantic_role_status": "TRUE",
        "PLANAR_SUPPORT": "TRUE",
        "FITS_ON": "TRUE",
        "NEAR_SEATING_AREA": "TRUE",
        "geometry_only_status": "TRUE",
        "semantic_only_status": "TRUE",
        "joint_status": "TRUE",
        "rejection_reason": None,
        "discovery_stage": 0,
    }
    modes = {
        mode: {
            "status": "COMPLETE",
            "selected_region_id": "region_0001",
            "completion_stage": 0,
        }
        for mode in ("geometry_only", "semantic_only", "joint")
    }
    region = {
        "geometric_properties": {
            "support_area_m2": {"value": 0.35},
            "planarity_score": {"value": 0.99},
        },
        "provenance": {
            "point_count": 1000,
            "contributing_camera_ids": ["inspection_front"],
            "measurement_cloud_path": "stages/000/region_evidence/fused.ply",
        },
        "functional_evaluations": {
            "NEAR_SEATING_AREA": {"measured_distance_m": 1.0}
        },
    }
    payload = {
        "identity": {"object_id": "object_0001"},
        "semantic_context": {"canonical_label": "serving_tray"},
        "geometric_properties": {
            "footprint_length_m": {"value": 0.4},
            "footprint_width_m": {"value": 0.28},
            "footprint_area_m2": {"value": 0.112},
            "measurement_quality": {
                "point_count": 500,
                "contributing_camera_count": 5,
            },
        },
        "provenance": {
            "measurement_cloud_path": "stages/000/payload_evidence/fused.ply"
        },
    }
    stage = {
        "title": "Stage 000: coffee_table",
        "overview": "stages/000/overview.png",
        "semantic": "stages/000/semantic_overview.png",
        "pointcloud": "stages/000/region_pointcloud.png",
        "graph": "stages/000/graph.png",
        "mask": "stages/000/evidence_masks.png",
        "camera_overlays": [
            {
                "camera_id": f"inspection_{name}",
                "consensus": f"stages/000/cameras/{name}/consensus.png",
                "raw": f"stages/000/cameras/{name}/overlay.png",
            }
            for name in ("left", "right", "top", "front", "close")
        ],
    }
    animations = {
        mode: {
            "gif": f"{mode}.gif",
            "mp4": f"{mode}.mp4",
        }
        for mode in modes
    }
    _write_html(
        report_dir=tmp_path,
        run_config={
            "scene_name": "living_room_test",
            "natural_language_goal": "Place the tray.",
            "capture_resolution": [1280, 960],
            "detector": {
                "name": "yolo_world",
                "checkpoint": "weights.pt",
                "version": "test",
                "device": "cpu",
            },
        },
        summary={"modes": modes},
        rows=[row],
        stage_assets=[stage],
        region_registry={"regions": {"region_0001": region}},
        payload_registry={"objects": {"object_0001": payload}},
        handoff=None,
        mode_animations=animations,
    )
    report = (tmp_path / "presentation_report.html").read_text()
    redirect = (tmp_path / "ablation_report.html").read_text()
    assert "Three individual policy ablations" in report
    assert "How the geometric checks are defined" in report
    assert "Rendered scene and component audit" in report
    assert "five RGB detections" in report
    assert report.count("detector overlay") == 5
    assert "RegionMeasurementEvidence" in report
    assert "presentation_report.html" in redirect


def test_living_room_report_embeds_displayed_media(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nportable-report-test")
    uri = _data_uri(tmp_path, "frame.png")
    assert uri.startswith("data:image/png;base64,")
    assert "frame.png" not in uri
