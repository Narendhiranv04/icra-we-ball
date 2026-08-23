"""Unit and regression tests for Workshop (W1) Phase 1 Frozen Architecture."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import yaml

from mujoco_scenes.workshop_scene import (
    WorkshopScene, privileged_validate_variant_feasibility,
)
from mujoco_scenes.workshop_phase1.capture import MultiViewCameraRig
from mujoco_scenes.workshop_phase1.types import (
    AblationType,
    EntityType,
    FunctionalRequirement,
    FunctionGroundingResult,
    FunctionalWitness,
    GroundingStatus,
    MaskBackendType,
    ObservedMask,
    ObservedObjectTrack,
    ObservedRegion,
    ProposalMode,
    RequirementSource,
    SemanticBackendType,
    TargetGeometryEvidence,
    ViewObservation,
    combine_status,
)
from mujoco_scenes.workshop_phase1.perception import (
    PrivilegedOracleMaskBackend,
    RGBDConnectedComponentProposalBackend,
    YOLOWorldProposalBackend,
)
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker
from mujoco_scenes.workshop_phase1.evidence_graph import GrowingObservedGraph
from mujoco_scenes.workshop_phase1.semantic_grounding import (
    ObjectSemanticBackend,
    PrivilegedOracleSemanticBackend,
    ProductionSemanticBackend,
    SemanticGrounder,
)
from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.region_grounding import RegionGrounder
from mujoco_scenes.workshop_phase1.functional_search import FunctionalSatisfactionSearch
from mujoco_scenes.workshop_phase1.inspection_controller import WorkshopPhase1InspectionController
from mujoco_scenes.workshop_phase1.requirements import (
    ManualWorkshopFMContract,
    StaticWorkshopRequirementProvider,
)
from mujoco_scenes.workshop_phase1.serialization import (
    assert_no_backend_names,
    sanitize_production_data,
)


def test_fm_contract_loading_and_neutrality():
    """Verify standard FM contract defines requirements and broad vocabulary without backend strings."""
    contract = ManualWorkshopFMContract()
    reqs = contract.get_requirements()
    assert len(reqs) == 2

    names = {r.function_name for r in reqs}
    assert "CAN_DRIVE_SCREW" in names
    assert "CAN_FASTEN" in names
    expected_relations = {
        "CAN_DRIVE_SCREW": {"REACHES_TARGET", "COMPATIBLE_WITH"},
        "CAN_FASTEN": {"COMPATIBLE_WITH_TARGET"},
    }
    for requirement in reqs:
        assert set(requirement.required_relations) == expected_relations[requirement.function_name]
        assert requirement.geometric_constraints == {}

    prompts = contract.get_detector_prompts()
    assert len(prompts) == 4
    assert len(prompts) == len(contract.get_semantic_vocabulary())
    assert prompts == [entry["detector_label"] for entry in contract.get_ranked_detector_vocabulary()]
    for alias in ("manual screwdriver", "powered screwdriver", "cordless drill"):
        assert alias not in prompts
    assert prompts == ["screwdriver", "power drill", "screw", "wooden hammer"]
    assert contract.get_detector_label_to_canonical_map()["screw"] == "screw"
    assert contract.get_detector_label_to_canonical_map()["wooden hammer"] == "hammer"
    # Detector prompts must be broad physical names, never variant-specific or simulator strings
    for p in prompts:
        assert "workshop_" not in p
        assert "F4" not in p
        assert "I6" not in p


def test_fm_contract_to_yolo_semantic_flow():
    """Mandatory test (Sec 63): Verify FM contract defines accepted categories and evaluates synthetic YOLO detections."""
    contract = ManualWorkshopFMContract()
    reqs = contract.get_requirements()
    driver_req = next(r for r in reqs if r.function_name == "CAN_DRIVE_SCREW")
    fastener_req = next(r for r in reqs if r.function_name == "CAN_FASTEN")

    grounder = SemanticGrounder()

    # Track A: detected as screwdriver
    track_driver = ObservedObjectTrack(
        instance_id="object_0001",
        current_semantic_belief={"canonical_label": "screwdriver", "confidence": 0.85},
    )
    # Track B: detected as wrench
    track_wrench = ObservedObjectTrack(
        instance_id="object_0002",
        current_semantic_belief={"canonical_label": "wrench", "confidence": 0.80},
    )
    # Track C: detected as screw
    track_screw = ObservedObjectTrack(
        instance_id="object_0003",
        current_semantic_belief={"canonical_label": "screw", "confidence": 0.90},
    )

    res_driver = grounder.ground_object_for_requirement(track_driver, driver_req)
    res_wrench = grounder.ground_object_for_requirement(track_wrench, driver_req)
    res_screw = grounder.ground_object_for_requirement(track_screw, fastener_req)

    assert res_driver.semantic_status == GroundingStatus.PASS
    assert res_wrench.semantic_status == GroundingStatus.FAIL
    assert res_screw.semantic_status == GroundingStatus.PASS


def test_changing_fm_contract_changes_detector_vocabulary(tmp_path: Path):
    """Mandatory test (Sec 64): Changing FM contract changes detector vocabulary without modifying perception code."""
    custom_yaml = tmp_path / "custom_contract.yaml"
    custom_data = {
        "task_instruction": "Hold liquids in kitchen",
        "functional_requirements": [
            {
                "requirement_id": "req_hold_liquid",
                "entity_type": "OBJECT",
                "function_name": "HOLD_LIQUID",
                "description": "Container capable of holding liquid",
                "accepted_categories": ["cup", "bowl"],
            }
        ],
        "vocabulary": {
            "canonical_labels": {
                "cup": ["cup", "drinking cup"],
                "bowl": ["bowl", "soup bowl"],
            }
        },
    }
    with open(custom_yaml, "w") as f:
        yaml.dump(custom_data, f)

    contract = ManualWorkshopFMContract(custom_yaml)
    prompts = contract.get_detector_prompts()
    assert "cup" in prompts
    assert "bowl" in prompts
    assert prompts == ["cup", "bowl"]
    assert contract.get_alias_to_canonical_map()["drinking cup"] == "cup"
    assert contract.get_alias_to_canonical_map()["soup bowl"] == "bowl"
    assert "screwdriver" not in prompts


def test_fm_rank_controls_active_yolo_vocabulary_budget(tmp_path: Path):
    config = yaml.safe_load(Path("mujoco_scenes/configs/workshop_phase1.yaml").read_text())
    config["perception"]["max_detector_vocabulary_size"] = 4
    config["perception"]["detector"]["checkpoint"] = str(tmp_path / "missing.pt")
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(config))
    controller = WorkshopPhase1InspectionController(config_path=path)
    assert controller.prompts == ["screwdriver", "power drill", "screw", "wooden hammer"]
    assert controller.proposal_backend._prompts == controller.prompts


def test_missing_or_malformed_manual_contract_fails_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Manual Workshop FM contract"):
        ManualWorkshopFMContract(tmp_path / "missing.yaml")
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("functional_requirements: []\n")
    with pytest.raises(ValueError, match="functional_requirements"):
        ManualWorkshopFMContract(malformed)


def test_no_function_description_prompts():
    """Mandatory test (Sec 65): Detector classes do not contain affordance sentences."""
    contract = ManualWorkshopFMContract()
    prompts = contract.get_detector_prompts()
    forbidden_phrases = ["tool that can", "capable of", "suitable for", "used to secure", "object capable"]
    for p in prompts:
        for f in forbidden_phrases:
            assert f not in p.lower(), f"Affordance prompt found: {p}"


def test_detector_vocabulary_owned_only_by_provider_layer():
    phase1_dir = Path(__file__).resolve().parent.parent / "workshop_phase1"
    for source_path in phase1_dir.glob("*.py"):
        if source_path.name == "requirements.py":
            continue
        source = source_path.read_text(encoding="utf-8")
        assert "set_classes([" not in source
        assert "detector_classes = [" not in source


def test_no_clip_imported_in_phase1():
    """Mandatory test (Sec 66): Workshop Phase 1 production modules do NOT import CLIP."""
    phase1_dir = Path(__file__).resolve().parent.parent / "workshop_phase1"
    for py_file in phase1_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "import clip" not in content, f"CLIP import in {py_file}"
        assert "open_clip" not in content, f"open_clip import in {py_file}"
        assert "CLIPModel" not in content, f"CLIPModel in {py_file}"
        assert "CLIPProcessor" not in content, f"CLIPProcessor in {py_file}"


def test_no_geometry_to_semantic_fabrication():
    """Mandatory test (Sec 67): Semantic grounder does NOT use shape or length thresholds to fabricate labels."""
    sem_file = Path(__file__).resolve().parent.parent / "workshop_phase1" / "semantic_grounding.py"
    content = sem_file.read_text(encoding="utf-8")
    assert "usable_reach_m" not in content
    assert "tip_aspect_ratio" not in content
    assert "min_driver_reach_m" not in content
    assert "min_fastener_length_m" not in content


def test_semantics_does_not_use_geometry():
    """Mandatory test (Sec 18): Semantic check produces identical results regardless of geometric point cloud."""
    grounder = SemanticGrounder()
    contract = ManualWorkshopFMContract()
    driver_req = next(r for r in contract.get_requirements() if r.function_name == "CAN_DRIVE_SCREW")

    identical_cloud = np.ones((50, 3), dtype=np.float32)

    # Identical cloud, different semantic labels
    track_driver = ObservedObjectTrack(
        instance_id="obj_1",
        fused_points=identical_cloud,
        current_semantic_belief={"canonical_label": "screwdriver", "confidence": 0.9},
    )
    track_wrench = ObservedObjectTrack(
        instance_id="obj_2",
        fused_points=identical_cloud,
        current_semantic_belief={"canonical_label": "wrench", "confidence": 0.9},
    )

    res_d = grounder.ground_object_for_requirement(track_driver, driver_req)
    res_w = grounder.ground_object_for_requirement(track_wrench, driver_req)
    assert res_d.semantic_status == GroundingStatus.PASS
    assert res_w.semantic_status == GroundingStatus.FAIL

    # Different clouds (short vs huge), same semantic label
    track_short = ObservedObjectTrack(
        instance_id="obj_3",
        fused_points=np.ones((10, 3)),
        current_semantic_belief={"canonical_label": "screwdriver", "confidence": 0.9},
    )
    track_huge = ObservedObjectTrack(
        instance_id="obj_4",
        fused_points=np.ones((5000, 3)) * 10.0,
        current_semantic_belief={"canonical_label": "screwdriver", "confidence": 0.9},
    )
    res_short = grounder.ground_object_for_requirement(track_short, driver_req)
    res_huge = grounder.ground_object_for_requirement(track_huge, driver_req)
    assert res_short.semantic_status == res_huge.semantic_status == GroundingStatus.PASS


def test_oracle_mask_backend_neutrality():
    """Mandatory test (Sec 68): PrivilegedOracleMaskBackend outputs 'object' and zero semantic typing."""
    scene = WorkshopScene("none", variant="F0_MANUAL_FIRST_ONE_REGION")
    oracle_backend = PrivilegedOracleMaskBackend(scene)
    rig = MultiViewCameraRig(scene=scene)
    obs_list = rig.capture_stage_observations(capture_segmentation=True)

    vol_min = np.array([-1.2, -0.2, 0.35])
    vol_max = np.array([1.2, 1.2, 1.40])
    for obs in obs_list:
        masks = oracle_backend.predict(obs, vol_min, vol_max)
        for m in masks:
            assert m.canonical_label == "object"
            assert m.predicted_label == "object"
            assert m.raw_label == "object"
            assert m.backend_name == "privileged_oracle"


def test_real_target_geometry():
    """Mandatory test (Sec 69): Target geometry is calculated from depth residuals or returns UNKNOWN."""
    scene = WorkshopScene("none", variant="F0_MANUAL_FIRST_ONE_REGION")
    rig = MultiViewCameraRig(scene=scene)
    obs_list = rig.capture_stage_observations()

    target_ev = GeometricGrounder.observe_target_recess(obs_list, scene=scene)
    if target_ev.validity == GroundingStatus.PASS:
        assert target_ev.estimated_recess_depth_m is not None
        assert 0.015 <= target_ev.estimated_recess_depth_m <= 0.050
        assert target_ev.estimated_opening_diameter_m is not None
        assert 0.004 <= target_ev.estimated_opening_diameter_m <= 0.025
    else:
        assert target_ev.validity == GroundingStatus.UNKNOWN
        assert target_ev.estimated_recess_depth_m is None


def test_cavity_openness():
    """The redesigned benchmark exposes only its fixed insertion surface."""
    scene = WorkshopScene("none", variant="F0_MANUAL_FIRST_ONE_REGION")
    rig = MultiViewCameraRig(scene=scene)
    obs_list = rig.capture_stage_observations()
    region_grounder = RegionGrounder()
    regions = region_grounder.discover_candidate_regions(scene, obs_list)

    assert len(regions) == 1
    for reg in regions:
        assert reg.observation_source == "calibrated_spatial_proposal"
        assert reg.region_instance_id.startswith("region_")


def test_tri_state_logic():
    """Mandatory test (Sec 71 & 35): Strict tri-state truth table."""
    # PASS + PASS = PASS
    assert combine_status(GroundingStatus.PASS, GroundingStatus.PASS) == GroundingStatus.PASS
    # PASS + UNKNOWN = UNKNOWN
    assert combine_status(GroundingStatus.PASS, GroundingStatus.UNKNOWN) == GroundingStatus.UNKNOWN
    # UNKNOWN + PASS = UNKNOWN
    assert combine_status(GroundingStatus.UNKNOWN, GroundingStatus.PASS) == GroundingStatus.UNKNOWN
    # UNKNOWN + UNKNOWN = UNKNOWN
    assert combine_status(GroundingStatus.UNKNOWN, GroundingStatus.UNKNOWN) == GroundingStatus.UNKNOWN
    # FAIL + PASS = FAIL
    assert combine_status(GroundingStatus.FAIL, GroundingStatus.PASS) == GroundingStatus.FAIL
    # PASS + FAIL = FAIL
    assert combine_status(GroundingStatus.PASS, GroundingStatus.FAIL) == GroundingStatus.FAIL
    # FAIL + UNKNOWN = FAIL
    assert combine_status(GroundingStatus.FAIL, GroundingStatus.UNKNOWN) == GroundingStatus.FAIL
    # UNKNOWN + FAIL = FAIL
    assert combine_status(GroundingStatus.UNKNOWN, GroundingStatus.FAIL) == GroundingStatus.FAIL
    # FAIL + FAIL = FAIL
    assert combine_status(GroundingStatus.FAIL, GroundingStatus.FAIL) == GroundingStatus.FAIL


def test_zero_detections_returns_insufficient_evidence():
    """Mandatory test (Sec 39): Zero detections + exhaustion returns INSUFFICIENT_EVIDENCE, not GLOBAL_CONFLICT."""
    search = FunctionalSatisfactionSearch()
    diag = search.diagnose_infeasibility(
        all_objects=[],
        all_regions=[],
        driver_candidates=[],
        fastener_candidates=[],
        work_surface_candidates=[],
        parts_container_candidates=[],
        evaluated_tuples=[],
    )
    assert diag == "INSUFFICIENT_EVIDENCE"


def test_no_joint_self_test():
    """Self-test (Sec 87): NO_JOINT_COUPLING selects unary-valid candidates without relational checks."""
    scene = WorkshopScene("none", variant="F0_MANUAL_FIRST_ONE_REGION")
    ctrl_full = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.NONE, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_full = ctrl_full.run_episode()

    # Each episode owns a fresh physical scene; the first controller opens
    # storage while inspecting and must not leak that state into the ablation.
    scene_no_joint = WorkshopScene("none", variant="F0_MANUAL_FIRST_ONE_REGION")
    ctrl_no_joint = WorkshopPhase1InspectionController(scene=scene_no_joint, ablation=AblationType.NO_JOINT_COUPLING, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_no_joint = ctrl_no_joint.run_episode()

    assert res_full.status == "FEASIBLE"
    assert res_no_joint.status == "FEASIBLE"
    assert res_no_joint.witness.verification_details.get("ablation") == "NO_JOINT_COUPLING"
    for relation in ("COMPATIBLE_WITH", "FITS_SET_ON", "FITS_IN"):
        assert ctrl_no_joint.geometric_grounder.relation_call_counts[relation] == 0


def test_semantic_only_self_test():
    """Self-test (Sec 88): SEMANTIC_ONLY bypasses geometry in witness selection."""
    scene = WorkshopScene("none", variant="F1_POWER_FIRST_ONE_REGION")
    ctrl_full = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.NONE, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_full = ctrl_full.run_episode()

    ctrl_sem = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.SEMANTIC_ONLY, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_sem = ctrl_sem.run_episode()

    # The redesigned fixed pair is valid with or without geometric ablation;
    # SEMANTIC_ONLY must still bypass every geometry call.
    assert res_full.status == "FEASIBLE"
    assert res_sem.status == "FEASIBLE"
    assert ctrl_sem.geometric_grounder.total_geometric_calls == 0
    assert all(count == 0 for count in ctrl_sem.geometric_grounder.relation_call_counts.values())


def test_config_wiring(tmp_path: Path):
    """Self-test (Sec 85): YAML config actually controls runtime components."""
    custom_cfg = tmp_path / "custom_config.yaml"
    default_geometry = Path("mujoco_scenes/configs/workshop_geometry_inference.yaml")
    geometry_data = yaml.safe_load(default_geometry.read_text())
    geometry_data["relations"]["grasp_allowance_m"] = 0.081
    geometry_data["relations"]["packing_edge_clearance_m"] = 0.012
    geometry_path = tmp_path / "geometry.yaml"
    geometry_path.write_text(yaml.safe_dump(geometry_data))
    cfg_data = {
        "pipeline": {
            "geometry_config_path": str(geometry_path),
            "fm_contract_path": "mujoco_scenes/configs/workshop_phase1_fm_contract.yaml",
            "requirements_source": "static",
            "inspection_policy": "fixed",
            "mask_backend": "production",
            "semantic_backend": "production",
            "ablation": "none",
            "image_width": 640,
            "image_height": 360,
        },
        "perception": {
            "max_detector_vocabulary_size": 7,
            "volume_margin_m": 0.031,
            "min_points_per_mask": 13,
            "multi_scale": {
                "full_frame": False,
                "stage_crop": True,
                "stage_tiles": True,
                "tile_overlap_fraction": 0.20,
            },
            "detector": {
                "checkpoint": str(tmp_path / "missing.pt"),
                "confidence_threshold": 0.12,
                "nms_iou_threshold": 0.50,
                "inference_size": 512,
                "device": "cpu",
                "max_detections": 17,
            },
            "duplicate_suppression": {
                "box_iou_threshold": 0.61,
                "mask_overlap_threshold": 0.71,
                "centroid_distance_m": 0.014,
                "aabb_overlap_threshold": 0.41,
            }
        },
        "tracking": {
            "cluster_distance_threshold_m": 0.055,
            "track_match_distance_threshold_m": 0.060,
            "voxel_size_m": 0.004,
            "min_cluster_points": 12,
        },
        "inspection": {
            "sequence": ["RIGHT_DRAWER", "LEFT_DRAWER"],
            "early_stop": False,
        }
    }
    with open(custom_cfg, "w") as f:
        yaml.dump(cfg_data, f)

    scene = WorkshopScene("none", variant="F0_MANUAL_FIRST_ONE_REGION")
    ctrl = WorkshopPhase1InspectionController(scene=scene, config_path=custom_cfg)

    assert ctrl.tracker.cluster_distance_threshold_m == 0.055
    assert ctrl.tracker.track_match_distance_threshold_m == 0.060
    assert ctrl.tracker.voxel_size_m == 0.004
    assert ctrl.tracker.min_cluster_points == 12
    assert ctrl.tracker.volume_margin_m == 0.031
    assert ctrl.tracker.min_points_per_mask == 13
    assert ctrl.camera_rig.width == 640 and ctrl.camera_rig.height == 360
    assert ctrl.proposal_backend.confidence_threshold == 0.12
    assert ctrl.proposal_backend.nms_iou_threshold == 0.50
    assert ctrl.proposal_backend.inference_size == 512
    assert ctrl.proposal_backend.max_detections == 17
    assert ctrl.proposal_backend.min_points_per_mask == 13
    assert len(ctrl.prompts) == 4
    assert ctrl.proposal_backend.enable_full_frame is False
    assert ctrl.proposal_backend.enable_stage_crop is True
    assert ctrl.proposal_backend.enable_stage_tiles is True
    assert ctrl.proposal_backend.tile_overlap_fraction == 0.20
    assert ctrl.proposal_backend.duplicate_box_iou_threshold == 0.61
    assert ctrl.geometric_grounder.config["relations"]["grasp_allowance_m"] == 0.081
    assert ctrl.geometric_grounder.config["relations"]["packing_edge_clearance_m"] == 0.012
    assert ctrl.inspection_sequence == ["RIGHT_DRAWER", "LEFT_DRAWER"]
    assert ctrl.early_stop_enabled is False


def _proposal(det_id: str, label: str, box: tuple[int, int, int, int],
              mask: np.ndarray, confidence: float, centroid: list[float]) -> ObservedMask:
    centre = np.asarray(centroid, dtype=float)
    return ObservedMask(
        detection_id=det_id, camera_id="camera", binary_mask=mask,
        bounding_box_xyxy=box, confidence=confidence,
        canonical_label=label, raw_label=label, predicted_label=label,
        refined_mask_area=int(mask.sum()), depth_point_count=int(mask.sum()),
        centroid_world_m=centre,
        cloud_bounds_world_m={
            "minimum_world_m": (centre - 0.01).tolist(),
            "maximum_world_m": (centre + 0.01).tolist(),
        },
    )


def test_pretracking_duplicate_suppression_same_object():
    backend = YOLOWorldProposalBackend(weights_path=Path("/missing/model.pt"))
    first = np.zeros((80, 80), bool); first[10:50, 10:50] = True
    second = np.zeros((80, 80), bool); second[11:51, 11:51] = True
    result = backend.suppress_duplicate_proposals([
        _proposal("a", "screwdriver", (10, 10, 50, 50), first, 0.8, [0, 0, 0.5]),
        _proposal("b", "screwdriver", (11, 11, 51, 51), second, 0.6, [0.002, 0, 0.5]),
    ])
    assert [item.detection_id for item in result] == ["a"]


def test_pretracking_dedup_preserves_distinct_same_category_objects():
    backend = YOLOWorldProposalBackend(weights_path=Path("/missing/model.pt"))
    first = np.zeros((80, 80), bool); first[5:20, 5:20] = True
    second = np.zeros((80, 80), bool); second[50:65, 50:65] = True
    result = backend.suppress_duplicate_proposals([
        _proposal("a", "screw", (5, 5, 20, 20), first, 0.8, [0, 0, 0.5]),
        _proposal("b", "screw", (50, 50, 65, 65), second, 0.7, [0.03, 0, 0.5]),
    ])
    assert len(result) == 2


def test_pretracking_dedup_keeps_one_cross_category_hypothesis():
    backend = YOLOWorldProposalBackend(weights_path=Path("/missing/model.pt"))
    mask = np.zeros((80, 80), bool); mask[10:50, 10:50] = True
    result = backend.suppress_duplicate_proposals([
        _proposal("a", "screw", (10, 10, 50, 50), mask, 0.7, [0, 0, 0.5]),
        _proposal("b", "bolt", (10, 10, 50, 50), mask.copy(), 0.8, [0, 0, 0.5]),
    ])
    assert len(result) == 1 and result[0].canonical_label == "bolt"
    assert result[0].semantic_alternatives[0]["canonical_label"] == "screw"


class _ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _ArrayTensor([[1, 2, 9, 10]])
        self.conf = _ArrayTensor([0.8])
        self.cls = _ArrayTensor([0])
    def __len__(self):
        return 1


class _FakeResult:
    boxes = _FakeBoxes()
    names = {0: "screw"}


class _FakeYOLO:
    def set_classes(self, prompts):
        self.prompts = prompts
    def predict(self, **kwargs):
        return [_FakeResult()]


def _simple_observation(depth_value: float = 1.0) -> ViewObservation:
    return ViewObservation(
        camera_id="camera", rgb=np.zeros((80, 100, 3), dtype=np.uint8),
        depth_m=np.full((80, 100), depth_value, dtype=float),
        intrinsics=np.array([[50.0, 0, 50.0], [0, 50.0, 40.0], [0, 0, 1.0]]),
        camera_position_world=np.zeros(3), camera_rotation_world=np.eye(3),
    )


def test_multiscale_boxes_map_to_global_coordinates_and_dedup():
    backend = YOLOWorldProposalBackend(weights_path=Path("/missing/model.pt"), min_points_per_mask=4)
    backend._model = _FakeYOLO()
    backend.set_vocabulary(["screw"], {"screw": "screw"})
    backend._inference_windows = lambda *_: [
        ("full_frame", (0, 0, 100, 80)), ("stage_crop", (20, 10, 80, 70)),
    ]
    proposals = backend.predict(
        _simple_observation(), np.array([-10, -10, -10]), np.array([10, 10, 10]))
    assert len(proposals) == 2
    assert {proposal.raw_yolo_bbox_xyxy for proposal in proposals} == {
        (1, 2, 9, 10), (21, 12, 29, 20),
    }
    assert {proposal.inference_source for proposal in proposals} == {"full_frame", "stage_crop"}


def test_same_object_across_full_and_crop_is_one_physical_proposal():
    backend = YOLOWorldProposalBackend(weights_path=Path("/missing/model.pt"))
    mask = np.zeros((80, 80), bool); mask[10:30, 10:30] = True
    full = _proposal("full", "screwdriver", (10, 10, 30, 30), mask, .8, [0, 0, .5])
    crop = _proposal("crop", "screwdriver", (11, 10, 31, 30), mask.copy(), .7, [.002, 0, .5])
    full.inference_source, crop.inference_source = "full_frame", "stage_crop"
    assert len(backend.suppress_duplicate_proposals([full, crop])) == 1


def test_refined_mask_retains_only_accepted_support_and_tracker_reuses_it():
    backend = YOLOWorldProposalBackend(weights_path=Path("/missing/model.pt"), min_points_per_mask=4)
    backend._model = _FakeYOLO()
    backend.set_vocabulary(["screw"], {"screw": "screw"})
    backend._inference_windows = lambda *_: [("full_frame", (0, 0, 100, 80))]
    observation = _simple_observation()
    proposals = backend.predict(observation, np.array([-10, -10, -10]), np.array([10, 10, 10]))
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.gated_points_world_m is not None
    assert int(proposal.binary_mask.sum()) == len(proposal.gated_points_world_m)
    assert proposal.refined_bbox_xyxy == proposal.bounding_box_xyxy
    # If tracking recomputed from depth this would fail after depth is removed.
    observation.depth_m[:] = np.nan
    observation.detected_masks = proposals
    tracker = PersistentInstanceTracker(
        min_cluster_points=4, min_points_per_mask=4, voxel_size_m=.03)
    updated = tracker.update_with_stage_observations(
        0, "TEST", [observation], np.array([-10, -10, -10]), np.array([10, 10, 10]))
    assert updated


def test_region_association_reuses_gated_points_and_weights_quality():
    class Scene:
        @staticmethod
        def get_candidate_regions():
            return [{"region_instance_id": "neutral", "proposal_bounds_m": {
                "minimum_world_m": [-.1, -.1, -.1], "maximum_world_m": [.1, .1, .1]}}]
    observation = _simple_observation()
    observation.depth_m[:] = np.nan
    mask = _proposal("tray", "parts_tray", (10, 10, 30, 30), np.zeros((80, 100), bool), .4, [0, 0, 0])
    mask.gated_points_world_m = np.zeros((20, 3))
    mask.physical_support_quality = .9
    observation.region_semantic_detections = [mask]
    grounder = RegionGrounder({"parts_tray", "shelf"})
    regions = grounder.discover_candidate_regions(Scene(), [observation], stage_index=1)
    assert regions[0].semantic_observations[0]["canonical_label"] == "parts_tray"
    assert regions[0].semantic_observations[0]["association_quality"] > 0

    belief = grounder._compute_consensus_region_semantic([
        {"stage_index": 0, "camera_id": "front", "canonical_label": "shelf",
         "raw_label": "shelf", "confidence": .8, "semantic_support": .04},
        {"stage_index": 0, "camera_id": "front", "canonical_label": "parts_tray",
         "raw_label": "parts tray", "confidence": .3, "semantic_support": .27},
    ])
    assert belief["canonical_label"] == "parts_tray"
    assert belief["total_observations"] == 1


def test_generic_no_valid_driver_diagnosis_has_no_decoy_category_dependency():
    search = FunctionalSatisfactionSearch()
    track = ObservedObjectTrack(
        instance_id="object_1", current_semantic_belief={"canonical_label": "novel_decoy"})
    result = FunctionGroundingResult(
        entity_id="object_1", requirement_id="future_driver_role",
        function_name="CAN_DRIVE_SCREW", semantic_status=GroundingStatus.FAIL,
        semantic_score=.1, semantic_evidence={"evaluated_label": "novel_decoy"},
        geometric_status=GroundingStatus.PASS, geometric_score=1.0,
        geometric_evidence={}, combined_status=GroundingStatus.FAIL)
    diagnosis = search.diagnose_infeasibility(
        [track], [], [], [], [], [], [], grounding_results=[result])
    assert diagnosis == "NO_COMPATIBLE_DRIVER"
    source = Path("mujoco_scenes/workshop_phase1/functional_search.py").read_text()
    assert '"wrench" in observed_categories' not in source
    assert '"pliers" in observed_categories' not in source


def test_no_persistence_clears_region_geometry_history():
    grounder = RegionGrounder()
    grounder._known_regions["proposal"] = ObservedRegion(
        region_instance_id="region_1", proposal_bounds_m={
            "minimum_world_m": [0, 0, 0], "maximum_world_m": [1, 1, 1]},
        observation_source="calibrated_spatial_proposal",
        fused_points=np.ones((20, 3)), semantic_observations=[{"canonical_label": "shelf"}],
        current_geometric_properties={"support_area_m2": 1.0})
    grounder.reset_persistent_evidence()
    retained = grounder._known_regions["proposal"]
    assert len(retained.fused_points) == 0
    assert retained.semantic_observations == []
    assert retained.current_geometric_properties == {}


def test_target_compatibility_does_not_depend_on_head_interface():
    grounder = GeometricGrounder(target_evidence=TargetGeometryEvidence(
        estimated_recess_depth_m=.03, estimated_opening_diameter_m=.01,
        validity=GroundingStatus.PASS))
    relation = grounder.evaluate_compatible_with_target({
        "total_length_m": .04, "shaft_diameter_m": .005, "head_interface": "HEX_LIKE"})
    assert relation["status"] == "TRUE"
    assert "interface_engagement_ok" not in relation


def test_required_relations_drive_unary_verification():
    contract = ManualWorkshopFMContract()
    requirement = next(r for r in contract.get_requirements() if r.function_name == "CAN_DRIVE_SCREW")
    no_relations = replace(requirement, required_relations=[])
    points = np.c_[np.linspace(0, 0.03, 40), np.zeros(40), np.zeros(40)]
    track = ObservedObjectTrack(instance_id="object_1", fused_points=points)
    grounder = GeometricGrounder(target_evidence=TargetGeometryEvidence(
        estimated_recess_depth_m=0.04, estimated_opening_diameter_m=0.01,
        validity=GroundingStatus.PASS))
    required_result = grounder.ground_object_geometry(track, requirement)
    calls_after_required = grounder.relation_call_counts["REACHES_TARGET"]
    optional_result = grounder.ground_object_geometry(track, no_relations)
    assert required_result.geometric_status == GroundingStatus.FAIL
    assert optional_result.geometric_status == GroundingStatus.PASS
    assert calls_after_required == 1
    assert grounder.relation_call_counts["REACHES_TARGET"] == 1


@pytest.mark.parametrize("variant", [
    "F0_MANUAL_FIRST_ONE_REGION",
    "F1_POWER_FIRST_ONE_REGION",
    "F2_MANUAL_FIRST_TWO_REGIONS",
    "F3_POWER_FIRST_TWO_REGIONS",
    "F4_MANUAL_FIRST_THREE_REGIONS",
    "F5_POWER_FIRST_THREE_REGIONS",
    "F6_MANUAL_ONLY",
    "F7_POWER_ONLY",
    "I0_NO_DRIVER",
    "I1_NO_SCREW",
])
def test_all_redesigned_variants_oracle_semantics_upper_bound(variant: str):
    """Verify every fixed-object scene against the compiled-geometry oracle."""
    scene = WorkshopScene("none", variant=variant)
    result = privileged_validate_variant_feasibility(scene)

    expected = {
        "I0_NO_DRIVER": "NO_COMPATIBLE_DRIVER",
        "I1_NO_SCREW": "NO_COMPATIBLE_SCREW",
    }
    if variant.startswith("F"):
        assert result["status"] == "FEASIBLE"
        assert result["selected_witness"] is not None
    else:
        assert result["status"] == "INFEASIBLE"
        assert result["rejection_reason"] == expected[variant]


def test_geometry_config_and_sources_are_category_free():
    config_text = Path("mujoco_scenes/configs/workshop_geometry_inference.yaml").read_text().lower()
    for token in ("workshop_", "stubby", "flathead", "power_driver", "variant_id", "accepted_categories"):
        assert token not in config_text
    source = Path("mujoco_scenes/workshop_phase1/geometric_grounding.py").read_text().lower()
    for token in ('"stubby"', '"flathead"', '"power_driver"', "if category", "if label"):
        assert token not in source


def test_target_unknown_propagates_to_target_relations():
    grounder = GeometricGrounder()
    props = {"usable_length_m": 0.2, "total_length_m": 0.04,
             "shaft_diameter_m": 0.005, "head_interface": "CROSS_LIKE"}
    assert grounder.evaluate_reaches_target(props)["status"] == "UNKNOWN"
    assert grounder.evaluate_compatible_with_target(props)["status"] == "UNKNOWN"


def test_support_footprint_is_measured_not_roi_sized():
    grounder = RegionGrounder({"workbench"})
    x, y = np.meshgrid(np.linspace(-0.10, 0.10, 40), np.linspace(-0.03, 0.03, 20))
    cloud = np.c_[x.ravel(), y.ravel(), np.full(x.size, 0.7)]
    measured = grounder._measure_support_plane(cloud)
    assert measured["predicate"]["status"] == "TRUE"
    assert 0.17 < measured["support_length_m"] < 0.22
    assert 0.04 < measured["support_width_m"] < 0.07
    # A hypothetical 1 m x 1 m selection ROI cannot inflate the measurement.
    assert measured["support_area_m2"] < 0.02


def test_oracle_semantics_use_broad_contract_taxonomy():
    scene = WorkshopScene("none", variant="F5_POWER_FIRST_THREE_REGIONS")
    controller = WorkshopPhase1InspectionController(
        scene=scene, mask_backend=MaskBackendType.ORACLE,
        semantic_backend=SemanticBackendType.ORACLE)
    controller.run_episode()
    labels = [track.current_semantic_belief.get("canonical_label")
              for track in controller.tracker.tracks.values()]
    assert labels.count("screwdriver") == 1
    assert labels.count("power_driver") == 1
    assert "flathead_driver" not in labels
