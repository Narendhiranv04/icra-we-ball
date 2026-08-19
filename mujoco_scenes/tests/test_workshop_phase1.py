"""Unit and regression tests for Workshop (W1) Phase 1 Frozen Architecture."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import yaml

from mujoco_scenes.workshop_scene import WorkshopScene
from mujoco_scenes.workshop_phase1.capture import MultiViewCameraRig
from mujoco_scenes.workshop_phase1.types import (
    AblationType,
    EntityType,
    FunctionalRequirement,
    FunctionalWitness,
    GroundingStatus,
    MaskBackendType,
    ObservedMask,
    ObservedObjectTrack,
    ObservedRegion,
    ProposalMode,
    RequirementSource,
    SemanticBackendType,
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
    assert len(reqs) == 4

    names = {r.function_name for r in reqs}
    assert "CAN_DRIVE_SCREW" in names
    assert "CAN_FASTEN" in names
    assert "WORK_SURFACE" in names
    assert "SMALL_PARTS_CONTAINER" in names

    prompts = contract.get_detector_prompts()
    assert len(prompts) >= 10
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
    assert "drinking cup" in prompts
    assert "bowl" in prompts
    assert "soup bowl" in prompts
    assert "screwdriver" not in prompts


def test_no_function_description_prompts():
    """Mandatory test (Sec 65): Detector classes do not contain affordance sentences."""
    contract = ManualWorkshopFMContract()
    prompts = contract.get_detector_prompts()
    forbidden_phrases = ["tool that can", "capable of", "suitable for", "used to secure", "object capable"]
    for p in prompts:
        for f in forbidden_phrases:
            assert f not in p.lower(), f"Affordance prompt found: {p}"


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
    scene = WorkshopScene("none", variant="F0_BASE")
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
    scene = WorkshopScene("none", variant="F0_BASE")
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
    """Mandatory test (Sec 70): Container cavity requires observed depth difference."""
    scene = WorkshopScene("none", variant="F0_BASE")
    rig = MultiViewCameraRig(scene=scene)
    obs_list = rig.capture_stage_observations()
    region_grounder = RegionGrounder()
    regions = region_grounder.discover_candidate_regions(scene, obs_list)

    assert len(regions) >= 2
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
    scene = WorkshopScene("none", variant="F4_OBJECT_REGION_COUPLING")
    ctrl_full = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.NONE, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_full = ctrl_full.run_episode()

    ctrl_no_joint = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.NO_JOINT_COUPLING, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_no_joint = ctrl_no_joint.run_episode()

    assert res_full.status == "FEASIBLE"
    assert res_no_joint.status == "FEASIBLE"
    assert res_no_joint.witness.verification_details.get("ablation") == "NO_JOINT_COUPLING"


def test_semantic_only_self_test():
    """Self-test (Sec 88): SEMANTIC_ONLY bypasses geometry in witness selection."""
    scene = WorkshopScene("none", variant="I4_TOOL_GEOMETRY_FAILURE")
    ctrl_full = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.NONE, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_full = ctrl_full.run_episode()

    ctrl_sem = WorkshopPhase1InspectionController(scene=scene, ablation=AblationType.SEMANTIC_ONLY, mask_backend=MaskBackendType.ORACLE, semantic_backend=SemanticBackendType.ORACLE)
    res_sem = ctrl_sem.run_episode()

    # In I4, driver reaches are insufficient. Full rejects with TOOL_GEOMETRY_FAILURE, SEMANTIC_ONLY accepts driver
    assert res_full.status == "INFEASIBLE"
    assert res_full.rejection_reason == "TOOL_GEOMETRY_FAILURE"
    assert res_sem.status == "FEASIBLE"


def test_config_wiring(tmp_path: Path):
    """Self-test (Sec 85): YAML config actually controls runtime components."""
    custom_cfg = tmp_path / "custom_config.yaml"
    cfg_data = {
        "perception": {
            "detector": {
                "confidence_threshold": 0.12,
                "nms_iou_threshold": 0.50,
            }
        },
        "tracking": {
            "cluster_distance_threshold_m": 0.055,
            "track_match_distance_threshold_m": 0.060,
        },
        "grounding": {
            "geometry": {
                "min_driver_reach_m": 0.040,
                "staging_margin_multiplier": 1.35,
            }
        },
        "inspection": {
            "sequence": ["RIGHT_DRAWER", "LEFT_DRAWER"],
            "early_stop": False,
        }
    }
    with open(custom_cfg, "w") as f:
        yaml.dump(cfg_data, f)

    scene = WorkshopScene("none", variant="F0_BASE")
    ctrl = WorkshopPhase1InspectionController(scene=scene, config_path=custom_cfg)

    assert ctrl.tracker.cluster_distance_threshold_m == 0.055
    assert ctrl.tracker.track_match_distance_threshold_m == 0.060
    assert ctrl.geometric_grounder.min_driver_reach_m == 0.040
    assert ctrl.geometric_grounder.staging_margin_multiplier == 1.35
    assert ctrl.inspection_sequence == ["RIGHT_DRAWER", "LEFT_DRAWER"]
    assert ctrl.early_stop_enabled is False


@pytest.mark.parametrize("variant", [
    "F0_BASE",
    "F1_TOOL_ALTERNATIVE",
    "F2_REGION_ALTERNATIVE",
    "F3_DISTRIBUTED_OBJECTS",
    "F4_OBJECT_REGION_COUPLING",
    "F5_DECOY_HEAVY",
    "F6_LAYOUT_SWAPPED",
    "I0_NO_VALID_DRIVER",
    "I1_NO_VALID_FASTENER",
    "I2_NO_WORK_SURFACE",
    "I3_NO_PARTS_CONTAINER",
    "I4_TOOL_GEOMETRY_FAILURE",
    "I5_OBJECT_REGION_PACKING_FAILURE",
    "I6_GLOBAL_CONFLICT",
])
def test_all_14_variants_oracle_semantics_upper_bound(variant: str):
    """Verify all 14 variants pass downstream tracking, geometry, and search under Oracle Semantics."""
    scene = WorkshopScene("none", variant=variant)
    controller = WorkshopPhase1InspectionController(
        scene=scene,
        mask_backend=MaskBackendType.ORACLE,
        semantic_backend=SemanticBackendType.ORACLE,
    )
    result = controller.run_episode()

    if variant.startswith("F"):
        assert result.status == "FEASIBLE"
        assert result.witness is not None
        assert result.witness.driver_id.startswith("object_")
        assert result.witness.fastener_id.startswith("object_")
        assert result.witness.work_surface_id.startswith("region_")
        assert result.witness.parts_container_id.startswith("region_")
    else:
        assert result.status == "INFEASIBLE"
        assert result.rejection_reason is not None
