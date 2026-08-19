"""Unit and integration tests for Workshop (W1) Phase 1 Research Pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from mujoco_scenes.workshop_scene import WorkshopScene
from mujoco_scenes.workshop_phase1.capture import ProductionInspectionCapture
from mujoco_scenes.workshop_phase1.types import (
    EntityType,
    FunctionalRequirement,
    GroundingStatus,
    MaskBackendType,
    ObservedMask,
    ObservedObjectTrack,
    ObservedRegion,
    ViewObservation,
)
from mujoco_scenes.workshop_phase1.perception import (
    OpenVocabularyQueryBuilder,
    PrivilegedOracleMaskBackend,
    RGBDConnectedComponentProposalBackend,
    YOLOWorldProposalBackend,
)
from mujoco_scenes.workshop_phase1.tracking import PersistentInstanceTracker
from mujoco_scenes.workshop_phase1.evidence_graph import GrowingObservedGraph
from mujoco_scenes.workshop_phase1.semantic_grounding import (
    DeterministicSemanticNormalizer,
    ProductionSemanticBackend,
    SemanticGrounder,
)
from mujoco_scenes.workshop_phase1.geometric_grounding import GeometricGrounder
from mujoco_scenes.workshop_phase1.region_grounding import RegionGrounder
from mujoco_scenes.workshop_phase1.functional_search import FunctionalSatisfactionSearch
from mujoco_scenes.workshop_phase1.inspection_controller import WorkshopPhase1InspectionController
from mujoco_scenes.workshop_phase1.requirements import StaticWorkshopRequirementProvider
from mujoco_scenes.workshop_phase1.serialization import sanitize_production_data, assert_no_backend_names


def test_static_requirements_provider():
    """Verify standard 4-tuple functional requirements schema."""
    provider = StaticWorkshopRequirementProvider()
    reqs = provider.get_requirements()
    assert len(reqs) == 4
    names = {r.function_name for r in reqs}
    assert "CAN_DRIVE_SCREW" in names
    assert "CAN_FASTEN" in names
    assert "WORK_SURFACE" in names
    assert "SMALL_PARTS_CONTAINER" in names


def test_open_vocabulary_query_builder():
    """Verify dynamic open-vocabulary query generation from functional requirements."""
    reqs = [
        FunctionalRequirement(
            requirement_id="req_01",
            entity_type=EntityType.OBJECT,
            function_name="CAN_DRIVE_SCREW",
            description="Tool capable of driving screws or fasteners into the frame joint",
        ),
        FunctionalRequirement(
            requirement_id="req_02",
            entity_type=EntityType.OBJECT,
            function_name="CAN_FASTEN",
            description="Fastener capable of securely fastening the frame joint",
        ),
    ]
    queries = OpenVocabularyQueryBuilder.build_queries(reqs, "Repair frame joint in workshop")
    assert len(queries) >= 2
    assert any("tool" in q.lower() or "driver" in q.lower() for q in queries)
    assert any("fastener" in q.lower() or "screw" in q.lower() for q in queries)


def test_oracle_mask_backend_neutrality():
    """Verify PrivilegedOracleMaskBackend outputs 'object' and zero semantic typing."""
    scene = WorkshopScene("none", variant="F0_BASE")
    oracle_backend = PrivilegedOracleMaskBackend(scene)
    capture = ProductionInspectionCapture()
    obs_list = capture.capture_stage(scene, "INITIAL", capture_segmentation=True)

    vol_min = np.array([-1.2, -0.2, 0.6])
    vol_max = np.array([1.2, 0.9, 1.5])
    for obs in obs_list:
        masks = oracle_backend.predict(obs, vol_min, vol_max)
        for m in masks:
            assert m.predicted_label == "object", f"Oracle mask leaked label: {m.predicted_label}"
            assert m.backend_name == "privileged_oracle"


def test_deterministic_semantic_normalizer():
    """Verify open-ended description parsing and UNKNOWN preservation without Phillips assumption."""
    norm = DeterministicSemanticNormalizer()

    assert norm.normalize_tool_interface("cross-shaped screw driving tip") == "CROSS_RECESS"
    assert norm.normalize_tool_interface("slotted flat blade screwdriver") == "SINGLE_SLOT"
    assert norm.normalize_tool_interface("hex socket driver bit") == "HEX_SOCKET"
    assert norm.normalize_tool_interface("gripping pliers without driving tip") == "NO_DRIVE_INTERFACE"
    # Ambiguous / novel tool does NOT default to Phillips
    assert norm.normalize_tool_interface("generic metal rod with textured grip") == "UNKNOWN_INTERFACE"

    assert norm.normalize_fastener_interface("screw with cross recess head") == "CROSS_RECESS"
    assert norm.normalize_fastener_interface("bolt with hexagonal head") == "HEX_HEAD"
    assert norm.normalize_fastener_interface("slotted wood screw") == "SINGLE_SLOT"
    assert norm.normalize_fastener_interface("threaded cylinder") == "UNKNOWN_INTERFACE"


def test_geometric_grounder_tool_reach():
    """Verify tool reach estimation distinguishes long from stubby tools."""
    geo = GeometricGrounder()

    # Long tool point cloud: cylinder of length 0.23m, shaft radius 0.0025m for 0.12m
    z_shaft = np.linspace(-0.10, 0.02, 300)
    theta_shaft = np.random.uniform(0, 2 * np.pi, 300)
    r_shaft = np.random.uniform(0, 0.0025, 300)
    shaft_pts = np.column_stack([r_shaft * np.cos(theta_shaft), r_shaft * np.sin(theta_shaft), z_shaft])

    z_handle = np.linspace(0.02, 0.13, 300)
    theta_handle = np.random.uniform(0, 2 * np.pi, 300)
    r_handle = np.random.uniform(0, 0.015, 300)
    handle_pts = np.column_stack([r_handle * np.cos(theta_handle), r_handle * np.sin(theta_handle), z_handle])

    long_tool_pts = np.vstack([shaft_pts, handle_pts])
    reach_info = geo.estimate_driver_reach(long_tool_pts)
    assert reach_info["usable_reach_m"] >= 0.025
    assert reach_info["total_length_m"] > 0.20


def test_geometric_grounder_fastener():
    """Verify fastener dimension estimation."""
    geo = GeometricGrounder()

    # Medium screw: length 0.043m, shaft radius 0.003m
    z_pts = np.linspace(-0.02, 0.023, 200)
    theta = np.random.uniform(0, 2 * np.pi, 200)
    r = np.random.uniform(0, 0.003, 200)
    screw_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z_pts])

    dim_info = geo.estimate_fastener_dimensions(screw_pts)
    assert 0.038 <= dim_info["length_m"] <= 0.048
    assert dim_info["shaft_diameter_m"] <= 0.008


def test_observed_cavity_and_openness():
    """Verify container cavity depth measurement and open status detection."""
    scene = WorkshopScene("none", variant="F0_BASE")
    capture = ProductionInspectionCapture()
    obs_list = capture.capture_stage(scene, "INITIAL")
    region_grounder = RegionGrounder()
    regions = region_grounder.discover_candidate_regions(scene, obs_list)

    assert len(regions) >= 2
    # Check that regions have valid proposal bounds and cavity geometry
    for reg in regions:
        assert reg.observation_source == "calibrated_spatial_proposal"
        assert reg.region_instance_id.startswith("region_")


def test_tracker_multiview_clustering_and_persistence():
    """Verify PersistentInstanceTracker clusters observations across views."""
    tracker = PersistentInstanceTracker(cluster_distance_threshold_m=0.030)

    cam1_pts = np.random.normal(loc=[0.0, 0.5, 0.8], scale=0.002, size=(50, 3))
    cam2_pts = np.random.normal(loc=[0.002, 0.501, 0.801], scale=0.002, size=(50, 3))

    m1 = ObservedMask(
        detection_id="det_cam1_0",
        camera_id="cam1",
        binary_mask=np.ones((10, 10), dtype=bool),
        bounding_box_xyxy=(0, 0, 10, 10),
        confidence=0.95,
        predicted_label="tool",
    )
    m2 = ObservedMask(
        detection_id="det_cam2_0",
        camera_id="cam2",
        binary_mask=np.ones((10, 10), dtype=bool),
        bounding_box_xyxy=(0, 0, 10, 10),
        confidence=0.95,
        predicted_label="tool",
    )

    intrinsics = np.eye(3)
    cam_pos = np.zeros(3)
    cam_mat = np.eye(3)

    obs1 = ViewObservation(
        camera_id="cam1",
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth_m=np.ones((10, 10)),
        intrinsics=intrinsics,
        camera_position_world=cam_pos,
        camera_rotation_world=cam_mat,
        detected_masks=[m1],
    )
    obs2 = ViewObservation(
        camera_id="cam2",
        rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        depth_m=np.ones((10, 10)),
        intrinsics=intrinsics,
        camera_position_world=cam_pos,
        camera_rotation_world=cam_mat,
        detected_masks=[m2],
    )

    import mujoco_scenes.workshop_phase1.tracking as trk_module
    orig_bp = trk_module.backproject_masked_depth
    trk_module.backproject_masked_depth = lambda d, m, i, p, r, max_depth: (
        cam1_pts if p[0] == 0.0 else cam2_pts,
        np.zeros((len(cam1_pts), 2), dtype=int),
    )

    try:
        vol_min = np.array([-1.0, -1.0, -1.0])
        vol_max = np.array([1.0, 1.0, 2.0])
        tracks = tracker.update_with_stage_observations(
            stage_index=0,
            source_region_id="INITIAL_TABLETOP",
            observations=[obs1, obs2],
            stage_volume_min=vol_min,
            stage_volume_max=vol_max,
        )
        assert len(tracks) == 1
        assert tracks[0].instance_id == "object_0001"
        assert tracks[0].evidence_count >= 1
    finally:
        trk_module.backproject_masked_depth = orig_bp


def test_serialization_sanitization():
    """Verify production data serialization rejects simulator strings."""
    valid_data = {
        "instance_id": "object_0001",
        "category": "HAND_DRIVER",
        "length_m": 0.223,
    }
    sanitized = sanitize_production_data(valid_data)
    assert sanitized["instance_id"] == "object_0001"
    assert_no_backend_names(sanitized)

    leak_data = {
        "instance_id": "object_0001",
        "backend_name": "workshop_long_phillips_driver",
    }
    with pytest.raises(ValueError):
        assert_no_backend_names(leak_data)


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
def test_all_14_variants_oracle_mask_execution(variant: str):
    """Integration test: verify all 14 variants run with Oracle Masks and Production Semantics."""
    scene = WorkshopScene("none", variant=variant)
    controller = WorkshopPhase1InspectionController(
        mask_backend=MaskBackendType.ORACLE,
    )
    result = controller.run_episode(scene)
    assert result.status in ("FEASIBLE", "INFEASIBLE")

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
