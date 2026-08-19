"""Data contracts and typed representations for Workshop Phase 1 Grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    OBJECT = "OBJECT"
    REGION = "REGION"
    WORKPIECE = "WORKPIECE"


class GroundingStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class RequirementSource(str, Enum):
    STATIC = "STATIC"
    FM = "FM"


class MaskBackendType(str, Enum):
    PRODUCTION = "PRODUCTION"
    ORACLE = "ORACLE"
    CONNECTED_COMPONENT = "CONNECTED_COMPONENT"


class SemanticBackendType(str, Enum):
    PRODUCTION = "PRODUCTION"
    ORACLE = "ORACLE"
    DETERMINISTIC_TEST = "DETERMINISTIC_TEST"


class InspectionPolicyType(str, Enum):
    FIXED = "FIXED"
    FM = "FM"


class AblationType(str, Enum):
    NONE = "NONE"
    SEMANTIC_ONLY = "SEMANTIC_ONLY"
    NO_GEOMETRY = "NO_GEOMETRY"
    NO_JOINT_COUPLING = "NO_JOINT_COUPLING"
    NO_PERSISTENCE = "NO_PERSISTENCE"
    SINGLE_VIEW = "SINGLE_VIEW"
    ORACLE_MASK = "ORACLE_MASK"
    ORACLE_SEMANTICS = "ORACLE_SEMANTICS"


@dataclass
class SemanticEvidence:
    """Open-ended structured semantic evidence extracted from visual crops."""

    free_text_description: str
    functional_description: str = ""
    visible_tool_interface: str = "UNKNOWN"
    visible_fastener_interface: str = "UNKNOWN"
    broad_object_type: str = "object"
    confidence: float = 1.0
    evidence_crops_provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_text_description": self.free_text_description,
            "functional_description": self.functional_description,
            "visible_tool_interface": self.visible_tool_interface,
            "visible_fastener_interface": self.visible_fastener_interface,
            "broad_object_type": self.broad_object_type,
            "confidence": round(self.confidence, 4),
            "evidence_crops_provenance": list(self.evidence_crops_provenance),
        }


@dataclass
class TargetGeometryEvidence:
    """Geometry of the target workpiece recess observed from tabletop RGB-D."""

    target_position: Any = None  # np.ndarray (3,)
    estimated_opening_diameter_m: float | None = None
    estimated_recess_depth_m: float | None = None
    point_count: int = 0
    source_views: list[str] = field(default_factory=list)
    confidence: float = 1.0
    validity: GroundingStatus = GroundingStatus.UNKNOWN
    quality_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "estimated_opening_diameter_m": self.estimated_opening_diameter_m,
            "estimated_recess_depth_m": self.estimated_recess_depth_m,
            "point_count": self.point_count,
            "source_views": list(self.source_views),
            "confidence": round(self.confidence, 4),
            "validity": self.validity.value,
            "quality_metadata": self.quality_metadata,
        }


@dataclass(frozen=True)
class FunctionalRequirement:
    """A broad functional capability required by the repair task."""

    requirement_id: str
    entity_type: EntityType
    function_name: str
    description: str
    rank: int = 1
    source: RequirementSource = RequirementSource.STATIC
    semantic_hints: list[str] = field(default_factory=list)
    geometric_constraints: dict[str, Any] = field(default_factory=dict)
    provenance: str = "default_provider"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "entity_type": self.entity_type.value,
            "function_name": self.function_name,
            "description": self.description,
            "rank": self.rank,
            "source": self.source.value,
            "semantic_hints": list(self.semantic_hints),
            "geometric_constraints": dict(self.geometric_constraints),
            "provenance": self.provenance,
        }


@dataclass
class ObservedMask:
    """A 2D instance mask detected in one camera view."""

    detection_id: str
    camera_id: str
    binary_mask: Any  # np.ndarray of shape (H, W) bool
    bounding_box_xyxy: tuple[int, int, int, int]
    confidence: float
    predicted_label: str
    backend_name: str = "yolo_world"
    features: Any | None = None  # optional visual embedding


@dataclass
class ViewObservation:
    """RGB-D and camera calibration data from one viewpoint."""

    camera_id: str
    rgb: Any  # np.ndarray of shape (H, W, 3) uint8
    depth_m: Any  # np.ndarray of shape (H, W) float64
    intrinsics: Any  # np.ndarray of shape (3, 3) float64
    camera_position_world: Any  # np.ndarray (3,)
    camera_rotation_world: Any  # np.ndarray (3, 3)
    validation: dict[str, Any] = field(default_factory=dict)
    detected_masks: list[ObservedMask] = field(default_factory=list)
    segmentation: Any | None = None  # np.ndarray (H, W, 2) int32, strictly oracle-only


@dataclass
class ObservedObjectTrack:
    """A persistent generic object instance aggregated across views and stages."""

    instance_id: str
    first_seen_stage: int = 0
    last_seen_stage: int = 0
    source_inspection_region_id: str = "INITIAL"
    fused_points: Any = None  # np.ndarray of shape (N, 3)
    fused_colors: Any = None  # np.ndarray of shape (N, 3)
    crop_evidence: dict[str, Any] = field(default_factory=dict)  # camera_id -> rgb crop (H, W, 3)
    points_by_camera: dict[str, Any] = field(default_factory=dict)
    contributing_cameras: tuple[str, ...] = field(default_factory=tuple)
    semantic_evidence_history: list[dict[str, Any]] = field(default_factory=list)
    geometric_evidence_history: list[dict[str, Any]] = field(default_factory=list)
    current_semantic_belief: dict[str, Any] = field(default_factory=dict)
    current_geometric_properties: dict[str, Any] = field(default_factory=dict)
    overall_confidence: float = 1.0
    evidence_count: int = 1
    status: str = "ACTIVE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "first_seen_stage": self.first_seen_stage,
            "last_seen_stage": self.last_seen_stage,
            "source_inspection_region_id": self.source_inspection_region_id,
            "point_count": len(self.fused_points) if self.fused_points is not None else 0,
            "contributing_cameras": list(self.contributing_cameras),
            "semantic_belief": self.current_semantic_belief,
            "geometric_properties": self.current_geometric_properties,
            "overall_confidence": round(self.overall_confidence, 4),
            "evidence_count": self.evidence_count,
            "status": self.status,
        }


@dataclass
class ObservedRegion:
    """An observable candidate functional region (work surface or parts container)."""

    region_instance_id: str
    proposal_bounds_m: dict[str, list[float]]  # minimum_world_m, maximum_world_m
    observation_source: str
    fused_points: Any | None = None
    fused_colors: Any | None = None
    crop_evidence: dict[str, Any] = field(default_factory=dict)
    support_plane: dict[str, Any] = field(default_factory=dict)
    cavity_geometry: dict[str, Any] = field(default_factory=dict)
    obstruction_evidence: dict[str, Any] = field(default_factory=dict)
    is_open: bool | None = None
    is_open_status: GroundingStatus = GroundingStatus.UNKNOWN
    current_semantic_belief: dict[str, Any] = field(default_factory=dict)
    current_geometric_properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_instance_id": self.region_instance_id,
            "proposal_bounds_m": self.proposal_bounds_m,
            "observation_source": self.observation_source,
            "point_count": len(self.fused_points) if self.fused_points is not None else 0,
            "support_plane": self.support_plane,
            "cavity_geometry": self.cavity_geometry,
            "obstruction_evidence": self.obstruction_evidence,
            "is_open": self.is_open,
            "is_open_status": self.is_open_status.value,
            "semantic_belief": self.current_semantic_belief,
            "geometric_properties": self.current_geometric_properties,
        }


@dataclass
class FunctionGroundingResult:
    """Verification result for one candidate entity against one functional requirement."""

    entity_id: str
    requirement_id: str
    function_name: str
    semantic_status: GroundingStatus
    semantic_score: float
    semantic_evidence: dict[str, Any]
    geometric_status: GroundingStatus
    geometric_score: float
    geometric_evidence: dict[str, Any]
    combined_status: GroundingStatus
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "requirement_id": self.requirement_id,
            "function_name": self.function_name,
            "semantic_status": self.semantic_status.value,
            "semantic_score": round(self.semantic_score, 4),
            "semantic_evidence": self.semantic_evidence,
            "geometric_status": self.geometric_status.value,
            "geometric_score": round(self.geometric_score, 4),
            "geometric_evidence": self.geometric_evidence,
            "combined_status": self.combined_status.value,
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass
class FunctionalCandidate:
    """An entity qualifying as a valid candidate for a functional role."""

    entity_id: str
    function_name: str
    confidence: float
    grounding_result: FunctionGroundingResult


@dataclass
class FunctionalWitness:
    """A jointly feasible 4-tuple fulfilling all functional prerequisites."""

    driver_id: str
    fastener_id: str
    work_surface_id: str
    parts_container_id: str
    overall_confidence: float
    verification_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "FEASIBLE",
            "driver": self.driver_id,
            "fastener": self.fastener_id,
            "work_surface": self.work_surface_id,
            "parts_container": self.parts_container_id,
            "overall_confidence": round(self.overall_confidence, 4),
            "verification": self.verification_details,
        }


@dataclass
class InspectionDecision:
    """Decision made by the inspection controller at one step."""

    stage_index: int
    inspection_region_id: str | None
    action: str  # "INSPECT", "STOP_FEASIBLE", "STOP_EXHAUSTED"
    rationale: str
    unresolved_requirements: list[str] = field(default_factory=list)


@dataclass
class InspectionTrace:
    """Full execution trace of the Phase 1 incremental inspection loop."""

    steps: list[InspectionDecision] = field(default_factory=list)
    inspected_regions: list[str] = field(default_factory=list)
    early_stopped: bool = False
    total_stages: int = 0


@dataclass
class EpisodeResult:
    """Final outcome of one Phase 1 grounding episode."""

    status: str  # "FEASIBLE" or "INFEASIBLE"
    rejection_reason: str | None
    witness: FunctionalWitness | None
    trace: InspectionTrace
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res = {
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "witness": self.witness.to_dict() if self.witness is not None else None,
            "inspection_count": len(self.trace.inspected_regions),
            "inspected_regions": self.trace.inspected_regions,
            "early_stopped": self.trace.early_stopped,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
        }
        return res
