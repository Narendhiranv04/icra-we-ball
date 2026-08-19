"""Semantic functional grounding for objects and regions in Workshop Phase 1.

Implements pure semantic category verification against the FM-contract functional
membership rules. Zero CLIP, zero second semantic model, zero geometric fabrication.
"""

from __future__ import annotations

import abc
from typing import Any
import numpy as np

from mujoco_scenes.workshop_phase1.types import (
    FunctionGroundingResult,
    FunctionalRequirement,
    GroundingStatus,
    ObservedObjectTrack,
    ObservedRegion,
    SemanticEvidence,
)


class ObjectSemanticBackend(abc.ABC):
    """Abstract interface for semantic description of object tracks and regions."""

    @abc.abstractmethod
    def describe_object(
        self,
        track: ObservedObjectTrack,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        """Extract semantic evidence for an object track."""
        pass

    @abc.abstractmethod
    def describe_region(
        self,
        region: ObservedRegion,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        """Extract semantic evidence for a candidate region."""
        pass


class ProductionSemanticBackend(ObjectSemanticBackend):
    """Production semantic backend consuming YOLO-World detection labels.

    Uses the consensus semantic category aggregated across camera views.
    Contains ZERO second vision models and ZERO geometric shape heuristics.
    """

    def describe_object(
        self,
        track: ObservedObjectTrack,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        belief = track.current_semantic_belief
        canonical_label = belief.get("canonical_label", "unknown")
        raw_label = belief.get("raw_label", canonical_label)
        confidence = float(belief.get("confidence", 0.0))
        observations = track.semantic_observations

        return SemanticEvidence(
            canonical_label=canonical_label,
            raw_label=raw_label,
            confidence=confidence,
            source_camera=str(track.contributing_cameras),
            evidence_provenance=observations,
        )

    def describe_region(
        self,
        region: ObservedRegion,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        belief = region.current_semantic_belief
        canonical_label = belief.get("canonical_label", "unknown")
        raw_label = belief.get("raw_label", canonical_label)
        confidence = float(belief.get("confidence", 0.0))
        observations = region.semantic_observations

        return SemanticEvidence(
            canonical_label=canonical_label,
            raw_label=raw_label,
            confidence=confidence,
            source_camera="multi_view",
            evidence_provenance=observations,
        )


class PrivilegedOracleSemanticBackend(ObjectSemanticBackend):
    """Privileged semantic backend using ground truth simulator metadata.

    ALLOWED ONLY under explicit ablation '--semantic-backend oracle'.
    """

    def __init__(self, scene: Any | None = None) -> None:
        self.scene = scene

    def describe_object(
        self,
        track: ObservedObjectTrack,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        # Match track to GT body by 3D centroid distance
        pts = track.fused_points
        if pts is None or len(pts) == 0 or self.scene is None:
            return SemanticEvidence(canonical_label="unknown", confidence=0.0)

        centroid = pts.mean(axis=0)
        best_dist = 1e6
        best_name = "unknown"
        best_canonical = "unknown"

        import mujoco
        for bid in range(self.scene.model.nbody):
            b_name = mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not b_name or self.scene.model.body_dofnum[bid] != 6:
                continue

            b_min = np.array([np.inf, np.inf, np.inf])
            b_max = np.array([-np.inf, -np.inf, -np.inf])
            for gid in range(self.scene.model.ngeom):
                if self.scene.model.geom_bodyid[gid] == bid:
                    gpos = self.scene.data.geom_xpos[gid]
                    gmat = self.scene.data.geom_xmat[gid].reshape(3, 3)
                    gtype = self.scene.model.geom_type[gid]
                    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                        mid = self.scene.model.geom_dataid[gid]
                        v_start = self.scene.model.mesh_vertadr[mid]
                        v_num = self.scene.model.mesh_vertnum[mid]
                        verts = self.scene.model.mesh_vert[v_start:v_start + v_num]
                        wverts = (gmat @ verts.T).T + gpos
                        b_min = np.minimum(b_min, wverts.min(axis=0))
                        b_max = np.maximum(b_max, wverts.max(axis=0))
                    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                        gsz = self.scene.model.geom_size[gid]
                        corners = np.array([[sx * gsz[0], sy * gsz[1], sz * gsz[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                        wcorners = (gmat @ corners.T).T + gpos
                        b_min = np.minimum(b_min, wcorners.min(axis=0))
                        b_max = np.maximum(b_max, wcorners.max(axis=0))

            center = (b_min + b_max) / 2.0 if np.all(np.isfinite(b_min)) else self.scene.data.xpos[bid].copy()
            d = float(np.linalg.norm(centroid - center))
            if d < best_dist and d < 0.25:
                best_dist = d
                best_name = b_name

        if "driver" in best_name or "screwdriver" in best_name:
            if "power" in best_name:
                best_canonical = "power_driver"
            else:
                best_canonical = "screwdriver"
        elif "screw" in best_name:
            best_canonical = "screw"
        elif "bolt" in best_name:
            best_canonical = "bolt"
        elif "wrench" in best_name:
            best_canonical = "wrench"
        elif "pliers" in best_name:
            best_canonical = "pliers"

        return SemanticEvidence(
            canonical_label=best_canonical,
            raw_label=best_name,
            confidence=1.0,
        )

    def describe_region(
        self,
        region: ObservedRegion,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        obs_source = region.observation_source.lower()
        if self.scene is not None and hasattr(self.scene, "get_candidate_regions"):
            r_min = np.array(region.proposal_bounds_m["minimum_world_m"])
            r_max = np.array(region.proposal_bounds_m["maximum_world_m"])
            r_center = (r_min + r_max) / 2.0
            for prop in self.scene.get_candidate_regions():
                p_min = np.array(prop["proposal_bounds_m"]["minimum_world_m"])
                p_max = np.array(prop["proposal_bounds_m"]["maximum_world_m"])
                p_center = (p_min + p_max) / 2.0
                if np.linalg.norm(r_center - p_center) < 0.05:
                    if hasattr(self.scene, "privileged_backend_name_for_region"):
                        bname = self.scene.privileged_backend_name_for_region(prop["region_instance_id"])
                        active = set(getattr(self.scene, "active_surfaces", [])) | set(
                            getattr(self.scene, "active_containers", []))
                        if bname not in active:
                            return SemanticEvidence(canonical_label="unknown", raw_label="inactive_region", confidence=0.0)
                        if bname == "MAIN_WORKBENCH_ZONE":
                            return SemanticEvidence(canonical_label="workbench", raw_label=bname, confidence=1.0)
                        elif bname == "TOOL_CART_TOP":
                            return SemanticEvidence(canonical_label="tool_cart", raw_label=bname, confidence=1.0)
                        elif bname == "NARROW_WALL_SHELF":
                            return SemanticEvidence(canonical_label="shelf", raw_label=bname, confidence=1.0)
                        elif bname == "PARTS_TRAY":
                            return SemanticEvidence(canonical_label="parts_tray", raw_label=bname, confidence=1.0)
                        elif bname == "HARDWARE_BIN":
                            return SemanticEvidence(canonical_label="hardware_bin", raw_label=bname, confidence=1.0)
                    obs_source = prop.get("observation_source", "").lower()
                    break

        if "workbench" in obs_source or "table" in obs_source:
            canonical = "workbench"
        elif "cart" in obs_source:
            canonical = "tool_cart"
        elif "shelf" in obs_source:
            canonical = "shelf"
        elif "tray" in obs_source:
            canonical = "parts_tray"
        elif "bin" in obs_source:
            canonical = "hardware_bin"
        else:
            canonical = "unknown"

        return SemanticEvidence(
            canonical_label=canonical,
            raw_label=obs_source,
            confidence=1.0,
        )


class SemanticGrounder:
    """Evaluates semantic requirement satisfaction against the FM contract."""

    def __init__(self, backend: ObjectSemanticBackend | None = None) -> None:
        self.backend = backend or ProductionSemanticBackend()
        self.total_semantic_calls: int = 0
        self._object_cache: dict[str, dict[str, Any]] = {}
        self._region_cache: dict[str, dict[str, Any]] = {}

    def reset_cache(self) -> None:
        self._object_cache.clear()
        self._region_cache.clear()

    def ground_object_for_requirement(
        self,
        track: ObservedObjectTrack,
        requirement: FunctionalRequirement,
    ) -> FunctionGroundingResult:
        """Evaluate semantic satisfaction of an object track for a functional requirement."""
        self.total_semantic_calls += 1

        evidence_obj = self.backend.describe_object(track, requirement.description)
        evidence_dict = evidence_obj.to_dict()

        canonical_label = evidence_obj.canonical_label.lower().strip()
        accepted_categories = [c.lower().strip() for c in requirement.accepted_categories]

        rejections: list[str] = []

        if not canonical_label or canonical_label in ("unknown", "object", "object_proposal"):
            status = GroundingStatus.UNKNOWN
            score = 0.50
            rejections.append("SEMANTIC_LABEL_UNKNOWN")
        elif canonical_label in accepted_categories:
            status = GroundingStatus.PASS
            score = max(0.60, evidence_obj.confidence)
        else:
            status = GroundingStatus.FAIL
            score = 0.10
            rejections.append(f"SEMANTIC_CATEGORY_MISMATCH_{canonical_label.upper()}_NOT_ACCEPTED")

        evidence_dict["accepted_categories"] = list(requirement.accepted_categories)
        evidence_dict["evaluated_label"] = canonical_label

        return FunctionGroundingResult(
            entity_id=track.instance_id,
            requirement_id=requirement.requirement_id,
            function_name=requirement.function_name,
            semantic_status=status,
            semantic_score=score,
            semantic_evidence=evidence_dict,
            geometric_status=GroundingStatus.UNKNOWN,
            geometric_score=0.0,
            geometric_evidence={},
            combined_status=status,
            rejection_reasons=rejections,
        )

    def ground_region_for_requirement(
        self,
        region: ObservedRegion,
        requirement: FunctionalRequirement,
    ) -> FunctionGroundingResult:
        """Evaluate semantic satisfaction of a candidate region for a functional requirement."""
        self.total_semantic_calls += 1

        evidence_obj = self.backend.describe_region(region, requirement.description)
        evidence_dict = evidence_obj.to_dict()

        canonical_label = evidence_obj.canonical_label.lower().strip()
        accepted_categories = [c.lower().strip() for c in requirement.accepted_categories]

        rejections: list[str] = []

        if not canonical_label or canonical_label in ("unknown", "object", "spatial_region"):
            status = GroundingStatus.UNKNOWN
            score = 0.50
            rejections.append("SEMANTIC_REGION_LABEL_UNKNOWN")
        elif canonical_label in accepted_categories:
            status = GroundingStatus.PASS
            score = max(0.60, evidence_obj.confidence)
        else:
            status = GroundingStatus.FAIL
            score = 0.10
            rejections.append(f"SEMANTIC_REGION_MISMATCH_{canonical_label.upper()}_NOT_ACCEPTED")

        evidence_dict["accepted_categories"] = list(requirement.accepted_categories)
        evidence_dict["evaluated_label"] = canonical_label

        return FunctionGroundingResult(
            entity_id=region.region_instance_id,
            requirement_id=requirement.requirement_id,
            function_name=requirement.function_name,
            semantic_status=status,
            semantic_score=score,
            semantic_evidence=evidence_dict,
            geometric_status=GroundingStatus.UNKNOWN,
            geometric_score=0.0,
            geometric_evidence={},
            combined_status=status,
            rejection_reasons=rejections,
        )
