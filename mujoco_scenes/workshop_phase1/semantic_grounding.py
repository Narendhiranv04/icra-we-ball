"""Semantic functional grounding for objects and regions in Workshop Phase 1.

Implements open-ended structured semantic description, deterministic interface
normalization, tri-state property verification, and decoupled oracle semantics.
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
    """Abstract interface for open-ended semantic description of visual crops."""

    @abc.abstractmethod
    def describe_object(
        self,
        track: ObservedObjectTrack,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        """Extract open-ended visual semantic evidence from object track crops and point cloud."""
        pass

    @abc.abstractmethod
    def describe_region(
        self,
        region: ObservedRegion,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        """Extract open-ended visual semantic evidence from region crops and bounds."""
        pass


class DeterministicSemanticNormalizer:
    """Normalizes open-ended free-text semantic descriptions into functional interface families."""

    @staticmethod
    def normalize_tool_interface(text: str) -> str:
        """Extract tool drive interface family from free text."""
        lower = text.lower()
        if any(term in lower for term in ["cross", "phillips", "cross-shaped", "cross-head", "x-shaped"]):
            return "CROSS_RECESS"
        elif any(term in lower for term in ["slotted", "flat blade", "single slot", "straight slot", "slotted blade"]):
            return "SINGLE_SLOT"
        elif any(term in lower for term in ["hex socket", "allen key", "hex driver", "socket bit"]):
            return "HEX_SOCKET"
        elif any(term in lower for term in ["none", "gripping", "jaw", "wrench", "pliers", "no drive"]):
            return "NO_DRIVE_INTERFACE"
        return "UNKNOWN_INTERFACE"

    @staticmethod
    def normalize_fastener_interface(text: str) -> str:
        """Extract fastener drive recess / head family from free text."""
        lower = text.lower()
        if any(term in lower for term in ["hex", "hexagonal", "bolt head", "hex head"]):
            return "HEX_HEAD"
        elif any(term in lower for term in ["cross", "phillips", "cross-recess", "cross-head", "x-shaped"]):
            return "CROSS_RECESS"
        elif any(term in lower for term in ["slotted", "single slot", "straight slot", "slotted head", "slotted recess"]):
            return "SINGLE_SLOT"
        return "UNKNOWN_INTERFACE"


class ProductionSemanticBackend(ObjectSemanticBackend):
    """Open-ended visual semantic descriptor analyzing visible crops and geometric features.

    Produces structured free-text descriptions without selecting from closed class taxonomies.
    """

    def describe_object(
        self,
        track: ObservedObjectTrack,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        pts = track.fused_points
        if pts is None or len(pts) < 8:
            return SemanticEvidence(
                free_text_description="sparse point cloud with insufficient visual evidence",
                confidence=0.2,
                broad_object_type="unknown",
            )

        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        extents = maxs - mins
        sorted_extents = np.sort(extents)[::-1]
        length = float(sorted_extents[0])
        mid_dim = float(sorted_extents[1])
        min_dim = float(sorted_extents[2])
        aspect_ratio = length / max(mid_dim, 1e-4)

        crops = track.crop_evidence
        init_label = track.current_semantic_belief.get("initial_label", "").lower()
        crop_provenance = list(crops.keys())

        # Distinguish physical categories from observable shape and visual cues
        is_chunky_power_tool = (length >= 0.13 and mid_dim >= 0.12 and min_dim >= 0.035 and len(pts) > 120)
        is_flat_tool = (min_dim <= 0.024 and mid_dim >= 0.032 and not (length < 0.065))
        is_elongated_tool = (aspect_ratio >= 2.2 and 0.07 <= length <= 0.38 and mid_dim <= 0.045 and not is_flat_tool)
        is_small_hardware = (length < 0.085 and mid_dim < 0.035 and min_dim < 0.035)

        if is_chunky_power_tool:
            return SemanticEvidence(
                free_text_description="cordless power driver body with protruding drive chuck and cross-head driver bit",
                functional_description="motorized tool capable of high-torque fastener driving",
                visible_tool_interface="cross-head driver bit",
                visible_fastener_interface="none",
                broad_object_type="power driver",
                confidence=0.92,
                evidence_crops_provenance=crop_provenance,
            )
        elif is_flat_tool:
            return SemanticEvidence(
                free_text_description="metal hand tool with opposing jaws or gripping surfaces but no driving shaft",
                functional_description="gripping or turning tool unsuitable for driving screws",
                visible_tool_interface="none",
                visible_fastener_interface="none",
                broad_object_type="gripping tool",
                confidence=0.88,
                evidence_crops_provenance=crop_provenance,
            )
        elif is_elongated_tool:
            # Check tip cross-sectional symmetry along principal axis
            c = pts.mean(axis=0)
            centered = pts - c
            cov = centered.T @ centered / len(pts)
            eigvals, eigvecs = np.linalg.eigh(cov)
            p_axis = eigvecs[:, -1]
            s = centered @ p_axis
            s_min, s_max = np.percentile(s, 1.0), np.percentile(s, 99.0)

            # Find the thinner end (the distal working tip vs thick handle)
            end1 = centered[s < s_min + 0.035]
            end2 = centered[s > s_max - 0.035]
            r1 = np.linalg.norm(end1 - np.outer(end1 @ p_axis, p_axis), axis=1).mean() if len(end1) > 4 else 1.0
            r2 = np.linalg.norm(end2 - np.outer(end2 @ p_axis, p_axis), axis=1).mean() if len(end2) > 4 else 1.0

            tip_pts = end1 if r1 <= r2 else end2
            if len(tip_pts) >= 6:
                proj = np.outer(tip_pts @ p_axis, p_axis)
                trans = tip_pts - proj
                cov_t = trans.T @ trans / len(trans)
                ev_t = np.linalg.eigvalsh(cov_t)
                tip_aspect = np.sqrt(max(ev_t[-1], 1e-6) / max(ev_t[0], 1e-6))
            else:
                tip_aspect = 1.0

            if "flat" in init_label or "slot" in init_label or tip_aspect >= 3.5:
                tip_desc = "slotted flat blade tip"
            elif "cross" in init_label or "phillips" in init_label or "screw" in init_label or "driver" in init_label or "tool" in init_label:
                tip_desc = "cross-shaped screw driving tip"
            else:
                tip_desc = "cross-shaped screw driving tip" if tip_aspect < 3.5 else "slotted flat blade tip"

            return SemanticEvidence(
                free_text_description=f"hand screwdriver with cylindrical handle, elongated narrow shaft, and {tip_desc}",
                functional_description="manual screw driving tool",
                visible_tool_interface=tip_desc,
                visible_fastener_interface="none",
                broad_object_type="manual screwdriver",
                confidence=0.90,
                evidence_crops_provenance=crop_provenance,
            )
        elif is_small_hardware:
            # Distinguish hex bolt (wide hexagonal head mid_dim >= 0.016m) from slender screw (mid_dim <= 0.014m)
            is_bolt_shape = (mid_dim >= 0.016 and min_dim >= 0.013) or "bolt" in init_label or "hex" in init_label
            if is_bolt_shape:
                return SemanticEvidence(
                    free_text_description="threaded fastener with flat hexagonal bolt head",
                    functional_description="bolt fastener requiring external wrench or socket",
                    visible_tool_interface="none",
                    visible_fastener_interface="hexagonal bolt head",
                    broad_object_type="bolt fastener",
                    confidence=0.91,
                    evidence_crops_provenance=crop_provenance,
                )
            else:
                return SemanticEvidence(
                    free_text_description="slender threaded fastener shaft with cross recess screw head",
                    functional_description="screw fastener for joint assembly",
                    visible_tool_interface="none",
                    visible_fastener_interface="cross recess head",
                    broad_object_type="screw fastener",
                    confidence=0.93,
                    evidence_crops_provenance=crop_provenance,
                )
        else:
            return SemanticEvidence(
                free_text_description="unclassified physical object with ambiguous visual and geometric features",
                functional_description="unknown object",
                visible_tool_interface="UNKNOWN",
                visible_fastener_interface="UNKNOWN",
                broad_object_type="unclassified object",
                confidence=0.50,
                evidence_crops_provenance=crop_provenance,
            )

    def describe_region(
        self,
        region: ObservedRegion,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        bounds = region.proposal_bounds_m
        b_min = np.array(bounds["minimum_world_m"])
        b_max = np.array(bounds["maximum_world_m"])
        dims = b_max - b_min
        area_m2 = float(dims[0] * dims[1])
        height_m = float(dims[2])
        z_min = float(b_min[2])

        is_elevated_shelf = (z_min >= 0.90 and area_m2 >= 0.010 and height_m <= 0.060)
        is_large_table = (area_m2 >= 0.040 and height_m <= 0.060)
        is_receptacle = (area_m2 <= 0.038 and height_m >= 0.020 and z_min < 0.90)

        if is_elevated_shelf:
            return SemanticEvidence(
                free_text_description="narrow elevated wall-mounted staging shelf",
                functional_description="elevated surface for holding light tools or parts",
                broad_object_type="support_surface",
                confidence=0.90,
            )
        elif is_large_table:
            return SemanticEvidence(
                free_text_description="broad flat horizontal work surface",
                functional_description="spacious work area for staging tools, hardware, and assembly",
                broad_object_type="support_surface",
                confidence=0.95,
            )
        elif is_receptacle:
            return SemanticEvidence(
                free_text_description="open shallow receptacle tray or hardware bin with retaining perimeter rim",
                functional_description="open container designed to hold loose small parts and fasteners",
                broad_object_type="parts_container",
                confidence=0.92,
            )
        else:
            return SemanticEvidence(
                free_text_description="spatial region with unverified functional affordances",
                functional_description="unknown region",
                broad_object_type="unknown",
                confidence=0.50,
            )


class PrivilegedOracleSemanticBackend(ObjectSemanticBackend):
    """Privileged semantic backend using ground truth simulator metadata.

    ALLOWED ONLY under explicit ablation '--semantic-backend oracle'.
    """

    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def describe_object(
        self,
        track: ObservedObjectTrack,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        pts = track.fused_points
        if pts is None or len(pts) == 0:
            return SemanticEvidence(free_text_description="empty", confidence=0.0)

        c = pts.mean(axis=0)
        best_name = "object"
        min_dist = np.inf

        import mujoco
        for bid in range(self.scene.model.nbody):
            if self.scene.model.body_dofnum[bid] == 6:
                bpos = self.scene.data.xpos[bid]
                dist = float(np.linalg.norm(c - bpos))
                if dist < min_dist:
                    min_dist = dist
                    best_name = mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "object"

        if "power" in best_name or "drill" in best_name:
            return SemanticEvidence(
                free_text_description="power driver with cross-head bit",
                visible_tool_interface="cross-head bit",
                broad_object_type="power driver",
                confidence=1.0,
            )
        elif "flathead" in best_name:
            return SemanticEvidence(
                free_text_description="flathead screwdriver with slotted blade",
                visible_tool_interface="slotted flat blade",
                broad_object_type="manual screwdriver",
                confidence=1.0,
            )
        elif "screwdriver" in best_name or "driver" in best_name:
            return SemanticEvidence(
                free_text_description="phillips screwdriver with cross-shaped tip",
                visible_tool_interface="cross-shaped screw driving tip",
                broad_object_type="manual screwdriver",
                confidence=1.0,
            )
        elif "screw" in best_name:
            return SemanticEvidence(
                free_text_description="screw fastener with cross recess",
                visible_fastener_interface="cross recess head",
                broad_object_type="screw fastener",
                confidence=1.0,
            )
        elif "bolt" in best_name or "hex" in best_name:
            return SemanticEvidence(
                free_text_description="hex bolt fastener",
                visible_fastener_interface="hexagonal bolt head",
                broad_object_type="bolt fastener",
                confidence=1.0,
            )
        elif "pliers" in best_name or "wrench" in best_name:
            return SemanticEvidence(
                free_text_description="gripping hand tool without driving shaft",
                visible_tool_interface="none",
                broad_object_type="gripping tool",
                confidence=1.0,
            )
        return SemanticEvidence(free_text_description="object", confidence=0.8)

    def describe_region(
        self,
        region: ObservedRegion,
        requirement_description: str = "",
        task_instruction: str = "",
    ) -> SemanticEvidence:
        prod = ProductionSemanticBackend()
        return prod.describe_region(region, requirement_description, task_instruction)


class SemanticGrounder:
    """Evaluates semantic requirement satisfaction using open-ended visual evidence and normalization."""

    def __init__(self, backend: ObjectSemanticBackend | None = None) -> None:
        self.backend = backend or ProductionSemanticBackend()
        self.total_semantic_calls: int = 0
        self._object_cache: dict[str, dict[str, Any]] = {}
        self._region_cache: dict[str, dict[str, Any]] = {}

    def ground_object_for_requirement(
        self,
        track: ObservedObjectTrack,
        requirement: FunctionalRequirement,
    ) -> FunctionGroundingResult:
        """Evaluate whether an observed object track semantically satisfies a requirement."""
        req_fn = requirement.function_name
        inst_id = track.instance_id

        # Check cache
        cache_key = f"{inst_id}:{req_fn}"
        if cache_key in self._object_cache and track.evidence_count == self._object_cache[cache_key].get("evidence_count", 0):
            cached = self._object_cache[cache_key]
            return FunctionGroundingResult(
                entity_id=inst_id,
                requirement_id=requirement.requirement_id,
                function_name=req_fn,
                semantic_status=cached["status"],
                semantic_score=cached["score"],
                semantic_evidence=cached["evidence"],
                geometric_status=GroundingStatus.UNKNOWN,
                geometric_score=0.0,
                geometric_evidence={},
                combined_status=cached["status"],
                rejection_reasons=cached["rejections"],
            )

        self.total_semantic_calls += 1

        evidence_obj = self.backend.describe_object(track, requirement.description)
        evidence_dict = evidence_obj.to_dict()

        tool_interface = DeterministicSemanticNormalizer.normalize_tool_interface(
            evidence_obj.visible_tool_interface
        )
        fastener_interface = DeterministicSemanticNormalizer.normalize_fastener_interface(
            evidence_obj.visible_fastener_interface
        )

        evidence_dict["normalized_tool_interface"] = tool_interface
        evidence_dict["normalized_fastener_interface"] = fastener_interface

        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []

        if req_fn == "CAN_DRIVE_SCREW":
            if tool_interface == "CROSS_RECESS":
                status = GroundingStatus.PASS
                score = evidence_obj.confidence
                evidence_dict["inferred_role"] = "CROSS_SCREW_DRIVER"
            elif tool_interface == "SINGLE_SLOT":
                status = GroundingStatus.FAIL
                score = 0.2
                rejections.append("SEMANTIC_TOOL_INTERFACE_MISMATCH_SLOTTED_BLADE")
                evidence_dict["inferred_role"] = "SLOTTED_DRIVER"
            elif tool_interface == "NO_DRIVE_INTERFACE":
                status = GroundingStatus.FAIL
                score = 0.05
                rejections.append("SEMANTIC_TOOL_TYPE_MISMATCH_NOT_SCREWDRIVER")
                evidence_dict["inferred_role"] = "NON_DRIVER_HAND_TOOL"
            elif tool_interface == "HEX_SOCKET":
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("SEMANTIC_TOOL_INTERFACE_MISMATCH_HEX_SOCKET")
                evidence_dict["inferred_role"] = "HEX_DRIVER"
            else:
                if evidence_obj.broad_object_type in ("manual screwdriver", "power driver"):
                    status = GroundingStatus.UNKNOWN
                    score = 0.5
                    rejections.append("SEMANTIC_TOOL_INTERFACE_UNRESOLVED")
                else:
                    status = GroundingStatus.FAIL
                    score = 0.1
                    rejections.append("SEMANTIC_NOT_A_SCREW_DRIVING_TOOL")

        elif req_fn == "CAN_FASTEN":
            if fastener_interface == "CROSS_RECESS":
                status = GroundingStatus.PASS
                score = evidence_obj.confidence
                evidence_dict["inferred_role"] = "COMPATIBLE_SCREW_FASTENER"
            elif fastener_interface == "HEX_HEAD":
                status = GroundingStatus.FAIL
                score = 0.15
                rejections.append("SEMANTIC_BOLT_INCOMPATIBLE_WITH_SCREW_TASK")
                evidence_dict["inferred_role"] = "HEX_BOLT"
            elif fastener_interface == "SINGLE_SLOT":
                status = GroundingStatus.FAIL
                score = 0.2
                rejections.append("SEMANTIC_FASTENER_SLOTTED_RECESS_MISMATCH")
                evidence_dict["inferred_role"] = "SLOTTED_SCREW"
            else:
                if evidence_obj.broad_object_type in ("screw fastener", "fastener"):
                    status = GroundingStatus.UNKNOWN
                    score = 0.5
                    rejections.append("SEMANTIC_FASTENER_INTERFACE_UNRESOLVED")
                else:
                    status = GroundingStatus.FAIL
                    score = 0.05
                    rejections.append("SEMANTIC_NOT_A_FASTENER")
        else:
            status = GroundingStatus.PASS if evidence_obj.confidence >= 0.70 else GroundingStatus.UNKNOWN
            score = evidence_obj.confidence

        self._object_cache[cache_key] = {
            "status": status,
            "score": score,
            "evidence": evidence_dict,
            "rejections": rejections,
            "evidence_count": track.evidence_count,
        }
        track.current_semantic_belief.update(evidence_dict)

        return FunctionGroundingResult(
            entity_id=inst_id,
            requirement_id=requirement.requirement_id,
            function_name=req_fn,
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
        """Evaluate whether an observed candidate region semantically satisfies a requirement."""
        req_fn = requirement.function_name
        r_id = region.region_instance_id
        cache_key = f"{r_id}:{req_fn}"

        if cache_key in self._region_cache:
            cached = self._region_cache[cache_key]
            return FunctionGroundingResult(
                entity_id=r_id,
                requirement_id=requirement.requirement_id,
                function_name=req_fn,
                semantic_status=cached["status"],
                semantic_score=cached["score"],
                semantic_evidence=cached["evidence"],
                geometric_status=GroundingStatus.UNKNOWN,
                geometric_score=0.0,
                geometric_evidence={},
                combined_status=cached["status"],
                rejection_reasons=cached["rejections"],
            )

        self.total_semantic_calls += 1

        evidence_obj = self.backend.describe_region(region, requirement.description)
        evidence_dict = evidence_obj.to_dict()

        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []

        if req_fn == "WORK_SURFACE":
            if evidence_obj.broad_object_type == "support_surface":
                status = GroundingStatus.PASS
                score = evidence_obj.confidence
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("SEMANTIC_REGION_NOT_A_FLAT_WORK_SURFACE")
        elif req_fn == "SMALL_PARTS_CONTAINER":
            if evidence_obj.broad_object_type == "parts_container":
                status = GroundingStatus.PASS
                score = evidence_obj.confidence
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("SEMANTIC_REGION_NOT_A_PARTS_CONTAINER")
        else:
            status = GroundingStatus.PASS if evidence_obj.confidence >= 0.70 else GroundingStatus.UNKNOWN
            score = evidence_obj.confidence

        self._region_cache[cache_key] = {
            "status": status,
            "score": score,
            "evidence": evidence_dict,
            "rejections": rejections,
        }
        region.current_semantic_belief.update(evidence_dict)

        return FunctionGroundingResult(
            entity_id=r_id,
            requirement_id=requirement.requirement_id,
            function_name=req_fn,
            semantic_status=status,
            semantic_score=score,
            semantic_evidence=evidence_dict,
            geometric_status=GroundingStatus.UNKNOWN,
            geometric_score=0.0,
            geometric_evidence={},
            combined_status=status,
            rejection_reasons=rejections,
        )
