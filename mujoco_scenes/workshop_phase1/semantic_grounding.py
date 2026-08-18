"""Semantic functional grounding for objects and regions in Workshop Phase 1."""

from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_scenes.workshop_phase1.types import (
    FunctionGroundingResult,
    FunctionalRequirement,
    GroundingStatus,
    ObservedObjectTrack,
    ObservedRegion,
)


class SemanticGrounder:
    """Classifies generic visual/depth evidence into functional and semantic roles."""

    def __init__(self) -> None:
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

        # Analyze observable properties from point cloud and predicted visual labels
        pts = track.fused_points
        colors = track.fused_colors
        if pts is None or len(pts) < 10:
            res = FunctionGroundingResult(
                entity_id=inst_id,
                requirement_id=requirement.requirement_id,
                function_name=req_fn,
                semantic_status=GroundingStatus.UNKNOWN,
                semantic_score=0.0,
                semantic_evidence={"reason": "insufficient_points"},
                geometric_status=GroundingStatus.UNKNOWN,
                geometric_score=0.0,
                geometric_evidence={},
                combined_status=GroundingStatus.UNKNOWN,
                rejection_reasons=["INSUFFICIENT_EVIDENCE"],
            )
            return res

        # Extents and aspect ratios
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        extents = maxs - mins
        sorted_extents = np.sort(extents)[::-1]
        length = float(sorted_extents[0])
        mid_dim = float(sorted_extents[1])
        min_dim = float(sorted_extents[2])
        aspect_ratio = length / max(mid_dim, 1e-4)

        # Visual label consensus
        init_label = track.current_semantic_belief.get("initial_label", "object").lower()

        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []
        evidence: dict[str, Any] = {
            "initial_label": init_label,
            "estimated_length_m": round(length, 4),
            "estimated_aspect_ratio": round(aspect_ratio, 2),
        }

        if req_fn == "CAN_DRIVE_SCREW":
            # Semantic criteria: driver-like elongated hand tool or power driver
            # 1. Power driver / drill: chunky in all 3 extents (two large spans >= 0.13m and thickness >= 0.038m)
            is_power_driver = (
                "drill" in init_label
                or (length >= 0.13 and mid_dim >= 0.13 and min_dim >= 0.038 and len(pts) > 150)
            )
            # 2. Distractor flat tools (wrench, pliers): very thin 3rd dimension (<= 0.020m)
            is_distractor_tool = (
                "pliers" in init_label
                or "wrench" in init_label
                or (min_dim <= 0.020 and mid_dim >= 0.035 and not ("screw" in init_label or length < 0.06))
            )
            # 3. Screwdriver / hand driver: elongated cylindrical profile (mid_dim and min_dim both 0.018-0.040m, aspect >= 2.5)
            is_hand_driver = (
                not is_distractor_tool
                and not is_power_driver
                and ("screwdriver" in init_label or "tool" in init_label or (aspect_ratio >= 2.5 and 0.07 <= length <= 0.35 and mid_dim <= 0.045))
            )

            if is_distractor_tool:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("SEMANTIC_TOOL_TYPE_MISMATCH_NOT_SCREWDRIVER")
                evidence["inferred_category"] = "OTHER_HAND_TOOL"
            elif is_power_driver:
                status = GroundingStatus.PASS
                score = 0.95
                evidence["inferred_category"] = "POWER_DRIVER"
                evidence["drive_profile"] = "PHILLIPS"  # standard workshop drill bit
            elif is_hand_driver:
                status = GroundingStatus.PASS
                score = 0.90
                evidence["inferred_category"] = "HAND_DRIVER"
                # Profile classification
                if "flathead" in init_label or "slotted" in init_label:
                    evidence["drive_profile"] = "SLOTTED"
                else:
                    evidence["drive_profile"] = "PHILLIPS"
            else:
                if length < 0.06:  # small hardware/screw
                    status = GroundingStatus.FAIL
                    score = 0.05
                    rejections.append("SEMANTIC_OBJECT_IS_FASTENER_NOT_DRIVER")
                else:
                    status = GroundingStatus.UNKNOWN
                    score = 0.5
                    evidence["inferred_category"] = "UNKNOWN_OBJECT"

        elif req_fn == "CAN_FASTEN":
            # Semantic criteria: screw-like threaded fastener with phillips/slotted drive recess
            is_screw = "screw" in init_label or (length < 0.075 and aspect_ratio >= 1.5)
            is_bolt = "bolt" in init_label or ("hex" in init_label and length < 0.08)
            is_tool = length > 0.09 or "screwdriver" in init_label or "drill" in init_label or "pliers" in init_label or "wrench" in init_label

            if is_tool:
                status = GroundingStatus.FAIL
                score = 0.05
                rejections.append("SEMANTIC_OBJECT_IS_TOOL_NOT_FASTENER")
                if track.current_semantic_belief.get("inferred_category") not in ("HAND_DRIVER", "POWER_DRIVER"):
                    evidence["inferred_category"] = "TOOL"
            elif is_bolt:
                status = GroundingStatus.FAIL
                score = 0.20
                rejections.append("SEMANTIC_BOLT_INCOMPATIBLE_WITH_SCREW_TASK")
                evidence["inferred_category"] = "BOLT"
                evidence["fastener_profile"] = "HEX_HEAD"
            elif is_screw:
                status = GroundingStatus.PASS
                score = 0.92
                evidence["inferred_category"] = "SCREW"
                evidence["fastener_profile"] = "PHILLIPS_RECESS"
            else:
                if length < 0.08:
                    status = GroundingStatus.PASS
                    score = 0.70
                    evidence["inferred_category"] = "FASTENER"
                    evidence["fastener_profile"] = "PHILLIPS_RECESS"
                else:
                    status = GroundingStatus.FAIL
                    score = 0.1
                    rejections.append("SEMANTIC_NOT_A_FASTENER")

        # Cache result
        self._object_cache[cache_key] = {
            "status": status,
            "score": score,
            "evidence": evidence,
            "rejections": rejections,
            "evidence_count": track.evidence_count,
        }
        track.current_semantic_belief.update(evidence)

        return FunctionGroundingResult(
            entity_id=inst_id,
            requirement_id=requirement.requirement_id,
            function_name=req_fn,
            semantic_status=status,
            semantic_score=score,
            semantic_evidence=evidence,
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

        bounds = region.proposal_bounds_m
        b_min = np.array(bounds["minimum_world_m"])
        b_max = np.array(bounds["maximum_world_m"])
        dims = b_max - b_min
        area_m2 = float(dims[0] * dims[1])
        height_m = float(dims[2])
        z_min = float(b_min[2])

        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []
        evidence: dict[str, Any] = {
            "bounding_dimensions_m": [round(float(v), 4) for v in dims],
            "planar_bounding_area_m2": round(area_m2, 4),
        }

        # Elevated wall shelf (z_min >= 0.95m) or large table/cart surfaces (area >= 0.05m2)
        is_wall_shelf = (z_min >= 0.95 and area_m2 >= 0.012 and height_m <= 0.050)
        is_large_surface = (area_m2 >= 0.050 and height_m <= 0.055)
        is_surface_like = is_large_surface or is_wall_shelf

        # Trays and bins resting on the table with cavity/rims (area <= 0.035m2, height >= 0.025m, z_min < 0.95m)
        is_container_like = (area_m2 <= 0.035 and height_m >= 0.025 and z_min < 0.95)

        if req_fn == "WORK_SURFACE":
            # Surfaces are flat, broad areas suitable for tool/hardware staging
            if is_surface_like and not is_container_like:
                status = GroundingStatus.PASS
                score = 0.95
                evidence["inferred_category"] = "SUPPORT_SURFACE"
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("SEMANTIC_REGION_NOT_A_FLAT_WORK_SURFACE")

        elif req_fn == "SMALL_PARTS_CONTAINER":
            # Containers have depth/cavity structure with rim walls for loose small parts
            if is_container_like:
                status = GroundingStatus.PASS
                score = 0.90
                evidence["inferred_category"] = "PARTS_CONTAINER"
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("SEMANTIC_REGION_NOT_A_PARTS_CONTAINER")

        self._region_cache[cache_key] = {
            "status": status,
            "score": score,
            "evidence": evidence,
            "rejections": rejections,
        }
        region.current_semantic_belief.update(evidence)

        return FunctionGroundingResult(
            entity_id=r_id,
            requirement_id=requirement.requirement_id,
            function_name=req_fn,
            semantic_status=status,
            semantic_score=score,
            semantic_evidence=evidence,
            geometric_status=GroundingStatus.UNKNOWN,
            geometric_score=0.0,
            geometric_evidence={},
            combined_status=status,
            rejection_reasons=rejections,
        )
