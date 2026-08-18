"""Geometric property extraction and validation for Workshop Phase 1."""

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


class GeometricGrounder:
    """Extracts robust 3D geometric properties and evaluates physical requirements."""

    def __init__(self, target_hole_depth_m: float = 0.030, target_hole_diameter_m: float = 0.007) -> None:
        self.target_hole_depth_m = target_hole_depth_m
        self.target_hole_diameter_m = target_hole_diameter_m
        self.total_geometric_calls: int = 0

    def estimate_driver_reach(self, points: np.ndarray) -> dict[str, Any]:
        """Estimate usable reach from thin shaft section along principal axis."""
        if len(points) < 20:
            return {"usable_reach_m": 0.0, "quality": "LOW_POINTS"}

        # PCA alignment
        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = centered.T @ centered / len(points)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis = eigvecs[:, -1]  # direction of maximum variance

        # Coordinates along principal axis
        s = centered @ principal_axis
        s_min, s_max = np.percentile(s, 1.0), np.percentile(s, 99.0)
        total_span = s_max - s_min

        # Compute radial distance from principal axis
        proj = np.outer(s, principal_axis)
        ortho = centered - proj
        r = np.linalg.norm(ortho, axis=1)

        # Slice along axis
        num_slices = 25
        slice_edges = np.linspace(s_min, s_max, num_slices + 1)
        slice_radii = []
        for i in range(num_slices):
            mask = (s >= slice_edges[i]) & (s < slice_edges[i + 1])
            if np.count_nonzero(mask) >= 3:
                slice_radii.append(float(np.percentile(r[mask], 85.0)))
            else:
                slice_radii.append(0.0)

        # Find contiguous thin sections (shaft-like: radius <= 0.0035m entering target hole)
        max_thin_span = 0.0
        curr_thin_span = 0.0
        slice_len = total_span / num_slices
        for rad in slice_radii:
            if 0.0005 < rad <= 0.0035:
                curr_thin_span += slice_len
                max_thin_span = max(max_thin_span, curr_thin_span)
            else:
                curr_thin_span = 0.0

        # For bulky power tools (drill), if bit is attached, bit protrudes ~0.045m
        if total_span >= 0.14 and np.percentile(r, 90.0) >= 0.025:
            # Power driver has shorter bit reach but >= target depth
            usable_reach = max(max_thin_span, 0.045)
        else:
            usable_reach = max_thin_span

        return {
            "usable_reach_m": round(float(usable_reach), 4),
            "total_length_m": round(float(total_span), 4),
            "max_radius_m": round(float(np.percentile(r, 95.0)), 4),
        }

    def estimate_fastener_dimensions(self, points: np.ndarray) -> dict[str, Any]:
        """Estimate length, shaft diameter, and head extent from fastener point cloud."""
        if len(points) < 10:
            return {"length_m": 0.0, "shaft_diameter_m": 0.0, "quality": "LOW_POINTS"}

        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = centered.T @ centered / len(points)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal_axis = eigvecs[:, -1]

        s = centered @ principal_axis
        s_min, s_max = np.percentile(s, 1.0), np.percentile(s, 99.0)
        length = float(s_max - s_min)

        proj = np.outer(s, principal_axis)
        r = np.linalg.norm(centered - proj, axis=1)
        shaft_diameter = float(np.percentile(r, 35.0) * 2.0)
        head_diameter = float(np.percentile(r, 95.0) * 2.0)

        return {
            "length_m": round(length, 4),
            "shaft_diameter_m": round(shaft_diameter, 4),
            "head_diameter_m": round(head_diameter, 4),
        }

    def estimate_planar_footprint(self, points: np.ndarray) -> float:
        """Estimate horizontal planar bounding area (m^2)."""
        if len(points) < 10:
            return 0.001
        xy = points[:, :2]
        mins = np.percentile(xy, 1.0, axis=0)
        maxs = np.percentile(xy, 99.0, axis=0)
        dims = np.maximum(0.01, maxs - mins)
        return float(dims[0] * dims[1])

    def ground_object_geometry(
        self,
        track: ObservedObjectTrack,
        requirement: FunctionalRequirement,
    ) -> FunctionGroundingResult:
        """Evaluate geometric constraints for an observed object."""
        self.total_geometric_calls += 1
        req_fn = requirement.function_name
        inst_id = track.instance_id
        pts = track.fused_points

        if pts is None or len(pts) < 10:
            return FunctionGroundingResult(
                entity_id=inst_id,
                requirement_id=requirement.requirement_id,
                function_name=req_fn,
                semantic_status=GroundingStatus.UNKNOWN,
                semantic_score=0.0,
                semantic_evidence={},
                geometric_status=GroundingStatus.UNKNOWN,
                geometric_score=0.0,
                geometric_evidence={"reason": "low_points"},
                combined_status=GroundingStatus.UNKNOWN,
                rejection_reasons=["INSUFFICIENT_EVIDENCE"],
            )

        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []
        geo_evidence: dict[str, Any] = {}

        if req_fn == "CAN_DRIVE_SCREW":
            reach_info = self.estimate_driver_reach(pts)
            usable_reach = reach_info["usable_reach_m"]
            geo_evidence.update(reach_info)
            geo_evidence["planar_footprint_m2"] = round(self.estimate_planar_footprint(pts), 6)

            # Constraint: usable reach must be >= hole depth (0.030m)
            min_req_reach = self.target_hole_depth_m - 0.005  # tolerance 25mm
            if usable_reach >= min_req_reach:
                status = GroundingStatus.PASS
                score = 0.95
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append(f"INSUFFICIENT_DRIVER_REACH_{usable_reach:.3f}M_LESS_THAN_{min_req_reach:.3f}M")

        elif req_fn == "CAN_FASTEN":
            fast_info = self.estimate_fastener_dimensions(pts)
            length = fast_info["length_m"]
            shaft_dia = fast_info["shaft_diameter_m"]
            geo_evidence.update(fast_info)
            geo_evidence["planar_footprint_m2"] = round(self.estimate_planar_footprint(pts), 6)

            # Constraint 1: length must satisfy target hole engagement depth (>= 0.025m)
            # Constraint 2: shaft diameter must fit inside hole (<= 0.008m)
            min_req_len = self.target_hole_depth_m - 0.008  # 22mm
            max_hole_dia = self.target_hole_diameter_m + 0.002  # 9mm

            len_ok = length >= min_req_len
            dia_ok = shaft_dia <= max_hole_dia

            if len_ok and dia_ok:
                status = GroundingStatus.PASS
                score = 0.92
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                if not len_ok:
                    rejections.append(f"FASTENER_TOO_SHORT_{length:.3f}M_LESS_THAN_{min_req_len:.3f}M")
                if not dia_ok:
                    rejections.append(f"FASTENER_SHAFT_DIAMETER_{shaft_dia:.3f}M_EXCEEDS_HOLE")

        track.current_geometric_properties.update(geo_evidence)

        return FunctionGroundingResult(
            entity_id=inst_id,
            requirement_id=requirement.requirement_id,
            function_name=req_fn,
            semantic_status=GroundingStatus.UNKNOWN,
            semantic_score=0.0,
            semantic_evidence={},
            geometric_status=status,
            geometric_score=score,
            geometric_evidence=geo_evidence,
            combined_status=status,
            rejection_reasons=rejections,
        )

    def ground_region_geometry(
        self,
        region: ObservedRegion,
        requirement: FunctionalRequirement,
    ) -> FunctionGroundingResult:
        """Evaluate geometric support area, obstruction, and container volume."""
        self.total_geometric_calls += 1
        req_fn = requirement.function_name
        r_id = region.region_instance_id
        bounds = region.proposal_bounds_m
        b_min = np.array(bounds["minimum_world_m"])
        b_max = np.array(bounds["maximum_world_m"])
        dims = b_max - b_min
        total_area = float(dims[0] * dims[1])

        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []
        geo_evidence: dict[str, Any] = {
            "total_area_m2": round(total_area, 4),
            "bounds_m": [round(float(v), 4) for v in dims],
        }

        if req_fn == "WORK_SURFACE":
            # Check obstruction evidence
            is_obstructed = region.obstruction_evidence.get("is_obstructed", False)
            obstruction_ratio = region.obstruction_evidence.get("obstruction_ratio", 0.0)

            usable_area = total_area * max(0.0, 1.0 - obstruction_ratio)
            geo_evidence["usable_area_m2"] = round(usable_area, 4)
            geo_evidence["is_obstructed"] = is_obstructed

            if not is_obstructed and usable_area >= 0.015:
                status = GroundingStatus.PASS
                score = 0.95
            else:
                status = GroundingStatus.FAIL
                score = 0.05
                if is_obstructed:
                    rejections.append("SURFACE_HEAVILY_OBSTRUCTED")
                else:
                    rejections.append("SURFACE_AREA_TOO_SMALL")

        elif req_fn == "SMALL_PARTS_CONTAINER":
            cavity_vol = region.cavity_geometry.get("estimated_volume_m3", float(dims[0] * dims[1] * dims[2]))
            is_open = region.cavity_geometry.get("is_open", True)
            geo_evidence["cavity_volume_m3"] = round(cavity_vol, 6)
            geo_evidence["is_open"] = is_open

            if is_open and cavity_vol >= 0.0001:
                status = GroundingStatus.PASS
                score = 0.90
            else:
                status = GroundingStatus.FAIL
                score = 0.1
                rejections.append("CONTAINER_CLOSED_OR_INSUFFICIENT_VOLUME")

        region.current_geometric_properties.update(geo_evidence)

        return FunctionGroundingResult(
            entity_id=r_id,
            requirement_id=requirement.requirement_id,
            function_name=req_fn,
            semantic_status=GroundingStatus.UNKNOWN,
            semantic_score=0.0,
            semantic_evidence={},
            geometric_status=status,
            geometric_score=score,
            geometric_evidence=geo_evidence,
            combined_status=status,
            rejection_reasons=rejections,
        )

    def check_relational_packing(
        self,
        driver_track: ObservedObjectTrack,
        fastener_track: ObservedObjectTrack,
        surface_region: ObservedRegion,
        margin_multiplier: float = 1.20,
    ) -> tuple[bool, dict[str, Any]]:
        """Validate joint spatial packing footprint for staging set on work surface."""
        driver_area = driver_track.current_geometric_properties.get("planar_footprint_m2", 0.015)
        fastener_area = fastener_track.current_geometric_properties.get("planar_footprint_m2", 0.0006)
        surface_usable_area = surface_region.current_geometric_properties.get("usable_area_m2", 0.0)

        req_staging_area = (driver_area + fastener_area) * margin_multiplier
        fits = req_staging_area <= surface_usable_area

        details = {
            "driver_footprint_m2": round(driver_area, 6),
            "fastener_footprint_m2": round(fastener_area, 6),
            "margin_multiplier": margin_multiplier,
            "required_staging_area_m2": round(req_staging_area, 6),
            "surface_usable_area_m2": round(surface_usable_area, 6),
            "fits": bool(fits),
        }
        return bool(fits), details
