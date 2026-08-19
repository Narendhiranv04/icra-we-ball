"""Geometric property extraction and validation for Workshop Phase 1.

Implements observable target recess estimation from RGB-D, multi-axis tool reach slicing,
tip transverse eigenvalue aspect analysis, fastener dimension fitting, cavity volume
calculation, relational packing, and quality metadata propagation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
)
from mujoco_scenes.workshop_phase1.types import (
    FunctionGroundingResult,
    FunctionalRequirement,
    GroundingStatus,
    ObservedObjectTrack,
    ObservedRegion,
    TargetGeometryEvidence,
    ViewObservation,
)


class GeometricGrounder:
    """Extracts robust 3D geometric properties and evaluates physical requirements."""

    def __init__(
        self,
        target_hole_depth_m: float = 0.030,
        target_hole_diameter_m: float = 0.007,
        min_driver_reach_m: float = 0.025,
        min_fastener_length_m: float = 0.022,
        min_surface_area_m2: float = 0.015,
        min_container_volume_m3: float = 0.0001,
        staging_margin_multiplier: float = 1.20,
        target_evidence: TargetGeometryEvidence | None = None,
    ) -> None:
        self.target_evidence = target_evidence
        if target_evidence and target_evidence.validity == GroundingStatus.PASS:
            self.target_hole_depth_m = (
                target_evidence.estimated_recess_depth_m
                if target_evidence.estimated_recess_depth_m is not None
                else target_hole_depth_m
            )
            self.target_hole_diameter_m = (
                target_evidence.estimated_opening_diameter_m
                if target_evidence.estimated_opening_diameter_m is not None
                else target_hole_diameter_m
            )
        else:
            self.target_hole_depth_m = target_hole_depth_m
            self.target_hole_diameter_m = target_hole_diameter_m

        self.min_driver_reach_m = float(min_driver_reach_m)
        self.min_fastener_length_m = float(min_fastener_length_m)
        self.min_surface_area_m2 = float(min_surface_area_m2)
        self.min_container_volume_m3 = float(min_container_volume_m3)
        self.staging_margin_multiplier = float(staging_margin_multiplier)

        self.total_geometric_calls: int = 0

    @staticmethod
    def observe_target_recess(
        observations: list[ViewObservation],
        scene: Any | None = None,
    ) -> TargetGeometryEvidence:
        """Estimate target workpiece recess position, opening diameter, and depth from multi-view RGB-D."""
        if scene is not None and hasattr(scene, "get_target_workpiece_specification"):
            spec = scene.get_target_workpiece_specification()
            fixture_center = np.array(spec.get("fixture_center_world_m", [-0.15, 0.50, 0.68]), dtype=float)
        else:
            fixture_center = np.array([-0.15, 0.50, 0.68], dtype=float)

        target_volume_min = fixture_center - np.array([0.12, 0.12, 0.10])
        target_volume_max = fixture_center + np.array([0.12, 0.12, 0.15])

        pts_list = []
        views = []
        for obs in observations:
            full_mask = np.ones(obs.depth_m.shape, dtype=bool)
            pts, _ = backproject_masked_depth(
                obs.depth_m,
                full_mask,
                obs.intrinsics,
                obs.camera_position_world,
                obs.camera_rotation_world,
                max_depth=3.0,
            )
            if len(pts) > 0:
                gated = gate_points_to_volume(
                    pts,
                    minimum_world_m=target_volume_min,
                    maximum_world_m=target_volume_max,
                    boundary_margin_m=0.0,
                )
                if np.count_nonzero(gated) > 20:
                    pts_list.append(pts[gated])
                    views.append(obs.camera_id)

        if not pts_list or sum(len(p) for p in pts_list) < 30:
            return TargetGeometryEvidence(
                target_position=None,
                estimated_opening_diameter_m=None,
                estimated_recess_depth_m=None,
                point_count=sum(len(p) for p in pts_list) if pts_list else 0,
                source_views=views,
                confidence=0.0,
                validity=GroundingStatus.UNKNOWN,
                quality_metadata={"reason": "insufficient_target_recess_points"},
            )

        all_pts = np.vstack(pts_list)
        z_vals = all_pts[:, 2]
        z_surface = float(np.percentile(z_vals, 95))
        depth_residuals = z_surface - z_vals

        recess_pts = all_pts[depth_residuals > 0.010]
        if len(recess_pts) >= 15:
            measured_depth = float(np.percentile(depth_residuals, 98.0))
            entry_pts = all_pts[(depth_residuals >= 0.005) & (depth_residuals <= 0.020)]
            if len(entry_pts) > 10:
                xy_span = np.percentile(entry_pts[:, :2], 95, axis=0) - np.percentile(entry_pts[:, :2], 5, axis=0)
                measured_diam = float(min(xy_span[0], xy_span[1]))
            else:
                measured_diam = float(0.010)

            return TargetGeometryEvidence(
                target_position=fixture_center,
                estimated_opening_diameter_m=round(measured_diam, 4),
                estimated_recess_depth_m=round(measured_depth, 4),
                point_count=len(all_pts),
                source_views=views,
                confidence=0.95,
                validity=GroundingStatus.PASS,
                quality_metadata={"observation_method": "multi_view_depth_residual"},
            )
        else:
            return TargetGeometryEvidence(
                target_position=fixture_center,
                estimated_opening_diameter_m=None,
                estimated_recess_depth_m=None,
                point_count=len(all_pts),
                source_views=views,
                confidence=0.30,
                validity=GroundingStatus.UNKNOWN,
                quality_metadata={"reason": "sparse_recess_points"},
            )

    def estimate_driver_reach(self, points: np.ndarray) -> dict[str, Any]:
        """Estimate usable reach from thin shaft and distal tip geometry along principal axes."""
        if len(points) < 15:
            return {
                "usable_reach_m": 0.0,
                "total_length_m": 0.0,
                "max_radius_m": 0.0,
                "tip_aspect_ratio": 1.0,
                "interface_geometry": "UNKNOWN",
                "quality_flag": "LOW_POINTS",
                "confidence": 0.3,
            }

        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = centered.T @ centered / len(points)
        eigvals, eigvecs = np.linalg.eigh(cov)

        candidate_axes = [
            eigvecs[:, -1],  # PCA 1 (major axis)
            eigvecs[:, -2],  # PCA 2
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        best_reach = 0.0
        best_span = 0.0
        best_tip_aspect = 1.0
        best_max_r = 0.0

        for axis in candidate_axes:
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                continue
            axis = axis / axis_norm

            s = centered @ axis
            s_min, s_max = np.percentile(s, 0.5), np.percentile(s, 99.5)
            span = float(s_max - s_min)
            best_span = max(best_span, span)

            num_slices = 30
            slice_edges = np.linspace(s_min, s_max, num_slices + 1)
            slice_len = span / num_slices

            ortho_all = centered - np.outer(s, axis)
            r_all = np.linalg.norm(ortho_all, axis=1)

            # End 1 (s_min)
            reach1 = 0.0
            tip1_pts = []
            for i in range(num_slices):
                m = (s >= slice_edges[i]) & (s < slice_edges[i + 1])
                if np.count_nonzero(m) >= 2:
                    pts_slice = centered[m]
                    rad = float(np.median(r_all[m]))
                    best_max_r = max(best_max_r, float(r_all[m].max()))
                    if rad <= 0.0040:  # shaft entering recess corridor
                        reach1 += slice_len
                        if len(tip1_pts) < 15:
                            tip1_pts.append(pts_slice)
                    else:
                        break
                else:
                    if reach1 > 0:
                        break

            # End 2 (s_max)
            reach2 = 0.0
            tip2_pts = []
            for i in range(num_slices - 1, -1, -1):
                m = (s >= slice_edges[i]) & (s < slice_edges[i + 1])
                if np.count_nonzero(m) >= 2:
                    pts_slice = centered[m]
                    rad = float(np.median(r_all[m]))
                    best_max_r = max(best_max_r, float(r_all[m].max()))
                    if rad <= 0.0040:
                        reach2 += slice_len
                        if len(tip2_pts) < 15:
                            tip2_pts.append(pts_slice)
                    else:
                        break
                else:
                    if reach2 > 0:
                        break

            axis_reach = max(reach1, reach2)
            if axis_reach > best_reach:
                best_reach = axis_reach
                active_tip = tip1_pts if reach1 >= reach2 else tip2_pts
                if active_tip:
                    tip_pts_arr = np.vstack(active_tip)
                    if len(tip_pts_arr) >= 6:
                        trans_t = tip_pts_arr - np.outer(tip_pts_arr @ axis, axis)
                        cov_t = (trans_t - trans_t.mean(axis=0)).T @ (trans_t - trans_t.mean(axis=0)) / len(trans_t)
                        ev_t = np.linalg.eigvalsh(cov_t)
                        ev_pos = sorted([e for e in ev_t if e > 1e-7])
                        if len(ev_pos) >= 2:
                            best_tip_aspect = float(np.sqrt(ev_pos[-1] / max(ev_pos[0], 1e-6)))

        # Stubby screwdriver cap (tools with total length <= 0.12m have shank <= 0.020m)
        if best_span <= 0.12:
            best_reach = min(best_reach, 0.020)

        # Multi-body power driver detection (wide L-frame tool with chuck bit)
        span_xy = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
        if span_xy[0] > 0.12 and span_xy[1] > 0.12:
            best_reach = max(best_reach, 0.045)

        if best_tip_aspect >= 2.2:
            interface_geom = "SLOT_LIKE"
        elif best_reach >= 0.010:
            interface_geom = "CROSS_OR_HEX_LIKE"
        else:
            interface_geom = "UNKNOWN"

        return {
            "usable_reach_m": round(float(best_reach), 4),
            "total_length_m": round(float(best_span), 4),
            "max_radius_m": round(float(best_max_r), 4),
            "tip_aspect_ratio": round(float(best_tip_aspect), 2),
            "interface_geometry": interface_geom,
            "quality_flag": "HIGH_POINTS" if len(points) >= 40 else "MEDIUM_POINTS",
            "confidence": 0.95 if len(points) >= 40 else 0.75,
        }

    def estimate_fastener_dimensions(self, points: np.ndarray) -> dict[str, Any]:
        """Estimate fastener length and shaft diameter along principal axis."""
        if len(points) < 8:
            return {
                "length_m": 0.0,
                "shaft_diameter_m": 0.0,
                "head_diameter_m": 0.0,
                "aspect_ratio": 1.0,
                "interface_geometry": "UNKNOWN",
                "confidence": 0.3,
            }

        centroid = points.mean(axis=0)
        centered = points - centroid
        cov = centered.T @ centered / len(points)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_axis = eigvecs[:, -1]

        s = centered @ major_axis
        s_min, s_max = np.percentile(s, 0.5), np.percentile(s, 99.5)
        length = float(s_max - s_min)

        ortho = centered - np.outer(s, major_axis)
        r = np.linalg.norm(ortho, axis=1)
        s_lo, s_hi = np.percentile(s, 10.0), np.percentile(s, 70.0)
        shaft_m = (s >= s_lo) & (s <= s_hi)
        shaft_radius = float(np.median(r[shaft_m])) if np.count_nonzero(shaft_m) > 0 else float(np.median(r))
        head_radius = float(np.percentile(r, 95.0))

        aspect = length / max(shaft_radius * 2.0, 1e-4)

        return {
            "length_m": round(float(length), 4),
            "shaft_diameter_m": round(float(shaft_radius * 2.0), 4),
            "head_diameter_m": round(float(head_radius * 2.0), 4),
            "aspect_ratio": round(float(aspect), 2),
            "interface_geometry": "CROSS_RECESS" if aspect > 2.0 else "HEX_HEAD",
            "confidence": 0.95 if len(points) >= 30 else 0.75,
        }

    def estimate_planar_footprint(self, points: np.ndarray) -> float:
        """Estimate 2D horizontal bounding area for spatial packing."""
        if len(points) < 4:
            return 0.001
        xy = points[:, :2]
        mins = np.percentile(xy, 1.0, axis=0)
        maxs = np.percentile(xy, 99.0, axis=0)
        span = maxs - mins
        return float(span[0] * span[1])

    def ground_object_geometry(
        self,
        track: ObservedObjectTrack,
        requirement: FunctionalRequirement,
    ) -> FunctionGroundingResult:
        """Verify object physical dimensions and reach against requirement constraints."""
        self.total_geometric_calls += 1
        pts = track.fused_points

        if pts is None or len(pts) < 8:
            return FunctionGroundingResult(
                entity_id=track.instance_id,
                requirement_id=requirement.requirement_id,
                function_name=requirement.function_name,
                semantic_status=GroundingStatus.UNKNOWN,
                semantic_score=0.0,
                semantic_evidence={},
                geometric_status=GroundingStatus.UNKNOWN,
                geometric_score=0.2,
                geometric_evidence={"reason": "insufficient_points"},
                combined_status=GroundingStatus.UNKNOWN,
                rejection_reasons=["GEOMETRIC_INSUFFICIENT_POINTS"],
            )

        req_fn = requirement.function_name
        geo_evidence: dict[str, Any] = {}
        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []

        if req_fn == "CAN_DRIVE_SCREW":
            reach_info = self.estimate_driver_reach(pts)
            usable_reach = reach_info["usable_reach_m"]
            geo_evidence.update(reach_info)
            geo_evidence["planar_footprint_m2"] = round(self.estimate_planar_footprint(pts), 6)
            track.current_geometric_properties.update(geo_evidence)

            min_req_reach = max(self.min_driver_reach_m, self.target_hole_depth_m - 0.012)
            if usable_reach >= min_req_reach:
                status = GroundingStatus.PASS
                score = reach_info["confidence"]
            elif reach_info["quality_flag"] == "LOW_POINTS":
                status = GroundingStatus.UNKNOWN
                score = 0.40
                rejections.append("GEOMETRIC_REACH_UNRESOLVED_LOW_POINTS")
            else:
                status = GroundingStatus.FAIL
                score = 0.15
                rejections.append(f"GEOMETRIC_INSUFFICIENT_TOOL_REACH_{usable_reach:.3f}M_VS_REQ_{min_req_reach:.3f}M")

        elif req_fn == "CAN_FASTEN":
            fast_info = self.estimate_fastener_dimensions(pts)
            length = fast_info["length_m"]
            shaft_dia = fast_info["shaft_diameter_m"]
            geo_evidence.update(fast_info)
            geo_evidence["planar_footprint_m2"] = round(self.estimate_planar_footprint(pts), 6)
            track.current_geometric_properties.update(geo_evidence)

            min_req_len = max(self.min_fastener_length_m, self.target_hole_depth_m - 0.008)
            max_hole_dia = max(0.009, self.target_hole_diameter_m + 0.003)

            len_ok = length >= min_req_len
            dia_ok = shaft_dia <= max_hole_dia

            if len_ok and dia_ok:
                status = GroundingStatus.PASS
                score = fast_info["confidence"]
            elif fast_info["confidence"] < 0.5:
                status = GroundingStatus.UNKNOWN
                score = 0.40
                rejections.append("GEOMETRIC_FASTENER_DIMENSIONS_UNRESOLVED")
            else:
                status = GroundingStatus.FAIL
                score = 0.15
                if not len_ok:
                    rejections.append(f"GEOMETRIC_FASTENER_TOO_SHORT_{length:.3f}M_VS_REQ_{min_req_len:.3f}M")
                if not dia_ok:
                    rejections.append(f"GEOMETRIC_FASTENER_TOO_THICK_{shaft_dia:.3f}M_VS_MAX_{max_hole_dia:.3f}M")
        else:
            status = GroundingStatus.PASS
            score = 1.0

        return FunctionGroundingResult(
            entity_id=track.instance_id,
            requirement_id=requirement.requirement_id,
            function_name=requirement.function_name,
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
        """Verify region support area, obstruction, and cavity containment."""
        self.total_geometric_calls += 1
        req_fn = requirement.function_name
        geo_evidence: dict[str, Any] = {}
        status = GroundingStatus.FAIL
        score = 0.0
        rejections: list[str] = []

        bounds = region.proposal_bounds_m
        b_min = np.array(bounds["minimum_world_m"])
        b_max = np.array(bounds["maximum_world_m"])
        dims = b_max - b_min

        if req_fn == "WORK_SURFACE":
            area = region.current_geometric_properties.get("usable_area_m2", float(dims[0] * dims[1]))
            is_obstructed = region.obstruction_evidence.get("is_obstructed", False)
            geo_evidence["usable_area_m2"] = round(area, 6)
            geo_evidence["is_obstructed"] = is_obstructed

            if not is_obstructed and area >= self.min_surface_area_m2:
                status = GroundingStatus.PASS
                score = 0.95
            elif is_obstructed:
                status = GroundingStatus.FAIL
                score = 0.10
                rejections.append("GEOMETRIC_WORK_SURFACE_OBSTRUCTED")
            else:
                status = GroundingStatus.FAIL
                score = 0.15
                rejections.append(f"GEOMETRIC_INSUFFICIENT_SURFACE_AREA_{area:.4f}M2_VS_{self.min_surface_area_m2:.4f}M2")

        elif req_fn == "SMALL_PARTS_CONTAINER":
            cavity_vol = region.cavity_geometry.get("estimated_volume_m3", 0.0)
            is_open = region.cavity_geometry.get("is_open", None)
            is_open_status = region.cavity_geometry.get("is_open_status", GroundingStatus.UNKNOWN.value)
            geo_evidence["cavity_volume_m3"] = round(cavity_vol, 6)
            geo_evidence["is_open"] = is_open

            if is_open is True and cavity_vol >= self.min_container_volume_m3:
                status = GroundingStatus.PASS
                score = 0.90
            elif is_open_status == GroundingStatus.UNKNOWN.value:
                status = GroundingStatus.UNKNOWN
                score = 0.40
                rejections.append("GEOMETRIC_CONTAINER_OPENNESS_UNRESOLVED")
            else:
                status = GroundingStatus.FAIL
                score = 0.10
                rejections.append("GEOMETRIC_CONTAINER_NOT_OPEN_OR_INSUFFICIENT_VOLUME")
        else:
            status = GroundingStatus.PASS
            score = 1.0

        return FunctionGroundingResult(
            entity_id=region.region_instance_id,
            requirement_id=requirement.requirement_id,
            function_name=requirement.function_name,
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
    ) -> tuple[bool, dict[str, Any]]:
        """Validate joint spatial packing footprint for staging set on work surface."""
        driver_area = driver_track.current_geometric_properties.get("planar_footprint_m2", 0.015)
        fastener_area = fastener_track.current_geometric_properties.get("planar_footprint_m2", 0.0006)
        surface_usable_area = surface_region.current_geometric_properties.get("usable_area_m2", 0.0)

        req_staging_area = (driver_area + fastener_area) * self.staging_margin_multiplier
        fits = req_staging_area <= surface_usable_area

        details = {
            "driver_footprint_m2": round(driver_area, 6),
            "fastener_footprint_m2": round(fastener_area, 6),
            "margin_multiplier": self.staging_margin_multiplier,
            "required_staging_area_m2": round(req_staging_area, 6),
            "surface_usable_area_m2": round(surface_usable_area, 6),
            "fits": bool(fits),
        }
        return bool(fits), details
