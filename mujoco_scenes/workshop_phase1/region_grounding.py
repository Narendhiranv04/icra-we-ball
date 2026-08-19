"""Candidate region discovery, YOLO region semantic association, and observable property extraction."""

from __future__ import annotations

from typing import Any
from collections import defaultdict

import numpy as np

from mujoco_scenes.geometry_checker import (
    MeasurementEvidence,
    backproject_masked_depth,
    gate_points_to_volume,
    voxel_downsample,
)
from mujoco_scenes.geometry_properties import extract_object_properties
from mujoco_scenes.workshop_phase1.geometric_grounding import DEFAULT_GEOMETRY_CONFIG
import yaml
from mujoco_scenes.workshop_phase1.types import GroundingStatus, ObservedRegion, ViewObservation


class RegionGrounder:
    """Discovers candidate regions, associates YOLO region detections, and extracts observable geometry."""

    def __init__(self, region_categories: set[str] | None = None,
                 geometry_config_path: str | None = None) -> None:
        self._region_id_counter: int = 1
        self._known_regions: dict[str, ObservedRegion] = {}
        self.region_categories = {str(v).lower() for v in (region_categories or set())}
        path = geometry_config_path or str(DEFAULT_GEOMETRY_CONFIG)
        with open(path, encoding="utf-8") as source:
            self.geometry_config = yaml.safe_load(source) or {}

    def _measure_support_plane(self, points: np.ndarray) -> dict[str, Any]:
        """Measure the dominant horizontal support from points, never ROI extents."""
        cfg = self.geometry_config.get("support_plane", {})
        points = points[np.all(np.isfinite(points), axis=1)]
        minimum = int(cfg.get("minimum_points", 40))
        unknown = {"relation": "PLANAR_SUPPORT", "status": "UNKNOWN",
                   "reason": "INSUFFICIENT_REGION_POINTS", "point_count": len(points)}
        if len(points) < minimum:
            return {"support_length_m": None, "support_width_m": None,
                    "support_area_m2": None, "support_plane_z_m": None,
                    "predicate": unknown, "plane_points": np.empty((0, 3))}
        bin_width = float(cfg.get("histogram_bin_m", 0.003))
        z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
        bins = max(2, int(np.ceil((z_max - z_min) / max(bin_width, 1e-5))))
        counts, edges = np.histogram(points[:, 2], bins=bins, range=(z_min, z_max + 1e-9))
        index = int(np.argmax(counts))
        plane_z = float(0.5 * (edges[index] + edges[index + 1]))
        tolerance = float(cfg.get("inlier_tolerance_m", 0.0045))
        plane = points[np.abs(points[:, 2] - plane_z) <= tolerance]
        if len(plane) < minimum:
            unknown.update({"reason": "SUPPORT_PLANE_UNRESOLVED", "plane_point_count": len(plane)})
            return {"support_length_m": None, "support_width_m": None,
                    "support_area_m2": None, "support_plane_z_m": plane_z,
                    "predicate": unknown, "plane_points": plane}
        centred = plane - np.median(plane, axis=0)
        values, vectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        normal = vectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        alignment = float(abs(normal[2]))
        planarity = float(1.0 - min(1.0, values[0] / max(values[1], 1e-12)))
        xy = plane[:, :2] - np.median(plane[:, :2], axis=0)
        _, axes = np.linalg.eigh(np.cov(xy, rowvar=False))
        projected = xy @ axes
        low = float(cfg.get("robust_lower_percentile", 2.0))
        high = float(cfg.get("robust_upper_percentile", 98.0))
        extents = np.sort(np.percentile(projected, high, axis=0) - np.percentile(projected, low, axis=0))[::-1]
        thickness = float(np.percentile(plane[:, 2], 98) - np.percentile(plane[:, 2], 2))
        passed = (planarity >= float(cfg.get("minimum_planarity_score", 0.70))
                  and alignment >= float(cfg.get("minimum_upward_alignment", 0.85))
                  and thickness <= float(cfg.get("maximum_thickness_m", 0.012)))
        predicate = {"relation": "PLANAR_SUPPORT", "status": "TRUE" if passed else "FALSE",
            "method": "dominant_height_plane_observed_footprint_v2", "plane_point_count": len(plane),
            "planarity_score": planarity, "upward_alignment": alignment,
            "support_thickness_m": thickness, "configured_roi_used_as_measurement": False}
        return {"support_length_m": float(extents[0]), "support_width_m": float(extents[1]),
                "support_area_m2": float(np.prod(extents)), "support_plane_z_m": plane_z,
                "support_normal_world": normal.tolist(), "predicate": predicate, "plane_points": plane}

    @staticmethod
    def _compute_consensus_region_semantic(observations: list[dict[str, Any]]) -> dict[str, Any]:
        if not observations:
            return {"canonical_label": "unknown", "raw_label": "unknown", "confidence": 0.0}

        score_by_label: dict[str, float] = defaultdict(float)
        raw_by_label: dict[str, str] = {}
        for obs in observations:
            label = obs.get("canonical_label", "unknown").lower()
            conf = float(obs.get("confidence", 1.0))
            score_by_label[label] += conf
            if label not in raw_by_label:
                raw_by_label[label] = obs.get("raw_label", label)

        best_label = max(score_by_label, key=score_by_label.get)
        total_score = sum(score_by_label.values())
        norm_conf = min(0.99, score_by_label[best_label] / max(1.0, total_score) * min(1.0, 0.5 + 0.25 * len(observations)))

        return {
            "canonical_label": best_label,
            "raw_label": raw_by_label.get(best_label, best_label),
            "confidence": round(norm_conf, 4),
            "total_observations": len(observations),
        }

    def discover_candidate_regions(
        self,
        scene: Any,
        observations: list[ViewObservation],
    ) -> list[ObservedRegion]:
        """Convert neutral spatial proposals to ObservedRegion instances with visual/depth evidence."""
        proposals = scene.get_candidate_regions()
        observed_regions: list[ObservedRegion] = []

        for prop in proposals:
            bounds = prop["proposal_bounds_m"]
            b_min = np.array(bounds["minimum_world_m"], dtype=np.float64)
            b_max = np.array(bounds["maximum_world_m"], dtype=np.float64)

            reg_id = f"region_{self._region_id_counter:04d}"
            self._region_id_counter += 1

            # Extract point clouds and crops from observations
            region_pts_list = []
            points_by_camera: dict[str, np.ndarray] = {}
            crop_evidence = {}
            semantic_obs = []

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
                        minimum_world_m=b_min,
                        maximum_world_m=b_max,
                        boundary_margin_m=0.02,
                    )
                    if np.count_nonzero(gated) > 0:
                        selected = pts[gated]
                        region_pts_list.append(selected)
                        points_by_camera[obs.camera_id] = selected

                # 2D region ROI projection to camera image
                corners = np.array([
                    [b_min[0], b_min[1], b_min[2]],
                    [b_min[0], b_min[1], b_max[2]],
                    [b_min[0], b_max[1], b_min[2]],
                    [b_min[0], b_max[1], b_max[2]],
                    [b_max[0], b_min[1], b_min[2]],
                    [b_max[0], b_min[1], b_max[2]],
                    [b_max[0], b_max[1], b_min[2]],
                    [b_max[0], b_max[1], b_max[2]],
                ])
                cam_p = obs.camera_position_world
                cam_r = obs.camera_rotation_world
                local = (corners - cam_p) @ cam_r
                z_depth = -local[:, 2]
                valid = z_depth > 0.1
                if np.count_nonzero(valid) >= 4:
                    fx, fy = obs.intrinsics[0, 0], obs.intrinsics[1, 1]
                    cx, cy = obs.intrinsics[0, 2], obs.intrinsics[1, 2]
                    u = local[valid, 0] * fx / z_depth[valid] + cx
                    v = -local[valid, 1] * fy / z_depth[valid] + cy
                    h, w = obs.rgb.shape[:2]
                    u_min = max(0, min(w - 1, int(np.min(u))))
                    u_max = min(w, max(0, int(np.max(u)) + 1))
                    v_min = max(0, min(h - 1, int(np.min(v))))
                    v_max = min(h, max(0, int(np.max(v)) + 1))
                    if (u_max - u_min) > 10 and (v_max - v_min) > 10:
                        crop_evidence[obs.camera_id] = obs.rgb[v_min:v_max, u_min:u_max].copy()

                        # Associate YOLO region detections overlapping this projected ROI
                        roi_box = (u_min, v_min, u_max, v_max)
                        semantic_detections = obs.region_semantic_detections or obs.detected_masks
                        for d_mask in semantic_detections:
                            if d_mask.canonical_label.lower() in self.region_categories:
                                bx1, by1, bx2, by2 = d_mask.bounding_box_xyxy
                                # Compute box overlap / intersection
                                ix1, iy1 = max(u_min, bx1), max(v_min, by1)
                                ix2, iy2 = min(u_max, bx2), min(v_max, by2)
                                if ix2 > ix1 and iy2 > iy1:
                                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                                    box_area = (bx2 - bx1) * (by2 - by1)
                                    if inter_area / max(1, box_area) > 0.15 or inter_area / max(1, (u_max - u_min) * (v_max - v_min)) > 0.15:
                                        semantic_obs.append({
                                            "camera_id": obs.camera_id,
                                            "canonical_label": d_mask.canonical_label,
                                            "raw_label": d_mask.raw_label,
                                            "confidence": d_mask.confidence,
                                        })

            if region_pts_list:
                all_pts = np.vstack(region_pts_list)
                fused_pts, _ = voxel_downsample(all_pts, np.ones_like(all_pts), voxel_size=0.005)
            else:
                fused_pts = np.empty((0, 3), dtype=np.float32)

            support = self._measure_support_plane(fused_pts)
            plane_z = support["support_plane_z_m"]
            clearance = float(self.geometry_config.get("support_plane", {}).get("obstruction_clearance_m", 0.012))
            elevated = fused_pts[fused_pts[:, 2] > plane_z + clearance] if plane_z is not None else np.empty((0, 3))
            obstruction = {"is_obstructed": len(elevated) > 80, "elevated_point_count": len(elevated),
                           "reference_support_plane_z_m": plane_z, "support_plane_points_excluded": True}
            evidence = MeasurementEvidence(instance_name=reg_id, measurement_points=fused_pts,
                measurement_colors=np.zeros_like(fused_pts), contributing_camera_ids=tuple(points_by_camera),
                points_by_camera=points_by_camera, source_stage=0, source_region=prop.get("region_instance_id", "CANDIDATE_REGION"),
                measurement_cloud_path=None, measurement_quality={"quality_is_valid": len(fused_pts) >= 30,
                    "raw_inside_point_count": len(fused_pts), "outlier_points_removed": 0})
            generic = extract_object_properties(evidence, config=self.geometry_config)
            cavity_pred = generic["geometric_predicates"]["OPEN_CAVITY"]
            cavity_props = generic["geometric_properties"]
            def value(key: str) -> float | None:
                return cavity_props.get(key, {}).get("value")
            cavity_depth_m, opening_w, opening_l = value("cavity_depth_m"), value("opening_width_m"), value("opening_length_m")
            is_open = True if cavity_pred["status"] == "TRUE" else False if cavity_pred["status"] == "FALSE" else None
            is_open_status = {"TRUE": GroundingStatus.PASS, "FALSE": GroundingStatus.FAIL}.get(cavity_pred["status"], GroundingStatus.UNKNOWN)
            cavity_geometry = {"is_open": is_open, "is_open_status": is_open_status.value,
                "estimated_depth_m": cavity_depth_m, "opening_width_m": opening_w,
                "opening_length_m": opening_l, "point_count": len(fused_pts), "predicate": cavity_pred}
            support_plane = {"center_world_m": np.median(support["plane_points"], axis=0).tolist() if len(support["plane_points"]) else None,
                "dimensions_m": [support["support_length_m"], support["support_width_m"]],
                "area_m2": support["support_area_m2"], "point_count": len(support["plane_points"]),
                "method": "dominant_height_plane_observed_footprint_v2"}

            sem_belief = self._compute_consensus_region_semantic(semantic_obs)

            region_entry = ObservedRegion(
                region_instance_id=reg_id,
                proposal_bounds_m=bounds,
                observation_source="calibrated_spatial_proposal",
                fused_points=fused_pts,
                crop_evidence=crop_evidence,
                support_plane=support_plane,
                cavity_geometry=cavity_geometry,
                obstruction_evidence=obstruction,
                is_open=is_open,
                is_open_status=is_open_status,
                semantic_observations=semantic_obs,
                current_semantic_belief=sem_belief,
                current_geometric_properties={
                    "support_length_m": support["support_length_m"],
                    "support_width_m": support["support_width_m"],
                    "support_area_m2": support["support_area_m2"],
                    "usable_area_m2": support["support_area_m2"],
                    "cavity_depth_m": cavity_depth_m,
                    "opening_width_m": opening_w,
                    "opening_length_m": opening_l,
                    "is_open": is_open,
                    "is_open_status": is_open_status.value,
                    "geometric_predicates": {"PLANAR_SUPPORT": support["predicate"], "OPEN_CAVITY": cavity_pred},
                    "measurement_provenance": {"camera_ids": list(points_by_camera), "point_count": len(fused_pts),
                        "selection_volume_is_measurement": False, "method": "stage_local_region_gated_rgbd"},
                },
            )
            observed_regions.append(region_entry)

        return observed_regions
