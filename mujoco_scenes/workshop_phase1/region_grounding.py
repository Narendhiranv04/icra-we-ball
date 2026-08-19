"""Candidate region discovery, YOLO region semantic association, and observable property extraction."""

from __future__ import annotations

from typing import Any
from collections import defaultdict

import numpy as np

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
    voxel_downsample,
)
from mujoco_scenes.workshop_phase1.types import GroundingStatus, ObservedRegion, ViewObservation


class RegionGrounder:
    """Discovers candidate regions, associates YOLO region detections, and extracts observable geometry."""

    def __init__(self) -> None:
        self._region_id_counter: int = 1
        self._known_regions: dict[str, ObservedRegion] = {}

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

        REGION_CATEGORIES = {"workbench", "tool_cart", "shelf", "parts_tray", "hardware_bin"}

        for prop in proposals:
            bounds = prop["proposal_bounds_m"]
            b_min = np.array(bounds["minimum_world_m"], dtype=np.float64)
            b_max = np.array(bounds["maximum_world_m"], dtype=np.float64)

            reg_id = f"region_{self._region_id_counter:04d}"
            self._region_id_counter += 1

            # Extract point clouds and crops from observations
            region_pts_list = []
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
                        region_pts_list.append(pts[gated])

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
                        for d_mask in obs.detected_masks:
                            if d_mask.canonical_label in REGION_CATEGORIES:
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

            dims = b_max - b_min
            support_plane = {
                "center_world_m": [float(c) for c in (b_min + b_max) / 2],
                "dimensions_m": [float(d) for d in dims],
                "area_m2": round(float(dims[0] * dims[1]), 6),
                "point_count": len(fused_pts),
            }

            # Obstruction check on support surfaces
            obstruction = {"is_obstructed": False, "obstructing_objects": []}
            if len(fused_pts) > 50:
                z_floor = float(b_min[2] + 0.015)
                elevated_pts = fused_pts[fused_pts[:, 2] > z_floor]
                if len(elevated_pts) > 80:
                    ext = elevated_pts.max(axis=0) - elevated_pts.min(axis=0)
                    if float(ext[0] * ext[1]) >= 0.010:
                        obstruction = {
                            "is_obstructed": True,
                            "obstructing_objects": ["unidentified_object_cluster"],
                            "elevated_point_count": len(elevated_pts),
                        }

            # Cavity depth and openness check for parts containers (observed strictly from RGB-D)
            is_open = None
            is_open_status = GroundingStatus.UNKNOWN
            cavity_depth_m = 0.0
            cavity_volume_m3 = 0.0
            observed_footprint_m2 = float(dims[0] * dims[1])

            if len(fused_pts) >= 30:
                z_pts = fused_pts[:, 2]
                z_rim = float(np.percentile(z_pts, 95))
                z_floor = float(np.percentile(z_pts, 10))
                depth_diff = z_rim - z_floor

                if depth_diff >= 0.015:
                    # Verified observed recessed cavity
                    cavity_depth_m = round(depth_diff, 4)
                    cavity_volume_m3 = round(float(observed_footprint_m2 * cavity_depth_m), 6)
                    is_open = True
                    is_open_status = GroundingStatus.PASS
                else:
                    # Flat surface or insufficient depth difference
                    is_open = False
                    is_open_status = GroundingStatus.FAIL
                    cavity_depth_m = round(max(0.0, depth_diff), 4)
                    cavity_volume_m3 = 0.0
            else:
                # Sparse points -> UNKNOWN
                is_open = None
                is_open_status = GroundingStatus.UNKNOWN
                cavity_depth_m = 0.0
                cavity_volume_m3 = 0.0

            cavity_geometry = {
                "is_open": is_open,
                "is_open_status": is_open_status.value,
                "estimated_depth_m": cavity_depth_m,
                "estimated_volume_m3": cavity_volume_m3,
                "point_count": len(fused_pts),
            }

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
                    "usable_area_m2": round(float(dims[0] * dims[1]), 6) if not obstruction["is_obstructed"] else 0.0,
                    "cavity_volume_m3": cavity_volume_m3,
                    "cavity_depth_m": cavity_depth_m,
                    "is_open": is_open,
                    "is_open_status": is_open_status.value,
                },
            )
            observed_regions.append(region_entry)

        return observed_regions
