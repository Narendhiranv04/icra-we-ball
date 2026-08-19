"""Candidate region discovery and observable property extraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
    voxel_downsample,
)
from mujoco_scenes.workshop_phase1.types import GroundingStatus, ObservedRegion, ViewObservation


class RegionGrounder:
    """Discovers candidate regions, extracts observable surface/cavity geometry, and detects obstructions."""

    def __init__(self) -> None:
        self._region_id_counter: int = 1
        self._known_regions: dict[str, ObservedRegion] = {}

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
            crop_evidence = {}

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

                # Approximate 2D region ROI crop
                # Project 8 corner points of bounding box to camera image
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
                cam_coords = (corners - cam_p) @ cam_r
                valid_front = cam_coords[:, 2] > 0.1
                if np.count_nonzero(valid_front) >= 4:
                    proj = cam_coords[valid_front] @ obs.intrinsics.T
                    uv = proj[:, :2] / proj[:, 2:3]
                    h, w = obs.rgb.shape[:2]
                    u_min = max(0, int(np.min(uv[:, 0])))
                    u_max = min(w, int(np.max(uv[:, 0])) + 1)
                    v_min = max(0, int(np.min(uv[:, 1])))
                    v_max = min(h, int(np.max(uv[:, 1])) + 1)
                    if (u_max - u_min) > 10 and (v_max - v_min) > 10:
                        crop_evidence[obs.camera_id] = obs.rgb[v_min:v_max, u_min:u_max].copy()

            if region_pts_list:
                all_pts = np.vstack(region_pts_list)
                fused_pts, _ = voxel_downsample(all_pts, np.ones_like(all_pts), voxel_size=0.005)
            else:
                fused_pts = np.empty((0, 3), dtype=np.float32)

            dims = b_max - b_min
            support_plane = {
                "base_height_m": round(float(b_min[2]), 4),
                "width_m": round(float(dims[0]), 4),
                "length_m": round(float(dims[1]), 4),
                "planar_area_m2": round(float(dims[0] * dims[1]), 4),
            }

            # Obstruction check: look for dense clusters in the upper volume (z > base + 0.035m)
            obstruction_evidence = {"is_obstructed": False, "obstruction_ratio": 0.0}
            if len(fused_pts) > 50:
                elevated_pts = fused_pts[fused_pts[:, 2] > (b_min[2] + 0.035)]
                if len(elevated_pts) > 150:
                    xy_span = elevated_pts.max(axis=0)[:2] - elevated_pts.min(axis=0)[:2]
                    obs_area = float(xy_span[0] * xy_span[1])
                    total_area = float(dims[0] * dims[1])
                    ratio = min(1.0, obs_area / max(total_area, 1e-4))
                    if ratio > 0.35:
                        obstruction_evidence = {
                            "is_obstructed": True,
                            "obstruction_ratio": round(ratio, 4),
                            "elevated_point_count": len(elevated_pts),
                        }

            # Cavity depth and openness check for parts containers (observed from RGB-D)
            is_open = None
            is_open_status = GroundingStatus.UNKNOWN
            cavity_depth_m = 0.0
            cavity_volume_m3 = 0.0

            if len(fused_pts) >= 30:
                z_pts = fused_pts[:, 2]
                z_rim = float(np.percentile(z_pts, 95))
                z_floor = float(np.percentile(z_pts, 10))
                depth_diff = z_rim - z_floor

                if depth_diff >= 0.015 and dims[2] >= 0.020:
                    # Verified recessed cavity
                    cavity_depth_m = round(depth_diff, 4)
                    cavity_volume_m3 = round(float(dims[0] * dims[1] * cavity_depth_m), 6)
                    is_open = True
                    is_open_status = GroundingStatus.PASS
                elif dims[2] < 0.015:
                    # Flat surface, not a cavity
                    is_open = False
                    is_open_status = GroundingStatus.FAIL
                else:
                    is_open = True
                    is_open_status = GroundingStatus.PASS
                    cavity_depth_m = round(float(dims[2] * 0.7), 4)
                    cavity_volume_m3 = round(float(dims[0] * dims[1] * cavity_depth_m), 6)
            else:
                # Fallback based on geometric proposal dimensions with UNKNOWN status
                if dims[2] >= 0.025:
                    cavity_depth_m = round(float(dims[2] * 0.7), 4)
                    cavity_volume_m3 = round(float(dims[0] * dims[1] * cavity_depth_m), 6)
                    is_open = True
                    is_open_status = GroundingStatus.UNKNOWN

            cavity_geometry = {
                "is_open": is_open,
                "is_open_status": is_open_status.value,
                "observed_cavity_depth_m": cavity_depth_m,
                "estimated_volume_m3": cavity_volume_m3,
            }

            observed_reg = ObservedRegion(
                region_instance_id=reg_id,
                proposal_bounds_m=bounds,
                observation_source="calibrated_spatial_proposal",
                fused_points=fused_pts,
                crop_evidence=crop_evidence,
                support_plane=support_plane,
                cavity_geometry=cavity_geometry,
                obstruction_evidence=obstruction_evidence,
                is_open=is_open,
                is_open_status=is_open_status,
            )
            observed_regions.append(observed_reg)
            self._known_regions[reg_id] = observed_reg

        return observed_regions
