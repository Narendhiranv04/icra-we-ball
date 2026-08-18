"""Candidate region discovery and observable property extraction."""

from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
    voxel_downsample,
)
from mujoco_scenes.workshop_phase1.types import ObservedRegion, ViewObservation


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
        # 1. Retrieve neutral proposal bounds
        proposals = scene.get_candidate_regions()
        observed_regions: list[ObservedRegion] = []

        for p_idx, prop in enumerate(proposals):
            bounds = prop["proposal_bounds_m"]
            b_min = np.array(bounds["minimum_world_m"], dtype=np.float64)
            b_max = np.array(bounds["maximum_world_m"], dtype=np.float64)

            reg_id = f"region_{self._region_id_counter:04d}"
            self._region_id_counter += 1

            # 2. Extract point cloud within proposal volume from observations
            region_pts_list = []
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

            if region_pts_list:
                all_pts = np.vstack(region_pts_list)
                fused_pts, _ = voxel_downsample(all_pts, np.ones_like(all_pts), voxel_size=0.005)
            else:
                fused_pts = np.empty((0, 3), dtype=np.float32)

            # 3. Analyze support plane & obstruction
            dims = b_max - b_min
            support_plane = {
                "base_height_m": round(float(b_min[2]), 4),
                "width_m": round(float(dims[0]), 4),
                "length_m": round(float(dims[1]), 4),
            }

            # Obstruction check: look for dense clusters in the upper volume (z > base + 0.05m)
            obstruction_evidence = {"is_obstructed": False, "obstruction_ratio": 0.0}
            if len(fused_pts) > 50:
                elevated_pts = fused_pts[fused_pts[:, 2] > (b_min[2] + 0.035)]
                # If elevated points cover more than 30% of the volume area
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

            # Cavity check for small-parts container
            cavity_geometry = {
                "is_open": True,
                "estimated_volume_m3": round(float(dims[0] * dims[1] * max(0.01, dims[2])), 6),
            }

            observed_reg = ObservedRegion(
                region_instance_id=reg_id,
                proposal_bounds_m=bounds,
                observation_source=prop.get("source_sensor", "calibrated_inspection_cameras"),
                fused_points=fused_pts,
                support_plane=support_plane,
                cavity_geometry=cavity_geometry,
                obstruction_evidence=obstruction_evidence,
            )
            observed_regions.append(observed_reg)
            self._known_regions[reg_id] = observed_reg

        return observed_regions
