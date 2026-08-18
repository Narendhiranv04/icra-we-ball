"""Multi-view fusion and persistent cross-stage instance tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
    remove_sparse_voxel_outliers,
    voxel_downsample,
)
from mujoco_scenes.workshop_phase1.types import (
    ObservedMask,
    ObservedObjectTrack,
    ViewObservation,
)


class PersistentInstanceTracker:
    """Associates 2D detections across 5 camera views and across sequential inspection stages."""

    def __init__(
        self,
        cluster_distance_threshold_m: float = 0.025,
        track_match_distance_threshold_m: float = 0.035,
        voxel_size_m: float = 0.003,
        min_cluster_points: int = 15,
    ) -> None:
        self.cluster_distance_threshold_m = cluster_distance_threshold_m
        self.track_match_distance_threshold_m = track_match_distance_threshold_m
        self.voxel_size_m = voxel_size_m
        self.min_cluster_points = min_cluster_points

        self._tracks: dict[str, ObservedObjectTrack] = {}
        self._next_instance_idx: int = 1

    @property
    def tracks(self) -> dict[str, ObservedObjectTrack]:
        return self._tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_instance_idx = 1

    def _allocate_instance_id(self) -> str:
        inst_id = f"object_{self._next_instance_idx:04d}"
        self._next_instance_idx += 1
        return inst_id

    def update_with_stage_observations(
        self,
        stage_index: int,
        source_region_id: str,
        observations: list[ViewObservation],
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
    ) -> list[ObservedObjectTrack]:
        """Fuse current stage detections across 5 views and associate with persistent tracks."""
        # 1. Backproject each detection to 3D point cloud
        detections_3d: list[dict[str, Any]] = []

        for obs in observations:
            cam_id = obs.camera_id
            for mask in obs.detected_masks:
                pts, pixel_indices = backproject_masked_depth(
                    obs.depth_m,
                    mask.binary_mask,
                    obs.intrinsics,
                    obs.camera_position_world,
                    obs.camera_rotation_world,
                    max_depth=3.0,
                )
                if len(pts) < 8:
                    continue

                gated = gate_points_to_volume(
                    pts,
                    minimum_world_m=stage_volume_min,
                    maximum_world_m=stage_volume_max,
                    boundary_margin_m=0.08,
                )
                pts = pts[gated]
                pixel_indices = pixel_indices[gated]
                if len(pts) < 8:
                    continue

                # Sample colors
                colors = obs.rgb[pixel_indices[:, 0], pixel_indices[:, 1]].astype(np.float32) / 255.0

                centroid = pts.mean(axis=0)
                detections_3d.append({
                    "mask": mask,
                    "camera_id": cam_id,
                    "points": pts,
                    "colors": colors,
                    "centroid": centroid,
                    "obs": obs,
                })

        # 2. Cluster detections across the 5 views of this stage
        clusters: list[list[dict[str, Any]]] = []
        for det in detections_3d:
            matched_cluster_idx = None
            min_dist = np.inf
            for c_idx, cluster in enumerate(clusters):
                # Distance to cluster centroid
                c_pts = np.vstack([d["points"] for d in cluster])
                c_centroid = c_pts.mean(axis=0)
                dist = float(np.linalg.norm(det["centroid"] - c_centroid))
                if dist < self.cluster_distance_threshold_m and dist < min_dist:
                    min_dist = dist
                    matched_cluster_idx = c_idx

            if matched_cluster_idx is not None:
                clusters[matched_cluster_idx].append(det)
            else:
                clusters.append([det])

        # 3. Fuse points for each cluster
        stage_objects: list[dict[str, Any]] = []
        for cluster in clusters:
            all_pts = np.vstack([d["points"] for d in cluster])
            all_colors = np.vstack([d["colors"] for d in cluster])
            if len(all_pts) < self.min_cluster_points:
                continue

            # Downsample and outlier removal
            fused_pts, fused_colors = voxel_downsample(all_pts, all_colors, voxel_size=self.voxel_size_m)
            fused_pts, fused_colors, _ = remove_sparse_voxel_outliers(
                fused_pts,
                fused_colors,
                voxel_radius_m=self.voxel_size_m * 2.5,
                minimum_neighbours=3,
                minimum_input_points=10,
            )
            if len(fused_pts) < self.min_cluster_points:
                continue

            cameras = tuple(sorted(set(d["camera_id"] for d in cluster)))
            pts_by_cam = {}
            for d in cluster:
                pts_by_cam[d["camera_id"]] = d["points"]

            # Predicted label consensus
            labels = [d["mask"].predicted_label for d in cluster]
            consensus_label = max(set(labels), key=labels.count)

            stage_objects.append({
                "points": fused_pts,
                "colors": fused_colors,
                "centroid": fused_pts.mean(axis=0),
                "cameras": cameras,
                "points_by_camera": pts_by_cam,
                "label": consensus_label,
                "cluster": cluster,
            })

        # 4. Associate stage objects with existing persistent tracks
        affected_tracks: list[ObservedObjectTrack] = []

        for st_obj in stage_objects:
            obj_centroid = st_obj["centroid"]
            best_track_id = None
            min_track_dist = np.inf

            for t_id, track in self._tracks.items():
                if track.fused_points is None or len(track.fused_points) == 0:
                    continue
                t_centroid = track.fused_points.mean(axis=0)
                dist = float(np.linalg.norm(obj_centroid - t_centroid))
                if dist < self.track_match_distance_threshold_m and dist < min_track_dist:
                    min_track_dist = dist
                    best_track_id = t_id

            if best_track_id is not None:
                # Update existing track
                track = self._tracks[best_track_id]
                combined_pts = np.vstack([track.fused_points, st_obj["points"]])
                combined_colors = np.vstack([track.fused_colors, st_obj["colors"]])
                fused_pts, fused_colors = voxel_downsample(combined_pts, combined_colors, voxel_size=self.voxel_size_m)

                track.fused_points = fused_pts
                track.fused_colors = fused_colors
                track.last_seen_stage = stage_index
                track.evidence_count += 1
                track.contributing_cameras = tuple(sorted(set(track.contributing_cameras + st_obj["cameras"])))
                track.points_by_camera.update(st_obj["points_by_camera"])
                affected_tracks.append(track)
            else:
                # Create brand new track with generic ID
                new_id = self._allocate_instance_id()
                new_track = ObservedObjectTrack(
                    instance_id=new_id,
                    first_seen_stage=stage_index,
                    last_seen_stage=stage_index,
                    source_inspection_region_id=source_region_id,
                    fused_points=st_obj["points"],
                    fused_colors=st_obj["colors"],
                    points_by_camera=st_obj["points_by_camera"],
                    contributing_cameras=st_obj["cameras"],
                    current_semantic_belief={"initial_label": st_obj["label"]},
                    overall_confidence=0.9,
                    evidence_count=1,
                    status="ACTIVE",
                )
                self._tracks[new_id] = new_track
                affected_tracks.append(new_track)

        return affected_tracks
