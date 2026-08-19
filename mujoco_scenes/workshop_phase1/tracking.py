"""Multi-view fusion and persistent cross-stage instance tracking."""

from __future__ import annotations

from typing import Any
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

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
        cluster_distance_threshold_m: float = 0.040,
        track_match_distance_threshold_m: float = 0.045,
        voxel_size_m: float = 0.003,
        min_cluster_points: int = 10,
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

    @staticmethod
    def _compute_consensus_semantic_belief(observations: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate multi-view semantic observations using confidence-weighted voting."""
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

        # Pick label with highest total score
        best_label = max(score_by_label, key=score_by_label.get)
        total_score = sum(score_by_label.values())
        norm_conf = min(0.99, score_by_label[best_label] / max(1.0, total_score) * min(1.0, 0.5 + 0.25 * len(observations)))

        return {
            "canonical_label": best_label,
            "raw_label": raw_by_label.get(best_label, best_label),
            "confidence": round(norm_conf, 4),
            "total_observations": len(observations),
        }

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

                colors = obs.rgb[pixel_indices[:, 0], pixel_indices[:, 1]].astype(np.float32) / 255.0

                x1, y1, x2, y2 = mask.bounding_box_xyxy
                crop = obs.rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()

                centroid = pts.mean(axis=0)
                b_min = pts.min(axis=0)
                b_max = pts.max(axis=0)

                detections_3d.append({
                    "mask": mask,
                    "camera_id": cam_id,
                    "points": pts,
                    "colors": colors,
                    "centroid": centroid,
                    "b_min": b_min,
                    "b_max": b_max,
                    "crop": crop,
                    "obs": obs,
                })

        if not detections_3d:
            return []

        # 2. Cluster detections across the 5 views of this stage (disallowing same-camera duplicate clustering)
        clusters: list[list[dict[str, Any]]] = []
        for det in detections_3d:
            matched_cluster_idx = None
            min_dist = np.inf
            for c_idx, cluster in enumerate(clusters):
                # Disallow merging if this cluster already has a detection from the SAME camera
                cluster_cameras = {d["camera_id"] for d in cluster}
                if det["camera_id"] in cluster_cameras:
                    continue

                c_pts = np.vstack([d["points"] for d in cluster])
                c_centroid = c_pts.mean(axis=0)
                c_min = c_pts.min(axis=0)
                c_max = c_pts.max(axis=0)

                dist = float(np.linalg.norm(det["centroid"] - c_centroid))

                # Check 3D bounding box overlap
                overlap = np.all(det["b_max"] >= c_min - 0.015) and np.all(det["b_min"] <= c_max + 0.015)

                if (dist < self.cluster_distance_threshold_m or overlap) and dist < min_dist:
                    min_dist = dist
                    matched_cluster_idx = c_idx

            if matched_cluster_idx is not None:
                clusters[matched_cluster_idx].append(det)
            else:
                clusters.append([det])

        # 3. Fuse points and semantic observations for each cluster
        stage_objects: list[dict[str, Any]] = []
        for cluster in clusters:
            all_pts = np.vstack([d["points"] for d in cluster])
            all_colors = np.vstack([d["colors"] for d in cluster])
            if len(all_pts) < self.min_cluster_points:
                continue

            fused_pts, fused_colors = voxel_downsample(all_pts, all_colors, voxel_size=self.voxel_size_m)
            fused_pts, fused_colors, _ = remove_sparse_voxel_outliers(
                fused_pts,
                fused_colors,
                voxel_radius_m=self.voxel_size_m * 2.5,
                minimum_neighbours=3,
                minimum_input_points=6,
            )
            if len(fused_pts) < self.min_cluster_points:
                continue

            cameras = tuple(sorted(set(d["camera_id"] for d in cluster)))
            pts_by_cam = {}
            crops_by_cam = {}
            semantic_obs = []
            for d in cluster:
                pts_by_cam[d["camera_id"]] = d["points"]
                if d["crop"] is not None and d["crop"].size > 0:
                    crops_by_cam[d["camera_id"]] = d["crop"]
                semantic_obs.append({
                    "stage_index": stage_index,
                    "camera_id": d["camera_id"],
                    "canonical_label": d["mask"].canonical_label,
                    "raw_label": d["mask"].raw_label,
                    "confidence": d["mask"].confidence,
                })

            stage_objects.append({
                "points": fused_pts,
                "colors": fused_colors,
                "centroid": fused_pts.mean(axis=0),
                "cameras": cameras,
                "points_by_camera": pts_by_cam,
                "crops_by_camera": crops_by_cam,
                "semantic_observations": semantic_obs,
            })

        # 4. Associate stage objects with existing persistent tracks using bipartite matching
        affected_tracks: list[ObservedObjectTrack] = []
        existing_track_ids = [t_id for t_id, t in self._tracks.items() if t.fused_points is not None and len(t.fused_points) > 0]

        if existing_track_ids and stage_objects:
            cost_matrix = np.full((len(stage_objects), len(existing_track_ids)), 1e6, dtype=float)
            for i, st_obj in enumerate(stage_objects):
                for j, t_id in enumerate(existing_track_ids):
                    t = self._tracks[t_id]
                    t_centroid = t.fused_points.mean(axis=0)
                    dist = float(np.linalg.norm(st_obj["centroid"] - t_centroid))
                    if dist < self.track_match_distance_threshold_m:
                        cost_matrix[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_stage_indices = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.track_match_distance_threshold_m:
                    matched_stage_indices.add(r)
                    t_id = existing_track_ids[c]
                    st_obj = stage_objects[r]
                    track = self._tracks[t_id]

                    combined_pts = np.vstack([track.fused_points, st_obj["points"]])
                    combined_colors = np.vstack([track.fused_colors, st_obj["colors"]])
                    fused_pts, fused_colors = voxel_downsample(combined_pts, combined_colors, voxel_size=self.voxel_size_m)

                    track.fused_points = fused_pts
                    track.fused_colors = fused_colors
                    track.last_seen_stage = stage_index
                    track.evidence_count += 1
                    track.contributing_cameras = tuple(sorted(set(track.contributing_cameras + st_obj["cameras"])))
                    track.points_by_camera.update(st_obj["points_by_camera"])
                    track.crop_evidence.update(st_obj["crops_by_camera"])
                    track.semantic_observations.extend(st_obj["semantic_observations"])
                    track.current_semantic_belief = self._compute_consensus_semantic_belief(track.semantic_observations)
                    affected_tracks.append(track)

            for i, st_obj in enumerate(stage_objects):
                if i not in matched_stage_indices:
                    new_id = self._allocate_instance_id()
                    sem_belief = self._compute_consensus_semantic_belief(st_obj["semantic_observations"])
                    new_track = ObservedObjectTrack(
                        instance_id=new_id,
                        first_seen_stage=stage_index,
                        last_seen_stage=stage_index,
                        source_inspection_region_id=source_region_id,
                        fused_points=st_obj["points"],
                        fused_colors=st_obj["colors"],
                        crop_evidence=st_obj["crops_by_camera"],
                        points_by_camera=st_obj["points_by_camera"],
                        contributing_cameras=st_obj["cameras"],
                        semantic_observations=list(st_obj["semantic_observations"]),
                        current_semantic_belief=sem_belief,
                        overall_confidence=0.9,
                        evidence_count=1,
                        status="ACTIVE",
                    )
                    self._tracks[new_id] = new_track
                    affected_tracks.append(new_track)
        else:
            for st_obj in stage_objects:
                new_id = self._allocate_instance_id()
                sem_belief = self._compute_consensus_semantic_belief(st_obj["semantic_observations"])
                new_track = ObservedObjectTrack(
                    instance_id=new_id,
                    first_seen_stage=stage_index,
                    last_seen_stage=stage_index,
                    source_inspection_region_id=source_region_id,
                    fused_points=st_obj["points"],
                    fused_colors=st_obj["colors"],
                    crop_evidence=st_obj["crops_by_camera"],
                    points_by_camera=st_obj["points_by_camera"],
                    contributing_cameras=st_obj["cameras"],
                    semantic_observations=list(st_obj["semantic_observations"]),
                    current_semantic_belief=sem_belief,
                    overall_confidence=0.9,
                    evidence_count=1,
                    status="ACTIVE",
                )
                self._tracks[new_id] = new_track
                affected_tracks.append(new_track)

        return affected_tracks
