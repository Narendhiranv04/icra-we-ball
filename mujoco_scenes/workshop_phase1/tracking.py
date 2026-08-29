"""Multi-view fusion and persistent cross-stage instance tracking."""

from __future__ import annotations

from typing import Any
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from mujoco_scenes.geometry_checker import (
    MeasurementEvidence,
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
        volume_margin_m: float = 0.08,
        min_points_per_mask: int = 8,
        stage_object_merge_distance_threshold_m: float = 0.0,
        fusion_config: dict[str, Any] | None = None,
    ) -> None:
        self.cluster_distance_threshold_m = cluster_distance_threshold_m
        self.track_match_distance_threshold_m = track_match_distance_threshold_m
        self.voxel_size_m = voxel_size_m
        self.min_cluster_points = min_cluster_points
        self.volume_margin_m = float(volume_margin_m)
        self.min_points_per_mask = int(min_points_per_mask)
        self.stage_object_merge_distance_threshold_m = float(
            stage_object_merge_distance_threshold_m)
        self.fusion_config = dict(fusion_config or {})

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
    def _measurement_evidence(instance_id: str, stage_index: int,
                              source_region_id: str,
                              stage_object: dict[str, Any]) -> MeasurementEvidence:
        return MeasurementEvidence(
            instance_name=instance_id,
            measurement_points=stage_object["points"],
            measurement_colors=stage_object["colors"],
            contributing_camera_ids=stage_object["cameras"],
            points_by_camera=stage_object["points_by_camera"],
            source_stage=stage_index,
            source_region=source_region_id,
            measurement_cloud_path=None,
            measurement_quality={
                "quality_is_valid": True,
                "raw_inside_point_count": len(stage_object["points"]),
                "outlier_points_removed": 0,
                "measurement_method": "stage_local_multiview_masked_rgbd",
            },
        )

    @staticmethod
    def _compute_consensus_semantic_belief(
        observations: list[dict[str, Any]],
        fusion_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Aggregate multi-view semantic observations using consensus fusion with ambiguity tracking."""
        if not observations:
            return {
                "status": "UNKNOWN",
                "canonical_label": None,
                "plausible_labels": [],
                "ambiguity_hypotheses": [],
                "reason_codes": ["NO_ASSOCIATED_DETECTION"],
                "raw_label": "unknown",
                "confidence": 0.0,
                "total_observations": 0,
                "supporting_view_count": 0,
                "label_supporting_view_count": {},
                "label_confidence_sum": {},
            }

        fusion_config = fusion_config or {}
        min_views = int(fusion_config.get("minimum_supporting_views", 2))
        min_mean_conf = float(fusion_config.get("minimum_mean_confidence", 0.03))
        min_winning_margin = float(fusion_config.get("minimum_winning_label_margin", 0.08))
        max_conflicting_view_frac = float(fusion_config.get("maximum_conflicting_view_fraction", 0.60))
        max_conflicting_score_frac = float(fusion_config.get("maximum_conflicting_score_fraction", 0.40))
        min_conflicting_mean_conf = float(fusion_config.get("minimum_conflicting_mean_confidence", 0.10))
        crop_mult = float(fusion_config.get("proposal_crop_score_multiplier", 20.0))

        per_camera_label = {}
        for obs in observations:
            camera = obs.get("camera_id")
            if not camera:
                continue

            quality = float(obs.get("observation_quality", 1.0))
            is_crop = (obs.get("inference_source") == "proposal_crop")
            mult = crop_mult if is_crop else 1.0
            hypotheses = [{
                "canonical_label": obs.get("canonical_label", "unknown"),
                "raw_label": obs.get("raw_label", "unknown"),
                "confidence": obs.get("confidence", 0.0),
            }, *list(obs.get("semantic_alternatives", []))]
            for hypothesis in hypotheses:
                label = str(hypothesis.get("canonical_label", "unknown")).lower()
                if not label or label in ("unknown", "object", "object_proposal"):
                    continue
                conf = float(hypothesis.get("confidence", 0.0))
                score = mult * conf * quality
                key = (camera, label)
                if key not in per_camera_label or score > per_camera_label[key]["score"]:
                    per_camera_label[key] = {
                        "camera_id": camera,
                        "canonical_label": label,
                        "raw_label": hypothesis.get("raw_label", label),
                        "confidence": conf,
                        "score": score,
                    }

        labels = sorted({label for (_, label) in per_camera_label})
        if not labels:
            return {
                "status": "UNKNOWN",
                "canonical_label": None,
                "plausible_labels": [],
                "ambiguity_hypotheses": [],
                "reason_codes": ["SEMANTIC_LABEL_UNKNOWN"],
                "raw_label": "unknown",
                "confidence": 0.0,
                "total_observations": len(observations),
                "supporting_view_count": 0,
                "label_supporting_view_count": {},
                "label_confidence_sum": {},
            }

        label_records = []
        raw_by_label: dict[str, str] = {}
        label_views: dict[str, set[str]] = defaultdict(set)
        label_confidence_sum: dict[str, float] = defaultdict(float)
        score_by_label: dict[str, float] = defaultdict(float)
        for label in labels:
            supporting = [entry for (cam, canon), entry in per_camera_label.items() if canon == label]
            views = len({entry["camera_id"] for entry in supporting})
            score = sum(entry["score"] for entry in supporting)
            mean_conf = float(np.mean([entry["confidence"] for entry in supporting])) if supporting else 0.0
            label_views[label] = {entry["camera_id"] for entry in supporting}
            label_confidence_sum[label] = score
            score_by_label[label] = score
            raw_by_label[label] = supporting[0]["raw_label"]
            label_records.append({
                "label": label,
                "supporting_view_count": views,
                "score": score,
                "mean_confidence": mean_conf,
            })

        winner_policy = fusion_config.get("winner_policy", "supporting_views_then_weighted_score")
        if winner_policy == "weighted_score_then_supporting_views":
            label_records.sort(
                key=lambda r: (
                    -r["score"],
                    -r["supporting_view_count"],
                    r["label"],
                )
            )
        elif winner_policy == "supporting_views_then_weighted_score":
            label_records.sort(
                key=lambda r: (
                    -r["supporting_view_count"],
                    -r["score"],
                    r["label"],
                )
            )
        else:
            raise ValueError(f"Unknown winner_policy: {winner_policy}")

        winner = label_records[0]
        runner = label_records[1] if len(label_records) > 1 else None

        reasons = []
        if winner["supporting_view_count"] < min_views:
            reasons.append("INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT")
        if winner["mean_confidence"] < min_mean_conf:
            reasons.append("INSUFFICIENT_DETECTOR_CONFIDENCE")

        for candidate in label_records[1:]:
            view_diff = winner["supporting_view_count"] - candidate["supporting_view_count"]
            score_diff = winner["score"] - candidate["score"]
            is_equal_views = (view_diff == 0)
            if is_equal_views and score_diff < min_winning_margin:
                reasons.append("CONFLICTING_MULTI_VIEW_LABELS")
                break
            elif (
                candidate["supporting_view_count"] >= min_views
                and candidate["mean_confidence"] >= min_conflicting_mean_conf
                and (candidate["supporting_view_count"] / max(winner["supporting_view_count"], 1)) >= max_conflicting_view_frac
                and (candidate["score"] / max(winner["score"], 1e-6)) >= max_conflicting_score_frac
            ):
                reasons.append("CONFLICTING_MULTI_VIEW_LABELS")
                break

        status = "SUPPORTED" if not reasons else "UNKNOWN"
        lack_of_evidence = any(
            r in reasons
            for r in (
                "NO_ASSOCIATED_DETECTION",
                "INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT",
                "INSUFFICIENT_DETECTOR_CONFIDENCE",
                "SEMANTIC_LABEL_UNKNOWN",
            )
        )

        plausible_labels: list[str] = []
        if status == "SUPPORTED":
            plausible_labels = [winner["label"]]
        elif status == "UNKNOWN" and "CONFLICTING_MULTI_VIEW_LABELS" in reasons and not lack_of_evidence:
            competing = [winner["label"]]
            for r in label_records[1:]:
                if (
                    r["supporting_view_count"] >= min_views
                    and r["label"] not in competing
                    and (r["score"] / max(winner["score"], 1e-6)) >= max_conflicting_score_frac
                ):
                    competing.append(r["label"])
            plausible_labels = competing

        best_label = winner["label"]
        total_score = sum(score_by_label.values())
        norm_conf = min(0.99, score_by_label[best_label] / max(1.0, total_score) * min(1.0, 0.5 + 0.25 * len(observations)))

        return {
            "status": status,
            "canonical_label": winner["label"] if status == "SUPPORTED" else None,
            "plausible_labels": plausible_labels,
            "ambiguity_hypotheses": list(plausible_labels),
            "reason_codes": reasons,
            "raw_label": raw_by_label.get(best_label, best_label),
            "confidence": round(norm_conf, 4),
            "total_observations": len(per_camera_label),
            "supporting_view_count": winner["supporting_view_count"],
            "label_supporting_view_count": {
                label: len(views) for label, views in label_views.items()
            },
            "label_confidence_sum": dict(label_confidence_sum),
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
                if mask.gated_points_world_m is not None and mask.gated_pixel_indices_yx is not None:
                    pts = np.asarray(mask.gated_points_world_m)
                    pixel_indices = np.asarray(mask.gated_pixel_indices_yx, dtype=int)
                else:
                    pts, pixel_indices = backproject_masked_depth(
                        obs.depth_m, mask.binary_mask, obs.intrinsics,
                        obs.camera_position_world, obs.camera_rotation_world,
                        max_depth=3.0)
                    gated = gate_points_to_volume(
                        pts, minimum_world_m=stage_volume_min,
                        maximum_world_m=stage_volume_max,
                        boundary_margin_m=self.volume_margin_m)
                    pts, pixel_indices = pts[gated], pixel_indices[gated]
                if len(pts) < self.min_points_per_mask:
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
                    "physical_proposal_id": d["mask"].duplicate_group_id,
                    "inference_source": d["mask"].inference_source,
                    "semantic_alternatives": list(d["mask"].semantic_alternatives),
                    "physical_support_quality": d["mask"].physical_support_quality,
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

        # A detector can retain two slightly different same-camera masks for
        # one physical item. The view-level clusterer intentionally keeps them
        # apart, so perform an optional conservative 3D merge afterward. This
        # is disabled by default and enabled only by profiles that have
        # calibrated the threshold against their close-view point clouds.
        merge_distance = self.stage_object_merge_distance_threshold_m
        if merge_distance > 0.0 and len(stage_objects) > 1:
            merged_objects: list[dict[str, Any]] = []
            for candidate in stage_objects:
                target = next((existing for existing in merged_objects
                               if np.linalg.norm(candidate["centroid"] - existing["centroid"])
                               < merge_distance), None)
                if target is None:
                    merged_objects.append(candidate)
                    continue
                points = np.vstack([target["points"], candidate["points"]])
                colors = np.vstack([target["colors"], candidate["colors"]])
                points, colors = voxel_downsample(
                    points, colors, voxel_size=self.voxel_size_m)
                target["points"], target["colors"] = points, colors
                target["centroid"] = points.mean(axis=0)
                target["cameras"] = tuple(sorted(set(
                    target["cameras"]) | set(candidate["cameras"])))
                for camera, camera_points in candidate["points_by_camera"].items():
                    if camera in target["points_by_camera"]:
                        target["points_by_camera"][camera] = np.vstack([
                            target["points_by_camera"][camera], camera_points])
                    else:
                        target["points_by_camera"][camera] = camera_points
                for camera, crop in candidate["crops_by_camera"].items():
                    old = target["crops_by_camera"].get(camera)
                    if old is None or crop.size > old.size:
                        target["crops_by_camera"][camera] = crop
                target["semantic_observations"].extend(
                    candidate["semantic_observations"])
            stage_objects = merged_objects

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
                    track.current_semantic_belief = self._compute_consensus_semantic_belief(
                        track.semantic_observations, fusion_config=self.fusion_config
                    )
                    track.current_measurement_evidence = self._measurement_evidence(
                        t_id, stage_index, source_region_id, st_obj)
                    track.current_geometric_properties = {}
                    affected_tracks.append(track)

            for i, st_obj in enumerate(stage_objects):
                if i not in matched_stage_indices:
                    new_id = self._allocate_instance_id()
                    sem_belief = self._compute_consensus_semantic_belief(
                        st_obj["semantic_observations"], fusion_config=self.fusion_config
                    )
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
                        current_measurement_evidence=self._measurement_evidence(
                            new_id, stage_index, source_region_id, st_obj),
                        overall_confidence=0.9,
                        evidence_count=1,
                        status="ACTIVE",
                    )
                    self._tracks[new_id] = new_track
                    affected_tracks.append(new_track)
        else:
            for st_obj in stage_objects:
                new_id = self._allocate_instance_id()
                sem_belief = self._compute_consensus_semantic_belief(
                    st_obj["semantic_observations"], fusion_config=self.fusion_config
                )
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
                    current_measurement_evidence=self._measurement_evidence(
                        new_id, stage_index, source_region_id, st_obj),
                    overall_confidence=0.9,
                    evidence_count=1,
                    status="ACTIVE",
                )
                self._tracks[new_id] = new_track
                affected_tracks.append(new_track)

        return affected_tracks
