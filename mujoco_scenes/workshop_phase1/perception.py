"""Object discovery and instance proposal backends for Workshop Phase 1.

Implements YOLO-World object detection from semantic vocabulary, depth-guided
mask refinement within bounding boxes, and semantic-free oracle segmentation.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
    reject_depth_discontinuities,
)
from mujoco_scenes.workshop_phase1.types import (
    FunctionalRequirement,
    ObservedMask,
    ViewObservation,
)


def find_yolo_world_weights() -> Path:
    """Find canonical YOLO-World weights with precedence order."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "semantic_model_cache" / "yolov8m-worldv2.pt",
        repo_root / "mujoco_scenes" / "yolov8m-worldv2.pt",
        repo_root / "mujoco_scenes" / "yolov8s-worldv2.pt",
        repo_root / "yolov8s-worldv2.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


class InstanceProposalBackend(abc.ABC):
    """Abstract interface for 2D instance proposal from calibrated RGB-D observation."""

    @abc.abstractmethod
    def set_vocabulary(self, prompts: list[str], alias_to_canonical: dict[str, str]) -> None:
        """Update detector classes based on semantic vocabulary."""
        pass

    @abc.abstractmethod
    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        """Predict 2D instance masks for objects inside the active stage volume."""
        pass


class YOLOWorldProposalBackend(InstanceProposalBackend):
    """YOLO-World open-vocabulary object detector with depth-guided mask refinement."""

    def __init__(
        self,
        weights_path: Path | str | None = None,
        confidence_threshold: float = 0.05,
        nms_iou_threshold: float = 0.45,
        inference_size: int = 640,
        device: str | None = None,
        max_detections: int = 100,
        min_points_per_mask: int = 8,
        duplicate_box_iou_threshold: float = 0.65,
        duplicate_mask_overlap_threshold: float = 0.72,
        duplicate_centroid_distance_m: float = 0.018,
        duplicate_aabb_overlap_threshold: float = 0.45,
        enable_full_frame: bool = True,
        enable_stage_crop: bool = True,
        enable_stage_tiles: bool = False,
        tile_overlap_fraction: float = 0.15,
        supplemental_prompts: list[str] | None = None,
        supplemental_confidence_threshold: float | None = None,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else find_yolo_world_weights()
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.inference_size = int(inference_size)
        self.device = device or "cpu"
        self.max_detections = int(max_detections)
        self.min_points_per_mask = int(min_points_per_mask)
        self.duplicate_box_iou_threshold = float(duplicate_box_iou_threshold)
        self.duplicate_mask_overlap_threshold = float(duplicate_mask_overlap_threshold)
        self.duplicate_centroid_distance_m = float(duplicate_centroid_distance_m)
        self.duplicate_aabb_overlap_threshold = float(duplicate_aabb_overlap_threshold)
        self.enable_full_frame = bool(enable_full_frame)
        self.enable_stage_crop = bool(enable_stage_crop)
        self.enable_stage_tiles = bool(enable_stage_tiles)
        self.tile_overlap_fraction = float(tile_overlap_fraction)
        self.supplemental_prompts = [str(label).strip() for label in (supplemental_prompts or [])]
        self.supplemental_confidence_threshold = float(
            supplemental_confidence_threshold
            if supplemental_confidence_threshold is not None else confidence_threshold
        )

        self._model = None
        self._prompts: list[str] = []
        self._alias_to_canonical: dict[str, str] = {}
        self.last_diagnostics: list[dict[str, Any]] = []
        self._full_text_features = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        if not self.weights_path.is_file():
            return
        try:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(str(self.weights_path))
        except Exception:
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.weights_path))
            except Exception:
                self._model = None

    def set_vocabulary(self, prompts: list[str], alias_to_canonical: dict[str, str]) -> None:
        """Establish detector classes from FM-contract vocabulary."""
        clean_prompts = [p.strip() for p in prompts if p.strip()]
        if clean_prompts != self._prompts:
            self._prompts = clean_prompts
            self._alias_to_canonical = {k.lower(): v for k, v in alias_to_canonical.items()}
            if self._model is not None and self._prompts:
                try:
                    self._model.set_classes(self._prompts)
                    self._full_text_features = self._model.model.txt_feats.detach().clone()
                except Exception:
                    pass

    def _activate_prompt_pass(self, prompt: str | None) -> None:
        """Select the full vocabulary or one cached prompt without rerunning CLIP."""
        if self._model is None or self._full_text_features is None:
            return
        if prompt is None:
            indices = list(range(len(self._prompts)))
            names = list(self._prompts)
        else:
            indices = [self._prompts.index(prompt)]
            names = [prompt]
        features = self._full_text_features[:, indices].clone()
        core = self._model.model
        core.txt_feats = features
        core.names = names
        core.model[-1].nc = len(names)
        predictor = getattr(self._model, "predictor", None)
        if predictor is not None and getattr(predictor, "model", None) is not None:
            predictor.model.names = names
            backend_core = getattr(predictor.model, "model", None)
            if backend_core is not None and hasattr(backend_core, "txt_feats"):
                backend_core.txt_feats = features
                backend_core.names = names
                backend_core.model[-1].nc = len(names)

    @staticmethod
    def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = max(1, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)
        return intersection / union

    @staticmethod
    def _mask_overlap_over_smaller(a: np.ndarray, b: np.ndarray) -> float:
        intersection = int(np.count_nonzero(a & b))
        return intersection / max(1, min(int(np.count_nonzero(a)), int(np.count_nonzero(b))))

    @staticmethod
    def _aabb_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        a_min, a_max = np.asarray(a["minimum_world_m"]), np.asarray(a["maximum_world_m"])
        b_min, b_max = np.asarray(b["minimum_world_m"]), np.asarray(b["maximum_world_m"])
        intersection = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
        inter_volume = float(np.prod(intersection))
        a_volume = float(np.prod(np.maximum(0.0, a_max - a_min)))
        b_volume = float(np.prod(np.maximum(0.0, b_max - b_min)))
        return inter_volume / max(1e-12, a_volume + b_volume - inter_volume)

    def suppress_duplicate_proposals(self, proposals: list[ObservedMask]) -> list[ObservedMask]:
        """Collapse same-camera semantic hypotheses for one physical RGB-D proposal."""
        if len(proposals) < 2:
            return proposals
        ordered = sorted(
            proposals,
            key=lambda proposal: (
                -(proposal.confidence * max(0.1, proposal.physical_support_quality)
                  * np.log1p(max(1, proposal.depth_point_count))),
                -proposal.depth_point_count,
                proposal.canonical_label,
                proposal.detection_id,
            ),
        )
        retained: list[ObservedMask] = []
        group_index = 0
        for candidate in ordered:
            duplicate_of: ObservedMask | None = None
            for accepted in retained:
                box_iou = self._box_iou(candidate.bounding_box_xyxy, accepted.bounding_box_xyxy)
                mask_overlap = self._mask_overlap_over_smaller(candidate.binary_mask, accepted.binary_mask)
                centroid_distance = float(np.linalg.norm(
                    np.asarray(candidate.centroid_world_m) - np.asarray(accepted.centroid_world_m)))
                aabb_iou = self._aabb_iou(candidate.cloud_bounds_world_m, accepted.cloud_bounds_world_m)
                strong_2d = (box_iou >= self.duplicate_box_iou_threshold
                             or mask_overlap >= self.duplicate_mask_overlap_threshold)
                strong_3d = (centroid_distance <= self.duplicate_centroid_distance_m
                             and aabb_iou >= self.duplicate_aabb_overlap_threshold)
                # Require overlapping image support. This preserves nearby small hardware
                # whose centroids alone happen to be close.
                if strong_2d and (strong_3d or mask_overlap >= 0.90):
                    duplicate_of = accepted
                    break
            if duplicate_of is None:
                group_index += 1
                candidate.duplicate_group_id = f"{candidate.camera_id}_physical_{group_index:03d}"
                retained.append(candidate)
                continue
            duplicate_of.semantic_alternatives.append({
                "canonical_label": candidate.canonical_label,
                "raw_label": candidate.raw_label,
                "confidence": candidate.confidence,
                "detection_id": candidate.detection_id,
                "inference_source": candidate.inference_source,
            })
            self.last_diagnostics.append({
                "camera_id": candidate.camera_id,
                "detection_id": candidate.detection_id,
                "raw_label": candidate.raw_label,
                "canonical_label": candidate.canonical_label,
                "confidence": candidate.confidence,
                "status": "SUPPRESSED_DUPLICATE",
                "raw_yolo_bbox_xyxy": list(
                    candidate.raw_yolo_bbox_xyxy or candidate.bounding_box_xyxy),
                "refined_bbox_xyxy": list(
                    candidate.refined_bbox_xyxy or candidate.bounding_box_xyxy),
                "inference_source": candidate.inference_source,
                "duplicate_group_id": duplicate_of.duplicate_group_id,
                "retained_detection_id": duplicate_of.detection_id,
            })
        return sorted(retained, key=lambda proposal: proposal.detection_id)

    @staticmethod
    def _project_volume_crop(
        observation: ViewObservation,
        minimum_world_m: np.ndarray,
        maximum_world_m: np.ndarray,
    ) -> tuple[int, int, int, int]:
        """Project the calibrated active-stage volume to a conservative image crop."""
        corners = np.array([
            [x, y, z]
            for x in (minimum_world_m[0], maximum_world_m[0])
            for y in (minimum_world_m[1], maximum_world_m[1])
            for z in (minimum_world_m[2], maximum_world_m[2])
        ])
        local = (corners - observation.camera_position_world) @ observation.camera_rotation_world
        depth = -local[:, 2]
        valid = depth > 0.05
        height, width = observation.rgb.shape[:2]
        if np.count_nonzero(valid) < 4:
            return 0, 0, width, height
        fx, fy = observation.intrinsics[0, 0], observation.intrinsics[1, 1]
        cx, cy = observation.intrinsics[0, 2], observation.intrinsics[1, 2]
        u = local[valid, 0] * fx / depth[valid] + cx
        v = -local[valid, 1] * fy / depth[valid] + cy
        padding_x, padding_y = int(0.04 * width), int(0.04 * height)
        x1 = max(0, int(np.floor(np.min(u))) - padding_x)
        y1 = max(0, int(np.floor(np.min(v))) - padding_y)
        x2 = min(width, int(np.ceil(np.max(u))) + padding_x)
        y2 = min(height, int(np.ceil(np.max(v))) + padding_y)
        if x2 - x1 < 64 or y2 - y1 < 64:
            return 0, 0, width, height
        return x1, y1, x2, y2

    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        self.last_diagnostics = []
        if self._model is None or not self._prompts:
            return []

        rgb, depth = observation.rgb, observation.depth_m
        height, width = rgb.shape[:2]
        windows = self._inference_windows(
            observation, stage_volume_min, stage_volume_max)
        masks: list[ObservedMask] = []
        global_index = 0
        active_supplements = [
            prompt for prompt in self.supplemental_prompts if prompt in self._prompts
        ]
        for inference_source, (crop_x1, crop_y1, crop_x2, crop_y2) in windows:
            detector_rgb = rgb[crop_y1:crop_y2, crop_x1:crop_x2]
            for supplemental_prompt in [None, *active_supplements]:
                self._activate_prompt_pass(supplemental_prompt)
                pass_source = (
                    inference_source if supplemental_prompt is None else
                    f"{inference_source}_supplemental_{supplemental_prompt.lower().replace(' ', '_')}"
                )
                pass_threshold = (
                    self.confidence_threshold if supplemental_prompt is None else
                    self.supplemental_confidence_threshold
                )
                try:
                    results = self._model.predict(
                        source=Image.fromarray(detector_rgb, mode="RGB"),
                        conf=pass_threshold,
                        iou=self.nms_iou_threshold,
                        imgsz=self.inference_size,
                        device=self.device,
                        max_det=self.max_detections,
                        verbose=False,
                    )
                except Exception as error:
                    self.last_diagnostics.append({
                        "camera_id": observation.camera_id,
                        "inference_source": pass_source,
                        "status": "INFERENCE_ERROR",
                        "error_type": type(error).__name__,
                    })
                    continue
                if not results or results[0].boxes is None:
                    continue
                result, boxes = results[0], results[0].boxes
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, conf, cls_id in zip(xyxy, confidences, classes):
                    idx = global_index
                    global_index += 1
                    if supplemental_prompt is not None:
                        raw_label = supplemental_prompt
                    elif isinstance(result.names, dict):
                        raw_label = str(result.names.get(int(cls_id), "object"))
                    else:
                        raw_label = (str(result.names[int(cls_id)])
                                     if cls_id < len(result.names) else "object")
                    canonical_label = self._alias_to_canonical.get(
                        raw_label.lower(), raw_label.lower())

                    x1 = max(0, min(width - 1, int(np.floor(box[0])) + crop_x1))
                    y1 = max(0, min(height - 1, int(np.floor(box[1])) + crop_y1))
                    x2 = max(0, min(width, int(np.ceil(box[2])) + crop_x1))
                    y2 = max(0, min(height, int(np.ceil(box[3])) + crop_y1))
                    detection_id = f"det_{observation.camera_id}_{pass_source}_{idx:03d}"

                    diagnostic = {
                        "camera_id": observation.camera_id,
                        "detection_id": detection_id,
                        "detector_vocabulary": list(self._prompts),
                        "raw_label": raw_label,
                        "canonical_label": canonical_label,
                        "confidence": float(conf),
                        "raw_yolo_bbox_xyxy": [x1, y1, x2, y2],
                        "detector_crop_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
                        "inference_source": pass_source,
                    }

                    if (x2 - x1) < 4 or (y2 - y1) < 4:
                        diagnostic.update({"status": "REJECTED_SMALL_BOX"})
                        self.last_diagnostics.append(diagnostic)
                        continue

                    box_width, box_height = x2 - x1, y2 - y1
                    compact_box = (
                        min(box_width, box_height) / max(1, max(box_width, box_height))
                        >= 0.35
                        and min(box_width, box_height) >= 64
                    )
                    bin_mask, support_quality, refinement_reason = self._refine_depth_support(
                        depth, (x1, y1, x2, y2),
                        prefer_volumetric=(canonical_label == "power_driver" and compact_box),
                    )
                    if bin_mask is None:
                        diagnostic.update({"status": refinement_reason})
                        self.last_diagnostics.append(diagnostic)
                        continue

                    pts, pixel_indices = backproject_masked_depth(
                        depth, bin_mask, observation.intrinsics,
                        observation.camera_position_world,
                        observation.camera_rotation_world, max_depth=3.0)
                    if len(pts) < self.min_points_per_mask:
                        diagnostic.update({
                            "status": "REJECTED_REFINEMENT",
                            "refined_mask_area": int(bin_mask.sum()),
                            "depth_point_count": len(pts),
                        })
                        self.last_diagnostics.append(diagnostic)
                        continue

                    gated = gate_points_to_volume(
                        pts, minimum_world_m=stage_volume_min,
                        maximum_world_m=stage_volume_max,
                        boundary_margin_m=volume_margin_m)
                    if np.count_nonzero(gated) < self.min_points_per_mask:
                        diagnostic.update({
                            "status": "REJECTED_STAGE_VOLUME",
                            "refined_mask_area": int(bin_mask.sum()),
                            "depth_point_count": int(np.count_nonzero(gated)),
                        })
                        self.last_diagnostics.append(diagnostic)
                        continue

                    gated_points = pts[gated]
                    gated_pixels = pixel_indices[gated]
                    final_mask = np.zeros((height, width), dtype=bool)
                    final_mask[gated_pixels[:, 0], gated_pixels[:, 1]] = True
                    rows, cols = np.nonzero(final_mask)
                    refined_box = (
                        int(cols.min()), int(rows.min()),
                        int(cols.max() + 1), int(rows.max() + 1),
                    )
                    centroid = gated_points.mean(axis=0)
                    cloud_bounds = {
                        "minimum_world_m": gated_points.min(axis=0).tolist(),
                        "maximum_world_m": gated_points.max(axis=0).tolist(),
                    }

                    mask_entry = ObservedMask(
                        detection_id=detection_id, camera_id=observation.camera_id,
                        binary_mask=final_mask, bounding_box_xyxy=refined_box,
                        confidence=float(conf), canonical_label=canonical_label,
                        raw_label=raw_label, predicted_label=canonical_label,
                        backend_name="yolo_world", refined_mask_area=int(final_mask.sum()),
                        depth_point_count=len(gated_points), centroid_world_m=centroid,
                        cloud_bounds_world_m=cloud_bounds,
                        raw_yolo_bbox_xyxy=(x1, y1, x2, y2),
                        refined_bbox_xyxy=refined_box,
                        gated_points_world_m=gated_points,
                        gated_pixel_indices_yx=gated_pixels,
                        inference_source=pass_source,
                        physical_support_quality=support_quality,
                    )
                    masks.append(mask_entry)
                    diagnostic.update({
                        "status": "ACCEPTED_PRE_DEDUP",
                        "refined_bbox_xyxy": list(refined_box),
                        "refined_mask_area": mask_entry.refined_mask_area,
                        "depth_point_count": mask_entry.depth_point_count,
                        "centroid_world_m": centroid.tolist(),
                        "cloud_bounds_world_m": cloud_bounds,
                        "physical_support_quality": support_quality,
                    })
                    self.last_diagnostics.append(diagnostic)

        deduplicated = self.suppress_duplicate_proposals(masks)
        retained_ids = {mask.detection_id for mask in deduplicated}
        for record in self.last_diagnostics:
            if record.get("detection_id") in retained_ids:
                record["status"] = "ACCEPTED"
                record["duplicate_group_id"] = next(
                    mask.duplicate_group_id for mask in deduplicated
                    if mask.detection_id == record["detection_id"])
        return deduplicated

    def _inference_windows(
        self, observation: ViewObservation,
        stage_volume_min: np.ndarray, stage_volume_max: np.ndarray,
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Return deterministic, calibration-driven inference windows."""
        height, width = observation.rgb.shape[:2]
        full = (0, 0, width, height)
        crop = self._project_volume_crop(observation, stage_volume_min, stage_volume_max)
        windows: list[tuple[str, tuple[int, int, int, int]]] = []
        if self.enable_full_frame:
            windows.append(("full_frame", full))
        crop_area = max(1, (crop[2] - crop[0]) * (crop[3] - crop[1]))
        if self.enable_stage_crop and (crop != full or not windows):
            windows.append(("stage_crop", crop))
        if self.enable_stage_tiles and crop_area >= 0.18 * width * height:
            x1, y1, x2, y2 = crop
            crop_w, crop_h = x2 - x1, y2 - y1
            tile_w = int(np.ceil(crop_w / (2.0 - self.tile_overlap_fraction)))
            tile_h = int(np.ceil(crop_h / (2.0 - self.tile_overlap_fraction)))
            x_starts = (x1, max(x1, x2 - tile_w))
            y_starts = (y1, max(y1, y2 - tile_h))
            for row, ty in enumerate(y_starts):
                for col, tx in enumerate(x_starts):
                    windows.append((f"tile_{row}{col}", (tx, ty, min(x2, tx + tile_w), min(y2, ty + tile_h))))
        # Avoid duplicate windows when a projected crop spans the whole frame.
        unique: list[tuple[str, tuple[int, int, int, int]]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for source, window in windows:
            if window not in seen:
                seen.add(window)
                unique.append((source, window))
        return unique

    def _refine_depth_support(
        self, depth: np.ndarray, box: tuple[int, int, int, int],
        prefer_volumetric: bool = False,
    ) -> tuple[np.ndarray | None, float, str]:
        """Extract a coherent foreground component inside an existing YOLO box."""
        x1, y1, x2, y2 = box
        roi = depth[y1:y2, x1:x2]
        valid = np.isfinite(roi) & (roi > 0.1) & (roi < 3.0)
        valid_count = int(valid.sum())
        if valid_count < self.min_points_per_mask:
            return None, 0.0, "REJECTED_DEPTH_SUPPORT"
        if prefer_volumetric:
            # Drills and similarly compact tools have multiple visible depth
            # layers. Retain the nearer volume inside the detector box instead
            # of collapsing it to a single handle/body plane.
            valid_depths = roi[valid]
            far_reference = float(np.percentile(valid_depths, 98.0))
            near_reference = float(np.percentile(valid_depths, 5.0))
            separation = max(0.002, 0.03 * (far_reference - near_reference))
            candidate = valid & (roi <= far_reference - separation)
            candidate = cv2.morphologyEx(
                candidate.astype(np.uint8), cv2.MORPH_CLOSE,
                np.ones((5, 5), dtype=np.uint8), iterations=1,
            ).astype(bool)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                candidate.astype(np.uint8), 8)
            kept = np.zeros_like(candidate)
            minimum_component = max(self.min_points_per_mask, int(0.002 * valid_count))
            for component in range(1, count):
                if int(stats[component, cv2.CC_STAT_AREA]) >= minimum_component:
                    kept |= labels == component
            if int(kept.sum()) >= self.min_points_per_mask:
                full = np.zeros(depth.shape, dtype=bool)
                full[y1:y2, x1:x2] = kept
                return full, float(min(1.0, kept.sum() / max(1, valid_count))), "ACCEPTED_REFINEMENT"
        h, w = roi.shape
        cy1, cy2 = int(0.25 * h), max(int(0.75 * h), int(0.25 * h) + 1)
        cx1, cx2 = int(0.25 * w), max(int(0.75 * w), int(0.25 * w) + 1)
        central = valid[cy1:cy2, cx1:cx2]
        depths = roi[cy1:cy2, cx1:cx2][central]
        if len(depths) < self.min_points_per_mask:
            depths = roi[valid]
        bin_width = 0.008
        lo, hi = float(depths.min()), float(depths.max())
        mode_centres: list[float]
        if hi - lo < bin_width:
            mode_centres = [float(np.median(depths))]
        else:
            edges = np.arange(lo, hi + 2 * bin_width, bin_width)
            counts, edges = np.histogram(depths, bins=edges)
            indices = np.argsort(counts)[::-1][:3]
            mode_centres = [float(0.5 * (edges[index] + edges[index + 1])) for index in indices if counts[index] > 0]
        nearest = float(np.percentile(roi[valid], 5.0))
        farthest = float(np.percentile(roi[valid], 95.0))
        best: tuple[float, np.ndarray] | None = None
        centre_xy = np.array([0.5 * (w - 1), 0.5 * (h - 1)])
        for mode in mode_centres:
            tolerance = max(0.008, 0.012 * mode)
            candidate = valid & (np.abs(roi - mode) <= tolerance)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < self.min_points_per_mask:
                    continue
                component_mask = labels == component
                distance = float(np.linalg.norm(centroids[component] - centre_xy)) / max(1.0, np.linalg.norm(centre_xy))
                centre_score = max(0.0, 1.0 - distance)
                near_score = 1.0 - (mode - nearest) / max(1e-6, farthest - nearest)
                area_score = min(1.0, area / max(1.0, 0.20 * valid_count))
                touches = sum(int(value) for value in (
                    np.any(component_mask[0]), np.any(component_mask[-1]),
                    np.any(component_mask[:, 0]), np.any(component_mask[:, -1])))
                edge_penalty = 0.08 * float(touches)
                score = 0.50 * centre_score + 0.30 * near_score + 0.20 * area_score - edge_penalty
                if best is None or score > best[0]:
                    best = (score, component_mask)
        if best is None:
            return None, 0.0, "REJECTED_NO_COHERENT_COMPONENT"
        full = np.zeros(depth.shape, dtype=bool)
        full[y1:y2, x1:x2] = best[1]
        cleaned = reject_depth_discontinuities(depth, full, threshold_m=0.03, radius_pixels=1)
        if int(cleaned.sum()) < self.min_points_per_mask:
            cleaned = full
        quality = float(np.clip(best[0], 0.0, 1.0))
        return cleaned, quality, "ACCEPTED_REFINEMENT"


class PrivilegedOracleMaskBackend(InstanceProposalBackend):
    """Ablation & upper-bound backend using MuJoCo segmentation masks.

    ALLOWED ONLY under explicit oracle/ablation testing.
    Provides segmentation masks ONLY with neutral label 'object'.
    Zero semantic typing or body name inspection is performed.
    """

    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def set_vocabulary(self, prompts: list[str], alias_to_canonical: dict[str, str]) -> None:
        pass

    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        if observation.segmentation is None:
            return []

        seg = observation.segmentation
        geom_ids = seg[:, :, 0]
        unique_gids = np.unique(geom_ids)

        # Collect distinct free-body instances (dofnum == 6)
        body_to_mask: dict[int, np.ndarray] = {}
        for gid in unique_gids:
            if gid < 0 or gid >= self.scene.model.ngeom:
                continue
            bid = self.scene.model.geom_bodyid[gid]
            if self.scene.model.body_dofnum[bid] != 6:
                continue

            gmask = (geom_ids == gid)
            if bid not in body_to_mask:
                body_to_mask[bid] = gmask
            else:
                body_to_mask[bid] |= gmask

        masks: list[ObservedMask] = []
        for bid, bmask in body_to_mask.items():
            if np.count_nonzero(bmask) < 8:
                continue

            rows, cols = np.nonzero(bmask)
            x1, y1 = int(cols.min()), int(rows.min())
            x2, y2 = int(cols.max() + 1), int(rows.max() + 1)

            pts, pixel_indices = backproject_masked_depth(
                observation.depth_m,
                bmask,
                observation.intrinsics,
                observation.camera_position_world,
                observation.camera_rotation_world,
                max_depth=3.0,
            )
            if len(pts) < 8:
                continue

            gated = gate_points_to_volume(
                pts,
                minimum_world_m=stage_volume_min,
                maximum_world_m=stage_volume_max,
                boundary_margin_m=volume_margin_m,
            )
            if np.count_nonzero(gated) < 8:
                continue
            gated_points = pts[gated]
            gated_pixels = pixel_indices[gated]
            final_mask = np.zeros_like(bmask, dtype=bool)
            final_mask[gated_pixels[:, 0], gated_pixels[:, 1]] = True
            rows, cols = np.nonzero(final_mask)
            refined_box = (int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1))

            masks.append(
                ObservedMask(
                    detection_id=f"oracle_{observation.camera_id}_b{bid:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=final_mask,
                    bounding_box_xyxy=refined_box,
                    confidence=1.0,
                    canonical_label="object",
                    raw_label="object",
                    predicted_label="object",
                    backend_name="privileged_oracle",
                    refined_mask_area=int(np.count_nonzero(final_mask)),
                    depth_point_count=len(gated_points),
                    centroid_world_m=gated_points.mean(axis=0),
                    cloud_bounds_world_m={
                        "minimum_world_m": gated_points.min(axis=0).tolist(),
                        "maximum_world_m": gated_points.max(axis=0).tolist(),
                    },
                    raw_yolo_bbox_xyxy=(x1, y1, x2, y2),
                    refined_bbox_xyxy=refined_box,
                    gated_points_world_m=gated_points,
                    gated_pixel_indices_yx=gated_pixels,
                    inference_source="oracle_mask",
                    physical_support_quality=1.0,
                )
            )

        return masks


class RGBDConnectedComponentProposalBackend(InstanceProposalBackend):
    """Diagnostic/ablation backend using depth clustering. NOT used in production."""

    def __init__(
        self,
        min_area_pixels: int = 40,
        max_area_pixels: int = 150000,
        max_physical_span_m: float = 0.35,
    ) -> None:
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.max_physical_span_m = max_physical_span_m

    def set_vocabulary(self, prompts: list[str], alias_to_canonical: dict[str, str]) -> None:
        pass

    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        depth = observation.depth_m
        height, width = depth.shape[:2]

        valid_depth = np.isfinite(depth) & (depth > 0.15) & (depth < 2.5)
        if np.count_nonzero(valid_depth) < 100:
            return []

        sobelx = cv2.Sobel(np.nan_to_num(depth, nan=0.0).astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(np.nan_to_num(depth, nan=0.0).astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        smooth_fg = valid_depth & (grad_mag < 0.05)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(smooth_fg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

        masks: list[ObservedMask] = []
        for l_idx in range(1, num_labels):
            area = stats[l_idx, cv2.CC_STAT_AREA]
            if area < self.min_area_pixels or area > self.max_area_pixels:
                continue

            x = stats[l_idx, cv2.CC_STAT_LEFT]
            y = stats[l_idx, cv2.CC_STAT_TOP]
            w = stats[l_idx, cv2.CC_STAT_WIDTH]
            h = stats[l_idx, cv2.CC_STAT_HEIGHT]

            comp_mask = (labels == l_idx)
            pts, _ = backproject_masked_depth(
                depth,
                comp_mask,
                observation.intrinsics,
                observation.camera_position_world,
                observation.camera_rotation_world,
                max_depth=3.0,
            )
            if len(pts) < 15:
                continue

            gated = gate_points_to_volume(
                pts,
                minimum_world_m=stage_volume_min,
                maximum_world_m=stage_volume_max,
                boundary_margin_m=volume_margin_m,
            )
            gated_pts = pts[gated]
            if len(gated_pts) < 15:
                continue

            extents = gated_pts.max(axis=0) - gated_pts.min(axis=0)
            if float(extents.max()) > self.max_physical_span_m:
                continue

            masks.append(
                ObservedMask(
                    detection_id=f"cc_{observation.camera_id}_{l_idx:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=comp_mask,
                    bounding_box_xyxy=(x, y, x + w, y + h),
                    confidence=0.50,
                    canonical_label="object",
                    raw_label="object_proposal",
                    predicted_label="object_proposal",
                    backend_name="connected_components",
                )
            )

        return masks
