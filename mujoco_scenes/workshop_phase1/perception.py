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
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else find_yolo_world_weights()
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.inference_size = int(inference_size)
        self.device = device or "cpu"
        self.max_detections = int(max_detections)

        self._model = None
        self._prompts: list[str] = []
        self._alias_to_canonical: dict[str, str] = {}
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
                except Exception:
                    pass

    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        if self._model is None or not self._prompts:
            return []

        rgb = observation.rgb
        depth = observation.depth_m
        height, width = rgb.shape[:2]

        # Explicit PIL Image format with mode RGB
        pil_image = Image.fromarray(rgb, mode="RGB")

        try:
            results = self._model.predict(
                source=pil_image,
                conf=self.confidence_threshold,
                iou=self.nms_iou_threshold,
                imgsz=self.inference_size,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception:
            return []

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)

        masks: list[ObservedMask] = []
        for idx, (box, conf, cls_id) in enumerate(zip(xyxy, confidences, classes)):
            raw_label = str(result.names[int(cls_id)]) if int(cls_id) in result.names else self._prompts[cls_id] if cls_id < len(self._prompts) else "object"
            canonical_label = self._alias_to_canonical.get(raw_label.lower(), raw_label.lower())

            x1 = max(0, min(width - 1, int(box[0])))
            y1 = max(0, min(height - 1, int(box[1])))
            x2 = max(0, min(width, int(box[2]) + 1))
            y2 = max(0, min(height, int(box[3]) + 1))

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue

            # Depth-guided foreground mask within bounding box
            roi_depth = depth[y1:y2, x1:x2]
            valid_roi = np.isfinite(roi_depth) & (roi_depth > 0.1) & (roi_depth < 3.0)
            if np.count_nonzero(valid_roi) < 8:
                continue

            # Segment foreground within bounding box based on median depth
            med_depth = np.median(roi_depth[valid_roi])
            depth_thresh = max(0.04, med_depth * 0.08)
            fg_roi = valid_roi & (np.abs(roi_depth - med_depth) <= depth_thresh)

            bin_mask = np.zeros((height, width), dtype=bool)
            bin_mask[y1:y2, x1:x2] = fg_roi

            # Reject edge depth discontinuities
            bin_mask = reject_depth_discontinuities(depth, bin_mask, threshold_m=0.03, radius_pixels=2)

            # Check 3D back-projection into active inspection volume
            pts, _ = backproject_masked_depth(
                depth,
                bin_mask,
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

            mask_entry = ObservedMask(
                detection_id=f"det_{observation.camera_id}_{idx:03d}",
                camera_id=observation.camera_id,
                binary_mask=bin_mask,
                bounding_box_xyxy=(x1, y1, x2, y2),
                confidence=float(conf),
                canonical_label=canonical_label,
                raw_label=raw_label,
                predicted_label=canonical_label,
                backend_name="yolo_world",
            )
            masks.append(mask_entry)

        return masks


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

            pts, _ = backproject_masked_depth(
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

            masks.append(
                ObservedMask(
                    detection_id=f"oracle_{observation.camera_id}_b{bid:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=bmask,
                    bounding_box_xyxy=(x1, y1, x2, y2),
                    confidence=1.0,
                    canonical_label="object",
                    raw_label="object",
                    predicted_label="object",
                    backend_name="privileged_oracle",
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
