"""Object discovery and instance proposal backends for Workshop Phase 1."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    gate_points_to_volume,
    reject_depth_discontinuities,
)
from mujoco_scenes.workshop_phase1.types import ObservedMask, ViewObservation

YOLO_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolov8s-worldv2.pt"


class InstanceProposalBackend(abc.ABC):
    """Abstract interface for 2D instance proposal from calibrated RGB-D observation."""

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
    """Open-vocabulary object detector + depth-guided foreground segmenter."""

    def __init__(
        self,
        weights_path: Path | None = None,
        confidence_threshold: float = 0.12,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self.weights_path = weights_path or YOLO_WEIGHTS_PATH
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._model = None
        self._classes = [
            "screwdriver",
            "screw",
            "bolt",
            "electric drill",
            "drill",
            "tool",
            "fastener",
            "pliers",
            "wrench",
        ]
        self._initialize_model()

    def _initialize_model(self) -> None:
        if not self.weights_path.is_file():
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.weights_path))
            self._model.set_classes(self._classes)
        except Exception:
            self._model = None

    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        if self._model is None:
            # Fallback to connected-component proposal if weights are unavailable
            fallback = RGBDConnectedComponentProposalBackend()
            return fallback.predict(observation, stage_volume_min, stage_volume_max, volume_margin_m)

        rgb = observation.rgb
        depth = observation.depth_m
        height, width = rgb.shape[:2]

        results = self._model.predict(
            source=rgb,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        masks: list[ObservedMask] = []
        for idx in range(len(boxes)):
            xyxy = boxes.xyxy[idx].cpu().numpy().astype(int)
            conf = float(boxes.conf[idx].cpu().numpy())
            cls_id = int(boxes.cls[idx].cpu().numpy())
            label = self._classes[cls_id] if 0 <= cls_id < len(self._classes) else "object"

            x1 = max(0, min(width - 1, xyxy[0]))
            y1 = max(0, min(height - 1, xyxy[1]))
            x2 = max(0, min(width, xyxy[2]))
            y2 = max(0, min(height, xyxy[3]))

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                continue

            # Depth-guided foreground mask within bounding box
            roi_depth = depth[y1:y2, x1:x2]
            valid_roi = np.isfinite(roi_depth) & (roi_depth > 0.1) & (roi_depth < 3.0)
            if np.count_nonzero(valid_roi) < 10:
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
            if len(pts) < 10:
                continue

            gated = gate_points_to_volume(
                pts,
                minimum_world_m=stage_volume_min,
                maximum_world_m=stage_volume_max,
                boundary_margin_m=volume_margin_m,
            )
            if np.count_nonzero(gated) < 10:
                continue

            mask_entry = ObservedMask(
                detection_id=f"det_{observation.camera_id}_{idx:03d}",
                camera_id=observation.camera_id,
                binary_mask=bin_mask,
                bounding_box_xyxy=(x1, y1, x2, y2),
                confidence=conf,
                predicted_label=label,
                backend_name="yolo_world",
            )
            masks.append(mask_entry)

        return masks


class RGBDConnectedComponentProposalBackend(InstanceProposalBackend):
    """Zero-weight heuristic proposal backend using depth clustering, 3D spatial gating, and CLIP zero-shot classification."""

    def __init__(
        self,
        min_area_pixels: int = 40,
        max_area_pixels: int = 150000,
        max_physical_span_m: float = 0.35,
        use_clip: bool = True,
    ) -> None:
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.max_physical_span_m = max_physical_span_m
        self.use_clip = use_clip
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_text_features = None
        self._device = "cpu"
        self._classes = [
            "phillips screwdriver",
            "flathead screwdriver",
            "stubby phillips screwdriver",
            "power drill",
            "phillips screw",
            "hex bolt",
            "combination wrench",
            "pliers",
        ]

        if self.use_clip:
            try:
                import clip
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load("ViT-B/32", device=self._device)
                self._clip_model = model
                self._clip_preprocess = preprocess

                prompts = [f"a photo of a {c}" for c in self._classes]
                text_tokens = clip.tokenize(prompts).to(self._device)
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                    self._clip_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            except Exception:
                self._clip_model = None

    def _classify_crop(self, roi_rgb: np.ndarray) -> tuple[str, float]:
        if self._clip_model is None or self._clip_text_features is None:
            return "object", 0.80

        try:
            import torch
            from PIL import Image
            img = Image.fromarray(roi_rgb)
            img_tensor = self._clip_preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                img_feat = self._clip_model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ self._clip_text_features.T).squeeze(0)
                probs = torch.softmax(sims * 100.0, dim=-1).cpu().numpy()
                best_idx = int(np.argmax(probs))
                return self._classes[best_idx], float(probs[best_idx])
        except Exception:
            return "object", 0.80

    def predict(
        self,
        observation: ViewObservation,
        stage_volume_min: np.ndarray,
        stage_volume_max: np.ndarray,
        volume_margin_m: float = 0.08,
    ) -> list[ObservedMask]:
        depth = observation.depth_m
        rgb = observation.rgb
        height, width = depth.shape[:2]

        # Valid depth mask
        valid_depth = np.isfinite(depth) & (depth > 0.15) & (depth < 2.5)
        if np.count_nonzero(valid_depth) < 100:
            return []

        # Find depth gradients/edges
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

            # Physical 3D extent check (reject walls, floors, large shelves)
            extents = gated_pts.max(axis=0) - gated_pts.min(axis=0)
            if float(extents.max()) > self.max_physical_span_m:
                continue

            # Classify crop with CLIP
            roi_rgb = rgb[max(0, y):min(height, y + h), max(0, x):min(width, x + w)]
            pred_label, conf = self._classify_crop(roi_rgb) if roi_rgb.size > 0 else ("object", 0.80)

            masks.append(
                ObservedMask(
                    detection_id=f"cc_{observation.camera_id}_{l_idx:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=comp_mask,
                    bounding_box_xyxy=(x, y, x + w, y + h),
                    confidence=conf,
                    predicted_label=pred_label,
                    backend_name="connected_components",
                )
            )

        return masks


class PrivilegedOracleMaskBackend(InstanceProposalBackend):
    """Ablation & upper-bound backend using MuJoCo segmentation masks.

    ALLOWED ONLY under explicit oracle/ablation testing. NEVER in production.
    """

    def __init__(self, scene: Any) -> None:
        self.scene = scene

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

        # Collect distinct body instances
        body_to_mask: dict[int, np.ndarray] = {}
        for gid in unique_gids:
            if gid < 0 or gid >= self.scene.model.ngeom:
                continue
            bid = self.scene.model.geom_bodyid[gid]
            # Strictly select pickable free bodies (dofnum == 6)
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

            bname = mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, bid) or "object"
            if "flathead" in bname:
                label = "flathead screwdriver"
            elif "stubby" in bname:
                label = "stubby phillips screwdriver"
            elif "power" in bname or "drill" in bname:
                label = "power drill"
            elif "driver" in bname or "screwdriver" in bname:
                label = "phillips screwdriver"
            elif "screw" in bname:
                label = "phillips screw"
            elif "hex" in bname or "bolt" in bname:
                label = "hex bolt"
            elif "wrench" in bname:
                label = "combination wrench"
            elif "pliers" in bname:
                label = "pliers"
            else:
                label = "object"

            masks.append(
                ObservedMask(
                    detection_id=f"oracle_{observation.camera_id}_b{bid:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=bmask,
                    bounding_box_xyxy=(x1, y1, x2, y2),
                    confidence=1.0,
                    predicted_label=label,
                    backend_name="privileged_oracle",
                )
            )

        return masks
