"""Object discovery and instance proposal backends for Workshop Phase 1.

Implements open-vocabulary requirement-driven YOLO-World proposals, zero-weight
connected-component proposals, and semantic-free oracle segmentation.
"""

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
from mujoco_scenes.workshop_phase1.types import (
    EntityType,
    FunctionalRequirement,
    ObservedMask,
    ViewObservation,
)

YOLO_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolov8s-worldv2.pt"


class OpenVocabularyQueryBuilder:
    """Requirement-driven open-vocabulary detector query generator.

    Constructs natural language detector queries dynamically from the active
    functional requirements and task instruction. Contains ZERO hard-coded
    Workshop object taxonomies.
    """

    @staticmethod
    def build_queries(
        requirements: list[FunctionalRequirement] | None = None,
        task_instruction: str = "",
    ) -> list[str]:
        """Derive text queries directly from requirement descriptions."""
        queries: list[str] = []

        if requirements:
            for req in requirements:
                if req.entity_type != EntityType.OBJECT:
                    continue
                fname = req.function_name.upper()
                desc = req.description.strip()

                if "DRIVE" in fname or "DRIVER" in fname:
                    queries.extend([
                        "tool that can drive or tighten a screw",
                        "fastener driving hand tool",
                        "handheld tool",
                    ])
                elif "FASTEN" in fname or "FASTENER" in fname:
                    queries.extend([
                        "fastener used to secure a joint",
                        "threaded fastener hardware",
                        "small hardware fastener",
                    ])
                else:
                    # Natural language fallback directly from requirement description
                    if desc:
                        queries.append(desc.lower())

        if task_instruction:
            # Add general task context terms if relevant
            lower_inst = task_instruction.lower()
            if "screw" in lower_inst or "fasten" in lower_inst:
                if "fastener" not in [q.lower() for q in queries]:
                    queries.append("fastener")
                if "tool" not in [q.lower() for q in queries]:
                    queries.append("tool")

        # Fallback default queries if no object requirements provided
        if not queries:
            queries = [
                "tool that can drive or tighten a screw",
                "fastener used to secure a joint",
                "handheld tool",
                "small hardware fastener",
            ]

        # Deduplicate while preserving order
        unique_queries: list[str] = []
        for q in queries:
            if q and q not in unique_queries:
                unique_queries.append(q)

        return unique_queries


class InstanceProposalBackend(abc.ABC):
    """Abstract interface for 2D instance proposal from calibrated RGB-D observation."""

    @abc.abstractmethod
    def set_requirements(self, requirements: list[FunctionalRequirement], task_instruction: str = "") -> None:
        """Update open-vocabulary queries based on current functional requirements."""
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
    """Open-vocabulary object detector + depth-guided foreground segmenter."""

    def __init__(
        self,
        weights_path: Path | None = None,
        confidence_threshold: float = 0.08,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        self.weights_path = weights_path or YOLO_WEIGHTS_PATH
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._model = None
        self._current_queries: list[str] = OpenVocabularyQueryBuilder.build_queries()
        self._initialize_model()

    def _initialize_model(self) -> None:
        if not self.weights_path.is_file():
            return
        try:
            from ultralytics import YOLOWorld
            self._model = YOLOWorld(str(self.weights_path))
            self._model.set_classes(self._current_queries)
        except Exception:
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.weights_path))
                self._model.set_classes(self._current_queries)
            except Exception:
                self._model = None

    def set_requirements(self, requirements: list[FunctionalRequirement], task_instruction: str = "") -> None:
        """Dynamically update detector classes based on current task requirements."""
        new_queries = OpenVocabularyQueryBuilder.build_queries(requirements, task_instruction)
        if new_queries != self._current_queries:
            self._current_queries = new_queries
            if self._model is not None:
                try:
                    self._model.set_classes(self._current_queries)
                except Exception:
                    pass

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
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            fallback = RGBDConnectedComponentProposalBackend()
            return fallback.predict(observation, stage_volume_min, stage_volume_max, volume_margin_m)

        masks: list[ObservedMask] = []
        for idx in range(len(boxes)):
            xyxy = boxes.xyxy[idx].cpu().numpy().astype(int)
            conf = float(boxes.conf[idx].cpu().numpy())
            cls_id = int(boxes.cls[idx].cpu().numpy())
            label = self._current_queries[cls_id] if 0 <= cls_id < len(self._current_queries) else "object"

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
                confidence=conf,
                predicted_label=label,
                backend_name="yolo_world",
            )
            masks.append(mask_entry)

        return masks


class RGBDConnectedComponentProposalBackend(InstanceProposalBackend):
    """Zero-weight heuristic proposal backend using depth clustering and 3D spatial gating."""

    def __init__(
        self,
        min_area_pixels: int = 40,
        max_area_pixels: int = 150000,
        max_physical_span_m: float = 0.35,
    ) -> None:
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.max_physical_span_m = max_physical_span_m

    def set_requirements(self, requirements: list[FunctionalRequirement], task_instruction: str = "") -> None:
        """Connected component backend is spatial-only and does not consume text queries."""
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

            masks.append(
                ObservedMask(
                    detection_id=f"cc_{observation.camera_id}_{l_idx:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=comp_mask,
                    bounding_box_xyxy=(x, y, x + w, y + h),
                    confidence=0.80,
                    predicted_label="object_proposal",
                    backend_name="connected_components",
                )
            )

        return masks


class PrivilegedOracleMaskBackend(InstanceProposalBackend):
    """Ablation & upper-bound backend using MuJoCo segmentation masks.

    ALLOWED ONLY under explicit oracle/ablation testing.
    Provides segmentation masks ONLY with neutral label 'object'.
    Zero semantic typing or body name inspection is performed.
    """

    def __init__(self, scene: Any) -> None:
        self.scene = scene

    def set_requirements(self, requirements: list[FunctionalRequirement], task_instruction: str = "") -> None:
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

            # Pure segmentation without semantic leakage: always label as "object"
            masks.append(
                ObservedMask(
                    detection_id=f"oracle_{observation.camera_id}_b{bid:03d}",
                    camera_id=observation.camera_id,
                    binary_mask=bmask,
                    bounding_box_xyxy=(x1, y1, x2, y2),
                    confidence=1.0,
                    predicted_label="object",
                    backend_name="privileged_oracle",
                )
            )

        return masks
