"""RGB-only semantic detection, mask association, and multi-view fusion.

The detector receives only rendered RGB pixels and a configurable vocabulary.
MuJoCo segmentation is used after detection solely to associate boxes with the
generic persistent IDs used by the observed-state registry.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "semantic_grounding.yaml"
VOCABULARY_PATH = (
    Path(__file__).resolve().parent / "configs" / "semantic_vocabulary.yaml"
)
SEMANTIC_OBSERVATION_SOURCE = "RGB_DETECTOR"
SEMANTIC_ASSOCIATION_METHOD = "mask_box_overlap_v1"


def _font(size: int, bold: bool = False):
    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True)
class Detection:
    """Normalized open-vocabulary detector output."""

    raw_label: str
    canonical_label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    source_camera: str | None = None
    detector_name: str | None = None
    checkpoint: str | None = None
    detector_version: str | None = None
    inference_resolution: tuple[int, int] | None = None
    input_kind: str = "FULL_FRAME"
    input_image_path: str | None = None
    input_crop_box_xyxy: tuple[int, int, int, int] | None = None
    detection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "bbox_xyxy",
            "inference_resolution",
            "input_crop_box_xyxy",
        ):
            if payload[key] is not None:
                payload[key] = list(payload[key])
        return payload


class SemanticDetector(Protocol):
    """Backend-independent detector boundary."""

    def detect(
        self,
        image: np.ndarray,
        vocabulary: Sequence[str],
    ) -> Sequence[Detection]:
        ...


class NullSemanticDetector:
    """Explicit no-detector backend used by legacy geometry-only runs."""

    name = "none"
    checkpoint = None
    version = None

    def detect(
        self,
        image: np.ndarray,
        vocabulary: Sequence[str],
    ) -> Sequence[Detection]:
        del image, vocabulary
        return ()


class YOLOWorldSemanticDetector:
    """Ultralytics YOLO-World adapter with configurable open vocabulary."""

    name = "ultralytics_yolo_world"

    def __init__(
        self,
        checkpoint: str,
        *,
        confidence_threshold: float,
        inference_size: int,
        device: str,
        max_detections: int,
    ):
        try:
            from ultralytics import YOLOWorld
        except ImportError as error:
            raise RuntimeError(
                "YOLO-World semantic grounding requires the pinned "
                "`ultralytics` dependency"
            ) from error
        self.checkpoint = checkpoint
        self.confidence_threshold = float(confidence_threshold)
        self.inference_size = int(inference_size)
        self.device = str(device)
        self.max_detections = int(max_detections)
        self.version = importlib.metadata.version("ultralytics")
        self._model = YOLOWorld(checkpoint)
        self._active_vocabulary: tuple[str, ...] | None = None

    def detect(
        self,
        image: np.ndarray,
        vocabulary: Sequence[str],
    ) -> Sequence[Detection]:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Semantic detector input must be uint8 H×W×3 RGB")
        prompts = tuple(str(label) for label in vocabulary)
        if not prompts:
            return ()
        if prompts != self._active_vocabulary:
            self._model.set_classes(list(prompts))
            self._active_vocabulary = prompts
        # PIL makes the RGB channel convention explicit. Ultralytics treats a
        # bare NumPy source as OpenCV/BGR.
        results = self._model.predict(
            source=Image.fromarray(rgb, mode="RGB"),
            conf=self.confidence_threshold,
            imgsz=self.inference_size,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = rgb.shape[:2]
        detections = []
        for box, confidence, class_id in zip(xyxy, confidences, classes):
            raw_label = str(result.names[int(class_id)])
            detections.append(
                Detection(
                    raw_label=raw_label,
                    canonical_label=raw_label.strip().lower(),
                    confidence=float(confidence),
                    bbox_xyxy=tuple(float(value) for value in box),
                    detector_name=self.name,
                    checkpoint=self.checkpoint,
                    detector_version=self.version,
                    inference_resolution=(width, height),
                )
            )
        return detections


def load_semantic_config(
    path: str | Path = CONFIG_PATH,
    *,
    vocabulary_path: str | Path | None = None,
) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Semantic grounding config must be a mapping")
    if vocabulary_path is not None:
        with Path(vocabulary_path).open(encoding="utf-8") as stream:
            vocabulary = yaml.safe_load(stream)
        config["vocabulary"] = {
            "canonical_labels": vocabulary["canonical_labels"]
        }
    canonical = config.get("vocabulary", {}).get("canonical_labels")
    if not isinstance(canonical, dict) or not canonical:
        raise ValueError("Semantic vocabulary needs canonical_labels")
    normalized: dict[str, list[str]] = {}
    alias_owner: dict[str, str] = {}
    for canonical_label, aliases in canonical.items():
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(
                f"Canonical label '{canonical_label}' needs detector aliases"
            )
        canonical_key = str(canonical_label).strip().lower()
        normalized[canonical_key] = []
        for alias in aliases:
            raw = str(alias).strip().lower()
            if not raw:
                raise ValueError("Detector aliases must be non-empty")
            if raw in alias_owner and alias_owner[raw] != canonical_key:
                raise ValueError(f"Detector alias '{raw}' is ambiguous")
            alias_owner[raw] = canonical_key
            normalized[canonical_key].append(raw)
    config["vocabulary"]["canonical_labels"] = normalized
    config["_alias_to_canonical"] = alias_owner
    return config


def detector_vocabulary(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        alias
        for aliases in config["vocabulary"]["canonical_labels"].values()
        for alias in aliases
    )


def create_semantic_detector(
    config: dict[str, Any],
    *,
    backend: str | None = None,
    checkpoint: str | None = None,
    confidence_threshold: float | None = None,
) -> SemanticDetector:
    detector_config = config["detector"]
    selected = str(backend or detector_config.get("backend", "none"))
    if selected in {"none", "disabled"}:
        return NullSemanticDetector()
    if selected not in {"yolo_world", "yoloworld"}:
        raise ValueError(f"Unsupported semantic detector backend: {selected}")
    return YOLOWorldSemanticDetector(
        checkpoint or detector_config["checkpoint"],
        confidence_threshold=(
            confidence_threshold
            if confidence_threshold is not None
            else detector_config["confidence_threshold"]
        ),
        inference_size=detector_config["inference_size"],
        device=detector_config.get("device", "cpu"),
        max_detections=detector_config.get("max_detections", 100),
    )


def canonicalize_detection(
    detection: Detection,
    config: dict[str, Any],
) -> Detection:
    raw = detection.raw_label.strip().lower()
    canonical = config["_alias_to_canonical"].get(raw, raw)
    return replace(detection, raw_label=raw, canonical_label=canonical)


def _clipped_box(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    left = max(0, min(width, int(math.floor(min(x1, x2)))))
    top = max(0, min(height, int(math.floor(min(y1, y2)))))
    right = max(0, min(width, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def _association_metrics(
    detection: Detection,
    mask: np.ndarray,
) -> dict[str, float | int] | None:
    height, width = mask.shape
    box = _clipped_box(detection.bbox_xyxy, width, height)
    mask_box = _mask_bbox(mask)
    if box is None or mask_box is None:
        return None
    mask_area = int(np.count_nonzero(mask))
    box_area = int((box[2] - box[0]) * (box[3] - box[1]))
    intersection = int(np.count_nonzero(mask[box[1] : box[3], box[0] : box[2]]))
    mask_fraction = intersection / max(mask_area, 1)
    box_fraction = intersection / max(box_area, 1)
    iou = _box_iou(box, mask_box)
    detection_center = np.asarray(
        ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    )
    mask_center = np.asarray(
        (
            (mask_box[0] + mask_box[2]) / 2.0,
            (mask_box[1] + mask_box[3]) / 2.0,
        )
    )
    diagonal = max(
        math.hypot(mask_box[2] - mask_box[0], mask_box[3] - mask_box[1]),
        1.0,
    )
    center_consistency = max(
        0.0, 1.0 - float(np.linalg.norm(detection_center - mask_center)) / diagonal
    )
    return {
        "visible_mask_pixels": mask_area,
        "box_pixels": box_area,
        "intersection_pixels": intersection,
        "mask_fraction_inside_box": float(mask_fraction),
        "box_fraction_on_mask": float(box_fraction),
        "box_mask_iou": float(iou),
        "center_consistency": float(center_consistency),
    }


def associate_detections_to_masks(
    detections: Sequence[Detection],
    object_masks: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically perform one-to-one box/mask association."""
    association_config = config["association"]
    weights = association_config["weights"]
    minimum_mask = int(
        association_config["minimum_visible_mask_pixels"]
    )
    minimum_intersection = int(
        association_config["minimum_box_mask_intersection_pixels"]
    )
    minimum_mask_fraction = float(
        association_config["minimum_mask_fraction_inside_box"]
    )
    minimum_iou = float(association_config["minimum_box_iou"])
    minimum_score = float(
        association_config["minimum_association_score"]
    )
    ambiguity_margin = float(association_config["ambiguity_margin"])
    candidates: list[dict[str, Any]] = []
    for detection_index, detection in enumerate(detections):
        for object_id, mask in sorted(object_masks.items()):
            metrics = _association_metrics(detection, np.asarray(mask, bool))
            if metrics is None or metrics["visible_mask_pixels"] < minimum_mask:
                continue
            score = sum(
                float(weights[key]) * float(metrics[key])
                for key in weights
            )
            # Spatial quality determines whether an association is valid.
            # Detector confidence is used only to break competition between
            # overlapping label proposals for the same physical object.
            matching_score = score * float(detection.confidence)
            weak = (
                metrics["intersection_pixels"] < minimum_intersection
                or (
                    metrics["mask_fraction_inside_box"] < minimum_mask_fraction
                    and metrics["box_mask_iou"] < minimum_iou
                )
                or score < minimum_score
            )
            candidates.append(
                {
                    "detection_index": detection_index,
                    "object_id": object_id,
                    "association_score": float(score),
                    "matching_score": float(matching_score),
                    "metrics": metrics,
                    "weak": bool(weak),
                }
            )

    ambiguous_detection_indices: set[int] = set()
    for detection_index in range(len(detections)):
        ranked = sorted(
            (
                candidate
                for candidate in candidates
                if candidate["detection_index"] == detection_index
                and not candidate["weak"]
            ),
            key=lambda record: (
                -record["association_score"],
                record["object_id"],
            ),
        )
        if (
            len(ranked) > 1
            and ranked[0]["association_score"]
            - ranked[1]["association_score"]
            < ambiguity_margin
        ):
            ambiguous_detection_indices.add(detection_index)

    valid = [
        candidate
        for candidate in candidates
        if not candidate["weak"]
        and candidate["detection_index"] not in ambiguous_detection_indices
    ]
    valid.sort(
        key=lambda record: (
            -record["matching_score"],
            -record["association_score"],
            record["detection_index"],
            record["object_id"],
        )
    )
    matched_detections: set[int] = set()
    matched_objects: set[str] = set()
    accepted = []
    for candidate in valid:
        detection_index = candidate["detection_index"]
        object_id = candidate["object_id"]
        if detection_index in matched_detections or object_id in matched_objects:
            continue
        matched_detections.add(detection_index)
        matched_objects.add(object_id)
        accepted.append(
            {
                **candidate,
                "status": "ACCEPTED",
                "detection": detections[detection_index].to_dict(),
            }
        )

    rejected = []
    for candidate in candidates:
        if any(
            accepted_candidate["detection_index"]
            == candidate["detection_index"]
            and accepted_candidate["object_id"] == candidate["object_id"]
            for accepted_candidate in accepted
        ):
            continue
        reason = (
            "AMBIGUOUS_ASSOCIATION"
            if candidate["detection_index"] in ambiguous_detection_indices
            else "WEAK_ASSOCIATION"
            if candidate["weak"]
            else "ONE_TO_ONE_CONFLICT"
        )
        rejected.append({**candidate, "status": "REJECTED", "reason": reason})
    return {
        "method": association_config.get(
            "method", SEMANTIC_ASSOCIATION_METHOD
        ),
        "accepted": accepted,
        "rejected": rejected,
        "unmatched_detection_indices": [
            index
            for index in range(len(detections))
            if index not in matched_detections
        ],
        "unmatched_object_ids": [
            object_id
            for object_id in sorted(object_masks)
            if object_id not in matched_objects
        ],
    }


def fuse_semantic_observations(
    observations: Sequence[dict[str, Any]],
    *,
    config: dict[str, Any],
    stage: int,
    region_id: str,
    detector_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Fuse associated labels across cameras without forcing a label."""
    fusion = config["fusion"]
    per_camera_label: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in observations:
        detection = observation["detection"]
        key = (
            str(detection["source_camera"]),
            str(detection["canonical_label"]),
        )
        current = per_camera_label.get(key)
        if current is None or (
            detection["confidence"],
            observation["association_score"],
        ) > (
            current["detection"]["confidence"],
            current["association_score"],
        ):
            per_camera_label[key] = observation

    label_records = []
    area_power = float(fusion.get("visible_area_weight_power", 0.15))
    association_power = float(
        fusion.get("association_quality_power", 1.0)
    )
    labels = sorted({key[1] for key in per_camera_label})
    for label in labels:
        supporting = [
            observation
            for (camera_id, canonical), observation in per_camera_label.items()
            if canonical == label
        ]
        weights = []
        for observation in supporting:
            detection = observation["detection"]
            visible_area = max(
                int(observation["metrics"]["visible_mask_pixels"]), 1
            )
            weights.append(
                float(detection["confidence"])
                * float(observation["association_score"]) ** association_power
                * visible_area**area_power
            )
        label_records.append(
            {
                "label": label,
                "score": float(sum(weights)),
                "supporting_view_count": len(
                    {
                        observation["detection"]["source_camera"]
                        for observation in supporting
                    }
                ),
                "mean_confidence": float(
                    np.mean(
                        [
                            observation["detection"]["confidence"]
                            for observation in supporting
                        ]
                    )
                ),
                "camera_ids": sorted(
                    {
                        observation["detection"]["source_camera"]
                        for observation in supporting
                    }
                ),
            }
        )
    winner_policy = fusion.get(
        "winner_policy", "supporting_views_then_weighted_score"
    )
    if winner_policy == "supporting_views_then_weighted_score":
        label_records.sort(
            key=lambda record: (
                -record["supporting_view_count"],
                -record["score"],
                record["label"],
            )
        )
    elif winner_policy == "weighted_score_then_supporting_views":
        label_records.sort(
            key=lambda record: (
                -record["score"],
                -record["supporting_view_count"],
                record["label"],
            )
        )
    else:
        raise ValueError(
            f"Unsupported semantic fusion winner_policy: {winner_policy}"
        )
    winner = label_records[0] if label_records else None
    runner = label_records[1] if len(label_records) > 1 else None
    weighted_score_margin = (
        winner["score"] - runner["score"]
        if winner is not None and runner is not None
        else (winner["score"] if winner is not None else 0.0)
    )
    supporting_view_margin = (
        winner["supporting_view_count"] - runner["supporting_view_count"]
        if winner is not None and runner is not None
        else (
            winner["supporting_view_count"]
            if winner is not None
            else 0
        )
    )
    margin = (
        float(supporting_view_margin)
        if (
            winner_policy == "supporting_views_then_weighted_score"
            and supporting_view_margin != 0
        )
        else float(weighted_score_margin)
    )
    reasons = []
    if winner is None:
        reasons.append("NO_ASSOCIATED_DETECTION")
    else:
        if winner["supporting_view_count"] < int(
            fusion["minimum_supporting_views"]
        ):
            reasons.append("INSUFFICIENT_SEMANTIC_CAMERA_SUPPORT")
        if winner["mean_confidence"] < float(
            fusion["minimum_mean_confidence"]
        ):
            reasons.append("INSUFFICIENT_DETECTOR_CONFIDENCE")
        equal_primary_support = (
            runner is not None
            and (
                winner_policy != "supporting_views_then_weighted_score"
                or supporting_view_margin == 0
            )
        )
        if equal_primary_support and margin < float(
            fusion["minimum_winning_label_margin"]
        ):
            reasons.append("CONFLICTING_MULTI_VIEW_LABELS")
    status = "SUPPORTED" if not reasons else "UNKNOWN"
    return {
        "status": status,
        "canonical_label": (
            winner["label"] if status == "SUPPORTED" else None
        ),
        "alternatives": label_records,
        "winning_label_margin": margin,
        "winning_label_margin_kind": (
            "supporting_view_count"
            if (
                winner_policy == "supporting_views_then_weighted_score"
                and supporting_view_margin != 0
            )
            else "weighted_score"
        ),
        "supporting_view_margin": supporting_view_margin,
        "weighted_score_margin": weighted_score_margin,
        "winner_policy": winner_policy,
        "source_stage": stage,
        "source_region": region_id,
        "contributing_camera_ids": (
            winner["camera_ids"] if winner is not None else []
        ),
        "supporting_view_count": (
            winner["supporting_view_count"] if winner is not None else 0
        ),
        "mean_confidence": (
            winner["mean_confidence"] if winner is not None else None
        ),
        "detector_name": detector_metadata.get("name"),
        "checkpoint": detector_metadata.get("checkpoint"),
        "detector_version": detector_metadata.get("version"),
        "association_method": config["association"].get(
            "method", SEMANTIC_ASSOCIATION_METHOD
        ),
        "semantic_evidence_paths": sorted(
            {
                observation["detection"].get("input_image_path")
                for observation in observations
                if observation["detection"].get("input_image_path")
            }
        ),
        "observation_source": SEMANTIC_OBSERVATION_SOURCE,
        "reason_codes": reasons,
        "quality": {
            "supporting_view_count": (
                winner["supporting_view_count"] if winner else 0
            ),
            "mean_confidence": (
                winner["mean_confidence"] if winner else None
            ),
            "winning_label_margin": margin,
            "winning_label_margin_kind": (
                "supporting_view_count"
                if (
                    winner_policy
                    == "supporting_views_then_weighted_score"
                    and supporting_view_margin != 0
                )
                else "weighted_score"
            ),
            "supporting_view_margin": supporting_view_margin,
            "weighted_score_margin": weighted_score_margin,
            "winner_policy": winner_policy,
            "validated": status == "SUPPORTED",
        },
    }


def _expanded_mask_crop(
    mask: np.ndarray,
    *,
    padding_fraction: float,
    minimum_size: int,
) -> tuple[int, int, int, int] | None:
    box = _mask_bbox(mask)
    if box is None:
        return None
    height, width = mask.shape
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    pad_x = max(int(round(box_width * padding_fraction)), 2)
    pad_y = max(int(round(box_height * padding_fraction)), 2)
    left = max(0, box[0] - pad_x)
    top = max(0, box[1] - pad_y)
    right = min(width, box[2] + pad_x)
    bottom = min(height, box[3] + pad_y)
    if min(right - left, bottom - top) < minimum_size:
        extra_x = max(0, minimum_size - (right - left))
        extra_y = max(0, minimum_size - (bottom - top))
        left = max(0, left - extra_x // 2)
        right = min(width, right + extra_x - extra_x // 2)
        top = max(0, top - extra_y // 2)
        bottom = min(height, bottom + extra_y - extra_y // 2)
    return left, top, right, bottom


def _translated_detection(
    detection: Detection,
    *,
    camera_id: str,
    detection_id: str,
    input_kind: str,
    input_image_path: str,
    crop_box: tuple[int, int, int, int] | None = None,
) -> Detection:
    if crop_box is None:
        box = detection.bbox_xyxy
    else:
        box = (
            detection.bbox_xyxy[0] + crop_box[0],
            detection.bbox_xyxy[1] + crop_box[1],
            detection.bbox_xyxy[2] + crop_box[0],
            detection.bbox_xyxy[3] + crop_box[1],
        )
    return replace(
        detection,
        bbox_xyxy=tuple(float(value) for value in box),
        source_camera=camera_id,
        input_kind=input_kind,
        input_image_path=input_image_path,
        input_crop_box_xyxy=crop_box,
        detection_id=detection_id,
    )


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    interior = (
        mask
        & np.roll(mask, 1, axis=0)
        & np.roll(mask, -1, axis=0)
        & np.roll(mask, 1, axis=1)
        & np.roll(mask, -1, axis=1)
    )
    return mask & ~interior


def _render_overlay(
    rgb: np.ndarray,
    object_masks: dict[str, np.ndarray],
    detections: Sequence[Detection],
    associations: dict[str, Any],
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, np.uint8), mode="RGB")
    pixels = np.asarray(image).copy()
    colors = (
        (41, 128, 185),
        (142, 68, 173),
        (22, 160, 133),
        (211, 84, 0),
    )
    for index, (object_id, mask) in enumerate(sorted(object_masks.items())):
        boundary = _mask_boundary(mask)
        pixels[boundary] = colors[index % len(colors)]
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    accepted_by_detection = {
        record["detection_index"]: record
        for record in associations["accepted"]
    }
    for index, detection in enumerate(detections):
        accepted = accepted_by_detection.get(index)
        color = (35, 166, 89) if accepted else (210, 55, 55)
        draw.rectangle(detection.bbox_xyxy, outline=color, width=2)
        association_label = (
            accepted["object_id"] if accepted else "unmatched"
        )
        label = (
            f"{detection.raw_label} {detection.confidence:.2f} · "
            f"{association_label}"
        )
        x1, y1, _x2, _y2 = detection.bbox_xyxy
        font = _font(11, bold=True)
        raw_box = draw.textbbox((0, 0), label, font=font)
        text_width = raw_box[2] - raw_box[0]
        text_height = raw_box[3] - raw_box[1]
        label_x = max(1, min(float(x1), image.width - text_width - 3))
        label_y = (
            max(1, float(y1) - text_height - 3)
            if y1 >= text_height + 4
            else min(image.height - text_height - 2, float(y1) + 2)
        )
        text_box = draw.textbbox(
            (label_x, label_y), label, font=font
        )
        draw.rectangle(text_box, fill=color)
        draw.text(
            (label_x, label_y),
            label,
            fill="white",
            font=font,
        )
    return image


def run_semantic_inspection(
    inspection,
    *,
    accepted_instance_to_object_id: dict[str, str],
    detector: SemanticDetector,
    config: dict[str, Any],
    stage: int,
    region_id: str,
    stage_dir: Path,
    save_overlays: bool,
    role_rank_by_label: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Detect RGB semantics and associate only region-accepted instances."""
    semantics_dir = Path(stage_dir) / "semantics"
    semantics_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = detector_vocabulary(config)
    detector_metadata = {
        "name": getattr(detector, "name", detector.__class__.__name__),
        "checkpoint": getattr(detector, "checkpoint", None),
        "version": getattr(detector, "version", None),
    }
    all_detections: list[dict[str, Any]] = []
    all_associations: list[dict[str, Any]] = []
    observations_by_object: dict[str, list[dict[str, Any]]] = {
        object_id: []
        for object_id in accepted_instance_to_object_id.values()
    }
    camera_summaries = []
    overlay_images: list[tuple[str, Image.Image]] = []
    inference_seconds = 0.0

    for camera_id, capture in inspection.cameras.items():
        if not capture.validation.get("usable", False):
            continue
        object_masks = {
            object_id: capture.instance_masks[instance_name]
            for instance_name, object_id in accepted_instance_to_object_id.items()
            if instance_name in capture.instance_masks
            and np.count_nonzero(capture.instance_masks[instance_name])
        }
        detections: list[Detection] = []
        camera_rgb_path = f"cameras/{camera_id}/rgb.png"
        started = time.perf_counter()
        raw = detector.detect(capture.rgb, vocabulary)
        inference_seconds += time.perf_counter() - started
        for local_index, detection in enumerate(raw):
            normalized = canonicalize_detection(detection, config)
            detections.append(
                _translated_detection(
                    normalized,
                    camera_id=camera_id,
                    detection_id=f"{camera_id}_full_{local_index:03d}",
                    input_kind="FULL_FRAME",
                    input_image_path=camera_rgb_path,
                )
            )

        crop_config = config.get("mask_crop", {})
        if crop_config.get("enabled", False):
            crop_root = semantics_dir / "cameras" / camera_id / "crops"
            crop_root.mkdir(parents=True, exist_ok=True)
            for crop_index, (object_id, mask) in enumerate(
                sorted(object_masks.items())
            ):
                crop_box = _expanded_mask_crop(
                    mask,
                    padding_fraction=float(
                        crop_config.get("padding_fraction", 0.3)
                    ),
                    minimum_size=int(
                        crop_config.get("minimum_crop_size_pixels", 32)
                    ),
                )
                if crop_box is None:
                    continue
                left, top, right, bottom = crop_box
                crop = capture.rgb[top:bottom, left:right].copy()
                if crop_config.get("neutralize_background", False):
                    crop_mask = mask[top:bottom, left:right]
                    crop[~crop_mask] = 127
                crop_path = (
                    crop_root
                    / f"{crop_index:03d}_{object_id}.png"
                )
                Image.fromarray(crop).save(crop_path)
                started = time.perf_counter()
                raw_crop = detector.detect(crop, vocabulary)
                inference_seconds += time.perf_counter() - started
                relative_crop_path = crop_path.relative_to(stage_dir).as_posix()
                for local_index, detection in enumerate(raw_crop):
                    normalized = canonicalize_detection(detection, config)
                    detections.append(
                        _translated_detection(
                            normalized,
                            camera_id=camera_id,
                            detection_id=(
                                f"{camera_id}_crop_{crop_index:03d}_"
                                f"{local_index:03d}"
                            ),
                            input_kind="MASK_BOUNDED_RGB_CROP",
                            input_image_path=relative_crop_path,
                            crop_box=crop_box,
                        )
                    )

        associations = associate_detections_to_masks(
            detections, object_masks, config
        )
        for accepted in associations["accepted"]:
            observations_by_object[accepted["object_id"]].append(accepted)
        all_detections.extend(
            {"camera_id": camera_id, **detection.to_dict()}
            for detection in detections
        )
        all_associations.append(
            {
                "camera_id": camera_id,
                **associations,
            }
        )
        camera_summaries.append(
            {
                "camera_id": camera_id,
                "detection_count": len(detections),
                "accepted_association_count": len(
                    associations["accepted"]
                ),
                "unmatched_detection_count": len(
                    associations["unmatched_detection_indices"]
                ),
                "unmatched_visible_object_ids": associations[
                    "unmatched_object_ids"
                ],
            }
        )
        if save_overlays:
            overlay = _render_overlay(
                capture.rgb, object_masks, detections, associations
            )
            overlay_path = (
                semantics_dir / "cameras" / camera_id / "overlay.png"
            )
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(overlay_path)
            overlay_images.append((camera_id, overlay))

    semantic_records = {}
    for object_id in sorted(observations_by_object):
        record = fuse_semantic_observations(
            observations_by_object[object_id],
            config=config,
            stage=stage,
            region_id=region_id,
            detector_metadata=detector_metadata,
        )
        object_dir = semantics_dir / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        semantic_path = object_dir / "semantic_evidence.json"
        record["semantic_record_path"] = (
            semantic_path.relative_to(stage_dir).as_posix()
        )
        semantic_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        semantic_records[object_id] = record

    (semantics_dir / "detections.json").write_text(
        json.dumps(
            {
                "observation_source": SEMANTIC_OBSERVATION_SOURCE,
                "detector": detector_metadata,
                "vocabulary": list(vocabulary),
                "detections": all_detections,
                "camera_summaries": camera_summaries,
                "inference_seconds": inference_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (semantics_dir / "associations.json").write_text(
        json.dumps(
            {
                "method": config["association"].get(
                    "method", SEMANTIC_ASSOCIATION_METHOD
                ),
                "cameras": all_associations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if save_overlays and overlay_images:
        columns = 2
        panel_width = max(image.width for _camera, image in overlay_images)
        panel_height = max(image.height for _camera, image in overlay_images) + 28
        rows = math.ceil(len(overlay_images) / columns)
        summary_height = 46 + 32 * max(len(semantic_records), 1)
        overview = Image.new(
            "RGB",
            (
                columns * panel_width,
                rows * panel_height + summary_height,
            ),
            "white",
        )
        draw = ImageDraw.Draw(overview)
        for index, (camera_id, overlay) in enumerate(overlay_images):
            x = (index % columns) * panel_width
            y = (index // columns) * panel_height
            draw.text(
                (x + 8, y + 4),
                camera_id,
                fill=(25, 35, 50),
                font=_font(16, bold=True),
            )
            overview.paste(overlay, (x, y + 28))
        summary_y = rows * panel_height
        summary_width = columns * panel_width
        draw.rectangle(
            (0, summary_y, summary_width, summary_y + summary_height),
            fill=(242, 246, 250),
        )
        draw.text(
            (12, summary_y + 8),
            "Fused multi-view semantic evidence",
            fill=(22, 35, 50),
            font=_font(20, bold=True),
        )
        for row, (object_id, record) in enumerate(
            sorted(semantic_records.items())
        ):
            canonical = record.get("canonical_label")
            status = record.get("status", "UNKNOWN")
            ranks = []
            if canonical is not None:
                for role, label_ranks in sorted(
                    (role_rank_by_label or {}).items()
                ):
                    rank = label_ranks.get(canonical)
                    if rank is not None:
                        ranks.append(f"{role}: rank {rank}")
            rank_text = ", ".join(ranks) if ranks else "no compatible role"
            label = canonical if canonical is not None else "unresolved"
            text = (
                f"{object_id}: {label} · {status} · "
                f"{record.get('supporting_view_count', 0)} supporting views · "
                f"{rank_text}"
            )
            color = (
                (21, 117, 72)
                if status == "SUPPORTED"
                else (100, 110, 120)
            )
            draw.text(
                (18, summary_y + 42 + row * 30),
                text,
                fill=color,
                font=_font(17, bold=status == "SUPPORTED"),
            )
        overview.save(stage_dir / "semantic_overview.png")

    return {
        "detector": detector_metadata,
        "vocabulary": list(vocabulary),
        "semantic_records": semantic_records,
        "detections": all_detections,
        "associations": all_associations,
        "camera_summaries": camera_summaries,
        "inference_seconds": inference_seconds,
    }
