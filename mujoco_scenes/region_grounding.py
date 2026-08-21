"""Stage-local support-region evidence, properties, semantics, and relations."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from mujoco_scenes.geometry_checker import (
    backproject_masked_depth,
    camera_intrinsics,
    gate_points_to_volume,
    look_at_camera_rotation,
    validate_camera_view,
    voxel_downsample,
    write_ply,
)
from mujoco_scenes.semantic_grounding import (
    Detection,
    SemanticDetector,
    canonicalize_detection,
    detector_vocabulary,
)


REGION_MEASUREMENT_PURPOSE = "REGION_MEASUREMENT_EVIDENCE"
REGION_VISUALIZATION_PURPOSE = (
    "REGION_CUMULATIVE_VISUALIZATION_NOT_MEASUREMENT"
)
REGION_EXTRACTOR_VERSION = "region_support_geometry_v1"
REGION_RELATION_VERSION = "region_payload_relations_v1"


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _property(
    value: Any,
    *,
    status: str,
    method: str,
    unit: str | None = None,
    **provenance: Any,
) -> dict[str, Any]:
    record = {
        "value": value,
        "status": status,
        "method": method,
    }
    if unit is not None:
        record["unit"] = unit
    record.update(provenance)
    return record


def _unknown(method: str, unit: str | None = None) -> dict[str, Any]:
    return _property(None, status="UNKNOWN", method=method, unit=unit)


@dataclass(frozen=True)
class RegionMeasurementEvidence:
    """Fresh region-local support points accepted for property extraction."""

    measurement_points: np.ndarray
    measurement_colors: np.ndarray
    points_by_camera: dict[str, np.ndarray]
    source_stage: int
    inspection_label: str
    measurement_cloud_path: str
    contributing_camera_ids: tuple[str, ...]
    measurement_quality: dict[str, Any]
    cloud_purpose: str = REGION_MEASUREMENT_PURPOSE


@dataclass(frozen=True)
class PayloadMeasurementEvidence:
    """Fresh visible-instance points for the fixed placement payload."""

    measurement_points: np.ndarray
    measurement_colors: np.ndarray
    points_by_camera: dict[str, np.ndarray]
    source_stage: int
    measurement_cloud_path: str
    contributing_camera_ids: tuple[str, ...]
    measurement_quality: dict[str, Any]
    cloud_purpose: str = "PAYLOAD_MEASUREMENT_EVIDENCE"


@dataclass
class RegionCameraCapture:
    camera_id: str
    model_camera_name: str
    rgb: np.ndarray
    depth_m: np.ndarray
    segmentation: np.ndarray
    intrinsics: np.ndarray
    position_world_m: np.ndarray
    rotation_world_from_camera: np.ndarray
    validation: dict[str, Any]
    region_mask: np.ndarray
    payload_mask: np.ndarray
    sofa_mask: np.ndarray
    region_points: np.ndarray
    region_colors: np.ndarray
    payload_points: np.ndarray
    payload_colors: np.ndarray
    sofa_points: np.ndarray


@dataclass
class RegionStageCapture:
    stage: int
    inspection_label: str
    cameras: dict[str, RegionCameraCapture]
    region_evidence: RegionMeasurementEvidence
    payload_evidence: PayloadMeasurementEvidence | None
    sofa_points: np.ndarray
    semantic_observations: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    semantic_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings_seconds: dict[str, float] = field(default_factory=dict)


def require_region_measurement_evidence(
    evidence: RegionMeasurementEvidence,
) -> None:
    if not isinstance(evidence, RegionMeasurementEvidence):
        raise TypeError(
            "Region extraction requires typed RegionMeasurementEvidence"
        )
    if evidence.cloud_purpose != REGION_MEASUREMENT_PURPOSE:
        raise ValueError(
            "Cumulative or combined clouds are invalid region measurement input"
        )
    normalized_path = evidence.measurement_cloud_path.lower()
    if any(
        marker in normalized_path
        for marker in ("cumulative", "combined_cloud", "full_room")
    ):
        raise ValueError(
            "Historical or full-room paths are invalid region measurement input"
        )


def require_payload_measurement_evidence(
    evidence: PayloadMeasurementEvidence,
) -> None:
    if not isinstance(evidence, PayloadMeasurementEvidence):
        raise TypeError(
            "Payload extraction requires typed PayloadMeasurementEvidence"
        )
    if evidence.cloud_purpose != "PAYLOAD_MEASUREMENT_EVIDENCE":
        raise ValueError("Visualization clouds are invalid payload input")
    normalized_path = evidence.measurement_cloud_path.lower()
    if any(
        marker in normalized_path
        for marker in ("cumulative", "combined_cloud", "full_room")
    ):
        raise ValueError(
            "Historical or full-room paths are invalid payload measurement input"
        )


def _single_free_rigid_instance_geom_ids(
    model: mujoco.MjModel,
) -> np.ndarray:
    """Return segmentation geom IDs for the benchmark's fixed payload.

    The L2 task contract contains exactly one independently movable rigid
    payload. Identify that visible instance from kinematic topology and
    segmentation IDs, never from a body/geom/asset name or intended size.
    Its dimensions are still measured exclusively from current RGB-D points.
    """
    roots = {
        int(model.jnt_bodyid[joint_id])
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id])
        == int(mujoco.mjtJoint.mjJNT_FREE)
    }
    if len(roots) != 1:
        raise ValueError(
            "Region benchmark requires exactly one free rigid payload "
            f"instance; observed {len(roots)}"
        )
    root = next(iter(roots))
    ids = []
    for geom_id, body_id in enumerate(model.geom_bodyid):
        cursor = int(body_id)
        while cursor > 0:
            if cursor == root:
                ids.append(geom_id)
                break
            cursor = int(model.body_parentid[cursor])
    return np.asarray(ids, dtype=np.int32)


def _volume_mask_from_world_points(
    points: np.ndarray,
    pixels: np.ndarray,
    shape: tuple[int, int],
    volume: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    selected = gate_points_to_volume(
        points,
        minimum_world_m=np.asarray(volume["minimum_world_m"], float),
        maximum_world_m=np.asarray(volume["maximum_world_m"], float),
        boundary_margin_m=0.0,
    )
    mask = np.zeros(shape, dtype=bool)
    chosen_pixels = pixels[selected]
    if len(chosen_pixels):
        mask[chosen_pixels[:, 0], chosen_pixels[:, 1]] = True
    return mask, points[selected]


def _select_upper_support_plane(
    points: np.ndarray,
    colors: np.ndarray,
    *,
    plane_band_m: float,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite = np.all(np.isfinite(points), axis=1)
    points = np.asarray(points, np.float32)[finite]
    colors = np.asarray(colors, np.uint8)[finite]
    if len(points) < 20:
        return points, colors, {"reason": "INSUFFICIENT_RAW_POINTS"}
    bin_size = max(plane_band_m / 2.0, 0.002)
    bins = np.floor(points[:, 2] / bin_size).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    maximum_count = int(counts.max())
    dense = values[counts >= max(20, int(0.12 * maximum_count))]
    selected_bin = int(dense.max())
    center_z = float(np.median(points[bins == selected_bin, 2]))
    selected = np.abs(points[:, 2] - center_z) <= plane_band_m
    plane_points, plane_colors = voxel_downsample(
        points[selected], colors[selected], voxel_size_m
    )
    return plane_points, plane_colors, {
        "selected_plane_z_m": center_z,
        "raw_point_count": len(points),
        "plane_point_count": len(plane_points),
        "histogram_bin_size_m": bin_size,
    }


def _robust_horizontal_footprint(
    points: np.ndarray,
    percentiles: tuple[float, float],
) -> dict[str, Any] | None:
    samples = np.asarray(points, np.float64)
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) < 20:
        return None
    centroid = np.median(samples, axis=0)
    centered_xy = samples[:, :2] - centroid[:2]
    covariance_xy = np.cov(centered_xy, rowvar=False)
    values_xy, vectors_xy = np.linalg.eigh(covariance_xy)
    order = np.argsort(values_xy)[::-1]
    axes_xy = vectors_xy[:, order]
    projected_xy = centered_xy @ axes_xy
    lower, upper = percentiles
    bounds_low = np.percentile(projected_xy, lower, axis=0)
    bounds_high = np.percentile(projected_xy, upper, axis=0)
    extents = bounds_high - bounds_low
    if not np.all(np.isfinite(extents)):
        return None
    order_extent = np.argsort(extents)[::-1]
    length = float(extents[order_extent[0]])
    width = float(extents[order_extent[1]])
    length_axis_xy = axes_xy[:, order_extent[0]]
    covariance_3d = np.cov(samples - centroid, rowvar=False)
    values_3d, vectors_3d = np.linalg.eigh(covariance_3d)
    normal = vectors_3d[:, int(np.argmin(values_3d))]
    if normal[2] < 0:
        normal *= -1
    thickness = float(
        np.percentile(samples[:, 2], upper)
        - np.percentile(samples[:, 2], lower)
    )
    alignment = float(abs(normal[2]))
    normal_angle = float(
        np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0)))
    )
    planarity = float(
        np.clip(1.0 - thickness / max(min(length, width), 1e-6), 0.0, 1.0)
    )
    return {
        "centroid_world_m": centroid,
        "length_m": length,
        "width_m": width,
        "area_m2": length * width,
        "thickness_m": thickness,
        "normal_world": normal,
        "normal_alignment": alignment,
        "normal_angle_degrees": normal_angle,
        "planarity_score": planarity,
        "principal_axis_world": np.array(
            [length_axis_xy[0], length_axis_xy[1], 0.0]
        ),
    }


def extract_region_properties(
    evidence: RegionMeasurementEvidence,
    *,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    """Measure support properties only from fresh typed region evidence."""
    require_region_measurement_evidence(evidence)
    method = REGION_EXTRACTOR_VERSION
    quality = evidence.measurement_quality
    provenance = {
        "source_stage": evidence.source_stage,
        "source_region": evidence.inspection_label,
        "measurement_cloud_path": evidence.measurement_cloud_path,
        "contributing_camera_ids": list(evidence.contributing_camera_ids),
        "point_count": len(evidence.measurement_points),
        "extractor_version": REGION_EXTRACTOR_VERSION,
    }
    if not quality.get("quality_is_valid", False):
        return {
            "property_status": "UNKNOWN",
            "centroid_world_m": _unknown(method, "m"),
            "support_length_m": _unknown(method, "m"),
            "support_width_m": _unknown(method, "m"),
            "support_area_m2": _unknown(method, "m2"),
            "support_thickness_m": _unknown(method, "m"),
            "support_normal_world": _unknown(method),
            "normal_angle_degrees": _unknown(method, "deg"),
            "planarity_score": _unknown(method),
            "PLANAR_SUPPORT": _unknown(method),
            "measurement_quality": quality,
        }
    result = _robust_horizontal_footprint(
        evidence.measurement_points, (2.0, 98.0)
    )
    if result is None:
        quality = {**quality, "quality_is_valid": False}
        return extract_region_properties(
            RegionMeasurementEvidence(
                **{
                    **evidence.__dict__,
                    "measurement_quality": quality,
                }
            ),
            task_config=task_config,
        )
    requirements = task_config["geometric_requirements"]["unary_region"]
    planar = (
        result["normal_angle_degrees"]
        <= float(requirements["maximum_normal_angle_degrees"])
        and result["planarity_score"]
        >= float(requirements["minimum_planarity_score"])
        and result["area_m2"]
        >= float(requirements["minimum_usable_area_m2"])
    )
    return {
        "property_status": "MEASURED",
        "centroid_world_m": _property(
            result["centroid_world_m"].tolist(),
            status="MEASURED",
            method=method,
            unit="m",
            **provenance,
        ),
        "support_length_m": _property(
            result["length_m"],
            status="MEASURED",
            method=method,
            unit="m",
            **provenance,
        ),
        "support_width_m": _property(
            result["width_m"],
            status="MEASURED",
            method=method,
            unit="m",
            **provenance,
        ),
        "support_area_m2": _property(
            result["area_m2"],
            status="DERIVED",
            method=method,
            unit="m2",
            **provenance,
        ),
        "support_thickness_m": _property(
            result["thickness_m"],
            status="MEASURED",
            method=method,
            unit="m",
            **provenance,
        ),
        "support_normal_world": _property(
            result["normal_world"].tolist(),
            status="MEASURED",
            method=method,
            **provenance,
        ),
        "principal_axis_world": _property(
            result["principal_axis_world"].tolist(),
            status="MEASURED",
            method=method,
            **provenance,
        ),
        "normal_angle_degrees": _property(
            result["normal_angle_degrees"],
            status="DERIVED",
            method=method,
            unit="deg",
            **provenance,
        ),
        "planarity_score": _property(
            result["planarity_score"],
            status="DERIVED",
            method=method,
            **provenance,
        ),
        "PLANAR_SUPPORT": _property(
            bool(planar),
            status="DERIVED",
            method=method,
            **provenance,
        ),
        "measurement_quality": quality,
    }


def extract_payload_properties(
    evidence: PayloadMeasurementEvidence | None,
) -> dict[str, Any]:
    method = "payload_robust_horizontal_obb_v1"
    if evidence is None:
        return {
            "property_status": "UNKNOWN",
            "footprint_length_m": _unknown(method, "m"),
            "footprint_width_m": _unknown(method, "m"),
            "footprint_area_m2": _unknown(method, "m2"),
            "principal_orientation_world": _unknown(method),
        }
    require_payload_measurement_evidence(evidence)
    if not evidence.measurement_quality.get("quality_is_valid", False):
        return {
            "property_status": "UNKNOWN",
            "footprint_length_m": _unknown(method, "m"),
            "footprint_width_m": _unknown(method, "m"),
            "footprint_area_m2": _unknown(method, "m2"),
            "principal_orientation_world": _unknown(method),
        }
    result = _robust_horizontal_footprint(
        evidence.measurement_points, (2.0, 98.0)
    )
    if result is None:
        return {
            "property_status": "UNKNOWN",
            "footprint_length_m": _unknown(method, "m"),
            "footprint_width_m": _unknown(method, "m"),
            "footprint_area_m2": _unknown(method, "m2"),
            "principal_orientation_world": _unknown(method),
        }
    provenance = {
        "source_stage": evidence.source_stage,
        "measurement_cloud_path": evidence.measurement_cloud_path,
        "contributing_camera_ids": list(evidence.contributing_camera_ids),
        "point_count": len(evidence.measurement_points),
        "extractor_version": method,
    }
    return {
        "property_status": "MEASURED",
        "footprint_length_m": _property(
            result["length_m"], status="MEASURED", method=method,
            unit="m", **provenance
        ),
        "footprint_width_m": _property(
            result["width_m"], status="MEASURED", method=method,
            unit="m", **provenance
        ),
        "footprint_area_m2": _property(
            result["area_m2"], status="DERIVED", method=method,
            unit="m2", **provenance
        ),
        "principal_orientation_world": _property(
            result["principal_axis_world"].tolist(),
            status="MEASURED",
            method=method,
            **provenance,
        ),
        "measurement_quality": evidence.measurement_quality,
    }


def evaluate_fits_on(
    payload: dict[str, Any],
    region: dict[str, Any],
    *,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    method = REGION_RELATION_VERSION
    try:
        payload_length = float(payload["footprint_length_m"]["value"])
        payload_width = float(payload["footprint_width_m"]["value"])
        region_length = float(region["support_length_m"]["value"])
        region_width = float(region["support_width_m"]["value"])
    except (KeyError, TypeError, ValueError):
        return {
            "relation": "FITS_ON",
            "status": "UNKNOWN",
            "value": None,
            "method": method,
            "reason": "MISSING_MEASUREMENT",
        }
    config = task_config["geometric_requirements"]["payload_region"]
    clearance = float(config["edge_clearance_margin_m"])
    tested = []
    for orientation in config["allowed_orientations_degrees"]:
        rotated = int(orientation) % 180 == 90
        candidate_length = payload_width if rotated else payload_length
        candidate_width = payload_length if rotated else payload_width
        margin_length = region_length - candidate_length - 2.0 * clearance
        margin_width = region_width - candidate_width - 2.0 * clearance
        tested.append(
            {
                "orientation_degrees": int(orientation),
                "length_margin_m": margin_length,
                "width_margin_m": margin_width,
                "signed_fit_margin_m": min(margin_length, margin_width),
                "fits": margin_length >= 0.0 and margin_width >= 0.0,
            }
        )
    selected = max(tested, key=lambda item: item["signed_fit_margin_m"])
    return {
        "relation": "FITS_ON",
        "status": "TRUE" if selected["fits"] else "FALSE",
        "value": bool(selected["fits"]),
        "method": method,
        "payload_length_m": payload_length,
        "payload_width_m": payload_width,
        "region_usable_length_m": region_length,
        "region_usable_width_m": region_width,
        "edge_clearance_margin_m": clearance,
        "tested_orientations": tested,
        "selected_orientation_degrees": selected["orientation_degrees"],
        "signed_fit_margin_m": selected["signed_fit_margin_m"],
    }


def evaluate_near_seating_area(
    region: dict[str, Any],
    sofa_points: np.ndarray,
    *,
    task_config: dict[str, Any],
) -> dict[str, Any]:
    method = REGION_RELATION_VERSION
    centroid_value = region.get("centroid_world_m", {}).get("value")
    sofa = np.asarray(sofa_points, np.float64)
    sofa = sofa[np.all(np.isfinite(sofa), axis=1)]
    if centroid_value is None or len(sofa) < 20:
        return {
            "relation": "NEAR_SEATING_AREA",
            "status": "UNKNOWN",
            "value": None,
            "method": method,
            "reason": "MISSING_OBSERVED_CENTROID",
        }
    region_centroid = np.asarray(centroid_value, float)
    sofa_centroid = np.median(sofa, axis=0)
    distance = float(np.linalg.norm(region_centroid[:2] - sofa_centroid[:2]))
    threshold = float(
        task_config["geometric_requirements"]["context"][
            "maximum_centroid_distance_m"
        ]
    )
    margin = threshold - distance
    return {
        "relation": "NEAR_SEATING_AREA",
        "status": "TRUE" if margin >= 0.0 else "FALSE",
        "value": margin >= 0.0,
        "method": method,
        "region_centroid_world_m": region_centroid.tolist(),
        "seating_centroid_world_m": sofa_centroid.tolist(),
        "measured_distance_m": distance,
        "maximum_distance_m": threshold,
        "signed_margin_m": margin,
        "point_count": len(sofa),
    }


def _semantic_overlap_score(
    detection: Detection,
    mask: np.ndarray,
) -> dict[str, float]:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return {
            "intersection_pixels": 0,
            "mask_fraction": 0.0,
            "box_fraction": 0.0,
            "score": 0.0,
        }
    x1, y1, x2, y2 = detection.bbox_xyxy
    left = max(0, int(math.floor(x1)))
    right = min(mask.shape[1], int(math.ceil(x2)))
    top = max(0, int(math.floor(y1)))
    bottom = min(mask.shape[0], int(math.ceil(y2)))
    intersection = (
        int(np.count_nonzero(mask[top:bottom, left:right]))
        if right > left and bottom > top
        else 0
    )
    mask_area = int(np.count_nonzero(mask))
    box_area = max((right - left) * (bottom - top), 1)
    mask_fraction = intersection / max(mask_area, 1)
    box_fraction = intersection / box_area
    score = 0.65 * mask_fraction + 0.35 * box_fraction
    return {
        "intersection_pixels": intersection,
        "mask_fraction": mask_fraction,
        "box_fraction": box_fraction,
        "score": score,
    }


def _fuse_region_semantics(
    observations: list[dict[str, Any]],
    *,
    minimum_views: int,
    minimum_confidence: float,
    minimum_margin: float,
) -> dict[str, Any]:
    per_camera_label: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in observations:
        detection = observation["detection"]
        key = (detection["source_camera"], detection["canonical_label"])
        current = per_camera_label.get(key)
        if current is None or observation["weighted_score"] > current[
            "weighted_score"
        ]:
            per_camera_label[key] = observation
    alternatives = []
    for label in sorted({key[1] for key in per_camera_label}):
        supporting = [
            value
            for key, value in per_camera_label.items()
            if key[1] == label
        ]
        alternatives.append(
            {
                "label": label,
                "supporting_view_count": len(supporting),
                "mean_confidence": float(
                    np.mean(
                        [item["detection"]["confidence"] for item in supporting]
                    )
                ),
                "weighted_score": float(
                    sum(item["weighted_score"] for item in supporting)
                ),
                "camera_ids": sorted(
                    item["detection"]["source_camera"] for item in supporting
                ),
            }
        )
    alternatives.sort(
        key=lambda item: (
            -item["weighted_score"],
            -item["supporting_view_count"],
            item["label"],
        )
    )
    winner = alternatives[0] if alternatives else None
    runner_score = alternatives[1]["weighted_score"] if len(alternatives) > 1 else 0.0
    margin = (
        winner["weighted_score"] - runner_score if winner is not None else 0.0
    )
    supported = bool(
        winner
        and winner["supporting_view_count"] >= minimum_views
        and winner["mean_confidence"] >= minimum_confidence
        and margin >= minimum_margin
    )
    return {
        "status": "SUPPORTED" if supported else "UNKNOWN",
        "canonical_label": winner["label"] if supported else None,
        "confidence": winner["mean_confidence"] if winner else None,
        "supporting_view_count": (
            winner["supporting_view_count"] if winner else 0
        ),
        "weighted_score_margin": margin,
        "alternatives": alternatives,
    }


def run_region_semantics(
    stage_capture: RegionStageCapture,
    *,
    detector: SemanticDetector,
    semantic_config: dict[str, Any],
    task_config: dict[str, Any],
    stage_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Associate RGB detections with region/payload/seating masks only."""
    vocabulary = detector_vocabulary(semantic_config)
    semantic_requirements = task_config["semantic_requirements"]
    allowed_labels = {
        "region_parent": set(
            semantic_requirements["accepted_parent_categories"]
        )
        | set(semantic_requirements["rejected_parent_categories"]),
        "payload": set(semantic_requirements["payload_categories"]),
        "seating": set(semantic_requirements["seating_categories"]),
    }
    minimum_overlap_score = {
        "region_parent": 0.12,
        "payload": 0.12,
        # Seating boxes naturally include the open space around the sofa and
        # therefore have lower projected-mask box fractions than compact
        # furniture. Multi-view support still guards acceptance.
        "seating": 0.05,
    }
    observations = {"region_parent": [], "payload": [], "seating": []}
    overlay_payloads: list[
        tuple[str, np.ndarray, list[tuple[str, Detection]]]
    ] = []
    detections_payload = []
    associations_payload = []
    for camera_id, capture in stage_capture.cameras.items():
        if not capture.validation.get("usable", False):
            continue
        raw = detector.detect(capture.rgb, vocabulary)
        detections = [
            canonicalize_detection(detection, semantic_config)
            for detection in raw
        ]
        accepted_for_overlay = []
        for target, mask in (
            ("region_parent", capture.region_mask),
            ("payload", capture.payload_mask),
            ("seating", capture.sofa_mask),
        ):
            candidates = []
            for detection in detections:
                if detection.canonical_label not in allowed_labels[target]:
                    continue
                metrics = _semantic_overlap_score(detection, mask)
                if (
                    metrics["intersection_pixels"] < 8
                    or metrics["score"] < minimum_overlap_score[target]
                ):
                    continue
                weighted = float(detection.confidence) * metrics["score"]
                candidates.append((weighted, detection, metrics))
            if not candidates:
                continue
            weighted, detection, metrics = max(
                candidates,
                key=lambda item: (
                    item[0],
                    item[1].confidence,
                    item[1].canonical_label,
                ),
            )
            observation = {
                "target": target,
                "detection": {
                    **detection.to_dict(),
                    "source_camera": camera_id,
                },
                "association_metrics": metrics,
                "weighted_score": weighted,
            }
            observations[target].append(observation)
            accepted_for_overlay.append((target, detection))
            associations_payload.append(
                {
                    "camera_id": camera_id,
                    "target": target,
                    "status": "ACCEPTED",
                    "detection": observation["detection"],
                    "metrics": metrics,
                }
            )
        detections_payload.extend(
            {"camera_id": camera_id, **detection.to_dict()}
            for detection in detections
        )
        image = Image.fromarray(capture.rgb)
        draw = ImageDraw.Draw(image)
        colors = {
            "region_parent": (29, 151, 76),
            "payload": (221, 96, 35),
            "seating": (48, 105, 190),
        }
        for target, detection in accepted_for_overlay:
            color = colors[target]
            draw.rectangle(detection.bbox_xyxy, outline=color, width=4)
            x1, y1, _x2, _y2 = detection.bbox_xyxy
            text = (
                f"{target}: {detection.raw_label} "
                f"{detection.confidence:.2f}"
            )
            draw.text((x1 + 3, max(2, y1 - 18)), text, fill=color, font=_font(14, True))
        overlay_path = (
            stage_dir / "semantics" / "cameras" / camera_id / "overlay.png"
        )
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(overlay_path)
        overlay_payloads.append(
            (camera_id, capture.rgb, accepted_for_overlay)
        )

    records = {}
    for target in observations:
        records[target] = _fuse_region_semantics(
            observations[target],
            minimum_views=int(
                semantic_requirements["minimum_supporting_views"]
            ),
            minimum_confidence=float(
                semantic_requirements.get(
                    "minimum_payload_mean_confidence",
                    semantic_requirements["minimum_mean_confidence"],
                )
                if target == "payload"
                else semantic_requirements["minimum_mean_confidence"]
            ),
            minimum_margin=float(
                semantic_requirements["minimum_winning_score_margin"]
            ),
        )
        records[target]["observations"] = observations[target]
    _atomic_json(
        stage_dir / "semantics" / "detections.json",
        {"detections": detections_payload},
    )
    _atomic_json(
        stage_dir / "semantics" / "associations.json",
        {"associations": associations_payload},
    )
    _atomic_json(stage_dir / "semantics" / "fused_semantics.json", records)
    overlay_views: list[tuple[str, Image.Image]] = []
    colors = {
        "region_parent": (29, 151, 76),
        "payload": (221, 96, 35),
        "seating": (48, 105, 190),
    }
    for camera_id, rgb, accepted_for_overlay in overlay_payloads:
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for target, detection in accepted_for_overlay:
            record = records[target]
            if (
                record.get("status") != "SUPPORTED"
                or detection.canonical_label
                != record.get("canonical_label")
            ):
                continue
            color = colors[target]
            draw.rectangle(detection.bbox_xyxy, outline=color, width=4)
            x1, y1, _x2, _y2 = detection.bbox_xyxy
            text = (
                f"{target}: {detection.raw_label} "
                f"{detection.confidence:.2f}"
            )
            draw.text(
                (x1 + 3, max(2, y1 - 18)),
                text,
                fill=color,
                font=_font(14, True),
            )
        consensus_path = (
            stage_dir
            / "semantics"
            / "cameras"
            / camera_id
            / "consensus_overlay.png"
        )
        image.save(consensus_path)
        overlay_views.append((camera_id, image))

    if overlay_views:
        width = max(image.width for _name, image in overlay_views)
        height = max(image.height for _name, image in overlay_views)
        canvas = Image.new("RGB", (2 * width, 3 * (height + 26)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (name, image) in enumerate(overlay_views):
            x = (index % 2) * width
            y = (index // 2) * (height + 26)
            draw.text((x + 5, y + 4), name, fill="black", font=_font(16, True))
            canvas.paste(image, (x, y + 26))
        canvas.save(stage_dir / "semantic_overview.png")
    stage_capture.semantic_observations = observations
    stage_capture.semantic_records = records
    return records


def semantic_region_role_status(
    semantic_record: dict[str, Any],
    task_config: dict[str, Any],
) -> dict[str, Any]:
    if semantic_record.get("status") != "SUPPORTED":
        return {
            "status": "UNKNOWN",
            "value": None,
            "reason": "INSUFFICIENT_PARENT_SEMANTICS",
        }
    label = semantic_record["canonical_label"]
    accepted = task_config["semantic_requirements"][
        "accepted_parent_categories"
    ]
    rejected = set(
        task_config["semantic_requirements"]["rejected_parent_categories"]
    )
    if label in accepted:
        return {
            "status": "TRUE",
            "value": True,
            "canonical_label": label,
            "semantic_rank": int(accepted[label]),
        }
    if label in rejected:
        return {
            "status": "FALSE",
            "value": False,
            "canonical_label": label,
            "reason": "EXCLUDED_PARENT_CATEGORY",
        }
    return {
        "status": "UNKNOWN",
        "value": None,
        "canonical_label": label,
        "reason": "UNCONFIGURED_PARENT_CATEGORY",
    }


def _depth_visual(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0)
    output = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        scaled = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
        output[..., 0] = np.uint8(255 * (1 - scaled))
        output[..., 1] = np.uint8(180 * scaled)
        output[..., 2] = np.uint8(255 * scaled)
        output[~valid] = 0
    return Image.fromarray(output)


def _segmentation_visual(segmentation: np.ndarray) -> Image.Image:
    ids = segmentation[..., 0].astype(np.int64)
    kind = segmentation[..., 1].astype(np.int64)
    output = np.zeros((*ids.shape, 3), dtype=np.uint8)
    output[..., 0] = (ids * 53 + kind * 17) % 255
    output[..., 1] = (ids * 97 + kind * 31) % 255
    output[..., 2] = (ids * 193 + kind * 47) % 255
    output[kind < 0] = 0
    return Image.fromarray(output)


class L2RegionEvidenceCapture:
    """Render a fresh calibrated candidate-region observation."""

    def __init__(
        self,
        scene,
        *,
        rig_config: str | Path,
        width: int,
        height: int,
    ):
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.width = width
        self.height = height
        with Path(rig_config).open(encoding="utf-8") as source:
            self.config = yaml.safe_load(source)

    def capture(
        self,
        inspection_label: str,
        *,
        stage: int,
        stage_dir: str | Path,
    ) -> RegionStageCapture:
        started = time.perf_counter()
        stage_dir = Path(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=False)
        region = self.config["regions"][inspection_label]
        for _ in range(int(region.get("settle_steps", 0))):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        camera_slots = self.config["camera_slots"]
        payload_geom_ids = _single_free_rigid_instance_geom_ids(self.model)
        configured = {}
        original = {}
        target_base = np.asarray(region["target_world_m"], float)
        rig_position = np.asarray(region["rig_position_world_m"], float)
        up = np.asarray(region["up_world"], float)
        for camera_id, model_name in camera_slots.items():
            model_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, model_name
            )
            if model_id < 0:
                raise ValueError(f"Missing L2 camera: {model_name}")
            camera = region["cameras"][camera_id]
            position = rig_position + np.asarray(
                camera["position_offset_m"], float
            )
            target = target_base + np.asarray(
                camera.get("look_at_offset_m", [0, 0, 0]), float
            )
            rotation = look_at_camera_rotation(position, target, up)
            quaternion = np.empty(4)
            mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
            original[camera_id] = {
                "model_id": model_id,
                "pos": self.model.cam_pos[model_id].copy(),
                "quat": self.model.cam_quat[model_id].copy(),
                "mat0": self.model.cam_mat0[model_id].copy(),
                "mode": int(self.model.cam_mode[model_id]),
                "target": int(self.model.cam_targetbodyid[model_id]),
                "fovy": float(self.model.cam_fovy[model_id]),
            }
            self.model.cam_pos[model_id] = position
            self.model.cam_quat[model_id] = quaternion
            self.model.cam_mat0[model_id] = rotation.reshape(-1)
            self.model.cam_mode[model_id] = int(
                mujoco.mjtCamLight.mjCAMLIGHT_FIXED
            )
            self.model.cam_targetbodyid[model_id] = -1
            self.model.cam_fovy[model_id] = float(camera["fovy_degrees"])
            configured[camera_id] = {
                "model_name": model_name,
                "model_id": model_id,
                "position": position,
                "target": target,
                "rotation": rotation,
            }
        mujoco.mj_forward(self.model, self.data)

        captures = {}
        all_region_points, all_region_colors = [], []
        all_payload_points, all_payload_colors = [], []
        all_sofa_points = []
        region_by_camera, payload_by_camera = {}, {}
        renderer = mujoco.Renderer(
            self.model, width=self.width, height=self.height
        )
        render_started = time.perf_counter()
        try:
            for camera_id, pose in configured.items():
                model_id = pose["model_id"]
                renderer.update_scene(self.data, camera=model_id)
                rgb = renderer.render().copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                intrinsics = camera_intrinsics(
                    float(self.model.cam_fovy[model_id]),
                    self.width,
                    self.height,
                )
                validation = validate_camera_view(
                    camera_position=pose["position"],
                    camera_rotation=pose["rotation"],
                    target_world=pose["target"],
                    intrinsics=intrinsics,
                    width=self.width,
                    height=self.height,
                    depth_m=depth,
                    near_depth_m=float(region["near_depth_m"]),
                    far_depth_m=float(region["far_depth_m"]),
                    maximum_target_angle_degrees=float(
                        self.config["view_validation"][
                            "maximum_target_angle_degrees"
                        ]
                    ),
                    minimum_valid_depth_pixels=int(
                        self.config["view_validation"][
                            "minimum_valid_depth_pixels"
                        ]
                    ),
                )
                valid_depth = (
                    np.isfinite(depth)
                    & (depth > float(region["near_depth_m"]))
                    & (depth <= float(region["far_depth_m"]))
                )
                world, pixels = backproject_masked_depth(
                    depth,
                    valid_depth,
                    intrinsics,
                    pose["position"],
                    pose["rotation"],
                    min_depth=float(region["near_depth_m"]),
                    max_depth=float(region["far_depth_m"]),
                )
                region_mask, region_points = _volume_mask_from_world_points(
                    world,
                    pixels,
                    depth.shape,
                    region["inspection_volume"],
                )
                sofa_mask, sofa_points = _volume_mask_from_world_points(
                    world,
                    pixels,
                    depth.shape,
                    self.config["sofa_context_volume"],
                )
                is_geom = segmentation[..., 1] == int(
                    mujoco.mjtObj.mjOBJ_GEOM
                )
                payload_mask = is_geom & np.isin(
                    segmentation[..., 0], payload_geom_ids
                )
                payload_points, payload_pixels = backproject_masked_depth(
                    depth,
                    payload_mask,
                    intrinsics,
                    pose["position"],
                    pose["rotation"],
                    min_depth=float(region["near_depth_m"]),
                    max_depth=float(region["far_depth_m"]),
                )
                region_pixels = np.column_stack(np.nonzero(region_mask))
                region_colors = (
                    rgb[region_pixels[:, 0], region_pixels[:, 1]]
                    if len(region_pixels)
                    else np.empty((0, 3), np.uint8)
                )
                payload_colors = (
                    rgb[payload_pixels[:, 0], payload_pixels[:, 1]]
                    if len(payload_pixels)
                    else np.empty((0, 3), np.uint8)
                )
                all_region_points.append(region_points)
                all_region_colors.append(region_colors)
                all_sofa_points.append(sofa_points)
                region_by_camera[camera_id] = region_points
                if len(payload_points):
                    all_payload_points.append(payload_points)
                    all_payload_colors.append(payload_colors)
                    payload_by_camera[camera_id] = payload_points
                captures[camera_id] = RegionCameraCapture(
                    camera_id=camera_id,
                    model_camera_name=pose["model_name"],
                    rgb=rgb,
                    depth_m=depth,
                    segmentation=segmentation,
                    intrinsics=intrinsics,
                    position_world_m=pose["position"],
                    rotation_world_from_camera=pose["rotation"],
                    validation=validation,
                    region_mask=region_mask,
                    payload_mask=payload_mask,
                    sofa_mask=sofa_mask,
                    region_points=region_points,
                    region_colors=region_colors,
                    payload_points=payload_points,
                    payload_colors=payload_colors,
                    sofa_points=sofa_points,
                )
        finally:
            renderer.close()
            for camera_id, state in original.items():
                model_id = state["model_id"]
                self.model.cam_pos[model_id] = state["pos"]
                self.model.cam_quat[model_id] = state["quat"]
                self.model.cam_mat0[model_id] = state["mat0"]
                self.model.cam_mode[model_id] = state["mode"]
                self.model.cam_targetbodyid[model_id] = state["target"]
                self.model.cam_fovy[model_id] = state["fovy"]
            mujoco.mj_forward(self.model, self.data)
        render_seconds = time.perf_counter() - render_started

        raw_region_points = np.concatenate(all_region_points)
        raw_region_colors = np.concatenate(all_region_colors)
        processing = self.config["processing"]
        plane_points, plane_colors, plane_diagnostics = (
            _select_upper_support_plane(
                raw_region_points,
                raw_region_colors,
                plane_band_m=float(processing["plane_band_m"]),
                voxel_size_m=float(processing["voxel_size_m"]),
            )
        )
        valid_cameras = tuple(
            camera_id
            for camera_id, capture in captures.items()
            if capture.validation.get("usable", False)
        )
        contributing_region = tuple(
            camera_id
            for camera_id in valid_cameras
            if len(region_by_camera.get(camera_id, ()))
        )
        region_quality = {
            "quality_is_valid": (
                len(valid_cameras)
                >= int(
                    self.config["view_validation"][
                        "minimum_valid_rig_cameras"
                    ]
                )
                and len(contributing_region)
                >= int(
                    self.config["view_validation"][
                        "minimum_region_camera_count"
                    ]
                )
                and len(plane_points)
                >= int(processing["minimum_region_points"])
            ),
            "valid_camera_count": len(valid_cameras),
            "contributing_camera_count": len(contributing_region),
            "point_count": len(plane_points),
            "cloud_purpose": REGION_MEASUREMENT_PURPOSE,
            **plane_diagnostics,
        }
        evidence_rel = (
            "region_evidence/fused.ply"
            if region_quality["quality_is_valid"]
            else "rejected_region_evidence/fused.ply"
        )
        region_evidence = RegionMeasurementEvidence(
            measurement_points=plane_points,
            measurement_colors=plane_colors,
            points_by_camera=region_by_camera,
            source_stage=stage,
            inspection_label=inspection_label,
            measurement_cloud_path=(
                f"stages/{stage_dir.name}/{evidence_rel}"
            ),
            contributing_camera_ids=contributing_region,
            measurement_quality=region_quality,
        )
        payload_evidence = None
        if all_payload_points:
            payload_points, payload_colors = voxel_downsample(
                np.concatenate(all_payload_points),
                np.concatenate(all_payload_colors),
                float(processing["voxel_size_m"]),
            )
            contributing_payload = tuple(sorted(payload_by_camera))
            payload_quality = {
                "quality_is_valid": (
                    len(payload_points)
                    >= int(processing["minimum_payload_points"])
                    and len(contributing_payload)
                    >= int(
                        self.config["view_validation"][
                            "minimum_payload_camera_count"
                        ]
                    )
                ),
                "point_count": len(payload_points),
                "contributing_camera_count": len(contributing_payload),
                "cloud_purpose": "PAYLOAD_MEASUREMENT_EVIDENCE",
            }
            payload_evidence = PayloadMeasurementEvidence(
                measurement_points=payload_points,
                measurement_colors=payload_colors,
                points_by_camera=payload_by_camera,
                source_stage=stage,
                measurement_cloud_path=(
                    f"stages/{stage_dir.name}/payload_evidence/fused.ply"
                ),
                contributing_camera_ids=contributing_payload,
                measurement_quality=payload_quality,
            )
        sofa_points = (
            np.concatenate(all_sofa_points)
            if any(len(points) for points in all_sofa_points)
            else np.empty((0, 3), np.float32)
        )
        sofa_colors = np.full((len(sofa_points), 3), (55, 105, 180), np.uint8)
        sofa_points, sofa_colors = voxel_downsample(
            sofa_points, sofa_colors, float(processing["voxel_size_m"])
        )

        write_ply(stage_dir / evidence_rel, plane_points, plane_colors)
        _atomic_json(
            (stage_dir / evidence_rel).parent / "quality.json",
            region_quality,
        )
        if payload_evidence is not None:
            write_ply(
                stage_dir / "payload_evidence" / "fused.ply",
                payload_evidence.measurement_points,
                payload_evidence.measurement_colors,
            )
        write_ply(
            stage_dir / "seating_context" / "observed_points.ply",
            sofa_points,
            sofa_colors,
        )
        for camera_id, capture in captures.items():
            camera_dir = stage_dir / "cameras" / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(capture.rgb).save(camera_dir / "rgb.png")
            _depth_visual(capture.depth_m).save(camera_dir / "depth.png")
            _segmentation_visual(capture.segmentation).save(
                camera_dir / "segmentation.png"
            )
            mask_image = np.zeros((*capture.region_mask.shape, 3), np.uint8)
            mask_image[capture.region_mask] = (45, 190, 90)
            mask_image[capture.payload_mask] = (235, 110, 35)
            mask_image[capture.sofa_mask] = (50, 110, 210)
            Image.fromarray(mask_image).save(camera_dir / "evidence_masks.png")
            write_ply(
                camera_dir / "region_cloud.ply",
                capture.region_points,
                capture.region_colors,
            )
        metadata = {
            "stage": stage,
            "configured_inspection_label": inspection_label,
            "rig_position_world_m": rig_position.tolist(),
            "target_world_m": target_base.tolist(),
            "camera_poses": {
                camera_id: {
                    "model_camera_name": capture.model_camera_name,
                    "position_world_m": capture.position_world_m.tolist(),
                    "rotation_world_from_camera": (
                        capture.rotation_world_from_camera.tolist()
                    ),
                    "intrinsics": capture.intrinsics.tolist(),
                    "validation": capture.validation,
                }
                for camera_id, capture in captures.items()
            },
            "capture_resolution": [self.width, self.height],
            "inspection_volume": {
                **region["inspection_volume"],
                "purpose": "EVIDENCE_SELECTION_ONLY_NOT_MEASUREMENT",
            },
            "valid_cameras": list(valid_cameras),
            "region_quality": region_quality,
            "payload_quality": (
                payload_evidence.measurement_quality
                if payload_evidence is not None
                else {"quality_is_valid": False}
            ),
        }
        _atomic_json(stage_dir / "inspection_metadata.json", metadata)
        _atomic_json(stage_dir / "inspection_quality.json", {
            "region": region_quality,
            "payload": (
                payload_evidence.measurement_quality
                if payload_evidence is not None
                else {"quality_is_valid": False}
            ),
        })
        return RegionStageCapture(
            stage=stage,
            inspection_label=inspection_label,
            cameras=captures,
            region_evidence=region_evidence,
            payload_evidence=payload_evidence,
            sofa_points=sofa_points,
            timings_seconds={
                "capture_and_reconstruction": render_seconds,
                "total": time.perf_counter() - started,
            },
        )


def load_region_task(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if config.get("natural_language_goal") is None:
        raise ValueError("Region task requires natural_language_goal")
    return config
