"""Category-independent geometric properties and compatibility relations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mujoco_scenes.geometry_checker import (
    MEASUREMENT_EVIDENCE_PURPOSE,
    MeasurementEvidence,
)

CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "geometry_inference.yaml"
)
SUPPORTED_STATUSES = {"MEASURED", "DERIVED", "UNKNOWN"}
EXTRACTOR_VERSION = "geometry_properties_v3"
OPEN_RECEPTACLE_EXTRACTOR_VERSION = "open_cavity_structure_v3"
GEOMETRIC_PROPERTY_KEYS = (
    "total_length_m",
    "usable_length_m",
    "maximum_cross_section_m",
    "elongation_ratio",
    "flatness_ratio",
    "dominant_plane_normal_world",
    "planarity_score",
    "support_length_m",
    "support_width_m",
    "support_thickness_m",
    "support_area_m2",
    "opening_width_m",
    "opening_length_m",
    "cavity_depth_m",
)
GEOMETRIC_PREDICATE_KEYS = (
    "OPEN_CAVITY",
    "ELONGATED_OBJECT",
    "PLANAR_SUPPORT",
)


def load_geometry_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load category-free extraction thresholds and relation margins."""
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Geometry configuration must be a mapping")
    forbidden = {
        "families",
        "category",
        "categories",
        "category_functions",
        "candidate_function",
        "object_family",
    }

    def find_keys(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden:
                    found.add(key)
                found.update(find_keys(child))
        elif isinstance(value, list):
            for child in value:
                found.update(find_keys(child))
        return found

    present = find_keys(config)
    if present:
        raise ValueError(
            "Geometry configuration cannot contain category mappings: "
            + ", ".join(sorted(present))
        )
    return config


# Compatibility alias for callers written before the geometry-only milestone.
load_semantics_config = load_geometry_config


def property_value(
    value: Any,
    *,
    unit: str | None,
    status: str,
    method: str,
) -> dict[str, Any]:
    """Create one explicitly-provenanced serializable property value."""
    if status not in SUPPORTED_STATUSES:
        raise ValueError(f"Unsupported property status: {status}")
    if value is None:
        status = "UNKNOWN"
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    return {
        "value": value,
        "unit": unit,
        "status": status,
        "method": method,
    }


def unknown_property(unit: str | None, method: str) -> dict[str, Any]:
    return property_value(None, unit=unit, status="UNKNOWN", method=method)


def geometric_predicate(
    status: str,
    *,
    method: str,
    evidence: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if status not in {"TRUE", "FALSE", "UNKNOWN"}:
        raise ValueError(f"Unsupported geometric predicate status: {status}")
    record = {
        "value": (
            True
            if status == "TRUE"
            else False
            if status == "FALSE"
            else None
        ),
        "status": status,
        "method": method,
        "evidence": evidence or {},
    }
    if reason is not None:
        record["reason"] = reason
    return record


def _unknown_geometry() -> dict[str, Any]:
    return {
        "centroid_world_m": unknown_property("m", "robust_pca_obb"),
        "dimensions_m": {
            key: unknown_property("m", "robust_pca_obb")
            for key in ("length", "width", "height")
        },
        "principal_axis_world": unknown_property(
            "unit_vector", "robust_pca_obb"
        ),
        "property_status": "UNKNOWN",
    }


def _finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    return points[np.all(np.isfinite(points), axis=1)]


def _robust_pca_arrays(
    points: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    extraction = config.get("property_extraction", {})
    minimum_points = int(extraction.get("minimum_point_count", 8))
    minimum_extent = float(extraction.get("minimum_extent_m", 0.001))
    lower = float(extraction.get("robust_lower_percentile", 2.0))
    upper = float(extraction.get("robust_upper_percentile", 98.0))
    valid = _finite_points(points)
    if len(valid) < minimum_points:
        return None

    coordinate_lower = np.percentile(valid, lower, axis=0)
    coordinate_upper = np.percentile(valid, upper, axis=0)
    inlier_mask = np.all(
        (valid >= coordinate_lower) & (valid <= coordinate_upper), axis=1
    )
    inliers = valid[inlier_mask]
    if len(inliers) < minimum_points:
        return None

    centre = np.median(inliers, axis=0)
    centred = inliers - centre
    covariance = np.cov(centred, rowvar=False)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    axes = eigenvectors[:, order]
    projected = centred @ axes
    robust_min = np.percentile(projected, lower, axis=0)
    robust_max = np.percentile(projected, upper, axis=0)
    extents = robust_max - robust_min
    if (
        not np.all(np.isfinite(extents))
        or float(np.max(extents)) < minimum_extent
    ):
        return None
    extent_order = np.argsort(extents)[::-1]
    extents = extents[extent_order]
    axes = axes[:, extent_order]
    eigenvalues = eigenvalues[extent_order]
    for column in range(3):
        axis = axes[:, column]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axes[:, column] = -axis
    return {
        "valid": valid,
        "inliers": inliers,
        "centre": centre,
        "axes": axes,
        "eigenvalues": eigenvalues,
        "extents": extents,
    }


def robust_pca_geometry(
    points: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Measure robust orientation-independent dimensions from visible points."""
    measured = _robust_pca_arrays(points, config)
    if measured is None:
        return _unknown_geometry()
    length, width, height = measured["extents"]
    return {
        "centroid_world_m": property_value(
            measured["centre"],
            unit="m",
            status="MEASURED",
            method="finite_point_coordinate_median",
        ),
        "dimensions_m": {
            key: property_value(
                float(value),
                unit="m",
                status="MEASURED",
                method="robust_pca_obb",
            )
            for key, value in zip(
                ("length", "width", "height"),
                (length, width, height),
            )
        },
        "principal_axis_world": property_value(
            measured["axes"][:, 0],
            unit="unit_vector",
            status="MEASURED",
            method="robust_pca_obb",
        ),
        "property_status": "MEASURED",
    }


def _property_template() -> dict[str, dict[str, Any]]:
    units = {
        "total_length_m": "m",
        "usable_length_m": "m",
        "maximum_cross_section_m": "m",
        "elongation_ratio": "ratio",
        "flatness_ratio": "ratio",
        "dominant_plane_normal_world": "unit_vector",
        "planarity_score": "ratio",
        "support_length_m": "m",
        "support_width_m": "m",
        "support_thickness_m": "m",
        "support_area_m2": "m2",
        "opening_width_m": "m",
        "opening_length_m": "m",
        "cavity_depth_m": "m",
    }
    return {
        key: unknown_property(units[key], "insufficient_visible_geometry")
        for key in GEOMETRIC_PROPERTY_KEYS
    }


def _predicate_template() -> dict[str, dict[str, Any]]:
    return {
        key: geometric_predicate(
            "UNKNOWN",
            method="insufficient_visible_geometry",
            reason="INSUFFICIENT_VISIBLE_GEOMETRY",
        )
        for key in GEOMETRIC_PREDICATE_KEYS
    }


def _estimate_open_cavity(
    measurement_evidence: MeasurementEvidence,
    config: dict[str, Any],
    measured_geometry: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Conservatively validate a gravity-aligned rim and observed interior."""
    method = OPEN_RECEPTACLE_EXTRACTOR_VERSION
    cavity = config.get("open_cavity", {})
    valid = _finite_points(measurement_evidence.measurement_points)
    unknown_properties = {
        key: unknown_property("m", method)
        for key in (
            "opening_width_m",
            "opening_length_m",
            "cavity_depth_m",
        )
    }
    diagnostics: dict[str, Any] = {
        "point_count": len(valid),
        "contributing_camera_count": len(
            measurement_evidence.contributing_camera_ids
        ),
        "definition": (
            "enclosed_rim AND open_centre AND observed_interior_below_rim"
        ),
        "structural_components": {
            "ENCLOSED_RIM": "UNKNOWN",
            "OPEN_CENTRE": "UNKNOWN",
            "INTERIOR_BELOW_RIM": "UNKNOWN",
        },
    }
    minimum_points = int(cavity.get("minimum_point_count", 120))
    minimum_cameras = int(
        cavity.get("minimum_contributing_camera_count", 2)
    )
    if len(valid) < minimum_points:
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INSUFFICIENT_POINT_COUNT",
        )
    if len(measurement_evidence.contributing_camera_ids) < minimum_cameras:
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INSUFFICIENT_CAMERA_COVERAGE",
        )
    if measured_geometry is None:
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INVALID_OBJECT_GEOMETRY",
        )

    vertical_lower, vertical_upper = np.percentile(
        valid[:, 2], (2.0, 98.0)
    )
    vertical_extent = float(vertical_upper - vertical_lower)
    diagnostics["vertical_extent_m"] = vertical_extent
    eigenvalues = measured_geometry["eigenvalues"]
    planarity_score = float(
        1.0
        - min(1.0, eigenvalues[2] / max(eigenvalues[1], 1e-12))
    )
    diagnostics["planarity_score"] = planarity_score
    if (
        planarity_score
        >= float(cavity.get("planar_rejection_score", 0.94))
        and vertical_extent
        <= float(cavity.get("planar_rejection_max_extent_m", 0.030))
    ):
        return unknown_properties, geometric_predicate(
            "FALSE",
            method=method,
            evidence=diagnostics,
            reason="PREDOMINANTLY_PLANAR_EVIDENCE",
        )

    minimum_vertical_extent = float(
        cavity.get(
            "minimum_resolvable_vertical_extent_m",
            cavity.get("minimum_vertical_extent_m", 0.006),
        )
    )
    diagnostics[
        "minimum_resolvable_vertical_extent_m"
    ] = minimum_vertical_extent
    if vertical_extent < minimum_vertical_extent:
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="UNRESOLVED_VERTICAL_STRUCTURE",
        )

    rim_percentile = float(cavity.get("rim_percentile_z", 86.0))
    rim_threshold = float(np.percentile(valid[:, 2], rim_percentile))
    top_mask = valid[:, 2] >= rim_threshold
    top = valid[top_mask]

    # Estimate the object's visible outer XY footprint first, then select the
    # high, outer-shell samples as rim candidates. This avoids treating a
    # shallow bowl's visible sloping interior as a filled top surface.
    centre_xy = np.median(valid[:, :2], axis=0)
    centred_xy = valid[:, :2] - centre_xy
    covariance = np.cov(centred_xy, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INVALID_RIM_PROJECTION",
        )
    _values, axes = np.linalg.eigh(covariance)
    axes = axes[:, ::-1]
    all_projected = centred_xy @ axes
    full_lower = np.percentile(all_projected, 2.0, axis=0)
    full_upper = np.percentile(all_projected, 98.0, axis=0)
    footprint_extents = np.maximum(full_upper - full_lower, 1e-6)
    footprint_half_extents = 0.5 * footprint_extents
    all_normalized = all_projected / footprint_half_extents
    normalized_radius = np.linalg.norm(all_normalized, axis=1)
    shell_radius = float(cavity.get("rim_shell_minimum_radius", 0.70))
    rim_mask = top_mask & (normalized_radius >= shell_radius)
    rim = valid[rim_mask]
    diagnostics["rim_point_count"] = len(rim)
    diagnostics["top_candidate_point_count"] = len(top)
    minimum_rim = int(cavity.get("minimum_rim_points", 24))
    if len(rim) < minimum_rim:
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INSUFFICIENT_RIM_POINTS",
        )

    centred_rim_3d = rim - np.median(rim, axis=0)
    rim_covariance = np.cov(centred_rim_3d, rowvar=False)
    if (
        rim_covariance.shape != (3, 3)
        or not np.all(np.isfinite(rim_covariance))
    ):
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INVALID_RIM_COVARIANCE",
        )
    _rim_values, rim_vectors = np.linalg.eigh(rim_covariance)
    rim_normal = rim_vectors[:, 0]
    if rim_normal[2] < 0.0:
        rim_normal = -rim_normal
    rim_normal_alignment = float(abs(rim_normal[2]))
    diagnostics["rim_normal_alignment"] = rim_normal_alignment
    diagnostics["rim_normal_world"] = rim_normal.tolist()
    minimum_alignment = float(
        cavity.get("minimum_rim_normal_alignment", 0.88)
    )
    if rim_normal_alignment < minimum_alignment:
        return unknown_properties, geometric_predicate(
            "FALSE",
            method=method,
            evidence=diagnostics,
            reason="RIM_NOT_GRAVITY_ALIGNED",
        )

    projected_rim = all_projected[rim_mask]
    lower = np.percentile(projected_rim, 2.0, axis=0)
    upper = np.percentile(projected_rim, 98.0, axis=0)
    opening_extents = np.sort(upper - lower)[::-1]
    opening_length = float(opening_extents[0])
    opening_width = float(opening_extents[1])
    diagnostics["estimated_opening_length_m"] = opening_length
    diagnostics["estimated_opening_width_m"] = opening_width
    minimum_opening = float(
        cavity.get(
            "minimum_resolvable_opening_m",
            cavity.get("minimum_opening_m", 0.009),
        )
    )
    diagnostics["minimum_resolvable_opening_m"] = minimum_opening
    if not np.isfinite(opening_width) or opening_width < minimum_opening:
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="UNRESOLVED_OPENING",
        )

    normalized_rim = all_normalized[rim_mask]
    angles = np.arctan2(normalized_rim[:, 1], normalized_rim[:, 0])
    bin_count = int(cavity.get("angular_bin_count", 12))
    bins = np.floor((angles + np.pi) * bin_count / (2.0 * np.pi))
    occupied_bins = len(
        np.unique(np.clip(bins.astype(int), 0, bin_count - 1))
    )
    enclosure_ratio = float(occupied_bins / max(bin_count, 1))
    shell_fraction = float(len(rim) / max(len(top), 1))
    central_top_fraction = float(
        np.mean(
            normalized_radius[top_mask]
            < float(cavity.get("central_top_radius_fraction", 0.52))
        )
    )
    central_top_point_count = int(
        np.count_nonzero(
            normalized_radius[top_mask]
            < float(cavity.get("central_top_radius_fraction", 0.52))
        )
    )
    top_height_spread = float(
        np.percentile(top[:, 2], 95.0)
        - np.percentile(top[:, 2], 5.0)
    )
    diagnostics.update(
        {
            "occupied_angular_bins": occupied_bins,
            "angular_bin_count": bin_count,
            "rim_enclosure_ratio": enclosure_ratio,
            "rim_shell_fraction": shell_fraction,
            "central_top_occupancy_fraction": central_top_fraction,
            "central_top_point_count": central_top_point_count,
            "top_height_spread_m": top_height_spread,
        }
    )
    maximum_central_top = float(
        cavity.get("maximum_central_top_occupancy_fraction", 0.30)
    )
    filled_coplanar_top = (
        central_top_point_count
        >= int(cavity.get("minimum_filled_top_central_points", 20))
        and top_height_spread
        <= float(cavity.get("maximum_filled_top_height_spread_m", 0.002))
    )
    if central_top_fraction > maximum_central_top or filled_coplanar_top:
        diagnostics["structural_components"]["OPEN_CENTRE"] = "FALSE"
        return unknown_properties, geometric_predicate(
            "FALSE",
            method=method,
            evidence=diagnostics,
            reason="FILLED_TOP_SURFACE",
        )
    diagnostics["structural_components"]["OPEN_CENTRE"] = "TRUE"
    if (
        enclosure_ratio
        < float(cavity.get("minimum_rim_enclosure_ratio", 0.75))
        or shell_fraction
        < float(cavity.get("minimum_rim_shell_fraction", 0.72))
    ):
        return unknown_properties, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INSUFFICIENT_RIM_ENCLOSURE",
        )
    diagnostics["structural_components"]["ENCLOSED_RIM"] = "TRUE"

    minimum_depth = float(
        cavity.get(
            "minimum_resolvable_depth_m",
            cavity.get("minimum_depth_m", 0.006),
        )
    )
    diagnostics["minimum_resolvable_depth_m"] = minimum_depth
    inner_fraction = float(cavity.get("inner_radius_fraction", 0.62))
    rim_height = float(np.median(rim[:, 2]))
    interior_mask = (
        (normalized_radius < inner_fraction)
        & (valid[:, 2] <= rim_height - minimum_depth)
    )
    interior = valid[interior_mask]
    minimum_interior = int(cavity.get("minimum_interior_points", 16))
    interior_camera_ids: list[str] = []
    for camera_id, camera_points in (
        measurement_evidence.points_by_camera.items()
    ):
        camera_valid = _finite_points(camera_points)
        if not len(camera_valid):
            continue
        camera_projected = (camera_valid[:, :2] - centre_xy) @ axes
        camera_radius = np.linalg.norm(
            camera_projected / footprint_half_extents, axis=1
        )
        camera_interior = (
            (camera_radius < inner_fraction)
            & (camera_valid[:, 2] <= rim_height - minimum_depth)
        )
        if np.count_nonzero(camera_interior) >= int(
            cavity.get("minimum_interior_points_per_camera", 3)
        ):
            interior_camera_ids.append(camera_id)
    diagnostics["interior_point_count"] = len(interior)
    diagnostics["interior_camera_ids"] = interior_camera_ids
    diagnostics["interior_camera_count"] = len(interior_camera_ids)
    opening_records = {
        "opening_width_m": property_value(
            opening_width,
            unit="m",
            status="MEASURED",
            method=method,
        ),
        "opening_length_m": property_value(
            opening_length,
            unit="m",
            status="MEASURED",
            method=method,
        ),
        "cavity_depth_m": unknown_properties["cavity_depth_m"],
    }
    if (
        len(interior) < minimum_interior
        or len(interior_camera_ids)
        < int(cavity.get("minimum_interior_camera_count", 2))
    ):
        return opening_records, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="INSUFFICIENT_OBSERVED_INTERIOR",
        )
    diagnostics["structural_components"][
        "INTERIOR_BELOW_RIM"
    ] = "TRUE"

    cavity_depth = float(
        rim_height - np.percentile(interior[:, 2], 10.0)
    )
    diagnostics["estimated_cavity_depth_m"] = cavity_depth
    if not np.isfinite(cavity_depth) or cavity_depth < minimum_depth:
        return opening_records, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="UNRESOLVED_INTERIOR_DEPTH",
        )
    removed_outliers = int(
        measurement_evidence.measurement_quality.get(
            "outlier_points_removed", 0
        )
    )
    raw_inside = int(
        measurement_evidence.measurement_quality.get(
            "raw_inside_point_count", len(valid)
        )
    )
    outlier_fraction = float(
        removed_outliers / max(raw_inside, 1)
    )
    diagnostics["outlier_fraction"] = outlier_fraction
    if outlier_fraction > float(
        cavity.get("maximum_outlier_fraction", 0.30)
    ):
        return opening_records, geometric_predicate(
            "UNKNOWN",
            method=method,
            evidence=diagnostics,
            reason="NOISY_DISCONNECTED_EVIDENCE",
        )

    opening_records["cavity_depth_m"] = property_value(
        cavity_depth,
        unit="m",
        status="MEASURED",
        method=method,
    )
    return opening_records, geometric_predicate(
        "TRUE",
        method=method,
        evidence=diagnostics,
    )


def _validate_measurement_evidence(
    measurement_evidence: MeasurementEvidence,
) -> None:
    if not isinstance(measurement_evidence, MeasurementEvidence):
        raise TypeError(
            "Property extraction requires a stage-local MeasurementEvidence "
            "object; raw, combined, and cumulative point arrays are rejected"
        )
    if measurement_evidence.cloud_purpose != MEASUREMENT_EVIDENCE_PURPOSE:
        raise ValueError(
            "Property extraction rejected cloud purpose "
            f"{measurement_evidence.cloud_purpose!r}"
        )
    if measurement_evidence.measurement_cloud_path:
        name = Path(measurement_evidence.measurement_cloud_path).name.lower()
        if name in {
            "cumulative.ply",
            "cumulative_visualization.ply",
            "combined_cloud.ply",
            "all_visible_objects.ply",
        }:
            raise ValueError(
                "Historical or scene-combined clouds are not valid "
                "measurement evidence"
            )


def _evidence_provenance(
    measurement_evidence: MeasurementEvidence,
    *,
    extractor_version: str,
) -> dict[str, Any]:
    return {
        "source_stage": measurement_evidence.source_stage,
        "source_region": measurement_evidence.source_region,
        "measurement_cloud_path": (
            measurement_evidence.measurement_cloud_path
        ),
        "contributing_camera_ids": list(
            measurement_evidence.contributing_camera_ids
        ),
        "point_count": int(
            len(measurement_evidence.measurement_points)
        ),
        "extractor_version": extractor_version,
        "cloud_purpose": measurement_evidence.cloud_purpose,
    }


def _attach_provenance(
    record: dict[str, Any],
    measurement_evidence: MeasurementEvidence,
) -> None:
    """Attach the same auditable source to every property/predicate record."""
    for key, value in record.items():
        if not isinstance(value, dict):
            continue
        if "status" in value and "method" in value:
            version = (
                OPEN_RECEPTACLE_EXTRACTOR_VERSION
                if key in {
                    "OPEN_CAVITY",
                    "opening_width_m",
                    "opening_length_m",
                    "cavity_depth_m",
                }
                else EXTRACTOR_VERSION
            )
            value.update(
                _evidence_provenance(
                    measurement_evidence,
                    extractor_version=version,
                )
            )
        _attach_provenance(value, measurement_evidence)


def extract_object_properties(
    measurement_evidence: MeasurementEvidence,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Measure one validated, stage-local, region-gated evidence cloud."""
    _validate_measurement_evidence(measurement_evidence)
    valid = _finite_points(measurement_evidence.measurement_points)
    geometry = robust_pca_geometry(valid, config)
    properties = _property_template()
    predicates = _predicate_template()
    measured = _robust_pca_arrays(valid, config)
    evidence_quality_valid = bool(
        measurement_evidence.measurement_quality.get(
            "quality_is_valid", False
        )
    )

    if not evidence_quality_valid:
        quality_reason = ",".join(
            measurement_evidence.measurement_quality.get("reasons", ())
        ) or "INVALID_MEASUREMENT_EVIDENCE"
        geometry = _unknown_geometry()
        properties = _property_template()
        predicates = {
            key: geometric_predicate(
                "UNKNOWN",
                method="measurement_evidence_quality_gate",
                evidence={
                    "measurement_quality": deepcopy(
                        measurement_evidence.measurement_quality
                    )
                },
                reason=quality_reason,
            )
            for key in GEOMETRIC_PREDICATE_KEYS
        }
        result = {
            **geometry,
            "point_count": property_value(
                int(len(valid)),
                unit="points",
                status="DERIVED",
                method="finite_stage_local_measurement_cloud",
            ),
            "contributing_camera_count": property_value(
                int(len(measurement_evidence.contributing_camera_ids)),
                unit="cameras",
                status="DERIVED",
                method="validated_inspection_camera_contributors",
            ),
            "geometric_properties": properties,
            "geometric_predicates": predicates,
            "measurement_quality": deepcopy(
                measurement_evidence.measurement_quality
            ),
            "measurement_cloud_path": (
                measurement_evidence.measurement_cloud_path
            ),
        }
        _attach_provenance(result, measurement_evidence)
        return result

    if measured is not None:
        length, width, height = map(float, measured["extents"])
        safe_width = max(width, 1e-9)
        safe_height = max(height, 1e-9)
        elongation = length / safe_width
        flatness = width / safe_height
        eigenvalues = measured["eigenvalues"]
        # A plane has two meaningful in-plane variances and a suppressed
        # normal variance. Comparing the smallest eigenvalue to the middle
        # one avoids penalising legitimate rectangular (non-square) planes.
        planarity = float(
            1.0
            - min(
                1.0,
                eigenvalues[2] / max(eigenvalues[1], 1e-12),
            )
        )
        plane_normal = measured["axes"][:, 2]
        if plane_normal[2] < 0.0:
            plane_normal = -plane_normal
        upward_alignment = abs(float(plane_normal[2]))
        support_area = length * width

        derived_values = {
            "total_length_m": (length, "m"),
            "usable_length_m": (length, "m"),
            "maximum_cross_section_m": (width, "m"),
            "elongation_ratio": (elongation, "ratio"),
            "flatness_ratio": (flatness, "ratio"),
            "dominant_plane_normal_world": (
                plane_normal,
                "unit_vector",
            ),
            "planarity_score": (planarity, "ratio"),
            "support_length_m": (length, "m"),
            "support_width_m": (width, "m"),
            "support_thickness_m": (height, "m"),
            "support_area_m2": (support_area, "m2"),
        }
        for key, (value, unit) in derived_values.items():
            properties[key] = property_value(
                value,
                unit=unit,
                status="DERIVED",
                method="robust_pca_obb",
            )

        elongated = config.get("elongated_object", {})
        minimum_dominance = float(
            elongated.get(
                "minimum_dominant_axis_ratio",
                elongated.get("minimum_elongation_ratio", 2.4),
            )
        )
        elongated_true = (
            elongation >= minimum_dominance
        )
        predicates["ELONGATED_OBJECT"] = geometric_predicate(
            "TRUE" if elongated_true else "FALSE",
            method="scale_independent_principal_axis_dominance_v2",
            evidence={
                "total_length_m": length,
                "dominant_axis_ratio": elongation,
                "minimum_dominant_axis_ratio": minimum_dominance,
                "absolute_length_used_for_decision": False,
                "definition": (
                    "largest robust extent dominates middle robust extent"
                ),
            },
        )

        support = config.get("planar_support", {})
        support_true = (
            planarity
            >= float(support.get("minimum_planarity_score", 0.72))
            and upward_alignment
            >= float(support.get("minimum_upward_alignment", 0.70))
            and length
            >= float(support.get("minimum_length_m", 0.070))
            and width
            >= float(support.get("minimum_width_m", 0.050))
            and height
            <= float(support.get("maximum_thickness_m", 0.035))
        )
        predicates["PLANAR_SUPPORT"] = geometric_predicate(
            "TRUE" if support_true else "FALSE",
            method="robust_pca_support_plane",
            evidence={
                "planarity_score": planarity,
                "upward_alignment": upward_alignment,
                "support_length_m": length,
                "support_width_m": width,
                "support_thickness_m": height,
                "support_area_m2": support_area,
            },
        )

    cavity_properties, cavity_predicate = _estimate_open_cavity(
        measurement_evidence,
        config,
        measured,
    )
    properties.update(cavity_properties)
    predicates["OPEN_CAVITY"] = cavity_predicate

    result = {
        **deepcopy(geometry),
        "point_count": property_value(
            int(len(valid)),
            unit="points",
            status="DERIVED",
            method="finite_stage_local_measurement_cloud",
        ),
        "contributing_camera_count": property_value(
            int(len(measurement_evidence.contributing_camera_ids)),
            unit="cameras",
            status="DERIVED",
            method="validated_inspection_camera_contributors",
        ),
        "geometric_properties": properties,
        "geometric_predicates": predicates,
        "measurement_quality": deepcopy(
            measurement_evidence.measurement_quality
        ),
        "measurement_cloud_path": (
            measurement_evidence.measurement_cloud_path
        ),
    }
    _attach_provenance(result, measurement_evidence)
    return result


def _record_number(record: dict[str, Any] | None) -> float | None:
    if not record or record.get("value") is None:
        return None
    try:
        value = float(record["value"])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def pairwise_relation_evaluation(
    relation: str,
    source_properties: dict[str, Any],
    target_properties: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate and explain category-free geometric compatibility."""
    source = source_properties.get("geometric_properties", {})
    target = target_properties.get("geometric_properties", {})
    target_cavity = target_properties.get(
        "geometric_predicates", {}
    ).get("OPEN_CAVITY", {})
    provenance = {
        "inference_basis": "GEOMETRY_ONLY",
        "source_measurement_cloud_path": source_properties.get(
            "measurement_cloud_path"
        ),
        "target_measurement_cloud_path": target_properties.get(
            "measurement_cloud_path"
        ),
    }
    if target_cavity.get("status") != "TRUE":
        return {
            "status": "UNKNOWN",
            "method": "cached_stage_local_geometry_relation_v1",
            "reason": "TARGET_OPEN_CAVITY_NOT_TRUE",
            **provenance,
        }
    relation_config = config.get("pairwise_relations", {})
    if relation == "INSERTABLE_IN":
        cross_section = _record_number(
            source.get("maximum_cross_section_m")
        )
        opening_width = _record_number(target.get("opening_width_m"))
        clearance = float(
            relation_config.get("clearance_margin_m", 0.005)
        )
        if cross_section is None or opening_width is None:
            status = "UNKNOWN"
            reason = "REQUIRED_MEASUREMENT_MISSING"
            pass_margin = None
        else:
            pass_margin = opening_width - (cross_section + clearance)
            status = (
                "TRUE"
                if pass_margin > 0.0
                else "FALSE"
            )
            reason = None
        return {
            "status": status,
            "method": "cross_section_clearance_vs_opening_v1",
            "reason": reason,
            "maximum_cross_section_m": cross_section,
            "clearance_margin_m": clearance,
            "opening_width_m": opening_width,
            "pass_margin_m": pass_margin,
            "evaluated_inequality": (
                "maximum_cross_section_m + clearance_margin_m "
                "< opening_width_m"
            ),
            **provenance,
        }
    if relation == "REACHES_BOTTOM":
        usable_length = _record_number(source.get("usable_length_m"))
        cavity_depth = _record_number(target.get("cavity_depth_m"))
        grip = float(relation_config.get("grip_allowance_m", 0.03))
        if usable_length is None or cavity_depth is None:
            status = "UNKNOWN"
            reason = "REQUIRED_MEASUREMENT_MISSING"
            pass_margin = None
        else:
            pass_margin = usable_length - grip - cavity_depth
            status = (
                "TRUE"
                if pass_margin >= 0.0
                else "FALSE"
            )
            reason = None
        return {
            "status": status,
            "method": "usable_length_after_grip_vs_cavity_v1",
            "reason": reason,
            "usable_length_m": usable_length,
            "grip_allowance_m": grip,
            "cavity_depth_m": cavity_depth,
            "pass_margin_m": pass_margin,
            "evaluated_inequality": (
                "usable_length_m - grip_allowance_m >= cavity_depth_m"
            ),
            **provenance,
        }
    raise ValueError(f"Unsupported pairwise relation: {relation}")


def pairwise_relation_status(
    relation: str,
    source_properties: dict[str, Any],
    target_properties: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Compatibility wrapper returning only the tri-state result."""
    return pairwise_relation_evaluation(
        relation,
        source_properties,
        target_properties,
        config,
    )["status"]
