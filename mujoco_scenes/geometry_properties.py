"""Point-cloud-only geometric properties and deterministic relations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "observed_state_semantics.yaml"
)
SUPPORTED_STATUSES = {"MEASURED", "DERIVED", "SEMANTIC", "UNKNOWN"}


def load_semantics_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load family, function, relation, and extraction configuration."""
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


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


def category_family(category: str, config: dict[str, Any]) -> str:
    """Map a semantic category to its configured shared family."""
    for family, family_config in config.get("families", {}).items():
        if category in family_config.get("categories", []):
            return family
    return "unknown"


def family_property_template(
    family: str, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return the complete, stable key set for a configured family."""
    keys = config.get("families", {}).get(family, {}).get("property_keys", [])
    return {
        key: unknown_property("m", "insufficient_visible_geometry")
        for key in keys
    }


def candidate_functions(category: str, config: dict[str, Any]) -> list[str]:
    return list(config.get("category_functions", {}).get(category, []))


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


def robust_pca_geometry(
    points: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    """Measure robust orientation-independent dimensions from visible points."""
    extraction = config.get("property_extraction", {})
    minimum_points = int(extraction.get("minimum_point_count", 8))
    minimum_extent = float(extraction.get("minimum_extent_m", 0.001))
    lower = float(extraction.get("robust_lower_percentile", 2.0))
    upper = float(extraction.get("robust_upper_percentile", 98.0))

    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    valid = points[np.all(np.isfinite(points), axis=1)]
    if len(valid) < minimum_points:
        return _unknown_geometry()

    # Prevent a handful of finite flying/background points from rotating the
    # covariance basis before robust projected bounds can be applied.
    coordinate_lower = np.percentile(valid, lower, axis=0)
    coordinate_upper = np.percentile(valid, upper, axis=0)
    inlier_mask = np.all(
        (valid >= coordinate_lower) & (valid <= coordinate_upper), axis=1
    )
    inliers = valid[inlier_mask]
    if len(inliers) < minimum_points:
        return _unknown_geometry()

    centre = np.median(inliers, axis=0)
    centred = inliers - centre
    covariance = np.cov(centred, rowvar=False)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        return _unknown_geometry()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    projected = centred @ axes
    robust_min = np.percentile(projected, lower, axis=0)
    robust_max = np.percentile(projected, upper, axis=0)
    extents = robust_max - robust_min
    if not np.all(np.isfinite(extents)) or float(np.max(extents)) < minimum_extent:
        return _unknown_geometry()

    extent_order = np.argsort(extents)[::-1]
    sorted_extents = extents[extent_order]
    principal_axis = axes[:, extent_order[0]]
    dominant_component = int(np.argmax(np.abs(principal_axis)))
    if principal_axis[dominant_component] < 0.0:
        principal_axis = -principal_axis

    return {
        "centroid_world_m": property_value(
            centre,
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
                ("length", "width", "height"), sorted_extents
            )
        },
        "principal_axis_world": property_value(
            principal_axis,
            unit="unit_vector",
            status="MEASURED",
            method="robust_pca_obb",
        ),
        "property_status": "MEASURED",
    }


def _record_number(record: dict[str, Any] | None) -> float | None:
    if not record or record.get("value") is None:
        return None
    value = float(record["value"])
    return value if np.isfinite(value) else None


def _estimate_receptacle_opening(
    points: np.ndarray, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservatively estimate a visible horizontal rim and cavity profile."""
    extraction = config.get("property_extraction", {})
    minimum_points = int(
        extraction.get("receptacle_minimum_point_count", 80)
    )
    minimum_rim = int(extraction.get("receptacle_minimum_rim_points", 12))
    minimum_interior = int(
        extraction.get("receptacle_minimum_interior_points", 8)
    )
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    valid = points[np.all(np.isfinite(points), axis=1)]
    unknown = (
        unknown_property("m", "visible_rim_interior_z_profile"),
        unknown_property("m", "visible_rim_interior_z_profile"),
    )
    if len(valid) < minimum_points:
        return unknown

    top_threshold = float(np.percentile(valid[:, 2], 88.0))
    rim = valid[valid[:, 2] >= top_threshold]
    if len(rim) < minimum_rim:
        return unknown
    centre_xy = np.median(rim[:, :2], axis=0)
    rim_radius = np.linalg.norm(rim[:, :2] - centre_xy, axis=1)
    opening_width = 2.0 * float(np.percentile(rim_radius, 90.0))
    if not np.isfinite(opening_width) or opening_width < 0.005:
        return unknown

    radial = np.linalg.norm(valid[:, :2] - centre_xy, axis=1)
    interior = valid[
        (radial < 0.35 * opening_width)
        & (valid[:, 2] < top_threshold - 0.005)
    ]
    if len(interior) < minimum_interior:
        return (
            property_value(
                opening_width,
                unit="m",
                status="MEASURED",
                method="visible_rim_interior_z_profile",
            ),
            unknown[1],
        )
    cavity_depth = float(np.median(rim[:, 2]) - np.percentile(interior[:, 2], 10.0))
    if not np.isfinite(cavity_depth) or cavity_depth < 0.005:
        return (
            property_value(
                opening_width,
                unit="m",
                status="MEASURED",
                method="visible_rim_interior_z_profile",
            ),
            unknown[1],
        )
    return (
        property_value(
            opening_width,
            unit="m",
            status="MEASURED",
            method="visible_rim_interior_z_profile",
        ),
        property_value(
            cavity_depth,
            unit="m",
            status="MEASURED",
            method="visible_rim_interior_z_profile",
        ),
    )


def extract_object_properties(
    points: np.ndarray,
    *,
    category: str,
    contributing_camera_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Extract the universal and configured family property records."""
    points = np.asarray(points).reshape((-1, 3))
    valid = points[np.all(np.isfinite(points), axis=1)]
    family = category_family(category, config)
    geometry = robust_pca_geometry(valid, config)
    family_properties = family_property_template(family, config)

    dimensions = geometry["dimensions_m"]
    if geometry["property_status"] == "MEASURED":
        if family == "utensil":
            family_properties["total_length_m"] = property_value(
                _record_number(dimensions["length"]),
                unit="m",
                status="DERIVED",
                method="robust_pca_obb",
            )
            family_properties["maximum_width_m"] = property_value(
                _record_number(dimensions["width"]),
                unit="m",
                status="DERIVED",
                method="robust_pca_obb",
            )
        elif family == "support":
            family_properties["support_length_m"] = property_value(
                _record_number(dimensions["length"]),
                unit="m",
                status="DERIVED",
                method="robust_pca_obb",
            )
            family_properties["support_width_m"] = property_value(
                _record_number(dimensions["width"]),
                unit="m",
                status="DERIVED",
                method="robust_pca_obb",
            )
    if family == "receptacle":
        opening, depth = _estimate_receptacle_opening(valid, config)
        family_properties["opening_width_m"] = opening
        family_properties["cavity_depth_m"] = depth

    return {
        **deepcopy(geometry),
        "point_count": property_value(
            int(len(valid)),
            unit="points",
            status="DERIVED",
            method="finite_cumulative_voxel_cloud",
        ),
        "contributing_camera_count": property_value(
            int(contributing_camera_count),
            unit="cameras",
            status="DERIVED",
            method="segmentation_pixel_contributors",
        ),
        "family_properties": family_properties,
    }


def pairwise_relation_status(
    relation: str,
    utensil_properties: dict[str, Any],
    receptacle_properties: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Evaluate one minimal measured utensil–receptacle relation."""
    utensil = utensil_properties.get("family_properties", {})
    receptacle = receptacle_properties.get("family_properties", {})
    relation_config = config.get("pairwise_relations", {})
    if relation == "INSERTABLE_IN":
        maximum_width = _record_number(utensil.get("maximum_width_m"))
        opening_width = _record_number(receptacle.get("opening_width_m"))
        if maximum_width is None or opening_width is None:
            return "UNKNOWN"
        clearance = float(relation_config.get("clearance_margin_m", 0.005))
        return "TRUE" if maximum_width + clearance < opening_width else "FALSE"
    if relation == "REACHES_BOTTOM":
        total_length = _record_number(utensil.get("total_length_m"))
        cavity_depth = _record_number(receptacle.get("cavity_depth_m"))
        if total_length is None or cavity_depth is None:
            return "UNKNOWN"
        grip = float(relation_config.get("grip_allowance_m", 0.03))
        return "TRUE" if total_length - grip >= cavity_depth else "FALSE"
    raise ValueError(f"Unsupported pairwise relation: {relation}")
