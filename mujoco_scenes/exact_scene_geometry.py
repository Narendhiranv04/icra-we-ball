"""Exact privileged geometry extracted from an instantiated MuJoCo model.

This module is used only by the offline feasibility oracle.  It never reads
RGB-D evidence, registries, detector output, or saved point clouds.  Production
geometry continues to use :class:`MeasurementEvidence` in geometry_properties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mujoco
import numpy as np

from mujoco_scenes.geometry_properties import load_geometry_config


@dataclass(frozen=True)
class ExactObjectGeometry:
    object_id: str
    object_kind: str
    extents_m: tuple[float, float, float]
    total_length_m: float | None = None
    maximum_cross_section_m: float | None = None
    elongation_ratio: float | None = None
    opening_width_m: float | None = None
    cavity_depth_m: float | None = None
    open_cavity: bool | None = None
    geometry_source: str = "ACTUAL_INSTANTIATED_MUJOCO_GEOMETRY"
    opening_length_m: float | None = None
    rim_height_m: float | None = None
    extraction_method: str = "instantiated_inner_rim_radial_quantile_v2"
    diagnostics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _body_mesh_points(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        rotation = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        vertices = vertices @ rotation.reshape(3, 3).T
        vertices += model.geom_pos[geom_id]
        parts.append(vertices)
    if not parts:
        # Some valid receptacles (notably the transparent glass) are analytic
        # primitives rather than meshes.  Their effective MuJoCo geom
        # parameters are still exact scene geometry, so construct conservative
        # local bounding points from those parameters.
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            geom_type = int(model.geom_type[geom_id])
            size = np.asarray(model.geom_size[geom_id], dtype=float)
            pos = np.asarray(model.geom_pos[geom_id], dtype=float)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                radius, half_height = size[:2]
                offsets = np.array([
                    [-radius, -radius, -half_height],
                    [radius, radius, half_height],
                ])
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                offsets = np.array([-size, size])
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                offsets = np.array([[-size[0]] * 3, [size[0]] * 3])
            else:
                continue
            parts.append(offsets + pos)
    if not parts:
        raise ValueError(f"Body {body_id} has no supported geometry")
    return np.concatenate(parts, axis=0)


def _bottom_top_z(model: mujoco.MjModel, body_id: int) -> float:
    candidates: list[float] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ) or ""
        if "bottom_collision" not in name:
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        candidates.append(
            float(model.geom_pos[geom_id][2] + model.geom_size[geom_id][1])
        )
    if not candidates:
        raise ValueError(f"Body {body_id} has no bottom collision geometry")
    return max(candidates)


def extract_exact_object_geometry(
    scene: Any,
    object_id: str,
    object_kind: str,
    *,
    geometry_config: dict[str, Any] | None = None,
) -> ExactObjectGeometry:
    """Extract object dimensions from the effective loaded MuJoCo model.

    Mesh vertices include the effective XML mesh scale. Primitive proxy geoms
    are used only to locate the exact container floor; no numeric object table
    is consulted.
    """
    model = scene.model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_id)
    if body_id < 0:
        raise ValueError(f"Object body {object_id!r} is missing from model")
    points = _body_mesh_points(model, body_id)
    extents = np.ptp(points, axis=0)
    if not np.all(np.isfinite(extents)) or np.any(extents <= 0):
        raise ValueError(f"Invalid exact mesh extents for {object_id}")

    config = geometry_config or load_geometry_config()
    ratio_threshold = float(
        config["elongated_object"]["minimum_dominant_axis_ratio"]
    )
    is_tool = object_kind in {
        "spoon", "fork", "knife", "stirrer", "marker",
    } or "spoon" in object_kind
    if is_tool:
        ordered = np.sort(extents)[::-1]
        ratio = float(ordered[0] / ordered[1])
        return ExactObjectGeometry(
            object_id=object_id,
            object_kind=object_kind,
            extents_m=tuple(float(x) for x in extents),
            total_length_m=float(ordered[0]),
            maximum_cross_section_m=float(ordered[1]),
            elongation_ratio=ratio,
        )

    is_container = (
        object_kind in {"cup", "mug", "glass", "bowl", "mixing_bowl"}
        or "cup" in object_kind
        or "mug" in object_kind
        or "bowl" in object_kind
    )
    if not is_container:
        return ExactObjectGeometry(
            object_id=object_id,
            object_kind=object_kind,
            extents_m=tuple(float(x) for x in extents),
        )

    # Estimate the *inner* rim envelope from the instantiated surface.  A
    # handle or other exterior protrusion is outside the upper-rim radial
    # cluster and therefore cannot enlarge the usable opening.
    z_max = float(points[:, 2].max())
    z_min = float(points[:, 2].min())
    rim_points = points[points[:, 2] >= z_max - 0.12 * (z_max - z_min)]
    centre = np.median(rim_points[:, :2], axis=0)
    radii = np.linalg.norm(rim_points[:, :2] - centre, axis=1)
    if len(radii) < 8 or not np.all(np.isfinite(radii)):
        raise ValueError(f"Insufficient instantiated rim samples for {object_id}")
    inner_radius = float(np.quantile(radii, 0.45))
    opening_width = opening_length = max(2.0 * inner_radius, 1e-6)
    cavity_depth = float(z_max - _bottom_top_z(model, body_id))
    open_cavity = bool(cavity_depth > 0.0 and opening_width > 0.0)
    return ExactObjectGeometry(
        object_id=object_id,
        object_kind=object_kind,
        extents_m=tuple(float(x) for x in extents),
        opening_width_m=opening_width,
        opening_length_m=opening_length,
        rim_height_m=z_max,
        cavity_depth_m=cavity_depth,
        open_cavity=open_cavity,
        diagnostics={
            "rim_sample_count": int(len(rim_points)),
            "inner_radius_quantile": 0.45,
            "outer_protrusions_excluded": True,
        },
    )
