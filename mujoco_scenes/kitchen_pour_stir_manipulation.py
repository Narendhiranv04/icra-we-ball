"""Geometry and fail-closed execution records for Kitchen Phase C."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import mujoco
import numpy as np


EVIDENCE_MODE = "KINEMATIC_ACTION_PROXY_NO_FLUID_DYNAMICS"
PHASE_C_OPERATOR_ELIGIBLE_TARGET_REGIONS = frozenset(("countertop", "B1"))


def phase_c_execution_plan(
    frozen_plan: list[dict[str, Any]], frozen_registry: dict[str, Any]
) -> list[dict[str, Any]]:
    """Exclude POUR/STIR whose target remains in a cupboard or drawer.

    B1 is the scene's BOX region.  All non-Phase-C operators, including
    cupboard PICK/PLACE, retain their original order and object identities.
    """
    objects = frozen_registry["objects"]
    result = []
    for row in frozen_plan:
        operator = row["action"].upper()
        arguments = list(row.get("arguments", []))
        if operator in {"POUR", "STIR"}:
            target_id = arguments[1]
            if objects[target_id].get("source_region") not in (
                PHASE_C_OPERATOR_ELIGIBLE_TARGET_REGIONS
            ):
                continue
        result.append(row)
    return result


@dataclass(frozen=True)
class TargetOpening:
    target_generic_id: str
    centre_world_m: tuple[float, float, float]
    rim_normal_world: tuple[float, float, float]
    opening_half_extents_m: tuple[float, float]
    cavity_depth_m: float
    safety_margin_m: float
    provenance: str


@dataclass(frozen=True)
class PourSpec:
    family: str
    outlet_local_m: tuple[float, float, float]
    outlet_provenance: str
    tilt_candidates_rad: tuple[float, ...]
    dwell_time_s: float


@dataclass(frozen=True)
class ToolTipGeometry:
    length_m: float
    longitudinal_axis_local: tuple[float, float, float]
    grasp_local_m: tuple[float, float, float]
    active_tip_local_m: tuple[float, float, float]
    active_tip_offset_from_gripper_m: tuple[float, float, float]
    provenance: str


def _value(properties: dict[str, Any], key: str) -> float | None:
    row = properties.get(key, {})
    value = row.get("value") if isinstance(row, dict) else None
    return None if value is None else float(value)


def _body_collision_rim_z(model: mujoco.MjModel, body_id: int) -> float:
    candidates = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = float(size[2])
        elif geom_type in {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
        }:
            extent = float(size[1] + (size[0] if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0))
        else:
            extent = float(max(size))
        candidates.append(float(model.geom_pos[geom_id, 2]) + extent)
    if not candidates:
        raise ValueError("Target has no physical collision geometry")
    return max(candidates)


def derive_target_opening(
    scene,
    registry_object: dict[str, Any],
    generic_id: str,
    backend_body: str,
    *,
    safety_margin_m: float = 0.006,
) -> TargetOpening:
    properties = registry_object.get("geometric_properties", {})
    width = _value(properties, "opening_width_m")
    length = _value(properties, "opening_length_m")
    depth = _value(properties, "cavity_depth_m")
    if width is None or length is None or depth is None:
        raise ValueError("POUR_OPENING_GEOMETRY_UNAVAILABLE")
    if min(width, length) <= 2.0 * safety_margin_m or depth <= safety_margin_m:
        raise ValueError("POUR_OPENING_GEOMETRY_UNAVAILABLE")
    body_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, backend_body
    )
    if body_id < 0:
        raise ValueError("POUR_TARGET_RESOLUTION_FAILED")
    rotation = scene.data.xmat[body_id].reshape(3, 3).copy()
    centre = scene.data.xpos[body_id] + rotation @ np.array(
        (0.0, 0.0, _body_collision_rim_z(scene.model, body_id))
    )
    normal = rotation[:, 2]
    normal /= np.linalg.norm(normal)
    return TargetOpening(
        generic_id,
        tuple(float(v) for v in centre),
        tuple(float(v) for v in normal),
        (0.5 * length, 0.5 * width),
        depth,
        safety_margin_m,
        "FROZEN_OPEN_CAVITY_DIMENSIONS_LOCALIZED_BY_LIVE_PHYSICAL_BODY",
    )


def _visual_mesh_outlet_local(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    points = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != body_id:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH) or mesh_id < 0:
            continue
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[start:start + count]
        geom_rotation = np.empty(9)
        mujoco.mju_quat2Mat(geom_rotation, model.geom_quat[geom_id])
        transformed = vertices @ geom_rotation.reshape(3, 3).T + model.geom_pos[geom_id]
        points.extend(transformed.tolist())
    if not points:
        raise ValueError("Source family has no physical visual mesh outlet evidence")
    vertices = np.asarray(points, float)
    upper = vertices[vertices[:, 2] >= np.quantile(vertices[:, 2], 0.55)]
    return upper[np.argmax(np.linalg.norm(upper[:, :2], axis=1))]


def derive_pour_spec(scene, backend_body: str, family: str) -> PourSpec:
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    if body_id < 0:
        raise ValueError("POUR_SOURCE_RESOLUTION_FAILED")
    if family == "JAR_SOURCE":
        site_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{backend_body}_opening"
        )
        if site_id < 0:
            raise ValueError("POUR_OPENING_GEOMETRY_UNAVAILABLE")
        outlet = scene.model.site_pos[site_id].copy()
        wall_centres = [
            scene.model.geom_pos[geom_id].copy()
            for geom_id in range(scene.model.ngeom)
            if int(scene.model.geom_bodyid[geom_id]) == body_id
            and (scene.model.geom_contype[geom_id] or scene.model.geom_conaffinity[geom_id])
            and float(np.linalg.norm(scene.model.geom_pos[geom_id, :2])) > 0.0
        ]
        if not wall_centres:
            raise ValueError("POUR_OPENING_GEOMETRY_UNAVAILABLE")
        rim = max(wall_centres, key=lambda point: (float(point[0]), -abs(float(point[1]))))
        outlet[:2] = rim[:2]
        provenance = "AUTHORED_PHYSICAL_MOUTH_SITE_PLUS_COLLISION_DERIVED_RIM_EDGE"
        tilts = tuple(math.radians(value) for value in (35.0, 45.0, 55.0))
    elif family == "KETTLE":
        outlet = _visual_mesh_outlet_local(scene.model, body_id)
        provenance = "PHYSICAL_VISUAL_MESH_UPPER_RADIAL_EXTREMUM"
        # The former 10-degree motion was almost visually indistinguishable
        # from an approach.  Start with a clear but still modest kettle pour.
        tilts = tuple(math.radians(value) for value in (22.5, 27.5, 32.5))
    else:
        raise ValueError(f"Unsupported pour family: {family}")
    # Keep both source families brisk; the tilt remains long enough to read
    # clearly in the rendered execution without stalling at the target.
    dwell_time_s = 0.18
    return PourSpec(
        family, tuple(float(v) for v in outlet), provenance, tilts, dwell_time_s
    )


def derive_tool_tip(scene, backend_body: str, observed_length_m: float) -> ToolTipGeometry:
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    site_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{backend_body}_grasp"
    )
    if body_id < 0 or site_id < 0 or observed_length_m <= 0.0:
        raise ValueError("STIR_ACTIVE_TIP_GEOMETRY_UNAVAILABLE")
    centres = np.asarray([
        scene.model.geom_pos[geom_id]
        for geom_id in range(scene.model.ngeom)
        if int(scene.model.geom_bodyid[geom_id]) == body_id
        and (scene.model.geom_contype[geom_id] or scene.model.geom_conaffinity[geom_id])
    ], float)
    if len(centres) < 2:
        raise ValueError("STIR_ACTIVE_TIP_GEOMETRY_UNAVAILABLE")
    centroid = centres.mean(axis=0)
    _, _, vh = np.linalg.svd(centres - centroid, full_matrices=False)
    axis = vh[0]
    grasp = scene.model.site_pos[site_id].copy()
    if float(np.dot(grasp - centroid, axis)) < 0.0:
        axis = -axis
    tip = grasp - axis * observed_length_m
    return ToolTipGeometry(
        observed_length_m,
        tuple(float(v) for v in axis),
        tuple(float(v) for v in grasp),
        tuple(float(v) for v in tip),
        tuple(float(v) for v in tip - grasp),
        "OBSERVED_LENGTH_PLUS_COLLISION_GEOMETRY_PCA_OPPOSITE_GRASP_END",
    )


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


class PhaseCExecutionLedger:
    """Exact-event ledger; records effects only from verified physical motion."""

    def __init__(self, frozen_plan: list[dict[str, Any]]):
        self.expected = {
            int(row["step"]): {
                "operator": row["action"].upper(),
                "arguments": list(row.get("arguments", [])),
            }
            for row in frozen_plan
            if row["action"].upper() in {"POUR", "STIR"}
        }
        self.events: dict[int, dict[str, Any]] = {}

    def commit(self, step: int, result: dict[str, Any]) -> bool:
        expected = self.expected.get(int(step))
        request = result.get("request", {})
        motion_key = "pour_motion_verified" if expected and expected["operator"] == "POUR" else "stir_motion_verified"
        valid = bool(
            expected
            and request.get("action") == expected["operator"]
            and list(request.get("arguments", []))[:2] == expected["arguments"][:2]
            and result.get("success") is True
            and result.get(motion_key) is True
        )
        if not valid:
            return False
        if int(step) in self.events:
            return self.events[int(step)] == result
        self.events[int(step)] = result
        return True

    def summary(self) -> dict[str, Any]:
        missing = sorted(set(self.expected) - set(self.events))
        return {
            "evidence_mode": EVIDENCE_MODE,
            "physical_fluid_transfer_modeled": False,
            "expected_event_count": len(self.expected),
            "verified_event_count": len(self.events),
            "missing_steps": missing,
            "complete": not missing and len(self.events) == len(self.expected),
            "events": [
                {"symbolic_step": step, **event}
                for step, event in sorted(self.events.items())
            ],
        }
