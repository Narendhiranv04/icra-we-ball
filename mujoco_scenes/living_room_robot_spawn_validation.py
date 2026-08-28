"""Evaluation-only Google Robot spawn and clearance validation for L2.

This module compiles and settles the physical scene, but never invokes robot
navigation, planning, IK, grasping, or manipulation.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from mujoco_scenes.living_room_region_scene import L2LivingRoomRegionScene


def _name(model: mujoco.MjModel, object_type, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"id_{object_id}"


def _is_robot_body(model: mujoco.MjModel, body_id: int) -> bool:
    while body_id > 0:
        name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name.startswith("google:"):
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _base_to_static_box_clearance(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    geom_id: int,
    base_center: np.ndarray,
    base_radius: float,
    base_halfheight: float,
) -> float:
    delta = base_center - data.geom_xpos[geom_id]
    local = data.geom_xmat[geom_id].reshape(3, 3).T @ delta
    half = model.geom_size[geom_id]
    dx = max(abs(float(local[0])) - float(half[0]), 0.0)
    dy = max(abs(float(local[1])) - float(half[1]), 0.0)
    radial_clearance = max(math.hypot(dx, dy) - base_radius, 0.0)
    vertical_clearance = max(
        abs(float(local[2])) - float(half[2]) - base_halfheight, 0.0
    )
    return math.hypot(radial_clearance, vertical_clearance)


def validate_google_robot_spawn(
    scene_name: str,
    *,
    settle_steps: int = 10,
    minimum_clearance_m: float = 0.10,
) -> dict[str, Any]:
    """Compile the Google Robot variant and validate its staging pose."""
    scene = L2LivingRoomRegionScene(scene_name, robot="google")
    model = scene.model
    data = scene.data
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    base_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "google:base_link"
    )
    base_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "google:base_collision_proxy"
    )
    spawn_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "l2_canonical_robot_spawn"
    )
    staging_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "a2_staging_top"
    )
    if min(base_body_id, base_geom_id, spawn_site_id, staging_geom_id) < 0:
        raise RuntimeError("Google Robot spawn validation entities are missing")

    base_position = data.xpos[base_body_id].copy()
    base_rotation = data.xmat[base_body_id].reshape(3, 3).copy()
    forward_xy = base_rotation[:, 0][:2]
    forward_xy /= np.linalg.norm(forward_xy)
    workspace_direction = -base_position[:2]
    workspace_direction /= np.linalg.norm(workspace_direction)
    facing_alignment = float(np.dot(forward_xy, workspace_direction))
    spawn_site = data.site_xpos[spawn_site_id].copy()
    staging_position = data.geom_xpos[staging_geom_id].copy()
    base_radius = float(model.geom_size[base_geom_id][0])
    base_halfheight = float(model.geom_size[base_geom_id][1])
    base_geom_center = data.geom_xpos[base_geom_id].copy()

    clearances = []
    excluded_names = {"floor", "a2_rug_surface", "a2_rug_border"}
    for geom_id in range(model.ngeom):
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = int(model.geom_bodyid[geom_id])
        if (
            geom_id == base_geom_id
            or _is_robot_body(model, body_id)
            or geom_name in excluded_names
            or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX
            or not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        ):
            continue
        clearance = _base_to_static_box_clearance(
            data,
            model,
            geom_id,
            base_geom_center,
            base_radius,
            base_halfheight,
        )
        clearances.append({"geom": geom_name, "clearance_m": clearance})
    clearances.sort(key=lambda item: item["clearance_m"])

    robot_contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        if not (_is_robot_body(model, body1) or _is_robot_body(model, body2)):
            continue
        robot_contacts.append(
            {
                "geom1": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                "distance_m": float(contact.dist),
            }
        )
    invalid_contacts = [
        item
        for item in robot_contacts
        if item["geom1"] != "floor" and item["geom2"] != "floor"
    ]
    minimum_static_clearance = min(
        item["clearance_m"] for item in clearances
    )
    checks = {
        "model_compiles_with_google_robot": True,
        "spawn_matches_canonical_site_xy": (
            float(np.linalg.norm(base_position[:2] - spawn_site[:2])) < 0.01
        ),
        "robot_is_behind_staging_surface": (
            float(base_position[1]) < float(staging_position[1])
        ),
        "robot_faces_workspace": facing_alignment >= 0.95,
        "minimum_static_clearance_satisfied": (
            minimum_static_clearance >= minimum_clearance_m
        ),
        "no_robot_furniture_contacts": not invalid_contacts,
    }
    return {
        "schema_version": 1,
        "validation_scope": "SPAWN_ONLY_NO_ROBOT_EXECUTION",
        "scene_name": scene_name,
        "robot": "google",
        "settle_steps": settle_steps,
        "base_position_world_m": base_position.tolist(),
        "canonical_spawn_site_world_m": spawn_site.tolist(),
        "base_forward_world_xy": forward_xy.tolist(),
        "workspace_facing_alignment": facing_alignment,
        "base_radius_m": base_radius,
        "base_halfheight_m": base_halfheight,
        "minimum_required_clearance_m": minimum_clearance_m,
        "minimum_static_clearance_m": minimum_static_clearance,
        "nearest_static_geometries": clearances[:8],
        "robot_contacts": robot_contacts,
        "invalid_robot_furniture_contacts": invalid_contacts,
        "checks": checks,
        "all_passed": all(checks.values()),
    }
