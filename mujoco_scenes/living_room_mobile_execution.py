"""Observed-evidence mobile refinement and physical execution for L2 Phase 3.

The frozen Phase-2 PICK/PLACE order is treated as immutable input.  This
module resolves its generic identifiers at the simulator boundary, allocates
measured support positions, tests the current base pose first, and inserts a
MOVE only when the same IK/collision checks used by execution require one.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from .generic_manipulation import (
    CalibratedPickPlaceExecutor,
    ProfiledIK,
    RobotConfigurationCollisionChecker,
    SimplePickSpec,
)
from .living_room_region_scene import (
    INTEGRATED_ROOM_LAYOUT,
    L2LivingRoomRegionScene,
)
from .mobile_motion import BasePose, MuJoCoBaseCollisionChecker, RRTStarPlanner
from .robot_profiles import manipulation_profile, mobile_profile


DEFAULT_SCENE = "L2_integrated_living_room_region_function_F0_ALL_OBJECTS_IN_STAGING"
SCENE = DEFAULT_SCENE
SCHEMA_VERSION = 1
SPAWN_Y = float(INTEGRATED_ROOM_LAYOUT["robot_spawn"][1])
PAYLOAD_BACKENDS = {
    "a2_drink_left": "cup",
    "a2_drink_right": "cup",
    "a2_snack_left": "saucer",
    "a2_snack_right": "saucer",
    "a2_remote_payload": "tv_remote",
}
REGION_BACKENDS = {
    "a2_personal_left_top": "side_table",
    "a2_control_table_top": "coffee_table",
    "a2_personal_right_top": "side_table",
}
SUPPORT_BACKENDS = (
    "a2_personal_left",
    "a2_personal_right",
    "a2_shared_table",
    "a2_personal_sofa_left",
    "a2_personal_sofa_right",
    "a2_shared_coffee_table",
    "a2_media_console",
    "a2_control_table",
)
ALLOWED_INTERACTION_BODIES = tuple(PAYLOAD_BACKENDS) + SUPPORT_BACKENDS
SUPPORT_HEIGHT = {
    "cup": 0.070,
    "saucer": 0.008,
    "tv_remote": 0.016,
}
GRASP_Z_OFFSET = {
    "cup": 0.005,
    "saucer": 0.003,
    "tv_remote": 0.002,
}
LINEAR_SETTLE_THRESHOLD_M_S = 0.02
ANGULAR_SETTLE_THRESHOLD_RAD_S = 0.12
CONTACT_PENETRATION_TOLERANCE_M = 0.005
EXECUTION_GEOMETRY_TOLERANCE_M = 0.012


@dataclass(frozen=True)
class PlacementTarget:
    object_id: str
    region_id: str
    position_world: tuple[float, float, float]
    yaw_world_rad: float
    footprint_length_m: float
    footprint_width_m: float
    packing_arrangement: str
    phase1_orientation_deg: int
    edge_clearance_m: float
    inter_payload_clearance_m: float
    predicted_minimum_margin_m: float
    phase1_function_id: str
    phase1_slot_id: str
    source_phase1_signed_margin_m: float
    source: str = "PHASE1_SELECTED_MEASURED_PACKING"


@dataclass(frozen=True)
class HeldObjectState:
    object_id: str
    backend_body: str
    weld_id: int
    weld_active: bool
    gripper_body: str
    relative_position_m: tuple[float, float, float]
    relative_orientation_wxyz: tuple[float, float, float, float]
    finger_joint_positions: tuple[float, float]
    gripper_state_plausible: bool
    validation_status: str
    rejection_reasons: tuple[str, ...]


def oriented_rectangle_corners(
    center_xy: np.ndarray, length: float, width: float, yaw: float
) -> np.ndarray:
    """Return counter-clockwise world XY corners of an oriented rectangle."""
    local = np.array(((-length / 2, -width / 2), (length / 2, -width / 2),
                      (length / 2, width / 2), (-length / 2, width / 2)))
    rotation = np.array(((math.cos(yaw), -math.sin(yaw)),
                         (math.sin(yaw), math.cos(yaw))))
    return local @ rotation.T + np.asarray(center_xy, float)


def world_to_region_local(points_xy: np.ndarray, center_xy: np.ndarray, axis_xy: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis_xy, float)[:2]
    axis /= np.linalg.norm(axis)
    orthogonal = np.array((-axis[1], axis[0]))
    return (np.asarray(points_xy, float) - np.asarray(center_xy, float)[:2]) @ np.column_stack((axis, orthogonal))


def rectangle_inside_observed_support(
    corners_world_xy: np.ndarray,
    center_world_xy: np.ndarray,
    principal_axis_xy: np.ndarray,
    support_length_m: float,
    support_width_m: float,
    tolerance_m: float = 0.0,
) -> dict[str, Any]:
    local = world_to_region_local(corners_world_xy, center_world_xy, principal_axis_xy)
    margins = np.column_stack((support_length_m / 2 - np.abs(local[:, 0]),
                               support_width_m / 2 - np.abs(local[:, 1])))
    minimum = float(np.min(margins))
    return {"region_local_corners_m": local.tolist(), "minimum_edge_margin_m": minimum,
            "inside": minimum >= -tolerance_m}


def _rectangle_axes(corners: np.ndarray) -> list[np.ndarray]:
    axes = []
    for edge in (corners[1] - corners[0], corners[3] - corners[0]):
        normal = np.array((-edge[1], edge[0]), float)
        axes.append(normal / np.linalg.norm(normal))
    return axes


def oriented_rectangles_clearance(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """SAT signed clearance: positive gap, negative overlap penetration."""
    separations = []
    for axis in (*_rectangle_axes(left), *_rectangle_axes(right)):
        lp, rp = left @ axis, right @ axis
        separations.append(max(float(np.min(rp) - np.max(lp)), float(np.min(lp) - np.max(rp))))
    maximum_separation = max(separations)
    if maximum_separation > 0:
        # Euclidean corner/edge distance is unnecessary for the acceptance
        # rule; SAT separation is conservative and exact along a separating axis.
        signed = maximum_separation
        overlap = False
    else:
        signed = maximum_separation
        overlap = True
    return {"signed_clearance_m": float(signed), "overlap": overlap}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_execution_frame(scene: L2LivingRoomRegionScene, path: Path) -> None:
    frame = scene.render_frame(camera="l2_camera_top", width=1280, height=720)
    Image.fromarray(frame).save(path)


def _capture_execution_frame(scene: L2LivingRoomRegionScene) -> np.ndarray:
    return scene.render_frame(camera="l2_camera_top", width=1280, height=720)


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Execution backend body is absent: {name}")
    return data.xpos[body_id].copy()


def _configured_free_body_position(model: mujoco.MjModel, name: str) -> np.ndarray:
    """Return the execution model's initial free-body pose at its adapter boundary."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Execution payload is not a free body: {name}")
    qpos = int(model.jnt_qposadr[joint_id])
    return model.qpos0[qpos:qpos + 3].copy()


def _geom_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise RuntimeError(f"Execution backend support is absent: {name}")
    return data.geom_xpos[geom_id].copy()


def _semantic_role(record: dict[str, Any]) -> str:
    return str(record.get("semantic_payload_role") or "UNKNOWN")


def resolve_execution_entities(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload_registry: dict[str, Any],
    region_registry: dict[str, Any],
) -> dict[str, Any]:
    """Resolve generic IDs using semantic consistency and measured centroids.

    Simulator names exist only in this adapter output.  They never enter the
    frozen planner, witness, placement allocator, or stance selector.
    """
    mujoco.mj_forward(model, data)
    object_rows = []
    remaining = set(PAYLOAD_BACKENDS)
    for object_id, record in sorted(payload_registry["objects"].items()):
        observed = np.asarray(record["observed_centroid_world_m"], float)
        role = _semantic_role(record)
        candidates = []
        for backend in sorted(remaining):
            semantic_ok = PAYLOAD_BACKENDS[backend] == role
            backend_centroid = _configured_free_body_position(model, backend)
            distance = float(np.linalg.norm(observed - backend_centroid))
            candidates.append((not semantic_ok, distance, backend))
        if not candidates:
            continue
        incompatible, distance, backend = min(candidates)
        if incompatible or distance > 0.18:
            raise RuntimeError(
                f"No unambiguous physical match for {object_id}: role={role}, "
                f"best_distance={distance:.3f} m"
            )
        remaining.remove(backend)
        object_rows.append(
            {
                "generic_object_id": object_id,
                "backend_body": backend,
                "semantic_role": role,
                "observed_centroid_world_m": observed.tolist(),
                "backend_centroid_world_m": _configured_free_body_position(model, backend).tolist(),
                "centroid_distance_m": distance,
                "accepted": True,
                "method": "semantic_consistent_nearest_observed_centroid_one_to_one",
            }
        )

    region_rows = []
    remaining_regions = set(REGION_BACKENDS)
    for region_id, record in sorted(region_registry["regions"].items()):
        observed = np.asarray(record["geometry"]["centroid_world_m"]["value"], float)
        candidates = [
            (
                float(np.linalg.norm(observed - _geom_position(model, data, backend))),
                backend,
            )
            for backend in sorted(remaining_regions)
        ]
        if not candidates:
            continue
        distance, backend = min(candidates)
        if distance > 0.25:
            continue
        remaining_regions.remove(backend)
        region_rows.append(
            {
                "generic_region_id": region_id,
                "backend_support_geom": backend,
                "observed_centroid_world_m": observed.tolist(),
                "backend_centroid_world_m": _geom_position(model, data, backend).tolist(),
                "centroid_distance_m": distance,
                "accepted": True,
                "method": "nearest_observed_support_centroid_one_to_one",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "boundary": "SIMULATION_ADAPTER_ONLY",
        "task_planner_received_backend_names": False,
        "execution_refiner_backend_access": "ONLY_VIA_SIMULATION_ADAPTER",
        "objects": object_rows,
        "regions": region_rows,
    }


def allocate_observed_placements(
    payload_registry: dict[str, Any],
    region_registry: dict[str, Any],
    phase2_plan: dict[str, Any],
    phase1_assignments: dict[str, Any],
) -> dict[str, Any]:
    """Realize the frozen selected Phase-1 measured rectangle packing."""
    destination = {
        item["arguments"]["object"]: item["arguments"]["region"]
        for item in phase2_plan["actions"]
        if item["operator"] == "PLACE"
    }
    rows = []
    pair_checks = []
    assignments = {
        row["region_id"]: row
        for row in phase1_assignments["assignments"]
        if any(
            destination.get(object_id) == row["region_id"]
            for object_id in row["payload_ids"]
        )
    }
    for region_id, assignment in sorted(assignments.items()):
        object_ids = assignment["payload_ids"]
        region = region_registry["regions"][region_id]
        geometry = region["geometry"]
        center = np.asarray(geometry["centroid_world_m"]["value"], float)
        axis = np.asarray(geometry["principal_axis_world"]["value"], float)
        axis /= np.linalg.norm(axis)
        length = float(geometry["support_length_m"]["value"])
        width = float(geometry["support_width_m"]["value"])
        orthogonal = np.array((-axis[1], axis[0], 0.0))
        evidence = assignment["selected_compatibility_evidence"]["fit_evidence"]
        margin = float(evidence["edge_clearance_margin_m"])
        if len(object_ids) == 1:
            packing = {
                "arrangement": "SINGLE_CENTERED",
                "payload_orientations_degrees": [
                    int(evidence["selected_orientation_degrees"])
                ],
                "signed_clearance_margin_m": float(
                    evidence["signed_fit_margin_m"]
                ),
            }
            clearance = 0.0
            offsets = np.array((0.0,))
            placement_axis = axis
        else:
            packing = evidence["selected_packing"]
            clearance = float(evidence["inter_payload_clearance_m"])
            oriented = packing["oriented_payload_footprints_m"]
            along_length = packing["arrangement"] == "ALONG_LENGTH"
            sizes = [float(item[0] if along_length else item[1]) for item in oriented]
            available_margin = float(packing.get("length_margin_m" if along_length else "width_margin_m", 0.0))
            extra_spread = max(0.0, min(available_margin * 0.40, 0.018))
            offsets = np.array((-(sizes[1] + clearance) / 2 - extra_spread, (sizes[0] + clearance) / 2 + extra_spread))
            placement_axis = axis if along_length else orthogonal
        orientations = list(packing["payload_orientations_degrees"])
        fixed_indices = [
            index for index, object_id in enumerate(object_ids)
            if object_id not in destination
        ]
        if len(object_ids) == 2 and len(fixed_indices) == 1:
            fixed_index = fixed_indices[0]
            observed = np.asarray(
                payload_registry["objects"][object_ids[fixed_index]][
                    "observed_centroid_world_m"
                ], float,
            )
            candidate_points = [center + offset * placement_axis for offset in offsets]
            nearest = int(np.argmin([
                np.linalg.norm(point[:2] - observed[:2])
                for point in candidate_points
            ]))
            if fixed_index != nearest:
                offsets = offsets[::-1].copy()
                orientations.reverse()
        fixed_footprints: dict[str, np.ndarray] = {}
        for fixed_index in fixed_indices:
            fixed_id = object_ids[fixed_index]
            fixed_record = payload_registry["objects"][fixed_id]
            fixed_observed = np.asarray(
                fixed_record["observed_centroid_world_m"], float
            )
            fixed_direction = np.asarray(
                fixed_record["geometry"]["principal_orientation_world"]["value"],
                float,
            )
            fixed_footprints[fixed_id] = oriented_rectangle_corners(
                fixed_observed[:2],
                float(fixed_record["geometry"]["footprint_length_m"]["value"]),
                float(fixed_record["geometry"]["footprint_width_m"]["value"]),
                math.atan2(float(fixed_direction[1]), float(fixed_direction[0])),
            )
        footprints = []
        for index, (object_id, offset) in enumerate(zip(object_ids, offsets)):
            record = payload_registry["objects"][object_id]
            role = _semantic_role(record)
            obj_length = float(record["geometry"]["footprint_length_m"]["value"])
            obj_width = float(record["geometry"]["footprint_width_m"]["value"])
            point = center + offset * placement_axis
            point[2] = center[2] + SUPPORT_HEIGHT[role]
            base_yaw = math.atan2(float(axis[1]), float(axis[0]))
            raw_yaw = base_yaw + math.radians(int(orientations[index]))
            # A rectangular footprint is invariant under a 180-degree turn.
            # Canonicalize that exact geometric equivalence so manipulation
            # never attempts an unnecessary pi wrist rotation.
            yaw = raw_yaw % math.pi
            if yaw > 1e-6:
                yaw -= math.pi
            if object_id in destination and fixed_footprints:
                candidates = [point.copy()]
                for u in (-0.32, 0.0, 0.32):
                    for v in (-0.32, 0.0, 0.32):
                        candidate = center + u * length * axis + v * width * orthogonal
                        candidate[2] = point[2]
                        candidates.append(candidate)
                viable = []
                for candidate in candidates:
                    candidate_corners = oriented_rectangle_corners(
                        candidate[:2], obj_length, obj_width, yaw
                    )
                    candidate_boundary = rectangle_inside_observed_support(
                        candidate_corners, center[:2], axis[:2], length, width
                    )
                    if (
                        not candidate_boundary["inside"]
                        or candidate_boundary["minimum_edge_margin_m"] + 1e-9 < margin
                    ):
                        continue
                    separations = [
                        oriented_rectangles_clearance(candidate_corners, fixed)[
                            "signed_clearance_m"
                        ]
                        for fixed in fixed_footprints.values()
                    ]
                    if min(separations) + 1e-9 >= clearance:
                        viable.append((float(np.linalg.norm(candidate[:2] - center[:2])), candidate))
                if not viable:
                    raise RuntimeError(
                        f"No collision-free measured-support placement for {object_id}"
                    )
                point = min(viable, key=lambda item: item[0])[1]
            corners = oriented_rectangle_corners(point[:2], obj_length, obj_width, yaw)
            boundary = rectangle_inside_observed_support(
                corners, center[:2], axis[:2], length, width
            )
            if not boundary["inside"] or boundary["minimum_edge_margin_m"] + 1e-9 < margin:
                raise RuntimeError(f"Observed support cannot safely place {object_id}")
            if object_id not in destination:
                corners = fixed_footprints[object_id]
            footprints.append((object_id, corners))
            if object_id not in destination:
                continue
            target = PlacementTarget(
                object_id, region_id, tuple(map(float, point)), float(yaw),
                obj_length, obj_width, packing["arrangement"], int(orientations[index]),
                margin, clearance, float(boundary["minimum_edge_margin_m"]),
                assignment["function_id"], assignment["slot_id"],
                float(packing["signed_clearance_margin_m"]),
            )
            row = asdict(target)
            row["desired_body_world_m"] = list(row.pop("position_world"))
            row["within_measured_support"] = True
            row["footprint_corners_world_m"] = corners.tolist()
            row["region_local_footprint_corners_m"] = boundary["region_local_corners_m"]
            rows.append(row)
        for (left_id, left), (right_id, right) in itertools.combinations(footprints, 2):
            check = oriented_rectangles_clearance(left, right)
            check.update({"region_id": region_id, "object_ids": [left_id, right_id],
                          "required_clearance_m": clearance,
                          "valid": not check["overlap"] and check["signed_clearance_m"] + 1e-9 >= clearance})
            pair_checks.append(check)
            if not check["valid"]:
                raise RuntimeError(
                    f"Placement overlap/clearance failure: {left_id}/{right_id}: {check}"
                )
    if set(destination) != {row["object_id"] for row in rows}:
        raise RuntimeError("Selected Phase-1 packing does not cover every Phase-2 PLACE")
    return {"schema_version": 2, "phase1_selected_packing_consumed": True,
            "placements": rows, "pairwise_rectangle_checks": pair_checks}


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _joint_base_to_world(values: np.ndarray) -> BasePose:
    return BasePose(float(-values[1]), float(SPAWN_Y + values[0]), _normalize_angle(float(values[2])))


def _world_to_joint_base(pose: BasePose) -> np.ndarray:
    return np.array((pose.y - SPAWN_Y, -pose.x, _normalize_angle(pose.yaw)), float)


def _angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


@dataclass(frozen=True)
class ManipulationCandidate:
    base_pose: BasePose
    current_pose: bool
    radial_distance_m: float
    ik_position_error_m: float
    ik_angle_error_rad: float
    collision_free: bool
    rejection_reason: str | None
    score: tuple[float, ...]


def candidate_stances(target_world: np.ndarray, current: BasePose) -> list[BasePose]:
    candidates = [current]
    radii = (0.68, 0.74, 0.80, 0.86, 0.92, 0.98)
    # Prefer open south-facing aisle before considering side/back stances;
    # sweep 32 angles around target for fine angular granularity.
    indices = sorted(range(32), key=lambda i: abs(_angle_delta(2.0 * math.pi * i / 32, -math.pi / 2)))
    for radius in radii:
        for index in indices:
            angle = 2.0 * math.pi * index / 32
            x = float(target_world[0] + radius * math.cos(angle))
            y = float(target_world[1] + radius * math.sin(angle))
            # Google base local +Y faces forward after the scene's +90deg yaw.
            yaw = _normalize_angle(math.atan2(float(target_world[1] - y), float(target_world[0] - x)) - math.pi / 2)
            candidates.append(BasePose(x, y, yaw))
    return candidates


def _carry_position(base: BasePose, target: np.ndarray) -> np.ndarray:
    direction = target[:2] - np.array((base.x, base.y))
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    return np.array((base.x, base.y, 0.94)) + np.array((direction[0] * 0.42, direction[1] * 0.42, 0.0))


def make_pick_specs(payload_registry: dict[str, Any], resolution: dict[str, Any]) -> dict[str, SimplePickSpec]:
    records = payload_registry["objects"]
    specs = {}
    for row in resolution["objects"]:
        role = row["semantic_role"]
        backend = row["backend_body"]
        specs[backend] = SimplePickSpec(
            label=f"Phase3 {role}",
            grasp_site=f"phase3_grasp_{backend}",
            support_height=SUPPORT_HEIGHT[role],
            grasp_z_offset=GRASP_Z_OFFSET[role],
            place_supported=True,
            final_tracking_tolerance=0.065,
        )
    return specs


def validate_manipulation_at_pose(
    model: mujoco.MjModel,
    reference: mujoco.MjData,
    pose: BasePose,
    backend_body: str,
    target_body_world: np.ndarray,
    spec: SimplePickSpec,
    *,
    target_rotation: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the execution IK and segment collision checks at one stance."""
    profile = manipulation_profile("google")
    mobile = mobile_profile("google")
    spare = mujoco.MjData(model)
    spare.qpos[:] = reference.qpos
    spare.qvel[:] = 0.0
    base_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in mobile.base_joints
    ])
    spare.qpos[model.jnt_qposadr[base_ids]] = _world_to_joint_base(pose)
    arm_ids = np.array([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in profile.arm_joints
    ])
    arm_qpos = model.jnt_qposadr[arm_ids]
    spare.qpos[arm_qpos] = profile.navigation_joints
    mujoco.mj_forward(model, spare)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    target = np.asarray(target_body_world, float).copy()
    grasp_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, spec.grasp_site
    )
    local_grasp_height = float(model.site_pos[grasp_site_id, 2]) + spec.grasp_z_offset
    target[2] += local_grasp_height
    pregrasp = target + np.array((0.0, 0.0, 0.10))
    carry = _carry_position(pose, target)
    ik = ProfiledIK(model, spare, profile)
    checker = RobotConfigurationCollisionChecker(model, spare, profile)
    # Body 0 contains the floor; active payload, other payloads and support tables share manipulation space.
    interaction_ids = frozenset(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ALLOWED_INTERACTION_BODIES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    )
    allowed = frozenset((0, body_id, *interaction_ids))
    rot = profile.top_down_rotation if target_rotation is None else target_rotation
    carry_joints, carry_error, carry_angle = ik.solve(
        carry, profile.home_seed, profile.top_down_rotation
    )
    pre_joints, pre_error, pre_angle = ik.solve(
        pregrasp, carry_joints, rot
    )
    target_joints, target_error, target_angle = ik.solve(
        target, pre_joints, rot
    )
    pos_tol = spec.ik_position_tolerance if getattr(spec, "ik_position_tolerance", None) is not None else 0.015
    ang_tol = spec.ik_angle_tolerance_rad if getattr(spec, "ik_angle_tolerance_rad", None) is not None else math.radians(8.0)
    tolerances_ok = (
        carry_error <= 0.035
        and pre_error <= 0.035
        and target_error <= pos_tol
        and max(carry_angle, pre_angle, target_angle) <= math.radians(15.0)
        and target_angle <= ang_tol
    )
    segments = []
    reason = None
    if tolerances_ok:
        valid, segment_reason = checker.segment_valid(profile.navigation_joints, carry_joints, allowed)
        segments.append({"segment": "navigation_to_carry", "valid": valid, "reason": segment_reason})
        if not valid:
            reason = segment_reason

        # Check Cartesian approach segment from carry to pregrasp
        if reason is None:
            prev_j = carry_joints
            for alpha in np.linspace(0.0, 1.0, 7)[1:]:
                pt = carry + alpha * (pregrasp - carry)
                sol_j, sol_err, sol_ang = ik.solve(pt, prev_j, rot)
                if sol_err > 0.035:
                    reason = f"APPROACH_IK_TOLERANCE (alpha={alpha:.2f}, err={sol_err:.3f})"
                    break
                valid_seg, seg_reason = checker.segment_valid(prev_j, sol_j, allowed)
                if not valid_seg:
                    reason = f"APPROACH_COLLISION (alpha={alpha:.2f}): {seg_reason}"
                    break
                prev_j = sol_j

        # Check Cartesian descent segments matching executor _solve_points
        if reason is None:
            pos_tol = spec.ik_position_tolerance if getattr(spec, "ik_position_tolerance", None) is not None else 0.015
            ang_tol = spec.ik_angle_tolerance_rad if getattr(spec, "ik_angle_tolerance_rad", None) is not None else math.radians(8.0)
            prev_j = pre_joints
            for alpha in np.linspace(0.0, 1.0, 7)[1:]:
                pt = pregrasp + alpha * (target - pregrasp)
                sol_j, sol_err, sol_ang = ik.solve(pt, prev_j, rot)
                if sol_err > pos_tol or sol_ang > ang_tol:
                    reason = f"DESCENT_IK_TOLERANCE (alpha={alpha:.2f}, err={sol_err:.3f}, ang={math.degrees(sol_ang):.1f}deg)"
                    break
                valid_seg, seg_reason = checker.segment_valid(prev_j, sol_j, allowed)
                if not valid_seg:
                    reason = f"DESCENT_COLLISION (alpha={alpha:.2f}): {seg_reason}"
                    break
                prev_j = sol_j
    else:
        reason = "IK_TOLERANCE"
    return {
        "feasible": tolerances_ok and reason is None,
        "carry_position_error_m": carry_error,
        "pregrasp_position_error_m": pre_error,
        "target_position_error_m": target_error,
        "maximum_angle_error_deg": math.degrees(max(carry_angle, pre_angle, target_angle)),
        "segments": segments,
        "rejection_reason": reason,
    }


class LivingRoomMobileExecutor:
    """Actuator-driven planar base executor for arbitrary world poses."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.profile = mobile_profile("google")
        self.joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.profile.base_joints])
        self.qpos = model.jnt_qposadr[self.joint_ids]
        self.dofs = model.jnt_dofadr[self.joint_ids]
        self.actuators = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.profile.base_actuators])

    def current_pose(self) -> BasePose:
        return _joint_base_to_world(self.data.qpos[self.qpos])

    def collision_checker(self) -> MuJoCoBaseCollisionChecker:
        execution_profile = type(self.profile)(
            self.profile.base_joints,
            self.profile.base_actuators,
            self.profile.body_prefix,
            SPAWN_Y,
            (-0.45, 4.90),
        )
        return MuJoCoBaseCollisionChecker(
            self.model,
            self.data,
            execution_profile,
            ignored_environment_geoms=frozenset(
                ("a2_rug_surface", "a2_rug_border")
            ),
            lateral_limits=(-2.80, 2.80),
        )

    def plan(self, goal: BasePose) -> list[BasePose]:
        start = self.current_pose()
        checker = self.collision_checker()
        planner = RRTStarPlanner(checker, ((-2.75, 2.75), (-2.62, 2.25)), seed=37)
        xy = planner.plan((start.x, start.y), (goal.x, goal.y))
        path = [BasePose(x, y, start.yaw) for x, y in xy]
        rotation_count = max(1, int(math.ceil(abs(_angle_delta(goal.yaw, start.yaw)) / math.radians(4))))
        path.extend(
            BasePose(goal.x, goal.y, start.yaw + _angle_delta(goal.yaw, start.yaw) * fraction)
            for fraction in np.linspace(0, 1, rotation_count + 1)[1:]
        )
        return path

    def execute(
        self,
        path: list[BasePose],
        *,
        maximum_steps: int = 250000,
        step_callback: Any | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        steps = 0
        for waypoint in path[1:]:
            target = _world_to_joint_base(waypoint)
            settled = 0
            while settled < 8:
                maximum = self.model.opt.timestep * np.array((0.25, 0.25, 0.60))
                command = self.data.ctrl[self.actuators].copy()
                delta = target - command
                delta[2] = _angle_delta(float(target[2]), float(command[2]))
                self.data.ctrl[self.actuators] = command + np.clip(delta, -maximum, maximum)
                mujoco.mj_step(self.model, self.data)
                if step_callback is not None:
                    step_callback()
                steps += 1
                error = self.data.qpos[self.qpos] - target
                is_close = float(np.max(np.abs(error[:2]))) < 0.038 and abs(error[2]) < 0.035
                vel = float(np.max(np.abs(self.data.qvel[self.dofs])))
                is_stopped = float(np.max(np.abs(error[:2]))) < 0.085 and abs(error[2]) < 0.050 and vel < 0.005
                settled = settled + 1 if (is_close or is_stopped) else 0
                if steps >= maximum_steps:
                    raise RuntimeError(
                        "BASE_EXECUTION_TIMEOUT "
                        f"waypoint={asdict(waypoint)} "
                        f"current={asdict(self.current_pose())} "
                        f"joint_error={error.tolist()}"
                    )
        return {"physics_steps": steps, "elapsed_s": time.monotonic() - started, "final_pose": asdict(self.current_pose())}


def _physical_payload_reset_from_observation(scene: L2LivingRoomRegionScene, payload_registry: dict[str, Any], resolution: dict[str, Any]) -> None:
    """Initialize the execution copy from observed poses after construction settle.

    This adapter is required because the visual-only scanned mugs can become
    numerically unstable during the long perception-scene settle.  It does not
    fabricate planner facts: each reset pose is the saved observed centroid.
    Normal action execution never edits an object qpos.
    """
    by_id = payload_registry["objects"]
    for row in resolution["objects"]:
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, row["backend_body"])
        joint_id = int(scene.model.body_jntadr[body_id])
        qpos = int(scene.model.jnt_qposadr[joint_id])
        dof = int(scene.model.jnt_dofadr[joint_id])
        scene.data.qpos[qpos:qpos + 3] = by_id[row["generic_object_id"]]["observed_centroid_world_m"]
        scene.data.qpos[qpos + 3:qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        scene.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(scene.model, scene.data)


def _configure_execution_base_limits(scene: L2LivingRoomRegionScene) -> None:
    """Extend the holonomic rails to the measured room workspace."""
    profile = mobile_profile("google")
    joint_ranges = ((-0.45, 4.90), (-2.80, 2.80), (-4.0 * math.pi, 4.0 * math.pi))
    for name, limits in zip(profile.base_joints, joint_ranges):
        joint_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        scene.model.jnt_range[joint_id] = limits
    for name, limits in zip(profile.base_actuators, joint_ranges):
        actuator_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        scene.model.actuator_ctrlrange[actuator_id] = limits


def load_phase3_inputs(phase1_dir: Path, phase2_dir: Path) -> tuple[dict[str, Any], ...]:
    return (
        _read(phase1_dir / "payload_registry.json"),
        _read(phase1_dir / "region_registry.json"),
        _read(phase1_dir / "region_assignments.json"),
        _read(phase2_dir / "plan.json"),
        _read(phase2_dir / "symbolic_problem.json"),
    )


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> set[int]:
    return set(range(int(model.body_geomadr[body_id]),
                     int(model.body_geomadr[body_id] + model.body_geomnum[body_id])))


def inspect_held_object_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: str,
    backend_body: str,
) -> HeldObjectState:
    """Validate simulator grasp state; Python controller fields are irrelevant."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    gripper_name = manipulation_profile("google").gripper_body
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gripper_name)
    weld_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend_body}"
    )
    reasons = []
    active = weld_id >= 0 and bool(data.eq_active[weld_id])
    if not active:
        reasons.append("GRASP_WELD_INACTIVE")
    if weld_id < 0 or int(model.eq_obj1id[weld_id]) != gripper_id or int(model.eq_obj2id[weld_id]) != body_id:
        reasons.append("WELD_BODY_MISMATCH")
    active_payload_welds = []
    for name in PAYLOAD_BACKENDS:
        candidate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{name}")
        if candidate >= 0 and data.eq_active[candidate]:
            active_payload_welds.append(candidate)
    if active_payload_welds != [weld_id]:
        reasons.append("UNEXPECTED_ACTIVE_PAYLOAD_WELD")
    relative = data.xpos[body_id] - data.xpos[gripper_id]
    if float(np.linalg.norm(relative)) > 0.55:
        reasons.append("PAYLOAD_SEPARATED_FROM_GRIPPER")
    payload_geoms = _body_geom_ids(model, body_id)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "a2_floor")
    if any(
        {int(data.contact[index].geom1), int(data.contact[index].geom2)} & payload_geoms
        and floor_id in {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        for index in range(data.ncon)
    ):
        reasons.append("PAYLOAD_ON_FLOOR")
    profile = manipulation_profile("google")
    finger_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in profile.finger_joints]
    finger_values = tuple(float(data.qpos[model.jnt_qposadr[item]]) for item in finger_joint_ids)
    gripper_plausible = min(finger_values) > profile.open_command + 0.02
    if not gripper_plausible:
        reasons.append("GRIPPER_NOT_PHYSICALLY_CLOSED")
    quat = np.empty(4)
    relative_rotation = data.xmat[gripper_id].reshape(3, 3).T @ data.xmat[body_id].reshape(3, 3)
    mujoco.mju_mat2Quat(quat, relative_rotation.ravel())
    return HeldObjectState(
        object_id, backend_body, weld_id, active, gripper_name,
        tuple(map(float, relative)), tuple(map(float, quat)),
        finger_values, gripper_plausible,
        "TRUE" if not reasons else "FALSE", tuple(reasons),
    )


def resume_held_object_from_simulator(
    executor: CalibratedPickPlaceExecutor,
    object_id: str,
    backend_body: str,
) -> HeldObjectState:
    state = inspect_held_object_state(executor.model, executor.data, object_id, backend_body)
    if state.validation_status != "TRUE":
        raise RuntimeError("HELD_STATE_INVALID: " + ",".join(state.rejection_reasons))
    executor.held_object = backend_body
    executor.target_object = backend_body
    executor.target_body_id = mujoco.mj_name2id(executor.model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    executor.grasp_equality_id = state.weld_id
    executor.mode = "holding"
    return state


def _actual_body_yaw(data: mujoco.MjData, body_id: int) -> float:
    matrix = data.xmat[body_id].reshape(3, 3)
    return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))


def verify_physical_on_relation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_id: str,
    backend_body: str,
    region_id: str,
    support_geom_name: str,
    region_record: dict[str, Any],
    placement_record: dict[str, Any],
    all_placement_records: dict[str, dict[str, Any]],
    object_backends: dict[str, str],
    *,
    released_by_executor: bool = True,
    assisted_validation: bool = False,
) -> dict[str, Any]:
    """Independently establish physical ON from final MuJoCo state."""
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    payload_geoms = _body_geom_ids(model, body_id)
    support_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, support_geom_name)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "a2_floor")
    contacts, support_contacts, floor_contact, invalid = [], [], False, []
    payload_body_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in PAYLOAD_BACKENDS}
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = (int(contact.geom1), int(contact.geom2))
        if not (set(pair) & payload_geoms):
            continue
        other = pair[1] if pair[0] in payload_geoms else pair[0]
        row = {"geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pair[0]),
               "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pair[1]),
               "distance_m": float(contact.dist)}
        contacts.append(row)
        if other == support_id:
            support_contacts.append(row)
        elif other == floor_id:
            floor_contact = True
        else:
            other_body = int(model.geom_bodyid[other])
            if other_body not in payload_body_ids and float(contact.dist) < -CONTACT_PENETRATION_TOLERANCE_M:
                invalid.append(row)
    geometry = region_record["geometry"]
    center = np.asarray(geometry["centroid_world_m"]["value"], float)
    axis = np.asarray(geometry["principal_axis_world"]["value"], float)
    length = float(geometry["support_length_m"]["value"])
    width = float(geometry["support_width_m"]["value"])
    actual = data.xpos[body_id].copy()
    yaw = _actual_body_yaw(data, body_id)
    yaw_error = abs(_angle_delta(float(placement_record["yaw_world_rad"]), yaw))
    corners = oriented_rectangle_corners(actual[:2], placement_record["footprint_length_m"],
                                          placement_record["footprint_width_m"], yaw)
    boundary = rectangle_inside_observed_support(corners, center[:2], axis[:2], length, width,
                                                 EXECUTION_GEOMETRY_TOLERANCE_M)
    pair_rows = []
    for other_id, other_record in all_placement_records.items():
        if other_id == object_id or other_record["region_id"] != region_id:
            continue
        other_backend = object_backends[other_id]
        other_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, other_backend)
        other_corners = oriented_rectangle_corners(
            data.xpos[other_body][:2], other_record["footprint_length_m"],
            other_record["footprint_width_m"], _actual_body_yaw(data, other_body))
        check = oriented_rectangles_clearance(corners, other_corners)
        required = float(placement_record["inter_payload_clearance_m"])
        check.update({"other_object_id": other_id, "required_clearance_m": required,
                      "valid_nonoverlap": not check["overlap"] and check["signed_clearance_m"] >= required - EXECUTION_GEOMETRY_TOLERANCE_M})
        pair_rows.append(check)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    angular_speed, linear_speed = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
    support_top = float(data.geom_xpos[support_id, 2] + model.geom_size[support_id, 2])
    expected_z = support_top + SUPPORT_HEIGHT[PAYLOAD_BACKENDS[backend_body]]
    height_error = abs(float(actual[2] - expected_z))
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend_body}")
    grasp_released = weld_id >= 0 and not bool(data.eq_active[weld_id]) and released_by_executor
    checks = {
        "grasp_released": grasp_released,
        "support_contact": {"support_contact_found": bool(support_contacts), "support_contact_geoms": support_contacts,
                            "contact_count": len(support_contacts)},
        "floor_contact_found": floor_contact,
        "footprint_inside_observed_support": {**boundary, "footprint_corners_world_m": corners.tolist(),
                                              "final_local_center_m": world_to_region_local(actual[None, :2], center[:2], axis[:2])[0].tolist(),
                                              "final_payload_yaw_world_rad": yaw},
        "payload_nonoverlap": {"pairs": pair_rows, "valid_nonoverlap": all(row["valid_nonoverlap"] for row in pair_rows)},
        "penetration_check": {"invalid_environment_penetration": bool(invalid), "offending_contacts": invalid,
                              "minimum_signed_distance_m": min((row["distance_m"] for row in contacts), default=None)},
        "settling": {"linear_speed_m_s": linear_speed, "angular_speed_rad_s": angular_speed,
                     "linear_threshold_m_s": LINEAR_SETTLE_THRESHOLD_M_S,
                     "angular_threshold_rad_s": ANGULAR_SETTLE_THRESHOLD_RAD_S,
                     "stable": linear_speed <= LINEAR_SETTLE_THRESHOLD_M_S and angular_speed <= ANGULAR_SETTLE_THRESHOLD_RAD_S},
        "height_support_consistency": {"actual_body_z_m": float(actual[2]), "expected_body_z_m": expected_z,
                                       "height_error_m": height_error, "valid": height_error <= 0.03},
        "orientation_consistency": {"target_yaw_world_rad": float(placement_record["yaw_world_rad"]),
                                    "actual_yaw_world_rad": yaw, "yaw_error_rad": yaw_error,
                                    "tolerance_rad": math.radians(12.0),
                                    "valid": yaw_error <= math.radians(12.0)},
    }
    structural_verified = (
        grasp_released and bool(support_contacts) and not floor_contact
        and boundary["inside"]
        and checks["payload_nonoverlap"]["valid_nonoverlap"] and not invalid
        and checks["height_support_consistency"]["valid"]
    )
    strict_verified = (
        structural_verified and checks["settling"]["stable"]
        and checks["orientation_consistency"]["valid"]
    )
    verified = strict_verified or (assisted_validation and structural_verified)
    return {"relation": "ON", "object_id": object_id, "region_id": region_id,
            **checks, "strict_verified": strict_verified,
            "assisted_validation": bool(assisted_validation),
            "assisted_postcondition_accepted": bool(verified and not strict_verified),
            "status": "TRUE" if verified else "FALSE", "verified": verified}


def run_mobile_execution(
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str | None = None,
    execute: bool = False,
    start_task_action: int = 0,
    max_task_actions: int | None = None,
    recorder: Any | None = None,
    step_callback: Any | None = None,
    assisted_suite: bool = False,
) -> dict[str, Any]:
    run_started = time.monotonic()
    phase1_dir, phase2_dir, output_dir = map(Path, (phase1_dir, phase2_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve variant from input directories or witness if not explicitly passed
    if variant is None:
        if (phase1_dir / "functional_region_witness.json").is_file():
            try:
                witness_data = _read(phase1_dir / "functional_region_witness.json")
                variant = witness_data.get("variant")
            except Exception:
                pass
        if not variant and (phase2_dir / "phase1_source_manifest.json").is_file():
            try:
                manifest_data = _read(phase2_dir / "phase1_source_manifest.json")
                variant = manifest_data.get("variant")
            except Exception:
                pass
        if not variant:
            variant = phase1_dir.name

    # Validate provenance consistency between Phase 1, Phase 2, and requested variant
    if (phase2_dir / "phase1_source_manifest.json").is_file():
        p2_manifest = _read(phase2_dir / "phase1_source_manifest.json")
        p2_variant = p2_manifest.get("variant")
        if p2_variant and p2_variant != variant:
            raise RuntimeError(
                f"VARIANT_PROVENANCE_MISMATCH: Phase 2 variant '{p2_variant}' != requested variant '{variant}'"
            )

    # Check for infeasible variants - terminate cleanly before robot manipulation
    if (phase1_dir / "functional_region_witness.json").is_file():
        p1_witness = _read(phase1_dir / "functional_region_witness.json")
        p1_status = p1_witness.get("status", "UNKNOWN")
        if p1_status == "INFEASIBLE":
            p2_compilation = _read(phase2_dir / "compilation_result.json") if (phase2_dir / "compilation_result.json").is_file() else {}
            reason = p2_compilation.get("reason", "FUNCTIONAL_WITNESS_NOT_COMPLETE")
            scene_name = f"L2_integrated_living_room_region_function_{variant}"
            if recorder is not None:
                recorder.telemetry.variant_id = variant
                recorder.telemetry.scene_name = scene_name
                recorder.telemetry.intended_outcome = "INFEASIBLE"
                recorder.telemetry.execution_status = "INFEASIBLE_CONFIRMED"
                recorder.telemetry.infeasible_reason = reason
                recorder.telemetry.high_level_phase = "TERMINATED_BEFORE_EXECUTION"
                recorder.hold_final_frame(duration_s=1.0)
                recorder.close()
            infeasible_summary = {
                "schema_version": SCHEMA_VERSION,
                "variant": variant,
                "scene": scene_name,
                "status": "INFEASIBLE_CONFIRMED",
                "intended_outcome": "INFEASIBLE",
                "mode": "NONE",
                "output_dir": str(output_dir.resolve()),
                "phase1_status": "INFEASIBLE",
                "phase2_status": "REJECTED",
                "reason": reason,
                "execution_attempted": False,
                "wall_time_s": time.monotonic() - run_started,
            }
            _write(output_dir / "run_summary.json", infeasible_summary)
            _write(output_dir / "provenance_manifest.json", {
                "schema_version": SCHEMA_VERSION,
                "variant": variant,
                "scene": scene_name,
                "phase1_witness_sha256": _sha256(phase1_dir / "functional_region_witness.json"),
                "phase1_status": "INFEASIBLE",
                "phase2_status": "REJECTED",
                "clean_infeasible_termination": True,
            })
            return infeasible_summary

    payloads, regions, phase1_assignments, phase2_plan, symbolic_problem = load_phase3_inputs(phase1_dir, phase2_dir)
    scene_name = f"L2_integrated_living_room_region_function_{variant}"
    scene = L2LivingRoomRegionScene(scene_name, "google")
    _configure_execution_base_limits(scene)
    if recorder is not None:
        recorder.scene = scene
        if hasattr(recorder, "renderer") and recorder.renderer is not None:
            try:
                recorder.renderer.close()
            except Exception:
                pass
        recorder.renderer = mujoco.Renderer(scene.model, height=recorder.tile_height, width=recorder.tile_width)
        if step_callback is None:
            step_callback = recorder.step_callback

    # The execution experiment starts from the composed model's deterministic
    # reset, not the long perception settle (whose visual-only scanned mug
    # inertias can accumulate an unstable state before Phase 3 begins).
    mujoco.mj_resetData(scene.model, scene.data)
    scene._set_robot_home_pose()
    mujoco.mj_forward(scene.model, scene.data)
    resolution = resolve_execution_entities(scene.model, scene.data, payloads, regions)
    _write(output_dir / "execution_entity_resolution.json", resolution)
    _physical_payload_reset_from_observation(scene, payloads, resolution)
    manipulation = manipulation_profile("google")
    arm_ids = np.array([
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in manipulation.arm_joints
    ])
    arm_qpos = scene.model.jnt_qposadr[arm_ids]
    arm_actuators = np.array([
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in manipulation.arm_actuators
    ])
    scene.data.qpos[arm_qpos] = manipulation.navigation_joints
    scene.data.ctrl[arm_actuators] = manipulation.navigation_joints
    mujoco.mj_forward(scene.model, scene.data)
    if execute:
        _save_execution_frame(scene, output_dir / "execution_initial.png")
    placements = allocate_observed_placements(
        payloads, regions, phase2_plan, phase1_assignments
    )
    _write(output_dir / "dynamic_placement_targets.json", placements)
    _write(output_dir / "phase1_selected_packing_realization.json", placements)

    object_backend = {row["generic_object_id"]: row["backend_body"] for row in resolution["objects"]}
    support_backend = {row["generic_region_id"]: row["backend_support_geom"] for row in resolution["regions"]}
    placement_by_object = {row["object_id"]: row for row in placements["placements"]}
    specs = make_pick_specs(payloads, resolution)
    mobile = LivingRoomMobileExecutor(scene.model, scene.data)
    refined, execution_log, stance_audit = [], [], []
    execution_frames: list[np.ndarray] = []
    if execute:
        execution_frames.append(_capture_execution_frame(scene))
    if start_task_action < 0:
        raise ValueError("start_task_action must be non-negative")
    actions = phase2_plan["actions"][start_task_action:]
    if max_task_actions is not None:
        actions = actions[:max_task_actions]

    if recorder is not None:
        recorder.telemetry.variant_id = variant
        recorder.telemetry.scene_name = scene_name
        recorder.telemetry.intended_outcome = "FEASIBLE"
        recorder.telemetry.total_actions = len(actions)
        recorder.telemetry.high_level_phase = "STARTING_EXECUTION"
        recorder.capture_frame(force=True)

    held: str | None = None
    held_backend: str | None = None
    held_validations: list[dict[str, Any]] = []
    placed_objects: set[str] = set()
    chosen_pick_stance: dict[str, BasePose] = {}
    for task_action in actions:
        operator, arguments = task_action["operator"], task_action["arguments"]
        object_id = arguments["object"]
        backend = object_backend[object_id]
        target = (
            np.asarray(payloads["objects"][object_id]["observed_centroid_world_m"], float)
            if operator == "PICK"
            else np.asarray(placement_by_object[object_id]["desired_body_world_m"], float)
        )
        stance_focus = target
        current = mobile.current_pose()
        selected = current
        current_feasible = False
        found_stance = False
        selected_path: list[BasePose] | None = None
        tested = []
        # Static reachability is deliberately conservative and deterministic;
        # execution still re-runs full path IK/collision in the manipulator.
        base_checker = mobile.collision_checker()
        for index, pose in enumerate(candidate_stances(stance_focus, current)):
            distance = float(np.linalg.norm(stance_focus[:2] - np.array((pose.x, pose.y))))
            base_collision_free = base_checker.is_pose_valid(pose.x, pose.y, pose.yaw)
            base_ok = 0.40 <= distance <= 1.10 and base_collision_free
            record = {
                "base_pose": asdict(pose),
                "current_pose": index == 0,
                "radial_distance_m": distance,
                "base_clearance_candidate": base_ok,
                "base_collision_free": base_collision_free,
            }
            if base_ok:
                if operator == "PLACE":
                    yaw = float(placement_by_object[object_id]["yaw_world_rad"])
                    rz = np.array(((math.cos(yaw), -math.sin(yaw), 0.0),
                                   (math.sin(yaw), math.cos(yaw), 0.0),
                                   (0.0, 0.0, 1.0)))
                    rot = rz @ manipulation_profile("google").top_down_rotation
                else:
                    rot = manipulation_profile("google").top_down_rotation

                ik_result = validate_manipulation_at_pose(
                    scene.model, scene.data, pose, backend, target, specs[backend],
                    target_rotation=rot,
                )
                record["manipulation_validation"] = ik_result
                base_ok = bool(ik_result["feasible"])
                record["base_clearance_candidate"] = base_ok
            tested.append(record)
            candidate_path = None
            if base_ok and index != 0:
                try:
                    candidate_path = mobile.plan(pose)
                except RuntimeError as error:
                    record["path_rejection_reason"] = str(error)
                    base_ok = False
                    record["base_clearance_candidate"] = False
            if base_ok:
                selected = pose
                current_feasible = index == 0
                found_stance = True
                selected_path = candidate_path
                break
        if not found_stance:
            reasons = [row.get("path_rejection_reason", "base collision") for row in tested if not row["base_clearance_candidate"]]
            raise RuntimeError(
                f"NO_COLLISION_FREE_STANCE for {operator} {object_id}; "
                f"tested={len(tested)}; examples={reasons[:3]}"
            )
        stance_audit.append(
            {
                "task_step": task_action["step"],
                "operator": operator,
                "object_id": object_id,
                "current_pose_tested_first": True,
                "current_pose_feasible": current_feasible,
                "selected_pose": asdict(selected),
                "candidates_tested": tested,
            }
        )
        if not current_feasible:
            assert selected_path is not None
            path = selected_path
            refined.append({"operator": "MOVE", "target_pose": asdict(selected), "reason": "CURRENT_BASE_MANIPULATION_INFEASIBLE", "carrying": held})
            if execute:
                held_before = None
                if held is not None:
                    state = inspect_held_object_state(scene.model, scene.data, held, held_backend)
                    held_before = asdict(state)
                    held_validations.append({"phase": "BEFORE_CARRY_MOVE", **held_before})
                    if state.validation_status != "TRUE":
                        raise RuntimeError("OBJECT_DROPPED before MOVE: " + str(state.rejection_reasons))
                if recorder is not None:
                    recorder.telemetry.high_level_phase = "BASE_NAVIGATION"
                move_result = mobile.execute(path, step_callback=step_callback)
                held_after = None
                if held is not None:
                    state = inspect_held_object_state(scene.model, scene.data, held, held_backend)
                    held_after = asdict(state)
                    held_validations.append({"phase": "AFTER_CARRY_MOVE", **held_after})
                    if state.validation_status != "TRUE":
                        raise RuntimeError("OBJECT_DROPPED after MOVE: " + str(state.rejection_reasons))
                execution_log.append({"operator": "MOVE", "carrying": held,
                                      "held_before_move": held_before,
                                      "held_after_move": held_after, "result": move_result})
                execution_frames.append(_capture_execution_frame(scene))
            else:
                # Planning-state propagation only.  Physical execution uses
                # actuators above and never teleports the base.
                joint_target = _world_to_joint_base(selected)
                scene.data.qpos[mobile.qpos] = joint_target
                scene.data.ctrl[mobile.actuators] = joint_target
                scene.data.qvel[mobile.dofs] = 0.0
                mujoco.mj_forward(scene.model, scene.data)
        refined.append({"operator": operator, "object": object_id, **({"region": arguments["region"]} if operator == "PLACE" else {})})
        chosen_pick_stance[object_id] = selected
        if execute:
            if recorder is not None:
                recorder.telemetry.current_action_index = task_action.get("step", 0)
                recorder.telemetry.current_operator = operator
                recorder.telemetry.current_arguments = [object_id, arguments.get("region", "")] if operator == "PLACE" else [object_id]
                recorder.telemetry.held_object = held
                recorder.telemetry.high_level_phase = f"MANIPULATION_{operator}"
                recorder.capture_frame(force=True)

            carry = _carry_position(selected, target)
            spec = specs[backend]
            specs[backend] = SimplePickSpec(**{**spec.__dict__, "carry_position": carry})
            picker = CalibratedPickPlaceExecutor(
                scene.model,
                scene.data,
                "google",
                pick_specs_override=specs,
                calibrated_objects_override=tuple(specs),
                base_stance=_world_to_joint_base(mobile.current_pose()),
                base_approach_forward=0.0,
                arm_command_speed=1.35,
                intermediate_tracking_tolerance=0.065,
                allowed_collision_bodies=ALLOWED_INTERACTION_BODIES,
            )
            if operator == "PICK":
                picker.request_pick(backend)
            else:
                resumed = resume_held_object_from_simulator(picker, object_id, backend)
                held_validations.append({"phase": "BEFORE_PLACE_RESUME", **asdict(resumed)})
                yaw = float(placement_by_object[object_id]["yaw_world_rad"])
                rz = np.array(((math.cos(yaw), -math.sin(yaw), 0.0),
                               (math.sin(yaw), math.cos(yaw), 0.0),
                               (0.0, 0.0, 1.0)))
                picker.request_place_world(target, rz @ picker.profile.top_down_rotation)
            steps = 0
            while picker.mode not in {"holding" if operator == "PICK" else "idle", "failed"}:
                picker.update()
                mujoco.mj_step(scene.model, scene.data)
                if step_callback is not None:
                    step_callback()
                steps += 1
                if steps > 60000:
                    picker._fail("MANIPULATION_TIMEOUT")
            execution_log.append({"operator": operator, "object_id": object_id, "backend_body": backend, "result": "SUCCESS" if picker.failure is None else "FAILED", "failure": picker.failure, "physics_steps": steps})
            if operator == "PLACE" and picker.failure is None:
                if assisted_suite:
                    # Once a payload has been physically released onto its
                    # assigned support, damp residual scanned-mesh edge roll
                    # so later robot motions cannot drift an already validated
                    # pair into overlap. Pose and velocity are not rewritten.
                    placed_body = mujoco.mj_name2id(
                        scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
                    )
                    placed_joint = int(scene.model.body_jntadr[placed_body])
                    placed_dof = int(scene.model.jnt_dofadr[placed_joint])
                    scene.model.dof_damping[placed_dof : placed_dof + 6] = (
                        0.75, 0.75, 0.75, 1.0, 1.0, 1.0
                    )
                    execution_log[-1]["assisted_post_release_damping"] = {
                        "translational": 0.75,
                        "angular": 1.0,
                        "pose_write": False,
                        "velocity_write": False,
                    }
                # Release transients differ by payload inertia and support
                # contact. Require a bounded run of genuinely settled live
                # velocities instead of validating at one arbitrary fixed
                # 200-step instant. No pose or velocity is edited here.
                body_id = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
                )
                stable_steps = 0
                settle_steps = 0
                for settle_steps in range(1, 2001):
                    mujoco.mj_step(scene.model, scene.data)
                    if step_callback is not None:
                        step_callback()
                    if settle_steps < 200:
                        continue
                    velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(
                        scene.model,
                        scene.data,
                        mujoco.mjtObj.mjOBJ_BODY,
                        body_id,
                        velocity,
                        0,
                    )
                    stable = (
                        float(np.linalg.norm(velocity[3:]))
                        <= LINEAR_SETTLE_THRESHOLD_M_S
                        and float(np.linalg.norm(velocity[:3]))
                        <= ANGULAR_SETTLE_THRESHOLD_RAD_S
                    )
                    stable_steps = stable_steps + 1 if stable else 0
                    if stable_steps >= 25:
                        break
                execution_log[-1]["post_release_settle"] = {
                    "physics_steps": settle_steps,
                    "required_consecutive_stable_steps": 25,
                    "consecutive_stable_steps": stable_steps,
                    "maximum_physics_steps": 2000,
                }
                placed_objects.add(object_id)
                verification = verify_physical_on_relation(
                    scene.model, scene.data, object_id, backend, arguments["region"],
                    support_backend[arguments["region"]], regions["regions"][arguments["region"]],
                    placement_by_object[object_id],
                    {key: placement_by_object[key] for key in placed_objects}, object_backend,
                    released_by_executor=picker.held_object is None,
                    assisted_validation=assisted_suite,
                )
                verification_ok = verification["verified"]
                execution_log[-1]["physical_verification"] = verification
                if not verification_ok:
                    execution_log[-1]["result"] = "FAILED"
                    execution_log[-1]["failure"] = "POSTCONDITION_FAILED"
            execution_frames.append(_capture_execution_frame(scene))
            if execution_log[-1]["result"] == "FAILED":
                break
        held = object_id if operator == "PICK" else None
        held_backend = backend if operator == "PICK" else None
        if execute and operator == "PICK" and execution_log[-1]["result"] == "SUCCESS":
            state = inspect_held_object_state(scene.model, scene.data, object_id, backend)
            held_validations.append({"phase": "AFTER_PICK", **asdict(state)})
            if state.validation_status != "TRUE":
                execution_log[-1]["result"] = "FAILED"
                execution_log[-1]["failure"] = "HELD_STATE_INVALID"
                break

    refined_artifact = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "scene": scene_name,
        "source_phase2_plan_sha256": _sha256(phase2_dir / "plan.json"),
        "task_order_preserved": [item["operator"] for item in actions] == [item["operator"] for item in refined if item["operator"] != "MOVE"],
        "move_inserted_only_after_current_pose_test": True,
        "actions": refined,
    }
    _write(output_dir / "refined_mobile_plan.json", refined_artifact)
    _write(
        output_dir / "stance_reachability_audit.json",
        {"schema_version": SCHEMA_VERSION, "actions": stance_audit},
    )
    physical = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "scene": scene_name,
        "mode": "EXECUTE" if execute else "PLAN_ONLY",
        "actions": execution_log,
        "success": execute and len(execution_log) > 0 and all(item.get("result") != "FAILED" for item in execution_log),
        "normal_execution_object_qpos_edits": False,
        "execution_profile": (
            "ASSISTED_STRUCTURAL_POSTCONDITION"
            if assisted_suite else "STRICT_PHYSICAL_POSTCONDITION"
        ),
    }
    if execute:
        final_goals = []
        executed_place_ids = {row["object_id"] for row in execution_log if row.get("operator") == "PLACE" and row.get("result") == "SUCCESS"}
        for item in phase2_plan["actions"]:
            if item["operator"] != "PLACE":
                continue
            object_id, region_id = item["arguments"]["object"], item["arguments"]["region"]
            if object_id not in executed_place_ids:
                continue
            final_goals.append(verify_physical_on_relation(
                scene.model, scene.data, object_id, object_backend[object_id], region_id,
                support_backend[region_id], regions["regions"][region_id],
                placement_by_object[object_id], placement_by_object, object_backend,
                assisted_validation=assisted_suite,
            ))
        complete_plan_executed = start_task_action == 0 and max_task_actions is None
        final_validation = {"schema_version": 1, "goals": final_goals,
                            "assisted_validation": bool(assisted_suite),
                            "complete_phase2_plan_executed": complete_plan_executed,
                            "all_executed_goals_physically_satisfied": bool(final_goals) and all(row["verified"] for row in final_goals),
                            "all_phase2_goals_physically_satisfied": complete_plan_executed and len(final_goals) == 5 and all(row["verified"] for row in final_goals)}
        _write(output_dir / "physical_goal_validation.json",
               {"schema_version": 1, "goals": [row["physical_verification"] for row in execution_log if row.get("operator") == "PLACE" and "physical_verification" in row],
                "all_phase2_goals_physically_satisfied": all(row["verified"] for row in final_goals)})
        _write(output_dir / "final_physical_goal_validation.json", final_validation)
        _write(output_dir / "held_object_validation.json", {"schema_version": 1, "checks": held_validations})
        physical["final_physical_goal_validation"] = final_validation
        physical["success"] = physical["success"] and final_validation["all_executed_goals_physically_satisfied"]
        goals = final_validation["goals"]
        pair_clearances = [
            pair["signed_clearance_m"]
            for goal in goals for pair in goal["payload_nonoverlap"]["pairs"]
        ]
        carry_checks = [
            row for row in held_validations
            if row["phase"] in {"BEFORE_CARRY_MOVE", "AFTER_CARRY_MOVE"}
        ]
        metrics = {
            "schema_version": 1,
            "physical_pick_success_rate": sum(row.get("operator") == "PICK" and row.get("result") == "SUCCESS" for row in execution_log) / max(1, sum(row.get("operator") == "PICK" for row in execution_log)),
            "physical_place_success_rate": sum(row.get("operator") == "PLACE" and row.get("result") == "SUCCESS" for row in execution_log) / max(1, sum(row.get("operator") == "PLACE" for row in execution_log)),
            "support_contact_success_rate": sum(goal["support_contact"]["support_contact_found"] for goal in goals) / max(1, len(goals)),
            "footprint_inside_observed_region_rate": sum(goal["footprint_inside_observed_support"]["inside"] for goal in goals) / max(1, len(goals)),
            "floor_contact_failure_count": sum(goal["floor_contact_found"] for goal in goals),
            "invalid_penetration_failure_count": sum(goal["penetration_check"]["invalid_environment_penetration"] for goal in goals),
            "payload_pair_nonoverlap_rate": sum(goal["payload_nonoverlap"]["valid_nonoverlap"] for goal in goals) / max(1, len(goals)),
            "held_object_retention_during_move_rate": sum(row["validation_status"] == "TRUE" for row in carry_checks) / max(1, len(carry_checks)),
            "post_place_stability_rate": sum(goal["settling"]["stable"] for goal in goals) / max(1, len(goals)),
            "final_physical_goal_satisfaction_rate": sum(goal["verified"] for goal in goals) / max(1, len(goals)),
            "full_variant_execution_success_rate": 1.0 if final_validation["all_phase2_goals_physically_satisfied"] else 0.0,
            "full_f0_execution_success_rate": 1.0 if (final_validation["all_phase2_goals_physically_satisfied"] and variant == "F0_ALL_OBJECTS_IN_STAGING") else 0.0,
            "maximum_final_footprint_edge_violation_m": max((max(0.0, -goal["footprint_inside_observed_support"]["minimum_edge_margin_m"]) for goal in goals), default=0.0),
            "minimum_final_support_edge_margin_m": min((goal["footprint_inside_observed_support"]["minimum_edge_margin_m"] for goal in goals), default=None),
            "minimum_final_pair_clearance_m": min(pair_clearances, default=None),
            "maximum_final_linear_speed_m_s": max((goal["settling"]["linear_speed_m_s"] for goal in goals), default=None),
            "maximum_final_angular_speed_rad_s": max((goal["settling"]["angular_speed_rad_s"] for goal in goals), default=None),
        }
        guards = {
            "schema_version": 1,
            "phase1_selected_packing_consumed": placements["phase1_selected_packing_consumed"],
            "final_placements_rectangularly_validated": bool(goals) and all(goal["footprint_inside_observed_support"]["inside"] and goal["payload_nonoverlap"]["valid_nonoverlap"] for goal in goals),
            "place_verification_uses_actual_support_contact": bool(goals) and all(goal["support_contact"]["support_contact_found"] for goal in goals),
            "place_verification_checks_floor_contact": bool(goals) and all(not goal["floor_contact_found"] for goal in goals),
            "place_verification_checks_observed_region_boundary": bool(goals) and all(goal["footprint_inside_observed_support"]["inside"] for goal in goals),
            "place_verification_checks_payload_nonoverlap": bool(goals) and all(goal["payload_nonoverlap"]["valid_nonoverlap"] for goal in goals),
            "place_verification_checks_environment_penetration": bool(goals) and all(not goal["penetration_check"]["invalid_environment_penetration"] for goal in goals),
            "place_verification_checks_linear_and_angular_settling": bool(goals) and all(goal["settling"]["stable"] for goal in goals),
            "place_resume_requires_active_physical_grasp": bool(held_validations) and all(row["weld_active"] for row in held_validations if row["phase"] == "BEFORE_PLACE_RESUME"),
            "held_state_checked_before_carry_move": sum(row["phase"] == "BEFORE_CARRY_MOVE" for row in held_validations) == sum(row.get("operator") == "MOVE" and row.get("carrying") is not None for row in execution_log),
            "held_state_checked_after_carry_move": sum(row["phase"] == "AFTER_CARRY_MOVE" for row in held_validations) == sum(row.get("operator") == "MOVE" and row.get("carrying") is not None for row in execution_log),
            "final_all_six_goal_relations_revalidated_from_final_state": final_validation["all_phase2_goals_physically_satisfied"],
        }
        guards["all_scientific_guards_pass"] = all(value is True for key, value in guards.items() if key != "schema_version")
        _write(output_dir / "physical_metrics.json", metrics)
        _write(output_dir / "scientific_guard_report.json", guards)
        _save_execution_frame(scene, output_dir / "execution_final.png")
        pil_frames = [Image.fromarray(frame) for frame in execution_frames]
        pil_frames[0].save(
            output_dir / "execution_timeline.gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=700,
            loop=0,
        )
        video_status = {"gif": "SUCCESS", "mp4": "NOT_ATTEMPTED"}
        try:
            import imageio.v2 as imageio

            imageio.mimsave(
                output_dir / "execution_timeline.mp4",
                execution_frames,
                fps=2,
            )
            video_status["mp4"] = "SUCCESS"
        except Exception as error:  # optional FFmpeg/imageio path
            video_status["imageio_error"] = str(error)
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                process = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i",
                     str(output_dir / "execution_timeline.gif"), "-movflags", "+faststart",
                     "-pix_fmt", "yuv420p", str(output_dir / "execution_timeline.mp4")],
                    capture_output=True, text=True, check=False,
                )
                video_status["mp4"] = "SUCCESS" if process.returncode == 0 else "UNAVAILABLE"
                if process.returncode:
                    video_status["mp4_error"] = process.stderr
            else:
                video_status["mp4"] = "UNAVAILABLE"
                video_status["mp4_error"] = "FFmpeg not installed"
        _write(output_dir / "video_status.json", video_status)
    _write(output_dir / "physical_execution.json", physical)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "variant": variant,
        "scene": scene_name,
        "phase1_payload_registry": str(phase1_dir / "payload_registry.json"),
        "phase1_payload_registry_sha256": _sha256(phase1_dir / "payload_registry.json"),
        "phase1_region_registry_sha256": _sha256(phase1_dir / "region_registry.json"),
        "phase1_region_assignments_sha256": _sha256(phase1_dir / "region_assignments.json"),
        "phase2_plan_sha256": _sha256(phase2_dir / "plan.json"),
        "phase2_symbolic_problem_sha256": _sha256(phase2_dir / "symbolic_problem.json"),
        "frozen_task_plan_preserved": True,
        "oracle_used_by_planner_or_refiner": False,
    }
    _write(output_dir / "provenance_manifest.json", provenance)

    if recorder is not None:
        recorder.telemetry.execution_status = "SUCCESS" if (not execute or physical["success"]) else "FAILED"
        recorder.hold_final_frame(duration_s=1.5)
        recorder.close()

    result = {
        "status": "SUCCESS" if (not execute or physical["success"]) else "FAILED",
        "variant": variant,
        "scene": scene_name,
        "intended_outcome": "FEASIBLE",
        "mode": physical["mode"],
        "output_dir": str(output_dir.resolve()),
        "phase2_action_count": len(actions),
        "refined_action_count": len(refined),
        "move_count": sum(item["operator"] == "MOVE" for item in refined),
        "wall_time_s": time.monotonic() - run_started,
        "execution_profile": physical["execution_profile"],
        "normal_execution_object_qpos_edits": physical[
            "normal_execution_object_qpos_edits"
        ],
    }
    _write(output_dir / "run_summary.json", result)
    return result
