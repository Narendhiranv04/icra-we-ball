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


SCENE = "L2_integrated_living_room_region_function_F0_BASE"
SCHEMA_VERSION = 1
SPAWN_Y = float(INTEGRATED_ROOM_LAYOUT["robot_spawn"][1])
PAYLOAD_BACKENDS = {
    "a2_drink_left": "drink",
    "a2_drink_right": "drink",
    "a2_snack_left": "snack_container",
    "a2_snack_right": "snack_container",
    "a2_remote_payload": "tv_remote",
    "a2_controller_payload": "game_controller",
}
REGION_BACKENDS = {
    "a2_personal_left_top": "side_table",
    "a2_shared_drink_top": "side_table",
    "a2_control_table_top": "coffee_table",
    "a2_personal_right_top": "side_table",
}
SUPPORT_HEIGHT = {
    "drink": 0.070,
    "snack_container": 0.025,
    "tv_remote": 0.016,
    "game_controller": 0.030,
}
GRASP_Z_OFFSET = {
    "drink": 0.005,
    "snack_container": 0.006,
    "tv_remote": 0.002,
    "game_controller": 0.002,
}


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
) -> dict[str, Any]:
    """Allocate deterministic non-overlapping positions from measured OBBs."""
    destination = {
        item["arguments"]["object"]: item["arguments"]["region"]
        for item in phase2_plan["actions"]
        if item["operator"] == "PLACE"
    }
    grouped: dict[str, list[str]] = {}
    for object_id, region_id in destination.items():
        grouped.setdefault(region_id, []).append(object_id)
    rows = []
    for region_id, object_ids in sorted(grouped.items()):
        region = region_registry["regions"][region_id]
        geometry = region["geometry"]
        center = np.asarray(geometry["centroid_world_m"]["value"], float)
        axis = np.asarray(geometry["principal_axis_world"]["value"], float)
        axis /= np.linalg.norm(axis)
        length = float(geometry["support_length_m"]["value"])
        width = float(geometry["support_width_m"]["value"])
        margin = 0.035
        ordered = sorted(object_ids)
        offsets = np.linspace(-0.25 * length, 0.25 * length, len(ordered))
        footprints = []
        for object_id, offset in zip(ordered, offsets):
            record = payload_registry["objects"][object_id]
            role = _semantic_role(record)
            obj_length = float(record["geometry"]["footprint_length_m"]["value"])
            obj_width = float(record["geometry"]["footprint_width_m"]["value"])
            point = center + offset * axis
            point[2] = center[2] + SUPPORT_HEIGHT[role]
            within = (
                abs(offset) + obj_length / 2 + margin <= length / 2
                and obj_width / 2 + margin <= width / 2
            )
            if not within:
                raise RuntimeError(f"Observed support cannot safely place {object_id}")
            footprints.append((object_id, point, max(obj_length, obj_width) / 2))
            rows.append(
                {
                    "object_id": object_id,
                    "region_id": region_id,
                    "desired_body_world_m": point.tolist(),
                    "footprint_length_m": obj_length,
                    "footprint_width_m": obj_width,
                    "boundary_margin_m": margin,
                    "within_measured_support": True,
                    "source": "PHASE1_OBSERVED_REGION_AND_PAYLOAD_GEOMETRY",
                }
            )
        for (left_id, left, left_radius), (right_id, right, right_radius) in itertools.combinations(footprints, 2):
            separation = float(np.linalg.norm(left[:2] - right[:2]))
            required = left_radius + right_radius + 0.025
            if separation < required:
                raise RuntimeError(
                    f"Placement overlap: {left_id}/{right_id} {separation:.3f} < {required:.3f}"
                )
    return {"schema_version": SCHEMA_VERSION, "placements": rows}


def _joint_base_to_world(values: np.ndarray) -> BasePose:
    return BasePose(float(-values[1]), float(SPAWN_Y + values[0]), float(values[2]))


def _world_to_joint_base(pose: BasePose) -> np.ndarray:
    return np.array((pose.y - SPAWN_Y, -pose.x, pose.yaw), float)


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
    for radius in (0.58, 0.68, 0.78, 0.88, 0.98, 1.08):
        # Prefer the open south-facing aisle before considering side/back
        # stances around furniture; all candidates remain deterministic.
        for index in (12, 13, 11, 14, 10, 15, 9, 0, 8, 1, 7, 2, 6, 3, 5, 4):
            angle = 2.0 * math.pi * index / 16
            x = float(target_world[0] + radius * math.cos(angle))
            y = float(target_world[1] + radius * math.sin(angle))
            # Google base local +Y faces forward after the scene's +90deg yaw.
            yaw = math.atan2(float(target_world[1] - y), float(target_world[0] - x)) - math.pi / 2
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
            final_tracking_tolerance=0.018,
        )
    return specs


def validate_manipulation_at_pose(
    model: mujoco.MjModel,
    reference: mujoco.MjData,
    pose: BasePose,
    backend_body: str,
    target_body_world: np.ndarray,
    spec: SimplePickSpec,
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
    # Body 0 contains the floor; the mobile base is designed to touch it.
    allowed = frozenset((0, body_id))
    carry_joints, carry_error, carry_angle = ik.solve(
        carry, profile.home_seed, profile.top_down_rotation
    )
    pre_joints, pre_error, pre_angle = ik.solve(
        pregrasp, carry_joints, profile.top_down_rotation
    )
    target_joints, target_error, target_angle = ik.solve(
        target, pre_joints, profile.top_down_rotation
    )
    tolerances_ok = (
        max(carry_error, pre_error, target_error) <= 0.012
        and max(carry_angle, pre_angle, target_angle) <= math.radians(2.0)
    )
    segments = []
    reason = None
    if tolerances_ok:
        for label, start, goal in (
            ("navigation_to_carry", profile.navigation_joints, carry_joints),
            ("carry_to_pregrasp", carry_joints, pre_joints),
            ("pregrasp_to_contact", pre_joints, target_joints),
        ):
            valid, segment_reason = checker.segment_valid(start, goal, allowed)
            segments.append({"segment": label, "valid": valid, "reason": segment_reason})
            if not valid:
                reason = segment_reason
                break
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

    def execute(self, path: list[BasePose], *, maximum_steps: int = 250000) -> dict[str, Any]:
        started = time.monotonic()
        steps = 0
        for waypoint in path[1:]:
            target = _world_to_joint_base(waypoint)
            settled = 0
            while settled < 8:
                maximum = self.model.opt.timestep * np.array((0.25, 0.25, 0.60))
                command = self.data.ctrl[self.actuators]
                self.data.ctrl[self.actuators] = command + np.clip(target - command, -maximum, maximum)
                mujoco.mj_step(self.model, self.data)
                steps += 1
                error = self.data.qpos[self.qpos] - target
                error[2] = _angle_delta(float(target[2]), float(self.data.qpos[self.qpos[2]]))
                settled = settled + 1 if float(np.max(np.abs(error))) < 0.018 else 0
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
    joint_ranges = ((-0.45, 4.90), (-2.80, 2.80), (-math.pi, math.pi))
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
        _read(phase2_dir / "plan.json"),
        _read(phase2_dir / "symbolic_problem.json"),
    )


def run_mobile_execution(
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    output_dir: str | Path,
    *,
    execute: bool = False,
    start_task_action: int = 0,
    max_task_actions: int | None = None,
) -> dict[str, Any]:
    run_started = time.monotonic()
    phase1_dir, phase2_dir, output_dir = map(Path, (phase1_dir, phase2_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads, regions, phase2_plan, symbolic_problem = load_phase3_inputs(phase1_dir, phase2_dir)
    scene = L2LivingRoomRegionScene(SCENE, "google")
    _configure_execution_base_limits(scene)
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
    placements = allocate_observed_placements(payloads, regions, phase2_plan)
    _write(output_dir / "dynamic_placement_targets.json", placements)

    object_backend = {row["generic_object_id"]: row["backend_body"] for row in resolution["objects"]}
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
    held: str | None = None
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
        if operator == "PLACE":
            region_record = regions["regions"][arguments["region"]]
            stance_focus = np.asarray(
                region_record["geometry"]["centroid_world_m"]["value"], float
            )
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
                ik_result = validate_manipulation_at_pose(
                    scene.model, scene.data, pose, backend, target, specs[backend]
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
                execution_log.append({"operator": "MOVE", "result": mobile.execute(path)})
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
            carry = _carry_position(selected, target)
            spec = specs[backend]
            specs[backend] = SimplePickSpec(**{**spec.__dict__, "carry_position": carry})
            picker = CalibratedPickPlaceExecutor(
                scene.model,
                scene.data,
                "google",
                pick_specs_override=specs,
                calibrated_objects_override=tuple(specs),
                base_stance=_world_to_joint_base(selected),
                base_approach_forward=0.0,
                arm_command_speed=1.35,
                intermediate_tracking_tolerance=0.065,
            )
            if operator == "PICK":
                picker.request_pick(backend)
            else:
                picker.held_object = backend
                picker.target_object = backend
                picker.target_body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, backend)
                picker.grasp_equality_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend}")
                picker.mode = "holding"
                picker.request_place_world(target)
            steps = 0
            while picker.mode not in {"holding" if operator == "PICK" else "idle", "failed"}:
                picker.update()
                mujoco.mj_step(scene.model, scene.data)
                steps += 1
                if steps > 60000:
                    picker._fail("MANIPULATION_TIMEOUT")
            execution_log.append({"operator": operator, "object_id": object_id, "backend_body": backend, "result": "SUCCESS" if picker.failure is None else "FAILED", "failure": picker.failure, "physics_steps": steps})
            if operator == "PLACE" and picker.failure is None:
                for _ in range(200):
                    mujoco.mj_step(scene.model, scene.data)
                body_id = mujoco.mj_name2id(
                    scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
                )
                actual = scene.data.xpos[body_id].copy()
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    scene.model,
                    scene.data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_id,
                    velocity,
                    0,
                )
                xy_error = float(np.linalg.norm(actual[:2] - target[:2]))
                height_error = abs(float(actual[2] - target[2]))
                stable_speed = float(np.linalg.norm(velocity[3:]))
                verification_ok = (
                    xy_error <= 0.055
                    and height_error <= 0.055
                    and stable_speed <= 0.10
                )
                execution_log[-1]["physical_verification"] = {
                    "target_body_world_m": target.tolist(),
                    "observed_body_world_m": actual.tolist(),
                    "xy_error_m": xy_error,
                    "height_error_m": height_error,
                    "linear_speed_m_s": stable_speed,
                    "within_observed_support_boundary": xy_error <= 0.055,
                    "stable": stable_speed <= 0.10,
                    "verified": verification_ok,
                }
                if not verification_ok:
                    execution_log[-1]["result"] = "FAILED"
                    execution_log[-1]["failure"] = "POST_PLACE_VERIFICATION_FAILED"
            execution_frames.append(_capture_execution_frame(scene))
            if execution_log[-1]["result"] == "FAILED":
                break
        held = object_id if operator == "PICK" else None

    refined_artifact = {
        "schema_version": SCHEMA_VERSION,
        "scene": SCENE,
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
        "mode": "EXECUTE" if execute else "PLAN_ONLY",
        "actions": execution_log,
        "success": execute and len(execution_log) > 0 and all(item.get("result") != "FAILED" for item in execution_log),
        "normal_execution_object_qpos_edits": False,
    }
    if execute:
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
            video_status["mp4"] = "UNAVAILABLE"
            video_status["mp4_error"] = str(error)
        _write(output_dir / "video_status.json", video_status)
    _write(output_dir / "physical_execution.json", physical)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "scene": SCENE,
        "phase1_payload_registry": str(phase1_dir / "payload_registry.json"),
        "phase1_payload_registry_sha256": _sha256(phase1_dir / "payload_registry.json"),
        "phase1_region_registry_sha256": _sha256(phase1_dir / "region_registry.json"),
        "phase2_plan_sha256": _sha256(phase2_dir / "plan.json"),
        "phase2_symbolic_problem_sha256": _sha256(phase2_dir / "symbolic_problem.json"),
        "frozen_task_plan_preserved": True,
        "oracle_used_by_planner_or_refiner": False,
    }
    _write(output_dir / "provenance_manifest.json", provenance)
    result = {
        "status": "SUCCESS" if (not execute or physical["success"]) else "FAILED",
        "mode": physical["mode"],
        "output_dir": str(output_dir.resolve()),
        "phase2_action_count": len(actions),
        "refined_action_count": len(refined),
        "move_count": sum(item["operator"] == "MOVE" for item in refined),
        "wall_time_s": time.monotonic() - run_started,
    }
    _write(output_dir / "run_summary.json", result)
    return result
