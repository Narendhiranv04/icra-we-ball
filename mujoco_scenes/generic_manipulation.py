"""Profile-driven vertical pick/place for calibrated kitchen robots.

Fetch keeps its object-specific manipulation controller in ``pick_motion``.
This module is deliberately smaller: it provides the reusable calibration
baseline used by Google Robot and future backends, beginning with regular
objects that admit a stable top-down pinch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_scenes.robot_profiles import (
    ManipulationProfile,
    manipulation_profile,
    mobile_profile,
)


APPROACH_CLEARANCE = 0.10
PATH_RESOLUTION = 0.035
# Intermediate waypoints are time-held, but the contact pose must settle
# tightly before the fingers close or Cartesian bias can push the object away.
JOINT_WAYPOINT_TOLERANCE = 0.010
INTERMEDIATE_TRACKING_TOLERANCE = 0.050
ARM_COMMAND_SPEED = 1.20
ARM_COMMAND_TOLERANCE = 0.001
WAYPOINT_HOLD_TICKS = 4
CONTACT_CONFIRM_TICKS = 12
RELEASE_SETTLE_TICKS = 100
MANIPULATION_BASE_FORWARD = 0.15
BASE_TARGET_TOLERANCE = 0.002
BASE_HOME_REQUEST_TOLERANCE = 0.03
BASE_LINEAR_COMMAND_SPEED = 0.25
BASE_YAW_COMMAND_SPEED = 0.60
BASE_COMMAND_TOLERANCE = 0.002
BASE_SETTLE_SPEED = 0.01
MANIPULATION_BASE_LINEAR_DAMPING = 2000.0
SELF_COLLISION_TOLERANCE = 0.003
ENVIRONMENT_COLLISION_TOLERANCE = 0.002
COLLISION_GUARD_INTERVAL = 5
SELF_COLLISION_MOUNT_ALLOWANCES = {
    # The shoulder rotates inside the base's outer housing by design.  The
    # upstream visual meshes overlap slightly at this mechanical interface;
    # deeper overlap is still rejected, as are all non-mounting link pairs.
    frozenset(("google:base_link", "google:link_shoulder")): -0.050,
}


@dataclass(frozen=True)
class SimplePickSpec:
    label: str
    grasp_site: str
    support_height: float
    # The Menagerie site's origin is below/above the most useful pad band after
    # rotation.  This is the per-object value refined during visual calibration.
    grasp_z_offset: float = 0.011


GOOGLE_PICK_SPECS = {
    "sugar_jar": SimplePickSpec(
        "Sugar jar (vertical)", "sugar_jar_grasp", 0.06724
    ),
}

CALIBRATED_SCENE_OBJECTS = {
    "S1_coffee_missing_mug": ("sugar_jar",),
}


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.ravel())
    if quat[0] < 0:
        quat = -quat
    norm = float(np.linalg.norm(quat[1:]))
    if norm < 1e-10:
        return np.zeros(3)
    return quat[1:] / norm * (2.0 * math.atan2(norm, float(quat[0])))


class ProfiledIK:
    """Damped least-squares pose IK over a profile's declared arm joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
    ):
        self.model = model
        self.profile = profile
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = reference.qpos
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, profile.grip_site
        )
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in profile.arm_joints
            ]
        )
        if self.site_id < 0 or np.any(self.joint_ids < 0):
            raise RuntimeError("Manipulation profile does not match the composed model")
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        limits = model.jnt_range[self.joint_ids]
        limited = model.jnt_limited[self.joint_ids].astype(bool)
        self.lower = np.where(limited, limits[:, 0] + 0.015, -math.pi)
        self.upper = np.where(limited, limits[:, 1] - 0.015, math.pi)

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        self.data.qpos[self.qpos_addresses] = np.clip(seed, self.lower, self.upper)
        self.data.qvel[:] = 0
        for _ in range(1200):
            mujoco.mj_forward(self.model, self.data)
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target - self.data.site_xpos[self.site_id]
            rotation_error = _rotation_vector(target_rotation @ rotation.T)
            error = np.concatenate((position_error, 0.30 * rotation_error))
            if (
                np.linalg.norm(position_error) < 0.0008
                and np.linalg.norm(rotation_error) < math.radians(0.7)
            ):
                break

            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, jac_pos, jac_rot, self.site_id
            )
            jacobian = np.vstack(
                (
                    jac_pos[:, self.dof_addresses],
                    0.30 * jac_rot[:, self.dof_addresses],
                )
            )
            damping = 0.0025
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            current = self.data.qpos[self.qpos_addresses]
            self.data.qpos[self.qpos_addresses] = np.clip(
                current + np.clip(delta, -0.055, 0.055),
                self.lower,
                self.upper,
            )

        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        return (
            self.data.qpos[self.qpos_addresses].copy(),
            float(np.linalg.norm(target - self.data.site_xpos[self.site_id])),
            float(np.linalg.norm(_rotation_vector(target_rotation @ rotation.T))),
        )


class RobotConfigurationCollisionChecker:
    """Check visual self-clipping and robot/environment penetration.

    MuJoCo normally filters direct parent/child contacts and Menagerie's base
    collision meshes are disabled in the holonomic kitchen adaptation.  Exact
    geometry-distance queries over visual meshes are therefore required to
    reject IK poses that look clipped even when the contact solver is silent.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
    ):
        self.model = model
        self.profile = profile
        self.data = mujoco.MjData(model)
        self.reference_qpos = reference.qpos.copy()
        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in profile.arm_joints
            ]
        )
        self.arm_qpos = model.jnt_qposadr[self.arm_joint_ids]
        self.body_prefix = profile.gripper_body.split(":", 1)[0] + ":"
        self.robot_visual_geoms = []
        self.environment_geoms = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or ""
            if body_name.startswith(self.body_prefix):
                if model.geom_group[geom_id] == 2:
                    self.robot_visual_geoms.append(geom_id)
            elif (
                geom_name != "floor"
                and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
            ):
                self.environment_geoms.append(geom_id)

        self.self_pairs: list[tuple[int, int, float]] = []
        baseline = mujoco.MjData(model)
        baseline.qpos[:] = self.reference_qpos
        baseline.qpos[self.arm_qpos] = profile.navigation_joints
        mujoco.mj_forward(model, baseline)
        for index, first_geom in enumerate(self.robot_visual_geoms):
            first_body = int(model.geom_bodyid[first_geom])
            first_parent = int(model.body_parentid[first_body])
            for second_geom in self.robot_visual_geoms[index + 1 :]:
                second_body = int(model.geom_bodyid[second_geom])
                if (
                    first_body == second_body
                    or first_parent == second_body
                    or int(model.body_parentid[second_body]) == first_body
                ):
                    continue
                baseline_distance = mujoco.mj_geomDistance(
                    model,
                    baseline,
                    first_geom,
                    second_geom,
                    SELF_COLLISION_TOLERANCE,
                    None,
                )
                minimum_distance = min(
                    -SELF_COLLISION_TOLERANCE,
                    baseline_distance - SELF_COLLISION_TOLERANCE,
                )
                first_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, first_body
                ) or ""
                second_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, second_body
                ) or ""
                mounting_allowance = SELF_COLLISION_MOUNT_ALLOWANCES.get(
                    frozenset((first_name, second_name))
                )
                if mounting_allowance is not None:
                    minimum_distance = min(minimum_distance, mounting_allowance)
                self.self_pairs.append(
                    (first_geom, second_geom, minimum_distance)
                )

    def evaluate(
        self,
        arm_joints: np.ndarray,
        allowed_environment_bodies: frozenset[int] = frozenset(),
    ) -> tuple[bool, str | None]:
        self.data.qpos[:] = self.reference_qpos
        self.data.qpos[self.arm_qpos] = arm_joints
        mujoco.mj_forward(self.model, self.data)
        return self._evaluate_current(allowed_environment_bodies)

    def evaluate_live(
        self,
        live_data: mujoco.MjData,
        allowed_environment_bodies: frozenset[int] = frozenset(),
    ) -> tuple[bool, str | None]:
        self.data.qpos[:] = live_data.qpos
        mujoco.mj_forward(self.model, self.data)
        return self._evaluate_current(allowed_environment_bodies)

    def _evaluate_current(
        self, allowed_environment_bodies: frozenset[int]
    ) -> tuple[bool, str | None]:
        for first_geom, second_geom, minimum_distance in self.self_pairs:
            distance = mujoco.mj_geomDistance(
                self.model,
                self.data,
                first_geom,
                second_geom,
                SELF_COLLISION_TOLERANCE,
                None,
            )
            if distance < minimum_distance:
                first_body = int(self.model.geom_bodyid[first_geom])
                second_body = int(self.model.geom_bodyid[second_geom])
                first_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, first_body
                ) or "unnamed robot body"
                second_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, second_body
                ) or "unnamed robot body"
                return False, (
                    f"self-collision {first_name} / {second_name} "
                    f"({distance * 100:.1f} cm signed distance)"
                )

        for robot_geom in self.robot_visual_geoms:
            for environment_geom in self.environment_geoms:
                environment_body = int(self.model.geom_bodyid[environment_geom])
                if environment_body in allowed_environment_bodies:
                    continue
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    robot_geom,
                    environment_geom,
                    ENVIRONMENT_COLLISION_TOLERANCE,
                    None,
                )
                if distance < -ENVIRONMENT_COLLISION_TOLERANCE:
                    robot_body = int(self.model.geom_bodyid[robot_geom])
                    robot_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body
                    ) or "unnamed robot body"
                    environment_name = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        environment_body,
                    ) or "unnamed environment body"
                    return False, (
                        f"environment collision {robot_name} / {environment_name} "
                        f"({distance * 100:.1f} cm signed distance)"
                    )
        return True, None

    def segment_valid(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        allowed_environment_bodies: frozenset[int] = frozenset(),
        resolution: float = 0.035,
    ) -> tuple[bool, str | None]:
        count = max(
            1,
            int(math.ceil(float(np.max(np.abs(goal - start))) / resolution)),
        )
        for fraction in np.linspace(0.0, 1.0, count + 1):
            joints = start + fraction * (goal - start)
            valid, reason = self.evaluate(joints, allowed_environment_bodies)
            if not valid:
                return False, reason
        return True, None


@dataclass
class JointWaypoint:
    joints: np.ndarray
    label: str


class CalibratedPickPlaceExecutor:
    """Execute calibrated vertical Google Robot pick and place actions."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_name: str,
        scene_name: str | None = None,
    ):
        self.model = model
        self.data = data
        self.robot_name = robot_name
        self.profile = manipulation_profile(robot_name)
        self.mobile_profile = mobile_profile(robot_name)
        scene_objects = CALIBRATED_SCENE_OBJECTS.get(scene_name, ())
        supported_objects = (
            self.profile.supported_objects if scene_name is None else scene_objects
        )
        self.pick_specs = {
            name: GOOGLE_PICK_SPECS[name]
            for name in supported_objects
            if name in self.profile.supported_objects
            if name in GOOGLE_PICK_SPECS
        }
        self.arm_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.profile.arm_joints
        )
        self.arm_qpos = model.jnt_qposadr[self.arm_joint_ids]
        self.arm_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.arm_actuators
        )
        self.finger_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.profile.finger_joints
        )
        self.finger_qpos = model.jnt_qposadr[self.finger_joint_ids]
        self.finger_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.finger_actuators
        )
        self.gripper_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.profile.gripper_body
        )
        self.grip_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, self.profile.grip_site
        )
        self.base_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.mobile_profile.base_joints
        )
        self.base_qpos = model.jnt_qposadr[self.base_joint_ids]
        self.base_dofs = model.jnt_dofadr[self.base_joint_ids]
        self.navigation_base_damping = model.dof_damping[self.base_dofs].copy()
        self.base_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.mobile_profile.base_actuators
        )
        self.base_forward_qpos = int(self.base_qpos[0])
        self.mode = "idle"
        self.status = "Manipulation idle: gripper empty"
        self.failure: str | None = None
        self.held_object: str | None = None
        self.target_object: str | None = None
        self.target_body_id = -1
        self.grasp_equality_id = -1
        self.close_target = self.profile.open_command
        self.contact_ticks = 0
        self.release_ticks = 0
        self.waypoints: list[JointWaypoint] = []
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.retreat_waypoints: list[JointWaypoint] = []
        self.pending_place_site = "serving_spot"
        self.configuration_checker: RobotConfigurationCollisionChecker | None = None
        self.collision_guard_tick = 0

    def _ids(self, object_type, names: tuple[str, ...]) -> np.ndarray:
        ids = np.array(
            [mujoco.mj_name2id(self.model, object_type, name) for name in names]
        )
        if np.any(ids < 0):
            missing = [name for name, item_id in zip(names, ids) if item_id < 0]
            raise RuntimeError(f"Missing calibrated model names: {', '.join(missing)}")
        return ids

    @property
    def busy(self) -> bool:
        return self.mode not in {"idle", "holding", "failed"}

    @property
    def can_place(self) -> bool:
        return self.mode == "holding" and self.held_object is not None

    @property
    def navigation_safe(self) -> bool:
        # This reports the arm/carry state, not the symbolic base location.  A
        # compact robot at a cupboard must still be allowed to Move home.
        return self.mode in {"idle", "holding"}

    def _command_base(self, forward: float) -> None:
        # The short table approach benefits from stronger damping than a long
        # navigation route. Restore the composed-model values as soon as the
        # approach or retreat has settled.
        self.model.dof_damping[self.base_dofs[:2]] = (
            MANIPULATION_BASE_LINEAR_DAMPING
        )
        target = np.array((forward, 0.0, 0.0))
        current = self.data.ctrl[self.base_actuators]
        max_step = self.model.opt.timestep * np.array(
            (
                BASE_LINEAR_COMMAND_SPEED,
                BASE_LINEAR_COMMAND_SPEED,
                BASE_YAW_COMMAND_SPEED,
            )
        )
        self.data.ctrl[self.base_actuators] = current + np.clip(
            target - current, -max_step, max_step
        )

    def _restore_navigation_base_damping(self) -> None:
        self.model.dof_damping[self.base_dofs] = self.navigation_base_damping

    def _base_at_target(self, forward: float) -> bool:
        target = np.array((forward, 0.0, 0.0))
        error = self.data.qpos[self.base_qpos] - target
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
        command_error = self.data.ctrl[self.base_actuators] - target
        command_error[2] = math.atan2(
            math.sin(command_error[2]), math.cos(command_error[2])
        )
        return (
            float(np.max(np.abs(error))) < BASE_TARGET_TOLERANCE
            and float(np.max(np.abs(command_error))) < BASE_COMMAND_TOLERANCE
            and float(np.max(np.abs(self.data.qvel[self.base_dofs])))
            < BASE_SETTLE_SPEED
        )

    def _near_navigation_home(self) -> bool:
        error = self.data.qpos[self.base_qpos].copy()
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
        return float(np.max(np.abs(error))) < BASE_HOME_REQUEST_TOLERANCE

    def _current_arm(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos].copy()

    @staticmethod
    def _cartesian_points(
        start: np.ndarray, goal: np.ndarray, resolution: float = PATH_RESOLUTION
    ) -> list[np.ndarray]:
        count = max(1, int(math.ceil(float(np.linalg.norm(goal - start)) / resolution)))
        return [start + fraction * (goal - start) for fraction in np.linspace(0, 1, count + 1)[1:]]

    @staticmethod
    def _joint_points(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        count = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / 0.06)))
        return [start + fraction * (goal - start) for fraction in np.linspace(0, 1, count + 1)[1:]]

    def _solve_points(
        self,
        ik: ProfiledIK,
        points: list[np.ndarray],
        seed: np.ndarray,
        label: str,
        collision_checker: RobotConfigurationCollisionChecker,
        allowed_environment_bodies: frozenset[int],
    ) -> tuple[list[JointWaypoint], np.ndarray]:
        result = []
        current = seed
        for point in points:
            previous = current
            current, position_error, angle_error = ik.solve(
                point, previous, self.profile.top_down_rotation
            )
            if position_error > 0.012 or angle_error > math.radians(2.0):
                raise RuntimeError(
                    f"IK misses {label} by {position_error * 100:.1f} cm "
                    f"with {math.degrees(angle_error):.1f} deg tilt"
                )
            collision_free, reason = collision_checker.segment_valid(
                previous,
                current,
                allowed_environment_bodies,
            )
            if not collision_free:
                raise RuntimeError(f"Unsafe IK segment during {label}: {reason}")
            result.append(JointWaypoint(current.copy(), label))
        return result, current

    def _plan_to_grip(
        self,
        target: np.ndarray,
        collision_checker: RobotConfigurationCollisionChecker,
        allowed_environment_bodies: frozenset[int],
    ) -> tuple[list[JointWaypoint], list[JointWaypoint]]:
        ik = ProfiledIK(self.model, self.data, self.profile)
        current = self._current_arm()
        carry, position_error, angle_error = ik.solve(
            self.profile.carry_position,
            self.profile.home_seed,
            self.profile.top_down_rotation,
        )
        if position_error > 0.012 or angle_error > math.radians(2.0):
            raise RuntimeError(
                f"Could not calibrate carry pose: {position_error * 100:.1f} cm, "
                f"{math.degrees(angle_error):.1f} deg"
            )
        collision_free, reason = collision_checker.segment_valid(
            current,
            carry,
            allowed_environment_bodies,
        )
        if not collision_free:
            raise RuntimeError(
                f"Unsafe path from navigation pose to carry: {reason}"
            )
        waypoints = [
            JointWaypoint(point, "Moving to carry clearance")
            for point in self._joint_points(current, carry)
        ]
        pregrasp = target + np.array((0.0, 0.0, APPROACH_CLEARANCE))
        approach, pregrasp_joints = self._solve_points(
            ik,
            self._cartesian_points(self.profile.carry_position, pregrasp),
            carry,
            "Approaching above object",
            collision_checker,
            allowed_environment_bodies,
        )
        descent, _ = self._solve_points(
            ik,
            self._cartesian_points(pregrasp, target, 0.012),
            pregrasp_joints,
            "Descending to grasp",
            collision_checker,
            allowed_environment_bodies,
        )
        waypoints.extend(approach)
        waypoints.extend(descent)
        return waypoints, [
            JointWaypoint(item.joints.copy(), "Lifting and returning to carry")
            for item in reversed([*approach, *descent[:-1]])
        ] + [JointWaypoint(carry.copy(), "Holding at carry pose")]

    def request_pick(self, object_name: str) -> None:
        if self.busy or self.held_object is not None:
            raise RuntimeError("The gripper is not available for another pick")
        if object_name not in self.pick_specs:
            choices = ", ".join(self.pick_specs)
            raise ValueError(f"Uncalibrated object '{object_name}'. Choose: {choices}")
        if not self._near_navigation_home():
            raise RuntimeError("Pick requires Move (home) first")
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if body_id < 0:
            raise RuntimeError(f"{object_name} is not present in this scene")
        self.target_object = object_name
        self.target_body_id = body_id
        self.close_target = self.profile.open_command
        self.contact_ticks = 0
        self.failure = None
        self.mode = "pick_base_approach"
        self.status = f"Pick {object_name}: approaching the manipulation stance"

    def _begin_pick_plan(self) -> None:
        assert self.target_object is not None
        spec = self.pick_specs[self.target_object]
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, spec.grasp_site
        )
        if site_id < 0:
            raise RuntimeError(f"Missing grasp site: {spec.grasp_site}")
        mujoco.mj_forward(self.model, self.data)
        target = self.data.site_xpos[site_id].copy()
        target[2] += spec.grasp_z_offset
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile
        )
        allowed_bodies = frozenset((self.target_body_id,))
        self.waypoints, self.retreat_waypoints = self._plan_to_grip(
            target,
            self.configuration_checker,
            allowed_bodies,
        )
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.mode = "pick_approach"
        self.status = f"Pick {self.target_object}: plan ready, opening gripper"

    def request_place(self, site_name: str = "serving_spot") -> None:
        if not self.can_place:
            raise RuntimeError("Pick an object before requesting place")
        if not self._near_navigation_home():
            raise RuntimeError("Place requires Move (home) first")
        self.pending_place_site = site_name
        self.failure = None
        self.mode = "place_base_approach"
        self.status = f"Place {self.held_object}: approaching manipulation stance"

    def _begin_place_plan(self) -> None:
        assert self.held_object is not None
        spec = self.pick_specs[self.held_object]
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.pending_place_site
        )
        if site_id < 0:
            raise RuntimeError(f"Unknown placement site: {self.pending_place_site}")
        mujoco.mj_forward(self.model, self.data)
        desired_body = self.data.site_xpos[site_id].copy()
        desired_body[2] += spec.support_height
        grip_to_body = (
            self.data.xpos[self.target_body_id] - self.data.site_xpos[self.grip_site_id]
        )
        target_grip = desired_body - grip_to_body
        ik = ProfiledIK(self.model, self.data, self.profile)
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile
        )
        allowed_bodies = frozenset((self.target_body_id,))
        current = self._current_arm()
        preplace = target_grip + np.array((0.0, 0.0, APPROACH_CLEARANCE))
        approach, preplace_joints = self._solve_points(
            ik,
            self._cartesian_points(self.data.site_xpos[self.grip_site_id], preplace),
            current,
            "Moving above placement site",
            self.configuration_checker,
            allowed_bodies,
        )
        descent, _ = self._solve_points(
            ik,
            self._cartesian_points(preplace, target_grip, 0.012),
            preplace_joints,
            "Descending to placement surface",
            self.configuration_checker,
            allowed_bodies,
        )
        self.waypoints = [*approach, *descent]
        retreat_to_carry = [
            JointWaypoint(item.joints.copy(), "Retreating after release")
            for item in reversed([*approach, *descent[:-1]])
        ]
        retreat_to_navigation = [
            JointWaypoint(point, "Folding arm into compact navigation pose")
            for point in self._joint_points(
                current, self.profile.navigation_joints
            )
        ]
        collision_free, reason = self.configuration_checker.segment_valid(
            current,
            self.profile.navigation_joints,
            allowed_bodies,
        )
        if not collision_free:
            raise RuntimeError(
                f"Unsafe post-place fold to navigation pose: {reason}"
            )
        self.retreat_waypoints = [*retreat_to_carry, *retreat_to_navigation]
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.release_ticks = 0
        self.mode = "place_approach"
        self.status = (
            f"Place {self.held_object}: moving to {self.pending_place_site}"
        )

    def _advance_waypoints(self) -> bool:
        if self.waypoint_index >= len(self.waypoints):
            return True
        waypoint = self.waypoints[self.waypoint_index]
        current_command = self.data.ctrl[self.arm_actuators]
        max_command_step = ARM_COMMAND_SPEED * self.model.opt.timestep
        next_command = current_command + np.clip(
            waypoint.joints - current_command,
            -max_command_step,
            max_command_step,
        )
        self.data.ctrl[self.arm_actuators] = next_command
        command_error = float(np.max(np.abs(waypoint.joints - next_command)))
        tracking_error = float(
            np.max(np.abs(self.data.qpos[self.arm_qpos] - waypoint.joints))
        )
        self.status = waypoint.label
        is_final = self.waypoint_index == len(self.waypoints) - 1
        tracking_tolerance = (
            JOINT_WAYPOINT_TOLERANCE
            if is_final
            else INTERMEDIATE_TRACKING_TOLERANCE
        )
        if (
            command_error < ARM_COMMAND_TOLERANCE
            and tracking_error < tracking_tolerance
        ):
            self.waypoint_ticks += 1
        else:
            self.waypoint_ticks = 0
        if self.waypoint_ticks >= WAYPOINT_HOLD_TICKS:
            self.waypoint_index += 1
            self.waypoint_ticks = 0
        return self.waypoint_index >= len(self.waypoints)

    def _finger_contact_sides(self) -> set[int]:
        sides: set[int] = set()
        for contact in self.data.contact:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self.target_body_id not in {body1, body2}:
                continue
            other = contact.geom2 if body1 == self.target_body_id else contact.geom1
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other
            ) or ""
            for side, names in enumerate(self.profile.finger_contact_geoms):
                if name in names:
                    sides.add(side)
        return sides

    def _activate_weld(self) -> None:
        assert self.target_object is not None
        equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"{self.robot_name}:pick_weld_{self.target_object}",
        )
        if equality_id < 0:
            raise RuntimeError("The composed scene has no calibrated grasp weld")
        inverse_pos = np.empty(3)
        inverse_quat = np.empty(4)
        relative_pos = np.empty(3)
        relative_quat = np.empty(4)
        mujoco.mju_negPose(
            inverse_pos,
            inverse_quat,
            self.data.xpos[self.gripper_body_id],
            self.data.xquat[self.gripper_body_id],
        )
        mujoco.mju_mulPose(
            relative_pos,
            relative_quat,
            inverse_pos,
            inverse_quat,
            self.data.xpos[self.target_body_id],
            self.data.xquat[self.target_body_id],
        )
        self.model.eq_data[equality_id, 3:6] = relative_pos
        self.model.eq_data[equality_id, 6:10] = relative_quat
        self.data.eq_active[equality_id] = 1
        self.grasp_equality_id = equality_id

    def _fail(self, message: str) -> None:
        self._restore_navigation_base_damping()
        self.mode = "failed"
        self.failure = message
        self.status = f"Manipulation failed: {message}"

    def _guard_live_configuration(self) -> bool:
        if self.configuration_checker is None:
            return True
        self.collision_guard_tick += 1
        if self.collision_guard_tick % COLLISION_GUARD_INTERVAL:
            return True
        target_contact_modes = {
            "pick_approach",
            "closing",
            "pick_retreat",
            "pick_base_retreat",
            "place_base_approach",
            "place_approach",
            "releasing",
        }
        allowed_bodies = (
            frozenset((self.target_body_id,))
            if self.target_body_id >= 0 and self.mode in target_contact_modes
            else frozenset()
        )
        collision_free, reason = self.configuration_checker.evaluate_live(
            self.data, allowed_bodies
        )
        if collision_free:
            return True
        self._fail(f"live collision guard stopped motion: {reason}")
        return False

    def update(self) -> None:
        if self.mode in {"idle", "holding", "failed"}:
            return
        guarded_modes = {
            "pick_approach",
            "closing",
            "pick_retreat",
            "pick_base_retreat",
            "place_base_approach",
            "place_approach",
            "releasing",
            "place_retreat",
            "place_base_retreat",
        }
        if self.mode in guarded_modes and not self._guard_live_configuration():
            return
        if self.mode == "pick_base_approach":
            self.data.ctrl[self.arm_actuators] = self.profile.navigation_joints
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            self._command_base(MANIPULATION_BASE_FORWARD)
            if self._base_at_target(MANIPULATION_BASE_FORWARD):
                self._restore_navigation_base_damping()
                try:
                    self._begin_pick_plan()
                except RuntimeError as error:
                    self._fail(str(error))
            return
        if self.mode == "place_base_approach":
            self._command_base(MANIPULATION_BASE_FORWARD)
            if self._base_at_target(MANIPULATION_BASE_FORWARD):
                self._restore_navigation_base_damping()
                try:
                    self._begin_place_plan()
                except RuntimeError as error:
                    self._fail(str(error))
            return
        if self.mode == "pick_approach":
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            if self._advance_waypoints():
                self.mode = "closing"
                self.status = f"Pick {self.target_object}: closing until bilateral contact"
            return
        if self.mode == "closing":
            self.close_target = min(
                self.profile.closed_command,
                self.close_target + self.profile.close_step,
            )
            self.data.ctrl[self.finger_actuators] = self.close_target
            if self._finger_contact_sides() == {0, 1}:
                self.contact_ticks += 1
                if self.contact_ticks >= CONTACT_CONFIRM_TICKS:
                    try:
                        self._activate_weld()
                    except RuntimeError as error:
                        self._fail(str(error))
                        return
                    self.waypoints = self.retreat_waypoints
                    self.waypoint_index = 0
                    self.waypoint_ticks = 0
                    self.mode = "pick_retreat"
                return
            self.contact_ticks = 0
            if self.close_target >= self.profile.closed_command:
                self._fail("gripper closed without bilateral object contact")
            return
        if self.mode == "pick_retreat":
            if self._advance_waypoints():
                self.held_object = self.target_object
                self.mode = "pick_base_retreat"
                self.status = (
                    f"Pick {self.held_object}: retreating to navigation home"
                )
            return
        if self.mode == "pick_base_retreat":
            self._command_base(0.0)
            if self._base_at_target(0.0):
                self._restore_navigation_base_damping()
                self.mode = "holding"
                self.status = (
                    f"Pick complete: holding {self.held_object} in compact carry"
                )
            return
        if self.mode == "place_approach":
            if self._advance_waypoints():
                if self.grasp_equality_id >= 0:
                    self.data.eq_active[self.grasp_equality_id] = 0
                self.mode = "releasing"
                self.status = f"Place {self.held_object}: opening gripper"
            return
        if self.mode == "releasing":
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            self.release_ticks += 1
            if self.release_ticks >= RELEASE_SETTLE_TICKS:
                self.waypoints = self.retreat_waypoints
                self.waypoint_index = 0
                self.waypoint_ticks = 0
                self.mode = "place_retreat"
            return
        if self.mode == "place_retreat" and self._advance_waypoints():
            self.mode = "place_base_retreat"
            self.status = "Place complete: retreating to navigation home"
            return
        if self.mode == "place_base_retreat":
            self._command_base(0.0)
            if not self._base_at_target(0.0):
                return
            self._restore_navigation_base_damping()
            placed = self.held_object
            self.held_object = None
            self.target_object = None
            self.target_body_id = -1
            self.grasp_equality_id = -1
            self.mode = "idle"
            self.status = f"Place complete: released {placed}"

    def progress(self) -> float:
        if self.mode in {"closing", "releasing"}:
            return 0.55
        if self.mode in {
            "pick_retreat",
            "pick_base_retreat",
            "place_retreat",
            "place_base_retreat",
        }:
            return 0.75
        if self.mode == "holding":
            return 1.0
        if not self.waypoints:
            return 0.0
        return min(0.5, 0.5 * self.waypoint_index / len(self.waypoints))
