"""Profile-driven vertical pick/place for calibrated kitchen robots.

Fetch keeps its object-specific manipulation controller in ``pick_motion``.
This module is deliberately smaller: it provides the reusable calibration
baseline used by Google Robot and future backends, beginning with regular
objects that admit a stable top-down pinch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

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
SELF_COLLISION_COMPARISON_EPSILON = 0.0002
ENVIRONMENT_COLLISION_TOLERANCE = 0.002
TRAVERSABLE_GROUND_GEOMS = frozenset(
    ("floor", "a2_floor", "a2_rug_surface", "a2_rug_border")
)
COLLISION_GUARD_INTERVAL = 5
SPOON_PIVOT_RELAXATION = 0.30
SPOON_PIVOT_DAMPING = 0.002
SPOON_PIVOT_MAX_TORQUE = 0.02
SPOON_VERTICAL_TOLERANCE = math.radians(3.0)
SPOON_SETTLED_ANGULAR_SPEED = 1.0
SPOON_SETTLE_TICKS = 50
SPOON_REGRASP_SQUEEZE = 0.015
SPOON_REGRASP_TIMEOUT_TICKS = 600
CALIBRATION_ATTEMPT_TIMEOUT_TICKS = 20000
SELF_COLLISION_MOUNT_ALLOWANCES = {
    # The shoulder rotates inside the base's outer housing by design.  The
    # upstream visual meshes overlap slightly at this mechanical interface;
    # deeper overlap is still rejected, as are all non-mounting link pairs.
    frozenset(("google:base_link", "google:link_shoulder")): -0.050,
}
GOOGLE_SPOON_TOP_DOWN_ROTATION = np.diag((1.0, -1.0, -1.0))


@dataclass(frozen=True)
class SimplePickSpec:
    label: str
    grasp_site: str
    support_height: float
    # The Menagerie site's origin is below/above the most useful pad band after
    # rotation.  This is the per-object value refined during visual calibration.
    grasp_z_offset: float = 0.011
    required_contact_geoms: tuple[str, ...] = ()
    place_supported: bool = True
    top_down_rotation: np.ndarray | None = None
    home_seed: np.ndarray | None = None
    carry_position: np.ndarray | None = None
    final_tracking_tolerance: float = JOINT_WAYPOINT_TOLERANCE
    carry_grip_relaxation: float = 0.0
    ik_position_tolerance: float | None = None
    ik_angle_tolerance_rad: float | None = None
    approach_clearance_m: float = APPROACH_CLEARANCE
    ik_orientation_weight: float = 0.30
    grasp_candidates: tuple["GraspPoseCandidate", ...] = ()
    ik_restart_offsets: tuple[tuple[float, ...], ...] = ()
    intermediate_ik_position_tolerance: float | None = None
    intermediate_ik_angle_tolerance_rad: float | None = None
    approach_offset_world_m: tuple[float, float, float] | None = None
    approach_route_offsets_world_m: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True)
class GraspPoseCandidate:
    """A bounded execution candidate expressed in the target-body frame."""

    candidate_id: str
    grasp_site_local_position_m: tuple[float, float, float]
    target_rotation_world: np.ndarray
    approach_clearance_m: float
    carry_rotation_world: np.ndarray | None = None
    approach_offset_world_m: tuple[float, float, float] | None = None
    approach_route_offsets_world_m: tuple[tuple[float, float, float], ...] = ()


GOOGLE_PICK_SPECS = {
    "sugar_jar": SimplePickSpec(
        "Sugar jar (vertical)", "sugar_jar_grasp", 0.06724
    ),
    "spoon": SimplePickSpec(
        "Spoon (far handle tip)",
        "spoon_grasp",
        0.01045,
        grasp_z_offset=0.020,
        required_contact_geoms=("spoon_handle_collision",),
        place_supported=False,
        # A 180-degree wrist roll preserves the jaw axis across the handle
        # while avoiding the joint-limit branch used for the centre jar.
        top_down_rotation=GOOGLE_SPOON_TOP_DOWN_ROTATION,
        home_seed=np.array((0.230, -0.195, 0.663, 1.732, 0.120, 1.564, -0.687)),
        # Four centimetres closer to the base keeps the hanging bowl clear of
        # B1 throughout the right-side yaw sweep.
        carry_position=np.array((0.0, -0.82, 0.94)),
    ),
}

# Candidate definitions are deliberately excluded from normal execution.  They
# provide a measurable starting point in ``--calibration-mode``: IK, collision,
# and bilateral-contact failures remain hard failures and must be tuned before
# moving an object into ``GOOGLE_PICK_SPECS``/the supported-object profile.
GOOGLE_CALIBRATION_PICK_SPECS = {
    "coffee_jar": SimplePickSpec(
        "Coffee jar (candidate upper-body pinch)",
        "coffee_jar_grasp",
        0.09287,
        grasp_z_offset=0.011,
        place_supported=False,
    ),
    "kettle": SimplePickSpec(
        "Kettle (candidate handle pinch)",
        "kettle_grasp",
        0.06817,
        grasp_z_offset=0.008,
        required_contact_geoms=("kettle_handle_collision",),
        place_supported=False,
    ),
}

CALIBRATED_SCENE_OBJECTS = {
    "S1_coffee_missing_mug": ("sugar_jar", "spoon"),
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
        orientation_weight: float = 0.30,
    ):
        self.model = model
        self.profile = profile
        self.orientation_weight = float(orientation_weight)
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
            error = np.concatenate(
                (position_error, self.orientation_weight * rotation_error)
            )
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
                    self.orientation_weight * jac_rot[:, self.dof_addresses],
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
        *,
        mounting_allowances: dict[frozenset[str], float] | None = None,
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
        self.mounting_allowances = (
            SELF_COLLISION_MOUNT_ALLOWANCES
            if mounting_allowances is None else mounting_allowances
        )
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
                mounting_allowance = self.mounting_allowances.get(
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
            if distance < minimum_distance - SELF_COLLISION_COMPARISON_EPSILON:
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
                environment_geom_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, environment_geom
                ) or ""
                if environment_geom_name in TRAVERSABLE_GROUND_GEOMS:
                    continue
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
    passive_pivot: bool = False


class CalibratedPickPlaceExecutor:
    """Execute calibrated vertical Google Robot pick and place actions."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_name: str,
        scene_name: str | None = None,
        calibration_mode: bool = False,
        pick_specs_override: dict[str, SimplePickSpec] | None = None,
        calibrated_objects_override: tuple[str, ...] | None = None,
        base_stance: np.ndarray | None = None,
        base_approach_forward: float = MANIPULATION_BASE_FORWARD,
        base_approach_delta: np.ndarray | None = None,
        arm_command_speed: float = ARM_COMMAND_SPEED,
        intermediate_tracking_tolerance: float = INTERMEDIATE_TRACKING_TOLERANCE,
        mounting_allowances: dict[frozenset[str], float] | None = None,
        ik_position_tolerance: float = 0.012,
        ik_angle_tolerance: float = math.radians(2.0),
    ):
        self.model = model
        self.data = data
        self.robot_name = robot_name
        self.profile = manipulation_profile(robot_name)
        self.mobile_profile = mobile_profile(robot_name)
        self.calibration_mode = calibration_mode
        if arm_command_speed <= 0.0:
            raise ValueError("arm_command_speed must be positive")
        self.arm_command_speed = float(arm_command_speed)
        if intermediate_tracking_tolerance <= 0.0:
            raise ValueError("intermediate_tracking_tolerance must be positive")
        self.intermediate_tracking_tolerance = float(
            intermediate_tracking_tolerance
        )
        self.mounting_allowances = mounting_allowances
        self.ik_position_tolerance = float(ik_position_tolerance)
        self.ik_angle_tolerance = float(ik_angle_tolerance)
        if pick_specs_override is None:
            scene_objects = CALIBRATED_SCENE_OBJECTS.get(scene_name, ())
            supported_objects = (
                self.profile.supported_objects
                if scene_name is None
                else scene_objects
            )
            calibrated_specs = {
                name: GOOGLE_PICK_SPECS[name]
                for name in supported_objects
                if name in self.profile.supported_objects
                if name in GOOGLE_PICK_SPECS
            }
            all_pick_specs = {
                **GOOGLE_CALIBRATION_PICK_SPECS,
                **calibrated_specs,
            }
        else:
            all_pick_specs = dict(pick_specs_override)
            calibrated_names = (
                tuple(all_pick_specs)
                if calibrated_objects_override is None
                else calibrated_objects_override
            )
            calibrated_specs = {
                name: all_pick_specs[name]
                for name in calibrated_names
                if name in all_pick_specs
            }
        self.calibrated_objects = frozenset(calibrated_specs)
        self.all_pick_specs = all_pick_specs
        self.pick_specs = (
            self.all_pick_specs if calibration_mode else calibrated_specs
        )
        self.calibration_attempt_ticks = 0
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
        self.base_stance = (
            np.zeros(3)
            if base_stance is None
            else np.asarray(base_stance, dtype=float).copy()
        )
        if self.base_stance.shape != (3,):
            raise ValueError("base_stance must contain forward, lateral, and yaw")
        approach_delta = (
            np.array((base_approach_forward, 0.0, 0.0))
            if base_approach_delta is None
            else np.asarray(base_approach_delta, dtype=float).copy()
        )
        if approach_delta.shape != (3,):
            raise ValueError(
                "base_approach_delta must contain forward, lateral, and yaw"
            )
        self.base_manipulation_target = self.base_stance + approach_delta
        self.base_forward_qpos = int(self.base_qpos[0])
        self.mode = "idle"
        self.status = "Manipulation idle: gripper empty"
        self.failure: str | None = None
        self.held_object: str | None = None
        self.target_object: str | None = None
        self.target_body_id = -1
        self.target_free_dof = -1
        self.grasp_equality_id = -1
        self.spoon_pivot_equality_id = -1
        self.spoon_settle_ticks = 0
        self.spoon_regrasp_command = self.profile.open_command
        self.spoon_regrasp_target = self.profile.open_command
        self.spoon_regrasp_ticks = 0
        self.spoon_regrasp_elapsed_ticks = 0
        self.close_target = self.profile.open_command
        self.contact_ticks = 0
        self.release_ticks = 0
        self.waypoints: list[JointWaypoint] = []
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.retreat_waypoints: list[JointWaypoint] = []
        self.pending_place_site = "serving_spot"
        self.pending_place_world: np.ndarray | None = None
        self.pending_place_rotation: np.ndarray | None = None
        self.configuration_checker: RobotConfigurationCollisionChecker | None = None
        self.collision_guard_tick = 0
        self.attachment_translation_snap_m: float | None = None
        self.attachment_angle_snap_rad: float | None = None
        # Persist the contact evidence that authorized weld activation.  Live
        # contacts usually disappear during lift, so post-pick validation
        # must not infer bilateral contact merely from an active weld.
        self.confirmed_contact_sides: tuple[int, ...] = ()
        self.confirmed_contact_geoms: tuple[str, ...] = ()
        self.confirmed_target_contact_geoms: tuple[str, ...] = ()
        self.selected_grasp_candidate_id: str | None = None

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
        return (
            self.mode == "holding"
            and self.held_object is not None
            and self.pick_specs[self.held_object].place_supported
        )

    @property
    def navigation_safe(self) -> bool:
        # This reports the arm/carry state, not the symbolic base location.  A
        # compact robot at a cupboard must still be allowed to Move home.
        return self.mode in {"idle", "holding"}

    def _command_base(self, target: np.ndarray) -> None:
        # The short table approach benefits from stronger damping than a long
        # navigation route. Restore the composed-model values as soon as the
        # approach or retreat has settled.
        self.model.dof_damping[self.base_dofs[:2]] = (
            MANIPULATION_BASE_LINEAR_DAMPING
        )
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

    def _base_at_target(self, target: np.ndarray) -> bool:
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
        error = self.data.qpos[self.base_qpos] - self.base_stance
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
        target_rotation: np.ndarray | None = None,
    ) -> tuple[list[JointWaypoint], np.ndarray]:
        rotation = (
            self.profile.top_down_rotation
            if target_rotation is None
            else target_rotation
        )
        result = []
        current = seed
        for point in points:
            previous = current
            current, position_error, angle_error = ik.solve(
                point, previous, rotation
            )
            active_spec = (
                self.pick_specs.get(self.target_object)
                if self.target_object is not None else None
            )
            position_tolerance = (
                active_spec.ik_position_tolerance
                if active_spec and active_spec.ik_position_tolerance is not None
                else self.ik_position_tolerance
            )
            angle_tolerance = (
                active_spec.ik_angle_tolerance_rad
                if active_spec and active_spec.ik_angle_tolerance_rad is not None
                else self.ik_angle_tolerance
            )
            if label == "Approaching above object" and active_spec:
                if active_spec.intermediate_ik_position_tolerance is not None:
                    position_tolerance = active_spec.intermediate_ik_position_tolerance
                if active_spec.intermediate_ik_angle_tolerance_rad is not None:
                    angle_tolerance = active_spec.intermediate_ik_angle_tolerance_rad
            if active_spec and active_spec.ik_restart_offsets and (
                position_error > position_tolerance
                or angle_error > angle_tolerance
            ):
                alternatives = [(current, position_error, angle_error)]
                for offset in active_spec.ik_restart_offsets:
                    restart_seed = previous + np.asarray(offset, float)
                    alternatives.append(
                        ik.solve(point, restart_seed, rotation)
                    )
                current, position_error, angle_error = min(
                    alternatives,
                    key=lambda item: (
                        item[1] / position_tolerance
                        + item[2] / angle_tolerance
                    ),
                )
            if (
                position_error > position_tolerance
                or angle_error > angle_tolerance
            ):
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
        target_rotation: np.ndarray,
        home_seed: np.ndarray,
        carry_position: np.ndarray,
        carry_rotation: np.ndarray | None = None,
    ) -> tuple[list[JointWaypoint], list[JointWaypoint]]:
        active_spec = self.pick_specs.get(self.target_object)
        ik = ProfiledIK(
            self.model,
            self.data,
            self.profile,
            orientation_weight=(
                active_spec.ik_orientation_weight
                if active_spec is not None else 0.30
            ),
        )
        current = self._current_arm()
        carry_rotation = target_rotation if carry_rotation is None else carry_rotation
        carry, position_error, angle_error = ik.solve(
            carry_position,
            home_seed,
            carry_rotation,
        )
        active_spec = self.pick_specs.get(self.target_object)
        position_tolerance = (
            active_spec.ik_position_tolerance
            if active_spec and active_spec.ik_position_tolerance is not None
            else self.ik_position_tolerance
        )
        angle_tolerance = (
            active_spec.ik_angle_tolerance_rad
            if active_spec and active_spec.ik_angle_tolerance_rad is not None
            else self.ik_angle_tolerance
        )
        if position_error > position_tolerance or angle_error > angle_tolerance:
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
        active_spec = self.pick_specs.get(self.target_object)
        approach_clearance = (
            active_spec.approach_clearance_m
            if active_spec is not None else APPROACH_CLEARANCE
        )
        pregrasp = target + np.asarray(
            (
                active_spec.approach_offset_world_m
                if active_spec is not None
                and active_spec.approach_offset_world_m is not None
                else (0.0, 0.0, approach_clearance)
            ),
            float,
        )
        approach = []
        pregrasp_joints = carry
        route_start = carry_position
        route_goals = [
            target + np.asarray(offset, float)
            for offset in (
                active_spec.approach_route_offsets_world_m
                if active_spec is not None else ()
            )
        ] or [pregrasp]
        if not np.allclose(route_goals[-1], pregrasp):
            route_goals.append(pregrasp)
        for route_goal in route_goals:
            segment, pregrasp_joints = self._solve_points(
                ik,
                self._cartesian_points(route_start, route_goal),
                pregrasp_joints,
                "Approaching above object",
                collision_checker,
                allowed_environment_bodies,
                target_rotation,
            )
            approach.extend(segment)
            route_start = route_goal
        descent, _ = self._solve_points(
            ik,
            self._cartesian_points(pregrasp, target, 0.012),
            pregrasp_joints,
            "Descending to grasp",
            collision_checker,
            allowed_environment_bodies,
            target_rotation,
        )
        waypoints.extend(approach)
        waypoints.extend(descent)
        passive_spoon = self.target_object == "spoon"
        return waypoints, [
            JointWaypoint(
                item.joints.copy(),
                (
                    "Returning while spoon hangs from handle"
                    if passive_spoon
                    else "Lifting and returning to carry"
                ),
                passive_pivot=passive_spoon,
            )
            for item in reversed([*approach, *descent[:-1]])
        ] + [
            JointWaypoint(
                carry.copy(),
                (
                    "Spoon hanging vertically in carry pose"
                    if passive_spoon
                    else "Holding at carry pose"
                ),
                passive_pivot=passive_spoon,
            )
        ]

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
        free_joint_id = int(self.model.body_jntadr[body_id])
        if (
            free_joint_id < 0
            or self.model.jnt_type[free_joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError(f"{object_name} does not have a free joint")
        self.target_free_dof = int(self.model.jnt_dofadr[free_joint_id])
        self.close_target = self.profile.open_command
        self.contact_ticks = 0
        self.spoon_pivot_equality_id = -1
        self.spoon_settle_ticks = 0
        self.spoon_regrasp_command = self.profile.open_command
        self.spoon_regrasp_target = self.profile.open_command
        self.spoon_regrasp_ticks = 0
        self.spoon_regrasp_elapsed_ticks = 0
        self.failure = None
        self.attachment_translation_snap_m = None
        self.attachment_angle_snap_rad = None
        self.confirmed_contact_sides = ()
        self.confirmed_contact_geoms = ()
        self.confirmed_target_contact_geoms = ()
        self.selected_grasp_candidate_id = None
        self.calibration_attempt_ticks = 0
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
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile,
            mounting_allowances=self.mounting_allowances,
        )
        allowed_bodies = frozenset((self.target_body_id,))
        candidates = spec.grasp_candidates or (
            GraspPoseCandidate(
                candidate_id="configured_grasp_site",
                grasp_site_local_position_m=tuple(self.model.site_pos[site_id]),
                target_rotation_world=(
                    self.profile.top_down_rotation
                    if spec.top_down_rotation is None else spec.top_down_rotation
                ),
                approach_clearance_m=spec.approach_clearance_m,
                approach_offset_world_m=spec.approach_offset_world_m,
                approach_route_offsets_world_m=spec.approach_route_offsets_world_m,
            ),
        )
        failures = []
        selected = None
        for candidate in candidates:
            self.model.site_pos[site_id] = candidate.grasp_site_local_position_m
            mujoco.mj_forward(self.model, self.data)
            target = self.data.site_xpos[site_id].copy()
            target[2] += spec.grasp_z_offset
            candidate_spec = replace(
                spec,
                top_down_rotation=candidate.target_rotation_world,
                approach_clearance_m=candidate.approach_clearance_m,
                approach_offset_world_m=candidate.approach_offset_world_m,
                approach_route_offsets_world_m=candidate.approach_route_offsets_world_m,
            )
            self.pick_specs[self.target_object] = candidate_spec
            try:
                self.waypoints, self.retreat_waypoints = self._plan_to_grip(
                    target,
                    self.configuration_checker,
                    allowed_bodies,
                    candidate.target_rotation_world,
                    self.profile.home_seed if spec.home_seed is None else spec.home_seed,
                    self.profile.carry_position if spec.carry_position is None else spec.carry_position,
                    candidate.carry_rotation_world,
                )
            except RuntimeError as error:
                failures.append(f"{candidate.candidate_id}: {error}")
                continue
            selected = candidate
            break
        if selected is None:
            self.pick_specs[self.target_object] = spec
            raise RuntimeError("No collision-free grasp candidate; " + "; ".join(failures))
        self.selected_grasp_candidate_id = selected.candidate_id
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.mode = "pick_approach"
        self.status = f"Pick {self.target_object}: plan ready, opening gripper"

    def request_place(self, site_name: str = "serving_spot") -> None:
        if not self.can_place:
            if self.held_object is not None:
                raise RuntimeError(
                    f"Place is not calibrated for {self.held_object}; "
                    "carry it or reset the scene"
                )
            raise RuntimeError("Pick an object before requesting place")
        if not self._near_navigation_home():
            raise RuntimeError("Place requires Move (home) first")
        self.pending_place_site = site_name
        self.pending_place_world = None
        self.pending_place_rotation = None
        self.failure = None
        self.calibration_attempt_ticks = 0
        self.mode = "place_base_approach"
        self.status = f"Place {self.held_object}: approaching manipulation stance"

    def request_place_world(
        self,
        desired_body_world: np.ndarray,
        target_rotation: np.ndarray | None = None,
    ) -> None:
        """Place the held body at a measured world-frame support position.

        This is the dynamic counterpart of :meth:`request_place`.  The input
        is a body-centre target derived by an upstream observed-evidence
        placement allocator; it is never inferred from simulator object
        metadata by this executor.
        """
        if not self.can_place:
            raise RuntimeError("Pick a place-supported object before placement")
        if not self._near_navigation_home():
            raise RuntimeError("Place requires Move (home) first")
        target = np.asarray(desired_body_world, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("desired_body_world must be a finite xyz vector")
        self.pending_place_world = target.copy()
        if target_rotation is None:
            self.pending_place_rotation = None
        else:
            rotation = np.asarray(target_rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("target_rotation must be a finite 3x3 matrix")
            self.pending_place_rotation = rotation.copy()
        self.pending_place_site = "<dynamic_world_target>"
        self.failure = None
        self.calibration_attempt_ticks = 0
        self.mode = "place_base_approach"
        self.status = f"Place {self.held_object}: approaching dynamic target"

    def _begin_place_plan(self) -> None:
        assert self.held_object is not None
        spec = self.pick_specs[self.held_object]
        mujoco.mj_forward(self.model, self.data)
        if self.pending_place_world is None:
            site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, self.pending_place_site
            )
            if site_id < 0:
                raise RuntimeError(f"Unknown placement site: {self.pending_place_site}")
            desired_body = self.data.site_xpos[site_id].copy()
            desired_body[2] += spec.support_height
        else:
            desired_body = self.pending_place_world.copy()
        grip_to_body = (
            self.data.xpos[self.target_body_id] - self.data.site_xpos[self.grip_site_id]
        )
        target_grip = desired_body - grip_to_body
        ik = ProfiledIK(self.model, self.data, self.profile)
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile,
            mounting_allowances=self.mounting_allowances,
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
            self.pending_place_rotation,
        )
        descent, _ = self._solve_points(
            ik,
            self._cartesian_points(preplace, target_grip, 0.012),
            preplace_joints,
            "Descending to placement surface",
            self.configuration_checker,
            allowed_bodies,
            self.pending_place_rotation,
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
        max_command_step = self.arm_command_speed * self.model.opt.timestep
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
            (
                self.pick_specs[self.target_object].final_tracking_tolerance
                if self.target_object in self.pick_specs
                else JOINT_WAYPOINT_TOLERANCE
            )
            if is_final
            else self.intermediate_tracking_tolerance
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
        assert self.target_object is not None
        required = self.pick_specs[self.target_object].required_contact_geoms
        sides: set[int] = set()
        for contact in self.data.contact:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self.target_body_id not in {body1, body2}:
                continue
            target_geom = (
                contact.geom1 if body1 == self.target_body_id else contact.geom2
            )
            target_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, target_geom
            ) or ""
            if required and target_name not in required:
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
        self.grasp_equality_id = equality_id
        sides = self._finger_contact_sides()
        if sides != {0, 1}:
            raise RuntimeError("grasp weld requires confirmed bilateral contact")
        contact_geoms: set[str] = set()
        target_contact_geoms: set[str] = set()
        for contact in self.data.contact:
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if self.target_body_id not in {body1, body2}:
                continue
            target_geom = contact.geom1 if body1 == self.target_body_id else contact.geom2
            target_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, target_geom
            ) or f"geom_{target_geom}"
            other = contact.geom2 if body1 == self.target_body_id else contact.geom1
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other
            ) or f"geom_{other}"
            if any(name in family for family in self.profile.finger_contact_geoms):
                contact_geoms.add(name)
                target_contact_geoms.add(target_name)
        self.confirmed_contact_sides = tuple(sorted(sides))
        self.confirmed_contact_geoms = tuple(sorted(contact_geoms))
        self.confirmed_target_contact_geoms = tuple(sorted(target_contact_geoms))
        before_pos = self.data.xpos[self.target_body_id].copy()
        before_quat = self.data.xquat[self.target_body_id].copy()
        self._set_grasp_weld_world_pose(before_pos, before_quat)
        self.data.eq_active[equality_id] = 1
        mujoco.mj_forward(self.model, self.data)
        self.attachment_translation_snap_m = float(
            np.linalg.norm(self.data.xpos[self.target_body_id] - before_pos)
        )
        dot = abs(float(np.dot(self.data.xquat[self.target_body_id], before_quat)))
        self.attachment_angle_snap_rad = 2.0 * math.acos(
            float(np.clip(dot, -1.0, 1.0))
        )

    def _set_grasp_weld_world_pose(
        self, object_pos: np.ndarray, object_quat: np.ndarray
    ) -> None:
        if self.grasp_equality_id < 0:
            raise RuntimeError("Cannot configure an inactive grasp weld")
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
            object_pos,
            object_quat,
        )
        self.model.eq_data[self.grasp_equality_id, 3:6] = relative_pos
        self.model.eq_data[self.grasp_equality_id, 6:10] = relative_quat

    def _activate_spoon_pivot(self) -> None:
        """Replace the transport weld with a free-rotation handle pivot."""
        if self.target_object != "spoon" or self.grasp_equality_id < 0:
            raise RuntimeError("Spoon pivot requested without a spoon grasp")
        equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"{self.robot_name}:pick_pivot_spoon",
        )
        grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "spoon_grasp"
        )
        if equality_id < 0 or grasp_site_id < 0:
            raise RuntimeError("Missing passive spoon pivot constraint")

        world_anchor = self.data.site_xpos[grasp_site_id].copy()
        gripper_rotation = self.data.xmat[self.gripper_body_id].reshape(3, 3)
        spoon_rotation = self.data.xmat[self.target_body_id].reshape(3, 3)
        gripper_anchor = gripper_rotation.T @ (
            world_anchor - self.data.xpos[self.gripper_body_id]
        )
        spoon_anchor = spoon_rotation.T @ (
            world_anchor - self.data.xpos[self.target_body_id]
        )
        self.model.eq_data[equality_id, :3] = gripper_anchor
        self.model.eq_data[equality_id, 3:6] = spoon_anchor
        self.data.eq_active[self.grasp_equality_id] = 0
        self.data.eq_active[equality_id] = 1
        self.spoon_pivot_equality_id = equality_id
        self.data.ctrl[self.finger_actuators] = max(
            self.profile.open_command,
            self.close_target - SPOON_PIVOT_RELAXATION,
        )

    @staticmethod
    def _limited(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= maximum or norm < 1e-12:
            return vector
        return vector * (maximum / norm)

    def _damp_spoon_pivot(self) -> tuple[float, float]:
        """Damp the freely hanging spoon without prescribing its angle."""
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.target_body_id,
            velocity,
            0,
        )
        angular_velocity = velocity[:3]
        self.data.xfrc_applied[self.target_body_id, 3:] = self._limited(
            -SPOON_PIVOT_DAMPING * angular_velocity,
            SPOON_PIVOT_MAX_TORQUE,
        )
        spoon_axis = -self.data.xmat[self.target_body_id].reshape(3, 3)[:, 0]
        bowl_down_angle = math.acos(
            float(np.clip(spoon_axis @ np.array((0.0, 0.0, -1.0)), -1.0, 1.0))
        )
        swing_velocity = angular_velocity - spoon_axis * float(
            angular_velocity @ spoon_axis
        )
        return bowl_down_angle, float(np.linalg.norm(swing_velocity))

    def _finish_spoon_pivot(self) -> None:
        """Capture the settled pose and prepare a physical handle re-grasp."""
        if self.spoon_pivot_equality_id < 0 or self.grasp_equality_id < 0:
            raise RuntimeError("Cannot finish an inactive spoon pivot")
        self.data.qvel[self.target_free_dof : self.target_free_dof + 6] = 0.0
        self._set_grasp_weld_world_pose(
            self.data.xpos[self.target_body_id],
            self.data.xquat[self.target_body_id],
        )
        self.model.eq_solref[self.grasp_equality_id] = (0.003, 1.0)
        self.data.eq_active[self.spoon_pivot_equality_id] = 0
        self.data.eq_active[self.grasp_equality_id] = 1
        self.data.xfrc_applied[self.target_body_id] = 0.0
        self.spoon_regrasp_command = max(
            self.profile.open_command,
            self.close_target - SPOON_PIVOT_RELAXATION,
        )
        self.spoon_regrasp_target = min(
            self.profile.closed_command,
            self.close_target + SPOON_REGRASP_SQUEEZE,
        )
        self.spoon_regrasp_ticks = 0
        self.spoon_regrasp_elapsed_ticks = 0

    def _fail(self, message: str) -> None:
        self._restore_navigation_base_damping()
        if self.spoon_pivot_equality_id >= 0:
            self.data.eq_active[self.spoon_pivot_equality_id] = 0
        if self.target_body_id >= 0:
            self.data.xfrc_applied[self.target_body_id] = 0.0
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
            "spoon_regrasp",
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
        if (
            self.calibration_mode
            and self.target_object not in self.calibrated_objects
        ):
            self.calibration_attempt_ticks += 1
            if self.calibration_attempt_ticks >= CALIBRATION_ATTEMPT_TIMEOUT_TICKS:
                active_status = self.status
                self._fail(f"candidate attempt timed out while: {active_status}")
                return
        guarded_modes = {
            "pick_approach",
            "closing",
            "pick_retreat",
            "spoon_regrasp",
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
            self._command_base(self.base_manipulation_target)
            if self._base_at_target(self.base_manipulation_target):
                self._restore_navigation_base_damping()
                try:
                    self._begin_pick_plan()
                except RuntimeError as error:
                    self._fail(str(error))
            return
        if self.mode == "place_base_approach":
            self._command_base(self.base_manipulation_target)
            if self._base_at_target(self.base_manipulation_target):
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
                    relaxation = self.pick_specs[
                        self.target_object
                    ].carry_grip_relaxation
                    if relaxation > 0.0:
                        self.close_target = max(
                            self.profile.open_command,
                            self.close_target - relaxation,
                        )
                        self.data.ctrl[
                            self.finger_actuators
                        ] = self.close_target
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
            waypoint = (
                self.waypoints[self.waypoint_index]
                if self.waypoint_index < len(self.waypoints)
                else None
            )
            if (
                waypoint is not None
                and waypoint.passive_pivot
                and self.spoon_pivot_equality_id < 0
            ):
                try:
                    self._activate_spoon_pivot()
                except RuntimeError as error:
                    self._fail(str(error))
                    return
            spoon_angle = spoon_speed = None
            if self.spoon_pivot_equality_id >= 0:
                self.data.ctrl[self.finger_actuators] = max(
                    self.profile.open_command,
                    self.close_target - SPOON_PIVOT_RELAXATION,
                )
                spoon_angle, spoon_speed = self._damp_spoon_pivot()
            if not self._advance_waypoints():
                return
            if self.spoon_pivot_equality_id >= 0:
                assert spoon_angle is not None and spoon_speed is not None
                if (
                    spoon_angle <= SPOON_VERTICAL_TOLERANCE
                    and spoon_speed <= SPOON_SETTLED_ANGULAR_SPEED
                ):
                    self.spoon_settle_ticks += 1
                else:
                    self.spoon_settle_ticks = 0
                if self.spoon_settle_ticks < SPOON_SETTLE_TICKS:
                    self.status = (
                        "Pick spoon: settling naturally into vertical hang "
                        f"({math.degrees(spoon_angle):.1f} deg)"
                    )
                    return
                self._finish_spoon_pivot()
                self.mode = "spoon_regrasp"
                self.status = "Pick spoon: closing fingers around the settled handle"
                return
            self.held_object = self.target_object
            self.mode = "pick_base_retreat"
            self.status = (
                f"Pick {self.held_object}: retreating to navigation home"
            )
            return
        if self.mode == "spoon_regrasp":
            self.spoon_regrasp_elapsed_ticks += 1
            self.spoon_regrasp_command = min(
                self.spoon_regrasp_target,
                self.spoon_regrasp_command + self.profile.close_step,
            )
            self.data.ctrl[self.finger_actuators] = self.spoon_regrasp_command
            if self._finger_contact_sides() == {0, 1}:
                self.spoon_regrasp_ticks += 1
            else:
                self.spoon_regrasp_ticks = 0
            if self.spoon_regrasp_ticks >= CONTACT_CONFIRM_TICKS:
                self.held_object = self.target_object
                self.mode = "pick_base_retreat"
                self.status = "Pick spoon: bilateral handle re-grasp confirmed"
                return
            if self.spoon_regrasp_elapsed_ticks >= SPOON_REGRASP_TIMEOUT_TICKS:
                self._fail("spoon handle re-grasp did not recover bilateral contact")
            return
        if self.mode == "pick_base_retreat":
            self._command_base(self.base_stance)
            if self._base_at_target(self.base_stance):
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
            self._command_base(self.base_stance)
            if not self._base_at_target(self.base_stance):
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
        if self.mode in {"closing", "spoon_regrasp", "releasing"}:
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
