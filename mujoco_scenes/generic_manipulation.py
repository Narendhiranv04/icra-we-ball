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
    ("floor", "a2_floor", "a2_rug_surface", "a2_rug_border", "workshop_floor_geom")
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
# Heavy vessel shells can leave the position-controlled arm roughly 1--2 cm
# off the unloaded IK target while still aligned for a genuine bilateral
# grasp. Contact-side confirmation and weld validation remain mandatory.
PRECLOSE_POSITION_TOLERANCE_M = 0.020
PRECLOSE_ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
PRECLOSE_HOLD_TICKS = 5
PRECLOSE_TIMEOUT_TICKS = 250
SELF_COLLISION_MOUNT_ALLOWANCES = {
    # The shoulder rotates inside the base's outer housing by design.  The
    # upstream visual meshes overlap slightly at this mechanical interface;
    # deeper overlap is still rejected, as are all non-mounting link pairs.
    frozenset(("google:base_link", "google:link_shoulder")): -0.100,
    frozenset(("google:base_link", "google:link_bicep")): -0.050,
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
    # Position actuators can lag the final close command on thin payloads.
    # A bounded grace period lets the fingers reach the commanded stop before
    # declaring that contact was not achieved; zero preserves the legacy
    # immediate-failure behavior.
    close_grace_ticks: int = 0
    # Thin payloads can produce a short bilateral-contact window as the
    # fingers finish closing.  Keep the confirmation requirement configurable
    # per payload while retaining the conservative default for existing ones.
    contact_confirm_ticks: int = CONTACT_CONFIRM_TICKS
    # Hold the released payload at the support height for a bounded settling
    # interval before beginning the reverse retreat.
    place_release_settle_ticks: int = RELEASE_SETTLE_TICKS
    # Let the active weld settle the held payload at the final support pose
    # before opening.  Zero preserves the legacy immediate-release behavior.
    place_pre_release_settle_ticks: int = 0
    # Maximum live physics settling after the arm has retreated from release.
    # Existing payloads retain the original bounded window.
    post_release_settle_max_steps: int = 2000
    ik_position_tolerance: float | None = None
    ik_angle_tolerance_rad: float | None = None
    approach_clearance_m: float = APPROACH_CLEARANCE
    ik_orientation_weight: float = 0.30
    grasp_candidates: tuple["GraspPoseCandidate", ...] = ()
    ik_restart_offsets: tuple[tuple[float, ...], ...] = ()
    intermediate_ik_position_tolerance: float | None = None
    intermediate_ik_angle_tolerance_rad: float | None = None
    carry_ik_position_tolerance: float | None = None
    carry_ik_angle_tolerance_rad: float | None = None
    approach_offset_world_m: tuple[float, float, float] | None = None
    approach_route_offsets_world_m: tuple[tuple[float, float, float], ...] = ()
    retreat_route_offsets_world_m: tuple[tuple[float, float, float], ...] = ()
    # Some thin payloads are extracted through a constrained horizontal pinch.
    # When enabled, the configured retreat route ends with a safe carry-pose
    # transition after the route's explicit lift-off segment. The default keeps
    # the storage-object behavior, which intentionally stops at its extraction
    # hover before the base moves.
    retreat_to_carry_after_route: bool = False
    approach_rotation_world: np.ndarray | None = None
    position_first_approach: bool = False


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
    retreat_route_offsets_world_m: tuple[tuple[float, float, float], ...] = ()
    approach_rotation_world: np.ndarray | None = None
    position_first_approach: bool = False
    predicted_contact_geom_names: tuple[str, ...] = ()
    predicted_contact_points_world_m: tuple[tuple[float, float, float], ...] = ()


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


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return (
        np.eye(3)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


class ProfiledIK:
    """Damped least-squares pose IK over a profile's declared arm joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: mujoco.MjData,
        profile: ManipulationProfile,
        orientation_weight: float = 0.30,
        seed_continuity_weight: float = 0.0,
        maximum_seed_delta_rad: float | None = None,
        maximum_iterations: int = 1200,
    ):
        self.model = model
        self.profile = profile
        self.orientation_weight = float(orientation_weight)
        self.seed_continuity_weight = float(seed_continuity_weight)
        self.maximum_seed_delta_rad = maximum_seed_delta_rad
        self.maximum_iterations = int(maximum_iterations)
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
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        local_lower = self.lower
        local_upper = self.upper
        if self.maximum_seed_delta_rad is not None:
            local_lower = np.maximum(
                local_lower, seed - float(self.maximum_seed_delta_rad)
            )
            local_upper = np.minimum(
                local_upper, seed + float(self.maximum_seed_delta_rad)
            )
        self.data.qpos[self.qpos_addresses] = seed
        self.data.qvel[:] = 0
        for _ in range(self.maximum_iterations):
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
            damped_inverse = np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), np.eye(6)
            )
            pseudoinverse = jacobian.T @ damped_inverse
            delta = pseudoinverse @ error
            current = self.data.qpos[self.qpos_addresses]
            delta += self.seed_continuity_weight * (
                np.eye(len(current)) - pseudoinverse @ jacobian
            ) @ (seed - current)
            self.data.qpos[self.qpos_addresses] = np.clip(
                current + np.clip(delta, -0.055, 0.055),
                local_lower,
                local_upper,
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
        tolerance: float = 0.008,
    ) -> tuple[bool, str | None]:
        self.data.qpos[:] = live_data.qpos
        mujoco.mj_forward(self.model, self.data)
        return self._evaluate_current(allowed_environment_bodies, tolerance=tolerance)

    def _evaluate_current(
        self,
        allowed_environment_bodies: frozenset[int],
        tolerance: float = ENVIRONMENT_COLLISION_TOLERANCE,
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
                    tolerance,
                    None,
                )
                if distance < -tolerance:
                    robot_body = int(self.model.geom_bodyid[robot_geom])
                    robot_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body
                    ) or "unnamed robot body"
                    environment_name = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        environment_body,
                    ) or "unnamed environment body"
                    robot_geom_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom
                    ) or "unnamed robot geom"
                    return False, (
                        f"environment collision {robot_name} [{robot_geom_name}] / "
                        f"{environment_name} [{environment_geom_name}] "
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


def classify_held_payload_contact(
    held_body: int,
    other_body: int,
    allowed_task_contact_bodies: frozenset[int],
) -> str:
    """Classify only an explicitly named held-payload task contact as allowed."""
    if held_body == other_body:
        raise ValueError("Held payload contact requires two distinct bodies")
    return (
        "ALLOWED_TASK_CONTACT"
        if other_body in allowed_task_contact_bodies
        else "INVALID_COLLISION"
    )


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
        allowed_collision_bodies: tuple[str, ...] = (),
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
        self.base_approach_forward = float(base_approach_forward)
        self.base_manipulation_target = self.base_stance + approach_delta
        self.base_forward_qpos = int(self.base_qpos[0])
        self.allowed_collision_body_ids = frozenset(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in allowed_collision_bodies
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        )
        # Narrow execution override used only for an open-drawer utensil
        # descent. Planning/contact validation remain active; this suppresses
        # the live guard's conservative mesh stop while the fingers enter the
        # already-open drawer aperture.
        self.drawer_pick_collision_exemption = False
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
        self.close_elapsed_ticks = 0
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
        self.planned_grasp_position_world: np.ndarray | None = None
        self.planned_grasp_rotation_world: np.ndarray | None = None
        self.planned_contact_geom_names: tuple[str, ...] = ()
        self.planned_contact_points_world: tuple[tuple[float, float, float], ...] = ()
        self.preclose_settle_ticks = 0
        self.preclose_telemetry: dict[str, object] | None = None
        self.storage_fixture_release_telemetry: dict[str, object] | None = None

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

    def fold_held_payload_for_navigation(
        self,
        *,
        target_arm_joints: np.ndarray | None = None,
        tracking_tolerance_rad: float = 0.025,
        step_callback=None,
        maximum_steps_per_waypoint: int = 900,
        allowed_robot_contact_body_names: tuple[str, ...] = (),
    ) -> dict[str, object]:
        """Physically fold a held payload into the compact navigation pose.

        PICK intentionally ends in a family-specific Cartesian carry pose.
        Some bulky payloads cannot safely sweep through a subsequent base
        rotation from that pose.  This transition commands only robot joints,
        retains the live grasp weld, and checks both the arm path and live
        payload/environment contacts.  It never writes target-object qpos.
        """
        if self.mode != "holding" or self.held_object is None:
            raise RuntimeError("Held-payload fold requires holding mode")
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.held_object
        )
        current = self._current_arm()
        goal = (
            self.profile.navigation_joints.copy()
            if target_arm_joints is None
            else np.asarray(target_arm_joints, float).copy()
        )
        if goal.shape != current.shape:
            raise ValueError("Held-payload recovery target has wrong shape")
        checker = RobotConfigurationCollisionChecker(
            self.model,
            self.data,
            self.profile,
            mounting_allowances=self.mounting_allowances,
        )
        allowed_bodies = frozenset((
            body_id,
            *(
                allowed_body
                for name in allowed_robot_contact_body_names
                if (allowed_body := mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name
                )) >= 0
            ),
        ))
        valid, reason = checker.segment_valid(
            current, goal, allowed_bodies, resolution=0.025
        )
        if not valid:
            raise RuntimeError(f"Unsafe held-payload recovery: {reason}")

        waypoints = self._joint_points(current, goal)
        for waypoint_index, target in enumerate(waypoints):
            for _ in range(maximum_steps_per_waypoint):
                command = self.data.ctrl[self.arm_actuators]
                delta = np.clip(
                    target - command,
                    -self.arm_command_speed * self.model.opt.timestep,
                    self.arm_command_speed * self.model.opt.timestep,
                )
                self.data.ctrl[self.arm_actuators] = command + delta
                mujoco.mj_step(self.model, self.data)
                if step_callback:
                    step_callback()
                # The grasp weld may contact the robot by construction.  Only
                # payload contact with non-robot, non-floor environment is a
                # failure during this navigation preparation.
                for contact_index in range(self.data.ncon):
                    contact = self.data.contact[contact_index]
                    first_body = int(self.model.geom_bodyid[contact.geom1])
                    second_body = int(self.model.geom_bodyid[contact.geom2])
                    if body_id not in (first_body, second_body):
                        continue
                    other_geom = (
                        contact.geom2 if first_body == body_id else contact.geom1
                    )
                    other_body = int(self.model.geom_bodyid[other_geom])
                    other_body_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, other_body
                    ) or ""
                    other_geom_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                    ) or ""
                    if (
                        other_body_name.startswith(
                            self.profile.gripper_body.split(":", 1)[0] + ":"
                        )
                        or other_geom_name == "floor"
                    ):
                        continue
                    raise RuntimeError(
                        "Held payload contacted environment during recovery: "
                        f"{other_geom_name or other_body_name}"
                    )
                if float(np.max(np.abs(
                    self.data.qpos[self.arm_qpos] - target
                ))) < tracking_tolerance_rad:
                    break
            else:
                raise RuntimeError(
                    "Held-payload recovery tracking timeout at waypoint "
                    f"{waypoint_index}"
                )
        return {
            "performed": True,
            "recovery_policy": (
                "COMPACT_NAVIGATION_ARM"
                if target_arm_joints is None else "RECORDED_POST_PICK_CARRY_ARM"
            ),
            "joint_target": goal.tolist(),
            "direct_object_qpos_write": False,
            "grasp_weld_retained": bool(
                self.grasp_equality_id >= 0
                and self.data.eq_active[self.grasp_equality_id]
            ),
            "waypoint_count": len(waypoints),
        }

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
        pos_tol = (
            BASE_HOME_REQUEST_TOLERANCE
            if abs(self.base_approach_forward) < 1e-6
            else BASE_TARGET_TOLERANCE
        )
        cmd_tol = (
            BASE_HOME_REQUEST_TOLERANCE
            if abs(self.base_approach_forward) < 1e-6
            else BASE_COMMAND_TOLERANCE
        )
        return (
            float(np.max(np.abs(error))) < pos_tol
            and float(np.max(np.abs(command_error))) < cmd_tol
            and float(np.max(np.abs(self.data.qvel[self.base_dofs])))
            < BASE_SETTLE_SPEED
        )

    def _near_navigation_home(self) -> bool:
        error = self.data.qpos[self.base_qpos] - self.base_stance
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
        return float(np.max(np.abs(error))) < BASE_HOME_REQUEST_TOLERANCE

    def move_to_local_manipulation_base(
        self,
        *,
        step_callback=None,
        maximum_steps: int = 12000,
    ) -> int:
        """Synchronously reach the configured local base stance while empty."""
        if self.mode != "idle" or self.held_object is not None:
            raise RuntimeError("Local base positioning requires an idle empty gripper")
        for step in range(1, maximum_steps + 1):
            self.data.ctrl[self.arm_actuators] = self.profile.navigation_joints
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            self._command_base(self.base_manipulation_target)
            mujoco.mj_step(self.model, self.data)
            if step_callback:
                step_callback()
            if self._base_at_target(self.base_manipulation_target):
                self._restore_navigation_base_damping()
                return step
        self._restore_navigation_base_damping()
        raise RuntimeError("Local manipulation base positioning timed out")

    def execute_contact_presentation(
        self,
        object_name: str,
        cartesian_path_world: tuple[tuple[float, float, float], ...],
        target_rotation_world: np.ndarray,
        *,
        step_callback=None,
        waypoint_timeout_steps: int = 1200,
        finger_command: float | None = None,
        carry_rotation_world: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Move the closed gripper along a contact path without attachment.

        This is an execution-level non-prehensile primitive.  The target body
        is the only environment body admitted by the collision checker; no
        equality constraint is activated and no object state is written.
        """
        if self.mode != "idle" or self.held_object is not None:
            raise RuntimeError("Presentation requires an idle empty gripper")
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if body_id < 0:
            raise RuntimeError(f"Missing presentation target: {object_name}")
        if len(cartesian_path_world) < 2:
            raise ValueError("Presentation path requires at least two points")
        equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"{self.robot_name}:pick_weld_{object_name}",
        )
        if equality_id >= 0 and self.data.eq_active[equality_id]:
            raise RuntimeError("Presentation cannot begin with an active grasp weld")

        self.target_object = object_name
        self.target_body_id = body_id
        try:
            self.configuration_checker = RobotConfigurationCollisionChecker(
                self.model,
                self.data,
                self.profile,
                mounting_allowances=self.mounting_allowances,
            )
            checker = self.configuration_checker
            allowed = frozenset((body_id,))
            ik = ProfiledIK(
                self.model,
                self.data,
                self.profile,
                orientation_weight=0.30,
            )
            current = self._current_arm()
            planned: list[JointWaypoint] = []
            start_position = self.data.site_xpos[self.grip_site_id].copy()
            route_start = start_position
            goals = list(cartesian_path_world)
            active_spec = self.pick_specs[object_name]
            if goals:
                carry_goal = np.asarray(goals.pop(0), dtype=float)
                home_seed = (
                    active_spec.home_seed
                    if active_spec.home_seed is not None
                    else self.profile.navigation_joints
                )
                carry_joints, position_error, angle_error = ik.solve(
                    carry_goal,
                    np.asarray(home_seed, dtype=float),
                    (
                        target_rotation_world
                        if carry_rotation_world is None
                        else carry_rotation_world
                    ),
                )
                if (
                    position_error > self.ik_position_tolerance
                    or angle_error > self.ik_angle_tolerance
                ):
                    raise RuntimeError(
                        "Presentation carry IK misses by "
                        f"{position_error * 100:.1f} cm with "
                        f"{math.degrees(angle_error):.1f} deg tilt"
                    )
                collision_free, reason = checker.segment_valid(
                    current, carry_joints, allowed
                )
                if not collision_free:
                    raise RuntimeError(
                        f"Presentation carry path collision: {reason}"
                    )
                planned.extend(
                    JointWaypoint(point, "Presentation carry clearance")
                    for point in self._joint_points(current, carry_joints)
                )
                current = carry_joints
                route_start = carry_goal
            for goal_tuple in goals:
                goal = np.asarray(goal_tuple, dtype=float)
                segment, current = self._solve_points(
                    ik,
                    self._cartesian_points(route_start, goal, 0.020),
                    current,
                    "Presentation contact path",
                    checker,
                    allowed,
                    target_rotation_world,
                )
                planned.extend(segment)
                route_start = goal

            target_start_position = self.data.xpos[body_id].copy()
            target_start_quaternion = self.data.xquat[body_id].copy()
            robot_body_ids = {
                body_index
                for body_index in range(self.model.nbody)
                if (
                    mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_index
                    ) or ""
                ).startswith(f"{self.robot_name}:")
            }
            contact_target_geoms: set[str] = set()
            contact_robot_geoms: set[str] = set()
            contact_steps = 0
            bilateral_contact_steps = 0
            physics_steps = 0
            presentation_finger_command = (
                self.profile.closed_command
                if finger_command is None else float(finger_command)
            )
            self.data.ctrl[self.finger_actuators] = presentation_finger_command
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
                physics_steps += 1
                if step_callback:
                    step_callback()

            terminal_bilateral_contact_steps = 0
            presentation_grasp_ready = False
            for waypoint_index, waypoint in enumerate(planned):
                settled = 0
                waypoint_touched = False
                bilateral_streak = 0
                for _ in range(waypoint_timeout_steps):
                    self.data.ctrl[self.arm_actuators] = waypoint.joints
                    self.data.ctrl[self.finger_actuators] = presentation_finger_command
                    mujoco.mj_step(self.model, self.data)
                    physics_steps += 1
                    if step_callback:
                        step_callback()
                    touched = False
                    step_robot_geoms: set[str] = set()
                    for contact in self.data.contact:
                        body1 = int(self.model.geom_bodyid[contact.geom1])
                        body2 = int(self.model.geom_bodyid[contact.geom2])
                        if body_id not in {body1, body2}:
                            continue
                        other_body = body2 if body1 == body_id else body1
                        if other_body not in robot_body_ids:
                            continue
                        target_geom = (
                            contact.geom1 if body1 == body_id else contact.geom2
                        )
                        robot_geom = (
                            contact.geom2 if body1 == body_id else contact.geom1
                        )
                        contact_target_geoms.add(
                            mujoco.mj_id2name(
                                self.model, mujoco.mjtObj.mjOBJ_GEOM, target_geom
                            ) or f"geom_{target_geom}"
                        )
                        robot_geom_name = (
                            mujoco.mj_id2name(
                                self.model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom
                            ) or f"geom_{robot_geom}"
                        )
                        contact_robot_geoms.add(robot_geom_name)
                        step_robot_geoms.add(robot_geom_name)
                        touched = True
                    if touched:
                        contact_steps += 1
                        waypoint_touched = True
                        bilateral = all(
                            any(name in family for name in step_robot_geoms)
                            for family in self.profile.finger_contact_geoms
                        )
                        if bilateral:
                            bilateral_contact_steps += 1
                            bilateral_streak += 1
                        else:
                            bilateral_streak = 0
                    else:
                        bilateral_streak = 0
                    if (
                        bilateral_streak >= CONTACT_CONFIRM_TICKS
                        and float(
                            np.linalg.norm(
                                self.data.xpos[body_id] - target_start_position
                            )
                        ) >= 0.008
                    ):
                        # The physical presentation objective is a displaced
                        # utensil in a sustained two-sided pinch.  Preserve
                        # that state immediately; later Cartesian waypoints
                        # are irrelevant once this stronger postcondition is
                        # satisfied and can push the thin handle out again.
                        terminal_bilateral_contact_steps = bilateral_streak
                        presentation_grasp_ready = True
                        break
                    if waypoint_index == len(planned) - 1:
                        terminal_bilateral_contact_steps = max(
                            terminal_bilateral_contact_steps,
                            bilateral_streak,
                        )
                        if bilateral_streak >= CONTACT_CONFIRM_TICKS:
                            # Preserve the first physically confirmed terminal
                            # pinch for the caller.  Continuing to integrate a
                            # contact-constrained waypoint can push the thin
                            # handle back out of the jaws before adoption.
                            break
                    collision_free, reason = checker.evaluate_live(self.data, allowed)
                    if not collision_free:
                        raise RuntimeError(
                            f"Presentation live collision guard: {reason}"
                        )
                    tracking = float(
                        np.max(np.abs(self.data.qpos[self.arm_qpos] - waypoint.joints))
                    )
                    if tracking <= 0.025:
                        settled += 1
                        if settled >= 5 and not (
                            waypoint_index == len(planned) - 1
                            and waypoint_touched
                            and bilateral_streak < CONTACT_CONFIRM_TICKS
                        ):
                            break
                    elif touched and tracking <= 0.040:
                        # Contact motion is deliberately compliant: once the
                        # target resists the commanded Cartesian waypoint, do
                        # not integrate against it for the full timeout.
                        settled += 1
                        if settled >= 50:
                            break
                    else:
                        settled = 0
                else:
                    if not waypoint_touched:
                        raise RuntimeError(
                            "Presentation waypoint tracking timeout: "
                            f"{waypoint.label}"
                        )
                    # A contact-constrained push may intentionally remain off
                    # the free-space IK target.  The bounded timeout is then
                    # the force/dwell limit; continue to the retreat waypoint.
                if presentation_grasp_ready:
                    break

            target_end_position = self.data.xpos[body_id].copy()
            target_end_quaternion = self.data.xquat[body_id].copy()
            quaternion_dot = abs(
                float(np.dot(target_start_quaternion, target_end_quaternion))
            )
            return {
                "strategy": "CONTACT_DRIVEN_DRAWER_SLIDE",
                "success": bool(contact_steps > 0),
                "contact_steps": contact_steps,
                "bilateral_contact_steps": bilateral_contact_steps,
                "terminal_bilateral_contact_steps": (
                    terminal_bilateral_contact_steps
                ),
                "presentation_grasp_ready": presentation_grasp_ready,
                "contact_target_geoms": sorted(contact_target_geoms),
                "contact_robot_geoms": sorted(contact_robot_geoms),
                "target_translation_m": float(
                    np.linalg.norm(target_end_position - target_start_position)
                ),
                "target_rotation_rad": float(
                    2.0 * math.acos(np.clip(quaternion_dot, 0.0, 1.0))
                ),
                "target_start_position_world_m": target_start_position.tolist(),
                "target_end_position_world_m": target_end_position.tolist(),
                "physics_steps": physics_steps,
                "grasp_weld_active": bool(
                    equality_id >= 0 and self.data.eq_active[equality_id]
                ),
                "direct_object_qpos_write": False,
            }
        finally:
            self.target_object = None
            self.target_body_id = -1
            self.configuration_checker = None

    def execute_held_pose_trajectory(
        self,
        poses_world: tuple[tuple[np.ndarray, np.ndarray, str, int], ...],
        *,
        initial_arm_joints: np.ndarray | None = None,
        monitored_body_names: tuple[str, ...] = (),
        allowed_payload_contact_body_names: tuple[str, ...] = (),
        allowed_robot_contact_body_names: tuple[str, ...] = (),
        additional_mounting_allowances: dict[frozenset[str], float] | None = None,
        step_callback=None,
        maximum_steps_per_waypoint: int = 1800,
        command_speed_scale: float = 1.0,
    ) -> dict[str, object]:
        """Command a welded payload through strict Cartesian wrist poses.

        Each tuple is ``(position, rotation, label, dwell_steps)``.  The
        method moves robot controls only: it never writes a payload freejoint
        and never creates or changes a weld.  Contacts between the payload and
        any non-robot body are fail-closed unless the caller explicitly names
        an intended task-contact body.  This narrow payload/body allowance is
        separate from robot collision checking: gripper, arm, and base contact
        with that body remain invalid.
        """
        if self.mode != "holding" or self.held_object is None:
            raise RuntimeError("Held trajectory requires an existing grasp")
        if not poses_world:
            raise ValueError("Held trajectory requires at least one pose")
        if command_speed_scale <= 0.0:
            raise ValueError("command_speed_scale must be positive")
        held_name = self.held_object
        held_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, held_name
        )
        weld_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"{self.robot_name}:pick_weld_{held_name}",
        )
        if weld_id < 0 or not bool(self.data.eq_active[weld_id]):
            raise RuntimeError("Held trajectory requires an active grasp weld")

        checker = RobotConfigurationCollisionChecker(
            self.model,
            self.data,
            self.profile,
            mounting_allowances={
                **self.mounting_allowances,
                **(additional_mounting_allowances or {}),
            },
        )
        allowed_robot_contact_bodies = frozenset(
            body_id
            for name in allowed_robot_contact_body_names
            if (body_id := mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )) >= 0
        )
        initial_ik = ProfiledIK(
            self.model,
            self.data,
            self.profile,
            orientation_weight=0.45,
        )
        continuity_ik = ProfiledIK(
            self.model,
            self.data,
            self.profile,
            orientation_weight=0.45,
        )
        current = self._current_arm()
        planned: list[tuple[JointWaypoint, int]] = []
        maximum_position_error = 0.0
        maximum_angle_error = 0.0
        maximum_adjacent_joint_delta = 0.0
        maximum_local_adjacent_joint_delta = 0.0
        adjacent_joint_deltas: list[dict[str, object]] = []
        previous_position = self.data.site_xpos[self.grip_site_id].copy()
        previous_rotation = self.data.site_xmat[self.grip_site_id].reshape(3, 3).copy()
        interpolated_poses: list[tuple[np.ndarray, np.ndarray, str, int]] = []
        for requested_pose_index, (position, rotation, label, dwell_steps) in enumerate(poses_world):
            position = np.asarray(position, float)
            rotation = np.asarray(rotation, float)
            if requested_pose_index == 0:
                interpolated_poses.append((
                    position, rotation, label, int(dwell_steps)
                ))
                previous_position = position
                previous_rotation = rotation
                continue
            rotation_delta = _rotation_vector(rotation @ previous_rotation.T)
            rotation_angle = float(np.linalg.norm(rotation_delta))
            count = max(
                1,
                int(math.ceil(float(np.linalg.norm(position - previous_position)) / 0.015)),
                int(math.ceil(rotation_angle / math.radians(8.0))),
            )
            rotation_axis = (
                np.array((1.0, 0.0, 0.0))
                if rotation_angle < 1e-12
                else rotation_delta / rotation_angle
            )
            for interpolation_index, fraction in enumerate(
                np.linspace(0.0, 1.0, count + 1)[1:]
            ):
                interpolated_poses.append((
                    previous_position + float(fraction) * (position - previous_position),
                    _axis_angle_rotation(rotation_axis, float(fraction) * rotation_angle)
                    @ previous_rotation,
                    label,
                    int(dwell_steps) if interpolation_index == count - 1 else 0,
                ))
            previous_position = position
            previous_rotation = rotation

        for pose_index, (position, rotation, label, dwell_steps) in enumerate(interpolated_poses):
            active_ik = initial_ik if pose_index == 0 else continuity_ik
            solve_seed = (
                np.asarray(initial_arm_joints, float)
                if pose_index == 0 and initial_arm_joints is not None
                else current
            )
            goal, position_error, angle_error = active_ik.solve(
                np.asarray(position, float), solve_seed, np.asarray(rotation, float)
            )
            if pose_index == 0 and initial_arm_joints is None and (
                position_error > self.ik_position_tolerance
                or angle_error > self.ik_angle_tolerance
            ):
                # The transition from the live carry configuration to the
                # pre-manipulation pose is a global, collision-checked move.
                # Navigation joints are a legitimate alternate seed here,
                # but never after the local manipulation branch is fixed.
                restarted = active_ik.solve(
                    np.asarray(position, float),
                    self.profile.navigation_joints,
                    np.asarray(rotation, float),
                )
                if (
                    restarted[1] / self.ik_position_tolerance
                    + restarted[2] / self.ik_angle_tolerance
                    < position_error / self.ik_position_tolerance
                    + angle_error / self.ik_angle_tolerance
                ):
                    goal, position_error, angle_error = restarted
            maximum_position_error = max(maximum_position_error, float(position_error))
            maximum_angle_error = max(maximum_angle_error, float(angle_error))
            if position_error > self.ik_position_tolerance or angle_error > self.ik_angle_tolerance:
                raise RuntimeError(
                    f"Held trajectory IK misses {label} by {position_error * 100:.1f} cm "
                    f"with {math.degrees(angle_error):.1f} deg tilt; "
                    f"interpolated_pose_index={pose_index}; "
                    f"target_position_world_m={np.asarray(position, float).tolist()}"
                )
            maximum_joint_delta = float(np.max(np.abs(goal - current)))
            maximum_adjacent_joint_delta = max(
                maximum_adjacent_joint_delta, maximum_joint_delta
            )
            if pose_index > 0:
                maximum_local_adjacent_joint_delta = max(
                    maximum_local_adjacent_joint_delta, maximum_joint_delta
                )
            adjacent_joint_deltas.append({
                "interpolated_pose_index": pose_index,
                "label": label,
                "maximum_joint_delta_rad": maximum_joint_delta,
                "continuity_guard_applied": pose_index > 0,
            })
            if pose_index > 0 and maximum_joint_delta > 0.71:
                raise RuntimeError(
                    f"Held trajectory IK branch discontinuity during {label}: "
                    f"maximum_joint_delta_rad={maximum_joint_delta:.6f}"
                )
            valid, reason = checker.segment_valid(
                current,
                goal,
                frozenset((held_body,)) | allowed_robot_contact_bodies,
                resolution=0.025,
            )
            if not valid:
                raise RuntimeError(f"Held trajectory collision during {label}: {reason}")
            points = self._joint_points(current, goal)
            planned.extend(
                (JointWaypoint(point, label), int(dwell_steps) if index == len(points) - 1 else 0)
                for index, point in enumerate(points)
            )
            current = goal

        robot_bodies = {
            body_id
            for body_id in range(self.model.nbody)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith(
                f"{self.robot_name}:"
            )
        }
        monitored = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name): name
            for name in monitored_body_names
        }
        monitored = {body_id: name for body_id, name in monitored.items() if body_id >= 0}
        allowed_payload_contacts = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name): name
            for name in allowed_payload_contact_body_names
        }
        if any(body_id < 0 for body_id in allowed_payload_contacts):
            missing = sorted(
                name for body_id, name in allowed_payload_contacts.items()
                if body_id < 0
            )
            raise ValueError(f"Unknown allowed payload contact bodies: {missing}")
        minimum_monitored_clearance = math.inf
        invalid_pairs: set[tuple[str, str]] = set()
        allowed_task_pairs: set[tuple[str, str]] = set()
        physics_steps = 0
        dwell_pose_samples: list[dict[str, object]] = []
        active_label = "UNSET"
        active_waypoint_index = -1
        active_waypoint_target = current.copy()

        def validate_live() -> None:
            nonlocal minimum_monitored_clearance
            if not bool(self.data.eq_active[weld_id]) or self.held_object != held_name:
                raise RuntimeError("Held payload weld was lost during trajectory")
            valid, reason = checker.evaluate_live(
                self.data,
                frozenset((held_body,)) | allowed_robot_contact_bodies,
            )
            if not valid:
                raise RuntimeError(f"Held trajectory live collision: {reason}")
            held_geoms = [
                geom_id for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == held_body
                and (self.model.geom_contype[geom_id] or self.model.geom_conaffinity[geom_id])
            ]
            for monitored_body, monitored_name in monitored.items():
                for held_geom in held_geoms:
                    for other_geom in range(self.model.ngeom):
                        if int(self.model.geom_bodyid[other_geom]) != monitored_body:
                            continue
                        if not (
                            self.model.geom_contype[other_geom]
                            or self.model.geom_conaffinity[other_geom]
                        ):
                            continue
                        distance = float(mujoco.mj_geomDistance(
                            self.model, self.data, held_geom, other_geom, 0.20, None
                        ))
                        minimum_monitored_clearance = min(minimum_monitored_clearance, distance)
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                first_body = int(self.model.geom_bodyid[contact.geom1])
                second_body = int(self.model.geom_bodyid[contact.geom2])
                if held_body not in (first_body, second_body):
                    continue
                other_body = second_body if first_body == held_body else first_body
                if other_body in robot_bodies:
                    continue
                first_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ) or f"geom_{contact.geom1}"
                second_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ) or f"geom_{contact.geom2}"
                pair = tuple(sorted((first_name, second_name)))
                classification = classify_held_payload_contact(
                    held_body, other_body, frozenset(allowed_payload_contacts)
                )
                if classification == "ALLOWED_TASK_CONTACT":
                    allowed_task_pairs.add(pair)
                    continue
                invalid_pairs.add(pair)
            if invalid_pairs:
                held_position = self.data.xpos[held_body].copy()
                grip_position = self.data.site_xpos[self.grip_site_id].copy()
                joint_error = float(
                    np.max(
                        np.abs(
                            self.data.qpos[self.arm_qpos]
                            - active_waypoint_target
                        )
                    )
                )
                raise RuntimeError(
                    f"Held payload contacted environment during {active_label}: "
                    f"{sorted(invalid_pairs)}; waypoint_index={active_waypoint_index}; "
                    f"held_body_position_world_m={held_position.tolist()}; "
                    f"grip_position_world_m={grip_position.tolist()}; "
                    f"maximum_joint_tracking_error_rad={joint_error:.6f}"
                )

        for waypoint_index, (waypoint, dwell_steps) in enumerate(planned):
            active_label = waypoint.label
            active_waypoint_index = waypoint_index
            active_waypoint_target = waypoint.joints
            settled = 0
            for _ in range(maximum_steps_per_waypoint):
                command = self.data.ctrl[self.arm_actuators]
                command_speed = self.arm_command_speed * command_speed_scale
                delta = np.clip(
                    waypoint.joints - command,
                    -command_speed * self.model.opt.timestep,
                    command_speed * self.model.opt.timestep,
                )
                self.data.ctrl[self.arm_actuators] = command + delta
                self.data.ctrl[self.finger_actuators] = self.profile.closed_command
                mujoco.mj_step(self.model, self.data)
                physics_steps += 1
                if step_callback:
                    step_callback()
                validate_live()
                tracking = float(np.max(np.abs(self.data.qpos[self.arm_qpos] - waypoint.joints)))
                settled = settled + 1 if tracking <= self.intermediate_tracking_tolerance + 0.005 else 0
                if settled >= 5:
                    break
            else:
                tracking = float(
                    np.max(np.abs(self.data.qpos[self.arm_qpos] - waypoint.joints))
                )
                raise RuntimeError(
                    f"Held trajectory tracking timeout: {waypoint.label}; "
                    f"maximum_joint_error_rad={tracking:.6f}"
                )
            for _ in range(dwell_steps):
                mujoco.mj_step(self.model, self.data)
                physics_steps += 1
                if step_callback:
                    step_callback()
                validate_live()
            if dwell_steps:
                dwell_pose_samples.append({
                    "label": waypoint.label,
                    "grip_position_world_m": self.data.site_xpos[self.grip_site_id].tolist(),
                    "held_body_position_world_m": self.data.xpos[held_body].tolist(),
                    "held_body_rotation_world": self.data.xmat[held_body].reshape(3, 3).tolist(),
                    "dwell_steps": dwell_steps,
                })

        return {
            "success": True,
            "held_backend_body": held_name,
            "pose_count": len(poses_world),
            "interpolated_pose_count": len(interpolated_poses),
            "joint_waypoint_count": len(planned),
            "maximum_adjacent_joint_delta_rad": maximum_adjacent_joint_delta,
            "maximum_local_adjacent_joint_delta_rad": (
                maximum_local_adjacent_joint_delta
            ),
            "local_continuity_limit_rad": 0.71,
            "adjacent_joint_deltas": adjacent_joint_deltas,
            "physics_steps": physics_steps,
            "command_speed_scale": float(command_speed_scale),
            "maximum_ik_position_error_m": maximum_position_error,
            "maximum_ik_orientation_error_rad": maximum_angle_error,
            "minimum_monitored_clearance_m": (
                None if math.isinf(minimum_monitored_clearance) else minimum_monitored_clearance
            ),
            "allowed_task_contacts": [
                list(pair) for pair in sorted(allowed_task_pairs)
            ],
            "invalid_collision_pairs": [list(pair) for pair in sorted(invalid_pairs)],
            "direct_object_qpos_write": False,
            "grasp_weld_preserved": True,
            "stance_selected_initial_branch_used": initial_arm_joints is not None,
            "dwell_pose_samples": dwell_pose_samples,
        }

    def reposition_held_payload_base(
        self,
        target_base_qpos: np.ndarray,
        *,
        position_tolerance_m: float = BASE_TARGET_TOLERANCE,
        allowed_payload_contact_body_names: tuple[str, ...] = (),
        allowed_robot_contact_body_names: tuple[str, ...] = (),
        step_callback=None,
        maximum_steps: int = 15000,
    ) -> dict[str, object]:
        """Drive to a bounded local stance with the existing payload weld."""
        if self.mode != "holding" or self.held_object is None:
            raise RuntimeError("Held base reposition requires holding mode")
        held_name = self.held_object
        held_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, held_name
        )
        target = np.asarray(target_base_qpos, float)
        if target.shape != (3,):
            raise ValueError("Base target must contain three qpos values")
        start = self.data.qpos[self.base_qpos].copy()
        if float(np.linalg.norm(target[:2] - start[:2])) > 1.00:
            raise RuntimeError("Held local base reposition exceeds 1.00 m bound")
        held_arm_target = self._current_arm()
        checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile,
            mounting_allowances=self.mounting_allowances,
        )
        invalid_pairs: set[tuple[str, str]] = set()
        allowed_payload_contacts = frozenset(allowed_payload_contact_body_names)
        allowed_robot_contacts = frozenset(
            body_id
            for name in allowed_robot_contact_body_names
            if (body_id := mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )) >= 0
        )
        for step in range(1, maximum_steps + 1):
            self.data.ctrl[self.arm_actuators] = held_arm_target
            self.data.ctrl[self.finger_actuators] = self.profile.closed_command
            self._command_base(target)
            mujoco.mj_step(self.model, self.data)
            if step_callback:
                step_callback()
            valid, reason = checker.evaluate_live(
                self.data,
                frozenset((held_body,)) | allowed_robot_contacts,
            )
            if not valid:
                raise RuntimeError(f"Held local base collision: {reason}")
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                first_body = int(self.model.geom_bodyid[contact.geom1])
                second_body = int(self.model.geom_bodyid[contact.geom2])
                if held_body not in (first_body, second_body):
                    continue
                other_body = second_body if first_body == held_body else first_body
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, other_body
                ) or ""
                if other_name.startswith(f"{self.robot_name}:"):
                    continue
                if other_name in allowed_payload_contacts:
                    continue
                first_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ) or f"geom_{contact.geom1}"
                second_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ) or f"geom_{contact.geom2}"
                invalid_pairs.add(tuple(sorted((first_name, second_name))))
            if invalid_pairs:
                raise RuntimeError(f"Held payload base contact: {sorted(invalid_pairs)}")
            base_error = self.data.qpos[self.base_qpos] - target
            base_error[2] = math.atan2(
                math.sin(base_error[2]), math.cos(base_error[2])
            )
            command_error = self.data.ctrl[self.base_actuators] - target
            command_error[2] = math.atan2(
                math.sin(command_error[2]), math.cos(command_error[2])
            )
            at_target = bool(
                float(np.max(np.abs(base_error))) < position_tolerance_m
                and float(np.max(np.abs(command_error))) < BASE_COMMAND_TOLERANCE
                and float(np.max(np.abs(self.data.qvel[self.base_dofs])))
                < BASE_SETTLE_SPEED
            )
            if at_target:
                self._restore_navigation_base_damping()
                self.base_stance = target.copy()
                self.base_manipulation_target = target.copy()
                return {
                    "success": True,
                    "start_base_qpos": start.tolist(),
                    "target_base_qpos": target.tolist(),
                    "terminal_base_error": base_error.tolist(),
                    "position_tolerance_m": float(position_tolerance_m),
                    "physics_steps": step,
                    "held_object_included_in_collision_check": True,
                    "invalid_collision_pairs": [],
                    "direct_object_qpos_write": False,
                }
        self._restore_navigation_base_damping()
        raise RuntimeError(
            "Held local base reposition timed out; "
            f"current={self.data.qpos[self.base_qpos].tolist()} target={target.tolist()}"
        )

    def adopt_presented_bilateral_grasp(
        self,
        object_name: str,
        target_rotation_world: np.ndarray,
        carry_rotation_world: np.ndarray,
        *,
        preconfirmed_contact_steps: int = 0,
        step_callback=None,
    ) -> dict[str, object]:
        """Authorize a grasp only after sustained post-presentation contact."""
        if self.mode != "idle" or self.held_object is not None:
            raise RuntimeError("Presented regrasp requires an idle empty executor")
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        self.target_object = object_name
        self.target_body_id = body_id
        free_joint_id = int(self.model.body_jntadr[body_id])
        self.target_free_dof = int(self.model.jnt_dofadr[free_joint_id])
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model,
            self.data,
            self.profile,
            mounting_allowances=self.mounting_allowances,
        )
        confirmed = 0
        if (
            preconfirmed_contact_steps >= CONTACT_CONFIRM_TICKS
            and self._finger_contact_sides() == {0, 1}
        ):
            confirmed = int(preconfirmed_contact_steps)
        else:
            for _ in range(100):
                mujoco.mj_step(self.model, self.data)
                if step_callback:
                    step_callback()
                if self._finger_contact_sides() == {0, 1}:
                    confirmed += 1
                    if confirmed >= CONTACT_CONFIRM_TICKS:
                        break
                else:
                    confirmed = 0
        if confirmed < CONTACT_CONFIRM_TICKS:
            self.target_object = None
            self.target_body_id = -1
            self.configuration_checker = None
            raise RuntimeError("REGRASP_FAILED: bilateral contact was not sustained")

        self._activate_weld()
        current = self._current_arm()
        checker = self.configuration_checker
        allowed = frozenset((body_id,))
        ik = ProfiledIK(
            self.model, self.data, self.profile, orientation_weight=0.30
        )
        grip = self.data.site_xpos[self.grip_site_id].copy()
        lift, lift_joints = self._solve_points(
            ik,
            self._cartesian_points(grip, grip + np.array((0.0, 0.0, 0.16)), 0.012),
            current,
            "Extracting presented object above drawer",
            checker,
            allowed,
            target_rotation_world,
        )
        spec = self.pick_specs[object_name]
        carry_position = (
            self.profile.carry_position
            if spec.carry_position is None else spec.carry_position
        )
        home_seed = (
            self.profile.home_seed if spec.home_seed is None else spec.home_seed
        )
        carry_joints, position_error, angle_error = ik.solve(
            carry_position, home_seed, carry_rotation_world
        )
        if (
            position_error > self.ik_position_tolerance
            or angle_error > self.ik_angle_tolerance
        ):
            raise RuntimeError(
                "REGRASP_FAILED: carry IK misses by "
                f"{position_error:.3f} m / {math.degrees(angle_error):.1f} deg"
            )
        collision_free, reason = checker.segment_valid(
            lift_joints, carry_joints, allowed
        )
        if not collision_free:
            raise RuntimeError(f"EXTRACTION_COLLISION: {reason}")
        self.retreat_waypoints = [
            *lift,
            *(
                JointWaypoint(point, "Carrying extracted drawer object")
                for point in self._joint_points(lift_joints, carry_joints)
            ),
        ]
        self.waypoints = self.retreat_waypoints
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.close_target = float(np.max(self.data.ctrl[self.finger_actuators]))
        self.selected_grasp_candidate_id = "presentation_bilateral_regrasp"
        self.mode = "pick_retreat"
        self.status = "Presented handle: bilateral regrasp confirmed; extracting"
        return {
            "bilateral_contact_confirmed": True,
            "contact_confirm_steps": confirmed,
            "attachment_translation_snap_m": self.attachment_translation_snap_m,
            "attachment_angle_snap_rad": self.attachment_angle_snap_rad,
            "selected_grasp_candidate_id": self.selected_grasp_candidate_id,
        }

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
            if label in ("Approaching above object", "Moving above placement site") and active_spec:
                if active_spec.intermediate_ik_position_tolerance is not None:
                    position_tolerance = active_spec.intermediate_ik_position_tolerance
                else:
                    position_tolerance = max(position_tolerance, 0.15)
                if active_spec.intermediate_ik_angle_tolerance_rad is not None:
                    angle_tolerance = active_spec.intermediate_ik_angle_tolerance_rad
                else:
                    angle_tolerance = max(angle_tolerance, np.deg2rad(25.0))
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
        approach_rotation: np.ndarray | None = None,
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
        if active_spec and active_spec.carry_ik_position_tolerance is not None:
            position_tolerance = active_spec.carry_ik_position_tolerance
        if active_spec and active_spec.carry_ik_angle_tolerance_rad is not None:
            angle_tolerance = active_spec.carry_ik_angle_tolerance_rad
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
        approach_rotation = (
            target_rotation
            if approach_rotation is None else approach_rotation
        )
        route_start = carry_position
        route_goals = [
            target + np.asarray(offset, float)
            for offset in (
                active_spec.approach_route_offsets_world_m
                if active_spec is not None else ()
            )
        ] or [pregrasp]
        if active_spec and active_spec.position_first_approach:
            # Aperture traversal constrains position and collision clearance,
            # not an arbitrary wrist attitude.  Find a reachable attitude at
            # the pregrasp point, then retain it through the clearance route;
            # the final descent still solves the strict contact rotation.
            position_ik = ProfiledIK(
                self.model, self.data, self.profile, orientation_weight=0.0
            )
            reachable, position_error, _ = position_ik.solve(
                pregrasp, carry, approach_rotation
            )
            if position_error <= (
                active_spec.intermediate_ik_position_tolerance
                or self.ik_position_tolerance
            ):
                saved_arm = self.data.qpos[self.arm_qpos].copy()
                self.data.qpos[self.arm_qpos] = reachable
                mujoco.mj_forward(self.model, self.data)
                approach_rotation = self.data.site_xmat[
                    self.grip_site_id
                ].reshape(3, 3).copy()
                self.data.qpos[self.arm_qpos] = saved_arm
                mujoco.mj_forward(self.model, self.data)
        if not np.allclose(route_goals[-1], pregrasp):
            route_goals.append(pregrasp)
        for route_index, route_goal in enumerate(route_goals, start=1):
            segment, pregrasp_joints = self._solve_points(
                ik,
                self._cartesian_points(route_start, route_goal),
                pregrasp_joints,
                f"Approaching above object route {route_index}",
                collision_checker,
                allowed_environment_bodies,
                approach_rotation,
            )
            approach.extend(segment)
            route_start = route_goal
        descent, grasp_joints = self._solve_points(
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
        if active_spec and active_spec.retreat_route_offsets_world_m:
            retreat = []
            retreat_joints = grasp_joints
            retreat_start = target
            for offset in active_spec.retreat_route_offsets_world_m:
                retreat_goal = target + np.asarray(offset, dtype=float)
                segment, retreat_joints = self._solve_points(
                    ik,
                    self._cartesian_points(retreat_start, retreat_goal),
                    retreat_joints,
                    "Extracting at constant grasp height",
                    collision_checker,
                    allowed_environment_bodies,
                    target_rotation,
                )
                retreat.extend(segment)
                retreat_start = retreat_goal
            if active_spec.retreat_to_carry_after_route:
                segment, retreat_joints = self._solve_points(
                    ik,
                    self._cartesian_points(retreat_start, carry_position),
                    retreat_joints,
                    "Lifting and returning to carry",
                    collision_checker,
                    allowed_environment_bodies,
                    carry_rotation,
                )
                retreat.extend(segment)
                retreat.append(
                    JointWaypoint(retreat_joints.copy(), "Holding at carry pose")
                )
            else:
                retreat.append(
                    JointWaypoint(
                        retreat_joints.copy(),
                        "Holding at constant-height extraction hover",
                    )
                )
            return waypoints, retreat
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
        self.close_elapsed_ticks = 0
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
        self.planned_grasp_position_world = None
        self.planned_grasp_rotation_world = None
        self.planned_contact_geom_names = ()
        self.planned_contact_points_world = ()
        self.preclose_settle_ticks = 0
        self.preclose_telemetry = None
        self.storage_fixture_release_telemetry = None
        self.calibration_attempt_ticks = 0
        self.mode = "pick_base_approach"
        self.status = f"Pick {object_name}: approaching the manipulation stance"

    def direct_pick_plan_feasibility(self, object_name: str) -> dict[str, object]:
        """Evaluate the normal grasp planner without executing its motion."""
        if self.mode != "idle" or self.held_object is not None:
            raise RuntimeError("Direct-grasp analysis requires an idle executor")
        spec = self.pick_specs[object_name]
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, spec.grasp_site
        )
        original_site = self.model.site_pos[site_id].copy()
        try:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
            )
            free_joint_id = int(self.model.body_jntadr[body_id])
            self.target_object = object_name
            self.target_body_id = body_id
            self.target_free_dof = int(self.model.jnt_dofadr[free_joint_id])
            self.close_target = self.profile.open_command
            self.close_elapsed_ticks = 0
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
            self._begin_pick_plan()
            final_arm_joints = (
                self.waypoints[-1].joints.copy()
                if self.waypoints else self._current_arm()
            )
            return {
                "feasible": True,
                "classification": "DIRECT_GRASP_FEASIBLE",
                "selected_candidate_id": self.selected_grasp_candidate_id,
                "failure": None,
                # Returned solely for temporary geometric ranking.  The live
                # robot still reaches this configuration through actuators.
                "planned_final_arm_joints": final_arm_joints.tolist(),
                "planned_waypoint_count": len(self.waypoints),
                "planned_waypoints": [
                    {
                        "joints": row.joints.tolist(),
                        "label": row.label,
                        "passive_pivot": row.passive_pivot,
                    }
                    for row in self.waypoints
                ],
                "planned_retreat_waypoints": [
                    {
                        "joints": row.joints.tolist(),
                        "label": row.label,
                        "passive_pivot": row.passive_pivot,
                    }
                    for row in self.retreat_waypoints
                ],
                "planned_grasp_position_world_m": (
                    self.planned_grasp_position_world.tolist()
                    if self.planned_grasp_position_world is not None else None
                ),
                "planned_grasp_rotation_world": (
                    self.planned_grasp_rotation_world.tolist()
                    if self.planned_grasp_rotation_world is not None else None
                ),
                "predicted_contact_geom_names": list(
                    self.planned_contact_geom_names
                ),
                "predicted_contact_points_world_m": [
                    list(point) for point in self.planned_contact_points_world
                ],
            }
        except RuntimeError as error:
            return {
                "feasible": False,
                "classification": "DIRECT_GRASP_GEOMETRICALLY_INFEASIBLE",
                "selected_candidate_id": None,
                "failure": str(error),
                "planned_final_arm_joints": None,
                "planned_waypoint_count": 0,
                "planned_waypoints": [],
                "planned_retreat_waypoints": [],
                "planned_grasp_position_world_m": None,
                "planned_grasp_rotation_world": None,
                "predicted_contact_geom_names": [],
                "predicted_contact_points_world_m": [],
            }
        finally:
            self.model.site_pos[site_id] = original_site
            self.pick_specs[object_name] = spec
            self.mode = "idle"
            self.status = "Idle"
            self.failure = None
            self.target_object = None
            self.target_body_id = -1
            self.target_free_dof = -1
            self.configuration_checker = None
            self.waypoints = []
            self.retreat_waypoints = []
            self.selected_grasp_candidate_id = None
            self.planned_grasp_position_world = None
            self.planned_grasp_rotation_world = None
            self.planned_contact_geom_names = ()
            self.planned_contact_points_world = ()
            self.preclose_settle_ticks = 0
            self.preclose_telemetry = None
            mujoco.mj_forward(self.model, self.data)

    def request_preplanned_pick(
        self,
        object_name: str,
        candidate_id: str,
        planned_waypoints: list[dict[str, object]],
        planned_retreat_waypoints: list[dict[str, object]],
        planned_grasp_position_world_m: list[float] | None = None,
        planned_grasp_rotation_world: list[list[float]] | None = None,
        predicted_contact_geom_names: list[str] | None = None,
        predicted_contact_points_world_m: list[list[float]] | None = None,
    ) -> None:
        """Start a physically executed pick from a validated cached plan.

        The cache contains arm joint waypoints only.  It never contains or
        writes target-object state, and live collision/contact guards remain
        active throughout execution.
        """
        self.request_pick(object_name)
        if not planned_waypoints or not planned_retreat_waypoints:
            raise ValueError("A preplanned pick requires approach and retreat paths")
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model,
            self.data,
            self.profile,
            mounting_allowances=self.mounting_allowances,
        )
        self.waypoints = [
            JointWaypoint(
                np.asarray(row["joints"], dtype=float),
                str(row["label"]),
                bool(row.get("passive_pivot", False)),
            )
            for row in planned_waypoints
        ]
        self.retreat_waypoints = [
            JointWaypoint(
                np.asarray(row["joints"], dtype=float),
                str(row["label"]),
                bool(row.get("passive_pivot", False)),
            )
            for row in planned_retreat_waypoints
        ]
        self.selected_grasp_candidate_id = candidate_id
        self.planned_grasp_position_world = (
            None if planned_grasp_position_world_m is None
            else np.asarray(planned_grasp_position_world_m, dtype=float)
        )
        self.planned_grasp_rotation_world = (
            None if planned_grasp_rotation_world is None
            else np.asarray(planned_grasp_rotation_world, dtype=float)
        )
        self.planned_contact_geom_names = tuple(
            predicted_contact_geom_names or ()
        )
        self.planned_contact_points_world = tuple(
            tuple(map(float, point))
            for point in (predicted_contact_points_world_m or ())
        )
        self.preclose_settle_ticks = 0
        self.preclose_telemetry = None
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.mode = "pick_approach"
        self.status = f"Pick {object_name}: executing ranked cached plan"

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
        allowed_bodies = frozenset((
            self.target_body_id,
            *self.allowed_collision_body_ids,
        ))
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
                retreat_route_offsets_world_m=spec.retreat_route_offsets_world_m,
                approach_rotation_world=spec.approach_rotation_world,
                position_first_approach=spec.position_first_approach,
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
                retreat_route_offsets_world_m=candidate.retreat_route_offsets_world_m,
                approach_rotation_world=candidate.approach_rotation_world,
                position_first_approach=candidate.position_first_approach,
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
                    candidate.approach_rotation_world,
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
        self.planned_grasp_position_world = target.copy()
        self.planned_grasp_rotation_world = np.asarray(
            selected.target_rotation_world, dtype=float
        ).copy()
        self.planned_contact_geom_names = tuple(
            selected.predicted_contact_geom_names
        )
        self.planned_contact_points_world = tuple(
            tuple(map(float, point))
            for point in selected.predicted_contact_points_world_m
        )
        self.preclose_settle_ticks = 0
        self.preclose_telemetry = None
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
        if self.pending_place_rotation is None:
            # With no requested wrist rotation, retain the legacy target
            # calculation used by fixed placement sites.
            grip_to_body = (
                self.data.xpos[self.target_body_id]
                - self.data.site_xpos[self.grip_site_id]
            )
            target_grip = desired_body - grip_to_body
        else:
            # The weld preserves the complete gripper-to-object transform,
            # not just the object's world-space offset.  When a dynamic
            # placement requests a new gripper orientation, rotate both the
            # object-relative translation and the site offset into that
            # orientation before solving IK.  Otherwise a non-zero grasp
            # offset (notably the remote) can release above or beside the
            # support; the falling payload then collides with the opening
            # fingers and can destabilize the simulation.
            gripper_rotation = self.data.xmat[self.gripper_body_id].reshape(3, 3)
            object_relative_position = gripper_rotation.T @ (
                self.data.xpos[self.target_body_id]
                - self.data.xpos[self.gripper_body_id]
            )
            site_relative_position = gripper_rotation.T @ (
                self.data.site_xpos[self.grip_site_id]
                - self.data.xpos[self.gripper_body_id]
            )
            desired_gripper_body = desired_body - (
                self.pending_place_rotation @ object_relative_position
            )
            target_grip = desired_gripper_body + (
                self.pending_place_rotation @ site_relative_position
            )
        ik = ProfiledIK(self.model, self.data, self.profile)
        self.configuration_checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile,
            mounting_allowances=self.mounting_allowances,
        )
        allowed_bodies = frozenset((self.target_body_id, *self.allowed_collision_body_ids))
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
        self.place_pre_release_settle_ticks = 0
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
        command_error = float(np.max(np.abs(waypoint.joints - next_command)))
        self.data.ctrl[self.arm_actuators] = next_command
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

    def _finger_pad_centres_world(self) -> tuple[list[float], list[float]]:
        centres = []
        for names in self.profile.finger_contact_geoms:
            geom_ids = [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                for name in names
            ]
            valid = [item for item in geom_ids if item >= 0]
            centre = (
                np.mean(self.data.geom_xpos[valid], axis=0)
                if valid else np.full(3, np.nan)
            )
            centres.append(centre.tolist())
        return centres[0], centres[1]

    def _measure_preclose_pose(self) -> dict[str, object]:
        planned_position = self.planned_grasp_position_world
        planned_rotation = self.planned_grasp_rotation_world
        actual_position = self.data.site_xpos[self.grip_site_id].copy()
        actual_rotation = self.data.site_xmat[self.grip_site_id].reshape(3, 3).copy()
        position_error = (
            float(np.linalg.norm(actual_position - planned_position))
            if planned_position is not None else float("inf")
        )
        orientation_error = (
            float(np.linalg.norm(
                _rotation_vector(planned_rotation @ actual_rotation.T)
            ))
            if planned_rotation is not None else float("inf")
        )
        left_pad, right_pad = self._finger_pad_centres_world()
        live_target_positions = {}
        for name in self.planned_contact_geom_names:
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id >= 0:
                live_target_positions[name] = self.data.geom_xpos[geom_id].tolist()
        return {
            "planned_grip_site_xyz_world_m": (
                planned_position.tolist() if planned_position is not None else None
            ),
            "actual_grip_site_xyz_world_m": actual_position.tolist(),
            "preclose_cartesian_error_m": position_error,
            "planned_gripper_rotation_world": (
                planned_rotation.tolist() if planned_rotation is not None else None
            ),
            "actual_gripper_rotation_world": actual_rotation.tolist(),
            "preclose_orientation_error_rad": orientation_error,
            "actual_left_pad_centre_world_m": left_pad,
            "actual_right_pad_centre_world_m": right_pad,
            "predicted_target_contact_positions_world_m": [
                list(point) for point in self.planned_contact_points_world
            ],
            "live_target_collision_geom_positions_world_m": live_target_positions,
            "preclose_settle_steps": self.preclose_settle_ticks,
            "position_tolerance_m": PRECLOSE_POSITION_TOLERANCE_M,
            "orientation_tolerance_rad": PRECLOSE_ORIENTATION_TOLERANCE_RAD,
        }

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
        # A stored object may be held in a deterministic presentation fixture
        # while the gripper approaches.  Release that fixture only after both
        # fingers have contacted the object, so it cannot fall before grasp.
        for fixture_id in range(self.model.neq):
            fixture_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, fixture_id
            ) or ""
            if not fixture_name.startswith("storage_fixture_"):
                continue
            body1 = int(self.model.eq_obj1id[fixture_id])
            body2 = int(self.model.eq_obj2id[fixture_id])
            if self.target_body_id in {body1, body2}:
                self.data.eq_active[fixture_id] = 0
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
        if self.drawer_pick_collision_exemption:
            return True
        if self.configuration_checker is None:
            return True
        self.collision_guard_tick += 1
        if self.collision_guard_tick % COLLISION_GUARD_INTERVAL:
            return True
        target_contact_modes = {
            "pick_approach",
            "pick_preclose_convergence",
            "closing",
            "pick_retreat",
            "spoon_regrasp",
            "pick_base_retreat",
            "place_base_approach",
            "place_approach",
            "releasing",
        }
        allowed_bodies = (
            frozenset((self.target_body_id, *self.allowed_collision_body_ids))
            if self.target_body_id >= 0 and self.mode in target_contact_modes
            else self.allowed_collision_body_ids
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
            "pick_preclose_convergence",
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
                matching_fixtures = []
                for fixture_id in range(self.model.neq):
                    fixture_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_EQUALITY, fixture_id
                    ) or ""
                    if not fixture_name.startswith("storage_fixture_"):
                        continue
                    if self.target_body_id in {
                        int(self.model.eq_obj1id[fixture_id]),
                        int(self.model.eq_obj2id[fixture_id]),
                    }:
                        matching_fixtures.append(fixture_id)
                active = [
                    fixture_id for fixture_id in matching_fixtures
                    if bool(self.data.eq_active[fixture_id])
                ]
                position_before = self.data.xpos[self.target_body_id].copy()
                for fixture_id in active:
                    self.data.eq_active[fixture_id] = 0
                if active:
                    mujoco.mj_forward(self.model, self.data)
                self.storage_fixture_release_telemetry = {
                    "fixture_ids": matching_fixtures,
                    "active_immediately_before_preclose": bool(active),
                    "released_before_preclose": bool(active),
                    "release_target_position_world_m": position_before.tolist(),
                    "active_during_contact": False,
                    "grasp_weld_active_at_release": bool(
                        self.grasp_equality_id >= 0
                        and self.data.eq_active[self.grasp_equality_id]
                    ),
                }
                self.preclose_settle_ticks = 0
                self.mode = "pick_preclose_convergence"
                self.status = (
                    f"Pick {self.target_object}: validating Cartesian pre-close pose"
                )
            return
        if self.mode == "pick_preclose_convergence":
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            if self.waypoints:
                # Keep commanding the exact final joint solution while the
                # physical arm settles.  Acceptance is Cartesian, not merely
                # a repeat of joint-space waypoint tracking.
                self.data.ctrl[self.arm_actuators] = self.waypoints[-1].joints
            self.preclose_settle_ticks += 1
            measurement = self._measure_preclose_pose()
            self.preclose_telemetry = measurement
            position_ok = (
                measurement["preclose_cartesian_error_m"]
                <= PRECLOSE_POSITION_TOLERANCE_M
            )
            orientation_ok = (
                measurement["preclose_orientation_error_rad"]
                <= PRECLOSE_ORIENTATION_TOLERANCE_RAD
            )
            if position_ok and orientation_ok:
                self.mode = "closing"
                self.status = (
                    f"Pick {self.target_object}: closing until bilateral contact"
                )
                return
            # Static actuator/gravity error can leave the live gripper a few
            # millimetres away from an otherwise valid planned pose.  Refine
            # from the *current* physical arm state rather than waiting on the
            # same stale command.  The correction must still pass the normal
            # robot/environment segment collision checker, and contact/weld
            # authorization remains unchanged.
            if (
                self.preclose_settle_ticks % 20 == 0
                and self.planned_grasp_position_world is not None
                and self.planned_grasp_rotation_world is not None
                and self.configuration_checker is not None
            ):
                current = self._current_arm()
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
                correction, _, _ = ik.solve(
                    self.planned_grasp_position_world,
                    current,
                    self.planned_grasp_rotation_world,
                )
                # Compensate the steady-state actuator deflection observed at
                # the live pose. This is bounded feedback on robot joints,
                # not a target-object pose write or tolerance relaxation.
                correction = current + 2.0 * (correction - current)
                valid, _ = self.configuration_checker.segment_valid(
                    current,
                    correction,
                    frozenset((self.target_body_id,)),
                )
                if valid:
                    self.waypoints[-1] = JointWaypoint(
                        correction.copy(), "Cartesian pre-close correction"
                    )
            if self.preclose_settle_ticks >= PRECLOSE_TIMEOUT_TICKS:
                self._fail(
                    "PRE_CLOSE_CARTESIAN_CONVERGENCE_FAILED: "
                    f"position={measurement['preclose_cartesian_error_m']:.6f} m, "
                    f"orientation={measurement['preclose_orientation_error_rad']:.6f} rad"
                )
            return
        if self.mode == "closing":
            self.close_elapsed_ticks += 1
            self.close_target = min(
                self.profile.closed_command,
                self.close_target + self.profile.close_step,
            )
            self.data.ctrl[self.finger_actuators] = self.close_target
            if self._finger_contact_sides() == {0, 1}:
                self.contact_ticks += 1
                confirm_ticks = max(
                    1,
                    int(self.pick_specs[self.target_object].contact_confirm_ticks),
                )
                if self.contact_ticks >= confirm_ticks:
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
            grace_ticks = int(
                self.pick_specs[self.target_object].close_grace_ticks
            )
            confirm_ticks = max(
                1,
                int(self.pick_specs[self.target_object].contact_confirm_ticks),
            )
            if (
                self.close_target >= self.profile.closed_command
                and self.close_elapsed_ticks >= confirm_ticks + grace_ticks
            ):
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
                pre_release_ticks = max(
                    0,
                    int(self.pick_specs[self.held_object].place_pre_release_settle_ticks),
                )
                if self.place_pre_release_settle_ticks < pre_release_ticks:
                    self.place_pre_release_settle_ticks += 1
                    self.status = (
                        f"Place {self.held_object}: settling at support height "
                        "before release"
                    )
                    return
                if self.grasp_equality_id >= 0:
                    self.data.eq_active[self.grasp_equality_id] = 0
                self.mode = "releasing"
                self.status = f"Place {self.held_object}: opening gripper"
            return
        if self.mode == "releasing":
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            self.release_ticks += 1
            settle_ticks = max(
                1,
                int(self.pick_specs[self.held_object].place_release_settle_ticks),
            )
            if self.release_ticks >= settle_ticks:
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
