"""Staged, object-specific top-down pick execution for the Fetch arm."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np


ARM_JOINTS = (
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
)
ARM_ACTUATORS = tuple(name.replace("_joint", "_actuator") for name in ARM_JOINTS)
FINGER_JOINTS = (
    "robot0:r_gripper_finger_joint",
    "robot0:l_gripper_finger_joint",
)
FINGER_ACTUATORS = tuple(name.replace("_joint", "_actuator") for name in FINGER_JOINTS)
FINGER_GEOMS = {
    "robot0:r_gripper_finger_link",
    "robot0:l_gripper_finger_link",
}

APPROACH_CLEARANCE = 0.08
LIFT_CLEARANCE = APPROACH_CLEARANCE
SPOON_POST_GRASP_CLEARANCE = 0.16
OPEN_WIDTH = 0.05
SPOON_PIVOT_WIDTH = 0.0065
SPOON_PIVOT_BLEND_TICKS = 120
SPOON_PIVOT_PAUSE_WAYPOINTS = 8
SPOON_SWING_DAMPING = 0.0015
SPOON_AXIAL_DOF_DAMPING = 0.02
SPOON_PIVOT_MAX_TORQUE = 0.01
SPOON_VERTICAL_TOLERANCE = math.radians(3.0)
SPOON_SETTLED_ANGULAR_SPEED = 0.6
SPOON_SETTLE_TICKS = 50
CARRY_POSITION = np.array((0.0, -0.82, 0.95))
HORIZONTAL_CARRY_POSITION = np.array((0.0, -0.75, 0.74))
# Joint-space knots are traversed on one continuous clock.  Segment duration is
# proportional to its largest joint displacement, so dense IK waypoints no
# longer become a series of stop-and-settle motions.
ARM_TRAJECTORY_SPEED = 0.24
ARM_TRAJECTORY_MIN_SEGMENT_TIME = 0.030
PIVOT_POSITION_STIFFNESS = 3000.0
PIVOT_POSITION_DAMPING = 55.0
PIVOT_MAX_FORCE = {
    "coffee_jar": 60.0,
    "sugar_jar": 120.0,
}
PIVOT_ORIENTATION_GAINS = {
    # tilt stiffness/damping/limit, then yaw stiffness/damping/limit. Yaw
    # inertia is especially small for the narrow sugar cylinder, so treating
    # it like the transverse axes produces a visible high-frequency spin.
    "coffee_jar": (20.0, 1.0, 3.0, 30.0, 2.0, 5.0),
    "sugar_jar": (4.0, 0.40, 0.60, 0.02, 0.0025, 0.03),
}
# A fixed seed prevents small actuator drift in the long-running viewer from
# selecting a different (and sometimes unreachable) IK branch.
HOME_ARM_SEED = np.array((1.32, 1.40, -0.20, 1.72, 0.0, 1.66, 0.0))

HOME_BASE_POSE = (0.0, -1.10, 0.0)
PICK_BASE_POSES = {
    "home": HOME_BASE_POSE,
    "cupboard1": (-1.025, -0.10, -math.pi / 2),
    "right_side": (1.025, -0.10, math.pi / 2),
}
PICK_LOCATION_REGIONS = {
    "home": (
        (-0.36, 0.36, -0.37, -0.14),
        (-0.25, 0.25, -0.71, -0.41),
    ),
    "cupboard1": ((-0.68, -0.36, -0.34, 0.22),),
    "right_side": ((0.36, 0.68, -0.37, -0.12),),
}

# The Fetch gripper approaches along its local +X axis. The matrix columns are
# local X/Y/Z in world coordinates: approach down, fingers close along world Y,
# and the remaining gripper axis lies along world X.
TOP_DOWN_ROTATION = np.array(
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
)


def _yaw_rotation(yaw: float) -> np.ndarray:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )


def carry_position_at(
    canonical_position: np.ndarray, physical_location: str
) -> np.ndarray:
    """Transform a home-frame carry point with the current named base pose."""
    try:
        base_x, base_y, yaw = PICK_BASE_POSES[physical_location]
    except KeyError as error:
        raise ValueError(f"Unknown pick base pose: {physical_location}") from error
    home_x, home_y, _ = HOME_BASE_POSE
    relative = canonical_position.copy()
    relative[:2] -= (home_x, home_y)
    return np.array((base_x, base_y, 0.0)) + _yaw_rotation(yaw) @ relative


def object_reachable_from_location(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_name: str,
    physical_location: str,
) -> bool:
    """Return whether an object's live centre lies in this base pose's region."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    if body_id < 0 or physical_location not in PICK_LOCATION_REGIONS:
        return False
    mujoco.mj_forward(model, data)
    x, y = data.xpos[body_id, :2]
    tolerance = 0.025
    return any(
        min_x - tolerance <= x <= max_x + tolerance
        and min_y - tolerance <= y <= max_y + tolerance
        for min_x, max_x, min_y, max_y in PICK_LOCATION_REGIONS[physical_location]
    )

# Cylindrical jars may be pinched on any horizontal diameter. A diagonal
# closing axis gives both table locations a continuous IK route. Pitching
# exactly 90 degrees about local Y then makes the gripper horizontal without
# changing that contact axis.
_JAR_YAW = math.radians(45.0)
_JAR_CLOSING_AXIS = np.array((math.sin(_JAR_YAW), math.cos(_JAR_YAW), 0.0))
_JAR_APPROACH_AXIS = np.array((0.0, 0.0, -1.0))
JAR_TOP_DOWN_ROTATION = np.column_stack(
    (
        _JAR_APPROACH_AXIS,
        _JAR_CLOSING_AXIS,
        np.cross(_JAR_APPROACH_AXIS, _JAR_CLOSING_AXIS),
    )
)
_PITCH_90_LOCAL_Y = np.array(
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
)
JAR_HORIZONTAL_ROTATION = JAR_TOP_DOWN_ROTATION @ _PITCH_90_LOCAL_Y


@dataclass(frozen=True)
class PickSpec:
    label: str
    grasp_site: str
    grasp_z_offset: float = 0.0
    required_contact_geoms: tuple[str, ...] = ()
    align_to_body: bool = False
    closing_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0)
    reorient_horizontal: bool = False


TABLE_PICK_SPECS = {
    "kettle": PickSpec(
        "Kettle (handle)",
        "kettle_grasp",
        # Fetch's finger pads extend below robot0:grip; this aligns their
        # contact band, rather than the abstract grip-site origin, to the tube.
        grasp_z_offset=0.008,
        required_contact_geoms=("kettle_handle_collision",),
        align_to_body=True,
        # The scanned handle tube runs along local (+X,-Y); close across it.
        closing_axis_local=(math.sqrt(0.5), math.sqrt(0.5), 0.0),
    ),
    "coffee_jar": PickSpec(
        "Coffee jar (upper body)",
        "coffee_jar_grasp",
        reorient_horizontal=True,
    ),
    "sugar_jar": PickSpec(
        "Sugar jar (upper body)",
        "sugar_jar_grasp",
        reorient_horizontal=True,
    ),
    # The thin handle is slightly above the support plane when the gripper
    # centre reaches it; the finger tips extend below the grip site.
    "spoon": PickSpec(
        "Spoon (handle)",
        "spoon_grasp",
        0.012,
        ("spoon_handle_collision",),
    ),
}


@dataclass
class ArmWaypoint:
    joints: np.ndarray
    label: str
    stabilize_object: bool = False
    passive_pivot: bool = False


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle vector represented by a rotation."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, matrix.ravel())
    if quat[0] < 0:
        quat = -quat
    norm = float(np.linalg.norm(quat[1:]))
    if norm < 1e-10:
        return np.zeros(3)
    return quat[1:] / norm * (2.0 * math.atan2(norm, float(quat[0])))


def top_down_rotation_for_body(
    body_rotation: np.ndarray,
    closing_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Map an object's grasp-cross-axis into world while approaching down."""
    closing_axis = body_rotation @ np.asarray(closing_axis_local, dtype=float)
    closing_axis[2] = 0.0
    norm = float(np.linalg.norm(closing_axis))
    if norm < 1e-8:
        return TOP_DOWN_ROTATION.copy()
    closing_axis /= norm
    # Finger closing is symmetric, so select the equivalent direction nearest
    # the canonical pose to avoid an unnecessary 180-degree wrist rotation.
    if float(closing_axis @ TOP_DOWN_ROTATION[:, 1]) < 0:
        closing_axis = -closing_axis
    approach_axis = np.array((0.0, 0.0, -1.0))
    remaining_axis = np.cross(approach_axis, closing_axis)
    return np.column_stack((approach_axis, closing_axis, remaining_axis))


class VerticalIK:
    """Damped least-squares IK constrained to a vertical gripper attitude."""

    def __init__(self, model: mujoco.MjModel, reference: mujoco.MjData):
        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = reference.qpos
        self.data.qvel[:] = 0
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip"
        )
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINTS
            ]
        )
        self.qpos_addresses = model.jnt_qposadr[self.joint_ids]
        self.dof_addresses = model.jnt_dofadr[self.joint_ids]
        limits = model.jnt_range[self.joint_ids]
        limited = model.jnt_limited[self.joint_ids].astype(bool)
        self.lower = np.where(limited, limits[:, 0] + 0.01, -math.pi + 0.01)
        self.upper = np.where(limited, limits[:, 1] - 0.01, math.pi - 0.01)

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        target_rotation: np.ndarray = TOP_DOWN_ROTATION,
    ) -> tuple[np.ndarray, float, float]:
        self.data.qpos[self.qpos_addresses] = seed
        for _ in range(900):
            mujoco.mj_forward(self.model, self.data)
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target - self.data.site_xpos[self.site_id]
            rotation_error = _rotation_vector(
                target_rotation @ current_rotation.T
            )
            error = np.concatenate((position_error, 0.40 * rotation_error))
            if (
                np.linalg.norm(position_error) < 0.0006
                and np.linalg.norm(rotation_error) < math.radians(0.5)
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
                    0.40 * jac_rot[:, self.dof_addresses],
                )
            )
            damping = 0.0015
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            current = self.data.qpos[self.qpos_addresses]
            self.data.qpos[self.qpos_addresses] = np.clip(
                current + np.clip(delta, -0.06, 0.06), self.lower, self.upper
            )

        mujoco.mj_forward(self.model, self.data)
        current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        position_error = float(
            np.linalg.norm(target - self.data.site_xpos[self.site_id])
        )
        angle_error = float(
            np.linalg.norm(_rotation_vector(target_rotation @ current_rotation.T))
        )
        return (
            self.data.qpos[self.qpos_addresses].copy(),
            position_error,
            angle_error,
        )


class PickExecutor:
    """Plan and execute one contact-confirmed pick from the home base pose."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINTS
            ]
        )
        self.arm_qpos = model.jnt_qposadr[self.arm_joint_ids]
        self.arm_actuators = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATORS
            ]
        )
        self.finger_joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FINGER_JOINTS
            ]
        )
        self.finger_qpos = model.jnt_qposadr[self.finger_joint_ids]
        self.finger_actuators = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in FINGER_ACTUATORS
            ]
        )
        self.waypoints: list[ArmWaypoint] = []
        self.waypoint_index = 0
        self.trajectory_times = np.zeros(0)
        self.trajectory_derivatives = np.empty((0, len(ARM_JOINTS)))
        self.trajectory_time = 0.0
        self.trajectory_segment = 0
        self.mode = "idle"
        self.target_object: str | None = None
        self.physical_location = "home"
        self.target_body_id = -1
        self.target_free_dof = -1
        self.held_object: str | None = None
        # Exact final joint command shared with the subsequent place action.
        # Using the live, gravity-sagged qpos here would create a second,
        # visibly different "place carry" pose.
        self.carry_goal_joints: np.ndarray | None = None
        self.close_target = OPEN_WIDTH
        self.close_ticks = 0
        self.contact_ticks = 0
        self.grasp_equality_id = -1
        self.spoon_pivot_equality_id = -1
        self.spoon_settle_ticks = 0
        self.spoon_pivot_ticks = 0
        self.spoon_pivot_start_width = SPOON_PIVOT_WIDTH
        self.spoon_original_axial_damping = 0.0
        self.gripper_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link"
        )
        self.grip_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip"
        )
        self.pivot_anchor_pos: np.ndarray | None = None
        self.pivot_anchor_rotation: np.ndarray | None = None
        self.pivot_anchor_gripper_pos: np.ndarray | None = None
        self.pivot_anchor_follows_gripper = False
        self.pivot_active = False
        self.status = "Pick idle: gripper empty"
        self.failure: str | None = None

    @property
    def busy(self) -> bool:
        return self.mode not in {"idle", "holding", "failed"}

    def _current_arm(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos].copy()

    @staticmethod
    def _pchip_derivatives(points: np.ndarray, times: np.ndarray) -> np.ndarray:
        """C1 path tangents without artificial stops at dense IK knots."""
        derivatives = np.zeros_like(points)
        if len(points) <= 2:
            return derivatives

        intervals = np.diff(times)
        slopes = np.diff(points, axis=0) / intervals[:, None]
        for index in range(1, len(points) - 1):
            previous_interval = intervals[index - 1]
            following_interval = intervals[index]
            previous = slopes[index - 1]
            following = slopes[index]
            monotonic = previous * following > 0.0
            weight_previous = 2.0 * following_interval + previous_interval
            weight_following = following_interval + 2.0 * previous_interval
            derivatives[index, monotonic] = (
                weight_previous + weight_following
            ) / (
                weight_previous / previous[monotonic]
                + weight_following / following[monotonic]
            )
            # Ignore tiny numerical reversals in otherwise smooth IK paths;
            # braking these low-motion joints at every dense knot is what
            # produced the visible accelerate-stop pattern. Large, genuine
            # reversals retain a zero tangent and cannot overshoot.
            reversal = previous * following < 0.0
            magnitude = np.maximum(np.abs(previous), np.abs(following))
            minor = reversal & (
                np.minimum(np.abs(previous), np.abs(following))
                < 0.12 * np.maximum(magnitude, 1e-12)
            )
            derivatives[index, minor] = (
                following_interval * previous[minor]
                + previous_interval * following[minor]
            ) / (previous_interval + following_interval)
        # Starting and finishing at rest keeps action boundaries smooth; only
        # genuine joint reversals stop at an internal knot.
        return derivatives

    def _start_trajectory(self, waypoints: list[ArmWaypoint]) -> None:
        """Time-parameterize a waypoint path as one continuous joint curve."""
        if not waypoints:
            raise RuntimeError("Cannot execute an empty pick trajectory")
        # Begin from the actuator's live command, not qpos.  Under gravity the
        # two differ at rest; commanding qpos immediately would itself create
        # a large force discontinuity at the start of every action.
        current_command = self.data.ctrl[self.arm_actuators].copy()
        if not np.allclose(current_command, waypoints[0].joints, atol=1e-8):
            waypoints = [
                ArmWaypoint(
                    current_command,
                    waypoints[0].label,
                    waypoints[0].stabilize_object,
                ),
                *waypoints,
            ]
        self.waypoints = waypoints
        points = np.asarray([waypoint.joints for waypoint in waypoints])
        displacements = np.max(np.abs(np.diff(points, axis=0)), axis=1)
        durations = np.maximum(
            displacements / ARM_TRAJECTORY_SPEED,
            ARM_TRAJECTORY_MIN_SEGMENT_TIME,
        )
        self.trajectory_times = np.concatenate(([0.0], np.cumsum(durations)))
        self.trajectory_derivatives = self._pchip_derivatives(
            points, self.trajectory_times
        )
        self.trajectory_time = 0.0
        self.trajectory_segment = 0
        self.waypoint_index = 0

    def _sample_trajectory(self) -> tuple[np.ndarray, ArmWaypoint]:
        """Sample the current Hermite segment and return its destination knot."""
        points = np.asarray([waypoint.joints for waypoint in self.waypoints])
        while (
            self.trajectory_segment < len(points) - 2
            and self.trajectory_time
            >= self.trajectory_times[self.trajectory_segment + 1]
        ):
            self.trajectory_segment += 1

        index = self.trajectory_segment
        start_time = self.trajectory_times[index]
        duration = self.trajectory_times[index + 1] - start_time
        phase = float(
            np.clip((self.trajectory_time - start_time) / duration, 0.0, 1.0)
        )
        phase2 = phase * phase
        phase3 = phase2 * phase
        target = (
            (2.0 * phase3 - 3.0 * phase2 + 1.0) * points[index]
            + (phase3 - 2.0 * phase2 + phase)
            * duration
            * self.trajectory_derivatives[index]
            + (-2.0 * phase3 + 3.0 * phase2) * points[index + 1]
            + (phase3 - phase2)
            * duration
            * self.trajectory_derivatives[index + 1]
        )
        self.waypoint_index = index + 1
        return target, self.waypoints[index + 1]

    def _advance_trajectory(
        self, final_tolerance: float = 0.012
    ) -> tuple[bool, ArmWaypoint]:
        target, waypoint = self._sample_trajectory()
        self.data.ctrl[self.arm_actuators] = target
        end_time = float(self.trajectory_times[-1])
        if self.trajectory_time < end_time:
            self.trajectory_time = min(
                end_time, self.trajectory_time + self.model.opt.timestep
            )
            return False, waypoint
        final_error = float(
            np.max(np.abs(self.data.qpos[self.arm_qpos] - self.waypoints[-1].joints))
        )
        return final_error < final_tolerance, waypoint

    @staticmethod
    def _joint_interpolation(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        count = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / 0.055)))
        return [start + f * (goal - start) for f in np.linspace(0, 1, count + 1)[1:]]

    @staticmethod
    def _cartesian_points(start: np.ndarray, goal: np.ndarray, resolution: float) -> list[np.ndarray]:
        count = max(1, int(math.ceil(float(np.linalg.norm(goal - start)) / resolution)))
        return [start + f * (goal - start) for f in np.linspace(0, 1, count + 1)[1:]]

    def _solve_path(
        self,
        ik: VerticalIK,
        points: list[np.ndarray],
        seed: np.ndarray,
        label: str,
        target_rotation: np.ndarray,
        *,
        position_tolerance: float = 0.012,
        angle_tolerance: float = math.radians(2.0),
    ) -> tuple[list[ArmWaypoint], np.ndarray]:
        output = []
        current = seed
        for point in points:
            current, pos_error, angle_error = ik.solve(
                point, current, target_rotation
            )
            if pos_error > position_tolerance or angle_error > angle_tolerance:
                raise RuntimeError(
                    f"vertical IK misses {label} by {pos_error * 100:.1f} cm "
                    f"with {math.degrees(angle_error):.1f} deg tilt"
                )
            output.append(ArmWaypoint(current.copy(), label))
        return output, current

    def request_pick(
        self, object_name: str, physical_location: str = "home"
    ) -> None:
        if self.busy:
            raise RuntimeError("A pick action is already running")
        if self.held_object is not None:
            raise RuntimeError(f"Gripper already holds {self.held_object}")
        if object_name not in TABLE_PICK_SPECS:
            raise ValueError(f"Unsupported tabletop pick object: {object_name}")
        spec = TABLE_PICK_SPECS[object_name]
        site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, spec.grasp_site
        )
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if site_id < 0 or body_id < 0:
            raise RuntimeError(f"{object_name} is not present in this scene")
        if not object_reachable_from_location(
            self.model, self.data, object_name, physical_location
        ):
            raise RuntimeError(
                f"{object_name} is not in the table region reachable from "
                f"{physical_location}"
            )

        mujoco.mj_forward(self.model, self.data)
        _, _, base_yaw = PICK_BASE_POSES[physical_location]
        base_rotation = _yaw_rotation(base_yaw)
        carry_position = carry_position_at(CARRY_POSITION, physical_location)
        horizontal_carry_position = carry_position_at(
            HORIZONTAL_CARRY_POSITION, physical_location
        )
        grasp = self.data.site_xpos[site_id].copy()
        grasp[2] += spec.grasp_z_offset
        if spec.reorient_horizontal and physical_location != "home":
            # A side release can leave a jar tilted while the hand retreats,
            # which displaces its body-fixed top site away from the visible
            # centre. Re-pick the live body centre and enter its side-wall
            # band instead of closing over that displaced site or the rim.
            grasp[:2] = self.data.xpos[body_id, :2]
            grasp[2] = self.data.xpos[body_id, 2] + 0.025
        pregrasp = grasp + np.array((0.0, 0.0, APPROACH_CLEARANCE))
        if spec.reorient_horizontal:
            # Jar grasp/carry orientation is base-relative at every location,
            # so every jar pick ends with the same horizontal gripper state.
            target_rotation = base_rotation @ JAR_TOP_DOWN_ROTATION
        else:
            target_rotation = base_rotation @ TOP_DOWN_ROTATION
        if spec.align_to_body:
            target_rotation = top_down_rotation_for_body(
                self.data.xmat[body_id].reshape(3, 3),
                spec.closing_axis_local,
            )

        ik = VerticalIK(self.model, self.data)
        current = self._current_arm()
        # Wrist roll directly supplies the top-down yaw. Seeding that degree
        # of freedom avoids an equivalent IK branch that winds several arm
        # rolls toward their limits and can self-collide on the way to carry.
        carry_seed = HOME_ARM_SEED.copy()
        local_target_rotation = base_rotation.T @ target_rotation
        carry_seed[-1] += math.atan2(
            float(local_target_rotation[0, 1]),
            float(local_target_rotation[1, 1]),
        )
        carry_joints, carry_error, carry_angle = ik.solve(
            carry_position, carry_seed, target_rotation
        )
        if carry_error > 0.006 or carry_angle > math.radians(1.0):
            raise RuntimeError("Could not reach the vertical carry pose")

        waypoints = [ArmWaypoint(current, "Opening gripper")]
        waypoints.extend(
            ArmWaypoint(q, "Moving from rest to carry clearance")
            for q in self._joint_interpolation(current, carry_joints)
        )
        approach_points = self._cartesian_points(
            carry_position, pregrasp, 0.035
        )
        approach, pregrasp_joints = self._solve_path(
            ik, approach_points, carry_joints, "pre-grasp", target_rotation
        )
        waypoints.extend(approach)
        descent_points = self._cartesian_points(pregrasp, grasp, 0.012)
        descent, _ = self._solve_path(
            ik, descent_points, pregrasp_joints, "vertical descent", target_rotation
        )
        waypoints.extend(descent)

        # First lift along the exact descent route so the held object clears
        # the table before any change in wrist attitude.
        post_grasp = [
            ArmWaypoint(w.joints.copy(), "Lifting vertically")
            for w in reversed(descent[:-1])
        ]
        post_grasp.append(ArmWaypoint(pregrasp_joints.copy(), "Lift clear"))

        if spec.reorient_horizontal:
            # Pitch the gripper 90 degrees around its unchanged finger-contact
            # axis. The grip-site origin is above the actual finger contact, so
            # it must travel around that contact rather than rotate in place.
            # Otherwise a short jar is geometrically swept out of the pads.
            pivot_joints = pregrasp_joints.copy()
            pivot_grip_position = pregrasp.copy()
            lifted_body_position = self.data.xpos[body_id] + (pregrasp - grasp)
            if physical_location != "home":
                # Recreate the known-good home pivot geometry in the current
                # base frame. The jar remains vertical while moving from the
                # side table strip into this clear corridor, then performs the
                # identical 90-degree compliant slip near the robot centreline.
                canonical_pivot = np.array((0.0, -0.30, pregrasp[2]))
                side_pivot = carry_position_at(
                    canonical_pivot, physical_location
                )
                corridor_points = self._cartesian_points(
                    pregrasp, side_pivot, 0.025
                )
                corridor_path, pivot_joints = self._solve_path(
                    ik,
                    corridor_points,
                    pivot_joints,
                    "Moving vertical jar into reorientation corridor",
                    target_rotation,
                    position_tolerance=0.018,
                    angle_tolerance=math.radians(3.0),
                )
                post_grasp.extend(corridor_path)
                pivot_grip_position = side_pivot
                lifted_body_position = self.data.xpos[body_id] + (
                    side_pivot - grasp
                )
            contact_offset_local = target_rotation.T @ (
                lifted_body_position - pivot_grip_position
            )
            if object_name == "coffee_jar":
                # Its broad upper shell surrounds the gripper origin's contact
                # band; translating around the body centre would instead drag
                # the pads across the polygonal rim. The narrow sugar jar is
                # pinched at the pad tips and needs the offset pivot above.
                contact_offset_local[:] = 0.0
            for angle in np.linspace(0.0, math.pi / 2.0, 19)[1:]:
                cosine = math.cos(float(angle))
                sine = math.sin(float(angle))
                local_pitch = np.array(
                    (
                        (cosine, 0.0, sine),
                        (0.0, 1.0, 0.0),
                        (-sine, 0.0, cosine),
                    )
                )
                pivot_rotation = target_rotation @ local_pitch
                pivot_grip_position = (
                    lifted_body_position
                    - pivot_rotation @ contact_offset_local
                )
                pivot_joints, pos_error, angle_error = ik.solve(
                    pivot_grip_position,
                    pivot_joints,
                    pivot_rotation,
                )
                pivot_position_tolerance = 0.012
                pivot_angle_tolerance = math.radians(2.0)
                if (
                    pos_error > pivot_position_tolerance
                    or angle_error > pivot_angle_tolerance
                ):
                    raise RuntimeError(
                        "Could not keep the jar near its pivot during "
                        f"reorientation ({pos_error * 100:.1f} cm, "
                        f"{math.degrees(angle_error):.1f} deg)"
                    )
                post_grasp.append(
                    ArmWaypoint(
                        pivot_joints.copy(),
                        "Reorienting gripper 90 degrees around held jar",
                        stabilize_object=True,
                    )
                )

            horizontal_points = self._cartesian_points(
                pivot_grip_position, horizontal_carry_position, 0.020
            )
            horizontal_rotation = target_rotation @ _PITCH_90_LOCAL_Y
            horizontal_return, horizontal_carry_joints = self._solve_path(
                ik,
                horizontal_points,
                pivot_joints,
                "Returning horizontally to carry pose",
                horizontal_rotation,
            )
            # Keep the same compliant world-orientation spring active while
            # translating away from the pivot. Its positional anchor follows
            # the gripper, so there is no hard constraint handoff mid-motion.
            post_grasp.extend(
                ArmWaypoint(
                    waypoint.joints.copy(),
                    waypoint.label,
                    # A short moving-anchor blend lets residual angular
                    # velocity decay before the live grasp is captured. The
                    # remainder of carry uses the ordinary physical grasp.
                    stabilize_object=index < 4,
                )
                for index, waypoint in enumerate(horizontal_return)
            )
            post_grasp.append(
                ArmWaypoint(
                    horizontal_carry_joints.copy(),
                    "Jar secured in horizontal carry pose",
                )
            )
        else:
            passive_spoon = object_name == "spoon"
            if passive_spoon:
                # Lift twice the ordinary clearance before beginning any
                # lateral carry motion, keeping the long spoon clear of the
                # tabletop while it transitions into its passive hang.
                high_hover = grasp + np.array(
                    (0.0, 0.0, SPOON_POST_GRASP_CLEARANCE)
                )
                extra_lift_points = self._cartesian_points(
                    pregrasp, high_hover, 0.012
                )
                extra_lift, high_hover_joints = self._solve_path(
                    ik,
                    extra_lift_points,
                    pregrasp_joints,
                    "Raising spoon to high post-grasp hover",
                    target_rotation,
                )
                post_grasp.extend(extra_lift)
                post_grasp.extend(
                    ArmWaypoint(
                        high_hover_joints.copy(),
                        "Blending spoon grasp into passive pivot",
                        passive_pivot=True,
                    )
                    for _ in range(SPOON_PIVOT_PAUSE_WAYPOINTS)
                )
                return_points = self._cartesian_points(
                    high_hover, carry_position, 0.035
                )
                return_path, _ = self._solve_path(
                    ik,
                    return_points,
                    high_hover_joints,
                    "Returning while spoon hangs from handle",
                    target_rotation,
                )
                post_grasp.extend(
                    ArmWaypoint(
                        waypoint.joints.copy(),
                        waypoint.label,
                        passive_pivot=True,
                    )
                    for waypoint in return_path
                )
            else:
                # Preserve the vertical grasp attitude and retrace the
                # collision-checked overhead approach corridor.
                post_grasp.extend(
                    ArmWaypoint(w.joints.copy(), "Returning to carry pose")
                    for w in reversed(approach[:-1])
                )
            post_grasp.append(
                ArmWaypoint(
                    carry_joints.copy(),
                    (
                        "Spoon hanging vertically in carry pose"
                        if passive_spoon
                        else "Object secured in carry pose"
                    ),
                    passive_pivot=passive_spoon,
                )
            )

        self.post_grasp_waypoints = post_grasp
        self.carry_goal_joints = post_grasp[-1].joints.copy()
        self.target_object = object_name
        self.physical_location = physical_location
        self.target_body_id = body_id
        free_joint_id = int(self.model.body_jntadr[body_id])
        if (
            free_joint_id < 0
            or self.model.jnt_type[free_joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError(f"{object_name} does not have a free joint")
        self.target_free_dof = int(self.model.jnt_dofadr[free_joint_id])
        self.mode = "approach"
        self.close_target = OPEN_WIDTH
        self.close_ticks = 0
        self.contact_ticks = 0
        self.grasp_equality_id = -1
        self.spoon_pivot_equality_id = -1
        self.spoon_settle_ticks = 0
        self.spoon_pivot_ticks = 0
        self.spoon_pivot_start_width = SPOON_PIVOT_WIDTH
        self.spoon_original_axial_damping = 0.0
        self.pivot_anchor_pos = None
        self.pivot_anchor_rotation = None
        self.pivot_anchor_gripper_pos = None
        self.pivot_anchor_follows_gripper = False
        self.pivot_active = False
        self.failure = None
        self._start_trajectory(waypoints)
        self.status = f"Pick {object_name}: planning complete, opening gripper"

    def _target_finger_contacts(self) -> set[str]:
        assert self.target_object is not None
        required = TABLE_PICK_SPECS[self.target_object].required_contact_geoms
        contacts: set[str] = set()
        for contact in self.data.contact:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if self.target_body_id not in {body1, body2}:
                continue
            target_geom = contact.geom1 if body1 == self.target_body_id else contact.geom2
            target_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, target_geom
            ) or ""
            if required and target_name not in required:
                continue
            other_geom = contact.geom2 if body1 == self.target_body_id else contact.geom1
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or ""
            if other_name in FINGER_GEOMS:
                contacts.add(other_name)
        return contacts

    def _activate_grasp_weld(self) -> None:
        assert self.target_object is not None
        equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"robot0:pick_weld_{self.target_object}",
        )
        if equality_id < 0:
            raise RuntimeError(f"Missing grasp constraint for {self.target_object}")
        self.grasp_equality_id = equality_id
        self._set_grasp_weld_world_pose(
            self.data.xpos[self.target_body_id],
            self.data.xquat[self.target_body_id],
        )
        self.data.eq_active[equality_id] = 1

    def _set_grasp_weld_world_pose(
        self, object_pos: np.ndarray, object_quat: np.ndarray
    ) -> None:
        """Set the weld so a requested object world pose is maintained."""
        if self.grasp_equality_id < 0:
            raise RuntimeError("Grasp constraint has not been activated")
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
            "robot0:pick_pivot_spoon",
        )
        grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "spoon_grasp"
        )
        if equality_id < 0 or grasp_site_id < 0:
            raise RuntimeError("Missing passive spoon pivot constraint")

        # Both local anchors are calculated from one live world point, so the
        # constraint activates without snapping either the hand or the spoon.
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
        self.spoon_pivot_ticks = 0
        self.spoon_pivot_start_width = float(
            np.mean(self.data.qpos[self.finger_qpos])
        )
        self.spoon_original_axial_damping = float(
            self.model.dof_damping[self.target_free_dof]
        )
        self.model.dof_damping[self.target_free_dof] = SPOON_AXIAL_DOF_DAMPING
        self.data.ctrl[self.finger_actuators] = self.spoon_pivot_start_width

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
        # The visible bowl is on the object's local -X end; the grasp is now
        # on the opposite +X red-handle tip.
        spoon_axis = -self.data.xmat[self.target_body_id].reshape(3, 3)[:, 0]
        angular_velocity = velocity[:3]
        axial_velocity = spoon_axis * float(angular_velocity @ spoon_axis)
        swing_velocity = angular_velocity - axial_velocity
        torque = self._limited(
            -SPOON_SWING_DAMPING * swing_velocity,
            SPOON_PIVOT_MAX_TORQUE,
        )
        self.data.xfrc_applied[self.target_body_id, 3:] = torque
        bowl_down_angle = math.acos(
            float(np.clip(spoon_axis @ np.array((0.0, 0.0, -1.0)), -1.0, 1.0))
        )
        return bowl_down_angle, float(np.linalg.norm(angular_velocity))

    def _finish_spoon_pivot(self) -> None:
        """Capture the settled live pose so the carried spoon cannot spin."""
        if self.spoon_pivot_equality_id < 0 or self.grasp_equality_id < 0:
            raise RuntimeError("Cannot finish an inactive spoon pivot")
        # Transverse swing is already settled here, but the nearly symmetric
        # handle can retain a fast, visually obvious axial spin. Remove that
        # residual momentum exactly at the transition to the final grasp.
        self.data.qvel[self.target_free_dof : self.target_free_dof + 6] = 0.0
        self._set_grasp_weld_world_pose(
            self.data.xpos[self.target_body_id],
            self.data.xquat[self.target_body_id],
        )
        # The ordinary transport weld is intentionally a little compliant.
        # Tighten it only for the settled spoon so gravity cannot leave a
        # visible residual twist around the very thin handle.
        self.model.eq_solref[self.grasp_equality_id] = (0.003, 1.0)
        self.data.eq_active[self.spoon_pivot_equality_id] = 0
        self.data.eq_active[self.grasp_equality_id] = 1
        self.model.dof_damping[
            self.target_free_dof
        ] = self.spoon_original_axial_damping
        self.data.xfrc_applied[self.target_body_id] = 0.0

    @staticmethod
    def _limited(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= maximum or norm < 1e-12:
            return vector
        return vector * (maximum / norm)

    def _begin_compliant_pivot(self) -> None:
        """Release the weld and start a soft, contact-retaining jar pivot."""
        if self.grasp_equality_id < 0 or self.target_free_dof < 0:
            raise RuntimeError("Jar pivot requested without an active grasp")
        self.pivot_anchor_pos = self.data.xpos[self.target_body_id].copy()
        self.pivot_anchor_rotation = self.data.xmat[self.target_body_id].reshape(
            3, 3
        ).copy()
        self.pivot_anchor_gripper_pos = self.data.site_xpos[
            self.grip_site_id
        ].copy()
        self.pivot_anchor_follows_gripper = False
        self.pivot_active = True
        self.data.eq_active[self.grasp_equality_id] = 0

    def _begin_compliant_translation(self) -> None:
        """Move the soft position anchor with the hand after pivoting."""
        self.pivot_anchor_pos = self.data.xpos[self.target_body_id].copy()
        self.pivot_anchor_gripper_pos = self.data.site_xpos[
            self.grip_site_id
        ].copy()
        self.pivot_anchor_follows_gripper = True

    def _apply_compliant_pivot(self) -> None:
        """Allow contact motion while softly retaining the jar's world pose."""
        assert self.pivot_anchor_pos is not None
        assert self.pivot_anchor_rotation is not None
        assert self.pivot_anchor_gripper_pos is not None
        object_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.target_body_id,
            object_velocity,
            0,
        )
        angular_velocity = object_velocity[:3]
        linear_velocity = object_velocity[3:]
        moving_anchor = self.pivot_anchor_pos
        if self.pivot_anchor_follows_gripper:
            moving_anchor = moving_anchor + (
                self.data.site_xpos[self.grip_site_id]
                - self.pivot_anchor_gripper_pos
            )
        position_error = (
            moving_anchor - self.data.xpos[self.target_body_id]
        )
        force = (
            PIVOT_POSITION_STIFFNESS * position_error
            - PIVOT_POSITION_DAMPING * linear_velocity
            - self.model.body_mass[self.target_body_id] * self.model.opt.gravity
        )

        rotation = self.data.xmat[self.target_body_id].reshape(3, 3)
        orientation_error = _rotation_vector(
            self.pivot_anchor_rotation @ rotation.T
        )
        gains = PIVOT_ORIENTATION_GAINS[self.target_object]
        if self.target_object == "coffee_jar":
            # Its inertia is sufficiently balanced for one isotropic spring;
            # one shared limit also avoids torque components adding at the rim.
            torque = self._limited(
                gains[3] * orientation_error - gains[4] * angular_velocity,
                gains[5],
            )
        else:
            (
                tilt_stiffness,
                tilt_damping,
                max_tilt_torque,
                yaw_stiffness,
                yaw_damping,
                max_yaw_torque,
            ) = gains
            upright_axis = self.pivot_anchor_rotation[:, 2]
            yaw_error = upright_axis * float(orientation_error @ upright_axis)
            tilt_error = orientation_error - yaw_error
            yaw_velocity = upright_axis * float(angular_velocity @ upright_axis)
            tilt_velocity = angular_velocity - yaw_velocity
            tilt_torque = self._limited(
                tilt_stiffness * tilt_error - tilt_damping * tilt_velocity,
                max_tilt_torque,
            )
            yaw_torque = self._limited(
                yaw_stiffness * yaw_error - yaw_damping * yaw_velocity,
                max_yaw_torque,
            )
            torque = tilt_torque + yaw_torque
        self.data.xfrc_applied[self.target_body_id, :3] = self._limited(
            force, PIVOT_MAX_FORCE[self.target_object]
        )
        self.data.xfrc_applied[self.target_body_id, 3:] = torque

    def _finish_compliant_pivot(self) -> None:
        """Capture the live horizontal grasp after the compliant slip."""
        self.data.xfrc_applied[self.target_body_id] = 0.0
        self._set_grasp_weld_world_pose(
            self.data.xpos[self.target_body_id],
            self.data.xquat[self.target_body_id],
        )
        self.data.eq_active[self.grasp_equality_id] = 1
        self.pivot_anchor_pos = None
        self.pivot_anchor_rotation = None
        self.pivot_anchor_gripper_pos = None
        self.pivot_anchor_follows_gripper = False
        self.pivot_active = False

    def _fail(self, message: str) -> None:
        self.mode = "failed"
        self.failure = message
        self.status = f"Pick failed: {message}"

    def update(self) -> None:
        if self.mode in {"idle", "holding", "failed"}:
            return
        if self.mode == "approach":
            # Only the final contact pose settles tightly. Intermediate IK
            # knots remain a single uninterrupted trajectory.
            finished, waypoint = self._advance_trajectory(
                final_tolerance=0.001 if self.target_object == "spoon" else 0.003
            )
            self.data.ctrl[self.finger_actuators] = OPEN_WIDTH
            finger_error = float(
                np.max(np.abs(self.data.qpos[self.finger_qpos] - OPEN_WIDTH))
            )
            self.status = f"Pick {self.target_object}: {waypoint.label}"
            if finished and finger_error < 0.008:
                self.mode = "closing"
                self.status = f"Pick {self.target_object}: closing until contact"
            return

        if self.mode == "closing":
            self.close_ticks += 1
            if self.close_ticks % 5 == 0:
                self.close_target = max(0.0, self.close_target - 0.001)
            self.data.ctrl[self.finger_actuators] = self.close_target
            contacts = self._target_finger_contacts()
            if contacts == FINGER_GEOMS:
                self.contact_ticks += 1
                # Add a small squeeze before fixing the contact-confirmed grasp.
                self.data.ctrl[self.finger_actuators] = max(
                    0.0, self.close_target - 0.002
                )
                if self.contact_ticks >= 12:
                    try:
                        self._activate_grasp_weld()
                    except RuntimeError as error:
                        self._fail(str(error))
                        return
                    self._start_trajectory(self.post_grasp_waypoints)
                    self.mode = "returning"
                    self.status = f"Pick {self.target_object}: contact confirmed, lifting"
                return
            self.contact_ticks = 0
            if self.close_target <= 0.0 and self.close_ticks > 300:
                self._fail("gripper closed without bilateral object contact")
            return

        if self.mode == "returning":
            return_tolerance = (
                0.080
                if self.target_object in {"coffee_jar", "sugar_jar"}
                and self.physical_location != "home"
                else 0.003
                if self.target_object == "spoon"
                else 0.012
            )
            finished, waypoint = self._advance_trajectory(
                final_tolerance=return_tolerance
            )
            if waypoint.passive_pivot and self.spoon_pivot_equality_id < 0:
                try:
                    self._activate_spoon_pivot()
                except RuntimeError as error:
                    self._fail(str(error))
                    return
            if self.spoon_pivot_equality_id >= 0:
                # Blend the pad opening while stationary at high hover. This
                # preserves visible handle contact as the weld becomes a live
                # point pivot instead of abruptly loosening the fingers.
                self.spoon_pivot_ticks += 1
                phase = min(
                    1.0,
                    self.spoon_pivot_ticks / SPOON_PIVOT_BLEND_TICKS,
                )
                pivot_width = (
                    (1.0 - phase) * self.spoon_pivot_start_width
                    + phase * SPOON_PIVOT_WIDTH
                )
                self.data.ctrl[self.finger_actuators] = pivot_width
                spoon_angle, spoon_speed = self._damp_spoon_pivot()
            if waypoint.stabilize_object:
                if not self.pivot_active:
                    self._begin_compliant_pivot()
                if (
                    "returning horizontally" in waypoint.label.lower()
                    and not self.pivot_anchor_follows_gripper
                ):
                    self._begin_compliant_translation()
                self._apply_compliant_pivot()
            elif self.pivot_active:
                self._finish_compliant_pivot()
            self.status = f"Pick {self.target_object}: {waypoint.label}"
            if finished:
                if self.spoon_pivot_equality_id >= 0:
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
                if self.pivot_active:
                    self._finish_compliant_pivot()
                if self.spoon_pivot_equality_id >= 0:
                    self._finish_spoon_pivot()
                self.held_object = self.target_object
                self.mode = "holding"
                self.status = f"Pick complete: holding {self.held_object} in carry pose"

    def progress(self) -> float:
        if self.mode == "closing":
            return 0.55
        if self.mode == "returning":
            ratio = self.trajectory_time / max(1e-9, self.trajectory_times[-1])
            return 0.60 + 0.40 * ratio
        if self.mode == "holding":
            return 1.0
        if self.mode == "approach":
            ratio = self.trajectory_time / max(1e-9, self.trajectory_times[-1])
            return 0.55 * ratio
        return 0.0
