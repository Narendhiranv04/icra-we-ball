"""Verified rigid-duster sweep for the living-room TV.

The screen is covered by three horizontal passes.  The arm supplies the row
height and tool orientation while the holonomic base supplies the horizontal
motion.  A cell is only marked clean after the live duster head reaches its
physical screen-facing pose; advancing the state machine alone is not enough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from mujoco_scenes.generic_manipulation import (
    ProfiledIK,
    RobotConfigurationCollisionChecker,
)
from mujoco_scenes.mobile_motion import MuJoCoBaseCollisionChecker
from mujoco_scenes.living_room_scene import LIVING_ROOM_FORWARD_LIMITS
from mujoco_scenes.robot_profiles import manipulation_profile, mobile_profile


DUST_ROWS = (
    (0, 1, 2, 3, 4),
    (9, 8, 7, 6, 5),
    (10, 11, 12, 13, 14),
)

# These are deterministic IK branch seeds, not commanded poses.  They select
# collision-free branches found during calibration; IK still solves against
# the live weld transform and cell positions on every request.
DUST_ROW_IK_SEEDS = (
    np.array((1.25475, 1.51759, -0.20447, 3.77500, -0.25376, 1.00590, -4.24833)),
    np.array((0.39792, 2.59112, -0.86439, 3.74523, -2.02756, 0.47203, -2.24956)),
    np.array((-0.07182, -0.14225, 0.92715, 1.77975, 0.10935, 1.44039, 0.83593)),
)

# The duster is used end-on: its rigid head stays between the robot and TV,
# keeping the fingers/forearm well clear of the screen.  The bottom row uses
# the equivalent 180-degree roll to avoid the shoulder/base collision branch.
DUST_TOOL_ROTATIONS = (
    np.array(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))),
    np.array(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))),
    np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
)

ARM_COMMAND_SPEED = 0.45
ARM_COMMAND_TOLERANCE = 0.001
ARM_TRACKING_TOLERANCE = 0.035
ARM_INTERMEDIATE_TOLERANCE = 0.035
ARM_HOLD_TICKS = 5
ARM_JOINT_RESOLUTION = 0.030
ARM_SEGMENT_RESOLUTION = 0.020
# The Google base occupies the space directly behind an end-on tool pose, so
# a conventional straight normal retract would pull the handle into the body.
# Clearance is instead supplied by the full held-tool joint-space detour.
HEAD_APPROACH_Y_OFFSET = 0.0
HEAD_SCREEN_GAP = 0.015
HEAD_POSITION_TOLERANCE = 0.030
HEAD_ORIENTATION_TOLERANCE = math.radians(6.0)
HEAD_GAP_RANGE = (-0.0015, 0.026)
CELL_CONFIRM_TICKS = 8
BASE_COMMAND_SPEED = 0.075
BASE_COMMAND_LEAD = 0.008
BASE_POSITION_TOLERANCE = 0.003
BASE_COMMAND_TOLERANCE = 0.002
BASE_SETTLE_SPEED = 0.008
BASE_SETTLE_TICKS = 25
CARRY_SETTLE_TICKS = 100
BASE_LOCK_TOLERANCE = 0.006
BASE_YAW_LOCK_TOLERANCE = math.radians(0.6)
ACTION_TIMEOUT_TICKS = 120_000
COLLISION_GUARD_INTERVAL = 5
TOOL_COLLISION_TOLERANCE = 0.002
TOOL_ENVIRONMENT_CLEARANCE = 0.020
SCREEN_PENETRATION_LIMIT = 0.0015


def _rotation_error(target: np.ndarray, current: np.ndarray) -> float:
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


@dataclass
class _ArmWaypoint:
    joints: np.ndarray
    label: str


class _HeldDusterCollisionChecker:
    """Check the actual rigid tool together with candidate arm poses."""

    def __init__(self, scene, gripper_body_id: int, tool_body_id: int):
        self.scene = scene
        self.model = scene.model
        self.reference = scene.data
        self.data = mujoco.MjData(self.model)
        self.profile = manipulation_profile("google")
        self.gripper_body_id = gripper_body_id
        self.tool_body_id = tool_body_id
        self.arm_joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                for name in self.profile.arm_joints
            ]
        )
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joint_ids]
        tool_joint = int(self.model.body_jntadr[tool_body_id])
        self.tool_qpos = int(self.model.jnt_qposadr[tool_joint])

        gripper_rotation = self.reference.xmat[gripper_body_id].reshape(3, 3)
        tool_rotation = self.reference.xmat[tool_body_id].reshape(3, 3)
        self.tool_position_in_gripper = gripper_rotation.T @ (
            self.reference.xpos[tool_body_id]
            - self.reference.xpos[gripper_body_id]
        )
        self.tool_rotation_in_gripper = gripper_rotation.T @ tool_rotation

        self.tool_geoms = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) == tool_body_id
            and (
                self.model.geom_contype[geom_id]
                or self.model.geom_conaffinity[geom_id]
            )
        ]
        self.robot_geoms: list[int] = []
        self.environment_geoms: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id == tool_body_id:
                continue
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            geom_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or ""
            if body_name.startswith("google:"):
                if not any(
                    token in body_name
                    for token in ("gripper", "finger")
                ):
                    self.robot_geoms.append(geom_id)
            elif (
                geom_name != "floor"
                and (
                    self.model.geom_contype[geom_id]
                    or self.model.geom_conaffinity[geom_id]
                )
            ):
                self.environment_geoms.append(geom_id)

        self.head_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "rigid_duster_head_collision",
        )
        self.screen_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tv_screen_collision"
        )

    def _position_tool(self, arm_joints: np.ndarray) -> None:
        self.data.qpos[:] = self.reference.qpos
        self.data.qpos[self.arm_qpos] = arm_joints
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        gripper_rotation = self.data.xmat[self.gripper_body_id].reshape(3, 3)
        tool_rotation = gripper_rotation @ self.tool_rotation_in_gripper
        tool_position = self.data.xpos[self.gripper_body_id] + gripper_rotation @ (
            self.tool_position_in_gripper
        )
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, tool_rotation.ravel())
        self.data.qpos[self.tool_qpos : self.tool_qpos + 3] = tool_position
        self.data.qpos[self.tool_qpos + 3 : self.tool_qpos + 7] = quaternion
        mujoco.mj_forward(self.model, self.data)

    def evaluate(self, arm_joints: np.ndarray) -> tuple[bool, str | None]:
        self._position_tool(arm_joints)
        for tool_geom in self.tool_geoms:
            tool_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, tool_geom
            ) or "rigid duster"
            for robot_geom in self.robot_geoms:
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    tool_geom,
                    robot_geom,
                    TOOL_COLLISION_TOLERANCE,
                    None,
                )
                if distance < -TOOL_COLLISION_TOLERANCE:
                    robot_body = int(self.model.geom_bodyid[robot_geom])
                    robot_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body
                    ) or "robot"
                    return False, f"{tool_name} clips {robot_name}"

            for environment_geom in self.environment_geoms:
                distance = mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    tool_geom,
                    environment_geom,
                    TOOL_ENVIRONMENT_CLEARANCE,
                    None,
                )
                if (
                    tool_geom == self.head_geom_id
                    and environment_geom == self.screen_geom_id
                ):
                    if distance < -SCREEN_PENETRATION_LIMIT:
                        return False, "duster head penetrates the TV screen"
                    continue
                if distance < TOOL_ENVIRONMENT_CLEARANCE:
                    environment_name = mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        environment_geom,
                    ) or "living-room geometry"
                    return False, f"{tool_name} is too close to {environment_name}"
        return True, None

    def segment_valid(
        self, start: np.ndarray, goal: np.ndarray
    ) -> tuple[bool, str | None]:
        count = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(goal - start)))
                    / ARM_SEGMENT_RESOLUTION
                )
            ),
        )
        for fraction in np.linspace(0.0, 1.0, count + 1):
            valid, reason = self.evaluate(start + fraction * (goal - start))
            if not valid:
                return False, f"{reason} at {fraction * 100:.0f}% of arm segment"
        return True, None


class TVDustExecutor:
    """Dust all 15 TV cells with collision-checked arm/base motion."""

    def __init__(self, scene, manipulation):
        if scene.robot_name != "google":
            raise ValueError("TV dusting currently requires Google Robot")
        self.scene = scene
        self.manipulation = manipulation
        self.model = scene.model
        self.data = scene.data
        self.profile = manipulation_profile("google")
        self.mobile_profile = replace(
            mobile_profile("google"),
            forward_limits=LIVING_ROOM_FORWARD_LIMITS,
        )
        self.arm_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.profile.arm_joints
        )
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.arm_actuators
        )
        self.base_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.mobile_profile.base_joints
        )
        self.base_qpos = self.model.jnt_qposadr[self.base_joint_ids]
        self.base_dofs = self.model.jnt_dofadr[self.base_joint_ids]
        self.base_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.mobile_profile.base_actuators
        )
        self.gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.profile.gripper_body
        )
        self.grip_site_id = scene.site_id(self.profile.grip_site)
        self.tool_body_id = scene.body_id("rigid_duster")
        self.head_site_id = scene.site_id("rigid_duster_head")
        self.head_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "rigid_duster_head_collision",
        )
        self.screen_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "tv_screen_collision"
        )
        self.grasp_equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "google:pick_weld_rigid_duster",
        )
        self.mode = "idle"
        self.status = "TV dusting idle"
        self.failure: str | None = None
        self.elapsed_ticks = 0
        self.row_index = 0
        self.cell_cursor = 0
        self.cell_ticks = 0
        self.base_ticks = 0
        self.waypoints: list[_ArmWaypoint] = []
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.retreat_waypoints: list[_ArmWaypoint] = []
        self.carry_joints = np.zeros(len(self.arm_qpos))
        self.row_contact_joints = np.zeros(len(self.arm_qpos))
        self.grip_to_tool_rotation = np.eye(3)
        self.head_offset_in_grip = np.zeros(3)
        self.desired_tool_rotation = np.eye(3)
        self.desired_grip_rotation = np.eye(3)
        self.tv_forward_qpos = 0.0
        self.tv_yaw_qpos = 0.0
        self.tool_checker: _HeldDusterCollisionChecker | None = None

    def _ids(self, object_type, names) -> np.ndarray:
        values = np.array(
            [mujoco.mj_name2id(self.model, object_type, name) for name in names]
        )
        if np.any(values < 0):
            raise RuntimeError("TV dust controller model interface is incomplete")
        return values

    @property
    def busy(self) -> bool:
        return self.mode not in {"idle", "complete", "failed"}

    @property
    def navigation_safe(self) -> bool:
        return self.mode in {"idle", "complete"}

    def _world_x(self) -> float:
        return -float(self.data.qpos[self.base_qpos[1]])

    def _world_y(self) -> float:
        return -1.25 + float(self.data.qpos[self.base_qpos[0]])

    def _joint_points(
        self, start: np.ndarray, goal: np.ndarray, label: str
    ) -> list[_ArmWaypoint]:
        count = max(
            1,
            int(
                math.ceil(
                    float(np.max(np.abs(goal - start)))
                    / ARM_JOINT_RESOLUTION
                )
            ),
        )
        return [
            _ArmWaypoint(start + fraction * (goal - start), label)
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]
        ]

    def _screen_contact_y(self) -> float:
        rotation = self.data.geom_xmat[self.screen_geom_id].reshape(3, 3)
        extent = float(
            np.sum(np.abs(rotation[1]) * self.model.geom_size[self.screen_geom_id])
        )
        screen_south_face = float(
            self.data.geom_xpos[self.screen_geom_id, 1] - extent
        )
        return screen_south_face - float(
            self.model.geom_size[self.head_geom_id, 0]
        ) - HEAD_SCREEN_GAP

    def _cell_head_target(self, cell: int, *, preapproach: bool) -> np.ndarray:
        target = self.data.site_xpos[self.scene.site_id(f"tv_cell_{cell}")].copy()
        target[1] = self._screen_contact_y()
        if preapproach:
            target[1] -= HEAD_APPROACH_Y_OFFSET
        return target

    def _grip_target(self, head_target: np.ndarray) -> np.ndarray:
        return head_target - self.desired_grip_rotation @ self.head_offset_in_grip

    def _check_arm_segment(
        self,
        checker: RobotConfigurationCollisionChecker,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> None:
        allowed = frozenset((self.tool_body_id,))
        valid, reason = checker.segment_valid(
            start, goal, allowed, resolution=ARM_SEGMENT_RESOLUTION
        )
        if not valid:
            raise RuntimeError(reason or "robot arm collision")
        assert self.tool_checker is not None
        valid, reason = self.tool_checker.segment_valid(start, goal)
        if not valid:
            raise RuntimeError(reason or "held duster collision")

    @staticmethod
    def _interpolate_rotation(
        start: np.ndarray, goal: np.ndarray, fraction: float
    ) -> np.ndarray:
        start_quat = np.empty(4)
        goal_quat = np.empty(4)
        mujoco.mju_mat2Quat(start_quat, start.ravel())
        mujoco.mju_mat2Quat(goal_quat, goal.ravel())
        dot = float(np.dot(start_quat, goal_quat))
        if dot < 0.0:
            goal_quat = -goal_quat
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            quaternion = start_quat + fraction * (goal_quat - start_quat)
            quaternion /= np.linalg.norm(quaternion)
        else:
            angle = math.acos(dot)
            scale = math.sin(angle)
            quaternion = (
                math.sin((1.0 - fraction) * angle) / scale * start_quat
                + math.sin(fraction * angle) / scale * goal_quat
            )
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        return matrix.reshape(3, 3)

    def _solve_pose_segment(
        self,
        ik: ProfiledIK,
        checker: RobotConfigurationCollisionChecker,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        start_rotation: np.ndarray,
        goal_rotation: np.ndarray,
        seed: np.ndarray,
        label: str,
    ) -> tuple[list[_ArmWaypoint], np.ndarray]:
        position_count = int(
            math.ceil(
                float(np.linalg.norm(goal_position - start_position)) / 0.025
            )
        )
        angle_count = int(
            math.ceil(
                _rotation_error(goal_rotation, start_rotation)
                / math.radians(2.0)
            )
        )
        count = max(1, position_count, angle_count)
        result: list[_ArmWaypoint] = []
        previous = seed
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            position = start_position + fraction * (
                goal_position - start_position
            )
            rotation = self._interpolate_rotation(
                start_rotation, goal_rotation, float(fraction)
            )
            joints, position_error, angle_error = ik.solve(
                position, previous, rotation
            )
            if position_error > 0.025 or angle_error > math.radians(6.0):
                raise RuntimeError(
                    f"{label} IK miss at {fraction * 100:.0f}%: "
                    f"{position_error * 100:.1f} cm / "
                    f"{math.degrees(angle_error):.1f} deg"
                )
            self._check_arm_segment(checker, previous, joints)
            result.append(_ArmWaypoint(joints.copy(), label))
            previous = joints
        return result, previous

    def _joint_path(
        self,
        checker: RobotConfigurationCollisionChecker,
        ik: ProfiledIK,
        start: np.ndarray,
        goal: np.ndarray,
        label: str,
    ) -> list[_ArmWaypoint]:
        """Plan a small deterministic 7-D detour around tool/self clipping."""

        def edge_valid(first: np.ndarray, second: np.ndarray) -> bool:
            try:
                self._check_arm_segment(checker, first, second)
            except RuntimeError:
                return False
            return True

        if edge_valid(start, goal):
            return self._joint_points(start, goal, label)

        def steer(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            delta = second - first
            magnitude = float(np.max(np.abs(delta)))
            if magnitude <= 0.16:
                return second.copy()
            return first + delta * (0.16 / magnitude)

        def nearest(nodes: list[np.ndarray], target: np.ndarray) -> int:
            scale = np.maximum(ik.upper - ik.lower, 0.5)
            distances = [
                float(np.linalg.norm((node - target) / scale)) for node in nodes
            ]
            return int(np.argmin(distances))

        def root_path(
            nodes: list[np.ndarray], parents: list[int], index: int
        ) -> list[np.ndarray]:
            path: list[np.ndarray] = []
            while index >= 0:
                path.append(nodes[index])
                index = parents[index]
            path.reverse()
            return path

        start_nodes = [start.copy()]
        start_parents = [-1]
        goal_nodes = [goal.copy()]
        goal_parents = [-1]
        rng = np.random.default_rng(1907 + self.row_index)
        for iteration in range(2400):
            active_is_start = iteration % 2 == 0
            if active_is_start:
                active_nodes, active_parents = start_nodes, start_parents
                other_nodes, other_parents = goal_nodes, goal_parents
            else:
                active_nodes, active_parents = goal_nodes, goal_parents
                other_nodes, other_parents = start_nodes, start_parents

            draw = rng.random()
            if draw < 0.72:
                fraction = rng.random()
                sample = start + fraction * (goal - start)
                sample += rng.normal(0.0, 0.42, len(start))
            elif draw < 0.90:
                sample = other_nodes[-1].copy()
            else:
                sample = rng.uniform(ik.lower, ik.upper)
            sample = np.clip(sample, ik.lower, ik.upper)

            active_nearest = nearest(active_nodes, sample)
            new_active = steer(active_nodes[active_nearest], sample)
            if not edge_valid(active_nodes[active_nearest], new_active):
                continue
            active_nodes.append(new_active)
            active_parents.append(active_nearest)
            active_index = len(active_nodes) - 1

            target = new_active
            connected_index: int | None = None
            for _ in range(40):
                other_nearest = nearest(other_nodes, target)
                new_other = steer(other_nodes[other_nearest], target)
                if not edge_valid(other_nodes[other_nearest], new_other):
                    break
                other_nodes.append(new_other)
                other_parents.append(other_nearest)
                connected_index = len(other_nodes) - 1
                if float(np.max(np.abs(new_other - target))) < 1e-8:
                    break
            if connected_index is None or float(
                np.max(np.abs(other_nodes[connected_index] - target))
            ) >= 1e-8:
                continue

            active_path = root_path(active_nodes, active_parents, active_index)
            other_path = root_path(other_nodes, other_parents, connected_index)
            if active_is_start:
                coarse = active_path + list(reversed(other_path[:-1]))
            else:
                coarse = other_path + list(reversed(active_path[:-1]))
            result: list[_ArmWaypoint] = []
            for first, second in zip(coarse, coarse[1:]):
                result.extend(self._joint_points(first, second, label))
            return result
        raise RuntimeError("could not find a held-duster arm detour")

    def _build_row_plan(self) -> None:
        first_cell = DUST_ROWS[self.row_index][0]
        self.desired_tool_rotation = DUST_TOOL_ROTATIONS[self.row_index]
        self.desired_grip_rotation = (
            self.desired_tool_rotation @ self.grip_to_tool_rotation.T
        )
        ik = ProfiledIK(self.model, self.data, self.profile)
        checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile
        )
        self.tool_checker = _HeldDusterCollisionChecker(
            self.scene, self.gripper_body_id, self.tool_body_id
        )
        current = self.data.qpos[self.arm_qpos].copy()
        contact_head = self._cell_head_target(first_cell, preapproach=False)
        contact, position_error, angle_error = ik.solve(
            self._grip_target(contact_head),
            DUST_ROW_IK_SEEDS[self.row_index],
            self.desired_grip_rotation,
        )
        if position_error > 0.012 or angle_error > math.radians(2.0):
            raise RuntimeError(
                "screen-row IK miss "
                f"{position_error * 100:.1f} cm / "
                f"{math.degrees(angle_error):.1f} deg"
            )
        pre_head = self._cell_head_target(first_cell, preapproach=True)
        count = max(1, int(math.ceil(HEAD_APPROACH_Y_OFFSET / 0.020)))
        outward: list[np.ndarray] = []
        previous = contact
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            head = contact_head + fraction * (pre_head - contact_head)
            joints, position_error, angle_error = ik.solve(
                self._grip_target(head), previous, self.desired_grip_rotation
            )
            if position_error > 0.012 or angle_error > math.radians(2.0):
                raise RuntimeError("TV preapproach leaves the calibrated IK branch")
            self._check_arm_segment(checker, previous, joints)
            outward.append(joints.copy())
            previous = joints
        precontact = previous
        approach_joints = [
            *[item.copy() for item in reversed(outward[:-1])],
            contact.copy(),
        ]
        approach = [
            _ArmWaypoint(item.copy(), "Approaching TV screen")
            for item in approach_joints
        ]
        staged = self._joint_path(
            checker,
            ik,
            current,
            precontact,
            "Orienting rigid duster through collision-free detour",
        )
        self.waypoints = [*staged, *approach]
        # Reversing the already checked path keeps the rigid tool in the same
        # clearance corridor on the way back to carry.
        self.retreat_waypoints = [
            *[
                _ArmWaypoint(item.joints.copy(), "Retracting from TV screen")
                for item in reversed(self.waypoints[:-1])
            ],
            _ArmWaypoint(
                self.carry_joints.copy(), "Returning duster to compact carry"
            ),
        ]
        self.row_contact_joints = contact.copy()
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.status = f"Dust row {self.row_index + 1}/3: orienting tool"

    def _build_retreat_plan(self) -> None:
        """Replan at the row's end side against that side's live furniture."""
        checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile
        )
        self.tool_checker = _HeldDusterCollisionChecker(
            self.scene, self.gripper_body_id, self.tool_body_id
        )
        ik = ProfiledIK(self.model, self.data, self.profile)
        current = self.data.qpos[self.arm_qpos].copy()
        self.waypoints = self._joint_path(
            checker,
            ik,
            current,
            self.carry_joints,
            "Returning duster through collision-free carry detour",
        )
        self.waypoint_index = 0
        self.waypoint_ticks = 0

    def _validate_base_path(self, target_x: float) -> None:
        checker = MuJoCoBaseCollisionChecker(
            self.model, self.data, self.mobile_profile
        )
        start_x = self._world_x()
        count = max(1, int(math.ceil(abs(target_x - start_x) / 0.025)))
        for fraction in np.linspace(0.0, 1.0, count + 1):
            x = start_x + fraction * (target_x - start_x)
            valid = checker.is_pose_valid(
                x, self._world_y(), self.tv_yaw_qpos
            )
            if not valid and self._base_contacts_are_screen_only(checker):
                valid = True
            if not valid:
                raise RuntimeError(
                    f"held-duster base path is blocked near x={x:.2f} m"
                )

    def _base_contacts_are_screen_only(
        self, checker: MuJoCoBaseCollisionChecker
    ) -> bool:
        """Accept only the intended head/screen pair in base-pose checks."""
        saw_screen = False
        for contact in checker.data.contact:
            first_robot = checker._geom_is_robot(int(contact.geom1))
            second_robot = checker._geom_is_robot(int(contact.geom2))
            if first_robot == second_robot:
                continue
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {self.head_geom_id, self.screen_geom_id}:
                if float(contact.dist) < -SCREEN_PENETRATION_LIMIT:
                    return False
                saw_screen = True
                continue
            other = int(contact.geom2) if first_robot else int(contact.geom1)
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other
            ) or ""
            if other_name != "floor":
                return False
        return saw_screen

    def request_dust(
        self,
        current_location: str,
        held_object: str | None = None,
    ) -> None:
        if self.busy:
            raise RuntimeError("TV dusting is already running")
        actual_held = self.manipulation.held_object
        if held_object is not None and held_object != actual_held:
            raise RuntimeError("Actions-panel held-object state is stale")
        if actual_held != "rigid_duster":
            raise RuntimeError("Pick the rigid duster before dusting")
        if current_location != "tv":
            raise RuntimeError("Move to TV before dusting")
        if not self.data.eq_active[self.grasp_equality_id]:
            raise RuntimeError("Rigid duster transport weld is not active")
        if not self.manipulation.navigation_safe:
            raise RuntimeError("Duster must be in compact carry before dusting")

        mujoco.mj_forward(self.model, self.data)
        grip_rotation = self.data.site_xmat[self.grip_site_id].reshape(3, 3)
        tool_rotation = self.data.site_xmat[self.head_site_id].reshape(3, 3)
        self.grip_to_tool_rotation = grip_rotation.T @ tool_rotation
        self.head_offset_in_grip = grip_rotation.T @ (
            self.data.site_xpos[self.head_site_id]
            - self.data.site_xpos[self.grip_site_id]
        )
        # Give the long handle a little more body clearance than the transport
        # carry used for navigation.  This is solved from the live pose (not a
        # hard-coded joint offset) and is used between sweep rows.
        carry_ik = ProfiledIK(self.model, self.data, self.profile)
        carry_target = self.data.site_xpos[self.grip_site_id].copy()
        carry_target[1] += 0.07
        self.carry_joints, position_error, angle_error = carry_ik.solve(
            carry_target,
            self.data.qpos[self.arm_qpos].copy(),
            grip_rotation,
        )
        if position_error > 0.012 or angle_error > math.radians(2.0):
            raise RuntimeError("Could not calibrate body-clear duster carry pose")
        carry_robot_checker = RobotConfigurationCollisionChecker(
            self.model, self.data, self.profile
        )
        valid, reason = carry_robot_checker.evaluate(
            self.carry_joints, frozenset((self.tool_body_id,))
        )
        if not valid:
            raise RuntimeError(f"Unsafe duster carry pose: {reason}")
        carry_tool_checker = _HeldDusterCollisionChecker(
            self.scene, self.gripper_body_id, self.tool_body_id
        )
        valid, reason = carry_tool_checker.evaluate(self.carry_joints)
        if not valid:
            raise RuntimeError(f"Unsafe duster carry pose: {reason}")
        self.tv_forward_qpos = float(self.data.qpos[self.base_qpos[0]])
        self.tv_yaw_qpos = float(self.data.qpos[self.base_qpos[2]])
        if abs(self.tv_yaw_qpos) > math.radians(0.8):
            raise RuntimeError("TV dusting requires the settled forward-facing TV pose")

        self.row_index = 0
        self.cell_cursor = 0
        self.cell_ticks = 0
        self.base_ticks = 0
        self.elapsed_ticks = 0
        self.failure = None
        first_x = float(
            self.data.site_xpos[
                self.scene.site_id(f"tv_cell_{DUST_ROWS[0][0]}")
            ][0]
        )
        self._validate_base_path(first_x)
        self.mode = "base_to_row_start"
        self.status = "TV dust: moving to the first safe sweep position"

    # Compatibility with the Actions panel's flexible request adapter.
    request = request_dust

    def _advance_arm(self) -> bool:
        if self.waypoint_index >= len(self.waypoints):
            return True
        waypoint = self.waypoints[self.waypoint_index]
        command = self.data.ctrl[self.arm_actuators]
        maximum = ARM_COMMAND_SPEED * self.model.opt.timestep
        next_command = command + np.clip(
            waypoint.joints - command, -maximum, maximum
        )
        self.data.ctrl[self.arm_actuators] = next_command
        command_error = float(np.max(np.abs(waypoint.joints - next_command)))
        tracking_error = float(
            np.max(np.abs(self.data.qpos[self.arm_qpos] - waypoint.joints))
        )
        final = self.waypoint_index == len(self.waypoints) - 1
        tolerance = ARM_TRACKING_TOLERANCE if final else ARM_INTERMEDIATE_TOLERANCE
        self.status = waypoint.label
        if command_error < ARM_COMMAND_TOLERANCE and tracking_error < tolerance:
            self.waypoint_ticks += 1
        else:
            self.waypoint_ticks = 0
        if self.waypoint_ticks >= ARM_HOLD_TICKS:
            self.waypoint_index += 1
            self.waypoint_ticks = 0
        return self.waypoint_index >= len(self.waypoints)

    def _command_base_x(self, target_x: float) -> bool:
        target = np.array(
            (self.tv_forward_qpos, -target_x, self.tv_yaw_qpos), dtype=float
        )
        qpos = self.data.qpos[self.base_qpos]
        command = self.data.ctrl[self.base_actuators]
        maximum = self.model.opt.timestep * np.array(
            (BASE_COMMAND_SPEED, BASE_COMMAND_SPEED, 0.35)
        )
        next_command = command + np.clip(target - command, -maximum, maximum)
        next_command[:2] = np.clip(
            next_command[:2],
            qpos[:2] - BASE_COMMAND_LEAD,
            qpos[:2] + BASE_COMMAND_LEAD,
        )
        self.data.ctrl[self.base_actuators] = next_command
        position_error = float(np.max(np.abs(qpos - target)))
        command_error = float(np.max(np.abs(next_command - target)))
        speed = float(np.max(np.abs(self.data.qvel[self.base_dofs])))
        if (
            position_error < BASE_POSITION_TOLERANCE
            and command_error < BASE_COMMAND_TOLERANCE
            and speed < BASE_SETTLE_SPEED
        ):
            self.base_ticks += 1
        else:
            self.base_ticks = 0
        return self.base_ticks >= BASE_SETTLE_TICKS

    def _head_gap(self) -> float:
        head_rotation = self.data.geom_xmat[self.head_geom_id].reshape(3, 3)
        head_extent = float(
            np.sum(np.abs(head_rotation[1]) * self.model.geom_size[self.head_geom_id])
        )
        screen_rotation = self.data.geom_xmat[self.screen_geom_id].reshape(3, 3)
        screen_extent = float(
            np.sum(
                np.abs(screen_rotation[1])
                * self.model.geom_size[self.screen_geom_id]
            )
        )
        screen_south = float(
            self.data.geom_xpos[self.screen_geom_id, 1] - screen_extent
        )
        head_north = float(self.data.geom_xpos[self.head_geom_id, 1] + head_extent)
        return screen_south - head_north

    def _cell_reached(self, cell: int) -> tuple[bool, str]:
        target = self._cell_head_target(cell, preapproach=False)
        head = self.data.site_xpos[self.head_site_id]
        position_error = float(np.linalg.norm(head - target))
        rotation = self.data.site_xmat[self.head_site_id].reshape(3, 3)
        angle_error = _rotation_error(self.desired_tool_rotation, rotation)
        gap = self._head_gap()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.tool_body_id,
            velocity,
            0,
        )
        speed = float(np.linalg.norm(velocity[3:]))
        reached = (
            position_error <= HEAD_POSITION_TOLERANCE
            and angle_error <= HEAD_ORIENTATION_TOLERANCE
            and HEAD_GAP_RANGE[0] <= gap <= HEAD_GAP_RANGE[1]
            and speed <= 0.10
        )
        detail = (
            f"head error={position_error * 100:.1f} cm, "
            f"angle={math.degrees(angle_error):.1f} deg, "
            f"gap={gap * 1000:.1f} mm"
        )
        return reached, detail

    def _forbidden_contact(self) -> str | None:
        for contact in self.data.contact:
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            first_robot = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body1
                )
                or ""
            ).startswith("google:")
            second_robot = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body2
                )
                or ""
            ).startswith("google:")
            first_tool = body1 == self.tool_body_id
            second_tool = body2 == self.tool_body_id
            if (first_robot and second_tool) or (second_robot and first_tool):
                continue
            if first_robot != second_robot and not (first_tool or second_tool):
                robot = geom1 if first_robot else geom2
                robot_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, robot
                ) or (
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(self.model.geom_bodyid[robot]),
                    )
                    or "robot"
                )
                other = geom2 if first_robot else geom1
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                ) or "living-room geometry"
                if other_name != "floor":
                    return f"{robot_name} contact with {other_name}"
            if first_tool != second_tool:
                other = geom2 if first_tool else geom1
                tool = geom1 if first_tool else geom2
                if {tool, other} == {self.head_geom_id, self.screen_geom_id}:
                    if float(contact.dist) < -SCREEN_PENETRATION_LIMIT:
                        return "duster head penetrated TV screen"
                    continue
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other
                ) or "living-room geometry"
                return f"duster contact with {other_name}"
        return None

    def _fail(self, message: str) -> None:
        self.data.ctrl[self.base_actuators] = self.data.qpos[self.base_qpos]
        self.data.ctrl[self.arm_actuators] = self.data.qpos[self.arm_qpos]
        self.failure = message
        self.mode = "failed"
        self.status = f"TV dust failed: {message}"

    def update(self) -> None:
        self.scene.update_visual_effects()
        if not self.busy:
            return
        self.elapsed_ticks += 1
        if self.elapsed_ticks >= ACTION_TIMEOUT_TICKS:
            self._fail(f"timeout while {self.status}")
            return
        if not self.data.eq_active[self.grasp_equality_id]:
            self._fail("duster transport weld became inactive")
            return
        if self.elapsed_ticks % COLLISION_GUARD_INTERVAL == 0:
            forbidden = self._forbidden_contact()
            if forbidden:
                self._fail(forbidden)
                return
            if (
                abs(float(self.data.qpos[self.base_qpos[0]]) - self.tv_forward_qpos)
                > BASE_LOCK_TOLERANCE
                or abs(float(self.data.qpos[self.base_qpos[2]]) - self.tv_yaw_qpos)
                > BASE_YAW_LOCK_TOLERANCE
            ):
                self._fail("base drifted away from the locked TV sweep line")
                return

        if self.mode == "base_to_row_start":
            first_cell = DUST_ROWS[self.row_index][0]
            target_x = float(
                self.data.site_xpos[
                    self.scene.site_id(f"tv_cell_{first_cell}")
                ][0]
            )
            self.status = f"Dust row {self.row_index + 1}/3: moving to start"
            if self._command_base_x(target_x):
                try:
                    self._build_row_plan()
                except RuntimeError as error:
                    self._fail(str(error))
                    return
                self.mode = "orienting"
            return

        if self.mode == "orienting":
            if self._advance_arm():
                self.mode = "sweeping"
                self.cell_cursor = 0
                self.cell_ticks = 0
                self.base_ticks = 0
                self.status = f"Dust row {self.row_index + 1}/3: verifying cells"
            return

        if self.mode == "sweeping":
            cell = DUST_ROWS[self.row_index][self.cell_cursor]
            target_x = float(
                self.data.site_xpos[
                    self.scene.site_id(f"tv_cell_{cell}")
                ][0]
            )
            self.data.ctrl[self.arm_actuators] = self.row_contact_joints
            if not self._command_base_x(target_x):
                self.cell_ticks = 0
                self.status = (
                    f"Dust row {self.row_index + 1}/3: moving to cell {cell + 1}/15"
                )
                return
            reached, detail = self._cell_reached(cell)
            if reached:
                self.cell_ticks += 1
            else:
                self.cell_ticks = 0
                self.status = f"Verifying TV cell {cell + 1}: {detail}"
                return
            if self.cell_ticks < CELL_CONFIRM_TICKS:
                return
            self.scene.mark_tv_cell_clean(cell)
            self.cell_ticks = 0
            self.cell_cursor += 1
            self.base_ticks = 0
            if self.cell_cursor < len(DUST_ROWS[self.row_index]):
                next_cell = DUST_ROWS[self.row_index][self.cell_cursor]
                try:
                    next_x = float(
                        self.data.site_xpos[
                            self.scene.site_id(f"tv_cell_{next_cell}")
                        ][0]
                    )
                    self._validate_base_path(next_x)
                except RuntimeError as error:
                    self._fail(str(error))
                return
            retreat_x = float(
                self.data.site_xpos[self.scene.site_id("tv_cell_0")][0]
            )
            try:
                self._validate_base_path(retreat_x)
            except RuntimeError as error:
                self._fail(str(error))
                return
            self.mode = "returning_row_start"
            self.base_ticks = 0
            self.status = (
                f"Dust row {self.row_index + 1}/3: returning to clear retract side"
            )
            return

        if self.mode == "returning_row_start":
            retreat_x = float(
                self.data.site_xpos[self.scene.site_id("tv_cell_0")][0]
            )
            self.data.ctrl[self.arm_actuators] = self.row_contact_joints
            if not self._command_base_x(retreat_x):
                return
            # We are back at the same side where this row's approach was
            # planned, so its exact reverse is already tool- and
            # environment-checked with the same furniture geometry.
            self.waypoints = self.retreat_waypoints
            self.waypoint_index = 0
            self.waypoint_ticks = 0
            self.mode = "retracting"
            self.status = f"Dust row {self.row_index + 1}/3: retracting"
            return

        if self.mode == "retracting":
            if not self._advance_arm():
                return
            self.mode = "carry_settling"
            self.base_ticks = 0
            self.status = "TV dust: settling rigid tool in compact carry"
            return

        if self.mode == "carry_settling":
            self.data.ctrl[self.arm_actuators] = self.carry_joints
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.tool_body_id,
                velocity,
                0,
            )
            tracking_error = float(
                np.max(
                    np.abs(
                        self.data.qpos[self.arm_qpos] - self.carry_joints
                    )
                )
            )
            if (
                tracking_error < ARM_TRACKING_TOLERANCE
                and float(np.linalg.norm(velocity)) < 0.03
            ):
                self.base_ticks += 1
            else:
                self.base_ticks = 0
            if self.base_ticks < CARRY_SETTLE_TICKS:
                return
            if self.row_index < len(DUST_ROWS) - 1:
                self.row_index += 1
                self.cell_cursor = 0
                try:
                    first_cell = DUST_ROWS[self.row_index][0]
                    first_x = float(
                        self.data.site_xpos[
                            self.scene.site_id(f"tv_cell_{first_cell}")
                        ][0]
                    )
                    self._validate_base_path(first_x)
                except RuntimeError as error:
                    self._fail(str(error))
                    return
                self.mode = "base_to_row_start"
                self.base_ticks = 0
                return
            try:
                self._validate_base_path(0.0)
            except RuntimeError as error:
                self._fail(str(error))
                return
            self.mode = "returning_base"
            self.base_ticks = 0
            self.status = "TV dust: returning to centered compact carry"
            return

        if self.mode == "returning_base":
            if not self._command_base_x(0.0):
                return
            if len(self.scene.cleaned_cells) != 15:
                self._fail(
                    f"coverage incomplete ({len(self.scene.cleaned_cells)}/15)"
                )
                return
            self.mode = "complete"
            self.status = "TV dust complete: verified coverage 15/15"

    def progress(self) -> float:
        if self.mode == "complete":
            return 1.0
        return min(0.99, len(self.scene.cleaned_cells) / 15.0)


__all__ = ["DUST_ROWS", "TVDustExecutor"]
