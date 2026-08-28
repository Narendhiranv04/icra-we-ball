"""Guarded held-remote aiming action for toggling the living-room TV."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from mujoco_scenes.generic_manipulation import (
    ProfiledIK,
    RobotConfigurationCollisionChecker,
)
from mujoco_scenes.robot_profiles import manipulation_profile


REMOTE_ARM_SPEED = 1.55
REMOTE_TRACKING_TOLERANCE = 0.085
REMOTE_HOLD_TICKS = 100
REMOTE_TIMEOUT_TICKS = 30000


class RemoteTVExecutor:
    """Aim the physically held remote, toggle TV state, and retract."""

    def __init__(self, scene, manipulation):
        self.scene = scene
        self.manipulation = manipulation
        self.model = scene.model
        self.data = scene.data
        self.profile = manipulation_profile("google")
        self.arm_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.profile.arm_joints
        )
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.arm_actuators
        )
        self.finger_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.finger_actuators
        )
        self.remote_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "remote_control"
        )
        self.grasp_equality_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "google:pick_weld_remote_control",
        )
        if self.remote_body_id < 0 or self.grasp_equality_id < 0:
            raise RuntimeError("TV remote controller model interface is incomplete")
        self.mode = "idle"
        self.status = "TV remote action idle"
        self.failure: str | None = None
        self.waypoints: list[np.ndarray] = []
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.hold_ticks = 0
        self.elapsed_ticks = 0
        self.toggled = False
        self.checker: RobotConfigurationCollisionChecker | None = None

    def _ids(self, object_type, names) -> np.ndarray:
        values = np.array(
            [mujoco.mj_name2id(self.model, object_type, name) for name in names]
        )
        if np.any(values < 0):
            raise RuntimeError("TV remote controller model interface is incomplete")
        return values

    @property
    def busy(self) -> bool:
        return self.mode in {"aiming", "holding", "retreating"}

    @property
    def navigation_safe(self) -> bool:
        return self.mode in {"idle", "complete"} and self.failure is None

    @staticmethod
    def _joint_points(start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        count = max(1, int(math.ceil(float(np.max(np.abs(goal - start))) / 0.05)))
        return [
            start + fraction * (goal - start)
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]
        ]

    def _plan(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        planning_data = mujoco.MjData(self.model)
        planning_data.qpos[:] = self.data.qpos
        planning_data.qvel[:] = self.data.qvel
        planning_data.eq_active[:] = self.data.eq_active
        mujoco.mj_forward(self.model, planning_data)
        ik = ProfiledIK(self.model, planning_data, self.profile)
        checker = RobotConfigurationCollisionChecker(
            self.model, planning_data, self.profile
        )
        self.checker = checker
        yaw = -math.pi / 2.0
        yaw_rotation = np.array(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        aim_rotation = yaw_rotation @ self.profile.top_down_rotation
        aim_position = np.array((0.0, 0.78, 0.82))
        current = self.data.qpos[self.arm_qpos].copy()
        aim, position_error, angle_error = ik.solve(
            aim_position, self.profile.navigation_joints, aim_rotation
        )
        if position_error > 0.012 or angle_error > math.radians(2.0):
            raise RuntimeError(
                "remote aim IK error "
                f"{position_error * 100:.1f} cm / "
                f"{math.degrees(angle_error):.1f} deg"
            )
        allowed = frozenset((self.remote_body_id,))
        valid, reason = checker.segment_valid(current, aim, allowed)
        if not valid:
            raise RuntimeError(f"unsafe remote aiming path: {reason}")
        valid, reason = checker.segment_valid(
            aim, self.profile.navigation_joints, allowed
        )
        if not valid:
            raise RuntimeError(f"unsafe remote retract path: {reason}")
        return (
            self._joint_points(current, aim),
            self._joint_points(aim, self.profile.navigation_joints),
        )

    def request_toggle(self, current_location: str) -> None:
        if self.busy:
            raise RuntimeError("A TV remote action is already running")
        if self.manipulation.held_object != "remote_control":
            raise RuntimeError("Pick up the TV remote first")
        if current_location != "tv":
            raise RuntimeError("Move to TV before using the remote")
        if self.grasp_equality_id < 0 or not bool(
            self.data.eq_active[self.grasp_equality_id]
        ):
            raise RuntimeError("The remote transport weld is not active")
        approach, retreat = self._plan()
        self.waypoints = approach
        self.retreat_waypoints = retreat
        self.waypoint_index = 0
        self.waypoint_ticks = 0
        self.hold_ticks = 0
        self.elapsed_ticks = 0
        self.toggled = False
        self.failure = None
        self.mode = "aiming"
        self.status = "TV remote: aiming at the screen receiver"

    def _fail(self, message: str) -> None:
        self.failure = message
        self.mode = "failed"
        self.status = f"TV remote action failed: {message}"

    def _advance_waypoints(self) -> bool:
        if self.waypoint_index >= len(self.waypoints):
            return True
        target = self.waypoints[self.waypoint_index]
        command = self.data.ctrl[self.arm_actuators]
        max_step = REMOTE_ARM_SPEED * self.model.opt.timestep
        next_command = command + np.clip(target - command, -max_step, max_step)
        self.data.ctrl[self.arm_actuators] = next_command
        command_error = float(np.max(np.abs(target - next_command)))
        tracking_error = float(
            np.max(np.abs(self.data.qpos[self.arm_qpos] - target))
        )
        if command_error < 0.001 and tracking_error < REMOTE_TRACKING_TOLERANCE:
            self.waypoint_ticks += 1
        else:
            self.waypoint_ticks = 0
        if self.waypoint_ticks >= 4:
            self.waypoint_index += 1
            self.waypoint_ticks = 0
        return self.waypoint_index >= len(self.waypoints)

    def _live_safe(self) -> bool:
        if self.checker is None:
            return True
        valid, reason = self.checker.evaluate_live(
            self.data, frozenset((self.remote_body_id,))
        )
        if not valid:
            self._fail(f"live collision guard stopped motion: {reason}")
        return valid

    def update(self) -> None:
        if not self.busy:
            return
        self.elapsed_ticks += 1
        if self.elapsed_ticks >= REMOTE_TIMEOUT_TICKS:
            self._fail(f"timeout while {self.status}")
            return
        if self.elapsed_ticks % 5 == 0 and not self._live_safe():
            return
        held_executor = self.manipulation.executor
        if held_executor is None:
            self._fail("held-remote manipulation state disappeared")
            return
        self.data.ctrl[self.finger_actuators] = held_executor.close_target
        if self.mode == "aiming":
            if self._advance_waypoints():
                self.mode = "holding"
                self.status = "TV remote: aim confirmed, sending power command"
            return
        if self.mode == "holding":
            self.data.ctrl[self.arm_actuators] = self.waypoints[-1]
            self.hold_ticks += 1
            if self.hold_ticks >= REMOTE_HOLD_TICKS:
                if not self.toggled:
                    self.scene.set_tv_power(not self.scene.tv_power_on)
                    self.toggled = True
                self.waypoints = self.retreat_waypoints
                self.waypoint_index = 0
                self.waypoint_ticks = 0
                self.mode = "retreating"
                state = "on" if self.scene.tv_power_on else "off"
                self.status = f"TV switched {state}; retracting remote"
            return
        if self.mode == "retreating" and self._advance_waypoints():
            self.mode = "complete"
            state = "on" if self.scene.tv_power_on else "off"
            self.status = f"TV remote action complete: screen is {state}"

    def progress(self) -> float:
        if self.mode == "complete":
            return 1.0
        return {
            "idle": 0.0,
            "aiming": 0.35,
            "holding": 0.65,
            "retreating": 0.85,
            "failed": 0.0,
        }[self.mode]
