"""Collision-checked navigation around live living-room furniture."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from mujoco_scenes.mobile_motion import (
    BASE_COMMAND_TOLERANCE,
    BASE_FINAL_POSITION_TOLERANCE,
    BASE_FINAL_YAW_TOLERANCE,
    BASE_LINEAR_COMMAND_SPEED,
    BASE_SETTLE_LINEAR_SPEED,
    BASE_SETTLE_TICKS,
    BASE_SETTLE_YAW_SPEED,
    BASE_YAW_COMMAND_SPEED,
    BASE_YAW_COMMAND_TOLERANCE,
    BasePose,
    MuJoCoBaseCollisionChecker,
    RRTStarPlanner,
)
from mujoco_scenes.robot_profiles import mobile_profile
from mujoco_scenes.living_room_scene import LIVING_ROOM_FORWARD_LIMITS


LIVING_ROOM_DESTINATIONS = (
    "home",
    "table_south",
    "table_north",
    "table_east",
    "table_west",
    "bookshelf",
    "drawer",
    "drawer_left",
    "drawer_right",
    "couch",
    "tv",
    "duster",
)
TABLE_NAVIGATION_HALF_EXTENTS = (0.78, 0.66)


def _angle_delta(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


@dataclass(frozen=True)
class LivingRoomLayout:
    # Fixed world pose corresponding to zero base qpos.  This is deliberately
    # separate from the symbolic Home destination, which follows the table.
    base_origin: BasePose = BasePose(0.0, -1.25, 0.0)
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-1.35, 1.42),
        (-1.58, 0.78),
    )

    @staticmethod
    def _table_relative_pose(
        table_pose: tuple[float, float, float],
        local_x: float,
        local_y: float,
        yaw_offset: float,
    ) -> BasePose:
        table_x, table_y, table_yaw = table_pose
        cosine = math.cos(table_yaw)
        sine = math.sin(table_yaw)
        return BasePose(
            table_x + cosine * local_x - sine * local_y,
            table_y + sine * local_x + cosine * local_y,
            math.atan2(
                math.sin(table_yaw + yaw_offset),
                math.cos(table_yaw + yaw_offset),
            ),
        )

    def destination_pose(self, scene, name: str) -> BasePose:
        if name not in LIVING_ROOM_DESTINATIONS:
            raise ValueError(
                f"Unknown living-room destination '{name}'. Choose: "
                f"{', '.join(LIVING_ROOM_DESTINATIONS)}"
            )
        table_pose = scene.table_pose
        poses = {
            # All four approach poses share the fixed table frame.
            "home": self._table_relative_pose(table_pose, 0.0, -0.90, 0.0),
            "table_south": self._table_relative_pose(
                table_pose, 0.0, -0.68, 0.0
            ),
            "table_north": self._table_relative_pose(
                table_pose, 0.0, 0.68, math.pi
            ),
            "table_east": self._table_relative_pose(
                table_pose, 0.82, 0.0, math.pi / 2
            ),
            "table_west": self._table_relative_pose(
                table_pose, -0.82, 0.0, -math.pi / 2
            ),
            "bookshelf": BasePose(-0.80, 0.48, 0.0),
            "drawer": BasePose(0.34, 0.34, 0.0),
            "drawer_left": BasePose(-0.34, 0.34, 0.0),
            "drawer_right": BasePose(0.34, 0.34, 0.0),
            # Approach the open east end of the sofa and face west. This keeps
            # the base outside both furniture envelopes while the low
            # cameras look laterally through the under-seat clearance.
            "couch": BasePose(0.10, -1.15, math.pi / 2),
            # A held duster extends beyond the compact arm envelope.  Stop
            # far enough from the console for RRT* to include the attached
            # rigid tool while leaving the screen within arm reach.
            "tv": BasePose(0.0, 0.50, 0.0),
            "duster": BasePose(1.28, 0.64, 0.0),
        }
        return poses[name]


class LivingRoomNavigationExecutor:
    """Plan arbitrary living-room moves around the fixed furniture."""

    def __init__(self, scene, layout: LivingRoomLayout | None = None):
        if scene.robot_name != "google":
            raise ValueError("Living-room navigation currently requires Google Robot")
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.layout = layout or LivingRoomLayout()
        self.profile = replace(
            mobile_profile("google"),
            forward_limits=LIVING_ROOM_FORWARD_LIMITS,
        )
        self.joint_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.profile.base_joints
        )
        self.qpos_addresses = tuple(
            int(self.model.jnt_qposadr[joint_id]) for joint_id in self.joint_ids
        )
        self.dof_addresses = tuple(
            int(self.model.jnt_dofadr[joint_id]) for joint_id in self.joint_ids
        )
        self.actuator_ids = tuple(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.profile.base_actuators
        )
        self.current_location = "home"
        self.requested_location: str | None = None
        self.targets: list[tuple[float, float, float]] = []
        self.target_index = 0
        self.settle_ticks = 0
        self.started_at = 0.0
        self.status = "Living-room navigation idle at home"
        self.failure: str | None = None
        self.checker: MuJoCoBaseCollisionChecker | None = None
        self.guard_tick = 0

    @property
    def busy(self) -> bool:
        return self.target_index < len(self.targets) and self.failure is None

    def _world_pose(self) -> BasePose:
        forward, lateral, yaw = self.data.qpos[list(self.qpos_addresses)]
        return BasePose(
            float(-lateral),
            float(self.layout.base_origin.y + forward),
            float(yaw),
        )

    @staticmethod
    def _rotation_targets(
        x: float, y: float, start: float, goal: float
    ) -> list[tuple[float, float, float]]:
        delta = _angle_delta(goal, start)
        count = max(1, int(math.ceil(abs(delta) / math.radians(4))))
        return [
            (x, y, start + delta * fraction)
            for fraction in np.linspace(0.0, 1.0, count + 1)[1:]
        ]

    @staticmethod
    def _outside_table_keepout(
        x: float,
        y: float,
        table_pose: tuple[float, float, float],
    ) -> bool:
        table_x, table_y, table_yaw = table_pose
        delta_x = x - table_x
        delta_y = y - table_y
        cosine = math.cos(table_yaw)
        sine = math.sin(table_yaw)
        local_x = cosine * delta_x + sine * delta_y
        local_y = -sine * delta_x + cosine * delta_y
        half_x, half_y = TABLE_NAVIGATION_HALF_EXTENTS
        return abs(local_x) >= half_x or abs(local_y) >= half_y

    @staticmethod
    def _outside_couch_keepout(x: float, y: float) -> bool:
        """Keep RRT* away from contact-only boundaries around the L sofa."""
        inside_west_section = -1.95 < x < -0.83 and -1.42 < y < 0.85
        inside_south_section = -1.47 < x < -0.18 and -1.98 < y < -0.83
        return not (inside_west_section or inside_south_section)

    def request_move(self, destination: str) -> None:
        if self.busy:
            raise RuntimeError("A living-room move is already running")
        goal = self.layout.destination_pose(self.scene, destination)
        current = self._world_pose()
        self.checker = MuJoCoBaseCollisionChecker(
            self.model, self.data, self.profile
        )
        table_pose = self.scene.table_pose

        def navigation_state_valid(x: float, y: float) -> bool:
            return bool(
                self.checker(x, y)
                and self._outside_table_keepout(x, y, table_pose)
                and self._outside_couch_keepout(x, y)
            )

        targets: list[tuple[float, float, float]] = []
        if not self._outside_table_keepout(current.x, current.y, table_pose):
            table_x, table_y, table_yaw = table_pose
            cosine = math.cos(table_yaw)
            sine = math.sin(table_yaw)
            delta_x = current.x - table_x
            delta_y = current.y - table_y
            local_x = cosine * delta_x + sine * delta_y
            local_y = -sine * delta_x + cosine * delta_y
            half_x, half_y = TABLE_NAVIGATION_HALF_EXTENTS
            if abs(local_x) / half_x >= abs(local_y) / half_y:
                local_x = math.copysign(half_x + 0.04, local_x or 1.0)
            else:
                local_y = math.copysign(half_y + 0.04, local_y or 1.0)
            retreat_x = table_x + cosine * local_x - sine * local_y
            retreat_y = table_y + sine * local_x + cosine * local_y
            distance = math.hypot(
                retreat_x - current.x, retreat_y - current.y
            )
            count = max(1, int(math.ceil(distance / 0.025)))
            retreat_targets = [
                (
                    current.x + (retreat_x - current.x) * fraction,
                    current.y + (retreat_y - current.y) * fraction,
                    current.yaw,
                )
                for fraction in np.linspace(0.0, 1.0, count + 1)[1:]
            ]
            for pose in retreat_targets:
                if not self.checker.is_pose_valid(*pose):
                    raise RuntimeError(
                        "Cannot leave the coffee-table interaction envelope"
                    )
            targets.extend(retreat_targets)
            current = BasePose(retreat_x, retreat_y, current.yaw)
        if abs(current.yaw) > math.radians(0.5):
            # Back away from the table before changing heading.  Rotating
            # at the close side pose can sweep the shoulder/base shell into a
            # table leg even though the final headings are each valid.
            if self.current_location in {"table_east", "table_west"}:
                sign = 1.0 if self.current_location == "table_east" else -1.0
                clearance = 0.12 if sign > 0 else 0.08
                retreat_x = current.x + sign * clearance
                count = max(1, int(math.ceil(clearance / 0.025)))
                retreat_targets = [
                    (
                        current.x + sign * clearance * fraction,
                        current.y,
                        current.yaw,
                    )
                    for fraction in np.linspace(0.0, 1.0, count + 1)[1:]
                ]
                for pose in retreat_targets:
                    if not self.checker.is_pose_valid(*pose):
                        raise RuntimeError(
                            "Cannot retreat from the table before rotating"
                        )
                targets.extend(retreat_targets)
                current = BasePose(retreat_x, current.y, current.yaw)
            rotation = self._rotation_targets(
                current.x, current.y, current.yaw, 0.0
            )
            for pose in rotation:
                if not self.checker.is_pose_valid(*pose):
                    raise RuntimeError(
                        "Cannot rotate to the neutral heading at the current pose"
                    )
            targets.extend(rotation)
        planner = RRTStarPlanner(
            navigation_state_valid,
            bounds=self.layout.bounds,
            max_iterations=900,
            step_size=0.18,
            neighbor_radius=0.28,
            seed=31,
        )
        path = planner.plan((current.x, current.y), (goal.x, goal.y))
        targets.extend((x, y, 0.0) for x, y in path[1:])
        rotation = self._rotation_targets(goal.x, goal.y, 0.0, goal.yaw)
        for pose in rotation:
            if not self.checker.is_pose_valid(*pose):
                raise RuntimeError(
                    f"Final rotation for {destination} is in collision"
                )
        targets.extend(rotation)
        if not targets:
            self.current_location = destination
            self.status = f"Already at {destination}"
            return
        self.targets = targets
        self.target_index = 0
        self.settle_ticks = 0
        self.guard_tick = 0
        self.requested_location = destination
        self.failure = None
        self.started_at = time.monotonic()
        self.status = (
            f"Moving to {destination}: RRT* path has {len(targets)} waypoints"
        )

    def _live_collision(self) -> str | None:
        if self.checker is None:
            return None
        for contact in self.data.contact:
            first_robot = self.checker._geom_is_robot(contact.geom1)
            second_robot = self.checker._geom_is_robot(contact.geom2)
            if first_robot == second_robot:
                continue
            other_geom = contact.geom2 if first_robot else contact.geom1
            other_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or "unnamed living-room geometry"
            if other_name != "floor":
                return other_name
        return None

    def update(self) -> None:
        if not self.busy:
            return
        self.guard_tick += 1
        if self.guard_tick % 5 == 0:
            collision = self._live_collision()
            if collision:
                self.failure = f"Live navigation collision with {collision}"
                self.targets = []
                self.status = f"Move failed: {self.failure}"
                return
        x, y, yaw = self.targets[self.target_index]
        target = np.array((y - self.layout.base_origin.y, -x, yaw))
        command = self.data.ctrl[list(self.actuator_ids)]
        max_step = self.model.opt.timestep * np.array(
            (
                BASE_LINEAR_COMMAND_SPEED,
                BASE_LINEAR_COMMAND_SPEED,
                BASE_YAW_COMMAND_SPEED,
            )
        )
        self.data.ctrl[list(self.actuator_ids)] = command + np.clip(
            target - command, -max_step, max_step
        )
        current = self.data.qpos[list(self.qpos_addresses)]
        velocity = self.data.qvel[list(self.dof_addresses)]
        position_error = float(np.linalg.norm(current[:2] - target[:2]))
        yaw_error = abs(_angle_delta(float(target[2]), float(current[2])))
        is_final = self.target_index == len(self.targets) - 1
        position_tolerance = BASE_FINAL_POSITION_TOLERANCE if is_final else 0.018
        yaw_tolerance = BASE_FINAL_YAW_TOLERANCE if is_final else math.radians(1.2)
        if position_error >= position_tolerance or yaw_error >= yaw_tolerance:
            self.settle_ticks = 0
            return
        if not is_final:
            self.target_index += 1
            return
        command_error = target - self.data.ctrl[list(self.actuator_ids)]
        command_error[2] = _angle_delta(
            float(target[2]), float(self.data.ctrl[self.actuator_ids[2]])
        )
        command_settled = (
            float(np.max(np.abs(command_error[:2]))) < BASE_COMMAND_TOLERANCE
            and abs(float(command_error[2])) < BASE_YAW_COMMAND_TOLERANCE
        )
        velocity_settled = (
            float(np.max(np.abs(velocity[:2]))) < BASE_SETTLE_LINEAR_SPEED
            and abs(float(velocity[2])) < BASE_SETTLE_YAW_SPEED
        )
        if not command_settled or not velocity_settled:
            self.settle_ticks = 0
            self.status = f"Moving to {self.requested_location}: settling"
            return
        self.settle_ticks += 1
        if self.settle_ticks < BASE_SETTLE_TICKS:
            return
        self.data.ctrl[list(self.actuator_ids)] = target
        self.target_index += 1
        assert self.requested_location is not None
        self.current_location = self.requested_location
        elapsed = time.monotonic() - self.started_at
        self.status = f"Move complete: {self.current_location} ({elapsed:.1f} s)"

    def progress(self) -> float:
        if not self.targets:
            return 0.0
        return min(1.0, self.target_index / len(self.targets))
