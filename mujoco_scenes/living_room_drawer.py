"""Lightweight guarded actuator control for the media-console drawer."""

from __future__ import annotations

import mujoco
import numpy as np


DRAWER_ACTIONS = ("open", "close")
DRAWER_SIDES = ("left", "right")
DRAWER_OPEN_POSITION = 0.27
DRAWER_COMMAND_SPEED = 0.20
DRAWER_POSITION_TOLERANCE = 0.004
DRAWER_SETTLE_SPEED = 0.008
DRAWER_SETTLE_TICKS = 80
DRAWER_TIMEOUT_TICKS = 12000


class MediaConsoleDrawerExecutor:
    """Open or close the rigid slide-joint drawer at a bounded speed."""

    def __init__(self, scene, side: str = "right"):
        if side not in DRAWER_SIDES:
            raise ValueError(f"Choose a drawer side from: {DRAWER_SIDES}")
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.side = side
        interface_prefix = f"media_console_{side}_drawer"
        self.joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{interface_prefix}_slide",
        )
        self.actuator_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"{interface_prefix}_actuator",
        )
        if self.joint_id < 0 or self.actuator_id < 0:
            raise RuntimeError(
                f"Media-console {side} drawer model interface is incomplete"
            )
        self.qpos_address = int(self.model.jnt_qposadr[self.joint_id])
        self.dof_address = int(self.model.jnt_dofadr[self.joint_id])
        self.mode = "idle"
        self.status = f"Media-console {side} drawer idle"
        self.failure: str | None = None
        self.action: str | None = None
        self.target = 0.0
        self.elapsed_ticks = 0
        self.settle_ticks = 0

    @property
    def busy(self) -> bool:
        return self.mode in {"moving", "settling"}

    @property
    def is_open(self) -> bool:
        return float(self.data.qpos[self.qpos_address]) >= (
            DRAWER_OPEN_POSITION - 0.015
        )

    @property
    def navigation_safe(self) -> bool:
        return not self.busy and self.failure is None

    def request(self, action: str, current_location: str) -> None:
        if self.busy:
            raise RuntimeError(
                f"The media-console {self.side} drawer is already moving"
            )
        if action not in DRAWER_ACTIONS:
            raise ValueError(f"Choose a drawer action from: {DRAWER_ACTIONS}")
        if current_location not in {"drawer", "drawer_left", "drawer_right"}:
            raise RuntimeError(
                "Move to Media-console drawer before opening or closing it"
            )
        if action == "open" and self.is_open:
            raise RuntimeError(
                f"The media-console {self.side} drawer is already open"
            )
        if action == "close" and not self.is_open:
            raise RuntimeError(
                f"The media-console {self.side} drawer is already closed"
            )
        self.action = action
        self.target = DRAWER_OPEN_POSITION if action == "open" else 0.0
        self.elapsed_ticks = 0
        self.settle_ticks = 0
        self.failure = None
        self.mode = "moving"
        self.status = f"{self.side.title()} drawer {action}: moving slowly"

    def _fail(self, message: str) -> None:
        self.failure = message
        self.mode = "failed"
        self.status = f"Drawer action failed: {message}"

    def update(self) -> None:
        if not self.busy:
            return
        self.elapsed_ticks += 1
        if self.elapsed_ticks >= DRAWER_TIMEOUT_TICKS:
            self._fail(f"timeout while trying to {self.action}")
            return
        command = float(self.data.ctrl[self.actuator_id])
        max_step = DRAWER_COMMAND_SPEED * self.model.opt.timestep
        self.data.ctrl[self.actuator_id] = command + float(
            np.clip(self.target - command, -max_step, max_step)
        )
        position_error = abs(
            float(self.data.qpos[self.qpos_address]) - self.target
        )
        if self.mode == "moving" and position_error <= DRAWER_POSITION_TOLERANCE:
            self.data.ctrl[self.actuator_id] = self.target
            self.mode = "settling"
            self.status = f"{self.side.title()} drawer {self.action}: settling"
            return
        if self.mode == "settling":
            velocity = abs(float(self.data.qvel[self.dof_address]))
            if (
                position_error <= DRAWER_POSITION_TOLERANCE
                and velocity <= DRAWER_SETTLE_SPEED
            ):
                self.settle_ticks += 1
            else:
                self.settle_ticks = 0
            if self.settle_ticks >= DRAWER_SETTLE_TICKS:
                self.mode = "complete"
                self.status = (
                    f"Media-console {self.side} drawer "
                    f"{self.action} complete"
                )

    def progress(self) -> float:
        if self.mode == "complete":
            return 1.0
        if self.mode == "settling":
            return 0.85
        if self.mode != "moving":
            return 0.0
        position = float(self.data.qpos[self.qpos_address])
        span = max(abs(self.target - position), DRAWER_POSITION_TOLERANCE)
        return min(0.8, 0.8 * (1.0 - span / DRAWER_OPEN_POSITION))
