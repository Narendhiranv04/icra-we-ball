"""Symmetric, contact-confirmed front opening motions for both drawers."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_scenes.pick_motion import (
    CARRY_POSITION,
    FINGER_GEOMS,
    HOME_ARM_SEED,
    OPEN_WIDTH,
    TOP_DOWN_ROTATION,
    ArmWaypoint,
    PickExecutor,
    VerticalIK,
    carry_position_at,
)


# The gripper approaches along world +Y from Home. Its local +Y closing axis
# is vertical, so the two pads straddle the level horizontal handle bar.
DRAWER_FRONT_GRASP_ROTATION = np.array(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
)
DRAWER_PREGRASP_DISTANCE = 0.080
DRAWER_GRIP_ORIGIN_OFFSET = -0.005
DRAWER_OVERHEAD_CLEARANCE = 0.20
DRAWER_ROUTE_HIGH = 0.95
DRAWER_ORIENTATION_HEIGHT = 0.85
DRAWER_ORIENTATION_Y = -0.65
DRAWER_RETREAT_DISTANCE = 0.200
DRAWER_RETURN_HEIGHT = 0.85
DRAWER_PATH_RESOLUTION = 0.010
DRAWER_PULL_SAMPLES = 41
DRAWER_CLOSE_STEP = 0.001
DRAWER_CONTACT_TICKS = 12
DRAWER_RELEASE_TICKS = 90
DRAWER_APPROACH_GRACE_TICKS = 1000
DRAWER_HANDLE_ARRIVAL_TOLERANCE = 0.050
DRAWER_ARM_TRACKING_COMPENSATION = 2.5


@dataclass(frozen=True)
class DrawerSpec:
    label: str
    handle_site: str
    handle_geoms: frozenset[str]
    tray_body: str
    joint: str
    actuator: str
    equality: str


DRAWER_SPECS = {
    name: DrawerSpec(
        label=f"Drawer {index}",
        handle_site=f"{name}_handle_grasp",
        handle_geoms=frozenset(
            {
                f"{name}_handle_left",
                f"{name}_handle_right",
                f"{name}_handle_bar",
            }
        ),
        tray_body=f"drawer_{name}_tray",
        joint=f"{name}_slide_joint",
        actuator=f"{name}_slide_actuator",
        equality=f"robot0:open_connect_{name}",
    )
    for index, name in enumerate(("D1", "D2"), start=1)
}


class DrawerOpenExecutor:
    """Open D1 or D2 from Home with the same straight front-grasp motion."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        picker: PickExecutor,
    ):
        self.model = model
        self.data = data
        self.picker = picker
        self.mode = "idle"
        self.status = "Drawer open idle"
        self.failure: str | None = None
        self.target: str | None = None
        self.spec: DrawerSpec | None = None
        self.handle_site_id = -1
        self.tray_body_id = -1
        self.slide_joint_id = -1
        self.slide_qpos = -1
        self.slide_actuator = -1
        self.equality_id = -1
        self.max_slide = 0.0
        self.close_target = OPEN_WIDTH
        self.close_ticks = 0
        self.contact_ticks = 0
        self.approach_wait_ticks = 0
        self.release_ticks = 0
        self.release_arm_target = self.picker._current_arm()
        self.pull_values = np.zeros(0)

    @property
    def busy(self) -> bool:
        return self.mode not in {"idle", "complete", "failed"}

    def is_fully_open(self, drawer_name: str) -> bool:
        spec = DRAWER_SPECS[drawer_name]
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint
        )
        if joint_id < 0:
            return True
        qpos = int(self.model.jnt_qposadr[joint_id])
        limit = float(self.model.jnt_range[joint_id, 1])
        return bool(self.data.qpos[qpos] >= limit - 0.015)

    def _fail(self, message: str) -> None:
        if self.equality_id >= 0:
            self.data.eq_active[self.equality_id] = 0
        self.mode = "failed"
        self.failure = message
        self.status = f"Open drawer failed: {message}"

    def _compensate_arm_tracking(self) -> None:
        command = self.data.ctrl[self.picker.arm_actuators].copy()
        live = self.data.qpos[self.picker.arm_qpos]
        compensated = command + DRAWER_ARM_TRACKING_COMPENSATION * (
            command - live
        )
        ranges = self.model.jnt_range[self.picker.arm_joint_ids]
        limited = self.model.jnt_limited[self.picker.arm_joint_ids].astype(bool)
        self.data.ctrl[self.picker.arm_actuators] = np.clip(
            compensated,
            np.where(limited, ranges[:, 0], -np.inf),
            np.where(limited, ranges[:, 1], np.inf),
        )

    def _configure_target(self, drawer_name: str) -> None:
        try:
            self.spec = DRAWER_SPECS[drawer_name]
        except KeyError as error:
            raise ValueError(f"Unknown drawer: {drawer_name}") from error
        self.target = drawer_name
        self.handle_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.spec.handle_site
        )
        self.tray_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.spec.tray_body
        )
        self.slide_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, self.spec.joint
        )
        self.slide_actuator = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.spec.actuator
        )
        self.equality_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, self.spec.equality
        )
        if min(
            self.handle_site_id,
            self.tray_body_id,
            self.slide_joint_id,
            self.slide_actuator,
            self.equality_id,
        ) < 0:
            raise RuntimeError(f"Missing physical model elements for {drawer_name}")
        self.slide_qpos = int(self.model.jnt_qposadr[self.slide_joint_id])
        self.max_slide = float(self.model.jnt_range[self.slide_joint_id, 1])

    def request_open(self, drawer_name: str, physical_location: str) -> None:
        if self.busy:
            raise RuntimeError("A drawer open action is already running")
        if physical_location != "home":
            raise RuntimeError("Move (Home) is required before opening a drawer")
        if self.picker.busy or self.picker.held_object is not None:
            raise RuntimeError("The gripper must be empty before opening a drawer")
        self._configure_target(drawer_name)
        assert self.spec is not None
        if self.is_fully_open(drawer_name):
            self.mode = "complete"
            self.status = f"Open complete: {self.spec.label} is already fully open"
            return

        mujoco.mj_forward(self.model, self.data)
        approach = DRAWER_FRONT_GRASP_ROTATION[:, 0]
        handle = self.data.site_xpos[self.handle_site_id].copy()
        grasp = handle - DRAWER_GRIP_ORIGIN_OFFSET * approach
        pregrasp = grasp - DRAWER_PREGRASP_DISTANCE * approach
        overhead = pregrasp + np.array((0.0, 0.0, DRAWER_OVERHEAD_CLEARANCE))

        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        carry_position = carry_position_at(CARRY_POSITION, "home")
        carry_joints, pos_error, angle_error = ik.solve(
            carry_position, HOME_ARM_SEED.copy(), TOP_DOWN_ROTATION
        )
        if pos_error > 0.012 or angle_error > np.deg2rad(2.0):
            raise RuntimeError(
                "Could not reach the safe carry corridor "
                f"({pos_error * 100:.1f} cm, {np.rad2deg(angle_error):.1f} deg)"
            )

        waypoints = [ArmWaypoint(current, "Opening gripper")]
        waypoints.extend(
            ArmWaypoint(joints, "Moving to the safe carry corridor")
            for joints in self.picker._joint_interpolation(current, carry_joints)
        )
        side_high = np.array((handle[0], -0.75, DRAWER_ROUTE_HIGH))
        high_path, seed = self.picker._solve_path(
            ik,
            self.picker._cartesian_points(
                carry_position, side_high, DRAWER_PATH_RESOLUTION * 2.0
            ),
            carry_joints,
            f"Moving above the {self.spec.label} side",
            TOP_DOWN_ROTATION,
        )
        waypoints.extend(high_path)
        orientation_point = np.array(
            (handle[0], DRAWER_ORIENTATION_Y, DRAWER_ORIENTATION_HEIGHT)
        )
        orientation_approach, seed = self.picker._solve_path(
            ik,
            self.picker._cartesian_points(
                side_high, orientation_point, DRAWER_PATH_RESOLUTION * 2.0
            ),
            seed,
            "Entering the front-grasp corridor",
            TOP_DOWN_ROTATION,
        )
        waypoints.extend(orientation_approach)
        front_joints, pos_error, angle_error = ik.solve(
            orientation_point, current, DRAWER_FRONT_GRASP_ROTATION
        )
        if pos_error > 0.012 or angle_error > np.deg2rad(2.0):
            raise RuntimeError(
                f"Could not face the {self.spec.label} handle "
                f"({pos_error * 100:.1f} cm, "
                f"{np.rad2deg(angle_error):.1f} deg)"
            )
        waypoints.extend(
            ArmWaypoint(joints, "Turning gripper to face drawer")
            for joints in self.picker._joint_interpolation(seed, front_joints)
        )
        seed = front_joints
        overhead_path, overhead_joints = self.picker._solve_path(
            ik,
            self.picker._cartesian_points(
                orientation_point, overhead, DRAWER_PATH_RESOLUTION * 2.0
            ),
            seed,
            f"Moving above {self.spec.label} handle",
            DRAWER_FRONT_GRASP_ROTATION,
        )
        waypoints.extend(overhead_path)
        descent, seed = self.picker._solve_path(
            ik,
            self.picker._cartesian_points(
                overhead, pregrasp, DRAWER_PATH_RESOLUTION
            ),
            overhead_joints,
            "Descending to front handle hover",
            DRAWER_FRONT_GRASP_ROTATION,
            angle_tolerance=np.deg2rad(7.0),
        )
        waypoints.extend(descent)
        insertion, _ = self.picker._solve_path(
            ik,
            self.picker._cartesian_points(
                pregrasp, grasp, DRAWER_PATH_RESOLUTION
            ),
            seed,
            "Approaching handle horizontally along +Y",
            DRAWER_FRONT_GRASP_ROTATION,
            angle_tolerance=np.deg2rad(7.0),
        )
        waypoints.extend(insertion)
        self.picker._start_trajectory(waypoints)
        self.data.ctrl[self.slide_actuator] = float(self.data.qpos[self.slide_qpos])
        self.close_target = OPEN_WIDTH
        self.close_ticks = 0
        self.contact_ticks = 0
        self.approach_wait_ticks = 0
        self.release_ticks = 0
        self.pull_values = np.zeros(0)
        self.failure = None
        self.mode = "approach"
        self.status = f"Open {self.spec.label}: approaching from the front"

    def _finger_contacts(self) -> set[str]:
        assert self.spec is not None
        contacts: set[str] = set()
        for contact in self.data.contact:
            first = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            second = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            if first in self.spec.handle_geoms and second in FINGER_GEOMS:
                contacts.add(second)
            elif second in self.spec.handle_geoms and first in FINGER_GEOMS:
                contacts.add(first)
        return contacts

    def _activate_handle_connect(self) -> None:
        gripper_rotation = self.data.xmat[
            self.picker.gripper_body_id
        ].reshape(3, 3)
        tray_rotation = self.data.xmat[self.tray_body_id].reshape(3, 3)
        anchor = self.data.site_xpos[self.handle_site_id].copy()
        self.model.eq_data[self.equality_id, :3] = gripper_rotation.T @ (
            anchor - self.data.xpos[self.picker.gripper_body_id]
        )
        self.model.eq_data[self.equality_id, 3:6] = tray_rotation.T @ (
            anchor - self.data.xpos[self.tray_body_id]
        )
        self.data.eq_active[self.equality_id] = 1

    def _plan_pull(self) -> None:
        assert self.spec is not None
        mujoco.mj_forward(self.model, self.data)
        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        start_slide = float(self.data.qpos[self.slide_qpos])
        start_grip = self.data.site_xpos[self.picker.grip_site_id].copy()
        start_rotation = self.data.site_xmat[
            self.picker.grip_site_id
        ].reshape(3, 3).copy()
        start_handle = self.data.site_xpos[self.handle_site_id].copy()
        contact_offset = start_rotation.T @ (start_handle - start_grip)

        waypoints = [ArmWaypoint(current, "Beginning horizontal drawer pull")]
        slides = [start_slide]
        seed = current
        values = np.linspace(start_slide, self.max_slide, DRAWER_PULL_SAMPLES)
        for value in values[1:]:
            handle = start_handle + np.array((0.0, -(value - start_slide), 0.0))
            grip = handle - start_rotation @ contact_offset
            seed, pos_error, angle_error = ik.solve(grip, seed, start_rotation)
            if pos_error > 0.015 or angle_error > np.deg2rad(2.0):
                raise RuntimeError(
                    f"Could not pull {self.spec.label} horizontally "
                    f"({pos_error * 100:.1f} cm, {np.rad2deg(angle_error):.1f} deg)"
                )
            waypoints.append(ArmWaypoint(seed.copy(), "Pulling straight along -Y"))
            slides.append(float(value))
        self.data.ctrl[self.picker.arm_actuators] = current
        self.picker._start_trajectory(waypoints)
        if len(self.picker.waypoints) > len(slides):
            slides = [slides[0]] * (len(self.picker.waypoints) - len(slides)) + slides
        self.pull_values = np.asarray(slides)

    def _plan_horizontal_retreat(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        position = self.data.site_xpos[self.picker.grip_site_id].copy()
        rotation = self.data.site_xmat[
            self.picker.grip_site_id
        ].reshape(3, 3).copy()
        if self.target == "D1":
            # Fetch's single right-mounted arm cannot preserve the fully
            # constrained front attitude while clearing the left drawer. Use
            # the same horizontal Cartesian objective but let the now-empty
            # wrist relax naturally; shoulder pan stays fixed to avoid the
            # torso collision branch.
            sample = mujoco.MjData(self.model)
            sample.qpos[:] = self.data.qpos
            qpos = self.picker.arm_qpos
            dofs = self.model.jnt_dofadr[self.picker.arm_joint_ids]
            ranges = self.model.jnt_range[self.picker.arm_joint_ids]
            lower = ranges[:, 0] + 0.01
            upper = ranges[:, 1] - 0.01
            fixed_pan = float(current[0])
            weights = np.diag((0.25, 1.0, 1.0))
            horizontal = []
            for _ in range(40):
                mujoco.mj_forward(self.model, sample)
                target = sample.site_xpos[self.picker.grip_site_id].copy()
                target[1] -= 0.005
                for _ in range(50):
                    mujoco.mj_forward(self.model, sample)
                    error = weights @ (
                        target - sample.site_xpos[self.picker.grip_site_id]
                    )
                    jacobian = np.zeros((3, self.model.nv))
                    mujoco.mj_jacSite(
                        self.model,
                        sample,
                        jacobian,
                        None,
                        self.picker.grip_site_id,
                    )
                    reduced = weights @ jacobian[:, dofs[1:]]
                    delta = reduced.T @ np.linalg.solve(
                        reduced @ reduced.T + 0.02 * np.eye(3), error
                    )
                    joints = sample.qpos[qpos].copy()
                    joints[1:] = np.clip(
                        joints[1:] + np.clip(delta, -0.01, 0.01),
                        lower[1:],
                        upper[1:],
                    )
                    joints[0] = fixed_pan
                    sample.qpos[qpos] = joints
                horizontal.append(
                    ArmWaypoint(
                        sample.qpos[qpos].copy(),
                        "Retreating horizontally from open drawer",
                    )
                )
                mujoco.mj_forward(self.model, sample)
                if sample.site_xpos[self.picker.grip_site_id, 1] <= -0.78:
                    break
            planned = sample.site_xpos[self.picker.grip_site_id]
            if planned[1] > -0.75 or abs(planned[2] - position[2]) > 0.025:
                raise RuntimeError("Could not find a horizontal left-drawer retreat")
        else:
            hover = (
                position
                - DRAWER_RETREAT_DISTANCE * DRAWER_FRONT_GRASP_ROTATION[:, 0]
            )
            horizontal, _ = self.picker._solve_path(
                ik,
                self.picker._cartesian_points(
                    position, hover, DRAWER_PATH_RESOLUTION
                ),
                current,
                "Retreating horizontally from open drawer",
                rotation,
                position_tolerance=0.050,
                angle_tolerance=np.deg2rad(6.0),
            )
        waypoints = [ArmWaypoint(current, "Beginning horizontal retreat")]
        waypoints.extend(horizontal)
        self.data.ctrl[self.picker.arm_actuators] = current
        self.picker._start_trajectory(waypoints)

    def _plan_vertical_clearance(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        position = self.data.site_xpos[self.picker.grip_site_id].copy()
        rotation = self.data.site_xmat[
            self.picker.grip_site_id
        ].reshape(3, 3).copy()
        raised = position.copy()
        raised[2] = max(raised[2], DRAWER_RETURN_HEIGHT)
        vertical, _ = self.picker._solve_path(
            ik,
            self.picker._cartesian_points(
                position, raised, DRAWER_PATH_RESOLUTION * 2.0
            ),
            current,
            "Rising clear after horizontal retreat",
            rotation,
            position_tolerance=0.025,
            angle_tolerance=np.deg2rad(4.0),
        )
        waypoints = [ArmWaypoint(current, "Beginning vertical clearance")]
        waypoints.extend(vertical)
        self.data.ctrl[self.picker.arm_actuators] = current
        self.picker._start_trajectory(waypoints)

    def _plan_carry_return(self) -> None:
        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        carry_position = carry_position_at(CARRY_POSITION, "home")
        carry_joints, pos_error, angle_error = ik.solve(
            carry_position, HOME_ARM_SEED.copy(), TOP_DOWN_ROTATION
        )
        if pos_error > 0.012 or angle_error > np.deg2rad(2.0):
            raise RuntimeError(
                "Could not solve the empty carry pose "
                f"({pos_error * 100:.1f} cm, {np.rad2deg(angle_error):.1f} deg)"
            )
        carry = [
            ArmWaypoint(joints, "Returning to empty carry pose")
            for joints in self.picker._joint_interpolation(current, carry_joints)
        ]
        waypoints = [ArmWaypoint(current, "Beginning carry return")]
        waypoints.extend(carry)
        self.data.ctrl[self.picker.arm_actuators] = current
        self.picker._start_trajectory(waypoints)

    def update(self) -> None:
        if self.mode in {"idle", "complete", "failed"}:
            return
        assert self.spec is not None

        if self.mode == "approach":
            finished, waypoint = self.picker._advance_trajectory(0.012)
            self._compensate_arm_tracking()
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self.status = f"Open {self.spec.label}: {waypoint.label}"
            if (
                not finished
                and self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            ):
                mujoco.mj_forward(self.model, self.data)
                target = (
                    self.data.site_xpos[self.handle_site_id]
                    - DRAWER_GRIP_ORIGIN_OFFSET
                    * DRAWER_FRONT_GRASP_ROTATION[:, 0]
                )
                distance = float(
                    np.linalg.norm(
                        self.data.site_xpos[self.picker.grip_site_id] - target
                    )
                )
                finished = distance < DRAWER_HANDLE_ARRIVAL_TOLERANCE
                if not finished:
                    self.approach_wait_ticks += 1
                    if self.approach_wait_ticks >= DRAWER_APPROACH_GRACE_TICKS:
                        self._fail(
                            "arm could not settle within finger reach of "
                            f"the handle ({distance * 100:.1f} cm)"
                        )
                        return
            if finished:
                self.mode = "closing"
                self.status = f"Open {self.spec.label}: closing until contact"
            return

        if self.mode == "closing":
            self.close_ticks += 1
            if self.close_ticks % 5 == 0:
                self.close_target = max(0.0, self.close_target - DRAWER_CLOSE_STEP)
            self.data.ctrl[self.picker.finger_actuators] = self.close_target
            if self._finger_contacts() == FINGER_GEOMS:
                self.contact_ticks += 1
                if self.contact_ticks >= DRAWER_CONTACT_TICKS:
                    self._activate_handle_connect()
                    try:
                        self._plan_pull()
                    except RuntimeError as error:
                        self._fail(str(error))
                        return
                    self.mode = "pulling"
                    self.status = f"Open {self.spec.label}: pulling horizontally"
                return
            self.contact_ticks = 0
            if self.close_target <= 0.0 and self.close_ticks > 300:
                self._fail("gripper closed without bilateral handle contact")
            return

        if self.mode == "pulling":
            self.picker._advance_trajectory(0.020)
            self._compensate_arm_tracking()
            target_slide = float(
                np.interp(
                    self.picker.trajectory_time,
                    self.picker.trajectory_times,
                    self.pull_values,
                )
            )
            self.data.ctrl[self.slide_actuator] = target_slide
            self.data.ctrl[self.picker.finger_actuators] = self.close_target
            self.status = (
                f"Open {self.spec.label}: pulling "
                f"{self.data.qpos[self.slide_qpos]:.2f}/{self.max_slide:.2f} m"
            )
            ended = self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            if ended and self.data.qpos[self.slide_qpos] >= self.max_slide - 0.015:
                self.release_arm_target = self.picker._current_arm()
                self.data.eq_active[self.equality_id] = 0
                self.data.ctrl[self.slide_actuator] = self.max_slide
                self.data.ctrl[self.picker.arm_actuators] = self.release_arm_target
                self.mode = "releasing"
                self.release_ticks = 0
                self.status = f"Open {self.spec.label}: releasing handle"
            return

        if self.mode == "releasing":
            self.release_ticks += 1
            self.data.ctrl[self.picker.arm_actuators] = self.release_arm_target
            self._compensate_arm_tracking()
            self.data.ctrl[self.slide_actuator] = self.max_slide
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            if self.release_ticks >= DRAWER_RELEASE_TICKS:
                try:
                    self._plan_horizontal_retreat()
                except RuntimeError as error:
                    self._fail(str(error))
                    return
                self.mode = "retreating"
                self.status = f"Open {self.spec.label}: retreating horizontally"
            return

        if self.mode == "retreating":
            finished, waypoint = self.picker._advance_trajectory(0.040)
            self._compensate_arm_tracking()
            self.data.ctrl[self.slide_actuator] = self.max_slide
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self.status = f"Open {self.spec.label}: {waypoint.label}"
            ended = self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            if finished or ended:
                try:
                    self._plan_vertical_clearance()
                except RuntimeError as error:
                    self._fail(str(error))
                    return
                self.mode = "clearing"
                self.status = f"Open {self.spec.label}: rising clear"
            return

        if self.mode == "clearing":
            finished, waypoint = self.picker._advance_trajectory(0.040)
            self._compensate_arm_tracking()
            self.data.ctrl[self.slide_actuator] = self.max_slide
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self.status = f"Open {self.spec.label}: {waypoint.label}"
            ended = self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            if finished or ended:
                try:
                    self._plan_carry_return()
                except RuntimeError as error:
                    self._fail(str(error))
                    return
                self.mode = "returning"
                self.status = f"Open {self.spec.label}: returning to carry pose"
            return

        if self.mode == "returning":
            finished, waypoint = self.picker._advance_trajectory(0.040)
            self._compensate_arm_tracking()
            self.data.ctrl[self.slide_actuator] = self.max_slide
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self.status = f"Open {self.spec.label}: {waypoint.label}"
            ended = self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            if finished or ended:
                self.mode = "complete"
                self.status = (
                    f"Open complete: {self.spec.label} fully open and gripper "
                    "at carry pose"
                )

    def progress(self) -> float:
        if self.mode == "approach":
            return 0.35 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "closing":
            return 0.40
        if self.mode == "pulling":
            return 0.45 + 0.30 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "releasing":
            return 0.75 + 0.05 * min(1.0, self.release_ticks / DRAWER_RELEASE_TICKS)
        if self.mode == "retreating":
            return 0.80 + 0.08 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "clearing":
            return 0.88 + 0.06 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "returning":
            return 0.94 + 0.06 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "complete":
            return 1.0
        return 0.0
