"""Contact-confirmed, hinge-centred opening motions for kitchen containers."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from mujoco_scenes.pick_motion import (
    FINGER_GEOMS,
    OPEN_WIDTH,
    ArmWaypoint,
    PickExecutor,
    VerticalIK,
)


BOX_HANDLE_GEOMS = {
    "B1_lid_handle_left",
    "B1_lid_handle_right",
    "B1_lid_handle_bar",
}
BOX_PREGRASP_DISTANCE = 0.075
BOX_OVERHEAD_CLEARANCE = 0.16
# The grip site sits behind the fingertips. Keeping its origin this far outside
# the metal bar puts the finger pads on the bar without driving their tips into
# B1's front wall.
BOX_GRIP_ORIGIN_OFFSET = 0.026
BOX_INSERT_RESOLUTION = 0.010
BOX_ARC_SAMPLES = 41
BOX_CLOSE_STEP = 0.001
BOX_CONTACT_TICKS = 12
BOX_RELEASE_TICKS = 90
BOX_FINAL_TOLERANCE = 0.025
BOX_ARM_TRACKING_COMPENSATION = 2.5
BOX_OPEN_TRACKING_WINDOW = 0.055

# Matrix columns are the gripper's local axes in the world frame.  Its +X
# approach axis points into the handle along world +Y, while the finger closing
# axis (local +Y) is vertical.  The pads therefore straddle the horizontal bar.
BOX_GRASP_ROTATION = np.array(
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a 3-D rotation matrix for a normalized axis and angle."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    return np.array(
        (
            (cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine),
            (y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine),
            (z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus),
        )
    )


class BoxOpenExecutor:
    """Open B1 by grasping its real handle and following its hinge arc."""

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
        self.status = "Open idle"
        self.failure: str | None = None
        self.close_target = OPEN_WIDTH
        self.close_ticks = 0
        self.contact_ticks = 0
        self.release_ticks = 0
        self.open_angles = np.zeros(0)
        self.lid_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "B1_lid"
        )
        self.handle_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "B1_lid_handle_grasp"
        )
        self.hinge_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "B1_lid_joint"
        )
        self.hinge_qpos = int(model.jnt_qposadr[self.hinge_joint_id])
        self.hinge_actuator = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "B1_lid_actuator"
        )
        self.grasp_equality_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, "robot0:open_weld_B1_lid"
        )
        self.max_angle = float(model.jnt_range[self.hinge_joint_id, 1])

    @property
    def busy(self) -> bool:
        return self.mode not in {"idle", "complete", "failed"}

    def _fail(self, message: str) -> None:
        if self.grasp_equality_id >= 0:
            self.data.eq_active[self.grasp_equality_id] = 0
        self.mode = "failed"
        self.failure = message
        self.status = f"Open failed: {message}"

    def _compensate_arm_tracking(self) -> None:
        """Counter the large sideways gravity load at the box-side pose."""
        command = self.data.ctrl[self.picker.arm_actuators].copy()
        live = self.data.qpos[self.picker.arm_qpos]
        compensated = command + BOX_ARM_TRACKING_COMPENSATION * (command - live)
        ranges = self.model.jnt_range[self.picker.arm_joint_ids]
        limited = self.model.jnt_limited[self.picker.arm_joint_ids].astype(bool)
        self.data.ctrl[self.picker.arm_actuators] = np.clip(
            compensated,
            np.where(limited, ranges[:, 0], -np.inf),
            np.where(limited, ranges[:, 1], np.inf),
        )

    def _handle_finger_contacts(self) -> set[str]:
        contacts: set[str] = set()
        for contact in self.data.contact:
            first_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            second_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            if first_name in BOX_HANDLE_GEOMS and second_name in FINGER_GEOMS:
                contacts.add(second_name)
            elif second_name in BOX_HANDLE_GEOMS and first_name in FINGER_GEOMS:
                contacts.add(first_name)
        return contacts

    def _activate_live_weld(self) -> None:
        if self.grasp_equality_id < 0:
            raise RuntimeError("Missing box-lid grasp constraint")
        gripper_id = self.picker.gripper_body_id
        inverse_pos = np.empty(3)
        inverse_quat = np.empty(4)
        relative_pos = np.empty(3)
        relative_quat = np.empty(4)
        mujoco.mju_negPose(
            inverse_pos,
            inverse_quat,
            self.data.xpos[gripper_id],
            self.data.xquat[gripper_id],
        )
        mujoco.mju_mulPose(
            relative_pos,
            relative_quat,
            inverse_pos,
            inverse_quat,
            self.data.xpos[self.lid_body_id],
            self.data.xquat[self.lid_body_id],
        )
        self.model.eq_data[self.grasp_equality_id, 3:6] = relative_pos
        self.model.eq_data[self.grasp_equality_id, 6:10] = relative_quat
        self.data.eq_active[self.grasp_equality_id] = 1

    def _lid_arc_samples(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Sample the real site transform at each value of the hinge joint."""
        sample_data = mujoco.MjData(self.model)
        sample_data.qpos[:] = self.data.qpos
        sample_data.qvel[:] = 0.0
        samples = []
        start_angle = float(self.data.qpos[self.hinge_qpos])
        mujoco.mj_forward(self.model, self.data)
        initial_lid_position = self.data.xpos[self.lid_body_id].copy()
        initial_lid_rotation = self.data.xmat[self.lid_body_id].reshape(3, 3).copy()
        initial_grip_position = self.data.site_xpos[self.picker.grip_site_id].copy()
        initial_grip_rotation = self.data.site_xmat[
            self.picker.grip_site_id
        ].reshape(3, 3).copy()
        for angle in np.linspace(start_angle, self.max_angle, BOX_ARC_SAMPLES):
            sample_data.qpos[self.hinge_qpos] = angle
            mujoco.mj_forward(self.model, sample_data)
            lid_rotation = sample_data.xmat[self.lid_body_id].reshape(3, 3).copy()
            lid_position = sample_data.xpos[self.lid_body_id].copy()
            delta_rotation = lid_rotation @ initial_lid_rotation.T
            rotation = delta_rotation @ initial_grip_rotation
            grip_position = (
                lid_position
                + delta_rotation @ (initial_grip_position - initial_lid_position)
            )
            samples.append(
                (
                    float(angle),
                    grip_position.copy(),
                    rotation,
                )
            )
        return samples

    def request_open(self, physical_location: str) -> None:
        if self.busy:
            raise RuntimeError("An open action is already running")
        if physical_location != "right_side":
            raise RuntimeError("Move (box) is required before opening B1")
        if self.picker.busy or self.picker.held_object is not None:
            raise RuntimeError("The gripper must be empty before opening B1")
        if self.data.qpos[self.hinge_qpos] >= self.max_angle - 0.05:
            self.mode = "complete"
            self.status = "Open complete: box lid is already at its limit"
            return

        mujoco.mj_forward(self.model, self.data)
        handle = self.data.site_xpos[self.handle_site_id].copy()
        grasp = handle - BOX_GRIP_ORIGIN_OFFSET * BOX_GRASP_ROTATION[:, 0]
        pregrasp = grasp - BOX_PREGRASP_DISTANCE * BOX_GRASP_ROTATION[:, 0]
        overhead = pregrasp + np.array((0.0, 0.0, BOX_OVERHEAD_CLEARANCE))
        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        overhead_joints, pos_error, angle_error = ik.solve(
            overhead, current, BOX_GRASP_ROTATION
        )
        if pos_error > 0.012 or angle_error > math.radians(2.0):
            raise RuntimeError(
                "Could not reach the box-handle overhead clearance "
                f"({pos_error * 100:.1f} cm, {math.degrees(angle_error):.1f} deg)"
            )
        waypoints = [ArmWaypoint(current, "Opening gripper")]
        waypoints.extend(
            ArmWaypoint(q, "Moving above the box handle")
            for q in self.picker._joint_interpolation(current, overhead_joints)
        )
        descent_points = self.picker._cartesian_points(
            overhead, pregrasp, 0.020
        )
        descent, pregrasp_joints = self.picker._solve_path(
            ik,
            descent_points,
            overhead_joints,
            "Descending to horizontal handle hover",
            BOX_GRASP_ROTATION,
            position_tolerance=0.012,
            angle_tolerance=math.radians(2.0),
        )
        waypoints.extend(descent)
        insertion_points = self.picker._cartesian_points(
            pregrasp, grasp, BOX_INSERT_RESOLUTION
        )
        insertion, _ = self.picker._solve_path(
            ik,
            insertion_points,
            pregrasp_joints,
            "Inserting along +Y around box handle",
            BOX_GRASP_ROTATION,
            position_tolerance=0.012,
            angle_tolerance=math.radians(2.0),
        )
        waypoints.extend(insertion)
        self.picker._start_trajectory(waypoints)
        self.data.ctrl[self.hinge_actuator] = float(self.data.qpos[self.hinge_qpos])
        self.close_target = OPEN_WIDTH
        self.close_ticks = 0
        self.contact_ticks = 0
        self.release_ticks = 0
        self.open_angles = np.zeros(0)
        self.failure = None
        self.mode = "approach"
        self.status = "Open box: approaching handle with vertical fingers"

    def _plan_opening_arc(self) -> None:
        samples = self._lid_arc_samples()
        ik = VerticalIK(self.model, self.data)
        current = self.picker._current_arm()
        waypoints = [ArmWaypoint(current, "Beginning hinge arc")]
        angles = [samples[0][0]]
        seed = current
        for angle, position, rotation in samples[1:]:
            seed, pos_error, angle_error = ik.solve(position, seed, rotation)
            # The sideways base pose leaves the last few degrees close to a
            # wrist limit. A small compliant IK residual is acceptable here:
            # the live handle weld and hinge still define the exact physical
            # arc, rather than prescribing a second incompatible lid pose.
            if pos_error > 0.030 or angle_error > math.radians(10.0):
                raise RuntimeError(
                    "Could not follow the box hinge at "
                    f"{math.degrees(angle):.0f} degrees "
                    f"({pos_error * 100:.1f} cm, {math.degrees(angle_error):.1f} deg)"
                )
            waypoints.append(ArmWaypoint(seed.copy(), "Opening around B1 hinge"))
            angles.append(angle)
        # End the hover compensation cleanly at the contact-confirmed live
        # joint state. Otherwise its over-command becomes a large synthetic
        # first knot and the tracking-gated hinge clock cannot begin.
        self.data.ctrl[self.picker.arm_actuators] = current
        self.picker._start_trajectory(waypoints)
        # Gravity compensation can leave the live actuator command slightly
        # ahead of qpos, in which case the shared helper prepends that command
        # as a smoothing knot. Keep the hinge stationary across any such knot.
        if len(self.picker.waypoints) > len(angles):
            angles = [angles[0]] * (len(self.picker.waypoints) - len(angles)) + angles
        self.open_angles = np.asarray(angles)

    def update(self) -> None:
        if self.mode in {"idle", "complete", "failed"}:
            return

        if self.mode == "approach":
            finished, waypoint = self.picker._advance_trajectory(0.003)
            self._compensate_arm_tracking()
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            self.status = f"Open box: {waypoint.label}"
            if (
                not finished
                and self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            ):
                # At this sideways reach the position actuators settle under
                # gravity a little outside their IK command. Accept that live
                # equilibrium only when the grip origin is still within the
                # physical finger-pad reach of the handle bar.
                mujoco.mj_forward(self.model, self.data)
                live_target = (
                    self.data.site_xpos[self.handle_site_id]
                    - BOX_GRIP_ORIGIN_OFFSET * BOX_GRASP_ROTATION[:, 0]
                )
                finished = (
                    np.linalg.norm(
                        self.data.site_xpos[self.picker.grip_site_id] - live_target
                    )
                    < 0.035
                )
            if finished:
                self.mode = "closing"
                self.status = "Open box: closing until both fingers touch handle"
            return

        if self.mode == "closing":
            self.close_ticks += 1
            if self.close_ticks % 5 == 0:
                self.close_target = max(0.0, self.close_target - BOX_CLOSE_STEP)
            self.data.ctrl[self.picker.finger_actuators] = self.close_target
            if self._handle_finger_contacts() == FINGER_GEOMS:
                self.contact_ticks += 1
                self.data.ctrl[self.picker.finger_actuators] = max(
                    0.0, self.close_target - 0.002
                )
                if self.contact_ticks >= BOX_CONTACT_TICKS:
                    try:
                        self._activate_live_weld()
                        self._plan_opening_arc()
                    except RuntimeError as error:
                        self._fail(str(error))
                        return
                    self.mode = "opening"
                    self.status = "Open box: contact confirmed, following hinge arc"
                return
            self.contact_ticks = 0
            if self.close_target <= 0.0 and self.close_ticks > 300:
                self._fail("gripper closed without bilateral handle contact")
            return

        if self.mode == "opening":
            previous_time = self.picker.trajectory_time
            finished, _ = self.picker._advance_trajectory(BOX_FINAL_TOLERANCE)
            reference = self.data.ctrl[self.picker.arm_actuators].copy()
            tracking_error = float(
                np.max(np.abs(self.data.qpos[self.picker.arm_qpos] - reference))
            )
            if tracking_error > BOX_OPEN_TRACKING_WINDOW:
                # Pause both clocks together until the physical arm catches
                # its reference. Advancing by wall time made the hinge motor
                # pull ahead of the hand and forced the weld to fight it.
                self.picker.trajectory_time = previous_time
                finished = False
            self._compensate_arm_tracking()
            target_angle = float(
                np.interp(
                    self.picker.trajectory_time,
                    self.picker.trajectory_times,
                    self.open_angles,
                )
            )
            self.data.ctrl[self.hinge_actuator] = target_angle
            self.data.ctrl[self.picker.finger_actuators] = self.close_target
            self.status = (
                "Open box: circular hinge motion "
                f"{math.degrees(self.data.qpos[self.hinge_qpos]):.0f}/"
                f"{math.degrees(self.max_angle):.0f} degrees"
            )
            trajectory_ended = (
                self.picker.trajectory_time >= self.picker.trajectory_times[-1]
            )
            if trajectory_ended and self.data.qpos[self.hinge_qpos] > self.max_angle - 0.12:
                self.data.eq_active[self.grasp_equality_id] = 0
                self.data.ctrl[self.hinge_actuator] = self.max_angle
                self.mode = "releasing"
                self.release_ticks = 0
                self.status = "Open box: releasing handle at maximum hinge angle"
            return

        if self.mode == "releasing":
            self.release_ticks += 1
            self.data.ctrl[self.hinge_actuator] = self.max_angle
            self.data.ctrl[self.picker.finger_actuators] = OPEN_WIDTH
            if self.release_ticks >= BOX_RELEASE_TICKS:
                self.mode = "complete"
                self.status = "Open complete: box lid is at its maximum hinge position"

    def progress(self) -> float:
        if self.mode == "approach":
            return 0.35 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "closing":
            return 0.40
        if self.mode == "opening":
            return 0.45 + 0.50 * self.picker.trajectory_time / max(
                1e-9, self.picker.trajectory_times[-1]
            )
        if self.mode == "releasing":
            return 0.95 + 0.05 * min(1.0, self.release_ticks / BOX_RELEASE_TICKS)
        if self.mode == "complete":
            return 1.0
        return 0.0
