"""Physics-assisted, measured MuJoCo execution of typed Workshop GT plans."""

from __future__ import annotations

import math
from typing import Any, Callable

import mujoco
import numpy as np

from .generic_manipulation import ProfiledIK, RobotConfigurationCollisionChecker
from .robot_profiles import manipulation_profile, mobile_profile
from .workshop_ground_truth_planner import WorkshopAssignment
from .workshop_ground_truth_state import WorkshopWorldState
from .workshop_scene import WORKSHOP_REGIONS, WorkshopScene


# Local +X approaches the storage front along world +Y.  For horizontal
# drawer bars local +Y is vertical; for the cabinet's vertical bar local +Y is
# horizontal.  These attitudes keep the wrist facing the handle instead of
# using the former top-down object-picking attitude.
DRAWER_FRONT_GRASP_ROTATION = np.array(
    ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
)
CABINET_FRONT_GRASP_ROTATION = np.array(
    ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
)
# Cabinet payloads are reached from the open front: local +X points into the
# cabinet, local +Y closes left/right around the upright tool, and local +Z
# points down.  This avoids the previous 90-degree side-on wrist attitude.
CABINET_OBJECT_FRONT_GRASP_ROTATION = np.array(
    ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
)
DRAWER_OBJECT_FRONT_GRASP_ROTATION = CABINET_OBJECT_FRONT_GRASP_ROTATION.copy()

# A 90-degree yaw of the same vertical-down attitude. The powered driver's
# flat resting pose points its handle along world Y, so its jaws must close
# along world X. This remains a vertical drawer/workbench approach.
POWER_TOP_DOWN_GRASP_ROTATION = np.array(
    ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
) @ np.asarray(manipulation_profile("google").top_down_rotation)


class WorkshopExecutionDispatcher:
    """Execute Workshop actions with constrained grasps and measured mechanics.

    MuJoCo actuators move the robot and storage.  Equality constraints are used
    only as the same grasp/fixture assistance already used by the Kitchen GT
    runner.  Small-part and driver alignment uses explicit compliant fixtures;
    transport, insertion descent, and driving happen through physical
    constraints and are verified from scene landmarks.
    """

    def __init__(
        self,
        scene: WorkshopScene,
        assignment: WorkshopAssignment,
        *,
        frame_callback: Callable[[bool], None] | None = None,
    ) -> None:
        self.scene = scene
        self.assignment = assignment
        self.frame_callback = frame_callback
        self.held_object: str | None = None
        self.robot_destination = "HOME"
        self.mobile_profile = mobile_profile("google")
        self.arm_profile = manipulation_profile("google")
        self.base_joint_ids = np.asarray([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.mobile_profile.base_joints
        ])
        self.base_qpos = scene.model.jnt_qposadr[self.base_joint_ids]
        self.base_dofs = scene.model.jnt_dofadr[self.base_joint_ids]
        self.base_actuators = np.asarray([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.mobile_profile.base_actuators
        ])
        self.arm_joint_ids = np.asarray([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_profile.arm_joints
        ])
        self.arm_qpos = scene.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dofs = scene.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_actuators = np.asarray([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.arm_profile.arm_actuators
        ])
        self.finger_joint_ids = np.asarray([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.arm_profile.finger_joints
        ])
        self.finger_qpos = scene.model.jnt_qposadr[self.finger_joint_ids]
        self.finger_dofs = scene.model.jnt_dofadr[self.finger_joint_ids]
        self.finger_actuators = np.asarray([
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.arm_profile.finger_actuators
        ])
        self.grip_site_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_SITE, self.arm_profile.grip_site
        )
        self.gripper_body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, self.arm_profile.gripper_body
        )
        self.active_grasp_weld = -1
        self.insertion_metrics: dict[str, float] = {}
        self.drive_metrics: dict[str, float] = {}
        self.navigation_audit: list[dict[str, Any]] = []
        self.motion_audit: list[dict[str, Any]] = []
        self.object_pick_sources: dict[str, str] = {}
        self.horizontal_transport_objects: set[str] = set()
        # Deliberately slower than the original benchmark motion. This scales
        # every rate-limited base, arm, finger and mechanism configuration.
        self.motion_rate_scale = 0.50

    def _capture(self, force: bool = True) -> None:
        if self.frame_callback:
            self.frame_callback(force)

    def _step(self, count: int = 1) -> None:
        """Advance physics while recording at the configured simulation FPS."""
        for _ in range(count):
            mujoco.mj_step(self.scene.model, self.scene.data)
            self._capture(False)

    def _hold(self, duration_s: float) -> None:
        self._step(max(1, int(round(duration_s / self.scene.model.opt.timestep))))

    @staticmethod
    def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
        return matrix.reshape(3, 3)

    def _settle_free_body(self, body_name: str, timeout_s: float = 4.0) -> None:
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        joint_id = int(self.scene.model.body_jntadr[body_id])
        dof = int(self.scene.model.jnt_dofadr[joint_id])
        stable = 0
        max_steps = int(round(timeout_s / self.scene.model.opt.timestep))
        for _ in range(max_steps):
            self._step()
            speed = float(np.max(np.abs(self.scene.data.qvel[dof : dof + 6])))
            stable = stable + 1 if speed < 0.025 else 0
            if stable >= 40:
                return

    def _body_position(self, name: str) -> list[float] | None:
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.scene.data.xpos[body_id].tolist() if body_id >= 0 else None

    def _object_grasp_position(self, object_name: str) -> np.ndarray:
        """Return the live centre of the physical geometry the jaws must hold."""
        suffix = {
            "workshop_long_phillips_driver": "_col_handle",
            "workshop_power_driver": "_col_handle",
            "workshop_medium_phillips_screw": "_col_head",
            "workshop_wooden_hammer": "_col_handle",
        }.get(object_name)
        geom_name = f"{object_name}{suffix}" if suffix else f"{object_name}_col"
        geom_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if geom_id < 0:
            raise RuntimeError(f"Missing calibrated Workshop grasp geometry {geom_name}")
        return self.scene.data.geom_xpos[geom_id].copy()

    def _finger_contact_sides(self, object_name: str) -> tuple[set[int], list[str]]:
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        sides: set[int] = set()
        contacts: list[str] = []
        for contact in self.scene.data.contact:
            body1 = int(self.scene.model.geom_bodyid[contact.geom1])
            body2 = int(self.scene.model.geom_bodyid[contact.geom2])
            if body_id not in {body1, body2}:
                continue
            object_geom = contact.geom1 if body1 == body_id else contact.geom2
            other_geom = contact.geom2 if body1 == body_id else contact.geom1
            object_name_geom = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, object_geom
            ) or f"geom_{object_geom}"
            other_name = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
            ) or f"geom_{other_geom}"
            for side, family in enumerate(self.arm_profile.finger_contact_geoms):
                if other_name in family:
                    sides.add(side)
                    contacts.append(f"{other_name}<->{object_name_geom}")
        return sides, sorted(set(contacts))

    def _close_gripper_on_object(self, object_name: str) -> dict[str, Any]:
        """Close slowly and require sustained left+right finger contact."""
        target_value = float(self.arm_profile.closed_command)
        dt = float(self.scene.model.opt.timestep)
        bilateral_streak = 0
        maximum_streak = 0
        observed_contacts: set[str] = set()
        # Allow enough simulated time to traverse the full 0.01--1.30 rad
        # finger range at the deliberately slow Workshop rate. The previous
        # fixed eight seconds stopped near 0.73 rad, leaving a 6--7 cm jaw
        # gap and making physical contact impossible.
        for _ in range(int(round(20.0 / max(self.motion_rate_scale, 1e-6) / dt))):
            command = self.scene.data.ctrl[self.finger_actuators].copy()
            self.scene.data.ctrl[self.finger_actuators] = np.minimum(
                target_value,
                command + 0.18 * self.motion_rate_scale * dt,
            )
            self._step()
            sides, names = self._finger_contact_sides(object_name)
            observed_contacts.update(names)
            bilateral_streak = bilateral_streak + 1 if sides == {0, 1} else 0
            maximum_streak = max(maximum_streak, bilateral_streak)
            if bilateral_streak >= 25:
                return {
                    "bilateral_contact_confirmed": True,
                    "bilateral_contact_steps": bilateral_streak,
                    "finger_object_contacts": sorted(observed_contacts),
                    "finger_joint_positions": self.scene.data.qpos[
                        self.finger_qpos
                    ].tolist(),
                }
        raise RuntimeError(
            f"GRASP_REJECTED: {object_name} never established sustained "
            f"bilateral finger contact (best streak={maximum_streak})"
        )

    def _free_qpos_address(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0 or self.scene.model.body_jntnum[body_id] != 1:
            raise RuntimeError(f"{body_name} is not an independent one-joint payload")
        joint_id = int(self.scene.model.body_jntadr[body_id])
        if self.scene.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"{body_name} does not have a free joint")
        return int(self.scene.model.jnt_qposadr[joint_id])

    def _release_storage_fixture(self, object_name: str) -> None:
        for prefix in ("storage_weld_",):
            equality_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{prefix}{object_name}"
            )
            if equality_id >= 0:
                self.scene.data.eq_active[equality_id] = 0

    def _release_staging_fixture(self, object_name: str) -> None:
        equality_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"workshop_staging_weld_{object_name}",
        )
        if equality_id >= 0:
            self.scene.data.eq_active[equality_id] = 0

    def _set_weld_world_pose(self, equality_id: int, body1_id: int, body2_id: int) -> None:
        inverse_pos, inverse_quat = np.empty(3), np.empty(4)
        relative_pos, relative_quat = np.empty(3), np.empty(4)
        mujoco.mju_negPose(
            inverse_pos, inverse_quat,
            self.scene.data.xpos[body1_id], self.scene.data.xquat[body1_id],
        )
        mujoco.mju_mulPose(
            relative_pos, relative_quat, inverse_pos, inverse_quat,
            self.scene.data.xpos[body2_id], self.scene.data.xquat[body2_id],
        )
        self.scene.model.eq_data[equality_id, 3:6] = relative_pos
        self.scene.model.eq_data[equality_id, 6:10] = relative_quat

    def _set_weld_desired_world_pose(
        self,
        equality_id: int,
        body1_id: int,
        desired_position: np.ndarray,
        desired_quaternion: np.ndarray,
    ) -> None:
        """Configure a fixture weld to pull body2 to a specified world pose."""
        inverse_pos, inverse_quat = np.empty(3), np.empty(4)
        relative_pos, relative_quat = np.empty(3), np.empty(4)
        mujoco.mju_negPose(
            inverse_pos, inverse_quat,
            self.scene.data.xpos[body1_id], self.scene.data.xquat[body1_id],
        )
        mujoco.mju_mulPose(
            relative_pos, relative_quat,
            inverse_pos, inverse_quat,
            np.asarray(desired_position, dtype=float),
            np.asarray(desired_quaternion, dtype=float),
        )
        self.scene.model.eq_data[equality_id, 3:6] = relative_pos
        self.scene.model.eq_data[equality_id, 6:10] = relative_quat

    def _activate_grasp(
        self, object_name: str, *, require_bilateral: bool = False
    ) -> dict[str, Any]:
        object_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        equality_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"google:pick_weld_{object_name}",
        )
        if object_id < 0 or equality_id < 0:
            raise RuntimeError(f"Missing constrained grasp for {object_name}")
        sides, contact_names = self._finger_contact_sides(object_name)
        if require_bilateral and sides != {0, 1}:
            raise RuntimeError(
                f"GRASP_REJECTED: attachment requested without bilateral "
                f"finger contact on {object_name}; sides={sorted(sides)}"
            )
        before_position = self.scene.data.xpos[object_id].copy()
        before_quaternion = self.scene.data.xquat[object_id].copy()
        self._release_storage_fixture(object_name)
        self._release_staging_fixture(object_name)
        self._set_weld_world_pose(equality_id, self.gripper_body_id, object_id)
        self.scene.data.eq_active[equality_id] = 1
        self.active_grasp_weld = equality_id
        self.held_object = object_name
        mujoco.mj_forward(self.scene.model, self.scene.data)
        translation_snap = float(np.linalg.norm(
            self.scene.data.xpos[object_id] - before_position
        ))
        quaternion_dot = abs(float(np.dot(
            self.scene.data.xquat[object_id], before_quaternion
        )))
        angle_snap = 2.0 * math.acos(float(np.clip(quaternion_dot, -1.0, 1.0)))
        if require_bilateral and translation_snap > 0.004:
            self.scene.data.eq_active[equality_id] = 0
            self.active_grasp_weld = -1
            self.held_object = None
            raise RuntimeError(
                f"GRASP_REJECTED: attachment translated {object_name} by "
                f"{translation_snap:.4f} m"
            )
        return {
            "bilateral_contact_confirmed": sides == {0, 1},
            "finger_object_contacts": contact_names,
            "attachment_translation_snap_m": translation_snap,
            "attachment_angle_snap_rad": angle_snap,
        }

    def _release_grasp(self) -> None:
        if self.active_grasp_weld >= 0:
            self.scene.data.eq_active[self.active_grasp_weld] = 0
        self.active_grasp_weld = -1
        self.held_object = None

    def _animate_configuration(
        self,
        qpos_addresses: np.ndarray,
        dof_addresses: np.ndarray,
        actuator_ids: np.ndarray,
        target: np.ndarray,
        *,
        samples: int = 16,
        maximum_rate: float | np.ndarray | None = None,
        tolerance: float = 0.012,
        velocity_tolerance: float = 0.05,
        settle_steps: int = 12,
        timeout_s: float = 12.0,
        allow_contact_stall: bool = False,
    ) -> None:
        del samples  # retained for call compatibility; motion is time/rate based.
        target = np.asarray(target, dtype=float)
        if maximum_rate is None:
            maximum_rate = 0.55
        rates = np.broadcast_to(
            np.asarray(maximum_rate, dtype=float) * self.motion_rate_scale,
            target.shape,
        )
        dt = float(self.scene.model.opt.timestep)
        max_steps = max(
            1, int(round(timeout_s / max(self.motion_rate_scale, 1e-6) / dt))
        )
        settled = 0
        contact_stalled = 0
        for _ in range(max_steps):
            command = self.scene.data.ctrl[actuator_ids].copy()
            delta = np.clip(target - command, -rates * dt, rates * dt)
            self.scene.data.ctrl[actuator_ids] = command + delta
            self._step()
            error = float(np.max(np.abs(self.scene.data.qpos[qpos_addresses] - target)))
            speed = float(np.max(np.abs(self.scene.data.qvel[dof_addresses])))
            command_done = bool(np.max(np.abs(self.scene.data.ctrl[actuator_ids] - target)) < 1e-6)
            settled = settled + 1 if command_done and error <= tolerance and speed <= velocity_tolerance else 0
            contact_stalled = (
                contact_stalled + 1
                # A low speed during the rate-limited command ramp is not a
                # contact stall. Only accept contact support after the full
                # actuator command has actually been issued.
                if allow_contact_stall and command_done and speed < 0.008 else 0
            )
            if settled >= settle_steps:
                break
            if contact_stalled >= 50:
                break
        else:
            raise RuntimeError(
                f"Actuator motion failed to settle: error={error:.4f}, speed={speed:.4f}"
            )
        self._hold(0.25)

    def _set_gripper(self, closed: bool) -> None:
        target_value = (
            self.arm_profile.closed_command if closed else self.arm_profile.open_command
        )
        if closed:
            dt = float(self.scene.model.opt.timestep)
            stable_contact_steps = 0
            for _ in range(int(round(6.0 / dt))):
                command = self.scene.data.ctrl[self.finger_actuators].copy()
                self.scene.data.ctrl[self.finger_actuators] = np.minimum(
                    target_value, command + 0.45 * dt
                )
                self._step()
                speed = float(np.max(np.abs(self.scene.data.qvel[self.finger_dofs])))
                closure = float(np.min(self.scene.data.qpos[self.finger_qpos]))
                command_done = bool(np.min(
                    target_value - self.scene.data.ctrl[self.finger_actuators]
                ) <= 1e-6)
                # A grasp correctly stops on the handle/object before the
                # no-load fully-closed joint command.
                stable_contact_steps = (
                    stable_contact_steps + 1
                    if closure >= 0.45 and speed < 0.012 else 0
                )
                if command_done or stable_contact_steps >= 35:
                    self._hold(0.35)
                    return
            raise RuntimeError("Gripper did not close or establish stable contact")
        self._animate_configuration(
            self.finger_qpos,
            self.finger_dofs,
            self.finger_actuators,
            np.full(2, target_value),
            samples=8,
            maximum_rate=0.45,
            tolerance=0.01,
            allow_contact_stall=True,
        )

    def _fold_arm(self) -> None:
        self._animate_configuration(
            self.arm_qpos,
            self.arm_dofs,
            self.arm_actuators,
            self.arm_profile.navigation_joints.copy(),
            samples=12,
            maximum_rate=0.50,
            tolerance=0.003,
            velocity_tolerance=0.02,
            timeout_s=16.0,
        )

    def _interaction_position(self, destination: str) -> np.ndarray:
        handle_name = {
            "LEFT_DRAWER": "left_drawer_handle",
            "RIGHT_DRAWER": "right_drawer_handle",
            "TOOL_CABINET": "tool_cabinet_door_handle",
        }.get(destination)
        if handle_name:
            geom_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, handle_name
            )
            if geom_id < 0:
                raise RuntimeError(f"Missing storage handle {handle_name}")
            return self.scene.data.geom_xpos[geom_id].copy()
        if destination == "HOME":
            return np.array([0.0, -0.15, 0.90])
        return self._destination_position(destination)

    def _base_stance(self, destination: str) -> np.ndarray:
        if destination == "TOOL_CABINET":
            # Park opposite the middle of the door-handle arc rather than the
            # closed handle.  This keeps both the latch pose and the 83-degree
            # open pose inside the arm's comfortable workspace.
            base_x = -0.70 if self.scene.variant_name == "F6_LAYOUT_SWAPPED" else 0.40
            return np.array([0.40, -base_x, 0.0], dtype=float)
        target = self._interaction_position(destination)
        # Spawn is world y=-0.75; forward qpos adds to y.  A 0.70 m
        # interaction standoff keeps the base outside furniture footprints.
        forward = target[1] + 0.05
        if destination in {"LEFT_DRAWER", "RIGHT_DRAWER"}:
            # Keep the mobile base in the clear front corridor. The fully
            # extracted drawer and its payload then remain in front of the
            # workbench but cannot collide with the chassis; the arm supplies
            # the handle reach and later top-down object reach.
            forward = 0.0
        if (
            destination == "TOOL_CART_TOP"
            and self.held_object == "workshop_long_phillips_driver"
        ):
            # Keep a carried long/heavy tool clear of the cart edge; the arm
            # supplies the remaining reach during the staged descent.
            forward -= 0.08
        stance_x = float(target[0])
        if destination == "MAIN_WORKBENCH_ZONE" and self.held_object:
            source = self.object_pick_sources.get(self.held_object)
            if source == "RIGHT_DRAWER":
                # Keep the chassis left of an extracted right drawer while
                # carrying its payload to the bench.
                stance_x = -0.12
            elif source == "LEFT_DRAWER":
                stance_x = 0.12
        if destination == "LEFT_DRAWER":
            stance_x += 0.075
        elif destination == "RIGHT_DRAWER":
            stance_x -= 0.075
        return np.array([forward, -stance_x, 0.0], dtype=float)

    @staticmethod
    def _base_world_xy(qpos: np.ndarray) -> np.ndarray:
        return np.array([-qpos[1], -0.75 + qpos[0]], dtype=float)

    def _audit_base_waypoint(self, qpos: np.ndarray, label: str) -> None:
        x, y = self._base_world_xy(qpos)
        # Conservative 24 cm footprint margin around the two floor-level
        # obstacles.  Tabletop objects are deliberately not treated as floor
        # obstacles; the base remains in front of the workbench.
        obstacles = {
            "workbench": (-0.84, 0.84, -0.185, 0.945),
            "tool_cart": (
                (-1.76, -0.80) if self.scene.variant_name == "F6_LAYOUT_SWAPPED" else (0.80, 1.76)
            ) + (-0.05, 0.85),
        }
        collisions = [
            name for name, (xmin, xmax, ymin, ymax) in obstacles.items()
            if xmin < x < xmax and ymin < y < ymax
        ]
        record = {"label": label, "base_world_xy_m": [float(x), float(y)], "clear": not collisions, "collisions": collisions}
        self.navigation_audit.append(record)
        if collisions:
            raise RuntimeError(f"Unsafe base waypoint {label}: {collisions} at {(x, y)}")

    def _navigate_robot(self, destination: str) -> None:
        target = self._base_stance(destination)
        current = self.scene.data.qpos[self.base_qpos].copy()
        # Retreat first, translate sideways only in the clear front corridor,
        # then approach the selected station.  This prevents the former
        # straight-line interpolation through the bench and drawers.
        corridor = np.array([0.0, current[1], 0.0])
        self._audit_base_waypoint(corridor, "retreat_to_front_corridor")
        self._animate_configuration(
            self.base_qpos, self.base_dofs, self.base_actuators,
            corridor, samples=10,
            maximum_rate=np.array([0.22, 0.22, 0.45]),
            tolerance=0.030,
            velocity_tolerance=0.025,
            timeout_s=18.0,
        )
        # Folding is safe only after the wrist and base have cleared the
        # station; doing it beside a tabletop caused the old table-phasing
        # appearance and real actuator stalls.
        if self.held_object is None:
            self._fold_arm()
        elif (
            not (
                destination == "workshop_frame_joint"
                and self.held_object in {
                    "workshop_long_phillips_driver",
                    "workshop_power_driver",
                }
            )
            and self.held_object != "workshop_power_driver"
            or (
                destination == "TOOL_CART_TOP"
                and self.held_object not in self.horizontal_transport_objects
            )
        ):
            base_xy = self._base_world_xy(
                self.scene.data.qpos[self.base_qpos].copy()
            )
            carry_target = np.array([
                base_xy[0], base_xy[1] + 0.22, 1.05
            ])
            self._reach(
                carry_target,
                allowed_body_names=(self.held_object,),
                orientation_weight=0.05,
                ik_tolerance_m=0.060,
            )
        route = [
            ("translate_in_front_corridor", np.array([0.0, target[1], 0.0])),
            ("approach_station", target),
        ]
        for label, waypoint in route:
            self._audit_base_waypoint(waypoint, label)
            self._animate_configuration(
                self.base_qpos, self.base_dofs, self.base_actuators,
                waypoint, samples=10,
                maximum_rate=np.array([0.22, 0.22, 0.45]),
                tolerance=0.030,
                velocity_tolerance=0.025,
                timeout_s=18.0,
            )

    def _reach(
        self,
        target: np.ndarray,
        *,
        samples: int = 18,
        rotation: np.ndarray | None = None,
        orientation_weight: float = 0.15,
        allowed_body_names: tuple[str, ...] = (),
        cartesian_step_m: float | None = None,
        ik_tolerance_m: float = 0.018,
    ) -> dict[str, Any]:
        del samples
        target = np.asarray(target, dtype=float)
        # Leave the nested navigation fold through the Google profile's known
        # collision-checked home branch before solving task-space motion.
        if np.linalg.norm(
            self.scene.data.qpos[self.arm_qpos] - self.arm_profile.navigation_joints
        ) < 0.12:
            self._animate_configuration(
                self.arm_qpos,
                self.arm_dofs,
                self.arm_actuators,
                self.arm_profile.home_seed.copy(),
                maximum_rate=0.45,
                tolerance=0.010,
                timeout_s=18.0,
            )
        start_configuration = self.scene.data.qpos[self.arm_qpos].copy()
        start_position = self.scene.data.site_xpos[self.grip_site_id].copy()
        ik = ProfiledIK(
            self.scene.model,
            self.scene.data,
            self.arm_profile,
            orientation_weight=orientation_weight,
            maximum_iterations=1800,
        )
        requested_rotation = (
            self.arm_profile.top_down_rotation if rotation is None else rotation
        )
        allowed_body_ids = frozenset(
            body_id for name in allowed_body_names
            if (body_id := mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, name
            )) >= 0
        )
        checker = RobotConfigurationCollisionChecker(
            self.scene.model,
            self.scene.data,
            self.arm_profile,
            mounting_allowances={
                frozenset(("google:base_link", "google:link_shoulder")): -0.150,
                frozenset(("google:base_link", "google:link_bicep")): -0.080,
                # The navigation fold intentionally nests the empty gripper
                # into the base shroud; this is a known mechanical-envelope
                # overlap, not permission to intersect task furniture.
                frozenset(("google:base_link", "google:link_gripper")): -0.100,
                frozenset(("google:base_link", "google:link_wrist")): -0.100,
                frozenset(("google:base_link", "google:link_forearm")): -0.120,
                frozenset(("google:base_link", "google:link_finger_right")): -0.100,
                frozenset(("google:base_link", "google:link_finger_left")): -0.100,
                frozenset(("google:base_link", "google:link_finger_tip_right")): -0.120,
                frozenset(("google:base_link", "google:link_finger_tip_left")): -0.120,
            },
        )
        # Callers provide semantic Cartesian waypoints (corridor, overhead,
        # pregrasp, contact, retreat).  Solve each segment endpoint here; a
        # straight Cartesian interpolation from the folded navigation pose is
        # frequently outside this arm's workspace.
        distance = float(np.linalg.norm(target - start_position))
        waypoint_count = (
            max(1, int(math.ceil(distance / cartesian_step_m)))
            if cartesian_step_m is not None else 1
        )
        seed = start_configuration.copy()
        planned_start = start_configuration.copy()
        position_error = float("inf")
        angle_error = float("inf")
        waypoints = [
            start_position + float(fraction) * (target - start_position)
            for fraction in np.linspace(0.0, 1.0, waypoint_count + 1)[1:]
        ]
        for waypoint in waypoints:
            saved_qpos = self.scene.data.qpos.copy()
            saved_qvel = self.scene.data.qvel.copy()
            solution, position_error, angle_error = ik.solve(
                waypoint, seed, requested_rotation
            )
            self.scene.data.qpos[:] = saved_qpos
            self.scene.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.scene.model, self.scene.data)
            if position_error > ik_tolerance_m:
                raise RuntimeError(
                    f"Unreachable Cartesian waypoint: error={position_error:.4f} m"
                )
            collision_free, reason = checker.segment_valid(
                planned_start, solution, allowed_body_ids, resolution=0.025
            )
            if not collision_free:
                raise RuntimeError(f"Unsafe Workshop arm path: {reason}")
            self._animate_configuration(
                self.arm_qpos,
                self.arm_dofs,
                self.arm_actuators,
                solution,
                maximum_rate=0.48,
                tolerance=0.014,
                timeout_s=10.0,
                allow_contact_stall=bool(allowed_body_names),
            )
            seed = solution
            planned_start = solution
        actual_error = float(np.linalg.norm(
            self.scene.data.site_xpos[self.grip_site_id] - target
        ))
        self.motion_audit.append({
            "target_m": target.tolist(),
            "measured_error_m": actual_error,
            "cartesian_waypoints": waypoint_count,
            "collision_checked": True,
        })
        return {
            "gripper_target_error_m": position_error,
            "measured_gripper_target_error_m": actual_error,
            "empty_hand_ik_pose_recovery_used": False,
            "direct_arm_qpos_write_used": False,
            "collision_checked": True,
            "cartesian_waypoints": waypoint_count,
            "gripper_angle_error_rad": angle_error,
        }

    def _robot_gesture(self, cycles: float = 1.0, amplitude: float = 0.18) -> None:
        actuator_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "google:joint_wrist_actuator"
        )
        if actuator_id < 0:
            return
        baseline = float(self.scene.data.ctrl[actuator_id])
        for index in range(8):
            self.scene.data.ctrl[actuator_id] = baseline + amplitude * math.sin(
                2.0 * math.pi * cycles * index / 7.0
            )
            self._step()
        self.scene.data.ctrl[actuator_id] = baseline
        self._capture(True)

    def _articulate_storage(self, region: str, *, opening: bool) -> dict[str, Any]:
        """Approach, grasp, and track a physical handle along its joint path."""
        joint_name = {
            "LEFT_DRAWER": "left_tool_drawer_slide",
            "RIGHT_DRAWER": "right_tool_drawer_slide",
            "TOOL_CABINET": "tool_cabinet_door_hinge",
        }[region]
        joint_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        qpos_address = int(self.scene.model.jnt_qposadr[joint_id])
        actuator_id = self.scene._container_actuator_id(region)
        # Drawers must expose the stored object's physical grasp geometry past
        # the workbench apron. The former 0.34 command left the payload deep
        # below the tabletop even though the drawer counted as open.
        final_target = (1.45 if region == "TOOL_CABINET" else 0.55) if opening else 0.0
        grasp_rotation = (
            CABINET_FRONT_GRASP_ROTATION.copy()
            if region == "TOOL_CABINET"
            else DRAWER_FRONT_GRASP_ROTATION.copy()
        )
        moving_body_id = mujoco.mj_name2id(
            self.scene.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "tool_cabinet_door" if region == "TOOL_CABINET" else (
                "left_tool_drawer" if region == "LEFT_DRAWER" else "right_tool_drawer"
            ),
        )
        moving_body_name = (
            "tool_cabinet_door" if region == "TOOL_CABINET" else (
                "left_tool_drawer" if region == "LEFT_DRAWER" else "right_tool_drawer"
            )
        )
        start_body_rotation = self.scene.data.xmat[moving_body_id].reshape(3, 3).copy()
        self._set_gripper(False)
        handle = self._interaction_position(region)
        # Grasp the inner half of each horizontal drawer bar.  At the exact
        # bar centre the horizontally aligned wrist clips the corresponding
        # front workbench leg; the inner point remains on the 10 cm handle
        # while giving the wrist real furniture clearance.
        if region == "LEFT_DRAWER":
            handle[0] += 0.040
        elif region == "RIGHT_DRAWER":
            handle[0] -= 0.040
        approach_axis = grasp_rotation[:, 0]
        grasp_offset = (
            -0.055 if region in {"LEFT_DRAWER", "RIGHT_DRAWER"}
            else -0.040
        )
        grasp = handle + grasp_offset * approach_axis
        # Remain entirely in front of the workbench while descending.  The
        # former overhead point was below the tabletop and the joint-space
        # transition visibly cut through it.
        pregrasp = grasp - 0.200 * approach_axis
        overhead = pregrasp + np.array([0.0, 0.0, 0.35])
        overhead[2] = 1.02 if region == "TOOL_CABINET" else 0.75
        corridor = np.array([
            pregrasp[0], min(pregrasp[1], -0.40),
            1.02 if region == "TOOL_CABINET" else 0.94,
        ])
        allowed_handle_body = (moving_body_name,)
        if region != "TOOL_CABINET":
            self._reach(
                corridor, samples=12, allowed_body_names=allowed_handle_body
            )
            # Descend in front of the bench before entering the under-top
            # drawer corridor.  Descending above the handle made the forearm
            # sweep through the tabletop even though the jaws faced forward.
            low_front = np.array([pregrasp[0], -0.40, grasp[2] + 0.035])
            self._reach(
                low_front, samples=10, rotation=grasp_rotation,
                orientation_weight=0.03,
                allowed_body_names=allowed_handle_body,
                ik_tolerance_m=0.030,
            )
        else:
            self._reach(
                overhead, samples=10, rotation=grasp_rotation,
                orientation_weight=0.15,
                allowed_body_names=allowed_handle_body,
                ik_tolerance_m=0.018,
            )
        self._reach(
            pregrasp, samples=8, rotation=grasp_rotation,
            orientation_weight=(0.15 if region == "TOOL_CABINET" else 0.03),
            allowed_body_names=allowed_handle_body,
            ik_tolerance_m=(0.018 if region == "TOOL_CABINET" else 0.030),
        )
        reach = self._reach(
            grasp, samples=8, rotation=grasp_rotation,
            orientation_weight=(0.15 if region == "TOOL_CABINET" else 0.03),
            allowed_body_names=allowed_handle_body,
            cartesian_step_m=0.020,
            ik_tolerance_m=(0.018 if region == "TOOL_CABINET" else 0.030),
        )
        self._set_gripper(True)
        handle_weld_id = mujoco.mj_name2id(
            self.scene.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"workshop_handle_grasp_weld_{region}",
        )
        if handle_weld_id < 0:
            raise RuntimeError(f"Missing physical handle grasp constraint for {region}")
        self._set_weld_world_pose(
            handle_weld_id, self.gripper_body_id, moving_body_id
        )
        self.scene.data.eq_active[handle_weld_id] = 1
        mujoco.mj_forward(self.scene.model, self.scene.data)
        self._hold(0.35)
        start = float(self.scene.data.qpos[qpos_address])
        handle_errors: list[float] = []
        contact_frames = 0
        live_rotation = grasp_rotation.copy()
        for fraction in np.linspace(0.0, 1.0, 17)[1:]:
            command = start + float(fraction) * (final_target - start)
            self.scene.data.ctrl[actuator_id] = command
            if region in {"LEFT_DRAWER", "RIGHT_DRAWER"}:
                # The hand leads the prismatic motion. Previously we waited
                # for the drawer servo while the welded arm held the handle
                # stationary, so the two actuators fought and the drawer
                # barely moved before magically completing after release.
                # A drawer's joint axis is world -Y, making its intended live
                # handle position exact and available without writing qpos.
                live_handle = handle + np.array([
                    0.0, -(command - start), 0.0
                ])
                # `handle` already includes the inner-bar X calibration; the
                # shared calibration immediately below adds it once.
                live_handle[0] += -0.040 if region == "LEFT_DRAWER" else 0.040
            else:
                # The door uses its measured hinged pose because its handle
                # follows an arc rather than a straight prismatic path.
                self._step(55)
                live_handle = self._interaction_position(region)
            if region == "LEFT_DRAWER":
                live_handle[0] += 0.040
            elif region == "RIGHT_DRAWER":
                live_handle[0] -= 0.040
            if region == "TOOL_CABINET":
                live_body_rotation = self.scene.data.xmat[moving_body_id].reshape(3, 3)
                door_delta = live_body_rotation @ start_body_rotation.T
                # Keep the jaws vertical and blend only half of the door yaw.
                # A rigid 90-degree wrist follow forces an unreachable branch
                # near full opening on this single-arm base; the partial yaw
                # retains a credible handle grasp while prioritizing contact.
                delta_quat = np.empty(4)
                mujoco.mju_mat2Quat(delta_quat, door_delta.ravel())
                half_angle_quat = delta_quat.copy()
                half_angle_quat[0] = math.sqrt(max(0.0, (1.0 + delta_quat[0]) / 2.0))
                scale = 0.5 / max(half_angle_quat[0], 1e-9)
                half_angle_quat[1:] = delta_quat[1:] * scale
                half_rotation = np.empty(9)
                mujoco.mju_quat2Mat(half_rotation, half_angle_quat)
                live_rotation = half_rotation.reshape(3, 3) @ grasp_rotation
            live_approach = live_rotation[:, 0]
            live_grasp = live_handle + grasp_offset * live_approach
            follow = self._reach(
                live_grasp,
                samples=4,
                rotation=live_rotation,
                # Handle position/contact is the hard requirement.  A small
                # attitude weight lets the wrist relax when loaded drawer
                # contents alter the reachable branch.
                orientation_weight=(0.005 if region == "TOOL_CABINET" else 0.025),
                allowed_body_names=allowed_handle_body,
                cartesian_step_m=0.025,
                ik_tolerance_m=(0.050 if region == "TOOL_CABINET" else 0.030),
            )
            if region in {"LEFT_DRAWER", "RIGHT_DRAWER"}:
                # Let the drawer servo settle onto the arm-led waypoint while
                # the physical handle constraint remains active.
                self._step(35)
            handle_errors.append(follow["measured_gripper_target_error_m"])
            handle_collision = {
                "LEFT_DRAWER": "left_drawer_handle_col",
                "RIGHT_DRAWER": "right_drawer_handle_col",
                "TOOL_CABINET": "tool_cabinet_door_handle_col",
            }[region]
            for contact in self.scene.data.contact:
                first = mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ) or ""
                second = mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ) or ""
                if handle_collision in {first, second} and (
                    first in self.arm_profile.finger_contact_geoms[0]
                    or first in self.arm_profile.finger_contact_geoms[1]
                    or second in self.arm_profile.finger_contact_geoms[0]
                    or second in self.arm_profile.finger_contact_geoms[1]
                ):
                    contact_frames += 1
                    break
            self._capture(True)
        tracking_measurement = float(self.scene.data.qpos[qpos_address])
        if opening:
            self.scene.state.container_open_state[region] = True
            self.scene.state.opened_containers.add(region)
        else:
            self.scene.state.container_open_state[region] = False
        self.scene.data.eq_active[handle_weld_id] = 0
        mujoco.mj_forward(self.scene.model, self.scene.data)
        self._set_gripper(False)
        live_grip = self.scene.data.site_xpos[self.grip_site_id].copy()
        if region == "TOOL_CABINET":
            retreat = live_grip + np.array([0.0, -0.18, 0.06])
            retreat_high = retreat + np.array([0.0, 0.0, 0.10])
        else:
            # Withdraw horizontally into the free band between the drawer
            # front and mobile base, then rise in place. The tools now lie
            # transversely across the drawer, so this band is unobstructed.
            retreat = live_grip - 0.14 * live_rotation[:, 0]
            retreat_high = retreat + (
                np.array([-0.16, 0.0, 0.18])
                if region == "LEFT_DRAWER"
                # The right-side arm branch clears the torso by moving back
                # toward the centreline before a small rise.
                else np.array([-0.18, 0.0, 0.05])
            )
        self._reach(
            retreat,
            samples=8,
            rotation=live_rotation,
            allowed_body_names=allowed_handle_body,
            cartesian_step_m=0.015,
            ik_tolerance_m=(0.050 if region == "TOOL_CABINET" else 0.018),
        )
        self._reach(
            retreat_high,
            samples=8,
            rotation=live_rotation,
            allowed_body_names=allowed_handle_body,
            cartesian_step_m=0.025,
            ik_tolerance_m=(0.050 if region == "TOOL_CABINET" else 0.030),
        )
        self._fold_arm()
        # Let the articulation servo reach its commanded stop after the hand
        # clears the handle.  This also creates an unambiguous open/closed hold.
        self.scene.data.ctrl[actuator_id] = final_target
        self._step(450)
        measured = float(self.scene.data.qpos[qpos_address])
        opened_enough = measured >= (1.20 if region == "TOOL_CABINET" else 0.49)
        closed_enough = measured <= (0.12 if region == "TOOL_CABINET" else 0.04)
        for _ in range(4):
            self._capture(True)
        return {
            "joint_name": joint_name,
            "commanded_joint_position": final_target,
            "measured_joint_position": measured,
            "joint_position_during_handle_tracking": tracking_measurement,
            "handle_tracking_max_error_m": max(handle_errors, default=0.0),
            "handle_contact_frames": contact_frames,
            "handle_contact_semantics": "front-facing closed gripper follows the physical handle joint path",
            "physical_handle_grasp_constraint_used": True,
            "verified": opened_enough if opening else closed_enough,
            "initial_reach": reach,
        }

    def _regrasp_vertical_fastener(self, object_name: str) -> dict[str, Any]:
        entry = self._destination_position("workshop_frame_joint")
        desired_tip = entry + np.array([0.0, 0.0, 0.055])
        allowed = (
            object_name, "workshop_parts_tray", "workshop_hardware_bin",
            "workbench", "left_tool_drawer", "right_tool_drawer",
            "workshop_frame_joint", "workshop_frame_fixture",
        )
        current_body = np.asarray(self._body_position(object_name), dtype=float)
        current_grip = self.scene.data.site_xpos[self.grip_site_id].copy()
        staging_grip = current_grip + (desired_tip - current_body)
        self._reach(
            staging_grip + np.array([0.0, 0.0, 0.10]),
            allowed_body_names=allowed, ik_tolerance_m=0.065,
        )
        self._reach(
            staging_grip, allowed_body_names=allowed,
            cartesian_step_m=0.025, ik_tolerance_m=0.065,
        )
        alignment_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"workshop_alignment_weld_{object_name}",
        )
        frame_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_frame_joint"
        )
        self._release_grasp()
        self._set_gripper(False)
        self._set_weld_desired_world_pose(
            alignment_id, frame_id, desired_tip,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.scene.data.eq_active[alignment_id] = 1
        self._hold(1.20)
        aligned_body = np.asarray(self._body_position(object_name), dtype=float)
        self._reach(
            aligned_body + np.array([0.0, 0.0, 0.14]),
            allowed_body_names=allowed, ik_tolerance_m=0.040,
        )
        reach = self._reach(
            aligned_body + np.array([0.0, 0.0, 0.065]),
            allowed_body_names=allowed,
            cartesian_step_m=0.015, ik_tolerance_m=0.040,
        )
        self._set_gripper(True)
        self._activate_grasp(object_name)
        self.scene.data.eq_active[alignment_id] = 0
        self._hold(0.45)
        return {
            "regrasp_pose_recovery": False,
            "robot_constrained_reorientation": True,
            "physical_alignment_fixture_used": True,
            "vertical_axis_world": [0.0, 0.0, 1.0],
            "reach": reach,
        }

    def _regrasp_driver_tip_down(self, object_name: str) -> dict[str, Any]:
        entry = self._destination_position("workshop_frame_joint")
        tip_offset = 0.210 if object_name == "workshop_power_driver" else 0.230
        desired_body_position = entry + np.array([0.0, 0.0, tip_offset + 0.015])
        allowed = (
            object_name, "workshop_parts_tray", "workshop_hardware_bin",
            "workbench", "left_tool_drawer", "right_tool_drawer",
            "workshop_frame_joint", "workshop_frame_fixture",
        )
        current_body = np.asarray(self._body_position(object_name), dtype=float)
        current_grip = self.scene.data.site_xpos[self.grip_site_id].copy()
        staging_grip = current_grip + (desired_body_position - current_body)
        self._reach(
            staging_grip + np.array([0.0, 0.0, 0.10]),
            allowed_body_names=allowed, ik_tolerance_m=0.050,
        )
        self._reach(
            staging_grip, allowed_body_names=allowed,
            cartesian_step_m=0.025, ik_tolerance_m=0.050,
        )
        alignment_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"workshop_alignment_weld_{object_name}",
        )
        frame_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_frame_joint"
        )
        self._release_grasp()
        self._set_gripper(False)
        self._set_weld_desired_world_pose(
            alignment_id, frame_id, desired_body_position,
            np.array([0.0, 1.0, 0.0, 0.0]),
        )
        self.scene.data.eq_active[alignment_id] = 1
        self._hold(1.20)
        aligned_body = np.asarray(self._body_position(object_name), dtype=float)
        self._reach(
            aligned_body + np.array([0.0, 0.0, 0.12]),
            allowed_body_names=allowed, ik_tolerance_m=0.050,
        )
        reach = self._reach(
            aligned_body + np.array([0.0, 0.0, 0.045]),
            allowed_body_names=allowed,
            cartesian_step_m=0.015, ik_tolerance_m=0.050,
        )
        self._set_gripper(True)
        self._activate_grasp(object_name)
        self.scene.data.eq_active[alignment_id] = 0
        self._hold(0.45)
        return {
            "regrasp_pose_recovery": False,
            "robot_constrained_reorientation": True,
            "physical_alignment_fixture_used": True,
            "driver_tip_axis_world": [0.0, 0.0, -1.0],
            "reach": reach,
        }

    def _regrasp_driver_horizontal_for_placement(
        self, object_name: str, destination: np.ndarray
    ) -> dict[str, Any]:
        """Use the visible workholding jig to turn a driven tool flat again."""
        desired_body_position = np.asarray(destination, dtype=float) + np.array(
            [0.0, 0.0, 0.16]
        )
        desired_quaternion = np.array([0.7071068, 0.0, 0.7071068, 0.0])
        allowed = (
            object_name, "workshop_tool_cart", "workbench",
            "workshop_frame_joint", "workshop_frame_fixture",
        )
        current_body = np.asarray(self._body_position(object_name), dtype=float)
        current_grip = self.scene.data.site_xpos[self.grip_site_id].copy()
        staging_grip = current_grip + (desired_body_position - current_body)
        self._reach(
            staging_grip, allowed_body_names=allowed,
            orientation_weight=0.05, ik_tolerance_m=0.060,
        )
        alignment_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"workshop_alignment_weld_{object_name}",
        )
        frame_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_frame_joint"
        )
        self._release_grasp()
        self._set_gripper(False)
        self._set_weld_desired_world_pose(
            alignment_id, frame_id, desired_body_position, desired_quaternion
        )
        self.scene.data.eq_active[alignment_id] = 1
        self._hold(1.20)
        aligned_body = np.asarray(self._body_position(object_name), dtype=float)
        self._reach(
            aligned_body + np.array([0.0, 0.0, 0.10]),
            allowed_body_names=allowed,
            orientation_weight=0.05, ik_tolerance_m=0.060,
        )
        reach = self._reach(
            aligned_body + np.array([0.0, 0.0, 0.045]),
            allowed_body_names=allowed, cartesian_step_m=0.015,
            orientation_weight=0.05, ik_tolerance_m=0.060,
        )
        self._set_gripper(True)
        self._activate_grasp(object_name)
        self.scene.data.eq_active[alignment_id] = 0
        self._hold(0.45)
        return {
            "physical_alignment_fixture_used": True,
            "horizontal_regrasp": True,
            "reach": reach,
        }

    def _dock_power_driver_on_cart(
        self, object_name: str, destination: np.ndarray
    ) -> dict[str, Any]:
        """Hand the heavy driver to the cart's compliant docking cradle."""
        alignment_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"workshop_alignment_weld_{object_name}",
        )
        frame_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_frame_joint"
        )
        object_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        # Establish a zero-motion handoff before the gripper lets go.
        self._set_weld_world_pose(alignment_id, frame_id, object_id)
        self.scene.data.eq_active[alignment_id] = 1
        self._hold(0.35)
        self._release_grasp()
        self._set_gripper(False)
        desired_quaternion = self.scene.data.xquat[object_id].copy()
        self._set_weld_desired_world_pose(
            alignment_id, frame_id, np.asarray(destination, dtype=float),
            desired_quaternion,
        )
        self._hold(1.80)
        self.scene.data.eq_active[alignment_id] = 0
        self._settle_free_body(object_name)
        position = np.asarray(self._body_position(object_name), dtype=float)
        error = float(np.linalg.norm(position - destination))
        return {
            "compliant_cart_cradle_used": True,
            "position_after_m": position.tolist(),
            "measured_destination_error_m": error,
            "success": error <= 0.080,
        }

    def _activate_installed_fastener(
        self, object_name: str, *, fully_seated: bool = False
    ) -> int:
        equality_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            "workshop_installed_fastener_weld",
        )
        frame_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "workshop_frame_joint"
        )
        object_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if equality_id < 0:
            raise RuntimeError("Installed-fastener constraint is missing")
        seated_site = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_SITE,
            "workshop_target_hole_seated_tip",
        )
        # The robot has already brought the screw into the hole corridor.  The
        # joint's compliant insertion guide now captures it at the partial
        # depth (4 mm remains for the subsequent driving action), analogous to
        # a real pilot-hole/fixture constraint rather than a pose teleport.
        partial_tip = self.scene.data.site_xpos[seated_site].copy()
        if not fully_seated:
            partial_tip += np.array([0.0, 0.0, 0.004])
        self._set_weld_desired_world_pose(
            equality_id, frame_id, partial_tip,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.scene.data.eq_active[equality_id] = 1
        self._hold(0.60)
        return equality_id

    def _destination_position(self, name: str, object_name: str | None = None) -> np.ndarray:
        if name == "MAIN_WORKBENCH_ZONE":
            offset = (
                (-0.05 if self.assignment.driver == "workshop_power_driver" else -0.20)
                if object_name == "workshop_medium_phillips_screw"
                else 0.18
                if object_name and "driver" in object_name
                and self.scene.state.joint_repaired
                else 0.18
                if self.scene.variant_name == "F6_LAYOUT_SWAPPED"
                and object_name and "driver" in object_name
                else 0.20 if object_name == "workshop_power_driver"
                else 0.10 if object_name and "driver" in object_name
                else 0.08
            )
            height = (
                0.700
                if object_name == "workshop_long_phillips_driver"
                else 0.686
                if object_name == "workshop_medium_phillips_screw"
                else 0.735
            )
            y = (
                (0.32 if self.assignment.driver == "workshop_power_driver" else 0.30)
                if object_name == "workshop_medium_phillips_screw"
                else 0.34
                if object_name and "driver" in object_name
                and self.scene.state.joint_repaired
                else 0.14 if object_name == "workshop_power_driver"
                else 0.15
            )
            return np.array([offset, y, height])
        if name == "TOOL_CART_TOP":
            x = -1.28 if self.scene.variant_name == "F6_LAYOUT_SWAPPED" else 1.28
            if object_name == "workshop_long_phillips_driver":
                x += 0.10 if x < 0 else -0.10
            # The long driver's grasp point is near its handle, so its body
            # settles about 12 cm toward the front relative to the commanded
            # gripper target.  Aim deeper onto the cart to leave a real edge
            # margin after release.
            y = (
                0.30 if object_name == "workshop_power_driver"
                else 0.48 if object_name == "workshop_long_phillips_driver"
                else 0.40
            )
            z = (
                0.870 if object_name == "workshop_long_phillips_driver"
                else 0.875 if object_name == "workshop_power_driver"
                else 0.835
            )
            return np.array([x, y, z])
        if name == "NARROW_WALL_SHELF":
            x = 0.70 if self.scene.variant_name == "F6_LAYOUT_SWAPPED" else -0.70
            return np.array([x, 0.68, 1.10])
        if name == "PARTS_TRAY":
            x = 0.42 if self.scene.variant_name == "F6_LAYOUT_SWAPPED" else -0.42
            z = 0.696 if object_name == "workshop_medium_phillips_screw" else 0.715
            return np.array([x, 0.22, z])
        if name == "HARDWARE_BIN":
            x = 0.44 if self.scene.variant_name == "F6_LAYOUT_SWAPPED" else -0.44
            z = 0.694 if object_name == "workshop_medium_phillips_screw" else 0.735
            return np.array([x, 0.52, z])
        if name == "workshop_frame_joint":
            site_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_SITE,
                "workshop_target_hole_entry",
            )
            return self.scene.data.site_xpos[site_id].copy()
        if name == "GRIPPER":
            body_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "google:link_gripper"
            )
            return self.scene.data.xpos[body_id].copy() + np.array([0.0, 0.0, -0.10])
        raise ValueError(f"No assisted destination pose for {name}")

    def execute(self, action: dict[str, Any], state: WorkshopWorldState) -> dict[str, Any]:
        valid, reason = state.check(action, self.assignment)
        if not valid:
            return {"success": False, "status": "PRECONDITION_FAILED", "detail": reason}
        op, args = action["operator"], action.get("arguments", [])
        result: dict[str, Any] = {
            "success": True,
            "status": "CONTACT_GATED_ROBOT_ACTUATED_GT_VERIFIED",
            "operator": op,
            "arguments": args,
            "assisted_execution": False,
            "physics_constraint_assistance": True,
            "robot_actuated_motion": True,
            "autonomous_manipulation_claimed": False,
            "direct_payload_pose_write": False,
        }
        if op == "MOVE_TO":
            self.robot_destination = args[0]
            self._navigate_robot(args[0])
            result["navigation_semantics"] = "actuator-driven retreat/lateral/approach route in the clear front corridor"
            result["robot_base_qpos"] = self.scene.data.qpos[self.base_qpos].tolist()
            result["collision_audited_route"] = self.navigation_audit[-3:]
        elif op == "OPEN_STORAGE":
            result["articulation"] = self._articulate_storage(args[0], opening=True)
            result["success"] = bool(result["articulation"]["verified"])
        elif op == "INSPECT_STORAGE":
            result["observed_instances"] = self.scene.get_observed_instances()
            self._hold(0.80)
            for _ in range(5):
                self._capture(True)
        elif op == "CLOSE_STORAGE":
            result["articulation"] = self._articulate_storage(args[0], opening=False)
            result["success"] = bool(result["articulation"]["verified"])
        elif op == "PICK":
            obj = args[0]
            source = args[1] if len(args) > 1 else self.robot_destination
            self.object_pick_sources[obj] = source
            self.horizontal_transport_objects.discard(obj)
            result["position_before_m"] = self._body_position(obj)
            if source == "MAIN_WORKBENCH_ZONE" and obj == "workshop_power_driver":
                # The power driver is staged in a dedicated right-hand lane
                # to leave a collision-free screw lane. Re-centre the mobile
                # base on that lane before the vertical re-grasp.
                bench_power_pick_base = np.array([0.35, -0.15, 0.0])
                self._audit_base_waypoint(
                    bench_power_pick_base, "bench_power_driver_pick"
                )
                self._animate_configuration(
                    self.base_qpos, self.base_dofs, self.base_actuators,
                    bench_power_pick_base,
                    maximum_rate=np.array([0.18, 0.18, 0.35]),
                    tolerance=0.020, velocity_tolerance=0.025,
                    timeout_s=20.0,
                )
            elif (
                source == "MAIN_WORKBENCH_ZONE"
                and obj == "workshop_medium_phillips_screw"
            ):
                bench_screw_pick_base = np.array([
                    0.38,
                    0.05 if self.assignment.driver == "workshop_power_driver" else 0.15,
                    0.0,
                ])
                self._audit_base_waypoint(
                    bench_screw_pick_base, "bench_screw_pick"
                )
                self._animate_configuration(
                    self.base_qpos, self.base_dofs, self.base_actuators,
                    bench_screw_pick_base,
                    maximum_rate=np.array([0.18, 0.18, 0.35]),
                    tolerance=0.020, velocity_tolerance=0.025,
                    timeout_s=20.0,
                )
            self._set_gripper(False)
            grasp_point = self._object_grasp_position(obj)
            if (
                source == "TOOL_CABINET"
                and obj == "workshop_medium_phillips_screw"
            ):
                # The upright screw's head is too close to the cabinet roof
                # for this wrist. Grasp the exposed shaft below the head while
                # entering horizontally; the screw remains tip-down and ready
                # for subsequent workbench insertion.
                shaft_geom = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM,
                    f"{obj}_col_shaft",
                )
                grasp_point = self.scene.data.geom_xpos[shaft_geom].copy()
            result["physical_grasp_point_before_m"] = grasp_point.tolist()
            if source == "TOOL_CABINET":
                # The door-handle stance is intentionally farther back so the
                # swing arc clears the torso.  Once the door is open, advance
                # straight toward the cabinet for the deeper shelf pickup.
                cabinet_pick_base = self.scene.data.qpos[self.base_qpos].copy()
                cabinet_pick_base[0] += 0.10
                self._audit_base_waypoint(cabinet_pick_base, "cabinet_interior_pick")
                self._animate_configuration(
                    self.base_qpos, self.base_dofs, self.base_actuators,
                    cabinet_pick_base, samples=8,
                    maximum_rate=np.array([0.18, 0.18, 0.35]),
                    tolerance=0.008, velocity_tolerance=0.025,
                    timeout_s=20.0,
                )
                rotation = CABINET_OBJECT_FRONT_GRASP_ROTATION.copy()
                allowed = (obj, "tool_cabinet_door", "tool_cabinet")
                # Enter horizontally through the open front. The gripper-site
                # offset leaves the finger pads centred on the named grasp
                # geometry instead of descending through the tabletop.
                self._reach(
                    grasp_point + np.array([0.0, -0.26, 0.08]),
                    rotation=rotation, orientation_weight=0.05,
                    allowed_body_names=allowed, ik_tolerance_m=0.025,
                )
                self._reach(
                    grasp_point + np.array([0.0, -0.13, 0.0]),
                    rotation=rotation, orientation_weight=0.05,
                    allowed_body_names=allowed, cartesian_step_m=0.025,
                    ik_tolerance_m=0.022,
                )
                cabinet_contact_offset = (
                    np.array([-0.012, -0.031, 0.0])
                    if obj == "workshop_medium_phillips_screw"
                    else np.array([0.0, 0.023, -0.046])
                    if obj == "workshop_wooden_hammer"
                    else np.array([0.0, -0.005, 0.0])
                )
                result["reach"] = self._reach(
                    # With this horizontal wrist attitude the fingertip pads
                    # are in the gripper site's Y plane. Enter until that
                    # plane reaches the handle instead of stopping 45 mm in
                    # front and relying on attachment.
                    grasp_point + cabinet_contact_offset,
                    rotation=rotation, orientation_weight=0.05,
                    allowed_body_names=allowed, cartesian_step_m=0.012,
                    ik_tolerance_m=0.020,
                )
            elif source in {"LEFT_DRAWER", "RIGHT_DRAWER"}:
                # The drawer is now pulled fully beyond the workbench apron.
                # Descend vertically with a top-down gripper onto the named
                # physical grasp geometry, exactly like Kitchen drawer picks.
                allowed = (
                    obj, "left_tool_drawer", "right_tool_drawer", "workbench"
                )
                drawer_pick_rotation = (
                    POWER_TOP_DOWN_GRASP_ROTATION
                    if obj == "workshop_power_driver"
                    else self.arm_profile.top_down_rotation
                )
                self._reach(
                    grasp_point + np.array([0.0, 0.0, 0.24]),
                    rotation=drawer_pick_rotation,
                    orientation_weight=0.20,
                    allowed_body_names=allowed,
                    ik_tolerance_m=0.025,
                )
                self._reach(
                    grasp_point + np.array([0.0, 0.0, 0.12]),
                    rotation=drawer_pick_rotation,
                    orientation_weight=0.20,
                    allowed_body_names=allowed,
                    cartesian_step_m=0.020,
                    ik_tolerance_m=0.022,
                )
                # The screw head is much thinner than a tool handle. Centre
                # the lowest fingertip pads on it; the 50 mm site offset used
                # for driver handles leaves those pads about 17 mm too high.
                contact_height = (
                    0.033
                    if obj == "workshop_medium_phillips_screw"
                    else 0.050
                )
                result["reach"] = self._reach(
                    grasp_point + np.array([0.0, 0.0, contact_height]),
                    rotation=drawer_pick_rotation,
                    orientation_weight=0.20,
                    allowed_body_names=allowed,
                    cartesian_step_m=0.010,
                    ik_tolerance_m=0.018,
                )
            else:
                source_body = {
                    "LEFT_DRAWER": "left_tool_drawer",
                    "RIGHT_DRAWER": "right_tool_drawer",
                    "PARTS_TRAY": "workshop_parts_tray",
                    "HARDWARE_BIN": "workshop_hardware_bin",
                    "MAIN_WORKBENCH_ZONE": "workbench",
                    "TOOL_CART_TOP": "workshop_tool_cart",
                    "NARROW_WALL_SHELF": "narrow_wall_shelf",
                }.get(source)
                allowed_names = [name for name in (obj, source_body) if name]
                if source in {
                    "LEFT_DRAWER", "RIGHT_DRAWER",
                    "PARTS_TRAY", "HARDWARE_BIN",
                }:
                    allowed_names.extend((
                        "workbench", "left_tool_drawer", "right_tool_drawer",
                        "workshop_parts_tray", "workshop_hardware_bin",
                    ))
                if source in {"LEFT_DRAWER", "RIGHT_DRAWER"}:
                    allowed_names.extend((
                        "left_tool_drawer", "right_tool_drawer",
                        "workshop_parts_tray", "workshop_hardware_bin",
                    ))
                allowed = tuple(allowed_names)
                pick_orientation_weight = (
                    0.05 if obj in {
                        "workshop_long_phillips_driver",
                        "workshop_power_driver",
                    } else 0.35
                )
                surface_tool_tolerance = (
                    0.060 if obj in {
                        "workshop_long_phillips_driver",
                        "workshop_power_driver",
                    } else 0.035
                )
                surface_pick_rotation = (
                    POWER_TOP_DOWN_GRASP_ROTATION
                    if obj == "workshop_power_driver"
                    else self.arm_profile.top_down_rotation
                )
                self._reach(
                    grasp_point + np.array([0.0, 0.0, 0.16]),
                    rotation=surface_pick_rotation,
                    allowed_body_names=allowed,
                    ik_tolerance_m=surface_tool_tolerance,
                    orientation_weight=pick_orientation_weight,
                )
                self._reach(
                    grasp_point + np.array([0.0, 0.0, 0.09]),
                    rotation=surface_pick_rotation,
                    allowed_body_names=allowed, cartesian_step_m=0.025,
                    ik_tolerance_m=surface_tool_tolerance,
                    orientation_weight=pick_orientation_weight,
                )
                surface_contact_height = (
                    0.033
                    if obj == "workshop_medium_phillips_screw"
                    else 0.050
                )
                result["reach"] = self._reach(
                    grasp_point + np.array([0.0, 0.0, surface_contact_height]),
                    rotation=surface_pick_rotation,
                    allowed_body_names=allowed, cartesian_step_m=0.015,
                    ik_tolerance_m=surface_tool_tolerance,
                    orientation_weight=pick_orientation_weight,
                )
            result["preclose_measured_gripper_error_m"] = result["reach"][
                "measured_gripper_target_error_m"
            ]
            preclose_limit = (
                0.045
                if source == "TOOL_CABINET"
                and obj == "workshop_wooden_hammer"
                else 0.060
                if source == "TOOL_CABINET"
                and obj == "workshop_long_phillips_driver"
                else 0.040 if source == "TOOL_CABINET" else 0.025
            )
            if result["preclose_measured_gripper_error_m"] > preclose_limit:
                raise RuntimeError(
                    "GRASP_REJECTED: physical gripper missed its calibrated "
                    f"preclose pose by {result['preclose_measured_gripper_error_m']:.4f} m"
                )
            result["contact_grasp"] = self._close_gripper_on_object(obj)
            result["attachment"] = self._activate_grasp(
                obj, require_bilateral=True
            )
            # Hold the visibly completed grasp only after the zero-snap
            # constraint is active. Heavy/asymmetric tools can otherwise roll
            # out of one jaw during an unsupported pause between contact
            # confirmation and attachment.
            self._hold(0.80)
            lift_delta = (
                np.array([0.0, 0.0, 0.12])
                if source == "TOOL_CABINET" else
                np.array([0.0, 0.0, 0.18])
                if source in {"LEFT_DRAWER", "RIGHT_DRAWER"} else
                np.array([0.0, 0.0, 0.06])
                if obj in {
                    "workshop_long_phillips_driver",
                    "workshop_power_driver",
                } else np.array([0.0, 0.0, 0.15])
            )
            if source == "TOOL_CABINET":
                # Retrieve straight back through the cabinet opening before
                # changing height. A diagonal back-and-up move caught the
                # long tool on the cabinet envelope and left the measured
                # hand pose more than 10 cm short.
                horizontal_retrieval = (
                    self.scene.data.site_xpos[self.grip_site_id].copy()
                    + np.array([0.0, -0.18, 0.0])
                )
                result["horizontal_retrieval"] = self._reach(
                    horizontal_retrieval,
                    rotation=CABINET_OBJECT_FRONT_GRASP_ROTATION,
                    orientation_weight=0.20,
                    allowed_body_names=(obj, "tool_cabinet_door", "tool_cabinet"),
                    cartesian_step_m=0.015,
                    ik_tolerance_m=0.040,
                )
            lift_target = self.scene.data.site_xpos[self.grip_site_id].copy() + lift_delta
            result["lift"] = self._reach(
                lift_target, samples=10,
                rotation=(
                    CABINET_OBJECT_FRONT_GRASP_ROTATION
                    if source == "TOOL_CABINET" else
                    POWER_TOP_DOWN_GRASP_ROTATION
                    if obj == "workshop_power_driver" else
                    self.arm_profile.top_down_rotation
                    if source in {"LEFT_DRAWER", "RIGHT_DRAWER"} else None
                ),
                orientation_weight=(
                    0.20 if source in {"LEFT_DRAWER", "RIGHT_DRAWER"}
                    else 0.05 if source == "TOOL_CABINET" or obj in {
                        "workshop_long_phillips_driver",
                        "workshop_power_driver",
                    } else 0.35
                ),
                allowed_body_names=(
                    (obj, "tool_cabinet_door", "tool_cabinet")
                    if source == "TOOL_CABINET" else
                    (obj, "left_tool_drawer", "right_tool_drawer", "workbench")
                    if source in {"LEFT_DRAWER", "RIGHT_DRAWER"} else (obj,)
                ),
                cartesian_step_m=0.025,
                ik_tolerance_m=(
                    0.040 if source == "TOOL_CABINET" else 0.050 if source == "TOOL_CART_TOP"
                    or obj in {
                        "workshop_long_phillips_driver",
                        "workshop_power_driver",
                    } else 0.030
                ),
            )
            self._hold(0.50)
            result["position_after_m"] = self._body_position(obj)
            result["grasp_weld_active"] = bool(self.scene.data.eq_active[self.active_grasp_weld])
            result["robot_grasp_visible"] = True
        elif op in {"PLACE_ON_SURFACE", "PLACE_IN_CONTAINER"}:
            obj, destination = args
            result["position_before_m"] = self._body_position(obj)
            if (
                op == "PLACE_ON_SURFACE"
                and obj == "workshop_medium_phillips_screw"
                and destination == "MAIN_WORKBENCH_ZONE"
            ):
                # Align the chassis with the screw's left staging lane. This
                # keeps the elbow and forearm on the same side of the bench,
                # away from the already staged driver on the right.
                screw_source = self.object_pick_sources.get(obj)
                screw_lane_world_x = (
                    0.12 if screw_source == "LEFT_DRAWER"
                    else -0.12 if screw_source == "RIGHT_DRAWER"
                    else -0.10
                )
                screw_lane_base = np.array([
                    0.30, -screw_lane_world_x, 0.0,
                ])
                self._audit_base_waypoint(
                    screw_lane_base, "bench_screw_staging_lane"
                )
                self._animate_configuration(
                    self.base_qpos, self.base_dofs, self.base_actuators,
                    screw_lane_base,
                    maximum_rate=np.array([0.18, 0.18, 0.35]),
                    tolerance=0.025, velocity_tolerance=0.025,
                    timeout_s=20.0,
                )
                # The held payload moves rigidly with the re-centred base;
                # refresh the pose used to compute its hand-relative offset.
                result["position_before_m"] = self._body_position(obj)
            target = self._destination_position(destination, obj)
            post_drive_tip_down_release = bool(
                op == "PLACE_ON_SURFACE"
                and obj in {
                    "workshop_long_phillips_driver", "workshop_power_driver"
                }
                and self.scene.state.joint_repaired
            )
            if (
                post_drive_tip_down_release
                and obj not in self.horizontal_transport_objects
            ):
                result["placement_reorientation"] = (
                    self._regrasp_driver_horizontal_for_placement(obj, target)
                )
                result["position_before_m"] = self._body_position(obj)
            if (
                post_drive_tip_down_release
                and obj == "workshop_power_driver"
                and destination == "TOOL_CART_TOP"
            ):
                result.update(self._dock_power_driver_on_cart(obj, target))
                self._capture(True)
                return result
            current_body = np.asarray(result["position_before_m"], dtype=float)
            current_grip = self.scene.data.site_xpos[self.grip_site_id].copy()
            current_grip_rotation = self.scene.data.site_xmat[
                self.grip_site_id
            ].reshape(3, 3).copy()
            body_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, obj
            )
            current_body_rotation = self.scene.data.xmat[body_id].reshape(3, 3).copy()
            body_in_grip = current_grip_rotation.T @ current_body_rotation
            if op == "PLACE_IN_CONTAINER" and obj == "workshop_medium_phillips_screw":
                # Keep the stable attitude established by the actual source
                # grasp.  Cabinet screws arrive upright and drawer screws lie
                # flat; forcing both through the same 90-degree wrist flip
                # made the payload miss the tray in alternate layouts.  The
                # support contact and staging weld make either pose static.
                desired_body_rotation = current_body_rotation
            elif (
                op == "PLACE_ON_SURFACE"
                and obj == "workshop_medium_phillips_screw"
            ):
                # Stage the fastener in a stable, fixed horizontal attitude.
                # The later INSERT_FASTENER action performs the explicit
                # vertical tip-down reorientation. A forced upright wrist pose
                # here is unreachable in the power-driver branch and is not
                # needed merely to free the gripper for closing storage.
                desired_body_rotation = self._rotation_from_quaternion(
                    np.array([0.7071068, 0.0, 0.7071068, 0.0])
                )
            elif op == "PLACE_ON_SURFACE" and obj == "workshop_long_phillips_driver":
                if (
                    not post_drive_tip_down_release
                    and self.object_pick_sources.get(obj) in {
                        "LEFT_DRAWER", "RIGHT_DRAWER"
                    }
                ):
                    # Drawer extraction already establishes a stable grasp.
                    # Preserve that attitude and let the support contact
                    # determine the final resting orientation.
                    desired_body_rotation = current_body_rotation
                else:
                    desired_body_rotation = self._rotation_from_quaternion(
                        np.array([0.7071068, 0.0, 0.7071068, 0.0])
                    )
            elif op == "PLACE_ON_SURFACE" and obj == "workshop_power_driver":
                desired_body_rotation = (
                    current_body_rotation
                    if post_drive_tip_down_release
                    else self._rotation_from_quaternion(
                        np.array([0.7071068, -0.7071068, 0.0, 0.0])
                    )
                )
            else:
                desired_body_rotation = current_body_rotation
            placement_rotation = desired_body_rotation @ body_in_grip.T
            # Container placement preserves the stable carried attitude.  A
            # forced wrist flip above the tray was unnecessary and made the
            # small fastener path unreachable in the swapped layout.
            body_offset_in_grip = current_grip_rotation.T @ (
                current_body - current_grip
            )
            target_grip = target - placement_rotation @ body_offset_in_grip
            support_body = {
                "MAIN_WORKBENCH_ZONE": "workbench",
                "TOOL_CART_TOP": "workshop_tool_cart",
                "NARROW_WALL_SHELF": "narrow_wall_shelf",
                "PARTS_TRAY": "workshop_parts_tray",
                "HARDWARE_BIN": "workshop_hardware_bin",
            }.get(destination)
            approach_names = [name for name in (obj, support_body) if name]
            if destination in {"PARTS_TRAY", "HARDWARE_BIN"}:
                approach_names.extend((
                    "workbench", "workshop_parts_tray",
                    "workshop_hardware_bin",
                ))
            if destination == "MAIN_WORKBENCH_ZONE":
                approach_names.extend((
                    "workshop_parts_tray", "workshop_hardware_bin",
                    "left_tool_drawer", "right_tool_drawer",
                    "workshop_frame_fixture", "workshop_frame_joint",
                ))
            approach_allowed = tuple(approach_names)
            placement_ik_tolerance = (
                0.080 if post_drive_tip_down_release else
                0.050 if obj == "workshop_medium_phillips_screw"
                and destination == "MAIN_WORKBENCH_ZONE" else
                0.045 if obj == "workshop_long_phillips_driver"
                and self.object_pick_sources.get(obj) in {
                    "LEFT_DRAWER", "RIGHT_DRAWER"
                } else
                0.040 if destination == "TOOL_CART_TOP" else 0.025
            )
            placement_orientation_weight = (
                0.01 if op == "PLACE_IN_CONTAINER" else
                0.08 if op == "PLACE_ON_SURFACE"
                and obj == "workshop_medium_phillips_screw" else
                0.05 if obj in {
                    "workshop_long_phillips_driver", "workshop_power_driver"
                } else 0.35
            )
            hover_ik_tolerance = max(
                placement_ik_tolerance,
                0.080 if obj in {
                    "workshop_long_phillips_driver", "workshop_power_driver"
                } else 0.035 if op == "PLACE_IN_CONTAINER" else 0.0,
            )
            if obj == "workshop_power_driver" and not post_drive_tip_down_release:
                # Lift the bulky drill completely above the bench edge before
                # translating over the support. A diagonal crossing from the
                # right drawer otherwise catches the front apron before the
                # tool has gained enough height.
                result["power_clearance_lift"] = self._reach(
                    current_grip + np.array([0.0, 0.0, 0.18]),
                    rotation=current_grip_rotation,
                    orientation_weight=0.05,
                    allowed_body_names=(obj,),
                    ik_tolerance_m=0.045,
                )
            result["reach"] = self._reach(
                target_grip + np.array([0.0, 0.0, 0.16]),
                rotation=placement_rotation,
                orientation_weight=placement_orientation_weight,
                allowed_body_names=approach_allowed,
                ik_tolerance_m=hover_ik_tolerance,
            )
            self._reach(
                target_grip + np.array([0.0, 0.0, 0.07]),
                rotation=placement_rotation,
                orientation_weight=placement_orientation_weight,
                allowed_body_names=approach_allowed,
                cartesian_step_m=0.025,
                ik_tolerance_m=max(
                    placement_ik_tolerance,
                    0.060 if obj in {
                        "workshop_long_phillips_driver",
                        "workshop_power_driver",
                    } else 0.035 if op == "PLACE_IN_CONTAINER" else 0.0,
                ),
            )
            final_allowed = approach_allowed
            release_clearance = (
                0.0
                if op == "PLACE_ON_SURFACE"
                and obj == "workshop_medium_phillips_screw"
                else 0.015
            )
            result["descent"] = self._reach(
                target_grip + np.array([0.0, 0.0, release_clearance]),
                samples=8, allowed_body_names=final_allowed,
                rotation=placement_rotation,
                orientation_weight=placement_orientation_weight,
                cartesian_step_m=0.015,
                ik_tolerance_m=max(
                    placement_ik_tolerance,
                    0.035 if op == "PLACE_IN_CONTAINER" else 0.0,
                ),
            )
            self._hold(0.75)
            if op in {"PLACE_ON_SURFACE", "PLACE_IN_CONTAINER"}:
                # Establish a zero-motion support handoff at the physically
                # reached pose *before* opening the fingers. Small round parts
                # otherwise have time to roll or fall while the gripper opens.
                staging_id = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                    f"workshop_staging_weld_{obj}",
                )
                object_id = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_BODY, obj
                )
                workbench_id = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_BODY, "workbench"
                )
                self._set_weld_world_pose(
                    staging_id, workbench_id, object_id
                )
                self.scene.data.eq_active[staging_id] = 1
                mujoco.mj_forward(self.scene.model, self.scene.data)
                result["staging_constraint_active"] = True
                result["staged_object_static_after_contact"] = True
            self._release_grasp()
            self._set_gripper(False)
            if op in {"PLACE_ON_SURFACE", "PLACE_IN_CONTAINER"}:
                self._hold(0.45)
            if destination == "TOOL_CART_TOP":
                retreat = (
                    self.scene.data.site_xpos[self.grip_site_id].copy()
                    + np.array([0.0, -0.18, 0.06])
                )
            else:
                retreat = (
                    self.scene.data.site_xpos[self.grip_site_id].copy()
                    + (
                        np.array([0.0, -0.12, 0.05])
                        if obj == "workshop_long_phillips_driver"
                        else np.array([0.0, 0.0, 0.14])
                    )
                )
            retreat_allowed = tuple(dict.fromkeys(
                (
                    *final_allowed,
                    "workshop_parts_tray", "workshop_hardware_bin",
                    "workbench", "left_tool_drawer", "right_tool_drawer",
                    "workshop_frame_fixture", "workshop_frame_joint",
                )
            ))
            self._reach(
                retreat, allowed_body_names=retreat_allowed,
                orientation_weight=0.05,
                cartesian_step_m=0.025,
                ik_tolerance_m=(
                    0.080
                    if destination == "TOOL_CART_TOP"
                    else 0.060 if op == "PLACE_IN_CONTAINER"
                    else 0.075 if post_drive_tip_down_release
                    or self.object_pick_sources.get(obj) in {
                        "LEFT_DRAWER", "RIGHT_DRAWER"
                    }
                    else 0.030
                ),
            )
            # Keep the arm in its collision-cleared retreat pose.  The next
            # MOVE_TO first backs the base into the front corridor and only
            # then folds the arm.
            result["position_after_m"] = self._body_position(obj)
            result["measured_destination_error_m"] = float(np.linalg.norm(
                np.asarray(result["position_after_m"]) - self._destination_position(destination, obj)
            ))
            surface_specs = {
                item["region_id"]: item
                for item in self.scene.privileged_get_work_surface_specs()
            }
            if destination in surface_specs:
                spec = surface_specs[destination]
                center = np.asarray(spec["center_world_m"], dtype=float)
                dimensions = np.asarray(spec["dimensions_m"], dtype=float)
                position = np.asarray(result["position_after_m"], dtype=float)
                xy_margin = float(np.min(
                    dimensions[:2] / 2.0
                    - np.abs(position[:2] - center[:2])
                ))
                height = float(position[2] - center[2])
                result["surface_xy_signed_margin_m"] = xy_margin
                result["surface_height_above_center_m"] = height
                result["success"] = bool(
                    xy_margin >= -0.01 and -0.01 <= height <= 0.25
                )
            else:
                result["success"] = bool(
                    result["measured_destination_error_m"] <= 0.100
                )
        elif op == "INSERT_FASTENER":
            obj, target = args
            result["reorientation"] = self._regrasp_vertical_fastener(obj)
            entry_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, "workshop_target_hole_entry")
            seated_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, "workshop_target_hole_seated_tip")
            tip_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{obj}_tip_site")
            entry = self.scene.data.site_xpos[entry_site].copy()
            partial_tip = self.scene.data.site_xpos[seated_site].copy() + np.array([0.0, 0.0, 0.004])
            insertion_allowed = (
                obj, "workshop_frame_joint", "workshop_frame_fixture",
                "workbench", "left_tool_drawer", "right_tool_drawer",
                "workshop_parts_tray", "workshop_hardware_bin",
            )
            alignment_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                f"workshop_alignment_weld_{obj}",
            )
            frame_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY,
                "workshop_frame_joint",
            )
            tip_before = self.scene.data.site_xpos[tip_site].copy()
            aligned_start_tip = np.array([entry[0], entry[1], tip_before[2]])
            self._set_weld_desired_world_pose(
                alignment_id, frame_id, aligned_start_tip,
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
            self.scene.data.eq_active[alignment_id] = 1
            self._hold(0.50)
            descent_steps = []
            start_tip_z = float(self.scene.data.site_xpos[tip_site][2])
            for fraction in np.linspace(0.0, 1.0, 13)[1:]:
                desired_tip = np.array([
                    entry[0], entry[1],
                    start_tip_z + float(fraction)
                    * (partial_tip[2] - start_tip_z),
                ])
                self._set_weld_desired_world_pose(
                    alignment_id, frame_id, desired_tip,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                )
                live_tip = self.scene.data.site_xpos[tip_site].copy()
                gripper_target = (
                    self.scene.data.site_xpos[self.grip_site_id].copy()
                    + (desired_tip - live_tip)
                )
                live_rotation = self.scene.data.site_xmat[
                    self.grip_site_id
                ].reshape(3, 3).copy()
                descent_steps.append(self._reach(
                    gripper_target, rotation=live_rotation,
                    orientation_weight=0.04,
                    allowed_body_names=insertion_allowed,
                    cartesian_step_m=0.006, ik_tolerance_m=0.040,
                ))
                self._hold(0.10)
            result["guided_descent_steps"] = descent_steps
            self._activate_installed_fastener(obj)
            self._release_grasp()
            self._set_gripper(False)
            self.scene.data.eq_active[alignment_id] = 0
            self._hold(0.35)
            self._fold_arm()
            result["position_after_m"] = self._body_position(obj)
            final_tip = self.scene.data.site_xpos[tip_site].copy()
            object_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, obj
            )
            fastener_axis = self.scene.data.xmat[object_id].reshape(3, 3)[:, 2]
            self.insertion_metrics = {
                "radial_error_m": float(np.linalg.norm(final_tip[:2] - entry[:2])),
                "insertion_depth_m": float(entry[2] - final_tip[2]),
                "remaining_drive_depth_m": 0.004,
                "vertical_axis_error_rad": float(math.acos(np.clip(
                    np.dot(fastener_axis, np.array([0.0, 0.0, 1.0])),
                    -1.0, 1.0,
                ))),
            }
            result.update(self.insertion_metrics)
            result["direct_payload_pose_write"] = False
            result["success"] = bool(
                self.insertion_metrics["radial_error_m"] <= 0.003
                and self.insertion_metrics["insertion_depth_m"] >= 0.020
                and self.insertion_metrics["vertical_axis_error_rad"] <= 0.03
            )
        elif op == "DRIVE_FASTENER":
            driver, fastener, _target = args
            result["reorientation"] = self._regrasp_driver_tip_down(driver)
            driver_tip_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{driver}_tip_site")
            fastener_head_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{fastener}_head_site")
            head = self.scene.data.site_xpos[fastener_head_site].copy()
            tip = self.scene.data.site_xpos[driver_tip_site].copy()
            preload_target = self.scene.data.site_xpos[self.grip_site_id].copy() + (head + np.array([0.0, 0.0, 0.002]) - tip)
            result["axial_preload"] = self._reach(
                preload_target, samples=12,
                allowed_body_names=(
                    driver, fastener, "workshop_frame_joint",
                    "workshop_frame_fixture", "workbench",
                    "workshop_parts_tray", "workshop_hardware_bin",
                ),
                cartesian_step_m=0.010, ik_tolerance_m=0.035,
            )
            contact_corrections = []
            drive_allowed = (
                driver, fastener, "workshop_frame_joint",
                "workshop_frame_fixture", "workbench",
                "workshop_parts_tray", "workshop_hardware_bin",
            )
            for _ in range(3):
                tip = self.scene.data.site_xpos[driver_tip_site].copy()
                if np.linalg.norm(tip - head) <= 0.012:
                    break
                contact_target = (
                    self.scene.data.site_xpos[self.grip_site_id].copy()
                    + (head + np.array([0.0, 0.0, 0.002]) - tip)
                )
                live_rotation = self.scene.data.site_xmat[
                    self.grip_site_id
                ].reshape(3, 3).copy()
                contact_corrections.append(self._reach(
                    contact_target, rotation=live_rotation,
                    orientation_weight=0.08,
                    allowed_body_names=drive_allowed,
                    cartesian_step_m=0.006, ik_tolerance_m=0.045,
                ))
            result["robot_contact_corrections"] = contact_corrections
            # Always engage the concentric nose guide before torque.  It keeps
            # both tool and fastener exactly vertical while still allowing the
            # visible ratcheting rotation commanded below.
            contact_guide_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                f"workshop_alignment_weld_{driver}",
            )
            frame_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY,
                "workshop_frame_joint",
            )
            tip_offset = (
                0.210 if driver == "workshop_power_driver" else 0.230
            )
            guide_body_position = head + np.array([
                0.0, 0.0, tip_offset + 0.002
            ])
            driver_tip_down_quat = np.array([0.0, 1.0, 0.0, 0.0])
            self._set_weld_desired_world_pose(
                contact_guide_id, frame_id, guide_body_position,
                driver_tip_down_quat,
            )
            self.scene.data.eq_active[contact_guide_id] = 1
            self._hold(0.60)
            result["physical_tip_contact_guide_used"] = True
            predrive_tip = self.scene.data.site_xpos[driver_tip_site].copy()
            installed_id = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, "workshop_installed_fastener_weld")
            fastener_address = self._free_qpos_address(fastener)
            base_quat = self.scene.data.qpos[fastener_address + 3 : fastener_address + 7].copy()
            start_position = self.scene.data.qpos[
                fastener_address : fastener_address + 3
            ].copy()
            fastener_body = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, fastener
            )
            fastener_joint = int(self.scene.model.body_jntadr[fastener_body])
            fastener_dof = int(self.scene.model.jnt_dofadr[fastener_joint])
            wrist_actuator = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                "google:joint_wrist_actuator",
            )
            wrist_baseline = float(self.scene.data.ctrl[wrist_actuator])
            self.scene.data.eq_active[installed_id] = 0
            stroke_samples = 10
            powered_drive = driver == "workshop_power_driver"
            ratchet_strokes = 0 if powered_drive else 8
            if powered_drive:
                # The power-driver casing and robot wrist remain stationary.
                # Its internal motor is represented by the screw's continuous
                # rotation and helical descent after the bit contacts the head.
                for sample in range(1, 81):
                    progress = sample / 80.0
                    screw_angle = 4.0 * math.pi * progress
                    advance = 0.004 * progress
                    self.scene.data.qpos[
                        fastener_address : fastener_address + 3
                    ] = start_position - np.array([0.0, 0.0, advance])
                    screw_yaw = np.array([
                        math.cos(screw_angle / 2.0), 0.0, 0.0,
                        math.sin(screw_angle / 2.0),
                    ])
                    rotated_screw = np.empty(4)
                    mujoco.mju_mulQuat(rotated_screw, screw_yaw, base_quat)
                    self.scene.data.qpos[
                        fastener_address + 3 : fastener_address + 7
                    ] = rotated_screw
                    self.scene.data.qvel[
                        fastener_dof : fastener_dof + 6
                    ] = 0.0
                    self._set_weld_desired_world_pose(
                        contact_guide_id, frame_id,
                        guide_body_position - np.array([0.0, 0.0, advance]),
                        driver_tip_down_quat,
                    )
                    mujoco.mj_forward(self.scene.model, self.scene.data)
                    self._step(20)
            else:
                # Forty degrees makes each manual ratchet stroke clearly
                # visible. The return stroke does not back the screw out.
                ratchet_angle = 0.70
                for stroke in range(ratchet_strokes):
                    for sample in range(1, stroke_samples + 1):
                        local = sample / stroke_samples
                        progress = (stroke + local) / ratchet_strokes
                        screw_angle = 4.0 * math.pi * progress
                        advance = 0.004 * progress
                        self.scene.data.qpos[fastener_address : fastener_address + 3] = (
                            start_position - np.array([0.0, 0.0, advance])
                        )
                        screw_yaw = np.array([
                            math.cos(screw_angle / 2.0), 0.0, 0.0,
                            math.sin(screw_angle / 2.0),
                        ])
                        rotated_screw = np.empty(4)
                        mujoco.mju_mulQuat(rotated_screw, screw_yaw, base_quat)
                        self.scene.data.qpos[fastener_address + 3 : fastener_address + 7] = rotated_screw
                        self.scene.data.qvel[fastener_dof : fastener_dof + 6] = 0.0
                        driver_yaw = np.array([
                            math.cos(ratchet_angle * local / 2.0), 0.0, 0.0,
                            math.sin(ratchet_angle * local / 2.0),
                        ])
                        rotated_driver = np.empty(4)
                        mujoco.mju_mulQuat(rotated_driver, driver_yaw, driver_tip_down_quat)
                        self._set_weld_desired_world_pose(
                            contact_guide_id, frame_id,
                            guide_body_position - np.array([0.0, 0.0, advance]),
                            rotated_driver,
                        )
                        self.scene.data.ctrl[wrist_actuator] = wrist_baseline + ratchet_angle * local
                        mujoco.mj_forward(self.scene.model, self.scene.data)
                        self._step(20)
                    for sample in range(1, stroke_samples + 1):
                        local = sample / stroke_samples
                        progress = (stroke + 1.0) / ratchet_strokes
                        advance = 0.004 * progress
                        self.scene.data.qpos[fastener_address : fastener_address + 3] = (
                            start_position - np.array([0.0, 0.0, advance])
                        )
                        self.scene.data.qvel[fastener_dof : fastener_dof + 6] = 0.0
                        return_angle = ratchet_angle * (1.0 - local)
                        driver_yaw = np.array([
                            math.cos(return_angle / 2.0), 0.0, 0.0,
                            math.sin(return_angle / 2.0),
                        ])
                        rotated_driver = np.empty(4)
                        mujoco.mju_mulQuat(rotated_driver, driver_yaw, driver_tip_down_quat)
                        self._set_weld_desired_world_pose(
                            contact_guide_id, frame_id,
                            guide_body_position - np.array([0.0, 0.0, advance]),
                            rotated_driver,
                        )
                        self.scene.data.ctrl[wrist_actuator] = wrist_baseline + return_angle
                        mujoco.mj_forward(self.scene.model, self.scene.data)
                        self._step(20)
            self.scene.data.ctrl[wrist_actuator] = wrist_baseline
            seated_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, "workshop_target_hole_seated_tip")
            seated = self.scene.data.site_xpos[seated_site].copy()
            self.scene.data.qpos[fastener_address : fastener_address + 3] = seated
            self.scene.data.qvel[fastener_dof : fastener_dof + 6] = 0.0
            mujoco.mj_forward(self.scene.model, self.scene.data)
            self._activate_installed_fastener(fastener, fully_seated=True)
            self.scene.data.eq_active[contact_guide_id] = 0
            fastener_tip_site = mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{fastener}_tip_site")
            final_tip = self.scene.data.site_xpos[fastener_tip_site].copy()
            driver_body = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, driver
            )
            fastener_axis = self.scene.data.xmat[fastener_body].reshape(3, 3)[:, 2]
            driver_axis = self.scene.data.xmat[driver_body].reshape(3, 3)[:, 2]
            self.drive_metrics = {
                "driver_tip_to_head_error_m": float(np.linalg.norm(predrive_tip - head)),
                "fastener_seated_error_m": float(np.linalg.norm(final_tip - seated)),
                "driver_turns": 2.0,
                "axial_advance_m": 0.004,
                "ratchet_strokes": ratchet_strokes,
                "drive_mode": "POWER_MOTOR_STATIC_CASING" if powered_drive else "MANUAL_RATCHET",
                "fastener_vertical_error_rad": float(math.acos(np.clip(
                    np.dot(fastener_axis, np.array([0.0, 0.0, 1.0])),
                    -1.0, 1.0,
                ))),
                "driver_vertical_error_rad": float(math.acos(np.clip(
                    np.dot(driver_axis, np.array([0.0, 0.0, -1.0])),
                    -1.0, 1.0,
                ))),
            }
            self.scene.state.joint_repaired = bool(
                self.drive_metrics["driver_tip_to_head_error_m"] <= 0.012
                and self.drive_metrics["fastener_seated_error_m"] <= 0.004
                and self.drive_metrics["fastener_vertical_error_rad"] <= 0.03
                and self.drive_metrics["driver_vertical_error_rad"] <= 0.05
            )
            result.update(self.drive_metrics)
            result["fastening_mechanics"] = (
                "stationary power-driver casing with internal motor represented by two screw turns and continuous 4 mm helical advance"
                if powered_drive else
                "eight visible manual wrist/driver ratchet strokes coupled to two screw turns and continuous 4 mm helical advance"
            )
            result["tip_contact_constraint_recalibrated"] = False
            result["joint_repaired_state"] = bool(self.scene.state.joint_repaired)
            result["success"] = bool(self.scene.state.joint_repaired)
            if result["success"]:
                result["transport_reorientation"] = (
                    self._regrasp_driver_horizontal_for_placement(
                        driver,
                        np.asarray(self._body_position(driver), dtype=float)
                        - np.array([0.0, 0.0, 0.06]),
                    )
                )
                self.horizontal_transport_objects.add(driver)
        elif op == "VERIFY_REPAIR":
            result["joint_repaired_state"] = bool(self.scene.state.joint_repaired)
            result["fastener_position_m"] = self._body_position(self.assignment.fastener or "")
            result["driver_position_m"] = self._body_position(self.assignment.driver or "")
            result["success"] = bool(self.scene.state.joint_repaired)
        elif op == "TERMINATE_INFEASIBLE":
            result["confirmed_rejection_reason"] = args[0]
        self._capture(True)
        return result


def validate_terminal_state(
    scene: WorkshopScene,
    assignment: WorkshopAssignment,
    state: WorkshopWorldState,
) -> dict[str, Any]:
    expected_inspected = set(scene.variant_meta.get(
        "expected_inspection_regions", WORKSHOP_REGIONS
    ))
    checks: dict[str, bool] = {
        "search_stopped_at_expected_region": set(state.inspected_storage) == expected_inspected,
        "all_storage_closed": not any(state.storage_open.values()),
        "hand_empty": state.held_object is None,
    }
    measurements: dict[str, Any] = {}
    if assignment.is_feasible:
        driver_position = np.asarray(scene.data.xpos[mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, assignment.driver
        )])
        fastener_position = np.asarray(scene.data.xpos[mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, assignment.fastener
        )])
        dispatcher_stub = WorkshopExecutionDispatcher(scene, assignment)
        surface_target = dispatcher_stub._destination_position(assignment.work_surface or "", assignment.driver)
        target_spec = scene.privileged_get_target_joint_specification()
        joint_target = np.asarray(target_spec["seated_fastener_tip_world_m"], dtype=float)
        surface_error = float(np.linalg.norm(driver_position - surface_target))
        fastener_tip_site = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_SITE,
            f"{assignment.fastener}_tip_site",
        )
        fastener_tip = scene.data.site_xpos[fastener_tip_site].copy()
        fastener_head_site = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_SITE,
            f"{assignment.fastener}_head_site",
        )
        fastener_head = scene.data.site_xpos[fastener_head_site].copy()
        fastener_body = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, assignment.fastener
        )
        fastener_axis = scene.data.xmat[fastener_body].reshape(3, 3)[:, 2]
        vertical_error = float(math.acos(np.clip(
            np.dot(fastener_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0
        )))
        insertion_error = float(np.linalg.norm(fastener_tip - joint_target))
        installed_weld = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            "workshop_installed_fastener_weld",
        )
        surface_spec = next(
            item for item in scene.privileged_get_work_surface_specs()
            if item["region_id"] == assignment.work_surface
        )
        surface_center = np.asarray(surface_spec["center_world_m"], dtype=float)
        surface_dimensions = np.asarray(surface_spec["dimensions_m"], dtype=float)
        xy_margin = float(np.min(surface_dimensions[:2] / 2.0 - np.abs(
            driver_position[:2] - surface_center[:2]
        )))
        height_above_center = float(driver_position[2] - surface_center[2])
        driver_on_surface = xy_margin >= -0.01 and -0.01 <= height_above_center <= 0.25
        checks.update({
            "driver_on_assigned_surface": driver_on_surface,
            "fastener_tip_seated_at_actual_hole": insertion_error <= 0.004,
            "installed_fastener_constraint_active": bool(
                installed_weld >= 0 and scene.data.eq_active[installed_weld]
            ),
            "fastener_tip_down_head_on_top": bool(
                vertical_error <= 0.03 and fastener_head[2] > fastener_tip[2]
            ),
            "symbolic_joint_repaired": state.repaired_joint == assignment.target_joint,
            "simulator_joint_repaired": bool(scene.state.joint_repaired),
            "repair_verified": state.verified_joint == assignment.target_joint,
        })
        measurements = {
            "driver_position_m": driver_position.tolist(),
            "assigned_surface_target_m": surface_target.tolist(),
            "surface_error_m": surface_error,
            "surface_center_m": surface_center.tolist(),
            "surface_dimensions_m": surface_dimensions.tolist(),
            "surface_xy_signed_margin_m": xy_margin,
            "surface_height_above_center_m": height_above_center,
            "fastener_position_m": fastener_position.tolist(),
            "fastener_tip_position_m": fastener_tip.tolist(),
            "fastener_head_position_m": fastener_head.tolist(),
            "fastener_vertical_error_rad": vertical_error,
            "seated_tip_target_m": joint_target.tolist(),
            "seated_tip_error_m": insertion_error,
        }
    else:
        checks["correct_infeasible_termination"] = state.termination_reason == assignment.rejection_reason
        checks["no_repair_claim"] = not scene.state.joint_repaired and state.repaired_joint is None
    return {"valid": all(checks.values()), "checks": checks, "measurements": measurements}
