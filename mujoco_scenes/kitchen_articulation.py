"""Contact-gated Google Robot articulation of kitchen storage.

The executor in this module is deliberately separate from
``KitchenScene.open_container``/``close_container``.  Those methods remain
deterministic perception adapters.  Here, container actuators are made passive
and the live joint moves only through the Google arm and a contact-preserving
handle weld.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import time
from typing import Any

import mujoco
import numpy as np

from mujoco_scenes.generic_manipulation import (
    ARM_COMMAND_SPEED,
    ProfiledIK,
    RobotConfigurationCollisionChecker,
)
from mujoco_scenes.kitchen_execution_policy import KitchenWorkspace
from mujoco_scenes.robot_profiles import manipulation_profile
from mujoco_scenes.robot_profiles import mobile_profile


class ArticulationFailureCode(str, Enum):
    WORKSPACE_PRECONDITION_UNSATISFIED = "WORKSPACE_PRECONDITION_UNSATISFIED"
    HAND_NOT_EMPTY_FOR_ARTICULATION = "HAND_NOT_EMPTY_FOR_ARTICULATION"
    HANDLE_SITE_MISSING = "HANDLE_SITE_MISSING"
    HANDLE_IK_FAILED = "HANDLE_IK_FAILED"
    HANDLE_APPROACH_COLLISION = "HANDLE_APPROACH_COLLISION"
    HANDLE_CONTACT_FAILED = "HANDLE_CONTACT_FAILED"
    HANDLE_GRASP_FAILED = "HANDLE_GRASP_FAILED"
    HANDLE_ATTACHMENT_FAILED = "HANDLE_ATTACHMENT_FAILED"
    HANDLE_ATTACHMENT_SNAP_EXCEEDED = "HANDLE_ATTACHMENT_SNAP_EXCEEDED"
    ARTICULATION_PATH_IK_FAILED = "ARTICULATION_PATH_IK_FAILED"
    ARTICULATION_PATH_COLLISION = "ARTICULATION_PATH_COLLISION"
    ARTICULATION_TRACKING_FAILED = "ARTICULATION_TRACKING_FAILED"
    ARTICULATION_WRONG_DIRECTION = "ARTICULATION_WRONG_DIRECTION"
    ARTICULATION_STALLED = "ARTICULATION_STALLED"
    OPEN_POSTCONDITION_FAILED = "OPEN_POSTCONDITION_FAILED"
    CLOSE_POSTCONDITION_FAILED = "CLOSE_POSTCONDITION_FAILED"
    UNEXPECTED_ARTICULATION_MOTION = "UNEXPECTED_ARTICULATION_MOTION"
    CONTAINER_DIRECT_ACTUATION_DETECTED = "CONTAINER_DIRECT_ACTUATION_DETECTED"
    RETREAT_FAILED = "RETREAT_FAILED"
    NAVIGATION_FAILED = "NAVIGATION_FAILED"


@dataclass(frozen=True)
class ArticulationSpec:
    container_id: str
    moving_body: str
    joint_name: str
    actuator_name: str
    joint_type: str
    handle_site: str
    handle_geoms: tuple[str, ...]
    attachment_name: str
    required_workspace: KitchenWorkspace
    closed_q: float
    open_q: float
    sample_count: int = 25
    final_tolerance: float = 0.018
    settled_velocity: float = 0.025
    pregrasp_distance: float = 0.080
    contact_offset: float = 0.002
    local_base_delta: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_yaw_world: float | None = None


ARTICULATION_SPECS = {
    "D1": ArticulationSpec(
        "D1", "drawer_D1_tray", "D1_slide_joint", "D1_slide_actuator",
        "PRISMATIC", "D1_handle_grasp",
        ("D1_handle_left", "D1_handle_right", "D1_handle_bar"),
        "google:container_grasp_D1", KitchenWorkspace.HOME, 0.0, 0.25,
        # Denser samples keep the arm on one collision-free IK branch as the
        # drawer passes the close-to-base portion of its stroke.
        sample_count=35, final_tolerance=0.020,
    ),
    "D2": ArticulationSpec(
        "D2", "drawer_D2_tray", "D2_slide_joint", "D2_slide_actuator",
        "PRISMATIC", "D2_handle_grasp",
        ("D2_handle_left", "D2_handle_right", "D2_handle_bar"),
        "google:container_grasp_D2", KitchenWorkspace.HOME, 0.0, 0.25,
        sample_count=21, final_tolerance=0.020,
    ),
    "C1": ArticulationSpec(
        "C1", "C1_door", "C1_door_joint", "C1_door_actuator",
        "HINGE", "C1_handle_grasp",
        ("C1_handle_left", "C1_handle_right", "C1_handle_bar"),
        "google:container_grasp_C1", KitchenWorkspace.LEFT_SIDE, 0.0, 1.40,
        sample_count=29, final_tolerance=0.035,
        local_base_delta=(0.0, -0.15, 0.0),
    ),
    "C2": ArticulationSpec(
        "C2", "C2_door", "C2_door_joint", "C2_door_actuator",
        "HINGE", "C2_handle_grasp",
        ("C2_handle_left", "C2_handle_right", "C2_handle_bar"),
        "google:container_grasp_C2", KitchenWorkspace.RIGHT_SIDE, 0.0, 1.40,
        sample_count=29, final_tolerance=0.035,
        local_base_delta=(0.0, 0.15, 0.0),
    ),
    "B1": ArticulationSpec(
        "B1", "B1_lid", "B1_lid_joint", "B1_lid_actuator",
        "HINGE", "B1_lid_handle_grasp",
        ("B1_lid_handle_left", "B1_lid_handle_right", "B1_lid_handle_bar"),
        "google:container_grasp_B1", KitchenWorkspace.RIGHT_SIDE, 0.0, 1.80,
        sample_count=37, final_tolerance=0.040,
    ),
}


FRONTAL_GRASP_ROTATION = np.array(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
)
WORKSPACE_YAWS = {
    KitchenWorkspace.HOME: 0.0,
    KitchenWorkspace.LEFT_SIDE: -math.pi / 2,
    KitchenWorkspace.RIGHT_SIDE: math.pi / 2,
}

# Menagerie's shoulder rotates concentrically inside the visual base housing.
# The D2 handle branch reaches a different yaw than ordinary top-down picks,
# where those two visual shells overlap without representing a link/link
# collision.  Keep this mechanism-specific calibration local so the frozen
# pick/place collision policy remains unchanged; every other link pair keeps
# the strict generic threshold.
ARTICULATION_MOUNT_ALLOWANCES = {
    frozenset(("google:base_link", "google:link_shoulder")): -0.150,
}


def _yaw_rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _quat_angle(first: np.ndarray, second: np.ndarray) -> float:
    dot = abs(float(np.dot(first, second)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


@dataclass
class ArticulationPlan:
    container: str
    action: str
    start_q: float
    target_q: float
    sampled_q: list[float]
    handle_positions: list[list[float]]
    arm_waypoints: list[list[float]]
    approach_joints: list[list[float]]
    pregrasp_joints: list[float]
    grasp_joints: list[float]
    retreat_joints: list[list[float]]
    ik_max_position_residual_m: float
    ik_max_angle_residual_rad: float
    collision_checked_segments: int
    status: str = "PLAN_VALID"


@dataclass
class KitchenExecutionResult:
    action_id: str
    action: str
    target_container: str
    required_workspace: str
    actual_workspace: str
    success: bool = False
    status: str = "PENDING"
    failure_code: str | None = None
    message: str = ""
    initial_articulation_q: float | None = None
    target_semantic_q: float | None = None
    final_articulation_q: float | None = None
    final_joint_error: float | None = None
    final_joint_velocity: float | None = None
    maximum_joint_velocity: float = 0.0
    handle_contact_sides: list[int] = field(default_factory=list)
    handle_contact_evidence: bool = False
    handle_attachment_evidence: bool = False
    attachment_translation_snap_m: float | None = None
    attachment_angle_snap_rad: float | None = None
    handle_pose_trajectory_count: int = 0
    arm_waypoint_count: int = 0
    ik_max_position_residual_m: float | None = None
    ik_max_angle_residual_rad: float | None = None
    collision_status: str = "NOT_RUN"
    unexpected_articulation_motion: bool = False
    direct_container_actuator_used: bool = False
    live_qpos_write_used: bool = False
    physical_motion_source: str = "GOOGLE_ROBOT_HANDLE_MANIPULATION"
    total_physics_steps: int = 0
    total_duration_s: float = 0.0
    final_postcondition: str | None = None
    local_base_approach_used: bool = False
    local_base_delta: list[float] = field(default_factory=list)
    local_base_retracted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArticulationExecutionError(RuntimeError):
    def __init__(self, code: ArticulationFailureCode, message: str):
        super().__init__(message)
        self.code = code


class GoogleKitchenArticulationExecutor:
    """Plan and synchronously execute one physical handle articulation."""

    def __init__(self, scene, *, held_object_getter=None, step_callback=None):
        if scene.robot_name != "google":
            raise ValueError("Physical kitchen articulation requires Google Robot")
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.profile = manipulation_profile("google")
        self.mobile_profile = mobile_profile("google")
        self.held_object_getter = held_object_getter or (lambda: None)
        self.step_callback = step_callback
        self.arm_joint_ids = self._ids(mujoco.mjtObj.mjOBJ_JOINT, self.profile.arm_joints)
        self.arm_qpos = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_dofs = self.model.jnt_dofadr[self.arm_joint_ids]
        self.arm_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.arm_actuators
        )
        self.finger_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.profile.finger_actuators
        )
        self.gripper_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, self.profile.gripper_body
        )
        self.grip_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, self.profile.grip_site)
        self.base_joint_ids = self._ids(
            mujoco.mjtObj.mjOBJ_JOINT, self.mobile_profile.base_joints
        )
        self.base_qpos = self.model.jnt_qposadr[self.base_joint_ids]
        self.base_dofs = self.model.jnt_dofadr[self.base_joint_ids]
        self.base_actuators = self._ids(
            mujoco.mjtObj.mjOBJ_ACTUATOR, self.mobile_profile.base_actuators
        )
        self._action_serial = 0
        self._successful_open_plans: dict[str, ArticulationPlan] = {}
        self._live_qpos_write_used = False
        self._direct_container_actuator_used = False

    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        if self.step_callback is not None:
            self.step_callback()

    def _run_base_target(self, target: np.ndarray, max_steps: int = 12000) -> int:
        """Physically execute the short container-specific fine approach."""
        settled = 0
        rate = np.array((0.18, 0.18, 0.45)) * self.model.opt.timestep
        for step in range(max_steps):
            current_ctrl = self.data.ctrl[self.base_actuators]
            self.data.ctrl[self.base_actuators] = current_ctrl + np.clip(
                target - current_ctrl, -rate, rate
            )
            self._step()
            position_error = np.abs(target - self.data.qpos[self.base_qpos])
            velocity = np.abs(self.data.qvel[self.base_dofs])
            if (
                float(np.max(position_error[:2])) < 0.003
                and float(position_error[2]) < math.radians(0.3)
                and float(np.max(velocity[:2])) < 0.004
                and float(velocity[2]) < 0.012
            ):
                settled += 1
                if settled >= 30:
                    return step + 1
            else:
                settled = 0
        raise ArticulationExecutionError(
            ArticulationFailureCode.NAVIGATION_FAILED,
            f"Fine base approach failed to settle at {target.tolist()}",
        )

    def _id(self, kind, name: str) -> int:
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ArticulationExecutionError(
                ArticulationFailureCode.HANDLE_SITE_MISSING, f"Missing model name: {name}"
            )
        return value

    def _ids(self, kind, names) -> np.ndarray:
        return np.asarray([self._id(kind, name) for name in names], dtype=int)

    def _resolved(self, spec: ArticulationSpec) -> dict[str, Any]:
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
        return {
            "joint_id": joint_id,
            "qpos": int(self.model.jnt_qposadr[joint_id]),
            "dof": int(self.model.jnt_dofadr[joint_id]),
            "actuator": self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, spec.actuator_name),
            "site": self._id(mujoco.mjtObj.mjOBJ_SITE, spec.handle_site),
            "body": self._id(mujoco.mjtObj.mjOBJ_BODY, spec.moving_body),
            "equality": self._id(mujoco.mjtObj.mjOBJ_EQUALITY, spec.attachment_name),
            "handle_geoms": frozenset(
                self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in spec.handle_geoms
            ),
        }

    def _assert_preconditions(
        self, spec: ArticulationSpec, current_workspace: KitchenWorkspace
    ) -> None:
        if current_workspace != spec.required_workspace:
            raise ArticulationExecutionError(
                ArticulationFailureCode.WORKSPACE_PRECONDITION_UNSATISFIED,
                f"{spec.container_id} requires {spec.required_workspace.value}; "
                f"actual workspace is {current_workspace.value}",
            )
        if self.held_object_getter() is not None:
            raise ArticulationExecutionError(
                ArticulationFailureCode.HAND_NOT_EMPTY_FOR_ARTICULATION,
                "Container articulation requires an empty gripper",
            )
        for equality_id in range(self.model.neq):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
            ) or ""
            if self.data.eq_active[equality_id] and (
                name.startswith("google:pick_")
                or name.startswith("google:container_grasp_")
            ):
                raise ArticulationExecutionError(
                    ArticulationFailureCode.HAND_NOT_EMPTY_FOR_ARTICULATION,
                    f"Active grasp constraint: {name}",
                )

    def _grasp_rotation(
        self, workspace: KitchenWorkspace, spec: ArticulationSpec | None = None
    ) -> np.ndarray:
        yaw = (
            spec.grasp_yaw_world
            if spec is not None and spec.grasp_yaw_world is not None
            else WORKSPACE_YAWS[workspace]
        )
        return _yaw_rotation(yaw) @ FRONTAL_GRASP_ROTATION

    def _collision_free_ik(
        self,
        ik: ProfiledIK,
        collision: RobotConfigurationCollisionChecker,
        position: np.ndarray,
        rotation: np.ndarray,
        preferred_seed: np.ndarray,
        allowed: frozenset[int],
        *,
        previous: np.ndarray | None = None,
        seed_key: int = 0,
    ) -> tuple[np.ndarray, float, float]:
        """Select a deterministic collision-free IK branch."""
        rng = np.random.default_rng(4100 + seed_key)
        seeds = [preferred_seed, self.profile.home_seed]
        seeds.extend(
            ik.lower + (ik.upper - ik.lower) * rng.random(len(ik.lower))
            for _ in range(48)
        )
        last_reason = "no converged IK branch"
        for seed in seeds:
            candidate, pe, ae = ik.solve(position, seed, rotation)
            if pe > 0.015 or ae > math.radians(3.0):
                continue
            valid, reason = collision.evaluate(candidate, allowed)
            if not valid:
                last_reason = str(reason)
                continue
            if previous is not None:
                # Reject coordinate-wrap branches before the dense segment
                # query; passing a multi-radian wrap into MuJoCo's native
                # distance sweep is both undesirable motion and numerically
                # fragile. Ordinary continuous branches are then checked at
                # dense resolution below.
                if float(np.max(np.abs(candidate - previous))) > 2.40:
                    last_reason = "IK branch discontinuity"
                    continue
                valid, reason = collision.segment_valid(
                    previous, candidate, allowed, resolution=0.012
                )
                if not valid:
                    last_reason = str(reason)
                    continue
            return candidate, pe, ae
        raise ArticulationExecutionError(
            ArticulationFailureCode.ARTICULATION_PATH_COLLISION, last_reason
        )

    def _joint_rrt(
        self,
        collision: RobotConfigurationCollisionChecker,
        start: np.ndarray,
        goal: np.ndarray,
        allowed: frozenset[int],
        *,
        seed_key: int,
    ) -> list[np.ndarray]:
        """Small deterministic joint-space connector for handle approach."""
        direct, _reason = collision.segment_valid(
            start, goal, allowed, resolution=0.018
        )
        if direct:
            return [goal.copy()]
        rng = np.random.default_rng(9200 + seed_key)
        nodes = [start.copy()]
        parents = [-1]
        step_size = 0.28
        for iteration in range(7000):
            sample = goal if iteration % 5 == 0 else (
                collision.data.qpos[collision.arm_qpos] * 0.0
            )
            if iteration % 5:
                # ProfiledIK exposes the authoritative padded joint bounds.
                helper = ProfiledIK(self.model, self.data, self.profile)
                sample = helper.lower + (helper.upper - helper.lower) * rng.random(7)
            distances = [float(np.linalg.norm(node - sample)) for node in nodes]
            nearest_index = int(np.argmin(distances))
            nearest = nodes[nearest_index]
            delta = sample - nearest
            norm = float(np.linalg.norm(delta))
            candidate = sample if norm <= step_size else nearest + step_size * delta / norm
            valid, _ = collision.segment_valid(
                nearest, candidate, allowed, resolution=0.025
            )
            if not valid:
                continue
            nodes.append(candidate.copy())
            parents.append(nearest_index)
            connected, _ = collision.segment_valid(
                candidate, goal, allowed, resolution=0.018
            )
            if not connected:
                continue
            path = [goal.copy(), candidate.copy()]
            cursor = len(nodes) - 1
            while parents[cursor] >= 0:
                cursor = parents[cursor]
                path.append(nodes[cursor].copy())
            path.reverse()
            # Greedy shortcut preserves checked segments.
            smoothed = [path[0]]
            index = 0
            while index < len(path) - 1:
                next_index = len(path) - 1
                while next_index > index + 1:
                    ok, _ = collision.segment_valid(
                        path[index], path[next_index], allowed, resolution=0.018
                    )
                    if ok:
                        break
                    next_index -= 1
                smoothed.append(path[next_index])
                index = next_index
            return smoothed[1:]
        raise ArticulationExecutionError(
            ArticulationFailureCode.HANDLE_APPROACH_COLLISION,
            "No collision-free joint-space route to handle pregrasp",
        )

    def plan(
        self,
        action: str,
        container: str,
        current_workspace: KitchenWorkspace,
        *,
        target_q_override: float | None = None,
    ) -> ArticulationPlan:
        action = action.upper()
        if action not in {"OPEN", "CLOSE"}:
            raise ValueError("Articulation action must be OPEN or CLOSE")
        spec = ARTICULATION_SPECS[container]
        self._assert_preconditions(spec, current_workspace)
        ids = self._resolved(spec)
        mujoco.mj_forward(self.model, self.data)
        start_q = float(self.data.qpos[ids["qpos"]])
        target_q = (
            float(target_q_override) if target_q_override is not None
            else (spec.open_q if action == "OPEN" else spec.closed_q)
        )
        if (
            action == "CLOSE"
            and target_q_override is None
            and container in self._successful_open_plans
        ):
            return self._reverse_open_plan(
                self._successful_open_plans[container], spec, ids,
                current_workspace, start_q,
            )
        sampled = np.linspace(start_q, target_q, spec.sample_count)
        planning = mujoco.MjData(self.model)
        planning.qpos[:] = self.data.qpos
        planning.qvel[:] = 0.0
        planning.ctrl[:] = self.data.ctrl
        planning.qpos[self.arm_qpos] = self.profile.navigation_joints
        planning.qpos[ids["qpos"]] = sampled[0]
        mujoco.mj_forward(self.model, planning)
        initial_handle_rotation = planning.site_xmat[ids["site"]].reshape(3, 3).copy()
        initial_grasp_rotation = self._grasp_rotation(current_workspace, spec)
        approach_axis = initial_grasp_rotation[:, 2]
        contact = planning.site_xpos[ids["site"]].copy() + spec.contact_offset * approach_axis
        pregrasp = contact - spec.pregrasp_distance * approach_axis
        ik = ProfiledIK(self.model, planning, self.profile)
        collision = RobotConfigurationCollisionChecker(
            self.model, planning, self.profile,
            mounting_allowances=ARTICULATION_MOUNT_ALLOWANCES,
        )
        allowed = frozenset((ids["body"],))
        max_position = 0.0
        max_angle = 0.0
        collision_segments = 0
        pre_q, pe, ae = self._collision_free_ik(
            ik, collision, pregrasp, initial_grasp_rotation,
            self.profile.home_seed, frozenset(), seed_key=10 + ord(container[0]),
        )
        max_position, max_angle = max(max_position, pe), max(max_angle, ae)
        if pe > 0.015 or ae > math.radians(3.0):
            raise ArticulationExecutionError(
                ArticulationFailureCode.HANDLE_IK_FAILED,
                f"Pregrasp IK residual {pe:.4f} m / {math.degrees(ae):.2f} deg",
            )
        collision.reference_qpos = planning.qpos.copy()
        approach_route = self._joint_rrt(
            collision, self.profile.navigation_joints, pre_q, frozenset(),
            seed_key=sum(map(ord, container)),
        )
        collision_segments += len(approach_route)
        grasp_q, pe, ae = self._collision_free_ik(
            ik, collision, contact, initial_grasp_rotation, pre_q, allowed,
            previous=pre_q, seed_key=30 + sum(map(ord, container)),
        )
        max_position, max_angle = max(max_position, pe), max(max_angle, ae)
        if pe > 0.012 or ae > math.radians(2.0):
            raise ArticulationExecutionError(
                ArticulationFailureCode.HANDLE_IK_FAILED,
                f"Contact IK residual {pe:.4f} m / {math.degrees(ae):.2f} deg",
            )
        valid, reason = collision.segment_valid(pre_q, grasp_q, allowed, resolution=0.018)
        collision_segments += 1
        if not valid:
            raise ArticulationExecutionError(
                ArticulationFailureCode.HANDLE_APPROACH_COLLISION, str(reason)
            )
        arm_waypoints: list[np.ndarray] = []
        handle_positions: list[np.ndarray] = []
        seed = grasp_q
        for q in sampled[1:]:
            planning.qpos[:] = self.data.qpos
            planning.qvel[:] = 0.0
            planning.qpos[ids["qpos"]] = q
            planning.qpos[self.arm_qpos] = seed
            mujoco.mj_forward(self.model, planning)
            handle_position = planning.site_xpos[ids["site"]].copy()
            handle_rotation = planning.site_xmat[ids["site"]].reshape(3, 3).copy()
            target_rotation = (
                handle_rotation @ initial_handle_rotation.T @ initial_grasp_rotation
            )
            ik.data.qpos[:] = planning.qpos
            collision.reference_qpos = planning.qpos.copy()
            candidate, pe, ae = self._collision_free_ik(
                ik, collision,
                handle_position + spec.contact_offset * target_rotation[:, 2],
                target_rotation, seed, allowed, previous=seed,
                seed_key=int(round(q * 1000)) + sum(map(ord, container)),
            )
            max_position, max_angle = max(max_position, pe), max(max_angle, ae)
            if pe > 0.015 or ae > math.radians(3.0):
                raise ArticulationExecutionError(
                    ArticulationFailureCode.ARTICULATION_PATH_IK_FAILED,
                    f"q={q:.4f}: residual {pe:.4f} m / {math.degrees(ae):.2f} deg",
                )
            collision_segments += 1
            arm_waypoints.append(candidate.copy())
            handle_positions.append(handle_position.copy())
            seed = candidate
        final_approach_axis = target_rotation[:, 2]
        final_pregrasp = (
            handle_position + spec.contact_offset * final_approach_axis
            - spec.pregrasp_distance * final_approach_axis
        )
        final_pre_q, final_pe, final_ae = self._collision_free_ik(
            ik, collision, final_pregrasp, target_rotation, seed,
            frozenset(), seed_key=3100 + sum(map(ord, container)),
        )
        max_position = max(max_position, final_pe)
        max_angle = max(max_angle, final_ae)
        valid, reason = collision.segment_valid(
            seed, final_pre_q, allowed, resolution=0.012
        )
        if not valid:
            raise ArticulationExecutionError(
                ArticulationFailureCode.ARTICULATION_PATH_COLLISION, str(reason)
            )
        final_retreat = self._joint_rrt(
            collision, final_pre_q, self.profile.navigation_joints,
            frozenset(), seed_key=4100 + sum(map(ord, container)),
        )
        return ArticulationPlan(
            container=container,
            action=action,
            start_q=start_q,
            target_q=target_q,
            sampled_q=sampled.tolist(),
            handle_positions=[item.tolist() for item in handle_positions],
            arm_waypoints=[item.tolist() for item in arm_waypoints],
            approach_joints=[item.tolist() for item in approach_route],
            pregrasp_joints=pre_q.tolist(),
            grasp_joints=grasp_q.tolist(),
            retreat_joints=[
                final_pre_q.tolist(),
                *[item.tolist() for item in final_retreat],
            ],
            ik_max_position_residual_m=max_position,
            ik_max_angle_residual_rad=max_angle,
            collision_checked_segments=collision_segments,
        )

    def _reverse_open_plan(
        self,
        open_plan: ArticulationPlan,
        spec: ArticulationSpec,
        ids: dict[str, Any],
        workspace: KitchenWorkspace,
        start_q: float,
    ) -> ArticulationPlan:
        """Construct CLOSE from the already validated OPEN kinematic branch."""
        planning = mujoco.MjData(self.model)
        planning.qpos[:] = self.data.qpos
        planning.qvel[:] = 0.0
        open_targets = [
            np.asarray(open_plan.grasp_joints),
            *[np.asarray(item) for item in open_plan.arm_waypoints],
        ]
        grasp_q = open_targets[-1]
        planning.qpos[self.arm_qpos] = grasp_q
        mujoco.mj_forward(self.model, planning)
        grasp_rotation = planning.site_xmat[self.grip_site_id].reshape(3, 3).copy()
        approach_axis = grasp_rotation[:, 2]
        contact = planning.site_xpos[ids["site"]].copy() + spec.contact_offset * approach_axis
        pregrasp = contact - spec.pregrasp_distance * approach_axis
        ik = ProfiledIK(self.model, planning, self.profile)
        collision = RobotConfigurationCollisionChecker(
            self.model, planning, self.profile,
            mounting_allowances=ARTICULATION_MOUNT_ALLOWANCES,
        )
        collision.reference_qpos = planning.qpos.copy()
        allowed = frozenset((ids["body"],))
        pre_q, pe, ae = self._collision_free_ik(
            ik, collision, pregrasp, grasp_rotation, grasp_q, frozenset(),
            seed_key=700 + sum(map(ord, spec.container_id)),
        )
        approach_route = self._joint_rrt(
            collision, self.profile.navigation_joints, pre_q, frozenset(),
            seed_key=1700 + sum(map(ord, spec.container_id)),
        )
        valid, reason = collision.segment_valid(
            pre_q, grasp_q, allowed, resolution=0.012
        )
        if not valid:
            raise ArticulationExecutionError(
                ArticulationFailureCode.HANDLE_APPROACH_COLLISION, str(reason)
            )
        close_targets = list(reversed(open_targets[:-1]))
        return ArticulationPlan(
            container=spec.container_id,
            action="CLOSE",
            start_q=start_q,
            target_q=spec.closed_q,
            sampled_q=np.linspace(
                start_q, spec.closed_q, len(close_targets) + 1
            ).tolist(),
            handle_positions=list(reversed(open_plan.handle_positions)),
            arm_waypoints=[item.tolist() for item in close_targets],
            approach_joints=[item.tolist() for item in approach_route],
            pregrasp_joints=pre_q.tolist(),
            grasp_joints=grasp_q.tolist(),
            retreat_joints=[
                *[item.tolist() for item in reversed(approach_route[:-1])],
                self.profile.navigation_joints.tolist(),
            ],
            ik_max_position_residual_m=pe,
            ik_max_angle_residual_rad=ae,
            collision_checked_segments=len(close_targets) + len(approach_route) + 1,
        )

    def _run_arm_target(
        self,
        target: np.ndarray,
        max_steps: int = 12000,
        tracking_tolerance: float = 0.018,
    ) -> int:
        hold = 0
        for step in range(max_steps):
            current = self.data.ctrl[self.arm_actuators]
            maximum = ARM_COMMAND_SPEED * self.model.opt.timestep
            self.data.ctrl[self.arm_actuators] = current + np.clip(
                target - current, -maximum, maximum
            )
            self._step()
            command_error = float(np.max(np.abs(target - self.data.ctrl[self.arm_actuators])))
            tracking_error = float(np.max(np.abs(target - self.data.qpos[self.arm_qpos])))
            if command_error < 0.001 and tracking_error < tracking_tolerance:
                hold += 1
                if hold >= 8:
                    return step + 1
            else:
                hold = 0
        raise ArticulationExecutionError(
            ArticulationFailureCode.ARTICULATION_TRACKING_FAILED,
            "Arm target did not settle; "
            f"command_error={float(np.max(np.abs(target - self.data.ctrl[self.arm_actuators]))):.4f}, "
            f"tracking_error={float(np.max(np.abs(target - self.data.qpos[self.arm_qpos]))):.4f}",
        )

    def _handle_contact_sides(self, ids: dict[str, Any]) -> set[int]:
        sides: set[int] = set()
        for contact in self.data.contact:
            if contact.geom1 in ids["handle_geoms"]:
                other = contact.geom2
            elif contact.geom2 in ids["handle_geoms"]:
                other = contact.geom1
            else:
                continue
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, other
            ) or ""
            for side, names in enumerate(self.profile.finger_contact_geoms):
                if name in names:
                    sides.add(side)
        return sides

    def _close_until_contact(self, ids: dict[str, Any], max_steps: int = 2500) -> tuple[int, set[int]]:
        stable = 0
        sides: set[int] = set()
        command = float(self.profile.open_command)
        for step in range(max_steps):
            command = min(self.profile.closed_command, command + self.profile.close_step)
            self.data.ctrl[self.finger_actuators] = command
            self._step()
            sides = self._handle_contact_sides(ids)
            stable = stable + 1 if sides == {0, 1} else 0
            if stable >= 12:
                return step + 1, sides
        raise ArticulationExecutionError(
            ArticulationFailureCode.HANDLE_CONTACT_FAILED,
            f"Bilateral handle contact not established; sides={sorted(sides)}",
        )

    def _set_live_relative_weld(self, equality_id: int, body_id: int) -> None:
        inverse_pos = np.empty(3)
        inverse_quat = np.empty(4)
        relative_pos = np.empty(3)
        relative_quat = np.empty(4)
        mujoco.mju_negPose(
            inverse_pos, inverse_quat,
            self.data.xpos[self.gripper_body_id], self.data.xquat[self.gripper_body_id],
        )
        mujoco.mju_mulPose(
            relative_pos, relative_quat, inverse_pos, inverse_quat,
            self.data.xpos[body_id], self.data.xquat[body_id],
        )
        self.model.eq_data[equality_id, 3:6] = relative_pos
        self.model.eq_data[equality_id, 6:10] = relative_quat

    def _make_actuator_passive(self, actuator_id: int):
        saved = (
            self.model.actuator_gainprm[actuator_id].copy(),
            self.model.actuator_biasprm[actuator_id].copy(),
            float(self.data.ctrl[actuator_id]),
        )
        self.model.actuator_gainprm[actuator_id] = 0.0
        self.model.actuator_biasprm[actuator_id] = 0.0
        self.data.ctrl[actuator_id] = 0.0
        return saved

    def _boost_arm_tracking(self):
        saved = (
            self.model.actuator_gainprm[self.arm_actuators].copy(),
            self.model.actuator_biasprm[self.arm_actuators].copy(),
        )
        self.model.actuator_gainprm[self.arm_actuators, 0] *= 1.8
        self.model.actuator_biasprm[self.arm_actuators, 1] *= 1.8
        return saved

    def _restore_arm_tracking(self, saved) -> None:
        gain, bias = saved
        self.model.actuator_gainprm[self.arm_actuators] = gain
        self.model.actuator_biasprm[self.arm_actuators] = bias

    def _restore_actuator(self, actuator_id: int, saved, hold_q: float) -> None:
        gain, bias, _ctrl = saved
        self.model.actuator_gainprm[actuator_id] = gain
        self.model.actuator_biasprm[actuator_id] = bias
        # Restore as a neutral hold at the state physically reached, never as
        # an OPEN/CLOSE target command.
        self.data.ctrl[actuator_id] = hold_q

    def execute(
        self,
        action: str,
        container: str,
        current_workspace: KitchenWorkspace,
        *,
        max_steps_per_waypoint: int = 12000,
        target_q_override: float | None = None,
    ) -> KitchenExecutionResult:
        action = action.upper()
        spec = ARTICULATION_SPECS[container]
        self._action_serial += 1
        result = KitchenExecutionResult(
            action_id=f"phaseA-{self._action_serial:05d}", action=action,
            target_container=container,
            required_workspace=spec.required_workspace.value,
            actual_workspace=current_workspace.value,
        )
        started = time.perf_counter()
        saved_actuator = None
        saved_arm_tracking = self._boost_arm_tracking()
        ids = None
        base_start = self.data.qpos[self.base_qpos].copy()
        local_base_active = False
        try:
            self._assert_preconditions(spec, current_workspace)
            ids = self._resolved(spec)
            initial_all = {
                key: float(self.data.qpos[self._resolved(other)["qpos"]])
                for key, other in ARTICULATION_SPECS.items()
            }
            start_q = float(self.data.qpos[ids["qpos"]])
            target_q = (
                float(target_q_override) if target_q_override is not None
                else (spec.open_q if action == "OPEN" else spec.closed_q)
            )
            result.initial_articulation_q = start_q
            result.target_semantic_q = target_q
            if abs(start_q - target_q) <= spec.final_tolerance:
                result.success = True
                result.status = "ALREADY_SATISFIED"
                result.final_articulation_q = start_q
                result.final_joint_error = abs(start_q - target_q)
                result.final_joint_velocity = abs(float(self.data.qvel[ids["dof"]]))
                result.final_postcondition = action
                return result
            local_delta = np.asarray(spec.local_base_delta, dtype=float)
            if np.any(np.abs(local_delta) > 0.0):
                result.local_base_approach_used = True
                result.local_base_delta = local_delta.tolist()
                result.total_physics_steps += self._run_base_target(
                    base_start + local_delta
                )
                local_base_active = True
            plan = self.plan(
                action, container, current_workspace,
                target_q_override=target_q_override,
            )
            result.handle_pose_trajectory_count = len(plan.sampled_q)
            result.arm_waypoint_count = (
                len(plan.approach_joints) + len(plan.arm_waypoints)
                + len(plan.retreat_joints) + 1
            )
            result.ik_max_position_residual_m = plan.ik_max_position_residual_m
            result.ik_max_angle_residual_rad = plan.ik_max_angle_residual_rad
            result.collision_status = "PLAN_VALID_AND_LIVE_GUARD_ACTIVE"
            for approach_target in plan.approach_joints:
                result.total_physics_steps += self._run_arm_target(
                    np.asarray(approach_target), max_steps_per_waypoint,
                    tracking_tolerance=0.050,
                )
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            for _ in range(100):
                self._step()
            result.total_physics_steps += 100
            result.total_physics_steps += self._run_arm_target(
                np.asarray(plan.grasp_joints), max_steps_per_waypoint,
                tracking_tolerance=0.060,
            )
            steps, sides = self._close_until_contact(ids)
            result.total_physics_steps += steps
            result.handle_contact_sides = sorted(sides)
            result.handle_contact_evidence = sides == {0, 1}
            before_position = self.data.site_xpos[ids["site"]].copy()
            before_quat = self.data.xquat[ids["body"]].copy()
            self._set_live_relative_weld(ids["equality"], ids["body"])
            self.data.eq_active[ids["equality"]] = 1
            self._step()
            result.total_physics_steps += 1
            mujoco.mj_forward(self.model, self.data)
            result.attachment_translation_snap_m = float(
                np.linalg.norm(self.data.site_xpos[ids["site"]] - before_position)
            )
            result.attachment_angle_snap_rad = _quat_angle(
                before_quat, self.data.xquat[ids["body"]]
            )
            if result.attachment_translation_snap_m > 0.002:
                raise ArticulationExecutionError(
                    ArticulationFailureCode.HANDLE_ATTACHMENT_SNAP_EXCEEDED,
                    f"Attachment snap {result.attachment_translation_snap_m:.6f} m",
                )
            result.handle_attachment_evidence = True
            saved_actuator = self._make_actuator_passive(ids["actuator"])
            previous_q = float(self.data.qpos[ids["qpos"]])
            direction = math.copysign(1.0, target_q - previous_q)
            for waypoint_index, arm_target in enumerate(plan.arm_waypoints):
                result.total_physics_steps += self._run_arm_target(
                    np.asarray(arm_target), max_steps_per_waypoint,
                    tracking_tolerance=0.055,
                )
                live_q = float(self.data.qpos[ids["qpos"]])
                result.maximum_joint_velocity = max(
                    result.maximum_joint_velocity,
                    abs(float(self.data.qvel[ids["dof"]])),
                )
                if direction * (live_q - previous_q) < -0.012:
                    raise ArticulationExecutionError(
                        ArticulationFailureCode.ARTICULATION_WRONG_DIRECTION,
                        f"Joint reversed from {previous_q:.4f} to {live_q:.4f}",
                    )
                previous_q = live_q
            settled = 0
            last_arm_target = np.asarray(plan.arm_waypoints[-1])
            for _ in range(2500):
                current = self.data.ctrl[self.arm_actuators]
                maximum = ARM_COMMAND_SPEED * self.model.opt.timestep
                self.data.ctrl[self.arm_actuators] = current + np.clip(
                    last_arm_target - current, -maximum, maximum
                )
                self._step()
                result.total_physics_steps += 1
                joint_error = abs(float(self.data.qpos[ids["qpos"]]) - target_q)
                joint_velocity = abs(float(self.data.qvel[ids["dof"]]))
                settled = settled + 1 if (
                    joint_error <= spec.final_tolerance
                    and joint_velocity <= spec.settled_velocity
                ) else 0
                if settled >= 50:
                    break
            if settled < 50:
                raise ArticulationExecutionError(
                    ArticulationFailureCode.ARTICULATION_TRACKING_FAILED,
                    "Container did not settle at the robot-driven target before release",
                )
            self.data.ctrl[self.finger_actuators] = self.profile.open_command
            for _ in range(150):
                self._step()
            result.total_physics_steps += 150
            self.data.eq_active[ids["equality"]] = 0
            final_physical_q = float(self.data.qpos[ids["qpos"]])
            self._restore_actuator(ids["actuator"], saved_actuator, final_physical_q)
            for retreat_target in plan.retreat_joints:
                result.total_physics_steps += self._run_arm_target(
                    np.asarray(retreat_target), max_steps_per_waypoint,
                    tracking_tolerance=0.050,
                )
            if local_base_active:
                result.total_physics_steps += self._run_base_target(base_start)
                local_base_active = False
                result.local_base_retracted = True
            final_settle = 0
            for _ in range(1500):
                self._step()
                result.total_physics_steps += 1
                if (
                    abs(float(self.data.qpos[ids["qpos"]]) - target_q)
                    <= spec.final_tolerance
                    and abs(float(self.data.qvel[ids["dof"]]))
                    <= spec.settled_velocity
                ):
                    final_settle += 1
                    if final_settle >= 50:
                        break
                else:
                    final_settle = 0
            final_q = float(self.data.qpos[ids["qpos"]])
            final_velocity = abs(float(self.data.qvel[ids["dof"]]))
            wrong_motion = {
                key: abs(
                    float(self.data.qpos[self._resolved(other)["qpos"]])
                    - initial_all[key]
                )
                for key, other in ARTICULATION_SPECS.items()
                if key != container
            }
            result.unexpected_articulation_motion = any(
                delta > 0.015 for delta in wrong_motion.values()
            )
            result.final_articulation_q = final_q
            result.final_joint_error = abs(final_q - target_q)
            result.final_joint_velocity = final_velocity
            result.live_qpos_write_used = self._live_qpos_write_used
            result.direct_container_actuator_used = self._direct_container_actuator_used
            if result.unexpected_articulation_motion:
                raise ArticulationExecutionError(
                    ArticulationFailureCode.UNEXPECTED_ARTICULATION_MOTION,
                    f"Unexpected joint deltas: {wrong_motion}",
                )
            if result.final_joint_error > spec.final_tolerance or final_velocity > spec.settled_velocity:
                code = (
                    ArticulationFailureCode.OPEN_POSTCONDITION_FAILED
                    if action == "OPEN" else ArticulationFailureCode.CLOSE_POSTCONDITION_FAILED
                )
                raise ArticulationExecutionError(
                    code,
                    f"q={final_q:.4f}, target={target_q:.4f}, velocity={final_velocity:.4f}",
                )
            if self.data.eq_active[ids["equality"]]:
                raise ArticulationExecutionError(
                    ArticulationFailureCode.RETREAT_FAILED,
                    "Handle attachment remained active",
                )
            result.success = True
            result.status = "EXECUTION_SUCCESS"
            result.final_postcondition = action
            if action == "OPEN" and target_q_override is None:
                self._successful_open_plans[container] = plan
            elif action == "CLOSE" and target_q_override is None:
                self._successful_open_plans.pop(container, None)
        except ArticulationExecutionError as error:
            result.failure_code = error.code.value
            result.message = str(error)
            result.status = "EXECUTION_FAILED"
            try:
                ids = self._resolved(spec)
                self.data.eq_active[ids["equality"]] = 0
                result.final_articulation_q = float(self.data.qpos[ids["qpos"]])
                result.final_joint_velocity = abs(float(self.data.qvel[ids["dof"]]))
            except Exception:
                pass
        finally:
            if ids is not None:
                self.data.eq_active[ids["equality"]] = 0
                if saved_actuator is not None:
                    current_q = float(self.data.qpos[ids["qpos"]])
                    self._restore_actuator(
                        ids["actuator"], saved_actuator, current_q
                    )
            self._restore_arm_tracking(saved_arm_tracking)
            if local_base_active and float(np.max(np.abs(
                self.data.qpos[self.arm_qpos] - self.profile.navigation_joints
            ))) < 0.08:
                try:
                    result.total_physics_steps += self._run_base_target(base_start)
                    result.local_base_retracted = True
                except ArticulationExecutionError:
                    result.local_base_retracted = False
            result.total_duration_s = time.perf_counter() - started
        return result


def articulation_specs_as_dict() -> dict[str, Any]:
    return {key: asdict(value) for key, value in ARTICULATION_SPECS.items()}
