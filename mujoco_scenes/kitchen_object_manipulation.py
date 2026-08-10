"""Physical Google PICK/PLACE primitives for perception-resolved kitchen objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any

import mujoco
import numpy as np

from .generic_manipulation import (
    CalibratedPickPlaceExecutor,
    GOOGLE_PICK_SPECS,
    GOOGLE_SPOON_TOP_DOWN_ROTATION,
    SimplePickSpec,
)
from .kitchen_execution_entities import ObjectSourceContext
from .kitchen_execution_policy import KitchenWorkspace
from .robot_profiles import manipulation_profile


PHASE_B_MOUNT_ALLOWANCES = {
    # Menagerie visual shells overlap at the concentric shoulder mount.  This
    # execution-local allowance is bounded to that mechanical pair; all other
    # self-collisions remain strict.
    frozenset(("google:base_link", "google:link_shoulder")): -0.055,
}


class ObjectExecutionFailureCode(str, Enum):
    ENTITY_RESOLUTION_FAILED = "ENTITY_RESOLUTION_FAILED"
    SOURCE_CONTEXT_UNKNOWN = "SOURCE_CONTEXT_UNKNOWN"
    SOURCE_RELATION_INVALID = "SOURCE_RELATION_INVALID"
    WORKSPACE_PRECONDITION_UNSATISFIED = "WORKSPACE_PRECONDITION_UNSATISFIED"
    CONTAINER_ACCESS_REQUIRED = "CONTAINER_ACCESS_REQUIRED"
    PICK_IK_FAILED = "PICK_IK_FAILED"
    PICK_PATH_COLLISION = "PICK_PATH_COLLISION"
    PICK_CONTACT_FAILED = "PICK_CONTACT_FAILED"
    GRASP_FAILED = "GRASP_FAILED"
    HELD_STATE_INVALID = "HELD_STATE_INVALID"
    OBJECT_DROPPED = "OBJECT_DROPPED"
    PLACEMENT_FAILED = "PLACEMENT_FAILED"
    UNSUPPORTED_PHASE_C_OPERATOR = "UNSUPPORTED_PHASE_C_OPERATOR"


@dataclass(frozen=True)
class HeldObjectState:
    generic_object_id: str
    backend_body: str
    weld_id: int
    weld_active: bool
    gripper_body: str
    relative_position_m: tuple[float, float, float]
    relative_orientation_wxyz: tuple[float, float, float, float]
    finger_joint_positions: tuple[float, float]
    exclusive_payload_weld: bool
    floor_contact: bool
    validation_status: str
    rejection_reasons: tuple[str, ...]


@dataclass
class PhysicalPickResult:
    generic_object_id: str
    backend_body: str
    source_context: dict[str, Any]
    required_workspace: str
    grasp_family: str
    success: bool
    status: str
    failure_code: str | None
    message: str
    physics_steps: int
    duration_s: float
    bilateral_contact: bool
    contact_sides: tuple[int, ...]
    contact_geoms: tuple[str, ...]
    attachment_translation_snap_m: float | None
    attachment_angle_snap_rad: float | None
    navigation_safe_carry_reached: bool
    held_state: dict[str, Any] | None
    direct_object_qpos_write: bool = False


@dataclass(frozen=True)
class PlacementTarget:
    generic_object_id: str
    symbolic_destination: str
    destination_kind: str
    target_position_world_m: tuple[float, float, float]
    target_yaw_world_rad: float
    support_backend: str | None
    target_object_id: str | None
    required_workspace: KitchenWorkspace
    edge_margin_m: float
    relation_to_verify: str
    provenance: str


@dataclass
class PhysicalPlaceResult:
    generic_object_id: str
    backend_body: str
    symbolic_destination: str
    placement_target: dict[str, Any]
    success: bool
    status: str
    failure_code: str | None
    message: str
    physics_steps: int
    duration_s: float
    grasp_released: bool
    support_contact: bool
    floor_contact: bool
    stable: bool
    physical_relation_verified: bool
    footprint_inside_support: bool = False
    edge_margin_m: float | None = None
    object_relative_distance_m: float | None = None
    final_body_position_world_m: tuple[float, float, float] | None = None
    object_contact_pairs: tuple[tuple[str, str], ...] = ()
    direct_object_qpos_write: bool = False


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> set[int]:
    first = int(model.body_geomadr[body_id])
    return set(range(first, first + int(model.body_geomnum[body_id])))


def make_kitchen_pick_specs(
    scene,
    inventory: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, SimplePickSpec]:
    inventory_by_id = {row["generic_object_id"]: row for row in inventory["objects"]}
    specs = {}
    table_top = 0.572
    for row in resolution["accepted"]:
        body = row["physical_backend_body"]
        family = row["grasp_family"]
        observed = inventory_by_id[row["generic_object_id"]]
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
        support_height = max(0.008, float(scene.data.xpos[body_id, 2] - table_top))
        rotation = GOOGLE_SPOON_TOP_DOWN_ROTATION if family == "UTENSIL" else None
        if family == "KETTLE":
            angle = np.deg2rad(45.0)
            rotation = np.array(
                ((np.cos(angle), -np.sin(angle), 0.0),
                 (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0))
            ) @ manipulation_profile("google").top_down_rotation
        utensil_reference = GOOGLE_PICK_SPECS["spoon"] if family == "UTENSIL" else None
        grasp_offset = {
            "UTENSIL": 0.020,
            "KETTLE": 0.008,
            "JAR_SOURCE": 0.011,
        }.get(family, 0.0)
        specs[body] = SimplePickSpec(
            label=f"Phase B {family} from {observed['source_context']['source_kind']}",
            grasp_site=f"{body}_grasp",
            support_height=support_height,
            grasp_z_offset=grasp_offset,
            place_supported=True,
            top_down_rotation=rotation,
            home_seed=(utensil_reference.home_seed if utensil_reference else None),
            carry_position=None,
            final_tracking_tolerance=0.018,
        )
    return specs


class KitchenPlacementResolver:
    """Deterministic dynamic placement allocation over symbolic destinations."""

    def __init__(self, scene, inventory: dict[str, Any], resolution: dict[str, Any]):
        self.scene = scene
        self.inventory = inventory
        self.binding = {row["generic_object_id"]: row for row in resolution["accepted"]}
        self.allocated_serving: dict[str, PlacementTarget] = {}
        self.support_height_by_id = {
            row["generic_object_id"]: max(
                0.008, float(row["observed_centroid_world_m"][2]) - 0.59
            )
            for row in inventory["objects"]
        }
        self.serving_slot_by_id: dict[str, tuple[float, float]] = {}
        for function, row_y in (("coffee_vessel", -0.48), ("soup_bowl", -0.64)):
            group = [
                row for row in inventory["objects"]
                if function in set(row["selected_functions"])
            ]
            # Keep each payload's transfer corridor close to its observed X
            # coordinate.  This avoids sweeping bowls across other tabletop
            # objects while preserving deterministic, non-overlapping slots.
            group.sort(key=lambda row: (row["observed_centroid_world_m"][0], row["generic_object_id"]))
            for slot_x, row in zip((-0.16, 0.0, 0.16), group):
                self.serving_slot_by_id[row["generic_object_id"]] = (slot_x, row_y)

    def resolve(self, object_id: str, destination: str) -> PlacementTarget:
        if object_id not in self.binding:
            raise ValueError("DESTINATION_RESOLUTION_FAILED: unresolved object")
        if destination == "serving_area":
            if object_id not in self.serving_slot_by_id:
                raise ValueError("DESTINATION_RESOLUTION_FAILED: object is not a serving target")
            x, y = self.serving_slot_by_id[object_id]
            support_height = self.support_height_by_id[object_id]
            target = PlacementTarget(
                object_id, destination, "SERVING_SUPPORT",
                (x, y, 0.58 + support_height), 0.0, "serving_surface", None,
                KitchenWorkspace.HOME, 0.035, "ON",
                "SYMBOLIC_DESTINATION_PLUS_PERSISTENT_SERVING_ALLOCATOR_V1",
            )
            self.allocated_serving[object_id] = target
            return target
        if destination == "countertop":
            row = next(row for row in self.inventory["objects"] if row["generic_object_id"] == object_id)
            position = tuple(float(x) for x in row["observed_centroid_world_m"])
            return PlacementTarget(
                object_id, destination, "SOURCE_RETURN", position, 0.0,
                "counter_surface", None, KitchenWorkspace.HOME, 0.03,
                "RETURNED_TO_SOURCE", "ORIGINAL_OBSERVED_SOURCE_POSE_V1",
            )
        if destination in self.binding:
            target_backend = self.binding[destination]["physical_backend_body"]
            target_body = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, target_backend
            )
            current = self.scene.data.xpos[target_body]
            position = (
                float(current[0] + 0.12), float(current[1]),
                0.59 + self.support_height_by_id[object_id],
            )
            return PlacementTarget(
                object_id, destination, "OBJECT_RELATIVE_DESTINATION", position,
                0.0, "counter_surface", destination, KitchenWorkspace.HOME,
                0.025, "WITH",
                "CURRENT_RESOLVED_TARGET_POSE_PLUS_SERVICE_OFFSET_V1",
            )
        raise ValueError(f"DESTINATION_RESOLUTION_FAILED: {destination}")


def inspect_held_object_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    generic_object_id: str,
    backend_body: str,
    supported_backend_bodies: set[str],
) -> HeldObjectState:
    profile = manipulation_profile("google")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, backend_body)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.gripper_body)
    weld_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend_body}"
    )
    reasons = []
    active = weld_id >= 0 and bool(data.eq_active[weld_id])
    if not active:
        reasons.append("GRASP_WELD_INACTIVE")
    active_payload_welds = []
    for body in supported_backend_bodies:
        candidate = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{body}"
        )
        if candidate >= 0 and data.eq_active[candidate]:
            active_payload_welds.append(candidate)
    exclusive = active_payload_welds == [weld_id]
    if not exclusive:
        reasons.append("UNEXPECTED_ACTIVE_PAYLOAD_WELD")
    if weld_id < 0 or int(model.eq_obj1id[weld_id]) != gripper_id or int(model.eq_obj2id[weld_id]) != body_id:
        reasons.append("WELD_BODY_MISMATCH")
    relative = data.xpos[body_id] - data.xpos[gripper_id]
    if float(np.linalg.norm(relative)) > 0.55:
        reasons.append("PAYLOAD_SEPARATED_FROM_GRIPPER")
    payload_geoms = _body_geom_ids(model, body_id)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_contact = any(
        floor_id in {int(data.contact[index].geom1), int(data.contact[index].geom2)}
        and bool(payload_geoms & {int(data.contact[index].geom1), int(data.contact[index].geom2)})
        for index in range(data.ncon)
    )
    if floor_contact:
        reasons.append("PAYLOAD_ON_FLOOR")
    finger_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in profile.finger_joints
    ]
    finger_values = tuple(float(data.qpos[model.jnt_qposadr[j]]) for j in finger_ids)
    relative_rotation = data.xmat[gripper_id].reshape(3, 3).T @ data.xmat[body_id].reshape(3, 3)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, relative_rotation.ravel())
    return HeldObjectState(
        generic_object_id=generic_object_id,
        backend_body=backend_body,
        weld_id=weld_id,
        weld_active=active,
        gripper_body=profile.gripper_body,
        relative_position_m=tuple(map(float, relative)),
        relative_orientation_wxyz=tuple(map(float, quaternion)),
        finger_joint_positions=finger_values,
        exclusive_payload_weld=exclusive,
        floor_contact=floor_contact,
        validation_status="TRUE" if not reasons else "FALSE",
        rejection_reasons=tuple(reasons),
    )


class KitchenObjectManipulationExecutor:
    """Synchronous typed facade over the existing Google pick/place executor."""

    def __init__(self, scene, inventory: dict[str, Any], resolution: dict[str, Any], *, step_callback=None):
        self.scene = scene
        self.inventory = inventory
        self.resolution = resolution
        self.step_callback = step_callback
        self.by_id = {row["generic_object_id"]: row for row in resolution["accepted"]}
        self.inventory_by_id = {row["generic_object_id"]: row for row in inventory["objects"]}
        self.backend_bodies = {row["physical_backend_body"] for row in resolution["accepted"]}
        self.placement_resolver = KitchenPlacementResolver(scene, inventory, resolution)
        specs = make_kitchen_pick_specs(scene, inventory, resolution)
        self.executor = CalibratedPickPlaceExecutor(
            scene.model,
            scene.data,
            "google",
            scene.scene_name,
            pick_specs_override=specs,
            calibrated_objects_override=tuple(specs),
            base_stance=np.zeros(3),
            mounting_allowances=PHASE_B_MOUNT_ALLOWANCES,
            ik_position_tolerance=0.020,
            ik_angle_tolerance=np.deg2rad(4.0),
        )

    def sync_workspace(self, workspace: KitchenWorkspace) -> None:
        """Make the generic local primitive relative to the live named pose."""
        base = self.executor.data.qpos[self.executor.base_qpos].copy()
        self.executor.base_stance = base
        self.executor.base_manipulation_target = base.copy()

    @staticmethod
    def _target_dict(target: PlacementTarget) -> dict[str, Any]:
        return {**asdict(target), "required_workspace": target.required_workspace.value}

    def _step_until_stable_mode(self, maximum_steps: int = 30000) -> int:
        for step in range(1, maximum_steps + 1):
            self.executor.update()
            mujoco.mj_step(self.scene.model, self.scene.data)
            if self.step_callback:
                self.step_callback()
            if self.executor.mode in {"holding", "idle", "failed"}:
                return step
        raise RuntimeError("MANIPULATION_EXECUTION_TIMEOUT")

    def pick(
        self,
        generic_object_id: str,
        current_workspace: KitchenWorkspace,
        physically_open_containers: set[str],
    ) -> PhysicalPickResult:
        started = time.perf_counter()
        if generic_object_id not in self.by_id:
            return PhysicalPickResult(generic_object_id, "", {}, "", "", False,
                "ENTITY_RESOLUTION_FAILED", ObjectExecutionFailureCode.ENTITY_RESOLUTION_FAILED.value, "Generic ID is unresolved",
                0, 0.0, False, (), (), None, None, False, None)
        binding = self.by_id[generic_object_id]
        context_row = self.inventory_by_id[generic_object_id]["source_context"]
        required = KitchenWorkspace(context_row["required_workspace"])
        if current_workspace != required:
            return PhysicalPickResult(generic_object_id, binding["physical_backend_body"], context_row,
                required.value, binding["grasp_family"], False,
                "WORKSPACE_PRECONDITION_UNSATISFIED",
                ObjectExecutionFailureCode.WORKSPACE_PRECONDITION_UNSATISFIED.value,
                f"PICK requires {required.value}",
                0, time.perf_counter() - started, False, (), (), None, None, False, None)
        container = context_row["source_container"]
        if container and container not in physically_open_containers:
            return PhysicalPickResult(generic_object_id, binding["physical_backend_body"], context_row,
                required.value, binding["grasp_family"], False, "CONTAINER_ACCESS_REQUIRED",
                ObjectExecutionFailureCode.CONTAINER_ACCESS_REQUIRED.value,
                f"PICK requires physical OPEN({container})",
                0, time.perf_counter() - started, False, (), (), None, None, False, None)
        backend = binding["physical_backend_body"]
        weld_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY, f"google:pick_weld_{backend}"
        )
        before_pos = self.scene.data.xpos[
            mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend)
        ].copy()
        # HOME covers the full main-table workspace, while a bounded local
        # approach aligns the arm with the selected observed object.  This is
        # not a named workspace transition and is retracted by the primitive.
        self.sync_workspace(current_workspace)
        local = np.zeros(3)
        if current_workspace == KitchenWorkspace.HOME:
            local = np.array((0.20, float(np.clip(-before_pos[0], -0.18, 0.18)), 0.0))
        self.executor.base_manipulation_target = self.executor.base_stance + local
        self.executor.request_pick(backend)
        try:
            steps = self._step_until_stable_mode()
        except RuntimeError as error:
            site_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_SITE, f"{backend}_grasp"
            )
            grip_error = None
            if site_id >= 0:
                grip_error = float(np.linalg.norm(
                    self.scene.data.site_xpos[self.executor.grip_site_id]
                    - self.scene.data.site_xpos[site_id]
                ))
            return PhysicalPickResult(
                generic_object_id, backend, context_row, required.value,
                binding["grasp_family"], False, "GRASP_FAILED",
                ObjectExecutionFailureCode.GRASP_FAILED.value,
                f"{error}; mode={self.executor.mode}; status={self.executor.status}; "
                f"grip_site_error_m={grip_error}",
                30000, time.perf_counter() - started, False, (), (), None,
                None, False, None,
            )
        if self.executor.mode != "holding":
            return PhysicalPickResult(generic_object_id, backend, context_row, required.value,
                binding["grasp_family"], False, "GRASP_FAILED",
                ObjectExecutionFailureCode.GRASP_FAILED.value,
                self.executor.failure or self.executor.status, steps,
                time.perf_counter() - started, False, (), (), None, None, False, None)
        state = inspect_held_object_state(
            self.scene.model, self.scene.data, generic_object_id, backend, self.backend_bodies
        )
        after_pos = self.scene.data.xpos[
            mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend)
        ].copy()
        success = state.validation_status == "TRUE" and float(after_pos[2] - before_pos[2]) > 0.03
        return PhysicalPickResult(
            generic_object_id, backend, context_row, required.value, binding["grasp_family"],
            success, "PICK_SUCCESS" if success else "HELD_STATE_INVALID",
            None if success else ObjectExecutionFailureCode.HELD_STATE_INVALID.value,
            "" if success else ",".join(state.rejection_reasons),
            steps, time.perf_counter() - started,
            self.executor.confirmed_contact_sides == (0, 1),
            self.executor.confirmed_contact_sides,
            self.executor.confirmed_contact_geoms,
            self.executor.attachment_translation_snap_m,
            self.executor.attachment_angle_snap_rad,
            self.executor.navigation_safe and self.executor.mode == "holding",
            asdict(state),
        )

    def place(
        self,
        generic_object_id: str,
        symbolic_destination: str,
        current_workspace: KitchenWorkspace,
    ) -> PhysicalPlaceResult:
        started = time.perf_counter()
        binding = self.by_id.get(generic_object_id)
        backend = binding["physical_backend_body"] if binding else ""
        if binding is None or self.executor.held_object != backend:
            return PhysicalPlaceResult(
                generic_object_id, backend, symbolic_destination, {}, False,
                "HELD_STATE_INVALID", ObjectExecutionFailureCode.HELD_STATE_INVALID.value,
                "The requested generic object is not physically held", 0,
                time.perf_counter() - started, False, False, False, False, False,
            )
        target = self.placement_resolver.resolve(generic_object_id, symbolic_destination)
        if current_workspace != target.required_workspace:
            return PhysicalPlaceResult(
                generic_object_id, backend, symbolic_destination, self._target_dict(target), False,
                "WORKSPACE_PRECONDITION_UNSATISFIED",
                ObjectExecutionFailureCode.WORKSPACE_PRECONDITION_UNSATISFIED.value,
                f"PLACE requires {target.required_workspace.value}", 0,
                time.perf_counter() - started, False, False, False, False, False,
            )
        position = np.asarray(target.target_position_world_m, float)
        self.sync_workspace(current_workspace)
        local = np.zeros(3)
        if current_workspace == KitchenWorkspace.HOME:
            local = np.array((0.20, float(np.clip(-position[0], -0.18, 0.18)), 0.0))
        self.executor.base_manipulation_target = self.executor.base_stance + local
        yaw = target.target_yaw_world_rad
        rotation = np.array(
            ((np.cos(yaw), -np.sin(yaw), 0.0),
             (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))
        ) @ manipulation_profile("google").top_down_rotation
        self.executor.request_place_world(position, rotation)
        steps = self._step_until_stable_mode()
        if self.executor.mode != "idle":
            return PhysicalPlaceResult(
                generic_object_id, backend, symbolic_destination, self._target_dict(target), False,
                "PLACEMENT_FAILED", ObjectExecutionFailureCode.PLACEMENT_FAILED.value,
                self.executor.failure or self.executor.status, steps,
                time.perf_counter() - started, False, False, False, False, False,
            )
        for _ in range(400):
            mujoco.mj_step(self.scene.model, self.scene.data)
            steps += 1
        mujoco.mj_forward(self.scene.model, self.scene.data)
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
        )
        object_geoms = _body_geom_ids(self.scene.model, body_id)
        support_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_GEOM,
            target.support_backend or "",
        )
        floor_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        support_contact = floor_contact = False
        contact_pairs = []
        for index in range(self.scene.data.ncon):
            pair = {
                int(self.scene.data.contact[index].geom1),
                int(self.scene.data.contact[index].geom2),
            }
            if not object_geoms & pair:
                continue
            contact_pairs.append(tuple(
                mujoco.mj_id2name(
                    self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom
                ) or f"geom_{geom}"
                for geom in sorted(pair)
            ))
            support_contact |= support_id in pair
            floor_contact |= floor_id in pair
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.scene.model, self.scene.data, mujoco.mjtObj.mjOBJ_BODY,
            body_id, velocity, 0,
        )
        stable = float(np.linalg.norm(velocity[:3])) <= 0.12 and float(
            np.linalg.norm(velocity[3:])
        ) <= 0.025
        weld_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_EQUALITY,
            f"google:pick_weld_{backend}",
        )
        released = weld_id >= 0 and not bool(self.scene.data.eq_active[weld_id])
        relative_ok = True
        relative_distance = None
        if target.target_object_id:
            target_backend = self.by_id[target.target_object_id]["physical_backend_body"]
            target_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, target_backend
            )
            distance = float(np.linalg.norm(
                self.scene.data.xpos[body_id, :2] - self.scene.data.xpos[target_id, :2]
            ))
            relative_distance = distance
            relative_ok = 0.06 <= distance <= 0.18
        final_xy = self.scene.data.xpos[body_id, :2]
        footprint_inside = True
        actual_edge_margin = None
        if target.destination_kind == "SERVING_SUPPORT":
            support_geom = support_id
            centre = self.scene.data.geom_xpos[support_geom, :2]
            half = self.scene.model.geom_size[support_geom, :2]
            actual_edge_margin = float(np.min(half - np.abs(final_xy - centre)))
            footprint_inside = actual_edge_margin >= target.edge_margin_m
        verified = (
            released and support_contact and not floor_contact and stable
            and relative_ok and footprint_inside
        )
        return PhysicalPlaceResult(
            generic_object_id=generic_object_id,
            backend_body=backend,
            symbolic_destination=symbolic_destination,
            placement_target={
                **asdict(target),
                "required_workspace": target.required_workspace.value,
            },
            success=verified,
            status="PLACE_SUCCESS" if verified else "POSTCONDITION_FAILED",
            failure_code=(
                None if verified else ObjectExecutionFailureCode.PLACEMENT_FAILED.value
            ),
            message="" if verified else "Physical placement relation did not validate",
            physics_steps=steps,
            duration_s=time.perf_counter() - started,
            grasp_released=released,
            support_contact=support_contact,
            floor_contact=floor_contact,
            stable=stable,
            physical_relation_verified=verified,
            footprint_inside_support=footprint_inside,
            edge_margin_m=actual_edge_margin,
            object_relative_distance_m=relative_distance,
            final_body_position_world_m=tuple(
                float(x) for x in self.scene.data.xpos[body_id]
            ),
            object_contact_pairs=tuple(contact_pairs),
        )
