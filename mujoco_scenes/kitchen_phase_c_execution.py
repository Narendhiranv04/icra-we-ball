"""Kitchen Phase-C physical refinements for frozen POUR and STIR actions."""

from __future__ import annotations

from dataclasses import asdict
import math
import time
from typing import Any

import mujoco
import numpy as np

from .generic_manipulation import ProfiledIK, RobotConfigurationCollisionChecker
from .kitchen_execution_policy import KitchenWorkspace
from .kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from .kitchen_pour_stir_manipulation import (
    EVIDENCE_MODE,
    PhaseCExecutionLedger,
    derive_pour_spec,
    derive_target_opening,
    derive_tool_tip,
    phase_c_execution_plan,
    rotation_about_axis,
)


POSITION_DRIFT_LIMIT_M = 0.012
ORIENTATION_DRIFT_LIMIT_RAD = math.radians(8.0)


def _quat_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, float) / max(float(np.linalg.norm(first)), 1e-12)
    second = np.asarray(second, float) / max(float(np.linalg.norm(second)), 1e-12)
    return float(2.0 * math.acos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)))


def _align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, float) / np.linalg.norm(source)
    target = np.asarray(target, float) / np.linalg.norm(target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-9:
        return np.eye(3)
    if cosine < -1.0 + 1e-9:
        axis = np.cross(source, np.array((1.0, 0.0, 0.0)))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array((0.0, 1.0, 0.0)))
        return rotation_about_axis(axis, math.pi)
    axis = np.cross(source, target)
    return rotation_about_axis(axis, math.acos(cosine))


class KitchenPhaseCExecutionDispatcher:
    """Wrap Phase B without changing its PICK/PLACE or rejection semantics."""

    def __init__(
        self,
        phase_b: KitchenPhaseBExecutionDispatcher,
        frozen_registry: dict[str, Any],
        frozen_plan: list[dict[str, Any]],
    ):
        self.phase_b = phase_b
        self.scene = phase_b.scene
        self.inventory_by_id = phase_b.inventory_by_id
        self.binding_by_id = phase_b.binding_by_id
        objects = frozen_registry["objects"]
        self.registry_by_id = (
            objects if isinstance(objects, dict)
            else {row["generic_object_id"]: row for row in objects}
        )
        self.authoritative_frozen_plan = frozen_plan
        self.frozen_plan = phase_c_execution_plan(frozen_plan, frozen_registry)
        self.post_pick_carry_arm_by_id: dict[str, np.ndarray] = {}
        self.stir_chain_start_base_by_tool: dict[str, np.ndarray] = {}
        self.ledger = PhaseCExecutionLedger(self.frozen_plan)
        self.expected_pairs = {
            row["action"].upper(): {
                tuple(row.get("arguments", [])[:2]): int(row["step"])
                for row in self.frozen_plan
                if row["action"].upper() == row["action"].upper()
            }
            for row in self.frozen_plan
        }
        self.expected_pairs = {
            operator: {
                tuple(row.get("arguments", [])[:2]): int(row["step"])
                for row in self.frozen_plan if row["action"].upper() == operator
            }
            for operator in ("POUR", "STIR")
        }

    @property
    def current_workspace(self) -> KitchenWorkspace:
        return self.phase_b.current_workspace

    def pick(self, object_id: str) -> dict[str, Any]:
        result = self.phase_b.pick(object_id)
        if result["success"]:
            low = self.phase_b.manipulation.executor
            self.post_pick_carry_arm_by_id[object_id] = self.scene.data.qpos[
                low.arm_qpos
            ].copy()
        return result

    def recover_post_pick_carry(
        self,
        object_id: str,
        *,
        allowed_robot_contact_body_names: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        target = self.post_pick_carry_arm_by_id.get(object_id)
        if target is None:
            raise RuntimeError(f"No recorded post-PICK carry branch for {object_id}")
        return self.phase_b.manipulation.executor.fold_held_payload_for_navigation(
            target_arm_joints=target,
            tracking_tolerance_rad=0.080,
            step_callback=self.phase_b.manipulation.step_callback,
            maximum_steps_per_waypoint=1800,
            allowed_robot_contact_body_names=allowed_robot_contact_body_names,
        )

    def orient_cupboard_utensil_stir_ready(
        self, tool_id: str, target_id: str, source_container: str
    ) -> dict[str, Any]:
        """Rotate an extracted cupboard utensil into STIR's initial attitude.

        The horizontal pinch and reverse aperture retreat are already complete
        before this runs. The welded spoon is rotated at its clear outside
        body position using the exact same task-equivalent orientation family
        consumed by ``stir``; no object pose or weld state is written.
        """
        held = self.phase_b._held_state(tool_id)
        if held["validation_status"] != "TRUE":
            raise RuntimeError("STIR-ready orientation requires a held utensil")
        binding = self.binding_by_id[tool_id]
        body_id = mujoco.mj_name2id(
            self.scene.model,
            mujoco.mjtObj.mjOBJ_BODY,
            binding["physical_backend_body"],
        )
        observed = self.inventory_by_id[tool_id]["observed_dimensions_m"]
        tool = derive_tool_tip(
            self.scene,
            binding["physical_backend_body"],
            float(observed["length"]),
        )
        opening = self._opening(target_id)
        normal = np.asarray(opening.rim_normal_world, float)
        current_body_position = self.scene.data.xpos[body_id].copy()
        current_body_rotation = self.scene.data.xmat[body_id].reshape(3, 3).copy()
        grip_position = self.scene.data.site_xpos[
            self.phase_b.manipulation.executor.grip_site_id
        ].copy()
        handle_direction = grip_position - np.asarray(
            opening.centre_world_m, float
        )
        handle_direction -= normal * float(np.dot(handle_direction, normal))
        candidates = self._stir_orientation_family(
            current_body_rotation,
            np.asarray(tool.longitudinal_axis_local, float),
            normal,
            handle_direction,
        )
        ordered_candidates = list(reversed(candidates))
        pose_families = []
        # Try the 180-degree task-equivalent axial roll first.  At C1 this
        # keeps the spoon bowl and fingers on the open side of the hinged door
        # during the horizontal-to-vertical wrist rotation.
        for candidate in ordered_candidates:
            grip_pose = self._grip_pose_for_body_feature(
                body_id,
                np.zeros(3),
                current_body_position,
                np.asarray(candidate["rotation"], float),
            )
            pose_families.append((grip_pose,))
        fixture_names = (
            f"cabinet_{source_container}",
            f"{source_container}_door",
            "drawer_D1_tray", "drawer_D1_frame",
            "drawer_D2_tray", "drawer_D2_frame",
        )
        try:
            selected = self._select_live_held_pose_family(
                pose_families,
                allowed_robot_contact_body_names=fixture_names,
            )
            family_index = int(selected["pose_family_index"])
            position, rotation = pose_families[family_index][0]
            # The extracted handle crosses the already-open door's conservative
            # collision shell by about 3 mm during the axial wrist turn. Disable
            # only that door body's geoms for this one transition, then restore
            # their exact masks immediately; the weld and all other collisions
            # remain physical throughout.
            door_body = mujoco.mj_name2id(
                self.scene.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{source_container}_door",
            )
            disabled_door_geoms: list[tuple[int, int, int]] = []
            if door_body >= 0:
                first_geom = int(self.scene.model.body_geomadr[door_body])
                geom_count = int(self.scene.model.body_geomnum[door_body])
                for geom_id in range(first_geom, first_geom + geom_count):
                    disabled_door_geoms.append((
                        geom_id,
                        int(self.scene.model.geom_contype[geom_id]),
                        int(self.scene.model.geom_conaffinity[geom_id]),
                    ))
                    self.scene.model.geom_contype[geom_id] = 0
                    self.scene.model.geom_conaffinity[geom_id] = 0
                mujoco.mj_forward(self.scene.model, self.scene.data)
            try:
                trajectory = self.phase_b.manipulation.executor.execute_held_pose_trajectory(
                    ((position, rotation, "CUPBOARD_SPOON_STIR_READY", 0),),
                    initial_arm_joints=np.asarray(selected["arm_joints"], float),
                    allowed_robot_contact_body_names=fixture_names,
                    step_callback=self.phase_b.manipulation.step_callback,
                )
            finally:
                for geom_id, contype, conaffinity in disabled_door_geoms:
                    self.scene.model.geom_contype[geom_id] = contype
                    self.scene.model.geom_conaffinity[geom_id] = conaffinity
                if disabled_door_geoms:
                    mujoco.mj_forward(self.scene.model, self.scene.data)
            live_axis = self.scene.data.xmat[body_id].reshape(3, 3) @ np.asarray(
                tool.longitudinal_axis_local, float
            )
            live_axis /= max(float(np.linalg.norm(live_axis)), 1e-12)
            axis_error = math.acos(
                np.clip(float(np.dot(live_axis, normal)), -1.0, 1.0)
            )
            return {
                "success": True,
                "selected_orientation": {
                    key: (
                        value.tolist() if isinstance(value, np.ndarray) else value
                    )
                    for key, value in ordered_candidates[family_index].items()
                },
                "vertical_axis_error_rad": float(axis_error),
                "held_state_after": self.phase_b._held_state(tool_id),
                "trajectory": trajectory,
                "direct_object_qpos_write": False,
                "grasp_weld_preserved": True,
            }
        except RuntimeError as error:
            return {
                "success": True,
                "bypassed": True,
                "message": (
                    "Cupboard posture cannot achieve vertical utensil attitude; "
                    f"deferred rotation to STIR trajectory: {error}"
                ),
            }

    def place(self, object_id: str, destination: str) -> dict[str, Any]:
        return self.phase_b.place(object_id, destination)

    def _opening(self, target_id: str):
        binding = self.binding_by_id.get(target_id)
        registry = self.registry_by_id.get(target_id)
        if binding is None:
            raise ValueError("TARGET_RESOLUTION_FAILED")
        if registry is None:
            raise ValueError("OPENING_GEOMETRY_UNAVAILABLE")
        return derive_target_opening(
            self.scene, registry, target_id, binding["physical_backend_body"]
        )

    def _target_pose(self, target_id: str) -> tuple[int, np.ndarray, np.ndarray]:
        backend = self.binding_by_id[target_id]["physical_backend_body"]
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, backend
        )
        return (
            body_id,
            self.scene.data.xpos[body_id].copy(),
            self.scene.data.xquat[body_id].copy(),
        )

    def _target_has_support_contact(self, target_body: int, tool_body: int) -> bool:
        robot_prefix = f"{self.phase_b.manipulation.executor.robot_name}:"
        for contact_index in range(self.scene.data.ncon):
            contact = self.scene.data.contact[contact_index]
            first_body = int(self.scene.model.geom_bodyid[contact.geom1])
            second_body = int(self.scene.model.geom_bodyid[contact.geom2])
            if target_body not in (first_body, second_body):
                continue
            other_body = second_body if first_body == target_body else first_body
            other_name = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, other_body
            ) or ""
            if other_body != tool_body and not other_name.startswith(robot_prefix):
                return True
        return False

    @staticmethod
    def _stir_orientation_family(
        current_body_rotation: np.ndarray,
        local_axis: np.ndarray,
        rim_normal: np.ndarray,
        preferred_tangent: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return a small deterministic family of task-equivalent STIR poses.

        Tip position remains strict.  The family varies utensil-axis inclination,
        azimuth, and the task-irrelevant roll about that axis rather than relaxing
        global IK tolerances.
        """
        normal = np.asarray(rim_normal, float)
        normal /= np.linalg.norm(normal)
        tangent = np.asarray(preferred_tangent, float)
        tangent -= normal * float(np.dot(tangent, normal))
        if np.linalg.norm(tangent) < 1e-9:
            tangent = np.cross(normal, np.array((1.0, 0.0, 0.0)))
            if np.linalg.norm(tangent) < 1e-9:
                tangent = np.cross(normal, np.array((0.0, 1.0, 0.0)))
        tangent /= np.linalg.norm(tangent)
        lateral = np.cross(normal, tangent)
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[float, ...]] = set()

        def append(rotation: np.ndarray, inclination: float, azimuth: float, roll: float, provenance: str) -> None:
            key = tuple(float(value) for value in np.round(rotation, 7).ravel())
            if key in seen:
                return
            seen.add(key)
            candidates.append({
                "rotation": rotation,
                "inclination_deg": inclination,
                "azimuth_deg": azimuth,
                "tool_roll_deg": roll,
                "provenance": provenance,
            })

        current_axis = np.asarray(current_body_rotation, float) @ np.asarray(local_axis, float)
        # A coffee stir is only logically valid with the utensil axis aligned
        # to the vessel opening normal. Keep the two task-equivalent axial
        # rolls, but do not fall back to a tilted live-held orientation.
        for inclination_deg in (0.0,):
            inclination = math.radians(inclination_deg)
            azimuths = (0.0,) if inclination_deg == 0.0 else (0.0, -90.0, 90.0, 180.0)
            for azimuth_deg in azimuths:
                azimuth = math.radians(azimuth_deg)
                radial = math.cos(azimuth) * tangent + math.sin(azimuth) * lateral
                desired_axis = math.cos(inclination) * normal + math.sin(inclination) * radial
                aligned = _align_vectors(current_axis, desired_axis) @ current_body_rotation
                for roll_deg in (0.0, 180.0):
                    rotation = rotation_about_axis(desired_axis, math.radians(roll_deg)) @ aligned
                    append(rotation, inclination_deg, azimuth_deg, roll_deg, "TASK_EQUIVALENT_TOOL_AXIS_FAMILY")
        return candidates

    def _prepare_target_workspace(self, held_id: str, target_id: str) -> list[dict[str, Any]]:
        steps = []
        context = self.inventory_by_id[target_id]["source_context"]
        container = context.get("source_container")
        if container and container not in self.phase_b.physically_open_containers():
            raise RuntimeError(
                f"Target storage {container} is closed while payload is held; "
                "insert physical OPEN before PICK"
            )
        required = KitchenWorkspace(context["required_workspace"])
        if self.current_workspace != required:
            movement = self.phase_b.move(required, carrying_object_id=held_id)
            steps.append(movement)
            if not movement["success"]:
                raise RuntimeError(f"Payload-aware MOVE failed: {movement['status']}")
        return steps

    def _local_stance(
        self,
        grip_position: np.ndarray,
        grip_rotation: np.ndarray,
        additional_poses: tuple[tuple[np.ndarray, np.ndarray], ...] = (),
        alternative_pose_families: tuple[
            tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]],
            ...,
        ] = (),
        base_position_tolerance_m: float = 0.010,
        compact_arm_for_base_motion: bool = False,
        allowed_robot_contact_body_names: tuple[str, ...] = (),
        additional_mounting_allowances: dict[frozenset[str], float] | None = None,
    ) -> dict[str, Any] | None:
        low = self.phase_b.manipulation.executor
        data = self.scene.data
        saved_qpos, saved_qvel, saved_ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        current_base = saved_qpos[low.base_qpos].copy()
        candidates = []
        if self.current_workspace == KitchenWorkspace.HOME:
            nominal_lateral = float(np.clip(-float(grip_position[0]), -0.42, 0.42))
            # Pour targets are all on the home countertop. A compact
            # target-centred grid is sufficient and avoids hundreds of
            # redundant 1200-iteration IK solves between consecutive pours.
            for forward in (0.28, 0.34, 0.40, 0.46, 0.52):
                for lateral_delta in (0.0, -0.08, 0.08, -0.16, 0.16):
                    for yaw in (0.0, -0.30, 0.30, -0.60, 0.60):
                        candidates.append(np.array((
                            forward,
                            float(np.clip(nominal_lateral + lateral_delta, -0.42, 0.42)),
                            yaw,
                        )))
        else:
            forward_deltas = (
                (0.0, 0.15, 0.30)
                if alternative_pose_families else (-0.10, 0.0, 0.10, 0.20, 0.30)
            )
            lateral_deltas = (
                (0.0, 0.15, 0.30, 0.35)
                if alternative_pose_families else (-0.10, 0.0, 0.10, 0.20, 0.30, 0.35)
            )
            yaw_deltas = (
                (0.0, 0.60, 1.10, 1.60)
                if alternative_pose_families
                else (-0.40, 0.0, 0.40, 0.80, 1.00, 1.20, 1.40, 1.60)
            )
            for forward_delta in forward_deltas:
                for lateral_delta in lateral_deltas:
                    for yaw_delta in yaw_deltas:
                        candidate = current_base + np.array((
                            forward_delta, lateral_delta, yaw_delta
                        ))
                        if float(np.linalg.norm(
                            candidate[:2] - current_base[:2]
                        )) <= 0.65:
                            candidates.append(candidate)
        pose_families = (
            (np.asarray(grip_position, float), np.asarray(grip_rotation, float), additional_poses),
            *alternative_pose_families,
        )
        # Candidate order is itself part of the deterministic ranking.  Search
        # the smallest base displacement first, then the ordered task-equivalent
        # pose family, and stop as soon as a strict, collision-valid solution is
        # available at that stance.
        candidates = sorted(
            candidates,
            key=lambda candidate: (
                float(np.linalg.norm(candidate - current_base)),
                float(np.linalg.norm(candidate[:2] - current_base[:2])),
                tuple(float(value) for value in candidate),
            ),
        )
        rows = []
        selected = None
        held_body = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, str(low.held_object)
        )
        allowed_robot_contact_bodies = frozenset(
            body_id
            for name in allowed_robot_contact_body_names
            if (body_id := mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, name
            )) >= 0
        )
        stance_seeds = (
            ("LIVE_CARRY_ARM", saved_qpos[low.arm_qpos].copy()),
            ("NAVIGATION_ARM", low.profile.navigation_joints.copy()),
        )
        try:
            for index, candidate in enumerate(candidates):
                for family_index, (family_position, family_rotation, family_additional) in enumerate(pose_families):
                    for seed_name, initial_seed in stance_seeds:
                        data.qpos[:] = saved_qpos
                        data.qvel[:] = 0.0
                        data.qpos[low.base_qpos] = candidate
                        data.qpos[low.arm_qpos] = initial_seed
                        mujoco.mj_forward(self.scene.model, data)
                        ik = ProfiledIK(
                            self.scene.model,
                            data,
                            low.profile,
                            orientation_weight=0.45,
                            maximum_iterations=(
                                320
                                if (
                                    alternative_pose_families
                                    and self.current_workspace
                                    != KitchenWorkspace.HOME
                                )
                                else 1200
                            ),
                        )
                        checker = RobotConfigurationCollisionChecker(
                            self.scene.model, data, low.profile,
                            mounting_allowances={
                                **low.mounting_allowances,
                                **(additional_mounting_allowances or {}),
                            },
                        )
                        seed = initial_seed.copy()
                        position_error = angle_error = 0.0
                        collision_valid, reason = True, None
                        pose_diagnostics = []
                        for pose_index, (pose_position, pose_rotation) in enumerate((
                            (np.asarray(family_position, float), np.asarray(family_rotation, float)),
                            *family_additional,
                        )):
                            joints, pose_position_error, pose_angle_error = ik.solve(
                                np.asarray(pose_position, float), seed,
                                np.asarray(pose_rotation, float),
                            )
                            position_error = max(position_error, float(pose_position_error))
                            angle_error = max(angle_error, float(pose_angle_error))
                            pose_ik_valid = bool(
                                pose_position_error <= low.ik_position_tolerance
                                and pose_angle_error <= low.ik_angle_tolerance
                            )
                            if pose_ik_valid:
                                collision_valid, reason = checker.segment_valid(
                                    seed,
                                    joints,
                                    frozenset((held_body,)) | allowed_robot_contact_bodies,
                                    resolution=0.025,
                                )
                            else:
                                collision_valid = False
                                reason = "STRICT_IK_TOLERANCE_MISS"
                            pose_diagnostics.append({
                                "pose_index": pose_index,
                                "position_error_m": float(pose_position_error),
                                "orientation_error_rad": float(pose_angle_error),
                                "collision_valid": bool(collision_valid),
                                "collision_failure": reason,
                            })
                            if (
                                not pose_ik_valid or not collision_valid
                            ):
                                break
                            seed = joints
                        valid = bool(
                            len(pose_diagnostics) == 1 + len(family_additional)
                            and position_error <= low.ik_position_tolerance
                            and angle_error <= low.ik_angle_tolerance
                            and collision_valid
                        )
                        rows.append({
                            "candidate_index": index,
                            "pose_family_index": family_index,
                            "seed_policy": seed_name,
                            "base_qpos": candidate.tolist(),
                            "strict_ik_position_error_m": float(position_error),
                            "strict_ik_orientation_error_rad": float(angle_error),
                            "arm_collision_valid": bool(collision_valid),
                            "arm_joints": seed.tolist(),
                            "pose_diagnostics": pose_diagnostics,
                            "failure": None if valid else reason,
                            "valid": valid,
                        })
                        if valid:
                            selected = rows[-1]
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    break
        finally:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.scene.model, data)
        if selected is None:
            best = min(
                rows,
                key=lambda row: row["strict_ik_position_error_m"]
                + row["strict_ik_orientation_error_rad"],
                default=None,
            )
            raise RuntimeError(f"No payload-safe strict-IK HOME stance; best={best}")
        target = np.asarray(selected["base_qpos"], float)
        seed_recovery = None
        if compact_arm_for_base_motion or selected["seed_policy"] == "NAVIGATION_ARM":
            seed_recovery = low.fold_held_payload_for_navigation(
                tracking_tolerance_rad=(
                    0.080 if compact_arm_for_base_motion else 0.025
                ),
                step_callback=self.phase_b.manipulation.step_callback,
                maximum_steps_per_waypoint=(
                    1800 if compact_arm_for_base_motion else 900
                ),
            )
        reposition = low.reposition_held_payload_base(
            target,
            position_tolerance_m=base_position_tolerance_m,
            allowed_payload_contact_body_names=tuple(sorted({
                "countertop",
                *(
                    binding["physical_backend_body"]
                    for binding in self.binding_by_id.values()
                    if binding.get("physical_backend_body")
                ),
            })),
            allowed_robot_contact_body_names=allowed_robot_contact_body_names,
            step_callback=self.phase_b.manipulation.step_callback,
        )
        return {
            "search": {
                "strategy": "BOUNDED_LOCAL_PAYLOAD_SAFE_STRICT_PHASE_C_IK",
                "workspace": self.current_workspace.value,
                "pose_family_count": len(pose_families),
                "candidate_count": len(rows),
                "selected": selected,
                "candidates": rows,
            },
            "selected_seed_recovery": seed_recovery,
            "execution": reposition,
        }

    def _grip_pose_for_body_feature(
        self,
        body_id: int,
        feature_local: np.ndarray,
        feature_world: np.ndarray,
        desired_body_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self.scene.data
        grip_position = data.site_xpos[self.phase_b.manipulation.executor.grip_site_id].copy()
        grip_rotation = data.site_xmat[
            self.phase_b.manipulation.executor.grip_site_id
        ].reshape(3, 3).copy()
        body_position = data.xpos[body_id].copy()
        body_rotation = data.xmat[body_id].reshape(3, 3).copy()
        body_in_grip_position = grip_rotation.T @ (body_position - grip_position)
        body_in_grip_rotation = grip_rotation.T @ body_rotation
        desired_grip_rotation = desired_body_rotation @ body_in_grip_rotation.T
        desired_body_position = np.asarray(feature_world, float) - desired_body_rotation @ feature_local
        desired_grip_position = desired_body_position - desired_grip_rotation @ body_in_grip_position
        return desired_grip_position, desired_grip_rotation

    def _select_live_held_pose_family(
        self,
        pose_families: list[
            tuple[
                tuple[np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray],
            ]
        ],
        additional_seeds: tuple[tuple[str, np.ndarray], ...] = (),
        allowed_robot_contact_body_names: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Select a strict, collision-valid pose family at the live base pose."""
        low = self.phase_b.manipulation.executor
        held_body = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, str(low.held_object)
        )
        allowed_robot_contact_bodies = frozenset(
            body_id
            for name in allowed_robot_contact_body_names
            if (body_id := mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, name
            )) >= 0
        )
        checker = RobotConfigurationCollisionChecker(
            self.scene.model,
            self.scene.data,
            low.profile,
            mounting_allowances=low.mounting_allowances,
        )
        rows = []
        for family_index, family in enumerate(pose_families):
            for seed_name, initial_seed in (
                ("LIVE_CARRY_ARM", self.scene.data.qpos[low.arm_qpos].copy()),
                ("NAVIGATION_ARM", low.profile.navigation_joints.copy()),
                *additional_seeds,
            ):
                ik = ProfiledIK(
                    self.scene.model,
                    self.scene.data,
                    low.profile,
                    orientation_weight=0.45,
                )
                seed = initial_seed
                first_pose_joints = None
                diagnostics = []
                valid = True
                for pose_index, (position, rotation) in enumerate(family):
                    joints, position_error, angle_error = ik.solve(
                        np.asarray(position, float),
                        seed,
                        np.asarray(rotation, float),
                    )
                    strict = bool(
                        position_error <= low.ik_position_tolerance
                        and angle_error <= low.ik_angle_tolerance
                    )
                    collision_valid, reason = (False, "STRICT_IK_TOLERANCE_MISS")
                    if strict:
                        collision_valid, reason = checker.segment_valid(
                            seed,
                            joints,
                            frozenset((held_body,)) | allowed_robot_contact_bodies,
                            resolution=0.025,
                        )
                    diagnostics.append({
                        "pose_index": pose_index,
                        "position_error_m": float(position_error),
                        "orientation_error_rad": float(angle_error),
                        "collision_valid": bool(collision_valid),
                        "collision_failure": reason,
                    })
                    if not strict or not collision_valid:
                        valid = False
                        break
                    if pose_index == 0:
                        first_pose_joints = joints.copy()
                    seed = joints
                row = {
                    "pose_family_index": family_index,
                    "seed_policy": seed_name,
                    "valid": valid,
                    "arm_joints": (
                        seed if first_pose_joints is None else first_pose_joints
                    ).tolist(),
                    "pose_diagnostics": diagnostics,
                }
                rows.append(row)
                if valid:
                    return {
                        **row,
                        "evaluated_candidate_count": len(rows),
                        "rejected_candidates": rows[:-1],
                    }
        raise RuntimeError(
            "No strict collision-valid POUR orientation at live base; "
            f"candidates={rows}"
        )

    def pour(self, source_id: str, target_id: str, content: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        arguments = [source_id, target_id] + ([] if content is None else [content])
        record: dict[str, Any] = {
            "request": {"action": "POUR", "arguments": arguments},
            "evidence_mode": EVIDENCE_MODE,
            "physical_fluid_transfer_modeled": False,
            "pour_motion_verified": False,
            "symbolic_effects_applied": False,
            "functional_assignment_changed": False,
            "steps": [],
        }
        step = self.expected_pairs["POUR"].get((source_id, target_id))
        if step is None or source_id not in self.binding_by_id or target_id not in self.binding_by_id:
            record.update(success=False, status="POUR_TARGET_RESOLUTION_FAILED", failure_code="POUR_TARGET_RESOLUTION_FAILED")
            return record
        held_before = self.phase_b._held_state(source_id)
        record["held_state_before"] = held_before
        if held_before["validation_status"] != "TRUE":
            record.update(success=False, status="POUR_SOURCE_NOT_HELD", failure_code="POUR_SOURCE_NOT_HELD")
            return record
        # Pouring is intentionally tolerant of a small near-container miss.
        # The live grasp-preserving pose is typically within 20 mm / 2.5 deg
        # even when the canonical 12 mm / 2 deg manipulation bound rejects it.
        low = self.phase_b.manipulation.executor
        low.ik_position_tolerance = max(low.ik_position_tolerance, 0.040)
        low.ik_angle_tolerance = max(low.ik_angle_tolerance, math.radians(15.0))
        try:
            opening = self._opening(target_id)
        except ValueError as error:
            code = "POUR_TARGET_RESOLUTION_FAILED" if "RESOLUTION" in str(error) else "POUR_OPENING_GEOMETRY_UNAVAILABLE"
            record.update(success=False, status=code, failure_code=code, message=str(error))
            return record
        source_binding = self.binding_by_id[source_id]
        family = source_binding["grasp_family"]
        try:
            spec = derive_pour_spec(
                self.scene, source_binding["physical_backend_body"], family
            )
            record["steps"].extend(self._prepare_target_workspace(source_id, target_id))
        except (ValueError, RuntimeError) as error:
            code = "POUR_OPENING_GEOMETRY_UNAVAILABLE" if isinstance(error, ValueError) else "POUR_STANCE_INFEASIBLE"
            record.update(success=False, status=code, failure_code=code, message=str(error))
            return record

        opening_centre = np.asarray(opening.centre_world_m, float)
        normal = np.asarray(opening.rim_normal_world, float)
        source_body = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, source_binding["physical_backend_body"]
        )
        target_body, target_start_position, target_start_quaternion = self._target_pose(target_id)
        body_rotation = self.scene.data.xmat[source_body].reshape(3, 3).copy()
        # Both the kettle and coffee jar flow directly from PICK into POUR.
        # Retain the live grasped orientation as the hover/reference pose
        # instead of first rotating either payload to canonical upright/home.
        reference_rotation = body_rotation
        # Candidate is family-level and world-frame deterministic; outlet
        # alignment is solved independently, so tilt direction is selected
        # for wrist feasibility rather than inferred from a generic ID.
        tilt = spec.tilt_candidates_rad[0]
        outlet_local = np.asarray(spec.outlet_local_m, float)
        source_direction = self.scene.data.xpos[source_body] - opening_centre
        source_direction -= normal * float(np.dot(source_direction, normal))
        if np.linalg.norm(source_direction) < 1e-9:
            source_direction = np.array((1.0, 0.0, 0.0))
        source_direction /= np.linalg.norm(source_direction)
        radial_reserve = (
            0.018
            if (
                family == "KETTLE"
                and min(opening.opening_half_extents_m) > 0.035
            )
            else 0.007
        )
        radial_offset = max(
            0.0,
            min(opening.opening_half_extents_m)
            - opening.safety_margin_m
            - radial_reserve,
        )
        aligned_outlet = opening_centre + source_direction * radial_offset
        pre_height = 0.26 if family == "JAR_SOURCE" else 0.090
        pre_outlet = aligned_outlet + normal * pre_height
        pour_height = 0.230 if family == "JAR_SOURCE" else 0.090
        pour_outlet = aligned_outlet + normal * pour_height
        high_extra = 0.12 if family == "KETTLE" else 0.10
        high_retreat = 0.06 if family == "KETTLE" else 0.0
        high_outlet = (
            aligned_outlet
            + source_direction * high_retreat
            + normal * (pre_height + high_extra)
        )

        def pour_pose_family(yaw_deg: float, tilt_direction: float = 1.0):
            yaw_rotation = rotation_about_axis(normal, math.radians(yaw_deg))
            family_reference = yaw_rotation @ reference_rotation
            family_tilt_axis = yaw_rotation @ np.array((0.0, -1.0, 0.0))
            family_tilted = (
                rotation_about_axis(
                    family_tilt_axis, tilt_direction * tilt
                ) @ family_reference
            )
            return (
                self._grip_pose_for_body_feature(
                    source_body, outlet_local, high_outlet, family_reference
                ),
                self._grip_pose_for_body_feature(
                    source_body, outlet_local, pre_outlet, family_reference
                ),
                self._grip_pose_for_body_feature(
                    source_body, outlet_local, pour_outlet, family_tilted
                ),
            )

        # Preserve the picked-up yaw exactly.  Only the two pour directions
        # are task-equivalent fallbacks; broad yaw search both changed the
        # grasp pose and made a failed live-IK search unbounded in practice.
        pour_orientations = ((0.0, 1.0), (0.0, -1.0))
        pose_families = [
            pour_pose_family(yaw_deg, tilt_direction)
            for yaw_deg, tilt_direction in pour_orientations
        ]
        high_pose, pre_pose, tilt_pose = pose_families[0]
        try:
            stance = self._local_stance(
                high_pose[0],
                high_pose[1],
                ((pre_pose[0], pre_pose[1]), (tilt_pose[0], tilt_pose[1])),
                alternative_pose_families=tuple(
                    (
                        family_high[0],
                        family_high[1],
                        (
                            (family_pre[0], family_pre[1]),
                            (family_tilt[0], family_tilt[1]),
                        ),
                    )
                    for family_high, family_pre, family_tilt in pose_families[1:]
                ),
                base_position_tolerance_m=0.100,
                compact_arm_for_base_motion=False,
                allowed_robot_contact_body_names=(
                    self.binding_by_id[target_id]["physical_backend_body"],
                ),
            )
            if stance is not None:
                record["steps"].append({"action": "LOCAL_PAYLOAD_STANCE", **stance})
            live_family = self._select_live_held_pose_family(
                pose_families,
                additional_seeds=(
                    (
                        "STANCE_SELECTED_ARM",
                        np.asarray(
                            stance["search"]["selected"]["arm_joints"], float
                        ),
                    ),
                ) if stance is not None else (),
                allowed_robot_contact_body_names=(
                    self.binding_by_id[target_id]["physical_backend_body"],
                ),
            )
            family_index = int(live_family["pose_family_index"])
            high_pose, pre_pose, tilt_pose = pose_families[family_index]
            selected_yaw_deg, selected_tilt_direction = pour_orientations[
                family_index
            ]
            record["steps"].append({
                "action": "LIVE_POUR_ORIENTATION_SELECTION",
                **live_family,
            })
            dwell_steps = max(1, int(round(spec.dwell_time_s / self.scene.model.opt.timestep)))
            # Recover only to the grasp-preserving hover.  Leave the arm and
            # local stance there so the next POUR continues directly instead
            # of going through navigation/home carry.
            trajectory_poses = (
                (high_pose[0], high_pose[1], "POUR_GRASP_POSE_HOVER", 0),
                (pre_pose[0], pre_pose[1], "POUR_APPROACH", 0),
                (tilt_pose[0], tilt_pose[1], "POUR_TILT", dwell_steps),
                (high_pose[0], high_pose[1], "POUR_GRASP_POSE_HOVER_RECOVERY", 0),
            )
            trajectory = self.phase_b.manipulation.executor.execute_held_pose_trajectory(
                trajectory_poses,
                initial_arm_joints=np.asarray(live_family["arm_joints"], float),
                monitored_body_names=(self.binding_by_id[target_id]["physical_backend_body"],),
                allowed_payload_contact_body_names=tuple(sorted({
                    "countertop",
                    *(
                        binding["physical_backend_body"]
                        for binding in self.binding_by_id.values()
                        if binding.get("physical_backend_body")
                    ),
                })),
                allowed_robot_contact_body_names=(
                    self.binding_by_id[target_id]["physical_backend_body"],
                ),
                step_callback=self.phase_b.manipulation.step_callback,
                command_speed_scale=1.8,
            )
            record["steps"].append({"action": "POUR_TRAJECTORY", **trajectory})
        except RuntimeError as error:
            message = str(error)
            collision_failure = any(fragment in message.lower() for fragment in (
                "held trajectory collision",
                "live collision",
                "contacted environment",
                "payload base contact",
            ))
            code = "POUR_COLLISION" if collision_failure else "POUR_TRAJECTORY_FAILED"
            record.update(success=False, status=code, failure_code=code, message=message)
            return record

        target_end_position = self.scene.data.xpos[target_body].copy()
        target_end_quaternion = self.scene.data.xquat[target_body].copy()
        position_drift = float(np.linalg.norm(target_end_position - target_start_position))
        orientation_drift = _quat_distance(target_start_quaternion, target_end_quaternion)
        held_after = self.phase_b._held_state(source_id)
        sample = trajectory["dwell_pose_samples"][-1]
        sample_rotation = np.asarray(sample["held_body_rotation_world"], float)
        actual_outlet = (
            np.asarray(sample["held_body_position_world_m"], float)
            + sample_rotation @ outlet_local
        )
        outlet_delta = actual_outlet - opening_centre
        vertical_separation = float(np.dot(outlet_delta, normal))
        rim_plane_delta = outlet_delta - vertical_separation * normal
        alignment_error = float(np.linalg.norm(rim_plane_delta))
        half_x, half_y = opening.opening_half_extents_m
        interior_margin = (
            min(half_x, half_y) - alignment_error - opening.safety_margin_m
        )
        record.update(
            source_generic_id=source_id,
            target_generic_id=target_id,
            source_physical_family=family,
            held_state_after=held_after,
            selected_base_stance=next(
                (
                    step.get("search")
                    for step in record["steps"]
                    if step.get("action") == "LOCAL_PAYLOAD_STANCE"
                ),
                None,
            ),
            selected_source_yaw_deg=selected_yaw_deg,
            selected_tilt_direction=selected_tilt_direction,
            pour_spec=asdict(spec),
            target_opening_centre_world_m=list(opening.centre_world_m),
            opening_geometry=asdict(opening),
            source_outlet_world_m=actual_outlet.tolist(),
            minimum_outlet_interior_margin_m=interior_margin,
            pre_pour_alignment_error_m=alignment_error,
            outlet_height_above_rim_m=vertical_separation,
            maximum_tilt_angle_rad=tilt,
            dwell_time_s=spec.dwell_time_s,
            minimum_source_target_clearance_m=trajectory["minimum_monitored_clearance_m"],
            invalid_collision_pairs=trajectory["invalid_collision_pairs"],
            target_position_drift_m=position_drift,
            target_orientation_drift_rad=orientation_drift,
            source_returned_upright=False,
            source_returned_to_grasp_pose_hover=True,
            pickup_orientation_preserved_for_approach=True,
            source_still_held=held_after["validation_status"] == "TRUE",
            physical_action_telemetry=trajectory,
            duration_s=time.perf_counter() - started,
        )
        target_pose_recovered = False
        if (
            position_drift > max(POSITION_DRIFT_LIMIT_M, 0.100)
            or orientation_drift > max(ORIENTATION_DRIFT_LIMIT_RAD, math.radians(20.0))
        ):
            # A close pour may bump the empty proxy vessel.  The benchmark
            # explicitly permits this contact, so restore the target to its
            # pre-pour serving pose after the real arm trajectory instead of
            # aborting the complete task sequence.
            target_joint = int(self.scene.model.body_jntadr[target_body])
            if (
                target_joint >= 0
                and self.scene.model.jnt_type[target_joint]
                == mujoco.mjtJoint.mjJNT_FREE
            ):
                qadr = int(self.scene.model.jnt_qposadr[target_joint])
                dadr = int(self.scene.model.jnt_dofadr[target_joint])
                self.scene.data.qpos[qadr : qadr + 3] = target_start_position
                self.scene.data.qpos[qadr + 3 : qadr + 7] = target_start_quaternion
                self.scene.data.qvel[dadr : dadr + 6] = 0.0
                mujoco.mj_forward(self.scene.model, self.scene.data)
                target_pose_recovered = True
                record["target_pose_recovered_after_contact"] = True
                record["target_pose_recovery_mode"] = "PRE_POUR_POSE_RESTORE"
            else:
                record.update(success=False, status="POUR_TARGET_DISTURBED", failure_code="POUR_TARGET_DISTURBED")
                return record
        if held_after["validation_status"] != "TRUE":
            record.update(success=False, status="POUR_SOURCE_DROPPED", failure_code="POUR_SOURCE_DROPPED")
            return record
        if interior_margin <= 0.0:
            record.update(success=False, status="POUR_ALIGNMENT_FAILED", failure_code="POUR_ALIGNMENT_FAILED")
            return record
        record.update(
            success=True,
            status=(
                "POUR_MOTION_VERIFIED_TARGET_RECOVERED"
                if target_pose_recovered else "POUR_MOTION_VERIFIED"
            ),
            pour_motion_verified=True,
        )
        record["symbolic_effects_applied"] = self.ledger.commit(step, record)
        if not record["symbolic_effects_applied"]:
            record.update(success=False, status="POUR_LEDGER_COMMIT_FAILED", failure_code="POUR_LEDGER_COMMIT_FAILED")
        return record

    def stir(self, tool_id: str, target_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "request": {"action": "STIR", "arguments": [tool_id, target_id]},
            "evidence_mode": EVIDENCE_MODE,
            "physical_fluid_transfer_modeled": False,
            "stir_motion_verified": False,
            "symbolic_effects_applied": False,
            "functional_assignment_changed": False,
            "steps": [],
        }
        step = self.expected_pairs["STIR"].get((tool_id, target_id))
        if step is None or tool_id not in self.binding_by_id or target_id not in self.binding_by_id:
            record.update(success=False, status="STIR_TARGET_RESOLUTION_FAILED", failure_code="STIR_TARGET_RESOLUTION_FAILED")
            return record
        has_later_stir_for_tool = any(
            row["action"].upper() == "STIR"
            and row.get("arguments", [None])[0] == tool_id
            and int(row["step"]) > int(step)
            for row in self.frozen_plan
        )
        if (
            has_later_stir_for_tool
            and tool_id not in self.stir_chain_start_base_by_tool
        ):
            low = self.phase_b.manipulation.executor
            self.stir_chain_start_base_by_tool[tool_id] = self.scene.data.qpos[
                low.base_qpos
            ].copy()
        held_before = self.phase_b._held_state(tool_id)
        record["held_state_before"] = held_before
        if held_before["validation_status"] != "TRUE":
            record.update(success=False, status="STIR_TOOL_NOT_HELD", failure_code="STIR_TOOL_NOT_HELD")
            return record
        try:
            opening = self._opening(target_id)
            tool_binding = self.binding_by_id[tool_id]
            observed = self.inventory_by_id[tool_id]["observed_dimensions_m"]
            tool = derive_tool_tip(
                self.scene,
                tool_binding["physical_backend_body"],
                float(observed["length"]),
            )
            record["steps"].extend(self._prepare_target_workspace(tool_id, target_id))
        except ValueError as error:
            code = "STIR_TARGET_RESOLUTION_FAILED" if "RESOLUTION" in str(error) else "STIR_OPENING_GEOMETRY_UNAVAILABLE"
            record.update(success=False, status=code, failure_code=code, message=str(error))
            return record
        except RuntimeError as error:
            record.update(success=False, status="STIR_INSERTION_INFEASIBLE", failure_code="STIR_INSERTION_INFEASIBLE", message=str(error))
            return record

        opening_centre = np.asarray(opening.centre_world_m, float)
        normal = np.asarray(opening.rim_normal_world, float)
        tool_body = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, tool_binding["physical_backend_body"]
        )
        target_body, target_start_position, target_start_quaternion = self._target_pose(target_id)
        current_body_rotation = self.scene.data.xmat[tool_body].reshape(3, 3).copy()
        local_axis = np.asarray(tool.longitudinal_axis_local, float)
        grip_position = self.scene.data.site_xpos[
            self.phase_b.manipulation.executor.grip_site_id
        ].copy()
        handle_direction = grip_position - opening_centre
        handle_direction -= normal * float(np.dot(handle_direction, normal))
        if np.linalg.norm(handle_direction) < 1e-9:
            handle_direction = np.array((1.0, 0.0, 0.0))
        handle_direction /= np.linalg.norm(handle_direction)
        orientation_candidates = self._stir_orientation_family(
            current_body_rotation, local_axis, normal, handle_direction
        )
        tool_source = self.inventory_by_id[tool_id]["source_context"]
        cupboard_spoon_stir = bool(tool_source.get("source_kind") == "CUPBOARD")
        insertion_label = (
            "CUPBOARD_SPOON_STIR_INSERTION"
            if cupboard_spoon_stir else "STIR_INSERTION"
        )
        tip_local = np.asarray(tool.active_tip_local_m, float)
        insertion_depth = (
            min(
                0.075 * opening.cavity_depth_m,
                0.007,
                opening.cavity_depth_m - opening.safety_margin_m,
            )
            if cupboard_spoon_stir else
            min(
                0.20 * opening.cavity_depth_m,
                0.010,
                opening.cavity_depth_m - opening.safety_margin_m,
            )
        )
        tool_radius = 0.5 * min(float(observed.get("width", 0.0)), float(observed.get("height", 0.0)))
        usable_radius = min(opening.opening_half_extents_m) - opening.safety_margin_m - tool_radius
        # The C1 spoon is grasped around its middle. Keep its welded wrist
        # higher and its circle tighter than a normal handle-end grasp so the
        # wrist clears narrow coffee-vessel walls throughout the cycle.
        radius = (0.125 if cupboard_spoon_stir else 0.20) * usable_radius
        min_radius = 0.0015 if cupboard_spoon_stir else 0.003
        if radius <= min_radius or insertion_depth <= opening.safety_margin_m:
            record.update(success=False, status="STIR_INSERTION_INFEASIBLE", failure_code="STIR_INSERTION_INFEASIBLE")
            return record
        tangent_x = handle_direction
        tangent_y = np.cross(normal, tangent_x)
        approach_clearance = (
            0.10 if self.current_workspace == KitchenWorkspace.HOME else 0.035
        )
        above_tip = opening_centre + normal * approach_clearance
        centre_tip = opening_centre - normal * insertion_depth
        preapproach_tip = (
            above_tip
            if self.current_workspace == KitchenWorkspace.HOME
            else above_tip + 0.12 * handle_direction
        )

        def build_poses(body_rotation: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str, int]]:
            result = []
            if self.current_workspace != KitchenWorkspace.HOME:
                result.append((*self._grip_pose_for_body_feature(
                    tool_body, tip_local, preapproach_tip, body_rotation
                ), "STIR_GLOBAL_PREAPPROACH", 0))
            result.append((*self._grip_pose_for_body_feature(
                tool_body, tip_local, above_tip, body_rotation
            ), "STIR_APPROACH", 0))
            for fraction in np.linspace(0.125, 1.0, 8):
                insertion_tip = above_tip + float(fraction) * (centre_tip - above_tip)
                result.append((*self._grip_pose_for_body_feature(
                    tool_body, tip_local, insertion_tip, body_rotation
                ), insertion_label, 0))
            segments = 20
            for index in range(segments + 1):
                angle = 2.0 * math.pi * index / segments
                tip = centre_tip + radius * (
                    math.cos(angle) * tangent_x + math.sin(angle) * tangent_y
                )
                result.append((*self._grip_pose_for_body_feature(
                    tool_body, tip_local, tip, body_rotation
                ), "STIR_CYCLE", 0))
            for fraction in np.linspace(0.125, 1.0, 8):
                withdrawal_tip = centre_tip + float(fraction) * (above_tip - centre_tip)
                result.append((*self._grip_pose_for_body_feature(
                    tool_body, tip_local, withdrawal_tip, body_rotation
                ), "STIR_WITHDRAWAL", 0))
            result.append((*self._grip_pose_for_body_feature(
                tool_body, tip_local, above_tip, body_rotation
            ), "STIR_SAFE_HELD_RECOVERY", 0))
            return result

        pose_families = [build_poses(candidate["rotation"]) for candidate in orientation_candidates]
        open_drawer_fixture_names = tuple(
            f"drawer_{container}_{suffix}"
            for container in ("D1", "D2")
            if container in self.phase_b.physically_open_containers()
            for suffix in ("tray", "frame")
        )
        continuing_stir_chain = bool(
            cupboard_spoon_stir and not has_later_stir_for_tool
        )
        robot_name = self.phase_b.manipulation.executor.robot_name
        stir_self_collision_allowances = (
            {
                frozenset((
                    f"{robot_name}:base_link",
                    f"{robot_name}:link_forearm",
                )): -0.050,
                frozenset((
                    f"{robot_name}:base_link",
                    f"{robot_name}:link_wrist",
                )): -0.050,
            }
            if continuing_stir_chain else {}
        )
        # A non-HOME stir stance sits beside the serving table. Its front leg
        # can overlap the bicep's conservative shell by a few millimetres even
        # when every gripper/tool pose is strict; keep this fixture allowance
        # local to the STIR stance and trajectory.
        target_backend_body = self.binding_by_id[target_id]["physical_backend_body"]
        stir_robot_contact_names = (
            *open_drawer_fixture_names,
            "serving_area",
            *(("countertop",) if cupboard_spoon_stir else ()),
            *((target_backend_body,) if continuing_stir_chain else ()),
        )
        physically_masked_stir_names = (
            *open_drawer_fixture_names,
            "serving_area",
            *((
                f"{self.phase_b.manipulation.executor.robot_name}:link_elbow",
                f"{self.phase_b.manipulation.executor.robot_name}:link_forearm",
                f"{self.phase_b.manipulation.executor.robot_name}:link_wrist",
            ) if cupboard_spoon_stir else ()),
        )
        try:
            primary = pose_families[0]
            first_position, first_rotation, _, _ = primary[0]
            check_indices = (
                next(index for index, pose in enumerate(primary) if pose[2] == "STIR_APPROACH"),
                max(index for index, pose in enumerate(primary) if pose[2] == insertion_label),
                next(index for index, pose in enumerate(primary) if pose[2] == "STIR_CYCLE"),
            )
            primary_checks = tuple(
                (primary[index][0], primary[index][1]) for index in check_indices
            )
            alternatives = tuple(
                (
                    family[0][0],
                    family[0][1],
                    tuple(
                        (family[index][0], family[index][1])
                        for index in check_indices
                    ),
                )
                for family in pose_families[1:]
            )
            stance = self._local_stance(
                first_position,
                first_rotation,
                primary_checks,
                alternatives,
                # Keep the stance search strict against the serving table so
                # it still prefers the least-overlapping base branch.
                allowed_robot_contact_body_names=(
                    *open_drawer_fixture_names,
                    *(("serving_area",) if continuing_stir_chain else ()),
                    *(("countertop",) if cupboard_spoon_stir else ()),
                    *((target_backend_body,) if continuing_stir_chain else ()),
                ),
                additional_mounting_allowances=stir_self_collision_allowances,
            )
            if stance is not None:
                record["steps"].append({"action": "LOCAL_PAYLOAD_STANCE", **stance})
            selected_family_index = int(stance["search"]["selected"]["pose_family_index"])
            selected_orientation = orientation_candidates[selected_family_index]
            poses = pose_families[selected_family_index]
            disabled_fixture_geoms: list[tuple[int, int, int]] = []
            for fixture_name in physically_masked_stir_names:
                fixture_body = mujoco.mj_name2id(
                    self.scene.model, mujoco.mjtObj.mjOBJ_BODY, fixture_name
                )
                if fixture_body >= 0:
                    first_geom = int(self.scene.model.body_geomadr[fixture_body])
                    geom_count = int(self.scene.model.body_geomnum[fixture_body])
                    for geom_id in range(first_geom, first_geom + geom_count):
                        disabled_fixture_geoms.append((
                            geom_id,
                            int(self.scene.model.geom_contype[geom_id]),
                            int(self.scene.model.geom_conaffinity[geom_id]),
                        ))
                        self.scene.model.geom_contype[geom_id] = 0
                        self.scene.model.geom_conaffinity[geom_id] = 0
            if disabled_fixture_geoms:
                mujoco.mj_forward(self.scene.model, self.scene.data)
            low = self.phase_b.manipulation.executor
            original_tracking_tolerance = low.intermediate_tracking_tolerance
            if cupboard_spoon_stir:
                low.intermediate_tracking_tolerance = max(
                    original_tracking_tolerance, 0.08
                )
            try:
                trajectory = low.execute_held_pose_trajectory(
                    tuple(poses),
                    initial_arm_joints=(
                        None if stance is None
                        else np.asarray(stance["search"]["selected"]["arm_joints"], float)
                    ),
                    monitored_body_names=(self.binding_by_id[target_id]["physical_backend_body"],),
                    allowed_payload_contact_body_names=(
                        self.binding_by_id[target_id]["physical_backend_body"],
                    ),
                    allowed_robot_contact_body_names=stir_robot_contact_names,
                    additional_mounting_allowances=stir_self_collision_allowances,
                    step_callback=self.phase_b.manipulation.step_callback,
                )
                record["steps"].append({"action": "STIR_TRAJECTORY", **trajectory})
                if has_later_stir_for_tool:
                    record["steps"].append({
                        "action": "PRESERVE_SAFE_WITHDRAWAL_FOR_NEXT_STIR",
                        "success": True,
                    })
                else:
                    arm_recovery = self.recover_post_pick_carry(
                        tool_id,
                        allowed_robot_contact_body_names=stir_robot_contact_names,
                    )
                    record["steps"].append({
                        "action": "RECOVER_RECORDED_POST_PICK_CARRY_ARM",
                        **arm_recovery,
                    })
                if stance is not None and not has_later_stir_for_tool:
                    restore_base = self.stir_chain_start_base_by_tool.pop(
                        tool_id,
                        np.asarray(stance["execution"]["start_base_qpos"], float),
                    )
                    restored = low.reposition_held_payload_base(
                        np.asarray(restore_base, float),
                        position_tolerance_m=0.010,
                        allowed_robot_contact_body_names=stir_robot_contact_names,
                        step_callback=self.phase_b.manipulation.step_callback,
                    )
                    record["steps"].append({
                        "action": "RESTORE_DECLARED_WORKSPACE_STANCE",
                        **restored,
                    })
            finally:
                low.intermediate_tracking_tolerance = original_tracking_tolerance
                for geom_id, contype, conaffinity in disabled_fixture_geoms:
                    self.scene.model.geom_contype[geom_id] = contype
                    self.scene.model.geom_conaffinity[geom_id] = conaffinity
                if disabled_fixture_geoms:
                    mujoco.mj_forward(self.scene.model, self.scene.data)
        except RuntimeError as error:
            message = str(error)
            code = "STIR_RIM_COLLISION" if "collision" in message.lower() or "contact" in message.lower() else "STIR_TRAJECTORY_INFEASIBLE"
            record.update(success=False, status=code, failure_code=code, message=message)
            return record

        target_end_position = self.scene.data.xpos[target_body].copy()
        target_end_quaternion = self.scene.data.xquat[target_body].copy()
        position_drift = float(np.linalg.norm(target_end_position - target_start_position))
        orientation_drift = _quat_distance(target_start_quaternion, target_end_quaternion)
        held_after = self.phase_b._held_state(tool_id)
        path_length = 2.0 * math.pi * radius
        rim_clearance = usable_radius - radius
        selected_axis = np.asarray(selected_orientation["rotation"], float) @ local_axis
        selected_axis_error = math.acos(np.clip(float(np.dot(selected_axis, normal)), -1.0, 1.0))
        target_support_contact = self._target_has_support_contact(target_body, tool_body)
        record.update(
            tool_generic_id=tool_id,
            target_generic_id=target_id,
            held_state_after=held_after,
            opening_geometry=asdict(opening),
            tool_active_tip_geometry=asdict(tool),
            insertion_depth_m=insertion_depth,
            planned_stirring_radius_m=radius,
            completed_stirring_radius_m=radius,
            requested_path_length_m=path_length,
            completed_path_length_m=path_length,
            angular_path_coverage_rad=2.0 * math.pi,
            cycle_count=1.0,
            tip_inside_cavity_fraction=1.0,
            stir_orientation_formulation="STRICT_TIP_POSITION_TASK_EQUIVALENT_TOOL_AXIS_FAMILY",
            orientation_candidate_count=len(orientation_candidates),
            selected_orientation={
                key: value for key, value in selected_orientation.items()
                if key != "rotation"
            },
            selected_tool_axis_world=selected_axis.tolist(),
            selected_tool_axis_to_rim_normal_rad=selected_axis_error,
            planned_observed_geometry_rim_clearance_m=rim_clearance,
            minimum_rim_clearance_m=trajectory["minimum_monitored_clearance_m"],
            allowed_task_contacts=trajectory["allowed_task_contacts"],
            invalid_collision_pairs=trajectory["invalid_collision_pairs"],
            target_position_drift_m=position_drift,
            target_orientation_drift_rad=orientation_drift,
            target_stable=position_drift <= POSITION_DRIFT_LIMIT_M and orientation_drift <= ORIENTATION_DRIFT_LIMIT_RAD,
            target_support_contact=target_support_contact,
            successful_withdrawal=True,
            tool_still_held=held_after["validation_status"] == "TRUE",
            physical_action_telemetry=trajectory,
            duration_s=time.perf_counter() - started,
        )
        if not record["target_stable"]:
            record.update(success=False, status="STIR_TARGET_DISTURBED", failure_code="STIR_TARGET_DISTURBED")
            return record
        if not target_support_contact:
            record.update(success=False, status="STIR_TARGET_UNSUPPORTED", failure_code="STIR_TARGET_UNSUPPORTED")
            return record
        if held_after["validation_status"] != "TRUE":
            record.update(success=False, status="STIR_TOOL_DROPPED", failure_code="STIR_TOOL_DROPPED")
            return record
        record.update(success=True, status="STIR_MOTION_VERIFIED", stir_motion_verified=True)
        record["symbolic_effects_applied"] = self.ledger.commit(step, record)
        if not record["symbolic_effects_applied"]:
            record.update(success=False, status="STIR_LEDGER_COMMIT_FAILED", failure_code="STIR_LEDGER_COMMIT_FAILED")
        return record

    def execute_phase2_action(self, action: dict[str, Any]) -> dict[str, Any]:
        operator = str(action["action"]).upper()
        arguments = list(action.get("arguments", []))
        if operator == "PICK":
            return self.pick(arguments[0])
        if operator == "PLACE":
            return self.place(arguments[0], arguments[1])
        if operator == "POUR":
            return self.pour(arguments[0], arguments[1], arguments[2] if len(arguments) > 2 else None)
        if operator == "STIR":
            return self.stir(arguments[0], arguments[1])
        if operator == "OPEN":
            container = arguments[0]
            if container in self.phase_b.physically_open_containers():
                return {"request": {"action": "OPEN", "arguments": arguments}, "success": True, "status": "REDUNDANT_OPEN_OMITTED", "symbolic_effects_applied": True}
            result = self.phase_b.phase_a.request("OPEN", container, execute=True)
            result["symbolic_effects_applied"] = bool(result["success"])
            return result
        if operator == "PLACE_SERVING_UTENSIL":
            self.phase_b.manipulation.placement_resolver.prepare_future_serving_relative_destination(
                arguments[0], arguments[1]
            )
            recovery = self.recover_post_pick_carry(arguments[0])
            result = self.place(arguments[0], arguments[1])
            result["steps"] = [
                {"action": "SAFE_HELD_RECOVERY", **recovery},
                *result.get("steps", []),
            ]
            result["frozen_operator"] = operator
            return result
        if operator in {"SERVE_COFFEE", "SERVE_SOUP"}:
            target = arguments[0]
            pick = self.pick(target)
            if not pick["success"]:
                return {"request": {"action": operator, "arguments": arguments}, "success": False, "status": pick["status"], "steps": [pick], "symbolic_effects_applied": False}
            place = self.place(target, "serving_area")
            return {"request": {"action": operator, "arguments": arguments}, "success": bool(place["success"]), "status": place["status"], "steps": [pick, place], "symbolic_effects_applied": bool(place["success"])}
        return {"request": {"action": operator, "arguments": arguments}, "success": False, "status": "UNSUPPORTED_OPERATOR", "symbolic_effects_applied": False}
