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
        self.frozen_plan = frozen_plan
        self.ledger = PhaseCExecutionLedger(frozen_plan)
        self.expected_pairs = {
            row["action"].upper(): {
                tuple(row.get("arguments", [])[:2]): int(row["step"])
                for row in frozen_plan
                if row["action"].upper() == row["action"].upper()
            }
            for row in frozen_plan
        }
        self.expected_pairs = {
            operator: {
                tuple(row.get("arguments", [])[:2]): int(row["step"])
                for row in frozen_plan if row["action"].upper() == operator
            }
            for operator in ("POUR", "STIR")
        }

    @property
    def current_workspace(self) -> KitchenWorkspace:
        return self.phase_b.current_workspace

    def pick(self, object_id: str) -> dict[str, Any]:
        return self.phase_b.pick(object_id)

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
    ) -> dict[str, Any] | None:
        low = self.phase_b.manipulation.executor
        data = self.scene.data
        saved_qpos, saved_qvel, saved_ctrl = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()
        current_base = saved_qpos[low.base_qpos].copy()
        candidates = []
        if self.current_workspace == KitchenWorkspace.HOME:
            nominal_lateral = float(np.clip(-float(grip_position[0]), -0.42, 0.42))
            for forward in (0.22, 0.25, 0.28):
                for lateral_delta in (0.0, -0.06, 0.06, -0.12, 0.12):
                    for yaw in (
                        0.0, -0.15, 0.15, -0.30, 0.30,
                        -0.50, 0.50, -0.75, 0.75, -1.00, 1.00,
                    ):
                        candidates.append(np.array((
                            forward,
                            float(np.clip(nominal_lateral + lateral_delta, -0.42, 0.42)),
                            yaw,
                        )))
        else:
            for forward_delta in (-0.10, 0.0, 0.10, 0.20, 0.30):
                for lateral_delta in (-0.10, 0.0, 0.10, 0.20, 0.30, 0.35):
                    for yaw_delta in (
                        -0.40, 0.0, 0.40, 0.80, 1.00, 1.20, 1.40, 1.60,
                    ):
                        candidate = current_base + np.array((
                            forward_delta, lateral_delta, yaw_delta
                        ))
                        if float(np.linalg.norm(
                            candidate[:2] - current_base[:2]
                        )) <= 0.65:
                            candidates.append(candidate)
        rows = []
        held_body = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, str(low.held_object)
        )
        stance_seeds = (
            ("LIVE_CARRY_ARM", saved_qpos[low.arm_qpos].copy()),
            ("NAVIGATION_ARM", low.profile.navigation_joints.copy()),
        )
        try:
            for index, candidate in enumerate(candidates):
                for seed_name, initial_seed in stance_seeds:
                    data.qpos[:] = saved_qpos
                    data.qvel[:] = 0.0
                    data.qpos[low.base_qpos] = candidate
                    data.qpos[low.arm_qpos] = initial_seed
                    mujoco.mj_forward(self.scene.model, data)
                    ik = ProfiledIK(
                        self.scene.model, data, low.profile, orientation_weight=0.45
                    )
                    checker = RobotConfigurationCollisionChecker(
                        self.scene.model, data, low.profile,
                        mounting_allowances=low.mounting_allowances,
                    )
                    seed = initial_seed.copy()
                    position_error = angle_error = 0.0
                    collision_valid, reason = True, None
                    pose_diagnostics = []
                    for pose_index, (pose_position, pose_rotation) in enumerate((
                        (np.asarray(grip_position, float), np.asarray(grip_rotation, float)),
                        *additional_poses,
                    )):
                        joints, pose_position_error, pose_angle_error = ik.solve(
                            np.asarray(pose_position, float), seed,
                            np.asarray(pose_rotation, float),
                        )
                        position_error = max(position_error, float(pose_position_error))
                        angle_error = max(angle_error, float(pose_angle_error))
                        collision_valid, reason = checker.segment_valid(
                            seed, joints, frozenset((held_body,)), resolution=0.025
                        )
                        pose_diagnostics.append({
                            "pose_index": pose_index,
                            "position_error_m": float(pose_position_error),
                            "orientation_error_rad": float(pose_angle_error),
                            "collision_valid": bool(collision_valid),
                            "collision_failure": reason,
                        })
                        if (
                            pose_position_error > low.ik_position_tolerance
                            or pose_angle_error > low.ik_angle_tolerance
                            or not collision_valid
                        ):
                            break
                        seed = joints
                    valid = bool(
                        len(pose_diagnostics) == 1 + len(additional_poses)
                        and position_error <= low.ik_position_tolerance
                        and angle_error <= low.ik_angle_tolerance
                        and collision_valid
                    )
                    rows.append({
                        "candidate_index": index,
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
        finally:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            data.ctrl[:] = saved_ctrl
            mujoco.mj_forward(self.scene.model, data)
        valid_rows = [row for row in rows if row["valid"]]
        if not valid_rows:
            best = min(
                rows,
                key=lambda row: row["strict_ik_position_error_m"]
                + row["strict_ik_orientation_error_rad"],
                default=None,
            )
            raise RuntimeError(f"No payload-safe strict-IK HOME stance; best={best}")
        selected = min(
            valid_rows,
            key=lambda row: (
                float(np.linalg.norm(np.asarray(row["base_qpos"]) - current_base)),
                row["strict_ik_position_error_m"] + row["strict_ik_orientation_error_rad"],
                row["candidate_index"],
            ),
        )
        target = np.asarray(selected["base_qpos"], float)
        reposition = low.reposition_held_payload_base(
            target, step_callback=self.phase_b.manipulation.step_callback
        )
        return {
            "search": {
                "strategy": "BOUNDED_LOCAL_PAYLOAD_SAFE_STRICT_PHASE_C_IK",
                "workspace": self.current_workspace.value,
                "candidate_count": len(rows),
                "selected": selected,
                "candidates": rows,
            },
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
        body_z = body_rotation[:, 2]
        upright_rotation = _align_vectors(body_z, np.array((0.0, 0.0, 1.0))) @ body_rotation
        # Candidate is family-level and world-frame deterministic; outlet
        # alignment is solved independently, so tilt direction is selected
        # for wrist feasibility rather than inferred from a generic ID.
        tilt_axis = np.array((0.0, -1.0, 0.0))
        tilt = spec.tilt_candidates_rad[0]
        tilted_rotation = rotation_about_axis(tilt_axis, tilt) @ upright_rotation
        outlet_local = np.asarray(spec.outlet_local_m, float)
        source_direction = self.scene.data.xpos[source_body] - opening_centre
        source_direction -= normal * float(np.dot(source_direction, normal))
        if np.linalg.norm(source_direction) < 1e-9:
            source_direction = np.array((1.0, 0.0, 0.0))
        source_direction /= np.linalg.norm(source_direction)
        radial_offset = max(
            0.0,
            min(opening.opening_half_extents_m)
            - opening.safety_margin_m
            - 0.009,
        )
        aligned_outlet = opening_centre + source_direction * radial_offset
        pre_height = 0.26 if family == "JAR_SOURCE" else 0.055
        pre_outlet = aligned_outlet + normal * pre_height
        pour_height = 0.230 if family == "JAR_SOURCE" else 0.040
        pour_outlet = aligned_outlet + normal * pour_height
        high_outlet = aligned_outlet + normal * (pre_height + 0.06)
        high_pose = self._grip_pose_for_body_feature(
            source_body, outlet_local, high_outlet, upright_rotation
        )
        pre_pose = self._grip_pose_for_body_feature(source_body, outlet_local, pre_outlet, upright_rotation)
        tilt_pose = self._grip_pose_for_body_feature(source_body, outlet_local, pour_outlet, tilted_rotation)
        try:
            stance = self._local_stance(
                high_pose[0],
                high_pose[1],
                ((pre_pose[0], pre_pose[1]), (tilt_pose[0], tilt_pose[1])),
            )
            if stance is not None:
                record["steps"].append({"action": "LOCAL_PAYLOAD_STANCE", **stance})
                # Recompute the grasp transform after the real base motion.
                high_pose = self._grip_pose_for_body_feature(source_body, outlet_local, high_outlet, upright_rotation)
                pre_pose = self._grip_pose_for_body_feature(source_body, outlet_local, pre_outlet, upright_rotation)
                tilt_pose = self._grip_pose_for_body_feature(source_body, outlet_local, pour_outlet, tilted_rotation)
            low = self.phase_b.manipulation.executor
            safe_pose = (
                self.scene.data.site_xpos[low.grip_site_id].copy(),
                self.scene.data.site_xmat[low.grip_site_id].reshape(3, 3).copy(),
            )
            dwell_steps = max(1, int(round(spec.dwell_time_s / self.scene.model.opt.timestep)))
            trajectory = self.phase_b.manipulation.executor.execute_held_pose_trajectory(
                (
                    (high_pose[0], high_pose[1], "POUR_HIGH_CLEARANCE_APPROACH", 0),
                    (pre_pose[0], pre_pose[1], "POUR_APPROACH", 0),
                    (tilt_pose[0], tilt_pose[1], "POUR_TILT", dwell_steps),
                    (pre_pose[0], pre_pose[1], "POUR_UPRIGHT_RECOVERY", 0),
                    (high_pose[0], high_pose[1], "POUR_HIGH_CLEARANCE_RECOVERY", 0),
                    (safe_pose[0], safe_pose[1], "POUR_SAFE_HELD_RECOVERY", 0),
                ),
                initial_arm_joints=(
                    None if stance is None
                    else np.asarray(stance["search"]["selected"]["arm_joints"], float)
                ),
                monitored_body_names=(self.binding_by_id[target_id]["physical_backend_body"],),
                step_callback=self.phase_b.manipulation.step_callback,
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
            selected_base_stance=record["steps"][-3].get("search") if len(record["steps"]) >= 3 and record["steps"][-3].get("action") == "LOCAL_PAYLOAD_STANCE" else None,
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
            source_returned_upright=True,
            source_still_held=held_after["validation_status"] == "TRUE",
            physical_action_telemetry=trajectory,
            duration_s=time.perf_counter() - started,
        )
        if position_drift > POSITION_DRIFT_LIMIT_M or orientation_drift > ORIENTATION_DRIFT_LIMIT_RAD:
            record.update(success=False, status="POUR_TARGET_DISTURBED", failure_code="POUR_TARGET_DISTURBED")
            return record
        if held_after["validation_status"] != "TRUE":
            record.update(success=False, status="POUR_SOURCE_DROPPED", failure_code="POUR_SOURCE_DROPPED")
            return record
        if interior_margin <= 0.0:
            record.update(success=False, status="POUR_ALIGNMENT_FAILED", failure_code="POUR_ALIGNMENT_FAILED")
            return record
        record.update(success=True, status="POUR_MOTION_VERIFIED", pour_motion_verified=True)
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
        if self.current_workspace == KitchenWorkspace.HOME:
            approach_body_rotation = current_body_rotation
        else:
            grip_position = self.scene.data.site_xpos[
                self.phase_b.manipulation.executor.grip_site_id
            ].copy()
            handle_direction = grip_position - opening_centre
            handle_direction -= normal * float(np.dot(handle_direction, normal))
            if np.linalg.norm(handle_direction) < 1e-9:
                handle_direction = np.array((1.0, 0.0, 0.0))
            handle_direction /= np.linalg.norm(handle_direction)
            target_tool_axis = (
                math.cos(math.radians(60.0)) * normal
                + math.sin(math.radians(60.0)) * handle_direction
            )
            approach_body_rotation = _align_vectors(
                current_body_rotation @ local_axis, target_tool_axis
            ) @ current_body_rotation
        cycle_body_rotation = approach_body_rotation
        tip_local = np.asarray(tool.active_tip_local_m, float)
        insertion_depth = min(
            0.20 * opening.cavity_depth_m,
            0.010,
            opening.cavity_depth_m - opening.safety_margin_m,
        )
        tool_radius = 0.5 * min(float(observed.get("width", 0.0)), float(observed.get("height", 0.0)))
        usable_radius = min(opening.opening_half_extents_m) - opening.safety_margin_m - tool_radius
        radius = 0.20 * usable_radius
        if radius <= 0.003 or insertion_depth <= opening.safety_margin_m:
            record.update(success=False, status="STIR_INSERTION_INFEASIBLE", failure_code="STIR_INSERTION_INFEASIBLE")
            return record
        tangent_x = cycle_body_rotation[:, 0].copy()
        tangent_x -= normal * float(np.dot(tangent_x, normal))
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(normal, tangent_x)
        approach_clearance = (
            0.10 if self.current_workspace == KitchenWorkspace.HOME else 0.035
        )
        above_tip = opening_centre + normal * approach_clearance
        centre_tip = opening_centre - normal * insertion_depth
        poses = [(*self._grip_pose_for_body_feature(
            tool_body, tip_local, above_tip, approach_body_rotation
        ), "STIR_APPROACH", 0)]
        for fraction in np.linspace(0.125, 1.0, 8):
            insertion_tip = above_tip + float(fraction) * (centre_tip - above_tip)
            poses.append((*self._grip_pose_for_body_feature(
                tool_body, tip_local, insertion_tip, approach_body_rotation
            ), "STIR_INSERTION", 0))
        segments = 20
        for index in range(segments + 1):
            angle = 2.0 * math.pi * index / segments
            tip = centre_tip + radius * (math.cos(angle) * tangent_x + math.sin(angle) * tangent_y)
            poses.append((*self._grip_pose_for_body_feature(
                tool_body, tip_local, tip, cycle_body_rotation
            ), "STIR_CYCLE", 0))
        for fraction in np.linspace(0.125, 1.0, 8):
            withdrawal_tip = centre_tip + float(fraction) * (above_tip - centre_tip)
            poses.append((*self._grip_pose_for_body_feature(
                tool_body, tip_local, withdrawal_tip, approach_body_rotation
            ), "STIR_WITHDRAWAL", 0))
        recovery_tip = opening_centre + normal * 0.10
        poses.append((*self._grip_pose_for_body_feature(
            tool_body, tip_local, recovery_tip, approach_body_rotation
        ), "STIR_SAFE_HELD_RECOVERY", 0))
        try:
            first_position, first_rotation, _, _ = poses[0]
            stance = self._local_stance(
                first_position,
                first_rotation,
            )
            if stance is not None:
                record["steps"].append({"action": "LOCAL_PAYLOAD_STANCE", **stance})
            trajectory = self.phase_b.manipulation.executor.execute_held_pose_trajectory(
                tuple(poses),
                initial_arm_joints=(
                    None if stance is None
                    else np.asarray(stance["search"]["selected"]["arm_joints"], float)
                ),
                monitored_body_names=(self.binding_by_id[target_id]["physical_backend_body"],),
                step_callback=self.phase_b.manipulation.step_callback,
            )
            record["steps"].append({"action": "STIR_TRAJECTORY", **trajectory})
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
            planned_observed_geometry_rim_clearance_m=rim_clearance,
            minimum_rim_clearance_m=trajectory["minimum_monitored_clearance_m"],
            invalid_collision_pairs=trajectory["invalid_collision_pairs"],
            target_position_drift_m=position_drift,
            target_orientation_drift_rad=orientation_drift,
            target_stable=position_drift <= POSITION_DRIFT_LIMIT_M and orientation_drift <= ORIENTATION_DRIFT_LIMIT_RAD,
            successful_withdrawal=True,
            tool_still_held=held_after["validation_status"] == "TRUE",
            physical_action_telemetry=trajectory,
            duration_s=time.perf_counter() - started,
        )
        if not record["target_stable"]:
            record.update(success=False, status="STIR_TARGET_DISTURBED", failure_code="STIR_TARGET_DISTURBED")
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
            result = self.place(arguments[0], arguments[1])
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
