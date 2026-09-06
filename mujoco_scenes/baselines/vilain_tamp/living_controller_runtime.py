"""Neutral live PICK/PLACE primitives for the Living Room baseline bridge."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import ExecutionProjection
from .identity import EntityBinding, resolve_geometry_entity_name


class LivingControllerError(RuntimeError):
    """The generic Living Room controller could not complete a primitive."""


class LivingRoomPhysicalController:
    """Use the existing mobile and calibrated manipulation controllers.

    All payload/support choices arrive through the execution projections.  The
    controller derives only collision-free stances and physical support poses.
    """

    def __init__(
        self,
        *,
        scene: Any,
        projections: Sequence[ExecutionProjection],
        bindings: Mapping[str, EntityBinding],
        fixed_bindings: Mapping[str, EntityBinding],
    ) -> None:
        import mujoco
        from mujoco_scenes.generic_manipulation import CalibratedPickPlaceExecutor
        from mujoco_scenes.living_room_mobile_execution import (
            ALLOWED_INTERACTION_BODIES,
            LivingRoomMobileExecutor,
            _configure_execution_base_limits,
            _world_to_joint_base,
            make_pick_specs,
        )

        del projections
        self.mujoco = mujoco
        self.scene = scene
        self.bindings = dict(bindings)
        self.fixed_bindings = dict(fixed_bindings)
        _configure_execution_base_limits(scene)
        self.mobile = LivingRoomMobileExecutor(scene.model, scene.data)
        payloads = {"objects": {}}
        resolution = {"objects": []}
        for object_id, binding in sorted(bindings.items()):
            role = binding.pddl_type.lower()
            if role == "remote":
                role = "tv_remote"
            if role not in {"cup", "saucer", "tv_remote"}:
                raise LivingControllerError(
                    f"unsupported Living Room physical type {binding.pddl_type!r}"
                )
            centroid = binding.entity_centroid_m or binding.observed_centroid_m
            if centroid is None:
                raise LivingControllerError(f"{object_id} has no physical centroid")
            payloads["objects"][object_id] = {
                "semantic_role": role,
                "observed_centroid_world_m": list(centroid),
            }
            resolution["objects"].append(
                {
                    "generic_object_id": object_id,
                    "backend_body": binding.entity_name,
                    "semantic_role": role,
                }
            )
        self.specs = make_pick_specs(payloads, resolution)
        self._Executor = CalibratedPickPlaceExecutor
        self._allowed = ALLOWED_INTERACTION_BODIES
        self._world_to_joint_base = _world_to_joint_base
        self.picker: Any | None = None
        self.selected_pose: Any | None = None
        self.selected_payload: str | None = None

    def move_to(
        self, target_entity: str, *, carrying_entity: str | None
    ) -> Mapping[str, Any]:
        from mujoco_scenes.living_room_mobile_execution import (
            SAUCER_BACKENDS,
            candidate_stances,
            gripper_rotation_for_object_target,
            saucer_pick_spec_for_stance,
            validate_manipulation_at_pose,
        )
        from mujoco_scenes.robot_profiles import manipulation_profile

        target = (
            self._body_position(target_entity)
            if carrying_entity is None
            else self._placement_position(carrying_entity, target_entity)
        )
        current = self.mobile.current_pose()
        checker = self.mobile.collision_checker()
        chosen = None
        path = None
        for pose in candidate_stances(target, current):
            distance = float(
                np.linalg.norm(target[:2] - np.asarray((pose.x, pose.y)))
            )
            if not 0.40 <= distance <= 1.10:
                continue
            if not checker.is_pose_valid(pose.x, pose.y, pose.yaw):
                continue
            payload = target_entity if carrying_entity is None else carrying_entity
            if payload in self.specs:
                spec = self.specs[payload]
                if payload in SAUCER_BACKENDS:
                    spec = saucer_pick_spec_for_stance(spec, target, pose)
                rotation = (
                    (
                        manipulation_profile("google").top_down_rotation
                        if spec.top_down_rotation is None
                        else spec.top_down_rotation
                    )
                    if carrying_entity is None
                    else gripper_rotation_for_object_target(
                        self.scene.model,
                        self.scene.data,
                        carrying_entity,
                        np.eye(3),
                    )
                )
                validation = validate_manipulation_at_pose(
                    self.scene.model,
                    self.scene.data,
                    pose,
                    payload,
                    target,
                    spec,
                    target_rotation=rotation,
                )
                if not validation["feasible"]:
                    continue
                self.specs[payload] = spec
            try:
                candidate_path = (
                    [current] if pose == current else self.mobile.plan(pose)
                )
            except RuntimeError:
                continue
            chosen, path = pose, candidate_path
            break
        if chosen is None or path is None:
            return {"success": False, "status": "NO_COLLISION_FREE_STANCE"}
        held_before = carrying_entity
        result = self.mobile.execute(path)
        self.selected_pose = chosen
        self.selected_payload = target_entity if carrying_entity is None else carrying_entity
        return {
            "success": True,
            "status": "PHYSICAL_BASE_MOTION_COMPLETE",
            "carrying_entity": held_before,
            **dict(result),
        }

    def pick(self, payload_entity: str) -> Mapping[str, Any]:
        if self.selected_pose is None:
            return {"success": False, "status": "MISSING_PICK_STANCE"}
        picker = self._new_picker()
        picker.request_pick(payload_entity)
        result = self._run(picker, "holding")
        if result["success"]:
            self.picker = picker
        return result

    def destination_for(
        self,
        *,
        payload_id: str,
        payload_entity: str,
        support_id: str,
        support_entity: str,
    ) -> Mapping[str, Any]:
        del payload_id, support_id
        lower, upper = self._body_aabb(support_entity)
        target = self._placement_position(payload_entity, support_entity)
        return {
            "success": True,
            "desired_body_world_m": target.tolist(),
            "support_aabb_min_m": lower.tolist(),
            "support_aabb_max_m": upper.tolist(),
        }

    def place(
        self,
        payload_entity: str,
        support_entity: str,
        destination: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from mujoco_scenes.living_room_mobile_execution import (
            gripper_rotation_for_object_target,
            resume_held_object_from_simulator,
        )

        del support_entity
        picker = self._new_picker()
        resumed = resume_held_object_from_simulator(
            picker, payload_entity, payload_entity
        )
        target = np.asarray(destination["desired_body_world_m"], dtype=float)
        rotation = gripper_rotation_for_object_target(
            picker.model, picker.data, payload_entity, np.eye(3)
        )
        picker.request_place_world(target, rotation)
        result = self._run(picker, "idle")
        return {**result, "resume_held_state": asdict(resumed)}

    def _new_picker(self) -> Any:
        if self.selected_pose is None:
            raise LivingControllerError("manipulation requires a selected base pose")
        pose_qpos = self._world_to_joint_base(self.mobile.current_pose())
        specs = dict(self.specs)
        if self.selected_payload in specs:
            target = self._body_position(str(self.selected_payload))
            direction = target[:2] - np.asarray(
                (self.selected_pose.x, self.selected_pose.y)
            )
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            carry = np.asarray(
                (
                    self.selected_pose.x + direction[0] * 0.42,
                    self.selected_pose.y + direction[1] * 0.42,
                    0.94,
                )
            )
            spec = specs[str(self.selected_payload)]
            specs[str(self.selected_payload)] = type(spec)(
                **{**spec.__dict__, "carry_position": carry}
            )
        return self._Executor(
            self.scene.model,
            self.scene.data,
            "google",
            pick_specs_override=specs,
            calibrated_objects_override=tuple(specs),
            base_stance=pose_qpos,
            base_approach_forward=0.0,
            arm_command_speed=1.35,
            intermediate_tracking_tolerance=0.065,
            allowed_collision_bodies=self._allowed,
        )

    def _run(self, picker: Any, success_mode: str) -> Mapping[str, Any]:
        steps = 0
        while picker.mode not in {success_mode, "failed"}:
            picker.update()
            self.mujoco.mj_step(self.scene.model, self.scene.data)
            steps += 1
            if steps > 60000:
                picker._fail("MANIPULATION_TIMEOUT")
        return {
            "success": picker.failure is None and picker.mode == success_mode,
            "status": (
                "PHYSICAL_MANIPULATION_COMPLETE"
                if picker.failure is None
                else "PHYSICAL_MANIPULATION_FAILED"
            ),
            "failure": picker.failure,
            "physics_steps": steps,
        }

    def _body_position(self, entity: str) -> np.ndarray:
        geometry_entity = resolve_geometry_entity_name(entity)
        body_id = self.mujoco.mj_name2id(
            self.scene.model, self.mujoco.mjtObj.mjOBJ_BODY, geometry_entity
        )
        if body_id < 0:
            raise LivingControllerError(f"unknown physical body {entity!r}")
        self.mujoco.mj_forward(self.scene.model, self.scene.data)
        return np.asarray(self.scene.data.xpos[body_id], dtype=float).copy()

    def _body_aabb(self, entity: str) -> tuple[np.ndarray, np.ndarray]:
        geometry_entity = resolve_geometry_entity_name(entity)
        body_id = self.mujoco.mj_name2id(
            self.scene.model, self.mujoco.mjtObj.mjOBJ_BODY, geometry_entity
        )
        if body_id < 0:
            raise LivingControllerError(f"unknown physical body {entity!r}")
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for geom_id in range(self.scene.model.ngeom):
            if int(self.scene.model.geom_bodyid[geom_id]) != body_id:
                continue
            rotation = np.asarray(self.scene.data.geom_xmat[geom_id]).reshape(3, 3)
            center = (
                np.asarray(self.scene.data.geom_xpos[geom_id])
                + rotation @ np.asarray(self.scene.model.geom_aabb[geom_id, :3])
            )
            half = np.abs(rotation) @ np.asarray(
                self.scene.model.geom_aabb[geom_id, 3:]
            )
            lower = np.minimum(lower, center - half)
            upper = np.maximum(upper, center + half)
        if not np.all(np.isfinite(lower)):
            raise LivingControllerError(f"physical body {entity!r} has no geometry")
        return lower, upper

    def _placement_position(self, payload_entity: str, support_entity: str) -> np.ndarray:
        lower, upper = self._body_aabb(support_entity)
        body = self._body_position(payload_entity)
        body_lower, _ = self._body_aabb(payload_entity)
        support_height = float(body[2] - body_lower[2])
        return np.asarray(
            (
                (lower[0] + upper[0]) / 2.0,
                (lower[1] + upper[1]) / 2.0,
                upper[2] + support_height,
            )
        )
