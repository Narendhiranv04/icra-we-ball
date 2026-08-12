"""Unified Phase-B refinement over the existing Phase-A execution stack."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import mujoco
import numpy as np

from .kitchen_execution_policy import KitchenWorkspace
from .kitchen_google_execution import KitchenGoogleExecutionDispatcher
from .kitchen_object_manipulation import (
    KitchenObjectManipulationExecutor,
    ObjectExecutionFailureCode,
    inspect_held_object_state,
)


class KitchenPhaseBExecutionDispatcher:
    """Refine generic PICK/PLACE into access, manipulation, and transport.

    The dispatcher never changes the frozen object assignment. Simulator body
    names are confined to the execution resolution passed to its constructor.
    """

    def __init__(self, scene, inventory: dict[str, Any], resolution: dict[str, Any], *, step_callback=None):
        self.scene = scene
        self.inventory = inventory
        self.resolution = resolution
        self.manipulation = KitchenObjectManipulationExecutor(
            scene, inventory, resolution, step_callback=step_callback
        )
        self.phase_a = KitchenGoogleExecutionDispatcher(
            scene,
            held_object_getter=lambda: self.manipulation.executor.held_object,
            step_callback=step_callback,
        )
        self.inventory_by_id = {
            row["generic_object_id"]: row for row in inventory["objects"]
        }
        self.binding_by_id = {
            row["generic_object_id"]: row for row in resolution["accepted"]
        }

    @property
    def current_workspace(self) -> KitchenWorkspace:
        return self.phase_a.current_workspace

    def physically_open_containers(self) -> set[str]:
        return {
            region
            for region, state in self.scene.get_region_observation_states().items()
            if bool(state["open"])
        }

    def _held_state(self, object_id: str) -> dict[str, Any]:
        binding = self.binding_by_id[object_id]
        state = inspect_held_object_state(
            self.scene.model,
            self.scene.data,
            object_id,
            binding["physical_backend_body"],
            {row["physical_backend_body"] for row in self.resolution["accepted"]},
        )
        return asdict(state)

    def move(self, workspace: KitchenWorkspace, *, carrying_object_id: str | None = None) -> dict[str, Any]:
        if self.current_workspace == workspace:
            return {
                "action": "MOVE",
                "target_workspace": workspace.value,
                "success": True,
                "status": "REDUNDANT_MOVE_OMITTED",
                "automatically_inserted": True,
                "physics_steps": 0,
            }
        before = None
        if carrying_object_id:
            before = self._held_state(carrying_object_id)
            if before["validation_status"] != "TRUE":
                return {
                    "action": "MOVE",
                    "success": False,
                    "status": "HELD_STATE_INVALID",
                    "failure_code": ObjectExecutionFailureCode.HELD_STATE_INVALID.value,
                    "held_state_before": before,
                }
        try:
            record = self.phase_a._move(workspace)
        except RuntimeError as error:
            if carrying_object_id and "compact navigation pose" in str(error):
                try:
                    preparation = (
                        self.manipulation.executor.fold_held_payload_for_navigation(
                            step_callback=self.manipulation.step_callback
                        )
                    )
                    record = self.phase_a._move(workspace)
                    record["held_navigation_preparation"] = preparation
                except RuntimeError as retry_error:
                    return {
                        "action": "MOVE",
                        "target_workspace": workspace.value,
                        "success": False,
                        "status": "MOVE_PLANNING_FAILED",
                        "failure_code": "MOVE_PLANNING_FAILED",
                        "message": str(retry_error),
                        "initial_planning_failure": str(error),
                        "held_state_before": before,
                    }
            else:
                return {
                    "action": "MOVE",
                    "target_workspace": workspace.value,
                    "success": False,
                    "status": "MOVE_PLANNING_FAILED",
                    "failure_code": "MOVE_PLANNING_FAILED",
                    "message": str(error),
                    "held_state_before": before,
                }
        record["automatically_inserted"] = True
        record["held_object_included_in_collision_check"] = bool(carrying_object_id)
        if carrying_object_id and record["success"]:
            after = self._held_state(carrying_object_id)
            record["held_state_before"] = before
            record["held_state_after"] = after
            before_position = before.get("relative_position_m")
            after_position = after.get("relative_position_m")
            before_quaternion = before.get("relative_orientation_wxyz")
            after_quaternion = after.get("relative_orientation_wxyz")
            position_drift = None
            orientation_drift = None
            if before_position is not None and after_position is not None:
                position_drift = float(
                    np.linalg.norm(
                        np.asarray(after_position, dtype=float)
                        - np.asarray(before_position, dtype=float)
                    )
                )
            if before_quaternion is not None and after_quaternion is not None:
                before_q = np.asarray(before_quaternion, dtype=float)
                after_q = np.asarray(after_quaternion, dtype=float)
                before_q /= max(float(np.linalg.norm(before_q)), 1e-12)
                after_q /= max(float(np.linalg.norm(after_q)), 1e-12)
                orientation_drift = float(
                    2.0
                    * np.arccos(
                        np.clip(abs(float(np.dot(before_q, after_q))), 0.0, 1.0)
                    )
                )
            record["relative_position_drift_m"] = position_drift
            record["relative_orientation_drift_rad"] = orientation_drift
            record["relative_transform_drift_thresholds"] = {
                "position_m": 0.005,
                "orientation_rad": float(np.deg2rad(2.0)),
            }
            drift_valid = (
                (position_drift is None or position_drift <= 0.005)
                and (
                    orientation_drift is None
                    or orientation_drift <= float(np.deg2rad(2.0))
                )
            )
            record["relative_transform_drift_valid"] = drift_valid
            if after["validation_status"] != "TRUE" or not drift_valid:
                record.update(
                    success=False,
                    status="OBJECT_DROPPED",
                    failure_code=ObjectExecutionFailureCode.OBJECT_DROPPED.value,
                )
        if record["success"]:
            self.manipulation.sync_workspace(workspace)
        return record

    def pick(self, object_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "request": {"action": "PICK", "arguments": [object_id]},
            "generic_object_id": object_id,
            "steps": [],
            "functional_assignment_changed": False,
        }
        row = self.inventory_by_id.get(object_id)
        if row is None or object_id not in self.binding_by_id:
            record.update(success=False, status="ENTITY_RESOLUTION_FAILED")
            return record
        context = row["source_context"]
        required = KitchenWorkspace(context["required_workspace"])
        if self.current_workspace != required:
            movement = self.move(required)
            record["steps"].append(movement)
            if not movement["success"]:
                record.update(success=False, status=movement["status"])
                return record
        container = context.get("source_container")
        if container and container not in self.physically_open_containers():
            opened = self.phase_a.request("OPEN", container, execute=True)
            opened["automatically_inserted"] = True
            record["steps"].append(opened)
            if not opened["success"]:
                record.update(success=False, status="CONTAINER_OPEN_FAILED")
                return record
            # Most fixtures preserve authored storage poses only through
            # opening.  C2 retains presentation support through collision/IK
            # approach, then the low-level executor releases it immediately
            # before Cartesian pre-close and live bilateral contact.
            defer_fixture_release = container == "C2"
            record["storage_fixture_release_deferred_to_manipulation_stance"] = (
                defer_fixture_release
            )
            record["storage_fixture_released"] = bool(
                False if defer_fixture_release
                else self.scene.release_storage_fixture(container)
            )
            record["storage_fixture_active_before_grasp_planning"] = bool(
                defer_fixture_release
            )
            if record["storage_fixture_released"]:
                for _ in range(120):
                    mujoco.mj_step(self.scene.model, self.scene.data)
        elif container:
            record["redundant_open_omitted"] = True
        result = self.manipulation.pick(
            object_id, self.current_workspace, self.physically_open_containers()
        )
        record["steps"].append(asdict(result))
        record.update(
            success=result.success,
            status=result.status,
            failure_code=result.failure_code,
            duration_s=time.perf_counter() - started,
            post_pick=asdict(result),
        )
        fixture_audit = self.manipulation.executor.storage_fixture_release_telemetry
        if fixture_audit is not None:
            record["storage_fixture_release"] = fixture_audit
            record["storage_fixture_released"] = bool(
                fixture_audit.get("released")
            )
            record["storage_fixture_active_during_contact"] = bool(
                fixture_audit.get("active_during_contact")
            )
        return record

    def place(self, object_id: str, destination: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "request": {"action": "PLACE", "arguments": [object_id, destination]},
            "steps": [],
            "functional_assignment_changed": False,
        }
        if object_id not in self.binding_by_id:
            record.update(success=False, status="ENTITY_RESOLUTION_FAILED")
            return record
        held_before = self._held_state(object_id)
        if held_before["validation_status"] != "TRUE":
            record.update(
                success=False,
                status="HELD_STATE_INVALID",
                held_state_before_place=held_before,
            )
            return record
        try:
            target = self.manipulation.placement_resolver.resolve(object_id, destination)
        except ValueError as error:
            record.update(success=False, status="DESTINATION_RESOLUTION_FAILED", message=str(error))
            return record
        if self.current_workspace != target.required_workspace:
            movement = self.move(target.required_workspace, carrying_object_id=object_id)
            record["steps"].append(movement)
            if not movement["success"]:
                record.update(success=False, status=movement["status"])
                return record
        result = self.manipulation.place(object_id, destination, self.current_workspace)
        record["steps"].append(asdict(result))
        record.update(
            success=result.success,
            status=result.status,
            failure_code=result.failure_code,
            held_state_before_place=held_before,
            post_place=asdict(result),
        )
        return record

    def execute_phase2_action(self, action: dict[str, Any]) -> dict[str, Any]:
        operator = str(action["action"]).upper()
        arguments = list(action.get("arguments", []))
        if operator == "PICK":
            return self.pick(arguments[0])
        if operator == "PLACE":
            return self.place(arguments[0], arguments[1])
        if operator in {"POUR", "STIR"}:
            return {
                "request": {"action": operator, "arguments": arguments},
                "success": False,
                "status": ObjectExecutionFailureCode.UNSUPPORTED_PHASE_C_OPERATOR.value,
                "failure_code": ObjectExecutionFailureCode.UNSUPPORTED_PHASE_C_OPERATOR.value,
                "symbolic_effects_applied": False,
            }
        return {
            "request": {"action": operator, "arguments": arguments},
            "success": False,
            "status": "UNSUPPORTED_OPERATOR",
            "symbolic_effects_applied": False,
        }
