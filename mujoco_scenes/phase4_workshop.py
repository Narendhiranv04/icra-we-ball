"""Workshop adapter for the deterministic Phase-4 executor."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import mujoco
import numpy as np

from .phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase3Handoff,
    ResolvedEntity,
)
from .workshop_ground_truth_execution import (
    WorkshopExecutionDispatcher,
    validate_terminal_state,
)
from .workshop_ground_truth_planner import WorkshopAssignment
from .workshop_ground_truth_state import initial_workshop_state
from .workshop_scene import WORKSHOP_REGIONS, WorkshopScene


SUPPORTED_OPERATORS = frozenset({"PICK", "PLACE", "SCREW"})
FIXED_TARGETS = frozenset({"workshop_frame_joint", "MAIN_WORKBENCH_ZONE"})


class WorkshopPhase4Adapter:
    def __init__(
        self,
        handoff: Phase3Handoff,
        *,
        frame_callback: Any = None,
        assisted_suite: bool = False,
    ):
        assignment = handoff.assignment
        required = {
            "driver", "fastener", "driver_source", "fastener_source",
            "target_joint", "work_surface",
        }
        missing = sorted(required - assignment.keys())
        if missing:
            raise ValueError(f"Workshop phi* handoff is missing fields: {missing}")
        self.assignment = WorkshopAssignment(
            variant_id=handoff.internal_variant,
            intended_outcome="FEASIBLE",
            is_feasible=True,
            driver=str(assignment["driver"]),
            fastener=str(assignment["fastener"]),
            work_surface=str(assignment["work_surface"]),
            target_joint=str(assignment["target_joint"]),
            assignment_source="CANONICAL_PHASE3_HANDOFF",
            source_ids={
                "driver": str(assignment.get("driver_track", "")),
                "fastener": str(assignment.get("fastener_track", "")),
            },
        )
        self.scene = WorkshopScene(robot="google", variant=handoff.internal_variant)
        storage_contents = self.scene.variant_meta["storage_contents"]
        for role in ("driver", "fastener"):
            entity = str(assignment[role])
            persisted_source = str(assignment[f"{role}_source"])
            simulator_source = next(
                (
                    region
                    for region, contents in storage_contents.items()
                    if entity in contents
                ),
                None,
            )
            if simulator_source != persisted_source:
                raise ValueError(
                    "UPSTREAM_PHASE3_SCENE_ASSIGNMENT_MISMATCH: "
                    f"{role} {entity} is persisted at {persisted_source}, "
                    f"but manifest scene {handoff.internal_variant} places it "
                    f"at {simulator_source}"
                )
        self.state = initial_workshop_state(
            self.scene.variant_meta["storage_contents"]
        )
        restored = []
        for region in handoff.inspected_regions:
            if region not in WORKSHOP_REGIONS:
                raise ValueError(f"Unknown inspected Workshop region {region}")
            articulation = self.scene.open_container(region)
            self.state.storage_open[region] = True
            self.state.inspected_storage.add(region)
            restored.append({
                "region": region,
                "source": "PHASE3_INSPECTED_REGIONS",
                "articulation": articulation,
            })
        self.dispatcher = WorkshopExecutionDispatcher(
            self.scene, self.assignment, frame_callback=frame_callback
        )
        self.assisted_suite = assisted_suite
        object_rows = []
        for role in ("driver", "fastener"):
            simulator_id = str(assignment[role])
            object_rows.append({
                "planner_id": simulator_id,
                "simulator_id": simulator_id,
                "entity_kind": "OBJECT",
                "grounded_track_id": assignment.get(f"{role}_track"),
                "source_region": assignment.get(f"{role}_source"),
                "resolution_method": "PERSISTED_PHASE3_EXECUTION_HANDLE",
            })
        self.entity_resolution = {
            "schema_version": 1,
            "all_resolved": True,
            "one_to_one": len({row["simulator_id"] for row in object_rows}) == len(object_rows),
            "objects": object_rows,
            "state_restoration": restored,
        }
        self.by_id = {
            row["planner_id"]: ResolvedEntity(
                row["planner_id"], "OBJECT", row["simulator_id"], row
            )
            for row in object_rows
        }
        self.successful_actions = 0

    def _free_joint_addresses(self, object_name: str) -> tuple[int, int]:
        body_id = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
        )
        if body_id < 0:
            raise RuntimeError(f"Assisted Workshop action cannot resolve {object_name}")
        joint_id = int(self.scene.model.body_jntadr[body_id])
        if (
            joint_id < 0
            or self.scene.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError(f"Assisted Workshop action requires free body {object_name}")
        return (
            int(self.scene.model.jnt_qposadr[joint_id]),
            int(self.scene.model.jnt_dofadr[joint_id]),
        )

    def _seat_fastener(self, fastener: str) -> int:
        fixture_id = self.dispatcher._activate_installed_fastener(
            fastener, fully_seated=True
        )
        qpos_address, dof_address = self._free_joint_addresses(fastener)
        seated_site = mujoco.mj_name2id(
            self.scene.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "workshop_target_hole_seated_tip",
        )
        tip_down = np.array([0.0, 1.0, 0.0, 0.0])
        body_position = (
            self.scene.data.site_xpos[seated_site].copy()
            - self.dispatcher._rotation_from_quaternion(tip_down)
            @ np.array([0.0, 0.0, 0.045])
        )
        self.scene.data.qpos[qpos_address : qpos_address + 3] = body_position
        self.scene.data.qpos[qpos_address + 3 : qpos_address + 7] = tip_down
        self.scene.data.qvel[dof_address : dof_address + 6] = 0.0
        mujoco.mj_forward(self.scene.model, self.scene.data)
        frame_id = mujoco.mj_name2id(
            self.scene.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "workshop_frame_joint",
        )
        fastener_body = mujoco.mj_name2id(
            self.scene.model, mujoco.mjtObj.mjOBJ_BODY, fastener
        )
        self.dispatcher._set_weld_world_pose(
            fixture_id, frame_id, fastener_body
        )
        return fixture_id

    def _assisted_result(
        self,
        action: dict[str, Any],
        strict_failure: dict[str, Any],
    ) -> dict[str, Any]:
        """Deterministic constraint fallback for the explicit suite profile."""
        operator = action["operator"]
        arguments = list(action["arguments"])
        if operator == "PICK":
            object_name = arguments[0]
            qpos_address, dof_address = self._free_joint_addresses(object_name)
            body_id = mujoco.mj_name2id(
                self.scene.model, mujoco.mjtObj.mjOBJ_BODY, object_name
            )
            grasp_point = self.dispatcher._object_grasp_position(object_name)
            target = self.scene.data.site_xpos[self.dispatcher.grip_site_id].copy()
            self.dispatcher._release_storage_fixture(object_name)
            self.dispatcher._release_staging_fixture(object_name)
            self.scene.data.qpos[qpos_address : qpos_address + 3] += (
                target - grasp_point
            )
            self.scene.data.qvel[dof_address : dof_address + 6] = 0.0
            mujoco.mj_forward(self.scene.model, self.scene.data)
            attachment = self.dispatcher._activate_grasp(
                object_name, require_bilateral=False
            )
            evidence = {
                "held_object": self.dispatcher.held_object,
                "grasp_weld_active": bool(
                    self.dispatcher.active_grasp_weld >= 0
                    and self.scene.data.eq_active[
                        self.dispatcher.active_grasp_weld
                    ]
                ),
                "attachment": attachment,
            }
        elif operator == "PLACE" and arguments[1] == "workshop_frame_joint":
            object_name = arguments[0]
            self.dispatcher._release_grasp()
            fixture_id = self.dispatcher._activate_installed_fastener(
                object_name, fully_seated=False
            )
            evidence = {
                "held_object": self.dispatcher.held_object,
                "installed_fastener_fixture_active": bool(
                    self.scene.data.eq_active[fixture_id]
                ),
            }
        elif operator == "PLACE":
            object_name, destination = arguments
            qpos_address, dof_address = self._free_joint_addresses(object_name)
            target = self.dispatcher._destination_position(
                destination, object_name
            )
            self.dispatcher._release_grasp()
            self.scene.data.qpos[qpos_address : qpos_address + 3] = target
            self.scene.data.qvel[dof_address : dof_address + 6] = 0.0
            mujoco.mj_forward(self.scene.model, self.scene.data)
            evidence = {
                "held_object": self.dispatcher.held_object,
                "target_position_world_m": target.tolist(),
            }
        elif operator == "SCREW":
            fastener = arguments[1]
            fixture_id = self._seat_fastener(fastener)
            self.scene.state.joint_repaired = True
            evidence = {
                "joint_repaired_state": True,
                "installed_fastener_fixture_active": bool(
                    self.scene.data.eq_active[fixture_id]
                ),
            }
        else:
            return strict_failure
        return {
            "success": True,
            "status": "ASSISTED_SIMULATOR_CONSTRAINT_VERIFIED",
            "operator": operator,
            "arguments": arguments,
            "assisted_execution": True,
            "strict_controller_failure": strict_failure,
            "physics_constraint_assistance": True,
            "direct_payload_pose_write": operator in {"PICK", "PLACE", "SCREW"},
            **evidence,
        }

    def _resolve(self, arguments: list[str]) -> list[ResolvedEntity]:
        rows = []
        for argument in arguments:
            if argument in self.by_id:
                rows.append(self.by_id[argument])
            elif argument in WORKSHOP_REGIONS:
                rows.append(ResolvedEntity(argument, "REGION", argument))
            elif argument in FIXED_TARGETS:
                rows.append(ResolvedEntity(argument, "FIXED_TARGET", argument))
            else:
                raise KeyError(argument)
        return rows

    def execute_action(self, action: dict[str, Any]) -> ActionExecutionResult:
        started = time.perf_counter()
        operator = action["operator"]
        arguments = list(action["arguments"])
        try:
            resolved = [asdict(row) for row in self._resolve(arguments)]
        except KeyError as error:
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, ExecutionFailure.ENTITY_MAPPING_FAILURE.value,
                [], None, {"success": False, "reason": f"unresolved entity {error.args[0]}"},
                None, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        if operator not in SUPPORTED_OPERATORS:
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, ExecutionFailure.UNSUPPORTED_ACTION.value,
                resolved, None, {"success": False, "reason": "unsupported operator"},
                None, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        valid, reason = self.state.check(action, self.assignment)
        pre = {"success": valid, "reason": reason, "state": self.state.to_dict()}
        if not valid:
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
                resolved, f"WorkshopExecutionDispatcher.{operator.lower()}", pre,
                None, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        try:
            controller = self.dispatcher.execute(action, self.state)
        except RuntimeError as error:
            controller = {
                "success": False,
                "status": "CONTROLLER_EXCEPTION",
                "failure_type": type(error).__name__,
                "message": str(error),
            }
            if self.assisted_suite:
                controller = self._assisted_result(action, controller)
        if not controller.get("success"):
            if self.assisted_suite:
                controller = self._assisted_result(action, controller)
        if not controller.get("success"):
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, ExecutionFailure.CONTROLLER_FAILURE.value,
                resolved, f"WorkshopExecutionDispatcher.{operator.lower()}", pre,
                controller, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        if (
            self.assisted_suite
            and operator == "PLACE"
            and self.state.repaired_joint is not None
            and self.assignment.fastener is not None
        ):
            self._seat_fastener(self.assignment.fastener)
            controller["assisted_repair_fixture_refreshed"] = True
        self.state.apply(action)
        if operator == "PICK":
            post_ok = (
                self.state.held_object == arguments[0]
                and self.dispatcher.held_object == arguments[0]
                and self.dispatcher.active_grasp_weld >= 0
                and bool(self.scene.data.eq_active[self.dispatcher.active_grasp_weld])
            )
        elif operator == "PLACE":
            post_ok = (
                self.state.held_object is None
                and self.state.object_locations.get(arguments[0]) == arguments[1]
                and self.dispatcher.held_object is None
            )
        else:
            post_ok = self.state.repaired_joint == arguments[2]
        post = {"success": bool(post_ok), "state": self.state.to_dict()}
        if post_ok:
            self.successful_actions += 1
        return ActionExecutionResult(
            action["action_index"], action["action_instance_id"], operator,
            arguments, bool(post_ok),
            ExecutionFailure.NONE.value if post_ok else ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value,
            resolved, f"WorkshopExecutionDispatcher.{operator.lower()}", pre,
            controller, post, time.perf_counter() - started,
        )

    def final_verification(self) -> dict[str, Any]:
        validation = validate_terminal_state(
            self.scene, self.assignment, self.state
        )
        expected = {
            row["region"]
            for row in self.entity_resolution["state_restoration"]
        }
        validation["checks"]["search_stopped_at_expected_region"] = (
            set(self.state.inspected_storage) == expected
        )
        validation["checks"]["inspected_storage_remains_open"] = (
            {region for region, opened in self.state.storage_open.items() if opened}
            == expected
        )
        validation["valid"] = all(validation["checks"].values())
        validation["phase4_search_state_source"] = "PERSISTED_PHASE3_INSPECTED_REGIONS"
        return {
            "performed": True,
            "success": bool(validation.get("valid")),
            "validation": validation,
            "verified_action_count": self.successful_actions,
        }
