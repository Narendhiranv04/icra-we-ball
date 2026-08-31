"""Workshop adapter for the deterministic Phase-4 executor."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

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
        for region in handoff.inspected_regions:
            if region not in WORKSHOP_REGIONS:
                raise ValueError(f"Unknown inspected Workshop region {region}")
        self.dispatcher = WorkshopExecutionDispatcher(
            self.scene, self.assignment, frame_callback=frame_callback
        )
        self.expected_inspected_regions = tuple(handoff.inspected_regions)
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
            "inspection_regions": list(handoff.inspected_regions),
            "direct_search_state_restoration_used": False,
        }
        self.by_id = {
            row["planner_id"]: ResolvedEntity(
                row["planner_id"], "OBJECT", row["simulator_id"], row
            )
            for row in object_rows
        }
        self.successful_actions = 0

    def execute_inspection_open(self, region: str) -> dict[str, Any]:
        """Replay a Phase-3 inspection by physically opening its storage."""
        started = time.perf_counter()
        action = {
            "action_index": 0,
            "action_instance_id": f"inspection_open_{region.lower()}",
            "operator": "OPEN",
            "arguments": [region],
        }
        valid, reason = self.state.check(action, self.assignment)
        if not valid:
            return {
                "region": region,
                "success": False,
                "failure": ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
                "pre_check": {"success": False, "reason": reason},
                "wall_duration_s": time.perf_counter() - started,
            }
        controller = self.dispatcher.execute(action, self.state)
        articulation = controller.get("articulation", {})
        verified = bool(controller.get("success") and articulation.get("verified"))
        if verified:
            self.state.apply(action)
        return {
            "region": region,
            "success": verified,
            "failure": (
                ExecutionFailure.NONE.value
                if verified
                else ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
            ),
            "primitive": "WorkshopExecutionDispatcher.OPEN",
            "pre_check": {"success": True, "reason": None},
            "controller_result": controller,
            "post_check": {
                "success": verified,
                "articulation_verified": bool(articulation.get("verified")),
                "storage_open": bool(self.state.storage_open.get(region)),
            },
            "direct_container_state_write_used": False,
            "wall_duration_s": time.perf_counter() - started,
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
        if operator == "SCREW":
            # The legacy dispatcher kinematically writes the fastener free
            # joint during its drive loop and then writes joint_repaired.
            # That routine is useful historical visualization code, but it
            # cannot satisfy Phase 4's strict no-task-state-write contract.
            controller = {
                "success": False,
                "status": "STRICT_PHYSICAL_SCREW_UNAVAILABLE",
                "message": (
                    "The current simulator has no force/joint-based fastening "
                    "primitive; the legacy SCREW routine uses direct fastener "
                    "qpos and task-success state writes and is disabled in "
                    "strict Phase 4."
                ),
                "legacy_direct_fastener_qpos_write_blocked": True,
                "legacy_direct_task_success_write_blocked": True,
                "direct_task_state_fallback_used": False,
            }
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, ExecutionFailure.CONTROLLER_FAILURE.value,
                resolved, "WorkshopExecutionDispatcher.screw", pre,
                controller, {"success": False, "performed": False},
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
        if not controller.get("success"):
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, ExecutionFailure.CONTROLLER_FAILURE.value,
                resolved, f"WorkshopExecutionDispatcher.{operator.lower()}", pre,
                controller, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
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
        expected = set(self.expected_inspected_regions)
        validation["checks"]["search_stopped_at_expected_region"] = (
            set(self.state.inspected_storage) == expected
        )
        validation["checks"]["inspected_storage_remains_open"] = (
            {region for region, opened in self.state.storage_open.items() if opened}
            == expected
        )
        validation["valid"] = all(validation["checks"].values())
        validation["phase4_search_state_source"] = (
            "PHYSICAL_REPLAY_OF_PERSISTED_PHASE3_INSPECTED_REGIONS"
        )
        return {
            "performed": True,
            "success": bool(validation.get("valid")),
            "validation": validation,
            "verified_action_count": self.successful_actions,
        }
