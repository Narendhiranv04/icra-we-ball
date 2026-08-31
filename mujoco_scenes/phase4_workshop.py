"""Workshop adapter for the deterministic Phase-4 executor."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any

import mujoco

from .phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase4EntityMappingError,
    Phase3Handoff,
    ResolvedEntity,
)
from .workshop_ground_truth_execution import (
    WorkshopExecutionDispatcher,
    validate_terminal_state,
)
from .phase4_workshop_entities import (
    WorkshopEntityResolutionError,
    resolve_workshop_entities,
    workshop_body_world_geometry_aabb_center,
)
from .workshop_ground_truth_planner import WorkshopAssignment
from .workshop_ground_truth_state import initial_workshop_state
from .workshop_scene import WORKSHOP_REGIONS, WorkshopScene


SUPPORTED_OPERATORS = frozenset({"PICK", "PLACE", "SCREW"})
FIXED_TARGETS = frozenset({"workshop_frame_joint"})
CONTEXT_SURFACES = frozenset({"MAIN_WORKBENCH_ZONE"})


def resolve_workshop_entities_for_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Translate resolver failures into the public Phase-4 classification."""
    try:
        return resolve_workshop_entities(*args, **kwargs)
    except WorkshopEntityResolutionError as error:
        raise Phase4EntityMappingError(str(error)) from error


def strict_workshop_place_block(
    action: dict[str, Any], planner_fastener: str
) -> dict[str, Any] | None:
    """Block legacy PLACE implementations that use task fixtures."""
    if action.get("operator") != "PLACE" or len(action.get("arguments", [])) != 2:
        return None
    obj, destination = action["arguments"]
    if destination == "workshop_frame_joint":
        if obj == planner_fastener:
            return None
        return {
            "status": "INVALID_IMMUTABLE_WORKSHOP_PLAN",
            "immutable_plan_precondition_mismatch": True,
            "legacy_alignment_fixture_blocked": True,
            "legacy_installed_fastener_fixture_blocked": True,
            "no_legacy_insertion_invoked": True,
        }
    return None


class WorkshopPhase4Adapter:
    def __init__(
        self,
        handoff: Phase3Handoff,
        *,
        frame_callback: Any = None,
    ):
        assignment = handoff.assignment
        required = {"driver", "fastener"}
        missing = sorted(required - assignment.keys())
        if missing:
            raise ValueError(f"Workshop phi* handoff is missing fields: {missing}")
        planner_driver = str(assignment["driver"])
        planner_fastener = str(assignment["fastener"])
        screw_actions = [
            action for action in handoff.actions if action["operator"] == "SCREW"
        ]
        if len(screw_actions) != 1 or len(screw_actions[0]["arguments"]) != 3:
            raise ValueError("Workshop final plan must contain one typed SCREW action")
        screw_driver, screw_fastener, target_joint = screw_actions[0]["arguments"]
        if (screw_driver, screw_fastener) != (planner_driver, planner_fastener):
            raise ValueError(
                "Workshop final plan objects differ from canonical phi* assignment"
            )
        driver_places = [
            action["arguments"][1]
            for action in handoff.actions
            if action["operator"] == "PLACE"
            and len(action["arguments"]) == 2
            and action["arguments"][0] == planner_driver
        ]
        if len(driver_places) != 1:
            raise ValueError("Workshop final plan has no unique driver destination")
        work_surface = str(driver_places[0])
        self.scene = WorkshopScene(robot="google", variant=handoff.internal_variant)
        observed_graph_path = handoff.artifacts.get("observed_graph")
        if observed_graph_path is None:
            raise ValueError("Workshop handoff has no final observed G_O artifact")
        observed_graph = json.loads(observed_graph_path.read_text(encoding="utf-8"))
        # Identity association uses a separate scene brought to the persisted
        # inspection configuration. The real execution scene above remains a
        # fresh, closed initial scene. Storage membership is currently backend
        # ground truth, never functional re-grounding or role selection.
        association_scene = WorkshopScene(
            robot="google", variant=handoff.internal_variant
        )
        source_regions = {
            action["arguments"][1]
            for action in handoff.actions
            if action["operator"] == "PICK" and len(action["arguments"]) >= 2
        }
        for source_region in sorted(source_regions):
            association_scene.open_container(source_region)
        simulator_candidates = []
        for source_region, bodies in association_scene.variant_meta[
            "storage_contents"
        ].items():
            for simulator_id in bodies:
                body_id = mujoco.mj_name2id(
                    association_scene.model, mujoco.mjtObj.mjOBJ_BODY, simulator_id
                )
                if body_id < 0:
                    raise ValueError(f"Workshop scene is missing body {simulator_id}")
                centroid, centroid_definition = (
                    workshop_body_world_geometry_aabb_center(
                        association_scene.model, association_scene.data, body_id
                    )
                )
                simulator_candidates.append({
                    "simulator_id": simulator_id,
                    "source_region": source_region,
                    "centroid_world_m": centroid,
                    "centroid_definition": centroid_definition,
                })
        resolution = resolve_workshop_entities_for_execution(
            (planner_driver, planner_fastener), observed_graph,
            handoff.actions, simulator_candidates,
        )
        resolution_by_id = {row["planner_id"]: row for row in resolution["objects"]}
        backend_driver = resolution_by_id[planner_driver]["simulator_id"]
        backend_fastener = resolution_by_id[planner_fastener]["simulator_id"]
        self.assignment = WorkshopAssignment(
            variant_id=handoff.internal_variant,
            intended_outcome="FEASIBLE",
            is_feasible=True,
            driver=planner_driver,
            fastener=planner_fastener,
            work_surface=work_surface,
            target_joint=str(target_joint),
            assignment_source="CANONICAL_PHASE3_HANDOFF",
            source_ids={
                "driver": planner_driver,
                "fastener": planner_fastener,
            },
        )
        self.controller_assignment = WorkshopAssignment(
            variant_id=handoff.internal_variant,
            intended_outcome="FEASIBLE",
            is_feasible=True,
            driver=backend_driver,
            fastener=backend_fastener,
            work_surface=work_surface,
            target_joint=str(target_joint),
            assignment_source="PHASE4_EXECUTION_ENTITY_RESOLUTION",
            source_ids={
                "driver": planner_driver,
                "fastener": planner_fastener,
            },
        )
        self.state = initial_workshop_state({})
        self.controller_state = initial_workshop_state(
            self.scene.variant_meta["storage_contents"]
        )
        for row in resolution["objects"]:
            self.state.object_locations[row["planner_id"]] = row["source_region"]
        for region in handoff.inspected_regions:
            if region not in WORKSHOP_REGIONS:
                raise ValueError(f"Unknown inspected Workshop region {region}")
        self.dispatcher = WorkshopExecutionDispatcher(
            self.scene, self.controller_assignment, frame_callback=frame_callback,
            strict_physical_execution=True,
        )
        self.expected_inspected_regions = tuple(handoff.inspected_regions)
        self.planner_fastener = planner_fastener
        self.entity_resolution = {
            **resolution,
            "inspection_regions": list(handoff.inspected_regions),
            "direct_search_state_restoration_used": False,
            "association_authority": (
                "SIMULATION_BACKEND_GROUND_TRUTH_ASSOCIATION_ONLY"
            ),
        }
        self.by_id = {
            row["planner_id"]: ResolvedEntity(
                row["planner_id"], "OBJECT", row["simulator_id"], row
            )
            for row in resolution["objects"]
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
        controller = self.dispatcher.execute(action, self.controller_state)
        articulation = controller.get("articulation", {})
        verified = bool(controller.get("success") and articulation.get("verified"))
        if verified:
            self.state.apply(action)
            self.controller_state.apply(action)
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
            elif argument in CONTEXT_SURFACES:
                rows.append(ResolvedEntity(argument, "REGION", argument, {
                    "planner_context_surface": True,
                }))
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
        blocked_place = strict_workshop_place_block(
            action, self.planner_fastener
        )
        if blocked_place is not None:
            controller = {
                "success": False,
                **blocked_place,
                "message": (
                    "Legacy Workshop placement uses a target-alignment, "
                    "staging, or installed-state fixture and is disabled in "
                    "strict Phase 4."
                ),
            }
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                arguments, False, (
                    ExecutionFailure.PRECONDITION_STATE_FAILURE.value
                    if blocked_place.get("immutable_plan_precondition_mismatch")
                    else ExecutionFailure.CONTROLLER_FAILURE.value
                ),
                resolved, None, pre, controller,
                {"success": False, "performed": False},
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
        physical_action = {
            **action,
            "arguments": [
                self.by_id[argument].simulator_id
                if argument in self.by_id else argument
                for argument in arguments
            ],
        }
        try:
            controller = self.dispatcher.execute(
                physical_action, self.controller_state
            )
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
        self.controller_state.apply(physical_action)
        if operator == "PICK":
            post_ok = (
                self.state.held_object == arguments[0]
                and self.controller_state.held_object
                == self.by_id[arguments[0]].simulator_id
                and self.dispatcher.held_object
                == self.by_id[arguments[0]].simulator_id
                and self.dispatcher.active_grasp_weld >= 0
                and bool(self.scene.data.eq_active[self.dispatcher.active_grasp_weld])
            )
        elif operator == "PLACE":
            post_ok = (
                self.state.held_object is None
                and self.state.object_locations.get(arguments[0]) == arguments[1]
                and self.controller_state.held_object is None
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
            self.scene, self.controller_assignment, self.controller_state
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
