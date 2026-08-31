"""Kitchen adapter for the common deterministic Phase-4 executor."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

from .kitchen_execution_entities import (
    KitchenExecutionEntityResolver,
    build_phase_b_inventory,
)
from .kitchen_ground_truth_execution import KitchenGroundTruthExecutionDispatcher
from .kitchen_ground_truth_planner import GroundTruthAssignment
from .phase4_execution import (
    ActionExecutionResult,
    ExecutionFailure,
    Phase3Handoff,
    ResolvedEntity,
)
from .scene_loader import KitchenScene


REGION_IDS = frozenset({"countertop", "serving_area", "D1", "D2", "C1", "C2", "B1"})
SUPPORTED_OPERATORS = frozenset({"PICK", "PLACE", "POUR", "STIR"})


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _binding_rows(handoff: Phase3Handoff, group: str) -> list[dict[str, Any]]:
    rows = []
    for binding in handoff.operation_bindings.get(group, []):
        tool = binding.get("tool_id")
        target = binding.get("target_id")
        if not isinstance(tool, str) or not isinstance(target, str):
            raise ValueError(f"Malformed {group} operation binding: {binding!r}")
        rows.append(
            {
                "tool_instance": tool,
                "target_instance": target,
                "tool_kind": "PHI_STAR_BOUND_OBJECT",
                "target_kind": "PHI_STAR_BOUND_OBJECT",
            }
        )
    return rows


def assignment_from_handoff(handoff: Phase3Handoff, scene_name: str) -> GroundTruthAssignment:
    """Translate phi* structurally into the legacy controller assignment shape."""
    assignment = handoff.assignment
    coffee_targets = _as_list(assignment.get("coffee_container"))
    soup_targets = _as_list(assignment.get("soup_container"))
    sources = {
        role: str(assignment[role])
        for role in ("water_source", "coffee_source")
        if role in assignment
    }
    coffee_rows = _binding_rows(handoff, "coffee_stirring")
    soup_rows = _binding_rows(handoff, "soup_serving")
    coffee_by_target = {
        row["target_instance"]: row["tool_instance"] for row in coffee_rows
    }
    soup_by_target = {
        row["target_instance"]: row["tool_instance"] for row in soup_rows
    }
    coffee_by_tool: dict[str, list[str]] = {}
    for row in coffee_rows:
        coffee_by_tool.setdefault(row["tool_instance"], []).append(
            row["target_instance"]
        )
    soup_by_tool = {
        row["tool_instance"]: row["target_instance"] for row in soup_rows
    }
    return GroundTruthAssignment(
        variant_id=handoff.variant,
        scene_name=scene_name,
        intended_outcome="FEASIBLE",
        is_feasible=True,
        failure_reason=None,
        coffee_targets=[{"instance_name": item} for item in coffee_targets],
        soup_targets=[{"instance_name": item} for item in soup_targets],
        sources=sources,
        coffee_assignments=coffee_rows,
        soup_assignments=soup_rows,
        coffee_tools_by_target=coffee_by_target,
        coffee_targets_by_tool=coffee_by_tool,
        soup_utensils_by_target=soup_by_target,
        soup_targets_by_utensil=soup_by_tool,
        unique_coffee_tools=sorted(set(coffee_by_target.values())),
        unique_soup_utensils=sorted(set(soup_by_target.values())),
    )


def _legacy_inventory_inputs(
    handoff: Phase3Handoff,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assignment = handoff.assignment
    legacy_assignment = {
        "coffee_targets": _as_list(assignment.get("coffee_container")),
        "soup_targets": _as_list(assignment.get("soup_container")),
        "source_roles": {
            role: str(assignment[role])
            for role in ("water_source", "coffee_source")
            if role in assignment
        },
        "coffee_stirring": [
            {
                "tool_object_id": row["tool_id"],
                "target_object_id": row["target_id"],
            }
            for row in handoff.operation_bindings.get("coffee_stirring", [])
        ],
        "soup_serving": [
            {
                "tool_object_id": row["tool_id"],
                "target_object_id": row["target_id"],
            }
            for row in handoff.operation_bindings.get("soup_serving", [])
        ],
    }
    legacy_plan = [
        {
            "step": action["action_index"],
            "action": action["operator"],
            "arguments": list(action["arguments"]),
        }
        for action in handoff.actions
    ]
    return legacy_assignment, legacy_plan


class KitchenPhase4Adapter:
    def __init__(
        self,
        handoff: Phase3Handoff,
        *,
        step_callback: Any = None,
    ):
        registry_path = (
            handoff.run_dir / "observed_search" / "phase1" / "object_registry.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        legacy_assignment, legacy_plan = _legacy_inventory_inputs(handoff)
        inventory = build_phase_b_inventory(
            registry, legacy_assignment, legacy_plan
        )
        # The legacy Phase-B inventory projection intentionally keeps only
        # the fields needed by PICK/PLACE.  Phase-C's cupboard-tool
        # orientation, however, consumes the measured opening geometry of
        # the already-grounded target.  Preserve that existing Phase-3
        # evidence verbatim at the execution handoff; do not recompute it or
        # infer a different target here.
        registry_rows = registry.get("objects", {})
        if isinstance(registry_rows, dict):
            registry_rows = registry_rows.values()
        registry_by_id = {row["object_id"]: row for row in registry_rows}
        for row in inventory["objects"]:
            measured = registry_by_id.get(row["generic_object_id"], {})
            if "geometric_properties" in measured:
                row["geometric_properties"] = measured["geometric_properties"]
        self.scene = KitchenScene(
            inventory["scene_name"], include_robot=True, robot="google"
        )
        resolver = KitchenExecutionEntityResolver()
        observed_regions = {
            row["source_context"]["source_container"]
            for row in inventory["objects"]
            if row["source_context"]["source_container"]
        }
        resolution = resolver.resolve(
            inventory,
            resolver.candidates_from_scene(
                self.scene, observed_regions=observed_regions
            ),
        )
        if not resolution["all_resolved"] or not resolution["one_to_one"]:
            raise RuntimeError(
                "Kitchen execution entity resolution failed: "
                f"unresolved={resolution['unresolved_object_ids']}"
            )
        controller_assignment = assignment_from_handoff(
            handoff, inventory["scene_name"]
        )
        self.dispatcher = KitchenGroundTruthExecutionDispatcher(
            self.scene,
            controller_assignment,
            inventory=inventory,
            resolution=resolution,
            step_callback=step_callback,
            assisted_suite=False,
            allow_assisted_pick_recovery=False,
        )
        self.expected_inspected_regions = tuple(handoff.inspected_regions)
        self.inventory = inventory
        self.entity_resolution = resolution
        self.by_id = {
            row["generic_object_id"]: ResolvedEntity(
                planner_id=row["generic_object_id"],
                entity_kind="OBJECT",
                simulator_id=row["physical_backend_body"],
                metadata={
                    "semantic_label": row.get("semantic_label"),
                    "grasp_family": row.get("grasp_family"),
                    "resolution_method": row.get("resolution_method"),
                    "centroid_error_m": row.get("centroid_error_m"),
                },
            )
            for row in resolution["accepted"]
        }
        self.successful_actions: list[dict[str, Any]] = []

    def execute_inspection_open(self, region: str) -> dict[str, Any]:
        """Physically replay one persisted Phase-3 inspection OPEN."""
        started = time.perf_counter()
        if region not in REGION_IDS - {"countertop", "serving_area"}:
            return {
                "region": region,
                "success": False,
                "failure": ExecutionFailure.ENTITY_MAPPING_FAILURE.value,
                "failure_reason": "unknown Kitchen articulated region",
                "wall_duration_s": time.perf_counter() - started,
            }
        controller = self.dispatcher.open_container(region)
        physically_open = region in self.dispatcher.physically_open_containers()
        success = bool(controller.get("success") and physically_open)
        return {
            "region": region,
            "success": success,
            "failure": (
                ExecutionFailure.NONE.value
                if success
                else ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
            ),
            "primitive": "KitchenGroundTruthExecutionDispatcher.open_container",
            "controller_result": controller,
            "post_check": {
                "success": physically_open,
                "physically_open_containers": sorted(
                    self.dispatcher.physically_open_containers()
                ),
            },
            "direct_container_state_write_used": False,
            "wall_duration_s": time.perf_counter() - started,
        }

    def _resolve_arguments(self, action: dict[str, Any]) -> list[ResolvedEntity]:
        resolved = []
        for argument in action["arguments"]:
            if argument in self.by_id:
                resolved.append(self.by_id[argument])
            elif argument in REGION_IDS:
                resolved.append(
                    ResolvedEntity(argument, "REGION", argument, {})
                )
            else:
                raise KeyError(argument)
        return resolved

    def _held_planner_id(self) -> str | None:
        backend = self.dispatcher.phase_b.manipulation.executor.held_object
        if backend is None:
            return None
        for planner_id, entity in self.by_id.items():
            if entity.simulator_id == backend:
                return planner_id
        return f"UNRESOLVED_BACKEND:{backend}"

    def _pre_check(self, action: dict[str, Any]) -> dict[str, Any]:
        operator = action["operator"]
        arguments = action["arguments"]
        held = self._held_planner_id()
        if operator == "PICK":
            valid = held is None
            reason = None if valid else f"hand already holds {held}"
        elif operator in {"PLACE", "POUR", "STIR"}:
            valid = bool(arguments) and held == arguments[0]
            reason = None if valid else f"expected held={arguments[0] if arguments else None}, observed={held}"
        else:
            valid, reason = False, f"unsupported operator {operator}"
        return {"success": valid, "held_object": held, "reason": reason}

    def _post_check(
        self, action: dict[str, Any], controller: dict[str, Any]
    ) -> dict[str, Any]:
        operator = action["operator"]
        first = action["arguments"][0] if action["arguments"] else None
        held = self._held_planner_id()
        if operator == "PICK":
            held_state = self.dispatcher.phase_b._held_state(first)
            valid = held == first and held_state.get("validation_status") == "TRUE"
            evidence = {"held_object": held, "held_state": held_state}
        elif operator == "PLACE":
            valid = held is None and bool(controller.get("success"))
            evidence = {
                "held_object": held,
                "controller_status": controller.get("status"),
                "placement_telemetry": controller.get("telemetry"),
            }
        elif operator == "POUR":
            valid = held == first and bool(controller.get("pour_motion_verified"))
            evidence = {
                "held_object": held,
                "pour_motion_verified": controller.get("pour_motion_verified"),
                "physical_fluid_dynamics_modeled": controller.get(
                    "physical_fluid_dynamics_modeled"
                ),
            }
        elif operator == "STIR":
            valid = held == first and bool(controller.get("stir_motion_verified"))
            evidence = {
                "held_object": held,
                "stir_motion_verified": controller.get("stir_motion_verified"),
                "physical_fluid_dynamics_modeled": controller.get(
                    "physical_fluid_dynamics_modeled"
                ),
            }
        else:
            valid, evidence = False, {}
        return {"success": bool(valid), **evidence}

    def execute_action(self, action: dict[str, Any]) -> ActionExecutionResult:
        started = time.perf_counter()
        operator = action["operator"]
        try:
            resolved = self._resolve_arguments(action)
        except KeyError as error:
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                list(action["arguments"]), False,
                ExecutionFailure.ENTITY_MAPPING_FAILURE.value, [], None,
                {"success": False, "reason": f"unresolved entity {error.args[0]}"},
                None, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        resolved_rows = [asdict(entity) for entity in resolved]
        if operator not in SUPPORTED_OPERATORS:
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                list(action["arguments"]), False,
                ExecutionFailure.UNSUPPORTED_ACTION.value, resolved_rows, None,
                {"success": False, "reason": f"unsupported operator {operator}"},
                None, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        pre = self._pre_check(action)
        if not pre["success"]:
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                list(action["arguments"]), False,
                ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
                resolved_rows, f"KitchenGroundTruthExecutionDispatcher.{operator.lower()}",
                pre, None, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        controller = self.dispatcher.execute_action(action)
        if not controller.get("success"):
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                list(action["arguments"]), False,
                ExecutionFailure.CONTROLLER_FAILURE.value,
                resolved_rows, f"KitchenGroundTruthExecutionDispatcher.{operator.lower()}",
                pre, controller, {"success": False, "performed": False},
                time.perf_counter() - started,
            )
        post = self._post_check(action, controller)
        success = bool(post["success"])
        if success:
            self.successful_actions.append(action)
        return ActionExecutionResult(
            action["action_index"], action["action_instance_id"], operator,
            list(action["arguments"]), success,
            (
                ExecutionFailure.NONE.value
                if success
                else ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
            ),
            resolved_rows, f"KitchenGroundTruthExecutionDispatcher.{operator.lower()}",
            pre, controller, post, time.perf_counter() - started,
        )

    def final_verification(self) -> dict[str, Any]:
        completed = len(self.successful_actions)
        open_regions = self.dispatcher.physically_open_containers()
        inspections_remain_open = set(self.expected_inspected_regions).issubset(
            open_regions
        )
        return {
            "performed": True,
            "success": inspections_remain_open,
            "verified_action_count": completed,
            "verification_basis": "PER_ACTION_SIMULATOR_POSTCONDITIONS",
            "held_object": self._held_planner_id(),
            "inspection_regions_remain_physically_open": inspections_remain_open,
            "physically_open_containers": sorted(open_regions),
        }
