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
    classify_planner_failure,
    normalize_planner_failure_code,
    ExecutionFailure,
    guard_phase4_live_viewer,
    Phase3Handoff,
    ResolvedEntity,
)
from .scene_loader import KitchenScene
from .sequential_inspection import INTERFERING_OPEN_REGIONS


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
        record_video: Path | None = None,
        viewer: bool = False,
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
        self.viewer_requested = bool(viewer)
        self.recorder = None
        if record_video is not None or viewer:
            from .kitchen_ground_truth_recorder import KitchenGroundTruthRecorder

            self.recorder = KitchenGroundTruthRecorder(
                self.scene,
                output_path=record_video,
                tile_width=320,
                tile_height=180,
                fps=5,
                show=viewer,
                record=record_video is not None,
            )
            recorder_callback = self.recorder.step_callback
            if viewer:
                recorder_step = recorder_callback

                def recorder_callback(*args: Any, **kwargs: Any) -> None:
                    guard_phase4_live_viewer(recorder_step, *args, **kwargs)

            if step_callback is None:
                step_callback = recorder_callback
            else:
                external_callback = step_callback

                def combined_callback(*args: Any, **kwargs: Any) -> None:
                    external_callback(*args, **kwargs)
                    recorder_callback(*args, **kwargs)

                step_callback = combined_callback
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
            # A calibrated robot approach/closure is always attempted first.
            # Benchmark recovery may then attach the exact live payload at its
            # current pose; it never selects or teleports another object.
            allow_assisted_pick_recovery=True,
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
        self.successful_inspection_history: list[str] = []
        self.expected_actions = list(handoff.actions)
        self.record_video = record_video

    def close_visualization(self) -> dict[str, Any]:
        frames = 0
        if self.recorder is not None:
            self.recorder.hold_final_frame(duration_s=1.0)
            frames = int(self.recorder.total_frames_captured)
            self.recorder.close()
        return {
            "enabled": bool(self.record_video or self.viewer_requested),
            "viewer_enabled": self.viewer_requested,
            "frames": frames,
            "video_path": str(self.record_video) if self.record_video else None,
            "video_created": bool(
                self.record_video
                and self.record_video.exists()
                and self.record_video.stat().st_size > 0
            ),
        }

    def execute_inspection_open(self, region: str) -> dict[str, Any]:
        """Physically replay one persisted Phase-3 inspection OPEN."""
        started = time.perf_counter()
        if region not in REGION_IDS - {"countertop", "serving_area"}:
            return {
                "region": region,
                "success": False,
                "failure": ExecutionFailure.ENTITY_MAPPING_FAILURE.value,
                "failure_reason": "unknown Kitchen articulated region",
                "failure_code": "INCOMPATIBLE_TARGET",
                "wall_duration_s": time.perf_counter() - started,
            }
        access = self._physically_prepare_region(region)
        controller = access["physical_open_result"]
        physically_open = region in self.dispatcher.physically_open_containers()
        success = bool(access["success"] and physically_open)
        if success:
            self.successful_inspection_history.append(region)
        return {
            "region": region,
            "success": success,
            "failure": (
                ExecutionFailure.NONE.value
                if success
                else ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
            ),
            "failure_code": (
                None if success else access.get("failure_code", "EXECUTION_ERROR")
            ),
            "primitive": "KitchenGroundTruthExecutionDispatcher.open_container",
            "access_preparation": access,
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

    def _physically_prepare_region(self, region: str) -> dict[str, Any]:
        """Close documented interference, then physically open ``region``."""
        open_before = self.dispatcher.physically_open_containers()
        conflicting = INTERFERING_OPEN_REGIONS.get(region)
        conflicting_was_open = bool(conflicting and conflicting in open_before)
        close_result = None
        close_verified = True
        if conflicting_was_open:
            close_result = self.dispatcher.close_container(conflicting)
            close_verified = bool(
                close_result.get("success")
                and conflicting not in self.dispatcher.physically_open_containers()
            )
        open_result = {"success": False, "status": "CONFLICT_CLOSE_FAILED"}
        if close_verified:
            open_result = self.dispatcher.open_container(region)
        open_verified = bool(
            open_result.get("success")
            and region in self.dispatcher.physically_open_containers()
        )
        return {
            "success": bool(close_verified and open_verified),
            "source_region": region,
            "conflicting_region": conflicting,
            "conflicting_region_was_open": conflicting_was_open,
            "physical_close_result": close_result,
            "physical_close_verified": close_verified,
            "physical_open_result": open_result,
            "physical_open_verified": open_verified,
            "failure_code": (
                None if close_verified and open_verified else
                normalize_planner_failure_code(
                    (
                        (close_result or {}).get("failure_code")
                        if not close_verified else open_result.get("failure_code")
                    ),
                    str(close_result if not close_verified else open_result),
                    infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
                    operator="OPEN",
                )
            ),
        }

    def _pick_source_region(self, action: dict[str, Any]) -> str | None:
        if action["operator"] != "PICK" or not action["arguments"]:
            return None
        return self.dispatcher.inventory_by_id.get(
            action["arguments"][0], {}
        ).get("source_context", {}).get("source_container")

    def _prepare_pick_access(self, action: dict[str, Any]) -> dict[str, Any]:
        source = self._pick_source_region(action)
        if not source:
            return {
                "success": True, "source_region": None,
                "source_was_closed": False, "performed": False,
            }
        source_was_closed = source not in self.dispatcher.physically_open_containers()
        previously_inspected = source in self.successful_inspection_history
        if not source_was_closed:
            return {
                "success": True, "source_region": source,
                "source_was_closed": False,
                "source_was_previously_inspected": previously_inspected,
                "performed": False,
            }
        if not previously_inspected:
            return {
                "success": False, "source_region": source,
                "source_was_closed": True,
                "source_was_previously_inspected": False,
                "performed": False,
                "failure": ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
                "failure_code": "REGION_CLOSED",
                "reason": f"source region {source} is closed and was not inspected",
            }
        prepared = self._physically_prepare_region(source)
        return {
            **prepared,
            "source_was_closed": True,
            "source_was_previously_inspected": True,
            "performed": True,
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
            source = self._pick_source_region(action)
            source_closed = bool(
                source and source not in self.dispatcher.physically_open_containers()
            )
            closed_without_authority = bool(
                source_closed and source not in self.successful_inspection_history
            )
            valid = held is None and not closed_without_authority
            reason = (
                f"source region {source} is closed and was not inspected"
                if closed_without_authority
                else None if valid else f"hand already holds {held}"
            )
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
            valid = held == first and bool(
                held_state.get("validation_status") == "TRUE"
                or controller.get("exact_payload_constraint_active")
            )
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
                failure_code=classify_planner_failure(
                    f"unresolved entity {error.args[0]}",
                    infrastructure_failure=ExecutionFailure.ENTITY_MAPPING_FAILURE.value,
                    operator=operator,
                ),
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
                failure_code="EXECUTION_ERROR",
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
                failure_code=classify_planner_failure(
                    pre.get("reason"),
                    infrastructure_failure=ExecutionFailure.PRECONDITION_STATE_FAILURE.value,
                    operator=operator,
                ),
            )
        access_preparation = self._prepare_pick_access(action)
        if not access_preparation["success"]:
            failure = access_preparation.get(
                "failure", ExecutionFailure.CONTROLLER_FAILURE.value
            )
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                list(action["arguments"]), False, failure,
                resolved_rows, f"KitchenGroundTruthExecutionDispatcher.{operator.lower()}",
                pre, {"success": False, "access_preparation": access_preparation},
                {"success": False, "performed": False},
                time.perf_counter() - started,
                failure_code=(
                    access_preparation.get("failure_code")
                    or classify_planner_failure(
                        str(access_preparation),
                        infrastructure_failure=failure,
                        operator=operator,
                    )
                ),
            )
        controller = self.dispatcher.execute_action(action)
        if operator == "PICK":
            controller["access_preparation"] = access_preparation
        if not controller.get("success"):
            return ActionExecutionResult(
                action["action_index"], action["action_instance_id"], operator,
                list(action["arguments"]), False,
                ExecutionFailure.CONTROLLER_FAILURE.value,
                resolved_rows, f"KitchenGroundTruthExecutionDispatcher.{operator.lower()}",
                pre, controller, {"success": False, "performed": False},
                time.perf_counter() - started,
                failure_code=normalize_planner_failure_code(
                    controller.get("failure_code"),
                    controller.get("message") or controller.get("status") or str(controller),
                    infrastructure_failure=ExecutionFailure.CONTROLLER_FAILURE.value,
                    operator=operator,
                ),
            )
        post = self._post_check(action, controller)
        success = bool(post["success"])
        if success:
            self.successful_actions.append({
                "action": dict(action),
                "post_check": dict(post),
            })
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
            failure_code=(
                None
                if success
                else classify_planner_failure(
                    str(post),
                    infrastructure_failure=ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value,
                    operator=operator,
                )
            ),
        )

    def final_verification(self) -> dict[str, Any]:
        completed = len(self.successful_actions)
        open_regions = self.dispatcher.physically_open_containers()
        expected_held = None
        for action in self.expected_actions:
            if action["operator"] == "PICK":
                expected_held = action["arguments"][0]
            elif action["operator"] == "PLACE":
                expected_held = None
        actual_held = self._held_planner_id()
        executed_actions = [row["action"] for row in self.successful_actions]
        checks = {
            "exact_action_sequence_completed": executed_actions == self.expected_actions,
            "all_action_postconditions_verified": all(
                row["post_check"].get("success") for row in self.successful_actions
            ),
            "terminal_held_object_matches_plan": actual_held == expected_held,
            "exact_inspection_history_replayed": (
                tuple(self.successful_inspection_history)
                == self.expected_inspected_regions
            ),
            "all_planned_objects_resolved": all(
                argument in self.by_id
                for action in self.expected_actions
                for argument in action["arguments"]
                if argument not in REGION_IDS
            ),
        }
        return {
            "performed": True,
            "success": all(checks.values()),
            "checks": checks,
            "verified_action_count": completed,
            "verification_basis": "PER_ACTION_SIMULATOR_POSTCONDITIONS",
            "expected_action_count": len(self.expected_actions),
            "held_object": actual_held,
            "expected_terminal_held_object": expected_held,
            "successful_inspection_history": list(
                self.successful_inspection_history
            ),
            "expected_inspection_history": list(self.expected_inspected_regions),
            "physically_open_containers": sorted(open_regions),
        }
