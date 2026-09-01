"""Living-Room bridge over the existing verified mobile execution loop."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from .living_room_mobile_execution import run_mobile_execution
from .phase4_execution import (
    ExecutionFailure,
    emit_phase4_progress,
    normalize_planner_failure_code,
    Phase4LiveViewer,
    Phase4EntityMappingError,
    Phase3Handoff,
    audit_strict_telemetry,
)


def normalize_living_room_action_result(
    action: dict[str, Any],
    controller: dict[str, Any] | None,
    unresolved: list[str],
    held_postcondition: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine mapping, controller, and independent physical post-checks."""
    mapping_ok = not unresolved
    controller_ok = bool(
        controller is not None and controller.get("result") == "SUCCESS"
    )
    post_ok = controller_ok
    if action["operator"] == "PICK":
        post_ok = bool(
            controller_ok
            and held_postcondition is not None
            and held_postcondition.get("validation_status") == "TRUE"
        )
    success = bool(mapping_ok and controller_ok and post_ok)
    if not mapping_ok:
        failure = ExecutionFailure.ENTITY_MAPPING_FAILURE.value
    elif not controller_ok:
        failure = (
            ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
            if controller and controller.get("failure") == "POSTCONDITION_FAILED"
            else ExecutionFailure.CONTROLLER_FAILURE.value
        )
    elif not post_ok:
        failure = ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
    else:
        failure = ExecutionFailure.NONE.value
    message = str(unresolved) if unresolved else str(
        controller or "controller result unavailable"
    )
    code = "ACCESS_BLOCKED" if (
        action["operator"] == "PICK" and controller_ok and not post_ok
    ) else (controller or {}).get("failure_code")
    return {
        "success": success,
        "failure": failure,
        "failure_code": normalize_planner_failure_code(
            code,
            message,
            infrastructure_failure=failure,
            operator=action["operator"],
        ),
        "post_check_success": post_ok,
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_living_room_action_arguments(
    arguments: list[str],
    object_map: dict[str, dict[str, Any]],
    region_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve every frozen planner argument and expose all misses."""
    resolved = []
    unresolved = []
    for argument in arguments:
        if argument in object_map:
            row = object_map[argument]
            resolved.append({
                "planner_id": argument,
                "entity_kind": "OBJECT",
                "simulator_id": row["backend_body"],
                "metadata": row,
            })
        elif argument in region_map:
            row = region_map[argument]
            resolved.append({
                "planner_id": argument,
                "entity_kind": "REGION",
                "simulator_id": row["backend_support_geom"],
                "metadata": row,
            })
        else:
            unresolved.append(argument)
    return resolved, unresolved


def validate_living_room_plan_ids(
    actions: list[dict[str, Any]],
    payload_registry: dict[str, Any],
    region_registry: dict[str, Any],
) -> None:
    """Fail before controller startup if any immutable argument is unknown."""
    known = set(payload_registry.get("objects", {})) | set(
        region_registry.get("regions", {})
    )
    for action in actions:
        unresolved = [
            argument for argument in action["arguments"] if argument not in known
        ]
        if unresolved:
            raise Phase4EntityMappingError(
                "Living-Room action "
                f"{action['action_instance_id']} has unresolved arguments: "
                f"{unresolved}"
            )


def execute_living_room_handoff(
    handoff: Phase3Handoff,
    *,
    output_dir: Path,
    max_actions: int | None = None,
    record_video: Path | None = None,
    viewer: bool = False,
) -> dict[str, Any]:
    """Execute the exact plan using the domain's existing per-action loop.

    The Living-Room stack already performs entity resolution, navigation
    refinement, held-state checks, support/contact verification, and final
    goal revalidation in one cohesive loop.  This bridge preserves that loop
    and normalizes its records into the common Phase-4 artifact schema.
    """
    if handoff.inspected_regions:
        raise ValueError(
            "Living-Room handoff contains inspection regions but has no "
            "domain physical OPEN primitive"
        )
    phase1_dir = handoff.run_dir / "observed_grounding"
    phase2_dir = handoff.run_dir / "action_sequence"
    validate_living_room_plan_ids(
        list(handoff.actions),
        _read(phase1_dir / "payload_registry.json"),
        _read(phase1_dir / "region_registry.json"),
    )
    native_dir = output_dir / "domain_execution"
    live_viewers: list[Phase4LiveViewer] = []

    def create_viewer(model: Any, data: Any) -> Any:
        live_viewer = Phase4LiveViewer(model, data)
        live_viewers.append(live_viewer)
        return live_viewer.sync

    def report_progress(
        event: str,
        index: int,
        total: int,
        operator: str,
        arguments: list[str],
        row: dict[str, Any] | None,
    ) -> None:
        if event == "start":
            emit_phase4_progress(
                "TASK", index, total, operator, arguments
            )
            return
        success = bool(row and row.get("result") == "SUCCESS")
        failure = (
            ExecutionFailure.NONE.value if success
            else ExecutionFailure.CONTROLLER_FAILURE.value
        )
        failure_code = normalize_planner_failure_code(
            None,
            str((row or {}).get("failure") or row),
            infrastructure_failure=failure,
            operator=operator,
        )
        emit_phase4_progress(
            "TASK", index, total, operator, arguments,
            success=success,
            controller_status=(row or {}).get("failure"),
            failure_code=failure_code,
        )

    try:
        native = run_mobile_execution(
            phase1_dir,
            phase2_dir,
            native_dir,
            variant=handoff.internal_variant,
            execute=True,
            max_task_actions=max_actions,
            assisted_suite=False,
            reset_payloads_from_observation=False,
            step_callback_factory=create_viewer if viewer else None,
            progress_callback=report_progress,
        )
    finally:
        for live_viewer in live_viewers:
            live_viewer.close()
    visual_output = {
        "enabled": bool(record_video is not None or viewer),
        "viewer_enabled": bool(viewer),
        "video_path": str(record_video) if record_video else None,
        "video_created": False,
    }
    if record_video is not None:
        native_video = native_dir / "execution_timeline.mp4"
        if native_video.is_file():
            record_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native_video, record_video)
            visual_output["video_created"] = bool(record_video.stat().st_size)
    resolution = _read(native_dir / "execution_entity_resolution.json")
    physical = _read(native_dir / "physical_execution.json")
    held_checks = _read(native_dir / "held_object_validation.json").get(
        "checks", []
    )
    after_pick_checks = iter(
        row for row in held_checks if row.get("phase") == "AFTER_PICK"
    )
    object_map = {
        row["generic_object_id"]: row for row in resolution["objects"]
    }
    region_map = {
        row["generic_region_id"]: row for row in resolution["regions"]
    }
    task_rows = [
        row for row in physical.get("actions", []) if row.get("operator") != "MOVE"
    ]
    selected_actions = list(handoff.actions)
    if max_actions is not None:
        selected_actions = selected_actions[:max_actions]
    action_results = []
    for index, action in enumerate(selected_actions):
        controller = task_rows[index] if index < len(task_rows) else None
        resolved, unresolved = resolve_living_room_action_arguments(
            list(action["arguments"]), object_map, region_map
        )
        mapping_success = not unresolved
        held_postcondition = (
            next(after_pick_checks, None)
            if action["operator"] == "PICK"
            else None
        )
        normalized = normalize_living_room_action_result(
            action, controller, unresolved, held_postcondition
        )
        action_results.append({
            "action_index": action["action_index"],
            "action_instance_id": action["action_instance_id"],
            "operator": action["operator"],
            "arguments": list(action["arguments"]),
            "success": normalized["success"],
            "failure": normalized["failure"],
            "failure_code": normalized["failure_code"],
            "resolved_arguments": resolved,
            "primitive": "living_room_mobile_execution.run_mobile_execution",
            "pre_check": {
                "success": controller is not None and mapping_success,
                "entity_mapping_success": mapping_success,
                "arguments_requested": len(action["arguments"]),
                "arguments_resolved": len(resolved),
                "unresolved_arguments": unresolved,
                "verification_basis": "DOMAIN_LOOP_STANCE_AND_HELD_STATE_CHECKS",
            },
            "controller_result": controller,
            "post_check": {
                "success": normalized["post_check_success"],
                "held_state": held_postcondition,
                "physical_verification": (
                    controller.get("physical_verification") if controller else None
                ),
            },
        })
    complete = len(selected_actions) == len(handoff.actions)
    final = physical.get("final_physical_goal_validation", {})
    all_actions = (
        len(action_results) == len(selected_actions)
        and all(row["success"] for row in action_results)
    )
    telemetry_rows = list(action_results)
    telemetry_rows.append({
        "physical_execution": {
            "initial_payload_qpos_reset_used": bool(
                physical.get("initial_payload_qpos_reset_used")
            )
        }
    })
    strict_audit = audit_strict_telemetry([], telemetry_rows)
    partial_smoke = not complete
    partial_smoke_success = bool(partial_smoke and all_actions)
    success = bool(
        complete
        and all_actions
        and final.get("all_phase2_goals_physically_satisfied", False)
    )
    failure = next(
        (row["failure"] for row in action_results if not row["success"]),
        ExecutionFailure.NONE.value,
    )
    failure_code = next(
        (row.get("failure_code") for row in action_results if not row["success"]),
        None,
    )
    if failure != ExecutionFailure.NONE.value:
        failure_stage = (
            "ENTITY_RESOLUTION"
            if failure == ExecutionFailure.ENTITY_MAPPING_FAILURE.value
            else "TASK_ACTION"
        )
    elif complete and not final.get("all_phase2_goals_physically_satisfied", False):
        failure = "FINAL_VERIFICATION_FAILURE"
        failure_stage = "FINAL_VERIFICATION"
    else:
        failure_stage = None
    return {
        "schema_version": 2,
        "phase": "PHASE_4_EXECUTION",
        "domain": handoff.domain,
        "variant": handoff.variant,
        "internal_variant": handoff.internal_variant,
        "functional_specification_source": handoff.source,
        "specification_sha256": handoff.specification_sha256,
        "phase3_artifacts": {
            key: str(value) for key, value in handoff.artifacts.items()
        },
        "phase3_artifact_sha256": dict(handoff.artifact_sha256),
        "final_action_sequence": list(handoff.actions),
        "entity_resolution": resolution,
        "inspection_execution": {
            "regions": [],
            "actions_requested": 0,
            "actions_completed": 0,
            "results": [],
            "success": True,
        },
        "task_plan_execution": {
            "actions": selected_actions,
            "results": action_results,
        },
        "actions_requested": len(selected_actions),
        "actions_completed": sum(row["success"] for row in action_results),
        "full_sequence_requested": complete,
        "partial_smoke": partial_smoke,
        "partial_smoke_success": partial_smoke_success,
        "action_results": action_results,
        "final_verification": (
            final if complete else {"performed": False, "reason": "PARTIAL_SEQUENCE"}
        ),
        "domain_execution_summary": native,
        "visual_output": visual_output,
        "failure": failure,
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "execution_mode": "P4_BENCH",
        "strict_execution": False,
        "strict_telemetry_verification": strict_audit,
        "strict_execution_violation_detected": strict_audit[
            "strict_execution_violation_detected"
        ],
        "direct_task_state_write_used": strict_audit[
            "direct_task_state_write_used"
        ],
        "direct_payload_state_write_used": strict_audit[
            "direct_payload_state_write_used"
        ],
        "assisted_task_fixture_used": strict_audit[
            "assisted_task_fixture_used"
        ],
        "post_release_dynamics_modified": strict_audit[
            "post_release_dynamics_modified"
        ],
        "direct_task_state_fallback_used": strict_audit[
            "direct_task_state_fallback_used"
        ],
        "initial_payload_qpos_reset_used": bool(
            physical.get("initial_payload_qpos_reset_used")
        ),
        "wall_duration_s": native.get("wall_time_s"),
        "success": success,
    }
