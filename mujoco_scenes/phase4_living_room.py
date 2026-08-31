"""Living-Room bridge over the existing verified mobile execution loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .living_room_mobile_execution import run_mobile_execution
from .phase4_execution import ExecutionFailure, Phase3Handoff


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def execute_living_room_handoff(
    handoff: Phase3Handoff,
    *,
    output_dir: Path,
    max_actions: int | None = None,
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
    native_dir = output_dir / "domain_execution"
    native = run_mobile_execution(
        phase1_dir,
        phase2_dir,
        native_dir,
        variant=handoff.internal_variant,
        execute=True,
        max_task_actions=max_actions,
        assisted_suite=False,
        reset_payloads_from_observation=False,
    )
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
        resolved = []
        for argument in action["arguments"]:
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
        controller_success = bool(
            controller is not None and controller.get("result") == "SUCCESS"
        )
        held_postcondition = (
            next(after_pick_checks, None)
            if action["operator"] == "PICK"
            else None
        )
        failure = ExecutionFailure.NONE.value
        if not controller_success:
            failure = (
                ExecutionFailure.POSTCONDITION_VERIFICATION_FAILURE.value
                if controller and controller.get("failure") == "POSTCONDITION_FAILED"
                else ExecutionFailure.CONTROLLER_FAILURE.value
            )
        action_results.append({
            "action_index": action["action_index"],
            "action_instance_id": action["action_instance_id"],
            "operator": action["operator"],
            "arguments": list(action["arguments"]),
            "success": controller_success,
            "failure": failure,
            "resolved_arguments": resolved,
            "primitive": "living_room_mobile_execution.run_mobile_execution",
            "pre_check": {
                "success": controller is not None,
                "verification_basis": "DOMAIN_LOOP_STANCE_AND_HELD_STATE_CHECKS",
            },
            "controller_result": controller,
            "post_check": {
                "success": bool(
                    controller_success
                    and (
                        held_postcondition is None
                        or held_postcondition.get("validation_status") == "TRUE"
                    )
                ),
                "held_state": held_postcondition,
                "physical_verification": (
                    controller.get("physical_verification") if controller else None
                ),
            },
        })
    complete = max_actions is None
    final = physical.get("final_physical_goal_validation", {})
    all_actions = (
        len(action_results) == len(selected_actions)
        and all(row["success"] for row in action_results)
    )
    success = bool(
        all_actions
        and (
            not complete
            or final.get("all_phase2_goals_physically_satisfied", False)
        )
    )
    failure = next(
        (row["failure"] for row in action_results if not row["success"]),
        ExecutionFailure.NONE.value,
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
        "action_results": action_results,
        "final_verification": (
            final if complete else {"performed": False, "reason": "PARTIAL_SEQUENCE"}
        ),
        "domain_execution_summary": native,
        "failure": failure,
        "failure_stage": failure_stage,
        "strict_execution": True,
        "direct_task_state_fallback_used": False,
        "initial_payload_qpos_reset_used": bool(
            physical.get("initial_payload_qpos_reset_used")
        ),
        "wall_duration_s": native.get("wall_time_s"),
        "success": success,
    }
