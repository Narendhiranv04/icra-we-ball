"""CONTROLLER_DEVELOPMENT_ONLY benchmark Workshop primitive harness.

This runner names simulator bodies directly and therefore is never an
end-to-end TAMP result or paper metric.  It performs no grounding and makes no
simulator-state repairs. It exercises the same calibrated benchmark backend
used by the planner-facing Phase-4 adapter.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from .phase4_execution import audit_strict_telemetry
from .workshop_ground_truth_execution import WorkshopExecutionDispatcher
from .workshop_ground_truth_planner import solve_gt_assignment
from .workshop_ground_truth_state import initial_workshop_state
from .workshop_scene import WorkshopScene


CONTROLLER_CASES = {
    "left_drawer_open": [{"operator": "OPEN", "arguments": ["LEFT_DRAWER"]}],
    "right_drawer_open": [{"operator": "OPEN", "arguments": ["RIGHT_DRAWER"]}],
    "tool_cabinet_open": [{"operator": "OPEN", "arguments": ["TOOL_CABINET"]}],
    "left_drawer_driver_pick": [
        {"operator": "OPEN", "arguments": ["LEFT_DRAWER"]},
        {"operator": "PICK", "arguments": ["workshop_long_phillips_driver", "LEFT_DRAWER"]},
    ],
    "right_drawer_power_pick": [
        {"operator": "OPEN", "arguments": ["RIGHT_DRAWER"]},
        {"operator": "PICK", "arguments": ["workshop_power_driver", "RIGHT_DRAWER"]},
    ],
    "cabinet_screw_pick": [
        {"operator": "OPEN", "arguments": ["TOOL_CABINET"]},
        {"operator": "PICK", "arguments": ["workshop_medium_phillips_screw", "TOOL_CABINET"]},
    ],
    "cabinet_driver_pick": [
        {"operator": "OPEN", "arguments": ["TOOL_CABINET"]},
        {"operator": "PICK", "arguments": ["workshop_long_phillips_driver", "TOOL_CABINET"]},
    ],
}


def _failure_class(message: str) -> str:
    for token in (
        "COLLISION_BLOCKED", "IK_UNREACHABLE", "ACTUATOR_STALL",
        "PREGRASP_POSITION_ERROR", "BILATERAL_CONTACT_NOT_ESTABLISHED",
        "ATTACHMENT_SNAP_TOO_LARGE", "OBJECT_DROPPED", "LIFT_CLEARANCE_FAILED",
    ):
        if token in message:
            return token
    if "Unsafe Workshop" in message:
        return "COLLISION_BLOCKED"
    if "missed its calibrated preclose" in message:
        return "PREGRASP_POSITION_ERROR"
    if "bilateral finger contact" in message:
        return "BILATERAL_CONTACT_NOT_ESTABLISHED"
    return "CONTROLLER_FAILURE"


def run_controller_sequence(
    variant: str, actions: list[dict[str, Any]], output: Path
) -> dict[str, Any]:
    scene = WorkshopScene(robot="google", variant=variant)
    assignment = solve_gt_assignment(variant)
    dispatcher = WorkshopExecutionDispatcher(
        scene, assignment, strict_physical_execution=False
    )
    state = initial_workshop_state(scene.variant_meta["storage_contents"])
    records = []
    for action in actions:
        try:
            result = dispatcher.execute(action, state)
        except Exception as error:
            result = {
                "success": False,
                "status": "CONTROLLER_EXCEPTION",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "failure_class": _failure_class(str(error)),
                "traceback": traceback.format_exc(),
            }
            if action.get("operator") == "PICK" and action.get("arguments"):
                object_name = action["arguments"][0]
                grasp = dispatcher._object_grasp_position(object_name)
                gripper = scene.data.site_xpos[dispatcher.grip_site_id].copy()
                result["failure_pose_diagnostics"] = {
                    "gripper_world_m": gripper.tolist(),
                    "grasp_world_m": grasp.tolist(),
                    "gripper_minus_grasp_m": (gripper - grasp).tolist(),
                }
        records.append({"action": action, "result": result})
        if not result.get("success"):
            break
        state.apply(action)
    payload = {
        "schema_version": 1,
        "label": "CONTROLLER_DEVELOPMENT_ONLY",
        "contributes_to_tamp_success_metrics": False,
        "variant": variant,
        "backend_object_ids_are_controller_test_inputs": True,
        "actions": actions,
        "results": records,
        "success": len(records) == len(actions) and all(
            row["result"].get("success") for row in records
        ),
        "direct_task_state_fallback_used": False,
    }
    audit = audit_strict_telemetry(
        [], [row["result"] for row in records]
    )
    payload["strict_telemetry_verification"] = audit
    payload.update({
        key: audit[key]
        for key in (
            "strict_execution_violation_detected",
            "assisted_task_fixture_used",
            "direct_task_state_write_used",
            "direct_task_state_fallback_used",
            "direct_payload_state_write_used",
            "post_release_dynamics_modified",
        )
    })
    payload["benchmark_execution_mode"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--actions-json", type=Path)
    choice.add_argument("--case", choices=sorted(CONTROLLER_CASES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actions = (
        CONTROLLER_CASES[args.case]
        if args.case else json.loads(args.actions_json.read_text())
    )
    if not isinstance(actions, list):
        raise ValueError("actions JSON must be a list")
    result = run_controller_sequence(args.variant, actions, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
