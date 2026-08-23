"""Run the redesigned 10-variant Workshop GT execution suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .workshop_execution_handoff import (
    load_frozen_production_assignment,
    validate_frozen_handoff_suite,
)
from .workshop_ground_truth_execution import WorkshopExecutionDispatcher, validate_terminal_state
from .workshop_ground_truth_planner import (
    generate_gt_plan,
    load_variant_specs,
    solve_gt_assignment,
)
from .workshop_ground_truth_recorder import WorkshopRecorder
from .workshop_ground_truth_state import initial_workshop_state, symbolic_preflight
from .workshop_scene import WorkshopScene, privileged_validate_variant_feasibility


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "workshop_ground_truth_execution_assisted_suite"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_suite_summary(
    output_root: Path,
    results: list[dict[str, Any]],
    *,
    assignment_source: str,
) -> dict[str, Any]:
    """Write a suite summary from completed per-variant result records."""
    handoff = validate_frozen_handoff_suite()
    write_json(output_root / "functional_grounding_handoff_validation.json", handoff)
    suite = {
        "schema_version": 1,
        "execution_profile": "CONTACT_GATED_ROBOT_ACTUATED_GT_EXECUTION",
        "assignment_source": assignment_source,
        "total_variants": len(results),
        "passed_variants": sum(result.get("success", False) for result in results),
        "feasible_passed": sum(result.get("success", False) and result.get("intended_outcome") == "FEASIBLE" for result in results),
        "infeasible_passed": sum(result.get("success", False) and result.get("intended_outcome") == "INFEASIBLE" for result in results),
        "total_actions": sum(result.get("total_actions", 0) for result in results),
        "actions_completed": sum(result.get("actions_completed", 0) for result in results),
        "grounding_execution_contract_exact": handoff["passed"],
        "only_foundation_model_pending": False,
        "pending_scopes": [
            "Run and freeze production YOLO-World-L five-view grounding for the redesigned 10-variant family.",
            "Replace ManualWorkshopFMContract with a live VLM/FM response.",
        ],
        "success": all(result.get("success", False) for result in results) and handoff["passed"],
        "results": results,
    }
    write_json(output_root / "suite_summary.json", suite)
    return suite


def run_variant(
    variant_id: str,
    *,
    output_root: Path = DEFAULT_OUTPUT,
    record: bool = False,
    dry_run: bool = False,
    assignment_source: str = "oracle",
    resolution: tuple[int, int] = (640, 360),
    fps: int = 20,
    show: bool = False,
) -> dict[str, Any]:
    specs = load_variant_specs()
    spec = specs[variant_id]
    assignment = (
        solve_gt_assignment(variant_id)
        if assignment_source == "oracle"
        else load_frozen_production_assignment(variant_id)
    )
    scene = WorkshopScene(robot="google", variant=variant_id)
    oracle_audit = privileged_validate_variant_feasibility(scene)
    state = initial_workshop_state(spec["storage_contents"])
    plan = generate_gt_plan(assignment)
    preflight = symbolic_preflight(state, plan, assignment)
    variant_dir = output_root / variant_id
    write_json(variant_dir / "assignment.json", assignment.to_dict())
    write_json(variant_dir / "action_plan.json", {"total_actions": len(plan), "actions": plan})
    write_json(variant_dir / "symbolic_preflight.json", preflight)
    write_json(variant_dir / "oracle_feasibility_audit.json", oracle_audit)
    if not preflight["success"]:
        summary = {"variant_id": variant_id, "success": False, "outcome": "PREFLIGHT_FAILED", "failure_reason": preflight["failure_reason"]}
        write_json(variant_dir / "summary.json", summary)
        return summary
    if dry_run:
        summary = {
            "variant_id": variant_id,
            "intended_outcome": assignment.intended_outcome,
            "assignment_source": assignment.assignment_source,
            "outcome": "DRY_RUN_PASSED",
            "total_actions": len(plan),
            "actions_completed": 0,
            "symbolic_preflight_passed": True,
            "success": True,
        }
        write_json(variant_dir / "summary.json", summary)
        return summary

    video = variant_dir / f"{variant_id}_5cam.mp4"
    recorder = WorkshopRecorder(
        scene, video if record else None, width=resolution[0], height=resolution[1],
        fps=fps, show=show,
    )
    recorder.telemetry["total"] = len(plan)
    dispatcher = WorkshopExecutionDispatcher(scene, assignment, frame_callback=recorder.capture)
    live = state.clone()
    trace = []
    physical_ok = True
    recorder.capture(True)
    for action in plan:
        recorder.telemetry.update({
            "index": action["action_index"],
            "operator": action["operator"],
            "arguments": action["arguments"],
        })
        before = live.to_dict()
        frame_start = recorder.frames
        try:
            physical = dispatcher.execute(action, live)
        except Exception as error:
            recorder.close()
            trace.append({
                "action": action,
                "state_before": before,
                "frame_start": frame_start,
                "frame_end": recorder.frames,
                "exception": repr(error),
                "traceback": traceback.format_exc(),
                "state_after": live.to_dict(),
            })
            write_json(variant_dir / "execution_trace.json", {"actions": trace})
            raise RuntimeError(
                f"Workshop action {action['action_index']} "
                f"{action['operator']} {action.get('arguments', [])} failed: {error}"
            ) from error
        if physical.get("success"):
            live.apply(action)
        else:
            physical_ok = False
        trace.append({
            "action": action,
            "state_before": before,
            "frame_start": frame_start,
            "frame_end": recorder.frames,
            "physical_result": physical,
            "state_after": live.to_dict(),
        })
        if not physical_ok:
            break
    validation = validate_terminal_state(scene, assignment, live)
    success = physical_ok and validation["valid"]
    outcome = "SUCCESS" if assignment.is_feasible and success else (
        "INFEASIBLE_CONFIRMED" if not assignment.is_feasible and success else "FAILED"
    )
    recorder.telemetry["status"] = outcome
    recorder.close()
    write_json(variant_dir / "execution_trace.json", {"actions": trace})
    write_json(variant_dir / "terminal_state.json", {"state": live.to_dict(), "validation": validation})
    write_json(variant_dir / "camera_manifest.json", {
        "layout": "3x2_five_views_plus_status",
        "cameras": ["left", "right", "top", "front", "close"],
        "resolution_per_view": list(resolution),
        "fps": fps,
        "frames": recorder.frames,
        "video": str(video) if record else None,
    })
    direct_payload_pose_write_count = sum(
        bool(row.get("physical_result", {}).get("direct_payload_pose_write", False))
        for row in trace
    )
    robot_actuated_action_count = sum(
        bool(row.get("physical_result", {}).get("robot_actuated_motion", False))
        for row in trace
    )
    if direct_payload_pose_write_count:
        success = False
        outcome = "DIRECT_PAYLOAD_POSE_WRITE_AUDIT_FAILED"
    summary = {
        "variant_id": variant_id,
        "intended_outcome": assignment.intended_outcome,
        "assignment_source": assignment.assignment_source,
        "execution_profile": "CONTACT_GATED_ROBOT_ACTUATED_GT_EXECUTION",
        "autonomous_manipulation_claimed": False,
        "outcome": outcome,
        "rejection_reason": assignment.rejection_reason,
        "total_actions": len(plan),
        "actions_completed": len(trace),
        "symbolic_preflight_passed": True,
        "physical_execution_success": physical_ok,
        "terminal_validation_passed": validation["valid"],
        "robot_actuated_action_count": robot_actuated_action_count,
        "direct_payload_pose_write_count": direct_payload_pose_write_count,
        "video_recorded": bool(record and video.exists() and video.stat().st_size > 0),
        "video_path": str(video) if record else None,
        "success": success,
    }
    write_json(variant_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="all", help="Variant ID or 'all'")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-feasible",
        action="store_true",
        help="Run only variants whose intended outcome is FEASIBLE.",
    )
    parser.add_argument("--assignment-source", choices=("oracle", "frozen-production"), default="oracle")
    parser.add_argument("--resolution", default="640x360")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument(
        "--summarize-existing", action="store_true",
        help="Rebuild the suite summary from existing per-variant summaries.",
    )
    args = parser.parse_args()
    specs = load_variant_specs()
    if args.list_variants:
        for name, spec in specs.items():
            print(f"{name:<36} {spec['intended_outcome']}")
        return 0
    variants = list(specs) if args.variant.lower() == "all" else [args.variant]
    if any(name not in specs for name in variants):
        parser.error("Unknown Workshop variant")
    if args.only_feasible:
        variants = [
            name for name in variants
            if specs[name]["intended_outcome"] == "FEASIBLE"
        ]
    try:
        width, height = (int(value) for value in args.resolution.lower().split("x"))
    except Exception:
        parser.error("--resolution must be WxH")
    if args.summarize_existing:
        results = []
        for variant in variants:
            summary_path = args.output_root / variant / "summary.json"
            if not summary_path.exists():
                parser.error(f"Missing existing summary: {summary_path}")
            results.append(json.loads(summary_path.read_text(encoding="utf-8")))
        suite = write_suite_summary(
            args.output_root, results,
            assignment_source=args.assignment_source,
        )
        print(
            f"SUITE {suite['passed_variants']}/{suite['total_variants']} | "
            f"actions {suite['actions_completed']}/{suite['total_actions']}"
        )
        return 0 if suite["success"] else 1
    results = []
    for variant in variants:
        print(f"[{len(results)+1}/{len(variants)}] {variant}", flush=True)
        try:
            result = run_variant(
                variant,
                output_root=args.output_root,
                record=args.record,
                dry_run=args.dry_run,
                assignment_source=args.assignment_source,
                resolution=(width, height),
                fps=args.fps,
                show=args.show,
            )
        except Exception as error:
            result = {"variant_id": variant, "outcome": "EXCEPTION", "error": repr(error), "success": False}
        results.append(result)
        print(f"  -> {result['outcome']} ({result.get('actions_completed', 0)}/{result.get('total_actions', 0)})", flush=True)
    suite = write_suite_summary(
        args.output_root, results, assignment_source=args.assignment_source
    )
    handoff = validate_frozen_handoff_suite()
    print(f"SUITE {suite['passed_variants']}/{suite['total_variants']} | actions {suite['actions_completed']}/{suite['total_actions']} | handoff {handoff['exact_matches']}/{handoff['total_variants']}")
    return 0 if suite["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
