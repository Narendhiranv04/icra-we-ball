"""Main CLI entry point for Ground-Truth Oracle kitchen execution and video recording.

Usage examples:
  # One variant with live visualization and recording
  MUJOCO_GL=glfw python -m mujoco_scenes.run_kitchen_ground_truth_execution \\
    --variant F2_HIDDEN_SOUP_BOWL --show --record

  # All variants with live visualization and recording
  MUJOCO_GL=glfw python -m mujoco_scenes.run_kitchen_ground_truth_execution \\
    --variant all --show --record

  # All variants headless recording
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m mujoco_scenes.run_kitchen_ground_truth_execution \\
    --variant all --record

  # Dry-run sequence generation and preflight verification
  python -m mujoco_scenes.run_kitchen_ground_truth_execution \\
    --variant F2_HIDDEN_SOUP_BOWL --dry-run
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback
import time
from typing import Any

import yaml

from .kitchen_ground_truth_execution import (
    KitchenGroundTruthExecutionDispatcher,
    build_oracle_inventory_and_resolution,
)
from .kitchen_ground_truth_planner import (
    generate_ground_truth_plan,
    solve_ground_truth_assignment,
)
from .kitchen_ground_truth_recorder import (
    FIVE_PROJECT_CAMERAS,
    KitchenGroundTruthRecorder,
    create_camera_manifest,
)
from .kitchen_ground_truth_state import (
    OracleWorldState,
    initialize_oracle_world_state,
    run_symbolic_preflight,
)
from .final_paper_variant_labels import resolve_variant_name
from .scene_loader import (
    KITCHEN_FEASIBILITY_VARIANTS,
    KitchenScene,
    load_all_configs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs/kitchen_ground_truth_execution"


def load_variants_config() -> dict[str, Any]:
    """Load feasibility benchmark variants configuration."""
    if not KITCHEN_FEASIBILITY_VARIANTS.exists():
        raise FileNotFoundError(f"Missing variant config: {KITCHEN_FEASIBILITY_VARIANTS}")
    data = yaml.safe_load(KITCHEN_FEASIBILITY_VARIANTS.read_text(encoding="utf-8")) or {}
    return data


def discover_variant_names() -> list[str]:
    """Return all variant identifiers defined in the YAML config."""
    config = load_variants_config()
    return list(config.get("variants", {}).keys())


def print_variant_list() -> None:
    """Print formatted summary of all discovered variants."""
    config = load_variants_config()
    variants = config.get("variants", {})
    print("\n" + "=" * 80)
    print(f"DISCOVERED KITCHEN FEASIBILITY VARIANTS ({len(variants)} total)")
    print("=" * 80)
    print(f"{'VARIANT ID':<36} {'OUTCOME':<12} {'SCENE NAME'}")
    print("-" * 80)
    for variant_id, info in variants.items():
        outcome = info.get("intended_outcome", "UNKNOWN")
        scene = info.get("scene_name", "")
        desc = info.get("description", "")
        print(f"{variant_id:<36} {outcome:<12} {scene}")
        if desc:
            print(f"  └─ {desc}")
    print("=" * 80 + "\n")


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WxH' string into (width, height) tuple."""
    try:
        parts = res_str.lower().split("x")
        if len(parts) != 2:
            raise ValueError()
        w, h = int(parts[0]), int(parts[1])
        return w, h
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{res_str}'. Expected format like '640x360' or '426x240'")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def count_true_flag(value: Any, flag: str) -> int:
    """Count true audit flags recursively in an execution result."""
    if isinstance(value, dict):
        return int(value.get(flag) is True) + sum(
            count_true_flag(child, flag) for child in value.values()
        )
    if isinstance(value, list):
        return sum(count_true_flag(child, flag) for child in value)
    return 0


def validate_feasible_final_state(
    assignment,
    final_state: OracleWorldState,
    dispatcher: KitchenGroundTruthExecutionDispatcher | None = None,
) -> dict[str, Any]:
    """Validate final task state for feasible variants."""
    checks = {}

    # Coffee checks
    coffee_target_ids = [t["instance_name"] for t in assignment.coffee_targets]
    water_source = assignment.sources.get("water_source")
    coffee_source = assignment.sources.get("coffee_source")

    # Water and coffee poured into all targets
    poured_all_water = bool(
        water_source and all((water_source, c) in final_state.poured_relations for c in coffee_target_ids)
    )
    poured_all_coffee = bool(
        coffee_source and all((coffee_source, c) in final_state.poured_relations for c in coffee_target_ids)
    )
    checks["coffee_water_poured"] = poured_all_water
    checks["coffee_grounds_poured"] = poured_all_coffee

    # Coffee stirring
    stirred_all = True
    for c in coffee_target_ids:
        tool = assignment.coffee_tools_by_target.get(c)
        if not tool or (tool, c) not in final_state.stirred_relations:
            stirred_all = False
            break
    checks["all_coffees_stirred"] = stirred_all

    # Coffee served
    coffees_served = all(c in final_state.served_objects for c in coffee_target_ids)
    checks["all_coffees_served"] = coffees_served

    # Soup checks
    soup_target_ids = [t["instance_name"] for t in assignment.soup_targets]
    soups_served = all(s in final_state.served_objects for s in soup_target_ids)
    checks["all_soups_served"] = soups_served

    soup_utensils_paired = True
    for s in soup_target_ids:
        utensil = assignment.soup_utensils_by_target.get(s)
        if not utensil or (utensil, s) not in final_state.utensil_bowl_pairs:
            soup_utensils_paired = False
            break
    checks["all_soup_utensils_paired"] = soup_utensils_paired

    # General hand empty check
    checks["hand_empty_at_end"] = final_state.held_object is None

    # Overall success
    overall_valid = all(checks.values())
    return {
        "valid": overall_valid,
        "checks": checks,
        "physical_fluid_dynamics_modeled": False,
        "pour_motion_verified": poured_all_water and poured_all_coffee,
        "stir_motion_verified": stirred_all,
    }


def run_variant_ground_truth(
    variant_id: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    show: bool = False,
    record: bool = False,
    dry_run: bool = False,
    fps: int = 20,
    camera_resolution: tuple[int, int] = (640, 360),
    no_overlay: bool = False,
    inspection_order: list[str] | None = None,
    assisted_suite: bool = False,
    strict_robot_execution: bool = False,
    speed: float = 1.0,
) -> dict[str, Any]:
    """Execute ground-truth workflow for a single variant."""
    variant_id = resolve_variant_name("kitchen", variant_id)
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    config = load_variants_config()
    variants = config.get("variants", {})
    if variant_id not in variants:
        raise ValueError(f"Variant '{variant_id}' not found in {list(variants.keys())}")

    variant_info = variants[variant_id]
    scene_name = variant_info["scene_name"]
    intended_outcome = variant_info.get("intended_outcome", "FEASIBLE")
    infeasible_reason = variant_info.get("intended_failure_reason")
    description = variant_info.get("description", "")

    variant_out_dir = output_root / variant_id
    variant_out_dir.mkdir(parents=True, exist_ok=True)
    video_path = variant_out_dir / f"{variant_id}_5cam.mp4"

    print("\n" + "=" * 70)
    print(f"RUNNING GROUND_TRUTH_ORACLE: {variant_id}")
    print(f"Scene: {scene_name} | Intended: {intended_outcome}")
    if description:
        print(f"Description: {description}")
    print("=" * 70)

    # 1. Instantiate KitchenScene
    scene = KitchenScene(scene_name, include_robot=True, robot="google")

    # 2. Solve Privileged Assignment
    assignment = solve_ground_truth_assignment(scene, variant_id, intended_outcome)
    assignment_dict = assignment.to_dict()
    write_json(variant_out_dir / "gt_assignment.json", assignment_dict)

    print(f"Assignment solved: Feasible={assignment.is_feasible}, CoffeeTools={len(assignment.unique_coffee_tools)}, SoupUtensils={len(assignment.unique_soup_utensils)}")
    if assignment.failure_reason:
        print(f"Oracle failure condition: {assignment.failure_reason}")

    # 3. Initialize Oracle World State & Generate Plan
    initial_state = initialize_oracle_world_state(scene._object_instance_records)
    plan = generate_ground_truth_plan(assignment, initial_state, inspection_order)
    write_json(variant_out_dir / "gt_plan.json", {"actions": plan, "total_actions": len(plan)})

    print(f"Generated GT plan with {len(plan)} deterministic actions.")

    # 4. Symbolic Preflight Pass
    preflight = run_symbolic_preflight(initial_state, plan)
    write_json(variant_out_dir / "symbolic_preflight.json", preflight)

    if not preflight["success"]:
        print(f"\n[ERROR] Symbolic Preflight FAILED on action #{preflight['failed_step_index']}: {preflight['failure_reason']}")
        if intended_outcome == "FEASIBLE":
            summary = {
                "variant_id": variant_id,
                "scene_name": scene_name,
                "execution_mode": "GROUND_TRUTH_ORACLE",
                "intended_outcome": intended_outcome,
                "execution_outcome": "PREFLIGHT_FAILED",
                "failure_reason": preflight["failure_reason"],
                "total_actions": len(plan),
                "actions_completed": 0,
                "success": False,
            }
            write_json(variant_out_dir / "summary.json", summary)
            return summary
    else:
        print("Symbolic preflight PASSED: All action preconditions verified.")

    # 5. Handle Dry-Run
    if dry_run:
        print("\n[DRY RUN] Skipping physical execution.")
        summary = {
            "variant_id": variant_id,
            "scene_name": scene_name,
            "execution_mode": "GROUND_TRUTH_ORACLE",
            "intended_outcome": intended_outcome,
            "execution_outcome": "DRY_RUN_PASSED",
            "is_feasible": assignment.is_feasible,
            "total_actions": len(plan),
            "actions_completed": 0,
            "preflight_status": preflight["preflight_status"],
            "success": True,
        }
        write_json(variant_out_dir / "summary.json", summary)
        return summary

    # 6. Physical Execution & Recording Setup
    tile_w, tile_h = camera_resolution
    recorder = KitchenGroundTruthRecorder(
        scene,
        output_path=video_path if record else None,
        tile_width=tile_w,
        tile_height=tile_h,
        fps=fps,
        show=show,
        record=record,
        no_overlay=no_overlay,
    )

    recorder.telemetry.variant_id = variant_id
    recorder.telemetry.scene_name = scene_name
    recorder.telemetry.intended_outcome = intended_outcome
    recorder.telemetry.total_actions = len(plan)
    recorder.telemetry.execution_status = "RUNNING"
    if not assignment.is_feasible:
        recorder.telemetry.infeasible_reason = assignment.failure_reason or infeasible_reason

    dispatcher = KitchenGroundTruthExecutionDispatcher(
        scene,
        assignment,
        step_callback=recorder.step_callback,
        assisted_suite=assisted_suite,
        allow_assisted_pick_recovery=not strict_robot_execution,
    )
    dispatcher.phase_b.manipulation.executor.arm_command_speed *= float(speed)

    # Initial frame capture
    recorder.capture_frame(force=True)

    # 7. Execute Actions
    execution_trace = []
    live_oracle_state = initial_state.clone()
    physical_execution_success = True
    execution_aborted = False
    failure_detail = None

    for action_idx, action in enumerate(plan, 1):
        if recorder.aborted_by_user:
            print("\n[ABORT] User requested termination (q/ESC).")
            execution_aborted = True
            physical_execution_success = False
            failure_detail = "ABORTED_BY_USER"
            break

        operator = action["operator"]
        arguments = action["arguments"]
        reason = action.get("reason", "")

        # Update telemetry
        recorder.telemetry.current_action_index = action_idx
        recorder.telemetry.current_operator = operator
        recorder.telemetry.current_arguments = arguments
        recorder.telemetry.current_reason = reason
        recorder.telemetry.held_object = live_oracle_state.held_object

        # Determine high-level phase
        if operator in {"OPEN", "CLOSE"}:
            recorder.telemetry.high_level_phase = "Phase A: ACCESS"
        elif operator == "PLACE" and arguments[1] == "countertop":
            recorder.telemetry.high_level_phase = "Phase B: RELOCATE"
        elif operator == "POUR":
            recorder.telemetry.high_level_phase = "Phase C: POUR"
        elif operator == "STIR":
            recorder.telemetry.high_level_phase = "Phase D: STIR"
        elif operator in {"SERVE_COFFEE", "SERVE_SOUP", "PLACE_SERVING_UTENSIL"}:
            recorder.telemetry.high_level_phase = "Phase E: SERVE"

        print(f"[{action_idx}/{len(plan)}] Executing {operator}({', '.join(arguments)}) ...")

        state_before = live_oracle_state.to_dict()
        action_result = dispatcher.execute_action(action)
        action_success = bool(action_result.get("success", False))

        trace_entry = {
            "action_index": action_idx,
            "action_instance_id": action.get("action_instance_id"),
            "operator": operator,
            "arguments": arguments,
            "reason": reason,
            "state_before": state_before,
            "physical_result": action_result,
            "success": action_success,
        }

        if action_success:
            live_oracle_state.apply_action(action)
            trace_entry["state_after"] = live_oracle_state.to_dict()
            recorder.telemetry.held_object = live_oracle_state.held_object
            execution_trace.append(trace_entry)
        else:
            trace_entry["state_after"] = state_before
            execution_trace.append(trace_entry)
            physical_execution_success = False
            failure_detail = action_result.get("status", "ACTION_FAILED")
            print(f"  └─ Physical action FAILED: {failure_detail}")
            diagnostic = action_result.get("message")
            if diagnostic:
                print(f"     Diagnostic: {diagnostic}")
            if action_result.get("primary_placement_failure"):
                print(
                    "     Primary placement failure: "
                    f"{action_result['primary_placement_failure']}"
                )
            break

    # 8. Post-Execution & Task Validation
    if not assignment.is_feasible:
        # Infeasible variant handling
        recorder.telemetry.execution_status = "INFEASIBLE_CONFIRMED"
        recorder.telemetry.high_level_phase = "TERMINATED"
        execution_outcome = "INFEASIBLE_CONFIRMED"
        task_validation = {
            "valid": True,
            "intended_outcome": "INFEASIBLE",
            "confirmed_reason": assignment.failure_reason,
        }
        overall_success = physical_execution_success and (assignment.failure_reason is not None)
    else:
        # Feasible variant validation
        task_validation = validate_feasible_final_state(assignment, live_oracle_state, dispatcher)
        if physical_execution_success and task_validation["valid"]:
            recorder.telemetry.execution_status = "SUCCESS"
            recorder.telemetry.high_level_phase = "COMPLETE"
            execution_outcome = "SUCCESS"
            overall_success = True
        else:
            recorder.telemetry.execution_status = "FAILED"
            execution_outcome = "FAILED"
            overall_success = False

    if execution_aborted:
        execution_outcome = "ABORTED_BY_USER"
        recorder.telemetry.execution_status = "ABORTED"

    # Hold final frame in video
    recorder.hold_final_frame(duration_s=2.0)
    recorder.close()

    # 9. Save Artifacts
    write_json(variant_out_dir / "execution_trace.json", {"actions": execution_trace})
    write_json(variant_out_dir / "final_state.json", {
        "oracle_world_state": live_oracle_state.to_dict(),
        "task_validation": task_validation,
    })

    camera_manifest = create_camera_manifest(
        video_path,
        recorder.mosaic_width,
        recorder.mosaic_height,
        recorder.tile_width,
        recorder.tile_height,
        recorder.fps,
        recorder.total_frames_captured,
        float(scene.data.time),
    )
    write_json(variant_out_dir / "camera_manifest.json", camera_manifest)

    assisted_action_count = sum(
        count_true_flag(entry.get("physical_result", {}), "assisted_execution")
        for entry in execution_trace
    )
    direct_payload_pose_write_count = sum(
        count_true_flag(entry.get("physical_result", {}), "direct_payload_pose_write")
        for entry in execution_trace
    )
    direct_object_qpos_write_count = sum(
        count_true_flag(entry.get("physical_result", {}), "direct_object_qpos_write")
        for entry in execution_trace
    )
    if strict_robot_execution and (
        assisted_action_count
        or direct_payload_pose_write_count
        or direct_object_qpos_write_count
    ):
        overall_success = False
        physical_execution_success = False
        execution_outcome = "STRICT_EXECUTION_AUDIT_FAILED"
        failure_detail = "ASSISTED_OR_DIRECT_PAYLOAD_MOTION_DETECTED"

    summary = {
        "variant_id": variant_id,
        "scene_name": scene_name,
        "execution_mode": "GROUND_TRUTH_ORACLE",
        "execution_profile": (
            "ASSISTED_DETERMINISTIC_DEMONSTRATION"
            if assisted_suite else (
                "STRICT_ROBOT_PHYSICAL_PRIMITIVES"
                if strict_robot_execution else "PHYSICAL_PRIMITIVES_WITH_ASSISTED_RECOVERY"
            )
        ),
        "intended_outcome": intended_outcome,
        "execution_outcome": execution_outcome,
        "failure_reason": assignment.failure_reason if not assignment.is_feasible else failure_detail,
        "is_feasible": assignment.is_feasible,
        "gt_assignment_valid": True,
        "symbolic_preflight_passed": preflight["success"],
        "physical_execution_success": physical_execution_success,
        "task_validation_passed": task_validation.get("valid", False),
        "total_actions": len(plan),
        "actions_completed": len(execution_trace),
        "sim_duration_s": float(scene.data.time),
        "video_path": str(video_path) if record else None,
        "video_recorded": bool(record and video_path.exists() and video_path.stat().st_size > 0),
        "assisted_action_count": assisted_action_count,
        "direct_payload_pose_write_count": direct_payload_pose_write_count,
        "direct_object_qpos_write_count": direct_object_qpos_write_count,
        "success": overall_success,
    }
    write_json(variant_out_dir / "summary.json", summary)

    print("-" * 70)
    print(f"Result for {variant_id}: {execution_outcome} (Success={overall_success})")
    if record and video_path.exists():
        print(f"Video saved: {video_path} ({video_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print("-" * 70 + "\n")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Ground-Truth Oracle kitchen execution and five-camera mosaic recording."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="F0_ALL_VISIBLE",
        help="Variant name (e.g. 'F2_HIDDEN_SOUP_BOWL') or 'all' to run every variant.",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List all discovered variants with intended outcomes and descriptions.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live five-camera mosaic during execution.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record synchronized five-camera mosaic MP4 video.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory to save run logs and videos (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frame rate for video capture and recording (default: 20).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solve assignment, generate plan, and run symbolic preflight without physics.",
    )
    parser.add_argument(
        "--only-feasible",
        action="store_true",
        help="Filter to only feasible variants when using --variant all.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        default=True,
        help="Continue to next variant if one fails in --variant all (default: True).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable camera labels and HUD on output frames.",
    )
    parser.add_argument(
        "--camera-resolution",
        type=parse_resolution,
        default=(640, 360),
        help="Per-camera tile resolution WxH (default: 640x360).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Execution speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--assisted-suite",
        action="store_true",
        help=(
            "Use deterministic articulation, grasp-weld, placement, POUR and "
            "STIR proxies for fast all-variant GT closure. Artifacts explicitly "
            "label direct payload pose writes and do not claim fluid dynamics."
        ),
    )
    parser.add_argument(
        "--strict-robot-execution",
        action="store_true",
        help=(
            "Require the robot physical-primitives path. Disable missed-grasp "
            "pose-write recovery and fail the run if any assisted action or "
            "direct payload pose write is detected."
        ),
    )

    args = parser.parse_args()

    if args.assisted_suite and args.strict_robot_execution:
        parser.error("--assisted-suite and --strict-robot-execution are mutually exclusive")

    if args.list_variants:
        print_variant_list()
        return 0

    all_variants = discover_variant_names()
    if not all_variants:
        print("[ERROR] No variants found in configuration.", file=sys.stderr)
        return 1

    if args.variant.lower() == "all":
        variants_to_run = all_variants
        if args.only_feasible:
            config = load_variants_config().get("variants", {})
            variants_to_run = [
                v for v in variants_to_run
                if config.get(v, {}).get("intended_outcome") == "FEASIBLE"
            ]
    else:
        resolved_variant = resolve_variant_name("kitchen", args.variant)
        if resolved_variant not in all_variants:
            print(f"[ERROR] Variant '{args.variant}' not recognized. Available: {', '.join(all_variants)}", file=sys.stderr)
            return 1
        variants_to_run = [resolved_variant]

    print(f"\nDiscovered {len(all_variants)} total variants. Selected {len(variants_to_run)} to run.")

    suite_results = []
    overall_suite_success = True

    for variant_name in variants_to_run:
        try:
            summary = run_variant_ground_truth(
                variant_name,
                output_root=args.output_root,
                show=args.show,
                record=args.record,
                dry_run=args.dry_run,
                fps=args.fps,
                camera_resolution=args.camera_resolution,
                no_overlay=args.no_overlay,
                assisted_suite=args.assisted_suite,
                strict_robot_execution=args.strict_robot_execution,
                speed=args.speed,
            )
            suite_results.append(summary)
            if not summary.get("success", False):
                overall_suite_success = False
                if not args.continue_on_failure:
                    print("[ABORT] Stopping suite execution after failure.")
                    break
        except Exception as error:
            print(f"\n[EXCEPTION] Error executing variant {variant_name}: {error}", file=sys.stderr)
            traceback.print_exc()
            overall_suite_success = False
            suite_results.append({
                "variant_id": variant_name,
                "execution_outcome": "EXCEPTION",
                "error": str(error),
                "success": False,
            })
            if not args.continue_on_failure:
                break

    if len(variants_to_run) > 1 or args.variant.lower() == "all":
        suite_summary = {
            "execution_mode": "GROUND_TRUTH_ORACLE",
            "execution_profile": (
                "ASSISTED_DETERMINISTIC_DEMONSTRATION"
                if args.assisted_suite else (
                    "STRICT_ROBOT_PHYSICAL_PRIMITIVES"
                    if args.strict_robot_execution
                    else "PHYSICAL_PRIMITIVES_WITH_ASSISTED_RECOVERY"
                )
            ),
            "total_variants_executed": len(suite_results),
            "passed_variants": sum(1 for r in suite_results if r.get("success", False)),
            "failed_variants": sum(1 for r in suite_results if not r.get("success", False)),
            "overall_success": overall_suite_success,
            "results": suite_results,
        }
        write_json(args.output_root / "suite_summary.json", suite_summary)

        print("\n" + "=" * 90)
        print(f"SUITE SUMMARY: {suite_summary['passed_variants']}/{suite_summary['total_variants_executed']} PASSED")
        print("=" * 90)
        print(f"{'VARIANT':<32} {'INTENDED':<12} {'OUTCOME':<22} {'ACTIONS':<10} {'SUCCESS'}")
        print("-" * 90)
        for r in suite_results:
            vid = r.get("variant_id", "")
            intended = r.get("intended_outcome", "")
            outcome = r.get("execution_outcome", "")
            acts = f"{r.get('actions_completed', 0)}/{r.get('total_actions', 0)}"
            succ = "PASS" if r.get("success", False) else "FAIL"
            print(f"{vid:<32} {intended:<12} {outcome:<22} {acts:<10} {succ}")
        print("=" * 90 + "\n")

    return 0 if overall_suite_success else 1


if __name__ == "__main__":
    sys.exit(main())
