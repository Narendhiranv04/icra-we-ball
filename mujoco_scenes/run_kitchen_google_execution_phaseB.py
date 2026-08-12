"""Kitchen Phase-B entry point: inventory and execution-entity resolution.

Physical PICK/CARRY/PLACE modes are added behind this same entry point; the
inventory mode intentionally runs first and writes no simulator name into the
frozen Phase-1/Phase-2 inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .kitchen_execution_entities import (
    KitchenExecutionEntityResolver,
    build_phase_b_inventory,
)
from .kitchen_phase_b_execution import KitchenPhaseBExecutionDispatcher
from .scene_loader import KitchenScene


DEFAULT_PHASE1 = Path(
    "runs/feasibility_benchmarks/kitchen_feasibility_phase1_closure_20260809/"
    "F1_INITIAL_COMPLETE"
)
DEFAULT_PHASE2 = Path(
    "mujoco_scenes/benchmark_reports/kitchen_symbolic_phase2/variants/"
    "F1_INITIAL_COMPLETE"
)


def _read(path: Path):
    return json.loads(path.read_text())


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-run", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2-run", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/kitchen_phaseB_inventory"))
    parser.add_argument("--robot", choices=("google",), default="google")
    parser.add_argument("--pick-object-id")
    parser.add_argument("--place-destination")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--viewer", action="store_true",
        help="Show a synchronized MuJoCo viewer during the physical PICK/PLACE",
    )
    parser.add_argument("--stop-before-phase-c", action="store_true")
    args = parser.parse_args()

    registry = _read(args.phase1_run / "object_registry.json")
    assignments = _read(args.phase2_run / "grounded_role_assignments.json")
    plan = _read(args.phase2_run / "generated_plan.json")
    inventory = build_phase_b_inventory(registry, assignments, plan)
    scene = KitchenScene(inventory["scene_name"], robot=args.robot)
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(scene.model, scene.data)
        mujoco.mjv_defaultFreeCamera(scene.model, viewer.cam)

    def sync_viewer():
        if viewer is not None and viewer.is_running():
            viewer.sync()

    resolver = KitchenExecutionEntityResolver()
    observed_regions = {
        row["source_context"]["source_container"]
        for row in inventory["objects"]
        if row["source_context"]["source_container"]
    }
    resolution = resolver.resolve(
        inventory,
        resolver.candidates_from_scene(scene, observed_regions=observed_regions),
    )
    _write(args.output_dir / "phaseB_object_inventory.json", inventory)
    _write(args.output_dir / "execution_entity_resolution.json", resolution)
    summary = {
        "phase": "KITCHEN_GOOGLE_EXECUTION_PHASE_B",
        "mode": "INVENTORY_AND_ENTITY_RESOLUTION",
        "scene": inventory["scene_name"],
        "object_count": len(inventory["objects"]),
        "resolved_count": len(resolution["accepted"]),
        "all_resolved": resolution["all_resolved"],
        "one_to_one": resolution["one_to_one"],
        "maximum_centroid_error_m": resolution["maximum_centroid_error_m"],
        "planner_received_backend_names": False,
        "entity_resolution_passed": resolution["all_resolved"],
    }
    _write(args.output_dir / "validation_summary.json", summary)
    if args.pick_object_id:
        if not args.execute:
            summary["pick"] = {
                "generic_object_id": args.pick_object_id,
                "status": "PLAN_ONLY",
            }
        else:
            dispatcher = KitchenPhaseBExecutionDispatcher(
                scene, inventory, resolution,
                step_callback=sync_viewer if viewer is not None else None,
            )
            result = dispatcher.pick(args.pick_object_id)
            summary["pick"] = result
            _write(args.output_dir / "pick_result.json", summary["pick"])
            if result["success"] and args.place_destination:
                place = dispatcher.place(args.pick_object_id, args.place_destination)
                summary["place"] = place
                _write(args.output_dir / "place_result.json", summary["place"])
            if not result["success"] or (
                args.place_destination and not summary.get("place", {}).get("success", False)
            ):
                summary["execution_success"] = False
            else:
                summary["execution_success"] = True
        _write(args.output_dir / "validation_summary.json", summary)
    elif args.stop_before_phase_c:
        dispatcher = KitchenPhaseBExecutionDispatcher(
            scene, inventory, resolution,
            step_callback=sync_viewer if viewer is not None else None,
        )
        actions = []
        if args.execute:
            for action in plan:
                result = dispatcher.execute_phase2_action(action)
                actions.append(result)
                if result["status"] == "UNSUPPORTED_PHASE_C_OPERATOR" or not result["success"]:
                    break
        else:
            for action in plan:
                operator = action["action"].upper()
                status = (
                    "SUPPORTED_PHASE_B"
                    if operator in {"PICK", "PLACE"}
                    else "UNSUPPORTED_PHASE_C_OPERATOR"
                )
                actions.append({"request": action, "status": status})
                if status == "UNSUPPORTED_PHASE_C_OPERATOR":
                    break
        summary["mode"] = "PHASE2_PLAN_SKELETON_STOP_BEFORE_PHASE_C"
        summary["phase2_action_results"] = actions
        summary["symbolic_effects_fabricated"] = False
        _write(args.output_dir / "phase2_adapter_results.json", actions)
        _write(args.output_dir / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if viewer is not None:
        print("Viewer open; close the MuJoCo window to finish.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    explicit_manipulation_requested = bool(
        args.pick_object_id or args.place_destination
    )
    manipulation_failed = (
        args.pick_object_id
        and args.execute
        and not summary["pick"]["success"]
    ) or (
        args.place_destination
        and args.execute
        and not summary.get("place", {}).get("success", False)
    )
    # Inventory-only mode remains a strict whole-run entity-resolution
    # audit.  An explicit PICK/PLACE request, however, succeeds when its own
    # generic object resolves and the requested physical action succeeds;
    # unrelated stale observations must not turn a verified manipulation
    # into a shell failure.
    if manipulation_failed or (
        not explicit_manipulation_requested and not resolution["all_resolved"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
