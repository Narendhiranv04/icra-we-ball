"""CLI for executing an immutable Phase-3 final action sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import mujoco

from .phase4_execution import (
    Phase4Executor,
    Phase4EntityMappingError,
    Phase4ViewerClosed,
    UpstreamPhase3Blocked,
    load_phase3_handoff,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3_ROOT = ROOT / "runs" / "functional_tamp_pipeline"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "phase4_execution"
SUPPORTED_MUJOCO_VERSION = "3.3.5"


def require_supported_mujoco_runtime() -> None:
    """Fail before simulation when physics differs from the calibrated runtime."""
    installed = str(mujoco.__version__)
    if installed != SUPPORTED_MUJOCO_VERSION:
        raise RuntimeError(
            "Phase-4 controllers are calibrated and validated with MuJoCo "
            f"{SUPPORTED_MUJOCO_VERSION}, but this interpreter has {installed}. "
            "Use the repository environment (`.venv/bin/python`) installed "
            "from mujoco_scenes/requirements.txt."
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def execute_phase3_run(
    run_dir: Path,
    *,
    output_dir: Path,
    max_actions: int | None = None,
    record_video: Path | None = None,
    viewer: bool = False,
) -> dict[str, Any]:
    require_supported_mujoco_runtime()
    handoff = load_phase3_handoff(run_dir)
    if handoff.domain == "living_room":
        from .phase4_living_room import execute_living_room_handoff

        result = execute_living_room_handoff(
            handoff,
            output_dir=output_dir,
            max_actions=max_actions,
            record_video=record_video,
            viewer=viewer,
        )
        _write_json(output_dir / "execution_result.json", result)
        _write_json(
            output_dir / "execution_entity_resolution.json",
            result["entity_resolution"],
        )
        _write_json(
            output_dir / "execution_trace.json",
            {
                "inspection_execution": result["inspection_execution"],
                "task_plan_execution": result["task_plan_execution"],
            },
        )
        return result
    if handoff.domain == "kitchen":
        # Container inspection/navigation can leave the empty arm in an
        # articulation posture.  Normalize it physically before each PICK so
        # grasp planning starts from the same calibrated navigation state in
        # all K1--K6 variants, including variants with inspections.
        from .kitchen_phase4_transition_fix import (
            install_patch as install_kitchen_transition_fix,
        )

        install_kitchen_transition_fix()

        # Kitchen soup bowls become task-terminal served assemblies only after
        # the normal physical PLACE(bowl, serving_area) has succeeded.  Install
        # this before composing the Kitchen scene so the required inactive
        # equality constraints exist in every K1--K6 execution.  The helper is
        # generic over the Phase-3 assignment; no K-variant or object ID is
        # hard-coded here.
        from .run_phase4_execution_served_static import (
            install_patch as install_kitchen_served_terminal_latch,
        )

        install_kitchen_served_terminal_latch()

        from .phase4_kitchen import KitchenPhase4Adapter

        adapter = KitchenPhase4Adapter(
            handoff, record_video=record_video, viewer=viewer
        )
        print("[Kitchen execution bindings]", flush=True)
        for planner_id, entity in sorted(adapter.by_id.items()):
            print(
                f"  {planner_id} -> {entity.simulator_id}",
                flush=True,
            )
    elif handoff.domain == "workshop":
        from .phase4_workshop import WorkshopPhase4Adapter

        adapter = WorkshopPhase4Adapter(
            handoff, record_video=record_video, viewer=viewer
        )
    else:
        raise NotImplementedError(
            f"Phase-4 adapter is not implemented yet for {handoff.domain}"
        )
    try:
        result = Phase4Executor(handoff, adapter).run(max_actions=max_actions)
    finally:
        visual_output = adapter.close_visualization()
    result["visual_output"] = visual_output
    _write_json(output_dir / "execution_result.json", result)
    _write_json(output_dir / "execution_entity_resolution.json", adapter.entity_resolution)
    _write_json(
        output_dir / "execution_trace.json",
        {
            "inspection_execution": result["inspection_execution"],
            "task_plan_execution": result["task_plan_execution"],
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the exact persisted Phase-3 symbolic action sequence"
    )
    parser.add_argument("--domain", choices=("kitchen", "living_room", "workshop"), required=True)
    parser.add_argument("--variant", required=True, help="Paper label, e.g. K1")
    parser.add_argument("--mode", choices=("gt", "vlm"), default="gt")
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--phase3-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument(
        "--record-video", type=Path,
        help="Write a synchronized manual-review MP4 to this path.",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help=(
            "Show the controlled model/data in a passive MuJoCo viewer. "
            "Task indices printed by this CLI are 1-based Phase-3 indices."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    run_dir = args.phase3_run or (
        args.phase3_root / args.domain / args.variant.upper() / args.mode
    )
    output_dir = args.output_root / args.domain / args.variant.upper() / args.mode
    print(
        "Phase-4 progress uses 1-based task indices from the persisted "
        "Phase-3 action sequence.",
        flush=True,
    )
    try:
        result = execute_phase3_run(
            run_dir, output_dir=output_dir, max_actions=args.max_actions,
            record_video=args.record_video, viewer=args.viewer,
        )
    except Exception as error:
        upstream_blocked = isinstance(error, UpstreamPhase3Blocked)
        entity_mapping_failed = isinstance(error, Phase4EntityMappingError)
        viewer_closed = isinstance(error, Phase4ViewerClosed)
        failure = {
            "schema_version": 2,
            "phase": "PHASE_4_EXECUTION",
            "domain": args.domain,
            "variant": args.variant.upper(),
            "success": False,
            "failure": (
                "BLOCKED_UPSTREAM_PHASE3"
                if upstream_blocked
                else (
                    "EXECUTION_ABORTED_BY_VIEWER"
                    if viewer_closed
                    else (
                        "ENTITY_MAPPING_FAILURE"
                        if entity_mapping_failed
                        else "INVALID_HANDOFF_OR_ADAPTER_SETUP"
                    )
                )
            ),
            "failure_stage": (
                "UPSTREAM_PHASE3_BLOCKED"
                if upstream_blocked
                else (
                    "TASK_ACTION" if viewer_closed
                    else ("ENTITY_RESOLUTION" if entity_mapping_failed else "HANDOFF")
                )
            ),
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
            "failure_code": (
                "EXECUTION_ABORTED_BY_VIEWER"
                if viewer_closed else "EXECUTION_ERROR"
            ),
            "execution_mode": "P4_BENCH",
            "strict_execution": False,
            "strict_execution_violation_detected": False,
            "direct_task_state_write_used": False,
            "direct_payload_state_write_used": False,
            "assisted_task_fixture_used": False,
            "post_release_dynamics_modified": False,
            "direct_task_state_fallback_used": False,
            "strict_telemetry_verification": {
                "verified": True,
                "strict_execution_violation_detected": False,
                "direct_task_state_write_used": False,
                "direct_task_state_fallback_used": False,
                "direct_payload_state_write_used": False,
                "assisted_task_fixture_used": False,
                "post_release_dynamics_modified": False,
                "violations": [],
                "reason": "NO_PRIMITIVE_EXECUTION_ATTEMPTED",
            },
            "inspection_execution": {
                "regions": [], "actions_requested": 0,
                "actions_completed": 0, "results": [], "success": False,
            },
            "task_plan_execution": {"actions": [], "results": []},
            "final_verification": {"performed": False},
        }
        _write_json(output_dir / "execution_result.json", failure)
        _write_json(
            output_dir / "execution_entity_resolution.json",
            {"all_resolved": False, "failure_stage": failure["failure_stage"]},
        )
        _write_json(
            output_dir / "execution_trace.json",
            {
                "inspection_execution": failure["inspection_execution"],
                "task_plan_execution": failure["task_plan_execution"],
            },
        )
        print(f"PHASE 4 FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"PHASE 4 STATUS: {'SUCCESS' if result['success'] else 'FAILED'} "
        f"({result['actions_completed']}/{result['actions_requested']} actions)"
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())