"""CLI for executing an immutable Phase-3 final action sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .phase4_execution import (
    Phase4Executor,
    Phase4EntityMappingError,
    UpstreamPhase3Blocked,
    load_phase3_handoff,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3_ROOT = ROOT / "runs" / "functional_tamp_pipeline"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "phase4_execution"


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
) -> dict[str, Any]:
    handoff = load_phase3_handoff(run_dir)
    if handoff.domain == "living_room":
        from .phase4_living_room import execute_living_room_handoff

        result = execute_living_room_handoff(
            handoff,
            output_dir=output_dir,
            max_actions=max_actions,
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
        from .phase4_kitchen import KitchenPhase4Adapter

        adapter = KitchenPhase4Adapter(handoff)
    elif handoff.domain == "workshop":
        from .phase4_workshop import WorkshopPhase4Adapter

        adapter = WorkshopPhase4Adapter(handoff)
    else:
        raise NotImplementedError(
            f"Phase-4 adapter is not implemented yet for {handoff.domain}"
        )
    result = Phase4Executor(handoff, adapter).run(max_actions=max_actions)
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


def main() -> int:
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
    args = parser.parse_args()

    run_dir = args.phase3_run or (
        args.phase3_root / args.domain / args.variant.upper() / args.mode
    )
    output_dir = args.output_root / args.domain / args.variant.upper() / args.mode
    try:
        result = execute_phase3_run(
            run_dir,
            output_dir=output_dir,
            max_actions=args.max_actions,
        )
    except Exception as error:
        upstream_blocked = isinstance(error, UpstreamPhase3Blocked)
        entity_mapping_failed = isinstance(error, Phase4EntityMappingError)
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
                    "ENTITY_MAPPING_FAILURE"
                    if entity_mapping_failed
                    else "INVALID_HANDOFF_OR_ADAPTER_SETUP"
                )
            ),
            "failure_stage": (
                "UPSTREAM_PHASE3_BLOCKED"
                if upstream_blocked
                else ("ENTITY_RESOLUTION" if entity_mapping_failed else "HANDOFF")
            ),
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
            "strict_execution": True,
            "direct_task_state_fallback_used": False,
            "strict_telemetry_verification": {
                "verified": True,
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
