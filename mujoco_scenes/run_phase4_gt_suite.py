"""Run the Phase-4 executor over the currently GT-feasible paper variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .run_phase4_execution import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHASE3_ROOT,
    execute_phase3_run,
)
from .phase4_execution import Phase4EntityMappingError, UpstreamPhase3Blocked


FEASIBLE_VARIANTS = {
    "kitchen": tuple(f"K{index}" for index in range(1, 7)),
    "living_room": tuple(f"L{index}" for index in range(1, 7)),
    "workshop": tuple(f"W{index}" for index in range(1, 9)),
}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=("all", "kitchen", "living_room", "workshop"),
        default="all",
    )
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-actions", type=int)
    args = parser.parse_args()

    domains = tuple(FEASIBLE_VARIANTS) if args.domain == "all" else (args.domain,)
    conditions = [
        (domain, variant)
        for domain in domains for variant in FEASIBLE_VARIANTS[domain]
    ]
    started = time.perf_counter()
    records = []
    for index, (domain, variant) in enumerate(conditions, start=1):
        print(f"[{index}/{len(conditions)}] {domain} {variant}", flush=True)
        run_dir = args.phase3_root / domain / variant / "gt"
        output_dir = args.output_root / domain / variant / "gt"
        try:
            result = execute_phase3_run(
                run_dir,
                output_dir=output_dir,
                max_actions=args.max_actions,
            )
            record = {
                "domain": domain,
                "variant": variant,
                "success": bool(result["success"]),
                "actions_completed": result["actions_completed"],
                "actions_requested": result["actions_requested"],
                "inspection_actions_requested": result["inspection_execution"]["actions_requested"],
                "inspection_actions_completed": result["inspection_execution"]["actions_completed"],
                "final_verification": result["final_verification"],
                "failure_stage": result.get("failure_stage"),
                "failure": result["failure"],
                "result_path": str(output_dir / "execution_result.json"),
            }
        except Exception as error:
            upstream_blocked = isinstance(error, UpstreamPhase3Blocked)
            entity_mapping_failed = isinstance(error, Phase4EntityMappingError)
            record = {
                "domain": domain,
                "variant": variant,
                "success": False,
                "actions_completed": 0,
                "actions_requested": None,
                "inspection_actions_requested": None,
                "inspection_actions_completed": 0,
                "final_verification": {"performed": False},
                "failure_stage": (
                    "UPSTREAM_PHASE3_BLOCKED"
                    if upstream_blocked
                    else (
                        "ENTITY_RESOLUTION" if entity_mapping_failed else "HANDOFF"
                    )
                ),
                "failure": (
                    "BLOCKED_UPSTREAM_PHASE3"
                    if upstream_blocked
                    else (
                        "ENTITY_MAPPING_FAILURE"
                        if entity_mapping_failed
                        else "SETUP_OR_EXECUTION_EXCEPTION"
                    )
                ),
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
            }
            failure_artifact = {
                "schema_version": 2,
                "phase": "PHASE_4_EXECUTION",
                "domain": domain,
                "variant": variant,
                "success": False,
                "failure": record["failure"],
                "failure_stage": record["failure_stage"],
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
            _write(output_dir / "execution_result.json", failure_artifact)
            _write(
                output_dir / "execution_entity_resolution.json",
                {"all_resolved": False, "failure_stage": record["failure_stage"]},
            )
            _write(
                output_dir / "execution_trace.json",
                {
                    "inspection_execution": failure_artifact["inspection_execution"],
                    "task_plan_execution": failure_artifact["task_plan_execution"],
                },
            )
        records.append(record)
        print(
            f"  -> {'PASS' if record['success'] else 'FAIL'} "
            f"{record['actions_completed']}/{record['actions_requested']} "
            f"{record['failure']}",
            flush=True,
        )
        if args.fail_fast and not record["success"]:
            break
    summary = {
        "schema_version": 1,
        "suite": "STRICT_GT_EXECUTION",
        "strict_execution": True,
        "direct_task_state_fallback_used": False,
        "max_actions": args.max_actions,
        "conditions_requested": len(conditions),
        "conditions_completed": len(records),
        "conditions_passed": sum(row["success"] for row in records),
        "all_passed": len(records) == len(conditions) and all(
            row["success"] for row in records
        ),
        "wall_duration_s": time.perf_counter() - started,
        "results": records,
    }
    _write(args.output_root / "suite_summary.json", summary)
    print(
        f"SUITE STATUS: {summary['conditions_passed']}/"
        f"{summary['conditions_completed']} passed",
        flush=True,
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
