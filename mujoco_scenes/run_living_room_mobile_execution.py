"""CLI for Living-Room Phase 3 mobile refinement and execution."""

from __future__ import annotations

import argparse
import json

from .living_room_mobile_execution import run_mobile_execution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="Living-room variant code (e.g. F0_BASE)")
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--phase2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-task-action", type=int, default=0)
    parser.add_argument("--max-task-actions", type=int)
    arguments = parser.parse_args()
    result = run_mobile_execution(
        arguments.phase1_dir,
        arguments.phase2_dir,
        arguments.output_dir,
        variant=arguments.variant,
        execute=arguments.execute,
        start_task_action=arguments.start_task_action,
        max_task_actions=arguments.max_task_actions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] not in {"SUCCESS", "INFEASIBLE_CONFIRMED"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
