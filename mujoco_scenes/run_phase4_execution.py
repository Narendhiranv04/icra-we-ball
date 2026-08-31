"""CLI for executing an immutable Phase-3 final action sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .phase4_execution import Phase4Executor, load_phase3_handoff


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
    assisted_suite: bool = False,
    allow_assisted_pick_recovery: bool = True,
) -> dict[str, Any]:
    handoff = load_phase3_handoff(run_dir)
    if handoff.domain == "living_room":
        from .phase4_living_room import execute_living_room_handoff

        result = execute_living_room_handoff(
            handoff,
            output_dir=output_dir,
            max_actions=max_actions,
            assisted_suite=assisted_suite,
        )
        _write_json(output_dir / "execution_result.json", result)
        _write_json(
            output_dir / "execution_entity_resolution.json",
            result["entity_resolution"],
        )
        _write_json(
            output_dir / "execution_trace.json",
            {"actions": result["action_results"]},
        )
        return result
    if handoff.domain == "kitchen":
        from .phase4_kitchen import KitchenPhase4Adapter

        adapter = KitchenPhase4Adapter(
            handoff,
            assisted_suite=assisted_suite,
            allow_assisted_pick_recovery=allow_assisted_pick_recovery,
        )
    elif handoff.domain == "workshop":
        from .phase4_workshop import WorkshopPhase4Adapter

        adapter = WorkshopPhase4Adapter(
            handoff, assisted_suite=assisted_suite
        )
    else:
        raise NotImplementedError(
            f"Phase-4 adapter is not implemented yet for {handoff.domain}"
        )
    result = Phase4Executor(handoff, adapter).run(max_actions=max_actions)
    _write_json(output_dir / "execution_result.json", result)
    _write_json(output_dir / "execution_entity_resolution.json", adapter.entity_resolution)
    _write_json(output_dir / "execution_trace.json", {"actions": result["action_results"]})
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
    parser.add_argument("--assisted-suite", action="store_true")
    parser.add_argument("--strict-pick", action="store_true")
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
            assisted_suite=args.assisted_suite,
            allow_assisted_pick_recovery=not args.strict_pick,
        )
    except Exception as error:
        failure = {
            "schema_version": 1,
            "phase": "PHASE_4_EXECUTION",
            "domain": args.domain,
            "variant": args.variant.upper(),
            "success": False,
            "failure": "INVALID_HANDOFF_OR_ADAPTER_SETUP",
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
        }
        _write_json(output_dir / "execution_result.json", failure)
        print(f"PHASE 4 FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"PHASE 4 STATUS: {'SUCCESS' if result['success'] else 'FAILED'} "
        f"({result['actions_completed']}/{result['actions_requested']} actions)"
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
