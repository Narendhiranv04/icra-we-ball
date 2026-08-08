"""Pure Phase-2 entry point: frozen COMPLETE witness -> validated plan."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from mujoco_scenes.symbolic_planning import compile_plan_and_save


DEFAULT_TASK = "mujoco_scenes/configs/s1_integrated_kitchen_object_function.yaml"


def run_pipeline(arguments: argparse.Namespace) -> dict:
    """Compile and plan without rendering, perception, inspection, or robots."""
    source = Path(arguments.phase1_run_dir).resolve()
    output = Path(arguments.output_root).resolve() / arguments.run_id
    result = compile_plan_and_save(
        source,
        arguments.task_requirements,
        output_dir=output,
    )
    print("\n" + (output / "generated_plan.txt").read_text(encoding="utf-8"))
    print(f"Phase-1 input: {source}")
    print(f"Phase-2 artifacts: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a frozen COMPLETE Phase-1 witness and search the generic "
            "PICK/PLACE/POUR/STIR symbolic domain"
        )
    )
    parser.add_argument("--phase1-run-dir", required=True)
    parser.add_argument("--task-requirements", default=DEFAULT_TASK)
    parser.add_argument("--output-root", default="runs/phase2_symbolic")
    parser.add_argument(
        "--run-id",
        default=f"phase2_symbolic_{datetime.now():%Y%m%d_%H%M%S}",
    )
    # Accepted only so old scripts fail neither mysteriously nor by invoking a
    # robot. The Phase-2 implementation contains no robot path.
    parser.add_argument("--no-robot", action="store_true", default=True)
    return parser


def main() -> None:
    run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    main()
