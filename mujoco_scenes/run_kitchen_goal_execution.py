"""Functional goal -> frozen witness -> symbolic plan -> live Google execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from inference_server.functional_client import request_decomposition
from baseline_common.artifacts import prepare_run_directory

from .kitchen_execution_bundle import (
    DEFAULT_TASK,
    KitchenExecutionBundleError,
    build_kitchen_execution_bundle,
)
from .run_kitchen_planner_execution import (
    execute_with_viewer,
    planner_actions,
)
from .symbolic_planning import ground_symbolic_sources
from .task_witness import load_task_requirements


DEFAULT_SOURCE_VOCABULARY = (
    Path(__file__).resolve().parent
    / "configs"
    / "symbolic_source_vocabulary.yaml"
)


class GoalContractError(RuntimeError):
    """Raised when an FM result does not select the configured benchmark."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def initial_observation_images(run_dir: str | Path) -> list[tuple[str, Path]]:
    stages = sorted(Path(run_dir).resolve().glob("stages/000_*"))
    if len(stages) != 1:
        raise GoalContractError(
            "Expected exactly one stage-000 observation in the Phase-1 run"
        )
    paths = sorted(stages[0].glob("cameras/*/rgb.png"))
    if not paths:
        raise GoalContractError("Stage-000 has no camera RGB observations")
    return [(path.parent.name, path) for path in paths]


def validate_goal_contract(
    goal: str,
    task: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    """Ensure the FM decomposition selects exactly the configured task scope."""
    configured_goal = str(task.get("goal_instruction", "")).strip()
    normalize = lambda text: " ".join(str(text).split()).casefold()
    if not configured_goal or normalize(goal) != normalize(configured_goal):
        raise GoalContractError(
            "The typed goal does not match this execution task contract. "
            f"Use exactly: {configured_goal}"
        )
    decomposition = response.get("decomposition", response)
    if not isinstance(decomposition, dict):
        raise GoalContractError("FM response has no decomposition object")
    if decomposition.get("status") != "DECOMPOSED":
        raise GoalContractError(
            f"FM did not decompose the goal: {decomposition.get('status')}"
        )
    functions = {
        str(row.get("function"))
        for row in decomposition.get("functional_requirements", ())
        if isinstance(row, dict) and row.get("function")
    }
    contract = task.get("execution_goal_contract") or {}
    required = set(map(str, contract.get("required_functions", ())))
    missing = sorted(required - functions)
    if missing:
        raise GoalContractError(
            "FM decomposition does not satisfy the configured task contract; "
            f"missing functions: {missing}"
        )
    return decomposition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-run-dir", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument(
        "--task-requirements",
        default=str(DEFAULT_TASK),
    )
    parser.add_argument(
        "--output-dir",
        help="Default: PHASE1_RUN_DIR/live_execution",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PLANNER_BASE_URL", "http://127.0.0.1:18080/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PLANNER_API_KEY", "")
        or os.environ.get("INFERENCE_API_KEY", ""),
    )
    parser.add_argument(
        "--decomposition",
        help="Reuse an existing functional API response instead of calling it",
    )
    parser.add_argument(
        "--semantic-model",
        default="semantic_model_cache/yolov8m-worldv2.pt",
    )
    parser.add_argument(
        "--source-vocabulary",
        default=str(DEFAULT_SOURCE_VOCABULARY),
    )
    parser.add_argument(
        "--refresh-source-grounding",
        action="store_true",
    )
    parser.add_argument("--camera", default="free")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Execute without opening the live viewer",
    )
    parser.add_argument(
        "--close-on-complete",
        action="store_true",
        help="Close the viewer immediately after execution terminates",
    )
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    phase1 = Path(arguments.phase1_run_dir).resolve()
    output = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else phase1 / "live_execution"
    )
    task = load_task_requirements(arguments.task_requirements)

    witness_path = phase1 / "latest_witness.json"
    if not witness_path.is_file():
        parser.error(f"Missing Phase-1 witness: {witness_path}")
    witness = _read(witness_path)
    if str(witness.get("status", "")).upper() != "COMPLETE":
        reasons = ", ".join(map(str, witness.get("reason_codes", ())))
        parser.error(
            f"Phase-1 witness is {witness.get('status')}, not COMPLETE"
            + (f" ({reasons})" if reasons else "")
        )
    try:
        output = prepare_run_directory(output)
    except ValueError as error:
        parser.error(str(error))

    print("[1/4] Validating the goal with the functional model")
    try:
        if arguments.decomposition:
            response = _read(Path(arguments.decomposition))
        else:
            response = request_decomposition(
                scene="kitchen",
                goal=arguments.goal,
                images=initial_observation_images(phase1),
                base_url=arguments.base_url,
                api_key=arguments.api_key,
            )
        validate_goal_contract(arguments.goal, task, response)
    except (GoalContractError, RuntimeError) as error:
        parser.error(str(error))
    _write(output / "functional_decomposition.json", response)

    print("[2/4] Grounding source objects from observed RGB crops")
    source_grounding = phase1 / "symbolic_source_semantics.json"
    if arguments.refresh_source_grounding or not source_grounding.is_file():
        ground_symbolic_sources(
            phase1,
            checkpoint=arguments.semantic_model,
            vocabulary_path=arguments.source_vocabulary,
        )
    else:
        print(f"      reusing {source_grounding}")

    print("[3/4] Sequencing and resolving observed IDs for Google execution")
    try:
        bundle = build_kitchen_execution_bundle(
            phase1,
            output_dir=output,
            task_requirements=arguments.task_requirements,
        )
    except KitchenExecutionBundleError as error:
        parser.error(str(error))
    actions = planner_actions(bundle.plan)
    print((output / "generated_plan.txt").read_text(encoding="utf-8"))

    print("[4/4] Opening the live viewer and executing the grounded plan")
    result = execute_with_viewer(
        bundle.scene,
        bundle.inventory,
        bundle.resolution,
        bundle.registry,
        bundle.witness,
        actions,
        goal=arguments.goal,
        show_viewer=not arguments.headless,
        camera=arguments.camera,
        close_on_complete=arguments.close_on_complete,
    )
    _write(output / "physical_execution_trace.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
