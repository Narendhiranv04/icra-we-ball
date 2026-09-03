"""Run planning-only OWL-TAMP on one Workshop W1--W10 variant."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from typing import Sequence

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.models import Action as SharedAction

from vlm_tamp_baseline.workshop_runtime import (
    WorkshopPlanningRuntime,
    WorkshopSymbolicExecutor,
    canonical_workshop_actions,
    compare_workshop_actions,
)

from .evaluation import load_expected
from .models import Action, Constraint
from .planner import OWLTAMPPlanner, OWLTAMPPlannerConfig, protocol_max_tokens
from .prompt import PROMPT_VERSION
from .receding_horizon import OWLTAMPRecedingHorizon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="W1-W10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", choices=(1, 3, 5), type=int, default=5)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument(
        "--protocol",
        choices=("native", "single_call", "receding_horizon"),
        default="native",
    )
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument("--max-total-actions", type=int, default=48)
    parser.add_argument(
        "--max-sketch-actions",
        type=int,
        default=24,
        help=(
            "Bound on the per-action constraint calls the native protocol "
            "issues.  OWL-TAMP asks for one constraint per sketch action, so a "
            "model that degenerates into a repeating sketch bills one request "
            "per repeat; truncation is recorded in the trace as "
            "constraint_generation_complete."
        ),
    )
    return parser


def _shared_action(action: Action) -> SharedAction:
    names = {
        "INSPECT": ("region_id",), "PICK": ("object_id",), "PLACE": ("object_id", "region_id"),
        "INSERT": ("fastener_id", "target_id"), "FASTEN": ("tool_id", "fastener_id", "target_id"),
    }[action.operator]
    return SharedAction(action.operator, {name: value for name, value in zip(names, action.arguments)})


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_run_directory(args.output_dir)
    runtime = WorkshopPlanningRuntime(
        args.variant, output, image_width=args.image_width, image_height=args.image_height, camera_count=args.camera_count,
    )
    config = OWLTAMPPlannerConfig.from_env()
    config = replace(
        config,
        max_tokens=protocol_max_tokens(
            args.max_tokens,
            "native" if args.protocol == "receding_horizon" else args.protocol,
        ),
        seed=args.seed,
        **({"base_url": args.base_url} if args.base_url else {}),
        **({"model": args.model} if args.model else {}),
    )
    planner = OWLTAMPPlanner(config)
    observation, images = runtime.observe()

    def oracle(action: Action, _constraints: Sequence[Constraint], _trial: int) -> bool:
        if action.operator == "INSPECT":
            return action.arguments[0] in runtime.storage_region_ids
        if action.operator == "PICK":
            return action.arguments[0] in observation.object_ids and action.arguments[0] != runtime.target_object_id
        if action.operator == "PLACE":
            return action.arguments[0] in observation.object_ids and action.arguments[1] in runtime.destination_ids
        if action.operator == "INSERT":
            return (
                action.arguments[0] in observation.object_ids
                and runtime.is_fastener(action.arguments[0])
                and action.arguments[1] == runtime.target_object_id
            )
        if action.operator == "FASTEN":
            tool, fastener, target = action.arguments
            return (
                tool in observation.object_ids
                and fastener in observation.object_ids
                and target == runtime.target_object_id
                and runtime.is_driver(tool)
                and runtime.is_fastener(fastener)
            )
        return False

    try:
        executor = WorkshopSymbolicExecutor(runtime)
        horizon = None
        if args.protocol == "receding_horizon":
            def observe():
                nonlocal observation, images
                observation, images = runtime.observe()
                return observation, images

            horizon = OWLTAMPRecedingHorizon(
                planner,
                observe,
                lambda action: executor.execute(_shared_action(action)),
                lambda _observation: runtime.goal_verifier(),
                oracle,
                movable_objects=lambda row: tuple(
                    object_id
                    for object_id in row.object_ids
                    if object_id != runtime.target_object_id
                ),
                max_replans=args.max_replans,
                max_total_actions=args.max_total_actions,
            ).run(args.goal or runtime.goal)
            history = [
                {
                    **row,
                    "action": _shared_action(Action.parse(row["action"])).as_dict(),
                }
                for row in horizon.action_history
            ]
            result_payload = horizon.as_dict()
            raw_vlm_requests = horizon.raw_vlm_requests
            planning_rounds = horizon.planning_rounds
            replans = horizon.replans
            write_json(output / "receding_horizon_trace.json", result_payload)
        else:
            result = planner.plan(
                args.goal or runtime.goal,
                observation,
                images,
                oracle,
                movable_object_ids=tuple(
                    object_id
                    for object_id in observation.object_ids
                    if object_id != runtime.target_object_id
                ),
                max_vlm_requests=(
                    1
                    if args.protocol == "single_call"
                    # One sketch request plus at most one constraint request
                    # per sketch action.
                    else 1 + args.max_sketch_actions
                ),
            )
            history = []
            for action in result.actions:
                shared = _shared_action(action)
                outcome = executor.execute(shared)
                history.append({"action": shared.as_dict(), "success": outcome.success, "failure_code": outcome.failure_code, "message": outcome.message})
                if not outcome.success:
                    break
            result_payload = {**result.as_dict(), "action_history": history}
            raw_vlm_requests = len(planner.response_trace)
            planning_rounds = 1
            replans = 0
        runtime.observe_state()
        backend_by_id = {**runtime.object_by_backend, **runtime.region_by_backend}
        predicted = canonical_workshop_actions(history, backend_by_id)
        expected = load_expected("workshop", runtime.variant)
        predicted_outcome = (
            "FEASIBLE" if runtime.goal_verifier()
            else "INFEASIBLE" if runtime.infeasibility_proven()
            else "UNRESOLVED"
        )
        comparison = compare_workshop_actions(predicted, expected["actions"])
        comparison.update({
            "variant": runtime.variant, "predicted_outcome": predicted_outcome,
            "expected_outcome": expected["intended_outcome"],
            "outcome_match": predicted_outcome == expected["intended_outcome"],
            "gt_was_model_input": False,
        })
        payload = {
            "baseline": "owl_tamp_paper_derived", "official_author_code": False,
            "environment": "workshop", "variant": runtime.variant, "goal": args.goal or runtime.goal,
            "model": config.model, "seed": config.seed, "camera_count": args.camera_count,
            "prompt_version": PROMPT_VERSION, "planning_rounds": planning_rounds,
            "replans": replans,
            "raw_vlm_requests": raw_vlm_requests,
            "protocol": args.protocol,
            "max_tokens": config.max_tokens,
            "physical_execution": False,
            "hidden_storage_contents_visible_to_model": False,
            "result": result_payload, "gt_comparison": comparison,
        }
        write_json(output / "method_manifest.json", {
            "method": "OWL-TAMP paper-derived reproduction", "official_author_code": False,
            "environment": "workshop", "evaluation_mode": "PLANNING_ONLY_GT_SEQUENCE_COMPARISON",
            "physical_execution": False, "prompt_version": PROMPT_VERSION, "model": config.model,
            "seed": config.seed, "camera_count": args.camera_count,
            "planner_input_contract": "ALIAS_ANNOTATED_RGB_PLUS_OBSERVABLE_ALIAS_ID_MAP",
            "hidden_storage_contents_visible_to_model": False, "gt_visible_to_model": False,
        })
        if args.protocol != "receding_horizon":
            write_json(output / "model_trace.json", planner.trace)
        write_json(output / "episode_result.json", payload)
        print("[OWL-TAMP refined plan]", flush=True)
        for index, row in enumerate(history, 1):
            action = row["action"]
            arguments = tuple(map(str, action.get("arguments", {}).values()))
            shown = ", ".join(
                f"{value} ({runtime.aliases[value]})" if value in runtime.aliases else value
                for value in arguments
            )
            print(f"  {index}. {action.get('skill')} {shown}", flush=True)
        print("[GT task-level comparison]", json.dumps(comparison["shared_task_vocabulary"], sort_keys=True), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
