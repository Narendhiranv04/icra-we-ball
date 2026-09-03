"""Run planning-only OWL-TAMP on a five-view Kitchen observation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Sequence

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.execution import MuJoCoActionExecutor
from baseline_common.models import Action as SharedAction
from baseline_common.physical_benchmark import write_execution_result

from mujoco_scenes.baseline_kitchen_runtime import BaselineKitchenRuntime
from vlm_tamp_baseline.kitchen_planning_runtime import KitchenPlanningState

from .evaluation import EXPECTED_ROOT, compare_kitchen_actions, load_expected
from .models import Action, Constraint
from .planner import OWLTAMPPlanner, OWLTAMPPlannerConfig, protocol_max_tokens
from .prompt import PROMPT_VERSION
from .receding_horizon import OWLTAMPRecedingHorizon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(f"K{i}" for i in range(1, 13)))
    parser.add_argument("--phase1-run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument(
        "--protocol",
        choices=("native", "single_call", "receding_horizon"),
        default="native",
    )
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument("--max-total-actions", type=int, default=48)
    parser.add_argument(
        "--physical-execution",
        action="store_true",
        help="Execute the VLM-produced plan through the shared Google-robot Kitchen skills.",
    )
    parser.add_argument("--camera", default="free")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--close-on-complete", action="store_true")
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


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_run_directory(args.output_dir)
    if args.phase1_run_dir is not None:
        raise SystemExit(
            "OWL-TAMP K1-K12 planning constructs variants directly; remove "
            "--phase1-run-dir"
        )
    expected_path = EXPECTED_ROOT / "kitchen" / args.variant / "expected_gt_actions.json"
    runtime = BaselineKitchenRuntime.from_variant(
        args.variant,
        output,
        expected_actions=expected_path if args.physical_execution else None,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_count=args.camera_count,
        viewer_camera=args.camera,
        show_viewer=args.physical_execution and not args.headless,
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
    inventory = {
        str(row["generic_object_id"]): row
        for row in runtime.bundle.inventory.get("objects", ())
        if str(row["generic_object_id"]) in observation.object_ids
    }

    def oracle(action: Action, _constraints: Sequence[Constraint], _trial: int) -> bool:
        if action.operator == "OPEN":
            return action.arguments[0] in observation.region_ids
        if action.arguments[0] not in inventory:
            return False
        if action.operator == "PICK":
            return bool(inventory[action.arguments[0]].get("observed_dimensions_m"))
        if action.operator == "PLACE":
            return action.arguments[1] in observation.object_ids | observation.region_ids
        if action.operator in {"POUR", "STIR", "PLACE_SERVING_UTENSIL"}:
            return action.arguments[1] in inventory
        return False

    def to_shared_action(action: Action) -> SharedAction:
        names = {
            "OPEN": ("region_id",),
            "PICK": ("object_id",),
            "PLACE": ("object_id", "region_id"),
            "POUR": ("source_id", "target_id"),
            "STIR": ("tool_id", "target_id"),
            "PLACE_SERVING_UTENSIL": ("object_id", "region_id"),
        }[action.operator]
        skill = "INSPECT" if action.operator == "OPEN" else (
            "PLACE" if action.operator == "PLACE_SERVING_UTENSIL" else action.operator
        )
        return SharedAction(skill, dict(zip(names, action.arguments)))

    def trace_latency_seconds(rows: Sequence[object]) -> float:
        total_ms = 0.0
        for row in rows:
            if isinstance(row, dict):
                trace = row.get("model_trace", row)
                if isinstance(trace, dict):
                    total_ms += float(trace.get("latency_ms", 0.0))
        return total_ms / 1000.0

    try:
        if args.physical_execution:
            runtime.open()
        episode_started = time.monotonic()
        goal = args.goal or str(runtime.scene.config.goal)
        physical_executor = (
            MuJoCoActionExecutor(
                runtime.dispatcher,
                effect_sink=runtime.accept_effects,
                status_sink=runtime.sync,
            )
            if args.physical_execution
            else None
        )
        horizon = None
        planning_state = None
        if args.protocol == "receding_horizon":
            if not args.physical_execution:
                planning_state = KitchenPlanningState(runtime)

            def observe():
                nonlocal observation, images
                nonlocal inventory
                if args.physical_execution:
                    observation, images = runtime.observe()
                else:
                    assert planning_state is not None
                    observation, images = planning_state.observe()
                inventory = {
                    str(row["generic_object_id"]): row
                    for row in runtime.bundle.inventory.get("objects", ())
                    if str(row["generic_object_id"]) in observation.object_ids
                }
                return observation, images

            def execute(action: Action):
                shared = to_shared_action(action)
                if physical_executor is not None:
                    return physical_executor.execute(shared)
                assert planning_state is not None
                return planning_state.execute(shared)

            horizon = OWLTAMPRecedingHorizon(
                planner,
                observe,
                execute,
                runtime.goal_verifier,
                oracle,
                max_replans=args.max_replans,
                max_total_actions=args.max_total_actions,
            ).run(goal)
            planned_actions = tuple(
                Action.parse(row["action"]) for row in horizon.action_history
            )
            result_payload = horizon.as_dict()
            planning_rounds = horizon.planning_rounds
            replans = horizon.replans
            raw_vlm_requests = horizon.raw_vlm_requests
            write_json(output / "receding_horizon_trace.json", result_payload)
        else:
            result = planner.plan(
                goal,
                observation,
                images,
                oracle,
                max_vlm_requests=(
                    1
                    if args.protocol == "single_call"
                    # One sketch request plus at most one constraint request
                    # per sketch action.
                    else 1 + args.max_sketch_actions
                ),
            )
            planned_actions = result.actions
            result_payload = result.as_dict()
            planning_rounds = 1
            replans = 0
            raw_vlm_requests = len(planner.response_trace)
            if physical_executor is not None and result.status == "PLAN":
                prepared = physical_executor.prepare(
                    tuple(to_shared_action(action) for action in planned_actions)
                )
                action_history = []
                if prepared.success:
                    for action in planned_actions:
                        shared = to_shared_action(action)
                        outcome = physical_executor.execute(shared)
                        action_history.append({
                            "action": shared.as_dict(),
                            "success": outcome.success,
                            "failure_code": outcome.failure_code,
                            "message": outcome.message,
                            "effects": list(outcome.effects),
                            "details": dict(outcome.details),
                        })
                        if not outcome.success:
                            break
                else:
                    action_history.append({
                        "action": None,
                        "success": False,
                        "failure_code": prepared.failure_code,
                        "message": prepared.message,
                        "effects": [],
                        "details": dict(prepared.details),
                    })
                result_payload = {
                    **result.as_dict(),
                    "action_history": action_history,
                    "physical_goal_satisfied": runtime.goal_verifier(runtime.observe_state()),
                }
        backend_by_id = {
            str(row["generic_object_id"]): str(row["physical_backend_body"])
            for row in runtime.bundle.resolution.get("accepted", ())
        }
        translated = tuple(
            Action(
                action.operator,
                tuple(backend_by_id.get(argument, argument) for argument in action.arguments),
            )
            for action in planned_actions
        )
        expected = load_expected("kitchen", args.variant)
        comparison = compare_kitchen_actions(translated, expected["actions"])
        physical_success = (
            runtime.goal_verifier(runtime.observe_state())
            if args.physical_execution
            else False
        )
        predicted_outcome = (
            "FEASIBLE" if args.physical_execution and physical_success else
            "UNRESOLVED" if args.physical_execution else
            "FEASIBLE" if horizon is not None and horizon.success else
            "UNRESOLVED" if horizon is not None else
            "FEASIBLE" if result.status == "PLAN" else "INFEASIBLE"
            if result.status == "NO_PLAN" else "UNRESOLVED"
        )
        comparison.update(
            {
                "variant": args.variant,
                "predicted_outcome": predicted_outcome,
                "expected_outcome": expected["intended_outcome"],
                "outcome_match": predicted_outcome == expected["intended_outcome"],
                "gt_was_model_input": False,
            }
        )
        payload = {
            "baseline": "owl_tamp_paper_derived",
            "official_author_code": False,
            "environment": "kitchen",
            "variant": args.variant,
            "goal": goal,
            "model": config.model,
            "seed": config.seed,
            "camera_count": args.camera_count,
            "prompt_version": PROMPT_VERSION,
            "planning_rounds": planning_rounds,
            "replans": replans,
            "raw_vlm_requests": raw_vlm_requests,
            "protocol": args.protocol,
            "max_tokens": config.max_tokens,
            "physical_execution": args.physical_execution,
            "observable_state_only": True,
            "closed_storage_contents_exposed": False,
            "result": result_payload,
            "private_id_translation_used_only_for_gt_comparison": True,
            "gt_comparison": comparison,
        }
        if horizon is None:
            write_json(output / "model_trace.json", planner.trace)
        write_json(output / "episode_result.json", payload)
        if args.physical_execution:
            planning_latency_s = (
                trace_latency_seconds(horizon.planning_trace)
                if horizon is not None
                else trace_latency_seconds((planner.trace,))
            )
            terminal_failure = (
                {"message": horizon.failure}
                if horizon is not None and horizon.failure
                else {}
            )
            terminal_status = (
                horizon.status
                if horizon is not None
                else "GOAL_COMPLETE" if physical_success else str(result.status)
            )
            write_execution_result(
                output,
                scene="kitchen",
                method="owl_tamp",
                protocol=args.protocol,
                variant=args.variant,
                camera_count=args.camera_count,
                seed=config.seed,
                success=physical_success,
                executed_actions=(
                    horizon.executed_actions
                    if horizon is not None
                    else len(result_payload.get("action_history", ()))
                ),
                model_calls=planning_rounds,
                raw_vlm_requests=raw_vlm_requests,
                replans=replans,
                planning_latency_s=planning_latency_s,
                elapsed_seconds=time.monotonic() - episode_started,
                terminal_status=terminal_status,
                terminal_failure=terminal_failure,
            )
        print("[OWL-TAMP refined plan]", flush=True)
        aliases = {
            generic: backend for generic, backend in backend_by_id.items()
        }
        for index, action in enumerate(planned_actions, start=1):
            shown = [f"{item} ({aliases[item]})" if item in aliases else item for item in action.arguments]
            print(f"  {index}. {action.operator} {', '.join(shown)}", flush=True)
        print("[GT comparison]", json.dumps(comparison, sort_keys=True), flush=True)
        if args.physical_execution and not args.close_on_complete:
            runtime.wait_for_viewer()
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
