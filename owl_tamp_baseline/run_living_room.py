"""Run planning-only OWL-TAMP on one Living Room benchmark variant."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
import json
from typing import Any, Mapping, Sequence

import time

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.living_room_execution import (
    LivingRoomPhysicalExecutor,
    build_living_room_physical_runtime,
)
from baseline_common.models import Action as SharedAction
from baseline_common.physical_benchmark import (
    physical_terminal_status,
    write_execution_result,
)

from vlm_tamp_baseline.living_room_runtime import (
    LivingRoomPlanningRuntime,
    LivingRoomSymbolicExecutor,
)
from vlm_tamp_baseline.pddlstream_refiner import LivingRoomGeometryOracle

from .evaluation import compare_actions, load_expected
from .models import Action, Constraint
from .planner import (
    OWLTAMPPlanner,
    OWLTAMPPlannerConfig,
    protocol_max_tokens,
    registry_sampling,
)
from .prompt import PROMPT_VERSION
from .receding_horizon import OWLTAMPRecedingHorizon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="L1-L10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument(
        "--protocol",
        choices=("native", "single_call", "receding_horizon"),
        default="native",
    )
    parser.add_argument(
        "--physical-execution",
        action="store_true",
        help=(
            "Execute the refined plan through the calibrated Google-robot "
            "Living Room skills instead of the symbolic executor."
        ),
    )
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="model-native",
        help=(
            "'paper' reproduces the baseline paper's own condition "
            "(temperature 0.2, top_p 1.0, thinking disabled).  'model-native' "
            "uses the served checkpoint's published sampling with thinking "
            "enabled.  Both baselines must run the same choice or the "
            "comparison measures decoding rather than method."
        ),
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
            "model that degenerates into a repeating sketch costs one request "
            "per repeat: an observed 64-action sketch billed 65 requests and 67 "
            "s for a plan the symbolic layer rejects either way.  The Living "
            "Room task needs 10 actions, so this bounds cost without reaching "
            "the length any real plan uses; truncation is recorded in the "
            "trace as constraint_generation_complete."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_run_directory(args.output_dir)
    if args.physical_execution:
        runtime = build_living_room_physical_runtime(
            args.variant,
            output,
            camera_count=args.camera_count,
            show_viewer=False,
        )
    else:
        runtime = LivingRoomPlanningRuntime(
            args.variant,
            output,
            image_width=args.image_width,
            image_height=args.image_height,
            camera_count=args.camera_count,
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
    if args.decoding == "paper":
        config = replace(
            config,
            enable_thinking=False,
            sampling={"temperature": 0.2, "top_p": 1.0},
        )
    else:
        # Re-resolve sampling: from_env picked the non-thinking block.
        config = replace(
            config,
            enable_thinking=True,
            sampling=registry_sampling(
                os.environ.get("OWL_TAMP_PROFILE", "qwen35-9b"), True
            ),
        )
    planner = OWLTAMPPlanner(config)
    geometry = LivingRoomGeometryOracle(runtime.inventory, runtime.region_registry)
    observation, images = runtime.observe()

    def oracle(action: Action, _constraints: Sequence[Constraint], trial: int) -> bool:
        if action.operator == "PICK":
            row = geometry.certify(
                "pick", (action.arguments[0], runtime.locations[action.arguments[0]], "home"),
                trial=trial, observation=observation,
            )
            return row is not None
        if action.operator == "PLACE":
            row = geometry.certify(
                "place", (action.arguments[0], action.arguments[1], "home"),
                trial=trial, observation=observation,
            )
            return row is not None
        return False

    ACTION_ARGUMENTS = {
        "PICK": ("object_id",),
        "PLACE": ("object_id", "region_id"),
    }

    def to_shared_action(action: Action) -> SharedAction:
        names = ACTION_ARGUMENTS[action.operator]
        return SharedAction(action.operator, dict(zip(names, action.arguments)))

    started_at = time.monotonic()
    try:
        goal = args.goal or runtime.goal
        horizon = None
        executed_actions = 0
        physical_goal_satisfied = False
        action_history: list[dict[str, Any]] = []
        executor = (
            LivingRoomPhysicalExecutor(runtime)
            if args.physical_execution
            else LivingRoomSymbolicExecutor(runtime)
        )
        if args.protocol == "receding_horizon":

            def observe():
                nonlocal observation, images
                observation, images = runtime.observe()
                return observation, images

            def execute(action: Action):
                return executor.execute(to_shared_action(action))

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
            if args.physical_execution and result.status == "PLAN":
                for action in planned_actions:
                    outcome = executor.execute(to_shared_action(action))
                    executed_actions += 1
                    action_history.append({
                        "action": to_shared_action(action).as_dict(),
                        "success": outcome.success,
                        "failure_code": outcome.failure_code,
                        "message": outcome.message,
                        "effects": list(outcome.effects),
                        "details": dict(outcome.details),
                    })
                    if not outcome.success:
                        break
                physical_goal_satisfied = bool(
                    runtime.goal_verifier(runtime.observe_state())
                )
                result_payload = {
                    **result.as_dict(),
                    "action_history": action_history,
                    "physical_goal_satisfied": physical_goal_satisfied,
                }
        expected = load_expected("living_room", runtime.variant)
        comparison = compare_actions(planned_actions, expected["actions"])
        predicted_outcome = (
            "FEASIBLE" if horizon is not None and horizon.success else
            "UNRESOLVED" if horizon is not None else
            "FEASIBLE" if result.status == "PLAN" else "INFEASIBLE"
            if result.status == "NO_PLAN" else "UNRESOLVED"
        )
        comparison.update(
            {
                "variant": runtime.variant,
                "predicted_outcome": predicted_outcome,
                "expected_outcome": expected["intended_outcome"],
                "outcome_match": predicted_outcome == expected["intended_outcome"],
                "gt_was_model_input": False,
            }
        )
        payload = {
            "baseline": "owl_tamp_paper_derived",
            "official_author_code": False,
            "environment": "living_room",
            "variant": runtime.variant,
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
            "physical_execution": bool(args.physical_execution),
            "result": result_payload,
            "gt_comparison": comparison,
        }
        if horizon is None:
            write_json(output / "model_trace.json", planner.trace)
        write_json(output / "episode_result.json", payload)
        if args.physical_execution:
            if horizon is not None:
                executed_actions = horizon.executed_actions
                physical_goal_satisfied = bool(horizon.success)
            write_execution_result(
                output,
                scene="living_room",
                method="owl_tamp",
                protocol=args.protocol,
                variant=runtime.variant,
                camera_count=args.camera_count,
                seed=config.seed,
                success=physical_goal_satisfied,
                executed_actions=executed_actions,
                model_calls=planning_rounds,
                raw_vlm_requests=raw_vlm_requests,
                replans=replans,
                planning_latency_s=float(
                    (planner.trace or {}).get("latency_ms", 0.0)
                ) / 1000.0,
                elapsed_seconds=time.monotonic() - started_at,
                # terminal_status is compared across methods, so on a physical
                # run it has to describe how execution ended.  result.status
                # only reports that planning produced a plan ("PLAN"), which
                # would sit in the same column as VLM-TAMP's execution-loop
                # statuses and read as a success.
                terminal_status=(
                    horizon.status
                    if horizon is not None
                    else physical_terminal_status(
                        result.status, physical_goal_satisfied, action_history
                    )
                ),
            )
        print("[OWL-TAMP refined plan]", flush=True)
        for index, action in enumerate(planned_actions, start=1):
            aliases = runtime.aliases
            shown = [f"{item} ({aliases[item]})" if item in aliases else item for item in action.arguments]
            print(f"  {index}. {action.operator} {', '.join(shown)}", flush=True)
        print("[GT comparison]", json.dumps(comparison, sort_keys=True), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
