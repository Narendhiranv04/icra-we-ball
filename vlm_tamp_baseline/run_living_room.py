"""Run planning-only VLM-TAMP on one Living Room variant and compare GT."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import time

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.inference import PlanningError
from baseline_common.living_room_execution import (
    LivingRoomPhysicalExecutor,
    build_living_room_physical_runtime,
)
from baseline_common.physical_benchmark import write_execution_result

from .executive import ObservationFrame, VLMTAMPExecutive
from .failure_feedback import model_failure_feedback
from .living_room_runtime import (
    DEFAULT_EXPECTED_ROOT,
    DEFAULT_PHASE1_ROOT,
    LivingRoomPlanningRuntime,
    LivingRoomSymbolicExecutor,
    canonical_actions,
    compare_action_sequences,
)
from .pddlstream_dependency import PDDLSTREAM_COMMIT
from .pddlstream_refiner import (
    LivingRoomGeometryOracle,
    PDDLStreamProtocol,
    PDDLStreamSubgoalRefiner,
)
from .planner import VLMTAMPPlanner, VLMTAMPPlannerConfig
from .prompt import PROMPT_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="L1-L10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal", help="Defaults to the frozen Living Room goal")
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="model-native",
        help=(
            "'paper' reproduces this baseline's own published condition "
            "(temperature 0.2, top_p 1.0, thinking disabled).  'model-native' "
            "uses the served checkpoint's published sampling with thinking "
            "enabled.  Both baselines must run the same choice or the "
            "comparison measures decoding rather than method."
        ),
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=1,
        help=(
            "Initial-plan comparison defaults to one two-stage VLM call. "
            "Increase only for a separately reported symbolic-reprompt condition."
        ),
    )
    parser.add_argument("--max-total-actions", type=int, default=40)
    parser.add_argument("--pddl-timeout", type=float, default=60.0)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument(
        "--physical-execution",
        action="store_true",
        help=(
            "Execute the refined actions through the calibrated Google-robot "
            "Living Room skills instead of the symbolic executor."
        ),
    )
    return parser


def _display(value: str, aliases: Mapping[str, str]) -> str:
    alias = aliases.get(value)
    return value if not alias else f"{value} ({alias})"


def _argument_text(arguments: Mapping[str, str], aliases: Mapping[str, str]) -> str:
    return ", ".join(
        f"{key}={_display(value, aliases)}" for key, value in arguments.items()
    )


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        output = prepare_run_directory(arguments.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if arguments.physical_execution:
        # The physical runtime owns its own scene/robot construction; its
        # phase-1 and expected-GT roots are the shared defaults.
        runtime = build_living_room_physical_runtime(
            arguments.variant,
            output,
            camera_count=arguments.camera_count,
            show_viewer=False,
        )
    else:
        runtime = LivingRoomPlanningRuntime(
            arguments.variant,
            output,
            phase1_root=arguments.phase1_root.resolve(),
            expected_root=arguments.expected_root.resolve(),
            image_width=arguments.image_width,
            image_height=arguments.image_height,
            camera_count=arguments.camera_count,
        )
    goal = arguments.goal or runtime.goal
    config = VLMTAMPPlannerConfig.from_env()
    overrides = {
        "max_tokens": arguments.max_tokens,
        "seed": arguments.seed,
        **({"base_url": arguments.base_url} if arguments.base_url else {}),
        **({"model": arguments.model} if arguments.model else {}),
    }
    config = replace(config, **overrides)
    if arguments.decoding == "paper":
        if not config.toggle_thinking and config.enable_thinking:
            raise SystemExit(
                "Paper decoding requires a checkpoint with toggleable thinking."
            )
        config = replace(
            config,
            enable_thinking=False,
            sampling={"temperature": 0.2, "top_p": 1.0},
        )

    planner_backend = VLMTAMPPlanner(config)

    class TracedPlanner:
        calls = 0
        raw_vlm_requests = 0
        planning_latency_s = 0.0

        def plan(self, *args: Any, **kwargs: Any):
            self.calls += 1
            observation = args[1]
            images = args[2]
            try:
                result = planner_backend.plan(*args, **kwargs)
            except PlanningError as error:
                self.raw_vlm_requests += len(planner_backend.request_trace)
                write_json(
                    output / "model_calls" / f"{self.calls:04d}.json",
                    {
                        "status": "INVALID_VLM_OUTPUT",
                        "error": str(error),
                        "model_visible_input": {
                            "textualized_state": observation.as_annotated_prompt_dict(),
                            "camera_ids": [item["camera"] for item in images],
                            "semantic_labels_exposed": True,
                        },
                        "model_requests": planner_backend.request_trace,
                        "model_responses": planner_backend.response_trace,
                    },
                )
                print(f"[vlm-tamp model {self.calls}] invalid: {error}", flush=True)
                raise
            self.raw_vlm_requests += len(planner_backend.request_trace)
            self.planning_latency_s += result.latency_ms / 1000.0
            print(f"[vlm-tamp English goals {self.calls}]", flush=True)
            for index, step in enumerate(result.english_plan.steps, start=1):
                print(f"  {index}. {step}", flush=True)
            print(
                f"[vlm-tamp formal subgoals {self.calls}] {result.plan.status}",
                flush=True,
            )
            for index, subgoal in enumerate(result.plan.subgoals, start=1):
                print(
                    f"  {index}. {subgoal.predicate} "
                    f"{_argument_text(subgoal.arguments, runtime.aliases)}".rstrip(),
                    flush=True,
                )
            trace = result.as_dict()
            trace["model_visible_input"] = {
                "textualized_state": observation.as_annotated_prompt_dict(),
                "camera_ids": [item["camera"] for item in images],
                "semantic_labels_exposed": True,
            }
            failure = kwargs.get("failure")
            trace["failure_feedback"] = model_failure_feedback(failure)
            trace["model_requests"] = planner_backend.request_trace
            trace["model_responses"] = planner_backend.response_trace
            write_json(output / "model_calls" / f"{self.calls:04d}.json", trace)
            return result

    def observe() -> ObservationFrame:
        observation, images = runtime.observe()
        return ObservationFrame(observation, images)

    refiner = PDDLStreamSubgoalRefiner(
        runtime.inventory,
        LivingRoomGeometryOracle(runtime.inventory, runtime.region_registry),
        protocol=PDDLStreamProtocol(timeout_seconds=arguments.pddl_timeout),
    )
    refinement_count = 0

    def record_refinement(subgoal: Any, actions: tuple[Any, ...]) -> None:
        nonlocal refinement_count
        refinement_count += 1
        print(f"[vlm-tamp PDDLStream actions] {subgoal.predicate}", flush=True)
        for index, action in enumerate(actions, start=1):
            print(
                f"  {index}. {action.skill} "
                f"{_argument_text(action.arguments, runtime.aliases)}".rstrip(),
                flush=True,
            )
        write_json(
            output / "refinements" / f"{refinement_count:04d}.json",
            {
                "subgoal": subgoal.as_dict(),
                "actions": [action.as_dict() for action in actions],
                "tamp_trace": refiner.last_trace,
            },
        )

    write_json(
        output / "method_manifest.json",
        {
            "method": "VLM-TAMP algorithm port",
            "environment": "living_room",
            "evaluation_mode": "PLANNING_ONLY_GT_SEQUENCE_COMPARISON",
            "physical_execution": False,
            "symbolic_rollout": True,
            "model_calls_condition": (
                "INITIAL_PLAN_ONLY"
                if arguments.max_model_calls == 1
                else "SYMBOLIC_REPROMPT_ABLATION"
            ),
            "prompt_version": PROMPT_VERSION,
            "model": config.model,
            "decoding": arguments.decoding,
            "sampling": dict(config.sampling),
            "thinking_enabled": config.enable_thinking,
            "seed": config.seed,
            "camera_count": arguments.camera_count,
            "refiner": "pddlstream",
            "pddlstream_revision": PDDLSTREAM_COMMIT,
            "planner_input_contract": "ALIAS_ANNOTATED_RGB_PLUS_OBSERVABLE_ALIAS_ID_MAP",
            "gt_visible_to_model": False,
            "semantic_labels_visible_to_model": True,
            "private_geometry_contains_functional_assignment": False,
            "comparison_target": str(
                arguments.expected_root.resolve()
                / runtime.variant
                / "expected_gt_actions.json"
            ),
        },
    )
    planner = TracedPlanner()
    executor = (
        LivingRoomPhysicalExecutor(runtime, effect_sink=None, status_sink=None)
        if arguments.physical_execution
        else LivingRoomSymbolicExecutor(runtime)
    )
    started_at = time.monotonic()
    executive = VLMTAMPExecutive(
        planner,
        observe,
        executor,
        refiner=refiner,
        goal_verifier=runtime.goal_verifier,
        state_observer=runtime.observe_state,
        refinement_sink=record_refinement,
        max_model_calls=arguments.max_model_calls,
        max_total_actions=arguments.max_total_actions,
    )
    try:
        print(
            f"[living-room] {runtime.variant}: {arguments.camera_count}-view VLM-TAMP -> "
            "PDDLStream -> symbolic rollout -> GT comparison",
            flush=True,
        )
        result = executive.run(goal)
        predicted = canonical_actions(result.action_history)
        predicted_outcome = "FEASIBLE" if result.success else "UNRESOLVED"
        if (
            not result.success
            and result.terminal_failure is not None
            and result.terminal_failure.code == "no_valid_subgoals"
        ):
            predicted_outcome = "INFEASIBLE"
            predicted.append(
                {
                    "operator": "TERMINATE_INFEASIBLE",
                    "arguments": ["NO_VALID_SUBGOALS"],
                }
            )
        comparison = compare_action_sequences(predicted, runtime.expected.actions)
        comparison.update(
            {
                "variant": runtime.variant,
                "predicted_outcome": predicted_outcome,
                "expected_outcome": runtime.expected.intended_outcome,
                "outcome_match": predicted_outcome == runtime.expected.intended_outcome,
                "gt_was_model_input": False,
            }
        )
        payload = {
            "baseline": "vlm_tamp",
            "environment": "living_room",
            "variant": runtime.variant,
            "goal": goal,
            # The serving model belongs in the episode result and not only in
            # method_manifest.json: the OWL runner records it here, so a
            # summary keyed on episode_result.json would otherwise read this
            # method's backbone as null and pool two models into one row.
            "model": config.model,
            "seed": config.seed,
            "camera_count": arguments.camera_count,
            "planning_rounds": result.model_calls,
            # Counted from the transport, not assumed: a round that stops after
            # the English stage issues one request, not the usual two.
            "raw_vlm_requests": planner.raw_vlm_requests,
            "protocol": arguments.protocol,
            "execution_started": bool(arguments.physical_execution),
            "physical_execution": bool(arguments.physical_execution),
            "result": result.as_dict(),
            "gt_comparison": comparison,
        }
        write_json(output / "episode_result.json", payload)
        if arguments.physical_execution and result.status != "INFERENCE_FAILED":
            # An unreachable model server is an infrastructure fault, not an
            # execution trial.  Recording it would put a 0% row in the physical
            # table for an episode the robot never attempted.
            write_execution_result(
                output,
                scene="living_room",
                method="vlm_tamp",
                protocol=arguments.protocol,
                variant=runtime.variant,
                camera_count=arguments.camera_count,
                seed=config.seed,
                success=bool(result.success and runtime.goal_verifier()),
                executed_actions=result.executed_actions,
                model_calls=result.model_calls,
                raw_vlm_requests=planner.raw_vlm_requests,
                replans=result.reprompts,
                planning_latency_s=planner.planning_latency_s,
                elapsed_seconds=time.monotonic() - started_at,
                terminal_status=result.status,
                terminal_failure=(
                    result.terminal_failure.as_dict()
                    if result.terminal_failure
                    else None
                ),
            )
        write_json(output / "gt_sequence_comparison.json", comparison)
        print("[GT sequence comparison]", flush=True)
        print(
            "  outcome: "
            f"{predicted_outcome} vs {runtime.expected.intended_outcome} "
            f"(match={comparison['outcome_match']})",
            flush=True,
        )
        print(
            "  sequence: "
            f"exact={comparison['exact_sequence_match']}, "
            f"LCS={comparison['lcs_action_count']}/"
            f"{comparison['expected_action_count']}, "
            f"F1={comparison['ordered_f1']:.3f}",
            flush=True,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
