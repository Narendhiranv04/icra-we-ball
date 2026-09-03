"""Run planning-only VLM-TAMP on one Workshop W1--W10 variant."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.inference import PlanningError

from .executive import ObservationFrame, VLMTAMPExecutive
from .failure_feedback import model_failure_feedback
from .pddlstream_dependency import PDDLSTREAM_COMMIT
from .pddlstream_refiner import PDDLStreamProtocol
from .planner import VLMTAMPPlanner, VLMTAMPPlannerConfig
from .prompt import PROMPT_VERSION
from .workshop_refiner import WorkshopPDDLStreamRefiner
from .workshop_runtime import (
    DEFAULT_EXPECTED_ROOT,
    WorkshopPlanningRuntime,
    WorkshopSymbolicExecutor,
    canonical_workshop_actions,
    compare_workshop_actions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="W1-W10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", choices=(1, 3, 5), type=int, default=5)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument("--pddl-timeout", type=float, default=60.0)
    parser.add_argument("--max-total-actions", type=int, default=48)
    parser.add_argument(
        "--max-model-calls", type=int, default=1,
        help="One initial plan by default; use >1 only for a separately reported replan condition.",
    )
    parser.add_argument("--decoding", choices=("paper", "model-native"), default="paper")
    return parser


def _shown(value: str, aliases: Mapping[str, str]) -> str:
    return f"{value} ({aliases[value]})" if value in aliases else value


def main() -> None:
    args = build_parser().parse_args()
    if args.max_model_calls < 1:
        raise SystemExit("--max-model-calls must be positive")
    try:
        output = prepare_run_directory(args.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    runtime = WorkshopPlanningRuntime(
        args.variant,
        output,
        expected_root=args.expected_root.resolve(),
        image_width=args.image_width,
        image_height=args.image_height,
        camera_count=args.camera_count,
    )
    config = VLMTAMPPlannerConfig.from_env()
    config = replace(
        config,
        max_tokens=args.max_tokens,
        seed=args.seed,
        **({"base_url": args.base_url} if args.base_url else {}),
        **({"model": args.model} if args.model else {}),
    )
    if args.decoding == "paper":
        if not config.toggle_thinking and config.enable_thinking:
            raise SystemExit("Paper decoding requires a checkpoint with toggleable thinking.")
        config = replace(config, enable_thinking=False, sampling={"temperature": 0.2, "top_p": 1.0})
    goal = args.goal or runtime.goal
    planner_backend = VLMTAMPPlanner(config)

    class TracedPlanner:
        calls = 0
        raw_vlm_requests = 0

        def plan(self, *values: Any, **kwargs: Any):
            self.calls += 1
            observation, images = values[1], values[2]
            try:
                result = planner_backend.plan(*values, **kwargs)
            except PlanningError as error:
                self.raw_vlm_requests += len(planner_backend.request_trace)
                write_json(output / "model_calls" / f"{self.calls:04d}.json", {
                    "status": "INVALID_VLM_OUTPUT", "error": str(error),
                    "model_visible_input": {"textualized_state": observation.as_annotated_prompt_dict(), "camera_ids": [item["camera"] for item in images]},
                    "model_requests": planner_backend.request_trace, "model_responses": planner_backend.response_trace,
                })
                print(f"[vlm-tamp model {self.calls}] invalid: {error}", flush=True)
                raise
            self.raw_vlm_requests += len(planner_backend.request_trace)
            print(f"[vlm-tamp English goals {self.calls}]", flush=True)
            for index, step in enumerate(result.english_plan.steps, 1):
                print(f"  {index}. {step}", flush=True)
            print(f"[vlm-tamp formal subgoals {self.calls}] {result.plan.status}", flush=True)
            for index, subgoal in enumerate(result.plan.subgoals, 1):
                text = ", ".join(f"{name}={_shown(value, runtime.aliases)}" for name, value in subgoal.arguments.items())
                print(f"  {index}. {subgoal.predicate} {text}", flush=True)
            trace = result.as_dict()
            trace["model_visible_input"] = {"textualized_state": observation.as_annotated_prompt_dict(), "camera_ids": [item["camera"] for item in images], "semantic_labels_exposed": True}
            trace["failure_feedback"] = model_failure_feedback(kwargs.get("failure"))
            trace["model_requests"] = planner_backend.request_trace
            trace["model_responses"] = planner_backend.response_trace
            write_json(output / "model_calls" / f"{self.calls:04d}.json", trace)
            return result

    def observe() -> ObservationFrame:
        observation, images = runtime.observe()
        return ObservationFrame(observation, images)

    refiner = WorkshopPDDLStreamRefiner(runtime, protocol=PDDLStreamProtocol(timeout_seconds=args.pddl_timeout))
    refinement_count = 0

    def record_refinement(subgoal: Any, actions: tuple[Any, ...]) -> None:
        nonlocal refinement_count
        refinement_count += 1
        print(f"[vlm-tamp PDDLStream actions] {subgoal.predicate}", flush=True)
        for index, action in enumerate(actions, 1):
            text = ", ".join(f"{name}={_shown(value, runtime.aliases)}" for name, value in action.arguments.items())
            print(f"  {index}. {action.skill} {text}", flush=True)
        write_json(output / "refinements" / f"{refinement_count:04d}.json", {
            "subgoal": subgoal.as_dict(), "actions": [action.as_dict() for action in actions], "tamp_trace": refiner.last_trace,
        })

    write_json(output / "method_manifest.json", {
        "method": "VLM-TAMP algorithm port", "environment": "workshop",
        "evaluation_mode": "PLANNING_ONLY_GT_SEQUENCE_COMPARISON", "physical_execution": False,
        "prompt_version": PROMPT_VERSION, "model": config.model, "seed": config.seed,
        "camera_count": args.camera_count, "decoding": args.decoding, "sampling": dict(config.sampling),
        "thinking_enabled": config.enable_thinking, "model_calls_condition": "INITIAL_PLAN_ONLY" if args.max_model_calls == 1 else "SYMBOLIC_REPROMPT_ABLATION",
        "refiner": "pddlstream", "pddlstream_revision": PDDLSTREAM_COMMIT,
        "planner_input_contract": "ALIAS_ANNOTATED_RGB_PLUS_OBSERVABLE_ALIAS_ID_MAP",
        "hidden_storage_contents_visible_to_model": False, "gt_visible_to_model": False,
        "comparison_target": str(args.expected_root.resolve() / runtime.variant / "expected_gt_actions.json"),
    })
    planner = TracedPlanner()
    executive = VLMTAMPExecutive(
        planner, observe, WorkshopSymbolicExecutor(runtime), refiner=refiner,
        goal_verifier=runtime.goal_verifier, state_observer=runtime.observe_state,
        refinement_sink=record_refinement, max_model_calls=args.max_model_calls,
        max_total_actions=args.max_total_actions,
    )
    try:
        print(f"[workshop] {runtime.variant}: {args.camera_count}-view VLM-TAMP -> PDDLStream -> symbolic rollout -> GT comparison", flush=True)
        result = executive.run(goal)
        runtime.observe_state()
        backend_by_id = {**runtime.object_by_backend, **runtime.region_by_backend}
        predicted = canonical_workshop_actions(result.action_history, backend_by_id)
        predicted_outcome = (
            "FEASIBLE" if result.success
            else "INFEASIBLE" if runtime.infeasibility_proven()
            else "UNRESOLVED"
        )
        comparison = compare_workshop_actions(predicted, runtime.expected.actions)
        comparison.update({
            "variant": runtime.variant, "predicted_outcome": predicted_outcome,
            "expected_outcome": runtime.expected.intended_outcome,
            "outcome_match": predicted_outcome == runtime.expected.intended_outcome,
            "gt_was_model_input": False,
        })
        payload = {
            "baseline": "vlm_tamp", "environment": "workshop", "variant": runtime.variant,
            "goal": goal, "model": config.model, "seed": config.seed, "camera_count": args.camera_count,
            "planning_rounds": result.model_calls,
            # Counted from the transport, not assumed: a round that stops after
            # the English stage issues one request, not the usual two.
            "raw_vlm_requests": planner.raw_vlm_requests,
            "protocol": args.protocol,
            "execution_started": False, "physical_execution": False, "result": result.as_dict(), "gt_comparison": comparison,
        }
        write_json(output / "episode_result.json", payload)
        write_json(output / "gt_sequence_comparison.json", comparison)
        print("[GT task-level comparison]", json.dumps(comparison["shared_task_vocabulary"], sort_keys=True), flush=True)
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
