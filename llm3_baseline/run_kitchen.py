"""Run an LLM3-style planner through live Google-robot kitchen execution."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.inference import PlanningError
from mujoco_scenes.baseline_kitchen_runtime import BaselineKitchenRuntime
from mujoco_scenes.kitchen_execution_bundle import DEFAULT_TASK

from .execution import LLM3MuJoCoExecutor
from .executive import LLM3Executive, ObservationFrame
from .planner import LLM3Planner, PlannerConfig
from .prompt import PROMPT_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-run-dir", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--task-requirements", default=str(DEFAULT_TASK))
    parser.add_argument("--output-dir")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="paper",
        help="The original LLM3 planner uses temperature 0 without thinking.",
    )
    parser.add_argument("--max-model-calls", type=int, default=10)
    parser.add_argument("--max-total-actions", type=int, default=80)
    parser.add_argument("--camera", default="free")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--close-on-complete", action="store_true")
    return parser


def _arguments_text(arguments: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in arguments.items())


def _print_plan(call: int, plan: Any) -> None:
    print(f"[llm3 plan {call}] {plan.status}", flush=True)
    for index, action in enumerate(plan.actions, start=1):
        parameters = (
            f" | params: {_arguments_text(action.parameters)}"
            if action.parameters
            else ""
        )
        print(
            f"  {index}. {action.skill} "
            f"{_arguments_text(action.arguments)}{parameters}".rstrip(),
            flush=True,
        )


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    phase1 = Path(arguments.phase1_run_dir).resolve()
    output = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else phase1 / "llm3_e2e"
    )
    try:
        output = prepare_run_directory(output)
    except ValueError as error:
        parser.error(str(error))
    config = PlannerConfig.from_env()
    overrides = {
        name: value
        for name, value in {
            "base_url": arguments.base_url,
            "model": arguments.model,
            "max_tokens": arguments.max_tokens,
        }.items()
        if value is not None
    }
    config = replace(config, **overrides)
    if arguments.decoding == "paper":
        if not config.toggle_thinking and config.enable_thinking:
            parser.error(
                "The selected checkpoint has mandatory thinking; use "
                "--decoding model-native or a toggleable checkpoint."
            )
        config = replace(
            config,
            enable_thinking=False,
            sampling={"temperature": 0.0, "top_p": 1.0},
        )
    runtime = BaselineKitchenRuntime.from_phase1(
        phase1,
        output,
        task_requirements=arguments.task_requirements,
        viewer_camera=arguments.camera,
        show_viewer=not arguments.headless,
    )

    def observe() -> ObservationFrame:
        observation, images = runtime.observe()
        return ObservationFrame(observation, images)

    planner_backend = LLM3Planner(config)

    class LivePlanner:
        calls = 0

        def plan(self, *args: Any, **kwargs: Any):
            self.calls += 1
            runtime.sync("Waiting for LLM3 VLM action plan")
            try:
                result = planner_backend.plan(*args, **kwargs)
            except PlanningError as error:
                write_json(
                    output / "model_calls" / f"{self.calls:04d}.json",
                    {
                        "error": str(error),
                        "observation_revision": args[1].revision,
                        "model_visible_input": {
                            "textualized_state": (
                                args[1].as_semantic_neutral_prompt_dict()
                            ),
                            "camera_ids": [image["camera"] for image in args[2]],
                            "semantic_labels_exposed": False,
                        },
                    },
                )
                runtime.sync("Rejected invalid LLM3 model output")
                raise
            _print_plan(self.calls, result.plan)
            failure = kwargs.get("failure")
            trace = result.as_dict()
            trace["observation_revision"] = args[1].revision
            trace["model_visible_input"] = {
                "textualized_state": args[1].as_semantic_neutral_prompt_dict(),
                "camera_ids": [image["camera"] for image in args[2]],
                "semantic_labels_exposed": False,
            }
            trace["failure_feedback"] = (
                failure.as_dict() if failure is not None else None
            )
            write_json(output / "model_calls" / f"{self.calls:04d}.json", trace)
            runtime.sync("Validated LLM3 action plan")
            return result

    executor = LLM3MuJoCoExecutor(
        runtime.dispatcher,
        effect_sink=runtime.accept_effects,
        status_sink=runtime.sync,
    )
    executive = LLM3Executive(
        LivePlanner(),
        observe,
        executor,
        goal_verifier=runtime.goal_verifier,
        state_observer=runtime.observe_state,
        max_model_calls=arguments.max_model_calls,
        max_total_actions=arguments.max_total_actions,
    )
    write_json(
        output / "method_manifest.json",
        {
            "method": "LLM3 algorithm port",
            "official_revision": "aca6f0c1ed5f7319b48b44523e4b317a15b3861f",
            "prompt_version": PROMPT_VERSION,
            "continuous_parameter_schema_version": 1,
            "model": config.model,
            "decoding": arguments.decoding,
            "sampling": dict(config.sampling),
            "thinking_enabled": config.enable_thinking,
            "trace_size": executive.trace_size,
            "max_model_calls": arguments.max_model_calls,
            "max_total_actions": arguments.max_total_actions,
            "manipulable_object_count": len(runtime.bundle.inventory["objects"]),
            "manipulable_object_ids": sorted(runtime.phase_b.inventory_by_id),
            "planner_input_contract": (
                "FIVE_ID_ANNOTATED_RGB_PLUS_SEMANTIC_NEUTRAL_TEXT_STATE"
            ),
            "semantic_detector_outputs_exposed": False,
            "instance_correspondence_is_oracle": True,
            "embodiment": "MuJoCo Google Robot adaptation",
        },
    )
    try:
        runtime.open()
        print(
            "[llm3] Starting full discrete+continuous VLM plan -> "
            "motion-feedback replanning -> physical execution",
            flush=True,
        )
        result = executive.run(arguments.goal)
        payload = {
            "baseline": "llm3",
            "goal": arguments.goal,
            "model": config.model,
            "protocol": "llm3_full_plan_continuous_parameters",
            "decoding": arguments.decoding,
            "execution_started": True,
            "result": result.as_dict(),
            "observed_effects": sorted(runtime.ledger.effects),
        }
        write_json(output / "episode_result.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        runtime.sync("Goal complete" if result.success else f"Stopped: {result.status}")
        if not arguments.close_on_complete:
            runtime.wait_for_viewer()
        if not result.success:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
