"""Run VLM-TAMP with either symbolic GT evaluation or live Kitchen execution."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.inference import PlanningError
from baseline_common.physical_benchmark import write_execution_result
from mujoco_scenes.baseline_kitchen_runtime import BaselineKitchenRuntime
from mujoco_scenes.kitchen_execution_bundle import DEFAULT_TASK

from .execution import VLMTAMPMuJoCoExecutor
from .executive import ObservationFrame, VLMTAMPExecutive
from .failure_feedback import model_failure_feedback
from .kitchen_planning_runtime import (
    DEFAULT_EXPECTED_ROOT,
    KitchenPlanningState,
    canonical_kitchen_actions,
    compare_kitchen_actions,
    load_expected,
)
from .planner import VLMTAMPPlanner, VLMTAMPPlannerConfig
from .pddlstream_refiner import (
    KitchenGeometryOracle,
    PDDLStreamSubgoalRefiner,
)
from .pddlstream_dependency import PDDLSTREAM_COMMIT
from .prompt import PROMPT_VERSION
from .refiner import CatalogSubgoalRefiner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-run-dir")
    parser.add_argument("--goal")
    parser.add_argument("--task-requirements", default=str(DEFAULT_TASK))
    parser.add_argument("--output-dir")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument(
        "--decoding",
        choices=("paper", "model-native"),
        default="paper",
        help="Paper uses temperature 0.2 without a thinking mode.",
    )
    parser.add_argument(
        "--max-model-calls",
        type=int,
        help="Defaults to 1 in planning-only mode and 3 in physical mode.",
    )
    parser.add_argument("--max-total-actions", type=int, default=80)
    parser.add_argument("--camera", default="free")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--close-on-complete", action="store_true")
    parser.add_argument(
        "--planning-only",
        action="store_true",
        help="Symbolically roll out the refined actions and compare them with GT.",
    )
    parser.add_argument("--variant", choices=tuple(f"K{i}" for i in range(1, 13)))
    parser.add_argument(
        "--physical-variant",
        choices=tuple(f"K{i}" for i in range(1, 13)),
        help=(
            "Run a benchmark Kitchen variant through the physical Google-robot "
            "adapter. The expected terminal relations remain evaluator-private."
        ),
    )
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument(
        "--refiner",
        choices=("pddlstream", "catalog-ablation"),
        default="pddlstream",
        help="Use the paper TAMP backend or the explicitly labeled fixed-template ablation.",
    )
    return parser


def _display_value(value: str, aliases: Mapping[str, str] | None = None) -> str:
    label = None if aliases is None else aliases.get(value)
    if not label or label == value:
        return value
    return f"{value} ({label})"


def _arguments_text(
    arguments: Mapping[str, str], aliases: Mapping[str, str] | None = None
) -> str:
    return ", ".join(
        f"{key}={_display_value(value, aliases)}"
        for key, value in arguments.items()
    )


def _print_subgoal_plan(
    call: int, plan: Any, aliases: Mapping[str, str] | None = None
) -> None:
    print(f"[vlm-tamp subgoals {call}] {plan.status}", flush=True)
    for index, subgoal in enumerate(plan.subgoals, start=1):
        print(
            f"  {index}. {subgoal.predicate} "
            f"{_arguments_text(subgoal.arguments, aliases)}".rstrip(),
            flush=True,
        )


def _print_refinement(
    subgoal: Any,
    actions: tuple[Any, ...],
    aliases: Mapping[str, str] | None = None,
) -> None:
    print(f"[vlm-tamp actions] {subgoal.predicate}", flush=True)
    for index, action in enumerate(actions, start=1):
        print(
            f"  {index}. {action.skill} "
            f"{_arguments_text(action.arguments, aliases)}".rstrip(),
            flush=True,
        )


def _method_manifest(
    arguments: argparse.Namespace,
    config: VLMTAMPPlannerConfig,
    *,
    max_model_calls: int,
    inventory_object_count: int,
) -> dict[str, Any]:
    """Describe the configured condition before any model call is made."""
    return {
        "method": "VLM-TAMP algorithm port",
        "prompt_version": PROMPT_VERSION,
        "model": config.model,
        "camera_count": arguments.camera_count,
        "seed": config.seed,
        "decoding": arguments.decoding,
        "sampling": dict(config.sampling),
        "thinking_enabled": config.enable_thinking,
        "refiner": arguments.refiner,
        "pddlstream_revision": (
            PDDLSTREAM_COMMIT if arguments.refiner == "pddlstream" else None
        ),
        "max_model_calls": max_model_calls,
        "max_total_actions": arguments.max_total_actions,
        "vlm_object_scope": "OBSERVED_OBJECTS_ACCUMULATED_DURING_EPISODE",
        "planner_input_contract": (
            "ALIAS_ANNOTATED_RGB_PLUS_OBSERVABLE_ALIAS_ID_MAP"
        ),
        "semantic_labels_visible_to_model": True,
        "semantic_detector_outputs_exposed": False,
        "instance_correspondence_is_oracle": True,
        "private_refinement_inventory_count": inventory_object_count,
        "embodiment": "MuJoCo Google Robot adaptation",
        "evaluation_mode": (
            "PLANNING_ONLY_GT_SEQUENCE_COMPARISON"
            if arguments.planning_only
            else "PHYSICAL_EXECUTION"
        ),
        "physical_execution": not arguments.planning_only,
        "gt_visible_to_model": False,
        "planning_rounds": max_model_calls,
        "raw_vlm_requests_per_round": 2,
        "protocol": arguments.protocol,
    }


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.planning_only and arguments.variant is None:
        parser.error("--planning-only requires --variant K1-K12")
    if arguments.planning_only and arguments.physical_variant is not None:
        parser.error("--planning-only cannot be combined with --physical-variant")
    if not arguments.planning_only and arguments.variant is not None:
        parser.error("--variant is only used with --planning-only")
    if (
        not arguments.planning_only
        and not arguments.phase1_run_dir
        and arguments.physical_variant is None
    ):
        parser.error("physical mode requires --phase1-run-dir or --physical-variant")
    if arguments.phase1_run_dir and arguments.physical_variant is not None:
        parser.error("choose either --phase1-run-dir or --physical-variant")
    if arguments.planning_only and arguments.phase1_run_dir:
        parser.error(
            "planning-only K1-K12 trials construct the variant directly; "
            "do not pass --phase1-run-dir"
        )
    max_model_calls = arguments.max_model_calls or (
        1 if arguments.planning_only else 3
    )
    if arguments.planning_only and max_model_calls != 1:
        parser.error(
            "Planning-only GT comparison requires exactly one planning round; "
            "use physical mode for failure-triggered replanning."
        )
    if arguments.protocol == "single_call" and max_model_calls != 1:
        parser.error("single_call protocol requires exactly one planning round")
    phase1 = (
        Path(arguments.phase1_run_dir).resolve()
        if arguments.phase1_run_dir
        else None
    )
    output = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else phase1 / "vlm_tamp_e2e" if phase1 is not None else None
    )
    if output is None:
        parser.error("planning-only mode requires --output-dir")
    try:
        output = prepare_run_directory(output)
    except ValueError as error:
        parser.error(str(error))
    config = VLMTAMPPlannerConfig.from_env()
    overrides = {
        name: value
        for name, value in {
            "base_url": arguments.base_url,
            "model": arguments.model,
            "max_tokens": arguments.max_tokens,
            "seed": arguments.seed,
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
            sampling={"temperature": 0.2, "top_p": 1.0},
        )
    if arguments.planning_only:
        runtime = BaselineKitchenRuntime.from_variant(
            arguments.variant,
            output,
            camera_count=arguments.camera_count,
            viewer_camera=arguments.camera,
            show_viewer=False,
        )
    elif arguments.physical_variant is not None:
        expected_actions = (
            arguments.expected_root.resolve()
            / arguments.physical_variant
            / "expected_gt_actions.json"
        )
        runtime = BaselineKitchenRuntime.from_variant(
            arguments.physical_variant,
            output,
            expected_actions=expected_actions,
            camera_count=arguments.camera_count,
            viewer_camera=arguments.camera,
            show_viewer=not arguments.headless,
        )
    else:
        runtime = BaselineKitchenRuntime.from_phase1(
            phase1,
            output,
            task_requirements=arguments.task_requirements,
            camera_count=arguments.camera_count,
            viewer_camera=arguments.camera,
            show_viewer=not arguments.headless,
        )
    goal = arguments.goal or str(runtime.scene.config.goal)
    planning_state = KitchenPlanningState(runtime) if arguments.planning_only else None

    def observe() -> ObservationFrame:
        observation, images = (
            planning_state.observe() if planning_state is not None else runtime.observe()
        )
        return ObservationFrame(observation, images)

    planner_backend = VLMTAMPPlanner(config)
    observed_aliases: dict[str, str] = {}

    class LivePlanner:
        calls = 0
        planning_latency_s = 0.0
        raw_vlm_requests = 0

        def plan(self, *args: Any, **kwargs: Any):
            runtime.sync("Waiting for VLM-TAMP subgoals")
            self.calls += 1
            observation = args[1]
            observed_aliases.update(
                {
                    item.entity_id: item.label
                    for item in observation.entities
                }
            )
            try:
                result = planner_backend.plan(*args, **kwargs)
            except PlanningError as error:
                self.raw_vlm_requests += len(planner_backend.request_trace)
                failure = kwargs.get("failure")
                write_json(
                    output / "model_calls" / f"{self.calls:04d}.json",
                    {
                        "status": "INVALID_VLM_OUTPUT",
                        "error": str(error),
                        "observation_revision": args[1].revision,
                        "model_visible_input": {
                            "textualized_state": (
                                args[1].as_annotated_prompt_dict()
                            ),
                            "camera_ids": [image["camera"] for image in args[2]],
                            "semantic_labels_exposed": True,
                        },
                        "failure_feedback": model_failure_feedback(failure),
                        "model_requests": planner_backend.request_trace,
                        "model_responses": planner_backend.response_trace,
                    },
                )
                print(
                    f"[vlm-tamp model output {self.calls}] invalid: {error}",
                    flush=True,
                )
                raise
            self.planning_latency_s += result.latency_ms / 1000.0
            self.raw_vlm_requests += len(planner_backend.request_trace)
            print(f"[vlm-tamp English goals {self.calls}]", flush=True)
            for index, step in enumerate(result.english_plan.steps, start=1):
                print(f"  {index}. {step}", flush=True)
            _print_subgoal_plan(self.calls, result.plan, observed_aliases)
            failure = kwargs.get("failure")
            trace = result.as_dict()
            trace["observation_revision"] = args[1].revision
            trace["model_visible_input"] = {
                "textualized_state": args[1].as_annotated_prompt_dict(),
                "camera_ids": [image["camera"] for image in args[2]],
                "semantic_labels_exposed": True,
            }
            trace["failure_feedback"] = model_failure_feedback(failure)
            trace["model_requests"] = planner_backend.request_trace
            trace["model_responses"] = planner_backend.response_trace
            write_json(output / "model_calls" / f"{self.calls:04d}.json", trace)
            runtime.sync("Validated VLM-TAMP subgoals")
            return result

    executor = (
        planning_state
        if planning_state is not None
        else VLMTAMPMuJoCoExecutor(
            runtime.dispatcher,
            effect_sink=runtime.accept_effects,
            status_sink=runtime.sync,
        )
    )
    inventory_object_count = len(runtime.bundle.inventory.get("objects", ()))
    if arguments.refiner == "pddlstream":
        refiner = PDDLStreamSubgoalRefiner(
            runtime.bundle.inventory,
            KitchenGeometryOracle(runtime.bundle.inventory, runtime),
        )
    else:
        refiner = CatalogSubgoalRefiner()
    write_json(
        output / "method_manifest.json",
        _method_manifest(
            arguments,
            config,
            max_model_calls=max_model_calls,
            inventory_object_count=inventory_object_count,
        ),
    )
    refinement_count = 0

    def record_refinement(subgoal: Any, actions: tuple[Any, ...]) -> None:
        nonlocal refinement_count
        refinement_count += 1
        _print_refinement(subgoal, actions, observed_aliases)
        trace = getattr(refiner, "last_trace", None)
        if not isinstance(trace, dict) or not trace:
            return
        final_attempt = trace.get("attempts", ())[-1]
        print("[vlm-tamp PDDLStream plan]", flush=True)
        for index, row in enumerate(final_attempt.get("pddl_plan", ()), start=1):
            arguments_text = ", ".join(
                _display_value(value, observed_aliases)
                for value in row.get("arguments", ())
            )
            print(
                f"  {index}. {str(row.get('operator', '')).upper()} "
                f"{arguments_text}".rstrip(),
                flush=True,
            )
        write_json(
            output / "refinements" / f"{refinement_count:04d}.json",
            {
                "subgoal": subgoal.as_dict(),
                "executable_actions": [item.as_dict() for item in actions],
                "tamp_trace": trace,
            },
        )
    live_planner = LivePlanner()
    executive = VLMTAMPExecutive(
        live_planner,
        observe,
        executor,
        refiner=refiner,
        goal_verifier=(
            planning_state.goal_verifier
            if planning_state is not None
            else runtime.goal_verifier
        ),
        state_observer=(
            planning_state.observe_state
            if planning_state is not None
            else runtime.observe_state
        ),
        object_universe=None,
        refinement_sink=record_refinement,
        max_model_calls=max_model_calls,
        max_total_actions=arguments.max_total_actions,
    )
    try:
        if not arguments.planning_only:
            runtime.open()
        episode_started = time.monotonic()
        print(
            "[vlm-tamp] Starting two-stage VLM subgoals -> "
            f"{arguments.refiner} refinement -> "
            + (
                "symbolic rollout -> GT comparison"
                if arguments.planning_only
                else "physical execution"
            ),
            flush=True,
        )
        result = executive.run(goal)
        payload = {
            "baseline": "vlm_tamp",
            "environment": "kitchen",
            "variant": arguments.variant or arguments.physical_variant,
            "goal": goal,
            "model": config.model,
            "seed": config.seed,
            "camera_count": arguments.camera_count,
            "protocol": arguments.protocol,
            "planning_rounds": result.model_calls,
            "raw_vlm_requests": live_planner.raw_vlm_requests,
            "refiner": arguments.refiner,
            "paper_protocol": arguments.refiner == "pddlstream",
            "decoding": arguments.decoding,
            "execution_started": not arguments.planning_only,
            "physical_execution": not arguments.planning_only,
            "result": result.as_dict(),
            "observed_effects": sorted(runtime.ledger.effects),
        }
        if arguments.planning_only:
            backend_by_id = {
                str(row["generic_object_id"]): str(row["physical_backend_body"])
                for row in runtime.bundle.resolution.get("accepted", ())
            }
            predicted = canonical_kitchen_actions(
                result.action_history, backend_by_id
            )
            expected = load_expected(
                arguments.expected_root.resolve(), arguments.variant
            )
            predicted_outcome = "FEASIBLE" if result.success else "UNRESOLVED"
            if (
                result.terminal_failure is not None
                and result.terminal_failure.code == "no_valid_subgoals"
            ):
                predicted_outcome = "INFEASIBLE"
            comparison = compare_kitchen_actions(
                predicted, expected.get("actions", ())
            )
            comparison.update(
                {
                    "variant": arguments.variant,
                    "predicted_outcome": predicted_outcome,
                    "expected_outcome": expected["intended_outcome"],
                    "outcome_match": (
                        predicted_outcome == expected["intended_outcome"]
                    ),
                    "gt_was_model_input": False,
                }
            )
            payload["gt_comparison"] = comparison
            write_json(output / "gt_sequence_comparison.json", comparison)
            print("[GT sequence comparison]", flush=True)
            shared = comparison["shared_task_vocabulary"]
            print(
                "  outcome: "
                f"{predicted_outcome} vs {expected['intended_outcome']} "
                f"(match={comparison['outcome_match']})",
                flush=True,
            )
            print(
                "  task-level sequence: "
                f"exact={shared['exact_sequence_match']}, "
                f"LCS={shared['lcs_action_count']}/"
                f"{shared['expected_action_count']}, "
                f"F1={shared['ordered_f1']:.3f}",
                flush=True,
            )
        write_json(output / "episode_result.json", payload)
        if not arguments.planning_only:
            write_execution_result(
                output,
                scene="kitchen",
                method="vlm_tamp",
                protocol=arguments.protocol,
                variant=str(arguments.physical_variant or "phase1_custom"),
                camera_count=arguments.camera_count,
                seed=config.seed,
                success=result.success,
                executed_actions=result.executed_actions,
                model_calls=result.model_calls,
                raw_vlm_requests=live_planner.raw_vlm_requests,
                replans=result.reprompts,
                planning_latency_s=live_planner.planning_latency_s,
                elapsed_seconds=time.monotonic() - episode_started,
                terminal_status=result.status,
                terminal_failure=(
                    result.terminal_failure.as_dict()
                    if result.terminal_failure is not None
                    else None
                ),
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        runtime.sync("Goal complete" if result.success else f"Stopped: {result.status}")
        if not arguments.planning_only and not arguments.close_on_complete:
            runtime.wait_for_viewer()
        if not arguments.planning_only and not result.success:
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
