"""Open-vocabulary retrieval baseline on one Living Room variant.

No language model is used.  The role structure is a fixed task template and
every role is filled by CLIP image-text similarity between the role's function
phrase and a crop from the raw camera frames.  Output matches the shared
planning-to-GT artifact contract so the standard batch runner and summarizer
consume it unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

from baseline_common.artifacts import prepare_run_directory, write_json
from baseline_common.living_room_execution import (
    LivingRoomPhysicalExecutor,
    build_living_room_physical_runtime,
)
from baseline_common.models import Action
from baseline_common.physical_benchmark import (
    physical_terminal_status,
    write_execution_result,
)
from vlm_tamp_baseline.living_room_runtime import (
    DEFAULT_EXPECTED_ROOT,
    DEFAULT_PHASE1_ROOT,
    LivingRoomPlanningRuntime,
    compare_action_sequences,
)

from .retrieval import (
    CROP_CONTEXT_FRACTION,
    CLIPRetriever,
    assign_distinct,
    read_annotations,
)
from .roles import LIVING_ROOM_ROLES, LIVING_ROOM_TASK_TEMPLATE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="L1-L10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--expected-root", type=Path, default=DEFAULT_EXPECTED_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--protocol", choices=("native", "single_call"), default="native")
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument(
        "--physical-execution",
        action="store_true",
        help=(
            "Execute the retrieved plan through the calibrated Google-robot "
            "Living Room skills instead of stopping at the plan.  Retrieval "
            "issues no model requests, so this isolates how far a purely "
            "similarity-grounded assignment gets on the real robot."
        ),
    )
    # Accepted for batch-runner compatibility; this baseline calls no model.
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens", type=int, help=argparse.SUPPRESS)
    return parser


def _plan(assignment: list[tuple[str, str]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for object_id, region_id in assignment:
        actions.append({"operator": "PICK", "arguments": [object_id]})
        actions.append({"operator": "PLACE", "arguments": [object_id, region_id]})
    return actions


def _role(key: str):
    return next(role for role in LIVING_ROOM_ROLES if role.key == key)


def _to_shared_action(row: dict[str, Any]) -> Action:
    """Map one retrieved PICK/PLACE row onto the shared executor contract."""
    operator = str(row["operator"]).upper()
    arguments = [str(value) for value in row["arguments"]]
    if operator == "PICK" and len(arguments) == 1:
        return Action("PICK", {"object_id": arguments[0]})
    if operator == "PLACE" and len(arguments) == 2:
        return Action("PLACE", {"object_id": arguments[0], "region_id": arguments[1]})
    raise ValueError(f"Retrieval produced an unexecutable action {row!r}")


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        output = prepare_run_directory(arguments.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    started_at = time.monotonic()
    if arguments.physical_execution:
        runtime = build_living_room_physical_runtime(
            arguments.variant,
            output,
            camera_count=arguments.camera_count,
            show_viewer=False,
            # Same render size as the planning path: CLIP scores the crops.
            image_width=arguments.image_width,
            image_height=arguments.image_height,
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
    try:
        runtime.observe()  # renders the raw and annotated frames
        observation_dir = output / "observations" / "initial"
        annotations = read_annotations(observation_dir)

        retriever = CLIPRetriever(device=arguments.clip_device)
        region_role_keys = ("personal_support", "shared_support")
        object_role_keys = ("drink_vessel", "under_dish", "handheld_control")
        # Candidate supports are the registered support regions the runtime
        # exposes to every baseline.  The staging area is where the payloads
        # start, not a placement target, and it is not a registered support -
        # electing it would let a missing table masquerade as a usable one.
        support_ids = sorted(runtime.region_backends)
        region_scores = retriever.score(
            annotations,
            observation_dir,
            [_role(key).phrase for key in region_role_keys],
            "region",
            candidate_ids=support_ids,
        )
        object_scores = retriever.score(
            annotations,
            observation_dir,
            [_role(key).phrase for key in object_role_keys],
            "object",
        )

        # Regions: personal supports first, then a distinct shared support.
        personals = assign_distinct(
            region_scores, _role("personal_support").phrase, 2, taken=()
        )
        shared = assign_distinct(
            region_scores, _role("shared_support").phrase, 1, taken=personals
        )
        vessels = assign_distinct(
            object_scores, _role("drink_vessel").phrase, 2, taken=()
        )
        dishes = assign_distinct(
            object_scores, _role("under_dish").phrase, 2, taken=vessels
        )
        controls = assign_distinct(
            object_scores,
            _role("handheld_control").phrase,
            1,
            taken=vessels + dishes,
        )

        selection = {
            "personal_support": personals,
            "shared_support": shared,
            "drink_vessel": vessels,
            "under_dish": dishes,
            "handheld_control": controls,
        }
        complete = (
            len(personals) == 2
            and len(shared) == 1
            and len(vessels) == 2
            and len(dishes) == 2
            and len(controls) == 1
        )

        assignment: list[tuple[str, str]] = []
        if complete:
            for vessel_index, dish_index, support_index in LIVING_ROOM_TASK_TEMPLATE:
                support = personals[support_index]
                assignment.append((vessels[vessel_index], support))
                assignment.append((dishes[dish_index], support))
            assignment.append((controls[0], shared[0]))

        predicted = _plan(assignment)
        predicted_outcome = "FEASIBLE" if complete else "INFEASIBLE"
        if not complete:
            predicted = [
                {"operator": "TERMINATE_INFEASIBLE", "arguments": ["NO_RETRIEVED_ROLE_FILLER"]}
            ]
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

        write_json(output / "retrieval_trace.json", {
            "schema_version": 1,
            "roles": [
                {"key": role.key, "phrase": role.phrase, "count": role.count, "kind": role.kind}
                for role in LIVING_ROOM_ROLES
            ],
            "selection": selection,
            "region_scores": region_scores.to_dict(),
            "object_scores": object_scores.to_dict(),
            "crops_taken_from": "raw_<camera>.png (unannotated)",
            "crop_context_fraction": CROP_CONTEXT_FRACTION,
            "candidate_support_ids": support_ids,
            "rejection_criterion": (
                "cardinality: a role is unfilled when the observable registered "
                "supports cannot supply enough distinct candidates"
            ),
        })
        status = "RETRIEVED" if complete else "NO_RETRIEVED_ROLE_FILLER"
        executed_actions = 0
        physical_goal_satisfied = False
        result_payload: dict[str, Any] = {
            "status": status,
            "actions": predicted,
            "selection": selection,
        }
        # Retrieval only reaches the robot when every role was actually filled;
        # its infeasible verdict is a TERMINATE_INFEASIBLE marker, not a plan.
        if arguments.physical_execution and complete:
            executor = LivingRoomPhysicalExecutor(runtime)
            action_history = []
            for row in predicted:
                outcome = executor.execute(_to_shared_action(row))
                executed_actions += 1
                action_history.append({
                    "action": row,
                    "success": outcome.success,
                    "failure_code": outcome.failure_code,
                    "message": outcome.message,
                    "effects": list(outcome.effects),
                })
                if not outcome.success:
                    break
            physical_goal_satisfied = bool(
                runtime.goal_verifier(runtime.observe_state())
            )
            result_payload.update({
                "action_history": action_history,
                "physical_goal_satisfied": physical_goal_satisfied,
            })
        payload = {
            "baseline": "retrieval",
            "environment": "living_room",
            "variant": runtime.variant,
            "goal": goal,
            "seed": arguments.seed,
            "camera_count": arguments.camera_count,
            "protocol": arguments.protocol,
            "uses_language_model": False,
            "planning_rounds": 0,
            "raw_vlm_requests": 0,
            "execution_started": bool(arguments.physical_execution and complete),
            "physical_execution": bool(arguments.physical_execution),
            "result": result_payload,
            "gt_comparison": comparison,
        }
        write_json(output / "episode_result.json", payload)
        write_json(output / "gt_sequence_comparison.json", comparison)
        if arguments.physical_execution:
            write_execution_result(
                output,
                scene="living_room",
                method="retrieval",
                protocol=arguments.protocol,
                variant=runtime.variant,
                camera_count=arguments.camera_count,
                seed=arguments.seed,
                success=physical_goal_satisfied,
                executed_actions=executed_actions,
                # Retrieval uses CLIP similarity, never a language model, so
                # every request counter is structurally zero.
                model_calls=0,
                raw_vlm_requests=0,
                replans=0,
                planning_latency_s=0.0,
                elapsed_seconds=time.monotonic() - started_at,
                terminal_status=physical_terminal_status(
                    status, physical_goal_satisfied,
                    result_payload.get("action_history", ()),
                ),
            )
        print(
            f"[retrieval] {runtime.variant}: {predicted_outcome} vs "
            f"{runtime.expected.intended_outcome} "
            f"(match={comparison['outcome_match']}, "
            f"LCS={comparison['lcs_action_count']}/{comparison['expected_action_count']})",
            flush=True,
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
