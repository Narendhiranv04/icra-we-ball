"""Run planning-only OWL-TAMP on a five-view Kitchen observation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from baseline_common.artifacts import prepare_run_directory, write_json

from mujoco_scenes.baseline_kitchen_runtime import BaselineKitchenRuntime

from .evaluation import compare_kitchen_actions, load_expected
from .models import Action, Constraint
from .planner import OWLTAMPPlanner, OWLTAMPPlannerConfig
from .prompt import PROMPT_VERSION


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_run_directory(args.output_dir)
    if args.phase1_run_dir is not None:
        raise SystemExit(
            "OWL-TAMP K1-K12 planning constructs variants directly; remove "
            "--phase1-run-dir"
        )
    runtime = BaselineKitchenRuntime.from_variant(
        args.variant,
        output,
        image_width=args.image_width,
        image_height=args.image_height,
        camera_count=args.camera_count,
        show_viewer=False,
    )
    config = OWLTAMPPlannerConfig.from_env()
    config = replace(
        config,
        max_tokens=args.max_tokens,
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

    try:
        goal = args.goal or str(runtime.scene.config.goal)
        result = planner.plan(goal, observation, images, oracle)
        backend_by_id = {
            str(row["generic_object_id"]): str(row["physical_backend_body"])
            for row in runtime.bundle.resolution.get("accepted", ())
        }
        translated = tuple(
            Action(
                action.operator,
                tuple(backend_by_id.get(argument, argument) for argument in action.arguments),
            )
            for action in result.actions
        )
        expected = load_expected("kitchen", args.variant)
        comparison = compare_kitchen_actions(translated, expected["actions"])
        predicted_outcome = (
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
            "planning_rounds": 1,
            "raw_vlm_requests": 1 + len(result.constraints),
            "physical_execution": False,
            "observable_state_only": True,
            "closed_storage_contents_exposed": False,
            "result": result.as_dict(),
            "private_id_translation_used_only_for_gt_comparison": True,
            "gt_comparison": comparison,
        }
        write_json(output / "model_trace.json", planner.trace)
        write_json(output / "episode_result.json", payload)
        print("[OWL-TAMP refined plan]", flush=True)
        aliases = {
            generic: backend for generic, backend in backend_by_id.items()
        }
        for index, action in enumerate(result.actions, start=1):
            shown = [f"{item} ({aliases[item]})" if item in aliases else item for item in action.arguments]
            print(f"  {index}. {action.operator} {', '.join(shown)}", flush=True)
        print("[GT comparison]", json.dumps(comparison, sort_keys=True), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
