"""Run planning-only OWL-TAMP on one Living Room benchmark variant."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from typing import Sequence

from baseline_common.artifacts import prepare_run_directory, write_json

from vlm_tamp_baseline.living_room_runtime import LivingRoomPlanningRuntime
from vlm_tamp_baseline.pddlstream_refiner import LivingRoomGeometryOracle

from .evaluation import compare_actions, load_expected
from .models import Action, Constraint
from .planner import OWLTAMPPlanner, OWLTAMPPlannerConfig
from .prompt import PROMPT_VERSION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="L1-L10 or internal ID")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-count", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = prepare_run_directory(args.output_dir)
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
        max_tokens=args.max_tokens,
        seed=args.seed,
        **({"base_url": args.base_url} if args.base_url else {}),
        **({"model": args.model} if args.model else {}),
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

    try:
        result = planner.plan(args.goal or runtime.goal, observation, images, oracle)
        expected = load_expected("living_room", runtime.variant)
        comparison = compare_actions(result.actions, expected["actions"])
        predicted_outcome = (
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
            "goal": args.goal or runtime.goal,
            "model": config.model,
            "seed": config.seed,
            "camera_count": args.camera_count,
            "prompt_version": PROMPT_VERSION,
            "planning_rounds": 1,
            "raw_vlm_requests": 1 + len(result.constraints),
            "physical_execution": False,
            "result": result.as_dict(),
            "gt_comparison": comparison,
        }
        write_json(output / "model_trace.json", planner.trace)
        write_json(output / "episode_result.json", payload)
        print("[OWL-TAMP refined plan]", flush=True)
        for index, action in enumerate(result.actions, start=1):
            aliases = runtime.aliases
            shown = [f"{item} ({aliases[item]})" if item in aliases else item for item in action.arguments]
            print(f"  {index}. {action.operator} {', '.join(shown)}", flush=True)
        print("[GT comparison]", json.dumps(comparison, sort_keys=True), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
