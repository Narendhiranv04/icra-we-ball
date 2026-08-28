"""Request one validated VLM-TAMP intermediate-subgoal sequence."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from baseline_common.client import data_url, image_argument
from baseline_common.inference import PlanningError
from baseline_common.models import Observation

from .models import ObjectUniverse
from .planner import VLMTAMPPlanner, VLMTAMPPlannerConfig


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--goal", required=True)
    result.add_argument(
        "--observation",
        type=Path,
        required=True,
        help="Visible-state JSON; never include hidden simulator inventory.",
    )
    result.add_argument(
        "--image",
        action="append",
        type=image_argument,
        required=True,
        help="Camera-labelled image: CAMERA=/path/to/image.png",
    )
    result.add_argument(
        "--object-universe",
        type=Path,
        help=(
            "Optional JSON list of persistent object IDs only. Omitting it "
            "uses the IDs in the visible textualized state."
        ),
    )
    result.add_argument("--base-url")
    result.add_argument("--model")
    result.add_argument("--max-tokens", type=int)
    result.add_argument("--no-thinking", action="store_true")
    return result


def main() -> None:
    arguments = parser().parse_args()
    environment = dict(os.environ)
    if arguments.no_thinking:
        environment["VLM_TAMP_ENABLE_THINKING"] = "false"
    config = VLMTAMPPlannerConfig.from_env(environment)
    config = replace(
        config,
        base_url=config.base_url if arguments.base_url is None else arguments.base_url,
        model=config.model if arguments.model is None else arguments.model,
        max_tokens=(
            config.max_tokens if arguments.max_tokens is None else arguments.max_tokens
        ),
    )
    observation = Observation.from_dict(
        json.loads(arguments.observation.read_text(encoding="utf-8"))
    )
    universe = None
    if arguments.object_universe:
        universe = ObjectUniverse.from_dict(
            json.loads(arguments.object_universe.read_text(encoding="utf-8"))
        )
    images = tuple(
        {"camera": camera, "data_url": data_url(path)}
        for camera, path in arguments.image
    )
    try:
        result = VLMTAMPPlanner(config).plan(
            arguments.goal, observation, images, universe=universe
        )
    except (OSError, ValueError, PlanningError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
