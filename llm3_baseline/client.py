"""Request one observation-bounded LLM3-style action plan."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from baseline_common.client import data_url, image_argument

from .models import Failure, Observation
from .planner import LLM3Planner, PlannerConfig, PlanningError


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
        "--failure",
        type=Path,
        help="Optional JSON failure feedback from a previous motion attempt.",
    )
    result.add_argument("--base-url")
    result.add_argument("--model")
    result.add_argument("--max-tokens", type=int)
    result.add_argument(
        "--no-thinking", action="store_true", help="Disable toggleable thinking."
    )
    return result


def _failure(path: Path | None) -> Failure | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Failure file must contain a JSON object")
    code = raw.get("code")
    message = raw.get("message")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Failure code must be a non-empty string")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Failure message must be a non-empty string")
    return Failure(code.strip(), message.strip())


def main() -> None:
    arguments = parser().parse_args()
    environment = dict(os.environ)
    if arguments.no_thinking:
        environment["LLM3_ENABLE_THINKING"] = "false"
    config = PlannerConfig.from_env(environment)
    overrides = {
        "base_url": config.base_url if arguments.base_url is None else arguments.base_url,
        "model": config.model if arguments.model is None else arguments.model,
        "max_tokens": (
            config.max_tokens if arguments.max_tokens is None else arguments.max_tokens
        ),
    }
    config = replace(config, **overrides)
    observation = Observation.from_dict(
        json.loads(arguments.observation.read_text(encoding="utf-8"))
    )
    images = tuple(
        {"camera": camera, "data_url": data_url(path)}
        for camera, path in arguments.image
    )
    try:
        result = LLM3Planner(config).plan(
            arguments.goal,
            observation,
            images,
            failure=_failure(arguments.failure),
        )
    except (OSError, ValueError, PlanningError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
