"""Request one observation-bounded LLM3-style action plan."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from dataclasses import replace
from pathlib import Path

from .models import Failure, Observation
from .planner import LLM3Planner, PlannerConfig, PlanningError


def image_argument(value: str) -> tuple[str, Path]:
    if "=" in value:
        camera, path = value.split("=", 1)
    else:
        path = value
        camera = Path(path).stem
    image = Path(path)
    if not camera.strip() or not image.is_file():
        raise argparse.ArgumentTypeError(
            "Use CAMERA=/path/to/an/existing/image.png"
        )
    return camera.strip(), image


def data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


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
    config = PlannerConfig.from_env()
    overrides = {
        "base_url": arguments.base_url or config.base_url,
        "model": arguments.model or config.model,
        "max_tokens": arguments.max_tokens or config.max_tokens,
        "enable_thinking": False if arguments.no_thinking else config.enable_thinking,
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
