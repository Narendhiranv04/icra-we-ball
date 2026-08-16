"""Send camera images and a goal for functional decomposition."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from pathlib import Path


def image_argument(value: str) -> tuple[str, Path]:
    if "=" in value:
        camera, path = value.split("=", 1)
    else:
        path = value
        camera = Path(path).stem
    if not camera.strip() or not path:
        raise argparse.ArgumentTypeError("Use CAMERA=/path/to/image.png")
    image_path = Path(path)
    if not image_path.is_file():
        raise argparse.ArgumentTypeError(f"Image not found: {image_path}")
    return camera.strip(), image_path


def data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--scene",
        choices=("kitchen", "living_room", "workshop"),
        required=True,
    )
    result.add_argument("--goal", required=True)
    result.add_argument(
        "--image",
        action="append",
        type=image_argument,
        required=True,
        help="Camera-labelled image: CAMERA=/path/to/image.png",
    )
    result.add_argument(
        "--base-url",
        default=os.environ.get("PLANNER_BASE_URL", "http://127.0.0.1:8080/v1"),
    )
    result.add_argument(
        "--api-key",
        default=os.environ.get("PLANNER_API_KEY", "")
        or os.environ.get("INFERENCE_API_KEY", ""),
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    payload = {
        "scene": arguments.scene,
        "goal": arguments.goal,
        "images": [
            {"camera": camera, "data_url": data_url(path)}
            for camera, path in arguments.image
        ],
    }
    headers = {"Content-Type": "application/json"}
    if arguments.api_key:
        headers["Authorization"] = f"Bearer {arguments.api_key}"
    request = urllib.request.Request(
        arguments.base_url.rstrip("/") + "/decompose",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=360) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"Cannot reach functional planning server: {error.reason}"
        ) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
