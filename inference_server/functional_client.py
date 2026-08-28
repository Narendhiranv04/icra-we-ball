"""Send camera images and a goal for functional decomposition."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import urllib.error
import urllib.parse
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


def request_decomposition(
    *,
    scene: str,
    goal: str,
    images: list[tuple[str, Path]],
    base_url: str,
    api_key: str = "",
    timeout_seconds: float = 360.0,
) -> dict:
    """Send one functional-decomposition request and return decoded JSON."""
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("scene must be a non-empty string")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a non-empty string")
    if not isinstance(images, list) or not images:
        raise ValueError("at least one camera image is required")
    cameras = []
    normalized_images = []
    for row in images:
        if not isinstance(row, tuple) or len(row) != 2:
            raise ValueError("images must contain (camera, path) tuples")
        camera, path = row
        if not isinstance(camera, str) or not camera.strip():
            raise ValueError("camera labels must be non-empty strings")
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Image not found: {path}")
        cameras.append(camera.strip())
        normalized_images.append((camera.strip(), path))
    if len(set(cameras)) != len(cameras):
        raise ValueError("camera labels must be unique")
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string")
    parsed_url = urllib.parse.urlparse(base_url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0.0
    ):
        raise ValueError("timeout_seconds must be finite and positive")
    payload = {
        "scene": scene.strip(),
        "goal": goal.strip(),
        "images": [
            {"camera": camera, "data_url": data_url(path)}
            for camera, path in normalized_images
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.strip().rstrip("/") + "/decompose",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds
        ) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach functional planning server: {error.reason}"
        ) from error
    except (TimeoutError, OSError) as error:
        raise RuntimeError(
            f"Functional planning request failed: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Functional planning server returned invalid JSON"
        ) from error
    if not isinstance(result, dict):
        raise RuntimeError("Functional planning server returned non-object JSON")
    return result


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
    try:
        result = request_decomposition(
            scene=arguments.scene,
            goal=arguments.goal,
            images=arguments.image,
            base_url=arguments.base_url,
            api_key=arguments.api_key,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
