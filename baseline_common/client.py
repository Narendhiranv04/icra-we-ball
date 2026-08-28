"""Small command-line image helpers shared by comparison clients."""

from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path


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
