"""HTTP client adapting the separate SAM 3.1 service to ImageSegmenter."""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image

from mujoco_scenes.perception import SegmentedInstance, validate_segmentations


def decode_rle(payload: Mapping[str, Any]) -> np.ndarray:
    """Decode row-major alternating zero/one run lengths."""
    height_value = payload.get("height")
    width_value = payload.get("width")
    raw_counts = payload.get("counts")
    if (
        isinstance(height_value, bool)
        or not isinstance(height_value, int)
        or isinstance(width_value, bool)
        or not isinstance(width_value, int)
        or height_value <= 0
        or width_value <= 0
    ):
        raise ValueError("mask RLE dimensions must be positive integers")
    if not isinstance(raw_counts, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_counts
    ):
        raise ValueError("mask RLE counts must be an integer array")
    height = height_value
    width = width_value
    counts = raw_counts
    if any(value < 0 for value in counts) or sum(counts) != height * width:
        raise ValueError("invalid mask RLE")
    flat = np.zeros(height * width, dtype=bool)
    offset = 0
    value = False
    for count in counts:
        if value:
            flat[offset : offset + count] = True
        offset += count
        value = not value
    return flat.reshape(height, width)


class Sam3HttpSegmenter:
    """Send only RGB pixels and prompts to a remote SAM 3.1 process."""

    name = "sam3.1_http"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("SAM 3.1 base URL must not be empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("SAM 3.1 timeout must be positive")
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = self._post_json if transport is None else transport

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        message = request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(message, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM 3.1 server returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"cannot reach SAM 3.1 server at {self.base_url}") from exc

    def segment(
        self,
        rgb: np.ndarray,
        *,
        camera_id: str,
        prompts: Sequence[str],
    ) -> tuple[SegmentedInstance, ...]:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        response = self._transport(
            "/v1/segment",
            {
                "camera_id": camera_id,
                "prompts": list(prompts),
                "image_png_base64": base64.b64encode(encoded.getvalue()).decode("ascii"),
            },
        )
        if not isinstance(response, dict):
            raise ValueError("SAM 3.1 response must be an object")
        raw_instances = response.get("instances")
        if not isinstance(raw_instances, list):
            raise ValueError("SAM 3.1 response instances must be an array")
        instances = []
        for item in raw_instances:
            if not isinstance(item, dict):
                raise ValueError("SAM 3.1 instances must be objects")
            instances.append(
                SegmentedInstance(
                    instance_id=str(item["instance_id"]),
                    label=str(item["label"]),
                    score=float(item["score"]),
                    mask=decode_rle(item["mask"]),
                )
            )
        return validate_segmentations(
            instances,
            image_shape=tuple(image.size[::-1]),
        )


def create_segmenter() -> Sam3HttpSegmenter:
    """Factory used by inspect_geometry's MODULE:FACTORY option."""
    return Sam3HttpSegmenter(
        os.getenv("SAM3_BASE_URL", "http://127.0.0.1:8010"),
        api_key=os.getenv("SAM3_API_KEY"),
        timeout_seconds=float(os.getenv("SAM3_TIMEOUT_SECONDS", "120")),
    )
