"""HTTP client adapting the separate SAM 3.1 service to ImageSegmenter."""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Callable, Sequence
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image

from mujoco_scenes.perception import SegmentedInstance


def decode_rle(payload: dict[str, Any]) -> np.ndarray:
    """Decode row-major alternating zero/one run lengths."""
    height = int(payload["height"])
    width = int(payload["width"])
    counts = [int(value) for value in payload["counts"]]
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
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._post_json

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
        instances = []
        for item in response.get("instances", []):
            instances.append(
                SegmentedInstance(
                    instance_id=str(item["instance_id"]),
                    label=str(item["label"]),
                    score=float(item["score"]),
                    mask=decode_rle(item["mask"]),
                )
            )
        return tuple(instances)


def create_segmenter() -> Sam3HttpSegmenter:
    """Factory used by inspect_geometry's MODULE:FACTORY option."""
    return Sam3HttpSegmenter(
        os.getenv("SAM3_BASE_URL", "http://127.0.0.1:8010"),
        api_key=os.getenv("SAM3_API_KEY"),
        timeout_seconds=float(os.getenv("SAM3_TIMEOUT_SECONDS", "120")),
    )
