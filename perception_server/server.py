"""Small SAM 3.1 HTTP service for RGB image segmentation."""

from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import os
import re
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a mask as row-major alternating zero/one run lengths."""
    array = np.asarray(mask, dtype=bool)
    flat = array.reshape(-1)
    counts: list[int] = []
    current = False
    count = 0
    for value in flat:
        value = bool(value)
        if value == current:
            count += 1
        else:
            counts.append(count)
            current = value
            count = 1
    counts.append(count)
    return {"height": array.shape[0], "width": array.shape[1], "counts": counts}


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    mask: np.ndarray
    box_xyxy: tuple[float, float, float, float] | None = None


class SegmentationEngine(Protocol):
    name: str

    def segment(self, image: Image.Image, prompts: list[str]) -> list[Detection]: ...


def normalize_prompts(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("prompts must be an array of strings")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("prompts must contain only strings")
    prompts = [item.strip() for item in value]
    prompts = [item for item in prompts if item]
    if not 1 <= len(prompts) <= 32:
        raise ValueError("provide between 1 and 32 non-empty prompts")
    return prompts


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def deduplicate(detections: list[Detection], threshold: float) -> list[Detection]:
    """Remove duplicate masks emitted for overlapping text prompts."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(_mask_iou(detection.mask, other.mask) < threshold for other in kept):
            kept.append(detection)
    return kept


class Sam31Engine:
    name = "sam3.1"

    def __init__(self) -> None:
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import (
                build_sam3_image_model,
                download_ckpt_from_hf,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SAM 3.1 is not installed; follow perception_server/README.md"
            ) from exc

        self.device = os.getenv("SAM3_DEVICE", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM3_DEVICE=cuda but PyTorch cannot see a CUDA GPU")
        checkpoint = os.getenv("SAM3_CHECKPOINT")
        if not checkpoint:
            checkpoint = download_ckpt_from_hf(version="sam3.1")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint,
            load_from_HF=False,
            device=self.device,
            compile=os.getenv("SAM3_COMPILE", "0") == "1",
        )
        confidence = float(os.getenv("SAM3_CONFIDENCE", "0.5"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("SAM3_CONFIDENCE must be between 0 and 1")
        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=confidence,
        )
        self.minimum_area = int(os.getenv("SAM3_MINIMUM_MASK_PIXELS", "32"))
        self.duplicate_iou = float(os.getenv("SAM3_DUPLICATE_IOU", "0.85"))
        if self.minimum_area < 1:
            raise ValueError("SAM3_MINIMUM_MASK_PIXELS must be positive")
        if not 0.0 <= self.duplicate_iou <= 1.0:
            raise ValueError("SAM3_DUPLICATE_IOU must be between 0 and 1")

    def segment(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        state = self.processor.set_image(image)
        detections: list[Detection] = []
        for prompt in prompts:
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(prompt=prompt, state=state)
            masks = output["masks"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            boxes = output["boxes"].detach().cpu().numpy()
            for mask, score, box in zip(masks, scores, boxes):
                mask = np.asarray(mask).squeeze().astype(bool)
                if mask.ndim != 2 or np.count_nonzero(mask) < self.minimum_area:
                    continue
                detections.append(
                    Detection(
                        label=prompt,
                        score=float(score),
                        mask=mask,
                        box_xyxy=tuple(float(value) for value in box),
                    )
                )
        return deduplicate(detections, self.duplicate_iou)


class ContractTestEngine:
    """Deterministic rectangle masks for testing transport without a GPU."""

    name = "contract_test_not_sam"

    def segment(self, image: Image.Image, prompts: list[str]) -> list[Detection]:
        width, height = image.size
        detections = []
        for index, prompt in enumerate(prompts):
            mask = np.zeros((height, width), dtype=bool)
            inset = min(index + 1, max(1, min(width, height) // 4))
            mask[inset : height - inset, inset : width - inset] = True
            detections.append(Detection(prompt, 1.0, mask))
        return detections


class SamServer(ThreadingHTTPServer):
    def __init__(self, address, engine: SegmentationEngine, api_key: str | None):
        super().__init__(address, SamRequestHandler)
        self.engine = engine
        self.api_key = api_key
        self.inference_lock = threading.Lock()


class SamRequestHandler(BaseHTTPRequestHandler):
    server: SamServer

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = self.server.api_key
        if not expected:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, f"Bearer {expected}")

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._json(HTTPStatus.OK, {"ok": True, "model": self.server.engine.name})

    def do_POST(self) -> None:
        if self.path != "/v1/segment":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        if not self._authorized():
            self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 25_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            prompts = normalize_prompts(payload["prompts"])
            image_bytes = base64.b64decode(payload["image_png_base64"], validate=True)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            with self.server.inference_lock:
                detections = self.server.engine.segment(image, prompts)
            instances = []
            for index, detection in enumerate(detections):
                slug = re.sub(r"[^a-z0-9]+", "_", detection.label.lower()).strip("_")
                instances.append(
                    {
                        "instance_id": f"{slug or 'object'}_{index + 1:03d}",
                        "label": detection.label,
                        "score": detection.score,
                        "box_xyxy": detection.box_xyxy,
                        "mask": encode_rle(detection.mask),
                    }
                )
            self._json(HTTPStatus.OK, {"instances": instances})
        except (KeyError, TypeError, ValueError, UnidentifiedImageError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self.log_error("Segmentation failed: %s", exc)
            self._json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "segmentation inference failed"},
            )

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("SAM3_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SAM3_PORT", "8010")))
    parser.add_argument(
        "--contract-test",
        action="store_true",
        help="test HTTP/RLE wiring with fake masks; does not run SAM",
    )
    args = parser.parse_args(argv)
    engine = ContractTestEngine() if args.contract_test else Sam31Engine()
    server = SamServer((args.host, args.port), engine, os.getenv("SAM3_API_KEY"))
    print(f"Serving {engine.name} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
