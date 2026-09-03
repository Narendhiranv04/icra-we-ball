"""CLIP image-text retrieval over annotated candidate crops.

Crops are taken from the RAW camera frames.  The annotated frames carry printed
semantic aliases, and CLIP reads text, so scoring those would leak the answer
through the pixels rather than testing perception.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_WEIGHTS = ROOT / "semantic_model_cache" / "weights" / "clip" / "ViT-B-32.pt"
MINIMUM_CROP_PIXELS = 4
# CLIP resizes every crop to 224x224, so a tight box on a distant object gives
# it almost no context.  Pad each box outward before cropping; this is an
# implementation choice of this baseline and is recorded in its trace.
CROP_CONTEXT_FRACTION = 0.25
CROP_CONTEXT_MINIMUM_PIXELS = 6


class RetrievalUnavailableError(RuntimeError):
    """CLIP or its weights are not installed."""


def load_clip(weights: str | Path = DEFAULT_CLIP_WEIGHTS, device: str = "cpu"):
    try:
        import clip  # type: ignore
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise RetrievalUnavailableError(f"CLIP is not importable: {error}") from error
    path = Path(weights)
    if not path.is_file():
        raise RetrievalUnavailableError(
            f"CLIP weights are missing at {path}. Run "
            "mujoco_scenes/scripts/prepare_semantic_models.py."
        )
    model, preprocess = clip.load(str(path), device=device)
    model.eval()
    return model, preprocess, clip, torch


@dataclass(frozen=True)
class RetrievalScores:
    """Similarity of every candidate to every role phrase."""

    by_phrase: dict[str, dict[str, float]] = field(default_factory=dict)
    cameras_used: tuple[str, ...] = ()
    candidates_scored: int = 0

    def ranking(self, phrase: str) -> list[tuple[str, float]]:
        return sorted(
            self.by_phrase.get(phrase, {}).items(), key=lambda row: (-row[1], row[0])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_phrase": {k: dict(v) for k, v in self.by_phrase.items()},
            "cameras_used": list(self.cameras_used),
            "candidates_scored": self.candidates_scored,
        }


def _crops(
    annotations: Mapping[str, Any],
    observation_dir: Path,
    kind: str,
    candidate_ids: Iterable[str] | None = None,
):
    """Yield (candidate_id, camera, PIL crop) from the raw frames."""
    from PIL import Image

    allowed = None if candidate_ids is None else set(candidate_ids)
    key = "regions" if kind == "region" else "objects"
    for camera, payload in sorted((annotations.get("cameras") or {}).items()):
        raw = observation_dir / f"raw_{camera}.png"
        if not raw.is_file():
            continue
        with Image.open(raw) as handle:
            frame = handle.convert("RGB")
            for row in payload.get(key, ()):
                box = row.get("bbox_xyxy")
                identifier = row.get("id")
                if not identifier or not box or len(box) != 4:
                    continue
                if allowed is not None and str(identifier) not in allowed:
                    continue
                left, top, right, bottom = (int(value) for value in box)
                if right - left < MINIMUM_CROP_PIXELS or bottom - top < MINIMUM_CROP_PIXELS:
                    continue
                pad_x = max(
                    CROP_CONTEXT_MINIMUM_PIXELS,
                    int(round((right - left) * CROP_CONTEXT_FRACTION)),
                )
                pad_y = max(
                    CROP_CONTEXT_MINIMUM_PIXELS,
                    int(round((bottom - top) * CROP_CONTEXT_FRACTION)),
                )
                window = (
                    max(0, left - pad_x),
                    max(0, top - pad_y),
                    min(frame.width, right + pad_x),
                    min(frame.height, bottom + pad_y),
                )
                yield str(identifier), camera, frame.crop(window)


class CLIPRetriever:
    """Score candidate crops against role function phrases."""

    def __init__(self, weights: str | Path = DEFAULT_CLIP_WEIGHTS, device: str = "cpu"):
        self.device = device
        self.model, self.preprocess, self._clip, self._torch = load_clip(weights, device)

    def _text_features(self, phrases: Sequence[str]):
        tokens = self._clip.tokenize(list(phrases)).to(self.device)
        with self._torch.no_grad():
            features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def score(
        self,
        annotations: Mapping[str, Any],
        observation_dir: str | Path,
        phrases: Sequence[str],
        kind: str,
        candidate_ids: Iterable[str] | None = None,
    ) -> RetrievalScores:
        """Cosine similarity per (phrase, candidate), max-pooled over cameras."""
        observation_dir = Path(observation_dir)
        text = self._text_features(phrases)
        by_phrase: dict[str, dict[str, float]] = {phrase: {} for phrase in phrases}
        cameras: set[str] = set()
        candidates: set[str] = set()
        for identifier, camera, crop in _crops(
            annotations, observation_dir, kind, candidate_ids
        ):
            tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
            with self._torch.no_grad():
                image = self.model.encode_image(tensor)
            image = image / image.norm(dim=-1, keepdim=True)
            similarity = (image @ text.T).squeeze(0).tolist()
            if not isinstance(similarity, list):
                similarity = [similarity]
            cameras.add(camera)
            candidates.add(identifier)
            for phrase, value in zip(phrases, similarity):
                previous = by_phrase[phrase].get(identifier)
                score = float(value)
                if previous is None or score > previous:
                    by_phrase[phrase][identifier] = score
        return RetrievalScores(
            by_phrase=by_phrase,
            cameras_used=tuple(sorted(cameras)),
            candidates_scored=len(candidates),
        )


def read_annotations(observation_dir: str | Path) -> dict[str, Any]:
    path = Path(observation_dir) / "annotations.json"
    if not path.is_file():
        raise FileNotFoundError(f"No annotations.json under {observation_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def assign_distinct(
    scores: RetrievalScores, phrase: str, count: int, taken: Iterable[str]
) -> list[str]:
    """Take the highest-scoring unclaimed candidates for one role."""
    claimed = set(taken)
    chosen: list[str] = []
    for identifier, _score in scores.ranking(phrase):
        if identifier in claimed:
            continue
        chosen.append(identifier)
        claimed.add(identifier)
        if len(chosen) == count:
            break
    return chosen
