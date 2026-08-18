"""Small boundary between MuJoCo RGB images and learned segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class SegmentedInstance:
    """One image mask with an identity unique within its camera frame."""

    instance_id: str
    mask: np.ndarray
    label: str | None = None
    score: float | None = None


@runtime_checkable
class ImageSegmenter(Protocol):
    """Backend contract for SAM 3, Grounded SAM, or another segmenter.

    ``instance_id`` only needs to be unique in the current frame. The geometry
    layer associates masks across cameras using their RGB-D world centroids.
    The backend must not use MuJoCo body or geometry identifiers.
    """

    name: str

    def segment(
        self,
        rgb: np.ndarray,
        *,
        camera_id: str,
        prompts: Sequence[str],
    ) -> Sequence[SegmentedInstance]:
        """Return visible instance masks for one RGB frame."""


def validate_segmentations(
    instances: Sequence[SegmentedInstance],
    *,
    image_shape: tuple[int, int],
) -> tuple[SegmentedInstance, ...]:
    """Validate untrusted backend output before it affects geometry."""
    accepted: list[SegmentedInstance] = []
    seen: set[str] = set()
    for instance in instances:
        instance_id = instance.instance_id.strip()
        if not instance_id:
            raise ValueError("segmentation instance_id must be non-empty")
        if instance_id in seen:
            raise ValueError(
                f"segmentation backend repeated instance_id {instance_id!r}"
            )
        mask = np.asarray(instance.mask)
        if mask.shape != image_shape:
            raise ValueError(
                f"mask for {instance_id!r} has shape {mask.shape}; "
                f"expected {image_shape}"
            )
        if instance.score is not None and not 0.0 <= instance.score <= 1.0:
            raise ValueError(
                f"score for {instance_id!r} must be between zero and one"
            )
        label = instance.label.strip() if instance.label else None
        accepted.append(
            SegmentedInstance(
                instance_id=instance_id,
                mask=mask.astype(bool, copy=False),
                label=label,
                score=instance.score,
            )
        )
        seen.add(instance_id)
    return tuple(accepted)
