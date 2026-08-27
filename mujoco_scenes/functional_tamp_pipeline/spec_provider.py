"""Functional-specification source boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import FunctionalSpecification


class FunctionalSpecProvider(ABC):
    @abstractmethod
    def provide(
        self,
        domain: str,
        task_instruction: str,
        observation_images: list[Path] | None = None,
    ) -> FunctionalSpecification:
        """Return a normalized specification; never a physical assignment."""


def provider_for_mode(mode: str) -> FunctionalSpecProvider:
    if mode == "gt":
        from .gt_spec_provider import GTSpecProvider
        return GTSpecProvider()
    if mode == "vlm":
        from .vlm_spec_provider import VLMSpecProvider
        return VLMSpecProvider()
    raise ValueError(f"Unknown functional-specification mode: {mode}")
