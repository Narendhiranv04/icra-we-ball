"""Foundation Model (FM) adapter protocol, schema validation, and call tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class FMBackendNotConfiguredError(RuntimeError):
    """Raised when an FM-backed requirement or search policy is invoked without a live configured FM provider."""
    pass


@dataclass
class FMCallMetrics:
    requirement_calls: int = 0
    search_prior_calls: int = 0
    total_calls: int = 0


class FMAdapter:
    """Handles structured generation from Foundation Models with strict budget tracking."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("FM_API_KEY")
        self.metrics = FMCallMetrics()

    def generate_task_requirements(self, task_instruction: str) -> dict[str, Any]:
        """Generate structured functional requirements from natural language instruction.

        Raises FMBackendNotConfiguredError when no live remote FM endpoint is configured.
        """
        raise FMBackendNotConfiguredError(
            "FM requirement generation requested (--requirements-source fm), "
            "but no real FM backend is configured. Phase 1 runs deterministically with --requirements-source static."
        )

    def generate_inspection_priors(
        self,
        task_instruction: str,
        search_region_descriptors: dict[str, str],
    ) -> list[str]:
        """Rank generic inspection regions prior to search.

        Raises FMBackendNotConfiguredError when no live remote FM endpoint is configured.
        """
        raise FMBackendNotConfiguredError(
            "FM inspection ranking requested (--inspection-policy fm_ranked), "
            "but no real FM backend is configured. Phase 1 runs deterministically with --inspection-policy fixed."
        )
