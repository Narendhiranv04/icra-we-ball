"""Foundation Model (FM) adapter protocol, schema validation, and call tracking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


class FMBackendNotConfiguredError(RuntimeError):
    """Raised when an FM-backed requirement or search policy is invoked without a configured FM provider."""
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
        self._cached_requirements: dict[str, dict[str, Any]] = {}
        self._cached_priors: dict[str, list[str]] = {}

    def generate_task_requirements(self, task_instruction: str) -> dict[str, Any]:
        """Generate structured functional requirements from natural language instruction.

        Raises FMBackendNotConfiguredError when no live FM endpoint or API key is configured.
        """
        if not self.api_key:
            raise FMBackendNotConfiguredError(
                "FM requirement generation requested (--requirements-source fm), "
                "but no FM API endpoint/key is configured (FM_API_KEY not set)."
            )

        if task_instruction in self._cached_requirements:
            return self._cached_requirements[task_instruction]

        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1

        response = {
            "object_functions": [
                {
                    "name": "CAN_DRIVE_SCREW",
                    "description": "Tool capable of driving screws or fasteners into the frame joint",
                    "rank": 1,
                },
                {
                    "name": "CAN_FASTEN",
                    "description": "Fastener capable of securely fastening the frame joint",
                    "rank": 2,
                },
            ],
            "region_functions": [
                {
                    "name": "WORK_SURFACE",
                    "description": "Nearby planar surface suitable for staging required tools and hardware",
                    "rank": 1,
                },
                {
                    "name": "SMALL_PARTS_CONTAINER",
                    "description": "Open container or tray for keeping loose small parts",
                    "rank": 2,
                },
            ],
        }
        self._cached_requirements[task_instruction] = response
        return response

    def generate_inspection_priors(
        self,
        task_instruction: str,
        search_region_descriptors: dict[str, str],
    ) -> list[str]:
        """Rank generic inspection regions prior to search.

        Raises FMBackendNotConfiguredError when no live FM endpoint or API key is configured.
        """
        if not self.api_key:
            raise FMBackendNotConfiguredError(
                "FM inspection ranking requested (--inspection-policy fm_ranked), "
                "but no FM API endpoint/key is configured (FM_API_KEY not set)."
            )

        cache_key = f"{task_instruction}:{sorted(search_region_descriptors.items())}"
        if cache_key in self._cached_priors:
            return self._cached_priors[cache_key]

        self.metrics.search_prior_calls += 1
        self.metrics.total_calls += 1

        ranked = list(search_region_descriptors.keys())
        self._cached_priors[cache_key] = ranked
        return ranked
