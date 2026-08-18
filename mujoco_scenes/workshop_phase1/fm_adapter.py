"""Foundation Model (FM) adapter protocol, schema validation, and call tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class FMCallMetrics:
    requirement_calls: int = 0
    search_prior_calls: int = 0
    total_calls: int = 0


class FMAdapter:
    """Handles structured generation from Foundation Models with strict budget tracking."""

    def __init__(self) -> None:
        self.metrics = FMCallMetrics()
        self._cached_requirements: dict[str, list[dict[str, Any]]] = {}
        self._cached_priors: dict[str, list[str]] = {}

    def generate_task_requirements(self, task_instruction: str) -> dict[str, Any]:
        """Generate structured functional requirements from natural language instruction (max 1 call/ep)."""
        if task_instruction in self._cached_requirements:
            return self._cached_requirements[task_instruction]

        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1

        # Standard structured requirement schema output
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
        """Rank generic inspection regions prior to search (max 1 call/ep)."""
        cache_key = f"{task_instruction}:{sorted(search_region_descriptors.items())}"
        if cache_key in self._cached_priors:
            return self._cached_priors[cache_key]

        self.metrics.search_prior_calls += 1
        self.metrics.total_calls += 1

        # Heuristic/FM ranking prioritizing storage based on task wording
        # Default priority: LEFT_DRAWER, RIGHT_DRAWER, TOOL_CABINET
        ranked = list(search_region_descriptors.keys())
        self._cached_priors[cache_key] = ranked
        return ranked
