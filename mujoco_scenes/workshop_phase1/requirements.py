"""Functional requirement providers for Workshop Phase 1."""

from __future__ import annotations

import abc
from typing import Any

from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter
from mujoco_scenes.workshop_phase1.types import (
    EntityType,
    FunctionalRequirement,
    RequirementSource,
)

CANONICAL_WORKSHOP_INSTRUCTION = (
    "Repair the loose frame joint using an appropriate tool and fastener. "
    "Arrange the required tool and hardware on a suitable nearby work surface, "
    "and keep loose small parts in a suitable container."
)


class RequirementProvider(abc.ABC):
    """Abstract interface for extracting broad task functional requirements."""

    @abc.abstractmethod
    def get_requirements(self, task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION) -> list[FunctionalRequirement]:
        pass


class StaticWorkshopRequirementProvider(RequirementProvider):
    """Deterministic static baseline returning the 4 canonical functional requirements."""

    def get_requirements(self, task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION) -> list[FunctionalRequirement]:
        return [
            FunctionalRequirement(
                requirement_id="req_obj_driver",
                entity_type=EntityType.OBJECT,
                function_name="CAN_DRIVE_SCREW",
                description="Tool capable of driving fasteners into the frame joint",
                rank=1,
                source=RequirementSource.STATIC,
                semantic_hints=["screwdriver", "driver", "drill"],
                geometric_constraints={"min_reach_m": 0.025},
                provenance="static_workshop_specification",
            ),
            FunctionalRequirement(
                requirement_id="req_obj_fastener",
                entity_type=EntityType.OBJECT,
                function_name="CAN_FASTEN",
                description="Fastener component capable of securing the frame joint",
                rank=2,
                source=RequirementSource.STATIC,
                semantic_hints=["screw", "fastener", "bolt"],
                geometric_constraints={"min_length_m": 0.025, "max_diameter_m": 0.008},
                provenance="static_workshop_specification",
            ),
            FunctionalRequirement(
                requirement_id="req_reg_work_surface",
                entity_type=EntityType.REGION,
                function_name="WORK_SURFACE",
                description="Nearby unobstructed planar work surface suitable for staging tool and fastener",
                rank=1,
                source=RequirementSource.STATIC,
                semantic_hints=["work_surface", "table", "cart", "shelf"],
                geometric_constraints={"min_area_m2": 0.015, "unobstructed": True},
                provenance="static_workshop_specification",
            ),
            FunctionalRequirement(
                requirement_id="req_reg_parts_container",
                entity_type=EntityType.REGION,
                function_name="SMALL_PARTS_CONTAINER",
                description="Open container or tray suitable for holding loose small parts",
                rank=2,
                source=RequirementSource.STATIC,
                semantic_hints=["tray", "bin", "container"],
                geometric_constraints={"min_volume_m3": 0.0001, "open_top": True},
                provenance="static_workshop_specification",
            ),
        ]


class FMRequirementProvider(RequirementProvider):
    """Generates requirements via Foundation Model with schema validation and caching."""

    def __init__(self, fm_adapter: FMAdapter | None = None) -> None:
        self.fm_adapter = fm_adapter or FMAdapter()

    def get_requirements(self, task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION) -> list[FunctionalRequirement]:
        raw = self.fm_adapter.generate_task_requirements(task_instruction)
        reqs: list[FunctionalRequirement] = []

        # Parse object functions
        for idx, item in enumerate(raw.get("object_functions", [])):
            name = item.get("name", f"OBJECT_FN_{idx}")
            desc = item.get("description", "")
            rank = item.get("rank", idx + 1)
            reqs.append(
                FunctionalRequirement(
                    requirement_id=f"fm_req_obj_{idx:02d}",
                    entity_type=EntityType.OBJECT,
                    function_name=name,
                    description=desc,
                    rank=rank,
                    source=RequirementSource.FM,
                    provenance="fm_task_prompt",
                )
            )

        # Parse region functions
        for idx, item in enumerate(raw.get("region_functions", [])):
            name = item.get("name", f"REGION_FN_{idx}")
            desc = item.get("description", "")
            rank = item.get("rank", idx + 1)
            reqs.append(
                FunctionalRequirement(
                    requirement_id=f"fm_req_reg_{idx:02d}",
                    entity_type=EntityType.REGION,
                    function_name=name,
                    description=desc,
                    rank=rank,
                    source=RequirementSource.FM,
                    provenance="fm_task_prompt",
                )
            )

        return reqs
