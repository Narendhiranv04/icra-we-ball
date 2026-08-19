"""Functional requirement providers and manual future-FM contract for Workshop Phase 1."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any
import yaml

from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter, FMBackendNotConfiguredError
from mujoco_scenes.workshop_phase1.types import (
    EntityType,
    FunctionalRequirement,
    RequirementSource,
)

DEFAULT_FM_CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "workshop_phase1_fm_contract.yaml"
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

    @abc.abstractmethod
    def get_semantic_vocabulary(self) -> dict[str, list[str]]:
        """Return canonical label to alias list mapping."""
        pass

    @abc.abstractmethod
    def get_detector_prompts(self) -> list[str]:
        """Return flattened list of prompt strings for YOLO-World set_classes."""
        pass

    @abc.abstractmethod
    def get_alias_to_canonical_map(self) -> dict[str, str]:
        """Return mapping from lowercase prompt string to canonical semantic category."""
        pass


class ManualWorkshopFMContract(RequirementProvider):
    """Manual surrogate contract representing the one-time FM output at episode start.

    Loads the structured contract from YAML (identical across all 14 variants).
    Zero variant-specific labels or backend simulator names are included.
    """

    def __init__(self, contract_path: Path | None = None) -> None:
        self.contract_path = contract_path or DEFAULT_FM_CONTRACT_PATH
        self._contract_data = self._load_contract()

    def _load_contract(self) -> dict[str, Any]:
        if self.contract_path.is_file():
            try:
                with open(self.contract_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        # Hardcoded safe defaults matching workshop_phase1_fm_contract.yaml
        return {
            "task_instruction": CANONICAL_WORKSHOP_INSTRUCTION,
            "functional_requirements": [
                {
                    "requirement_id": "req_obj_driver",
                    "entity_type": "OBJECT",
                    "function_name": "CAN_DRIVE_SCREW",
                    "description": "Tool capable of driving fasteners into the frame joint",
                    "rank": 1,
                    "accepted_categories": ["screwdriver", "power_driver"],
                    "geometric_constraints": {"min_reach_m": 0.025},
                },
                {
                    "requirement_id": "req_obj_fastener",
                    "entity_type": "OBJECT",
                    "function_name": "CAN_FASTEN",
                    "description": "Fastener component capable of securing the frame joint",
                    "rank": 2,
                    "accepted_categories": ["screw", "bolt"],
                    "geometric_constraints": {"min_length_m": 0.022, "max_diameter_m": 0.009},
                },
                {
                    "requirement_id": "req_reg_work_surface",
                    "entity_type": "FUNCTIONAL_REGION",
                    "function_name": "WORK_SURFACE",
                    "description": "Nearby unobstructed planar work surface suitable for staging tool and fastener",
                    "rank": 1,
                    "accepted_categories": ["workbench", "tool_cart", "shelf"],
                    "geometric_constraints": {"min_area_m2": 0.015, "unobstructed": True},
                },
                {
                    "requirement_id": "req_reg_parts_container",
                    "entity_type": "FUNCTIONAL_REGION",
                    "function_name": "SMALL_PARTS_CONTAINER",
                    "description": "Open container or tray suitable for holding loose small parts",
                    "rank": 2,
                    "accepted_categories": ["parts_tray", "hardware_bin"],
                    "geometric_constraints": {"min_volume_m3": 0.0001, "open_top": True},
                },
            ],
            "vocabulary": {
                "canonical_labels": {
                    "screwdriver": ["screwdriver", "hand screwdriver", "manual screwdriver"],
                    "power_driver": ["power drill", "powered screwdriver", "cordless drill", "power driver", "drill driver"],
                    "screw": ["screw", "machine screw", "threaded screw", "fastener"],
                    "bolt": ["bolt", "hex bolt", "machine bolt"],
                    "wrench": ["wrench", "adjustable wrench", "combination wrench", "spanner"],
                    "pliers": ["pliers", "combination pliers", "locking pliers"],
                    "workbench": ["workbench", "work table", "wooden table", "table"],
                    "tool_cart": ["tool cart", "cart", "rolling cart", "metal cart"],
                    "shelf": ["shelf", "wall shelf", "narrow shelf"],
                    "parts_tray": ["tray", "parts tray", "shallow tray"],
                    "hardware_bin": ["bin", "hardware bin", "plastic bin", "parts bin", "storage bin"],
                }
            },
        }

    def get_requirements(self, task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION) -> list[FunctionalRequirement]:
        raw_reqs = self._contract_data.get("functional_requirements", [])
        requirements: list[FunctionalRequirement] = []
        for r in raw_reqs:
            e_type = EntityType(r.get("entity_type", "OBJECT"))
            requirements.append(
                FunctionalRequirement(
                    requirement_id=r["requirement_id"],
                    entity_type=e_type,
                    function_name=r["function_name"],
                    description=r["description"],
                    rank=r.get("rank", 1),
                    source=RequirementSource.STATIC,
                    accepted_categories=list(r.get("accepted_categories", [])),
                    semantic_hints=list(r.get("accepted_categories", [])),
                    geometric_constraints=dict(r.get("geometric_constraints", {})),
                    provenance="manual_workshop_fm_contract",
                )
            )
        return requirements

    def get_semantic_vocabulary(self) -> dict[str, list[str]]:
        vocab = self._contract_data.get("vocabulary", {})
        return dict(vocab.get("canonical_labels", {}))

    def get_detector_prompts(self) -> list[str]:
        vocab = self.get_semantic_vocabulary()
        prompts: list[str] = []
        for aliases in vocab.values():
            for alias in aliases:
                if alias not in prompts:
                    prompts.append(alias)
        return prompts

    def get_alias_to_canonical_map(self) -> dict[str, str]:
        vocab = self.get_semantic_vocabulary()
        mapping: dict[str, str] = {}
        for canonical, aliases in vocab.items():
            mapping[canonical.lower()] = canonical
            for alias in aliases:
                mapping[alias.lower()] = canonical
        return mapping


# Alias for backward compatibility
StaticWorkshopRequirementProvider = ManualWorkshopFMContract


class FMRequirementProvider(RequirementProvider):
    """Generates requirements via live Foundation Model when configured."""

    def __init__(self, fm_adapter: FMAdapter | None = None) -> None:
        self.fm_adapter = fm_adapter or FMAdapter()

    def get_requirements(self, task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION) -> list[FunctionalRequirement]:
        raise FMBackendNotConfiguredError(
            "Live FM requirement generation requested, but no FM endpoint is configured. "
            "Phase 1 runs deterministically with ManualWorkshopFMContract."
        )

    def get_semantic_vocabulary(self) -> dict[str, list[str]]:
        raise FMBackendNotConfiguredError("Live FM backend not configured.")

    def get_detector_prompts(self) -> list[str]:
        raise FMBackendNotConfiguredError("Live FM backend not configured.")

    def get_alias_to_canonical_map(self) -> dict[str, str]:
        raise FMBackendNotConfiguredError("Live FM backend not configured.")
