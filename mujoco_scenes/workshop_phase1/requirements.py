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
    "Find the compatible screw and first compatible driver encountered, "
    "insert the screw tip-down into the workbench hole, and drive it fully."
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
        """Return exactly one detector-friendly label per canonical category."""
        pass

    @abc.abstractmethod
    def get_ranked_detector_vocabulary(self) -> list[dict[str, Any]]:
        """Return the one-time FM-owned canonical detector vocabulary in rank order."""
        pass

    @abc.abstractmethod
    def get_detector_label_to_canonical_map(self) -> dict[str, str]:
        """Map the detector's display labels, and only those labels, to canonicals."""
        pass

    @abc.abstractmethod
    def get_alias_to_canonical_map(self) -> dict[str, str]:
        """Return mapping from lowercase prompt string to canonical semantic category."""
        pass


class ManualWorkshopFMContract(RequirementProvider):
    """Manual surrogate contract representing the one-time FM output at episode start.

    Loads the structured contract from YAML (identical across all 10 variants).
    Zero variant-specific labels or backend simulator names are included.
    """

    def __init__(self, contract_path: Path | None = None) -> None:
        self.contract_path = contract_path or DEFAULT_FM_CONTRACT_PATH
        self._contract_data = self._load_contract()

    def _load_contract(self) -> dict[str, Any]:
        if not self.contract_path.is_file():
            raise FileNotFoundError(
                f"Manual Workshop FM contract is unavailable: {self.contract_path}")
        with open(self.contract_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Workshop FM contract must be a mapping")
        if not isinstance(data.get("functional_requirements"), list) or not data["functional_requirements"]:
            raise ValueError("Workshop FM contract requires a non-empty functional_requirements list")
        canonical = data.get("vocabulary", {}).get("canonical_labels")
        if not isinstance(canonical, dict) or not canonical:
            raise ValueError("Workshop FM contract requires vocabulary.canonical_labels")
        forbidden_fragments = (
            "min_reach", "minimum_reach", "min_length", "max_diameter",
            "min_area", "minimum_area", "min_volume", "minimum_volume",
        )
        def keys(value: Any) -> list[str]:
            if isinstance(value, dict):
                return [str(k).lower() for k in value] + [item for child in value.values() for item in keys(child)]
            if isinstance(value, list):
                return [item for child in value for item in keys(child)]
            return []
        offending = sorted({key for key in keys(data) if any(fragment in key for fragment in forbidden_fragments)})
        if offending:
            raise ValueError("FM contract contains deterministic metric thresholds: " + ", ".join(offending))
        return data

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
                    required_relations=list(r.get("required_relations", [])),
                    provenance="manual_workshop_fm_contract",
                )
            )
        return requirements

    def get_semantic_vocabulary(self) -> dict[str, list[str]]:
        entries = self._vocabulary_entries()
        return {
            canonical: list(dict.fromkeys([entry["detector_label"], *entry["aliases"]]))
            for canonical, entry in entries.items()
        }

    def _vocabulary_entries(self) -> dict[str, dict[str, Any]]:
        """Normalize both the current schema and legacy list-valued test contracts."""
        raw = self._contract_data.get("vocabulary", {}).get("canonical_labels", {})
        entries: dict[str, dict[str, Any]] = {}
        for insertion_index, (canonical, value) in enumerate(raw.items(), start=1):
            canonical = str(canonical).strip().lower()
            if isinstance(value, dict):
                detector_label = str(value.get("detector_label", canonical.replace("_", " "))).strip()
                aliases = [str(alias).strip() for alias in value.get("aliases", []) if str(alias).strip()]
                detector_rank = (int(value["detector_rank"])
                                 if "detector_rank" in value else None)
            else:
                aliases = [str(alias).strip() for alias in (value or []) if str(alias).strip()]
                detector_label = aliases[0] if aliases else canonical.replace("_", " ")
                detector_rank = None
            if not detector_label:
                raise ValueError(f"Canonical category {canonical!r} has an empty detector label")
            entries[canonical] = {
                "detector_label": detector_label,
                "aliases": aliases,
                "detector_rank": detector_rank,
            }
        return entries

    def get_ranked_detector_vocabulary(self) -> list[dict[str, Any]]:
        entries = self._vocabulary_entries()
        rank_by_category: dict[str, tuple[int, int]] = {}
        for requirement in sorted(self.get_requirements(), key=lambda item: item.rank):
            for category_index, category in enumerate(requirement.accepted_categories):
                key = category.lower()
                rank_by_category.setdefault(key, (requirement.rank, category_index))
        explicit_ranks = [entry["detector_rank"] for entry in entries.values()
                          if entry["detector_rank"] is not None]
        if len(explicit_ranks) != len(set(explicit_ranks)):
            raise ValueError("FM contract detector_rank values must be unique")
        ordered = sorted(entries, key=lambda key: (
            entries[key]["detector_rank"] if entries[key]["detector_rank"] is not None else 10_000,
            *rank_by_category.get(key, (10_000, 10_000)),
            list(entries).index(key),
        ))
        return [
            {
                "canonical_label": canonical,
                "detector_label": entries[canonical]["detector_label"],
                "aliases": list(entries[canonical]["aliases"]),
                "detector_rank": (entries[canonical]["detector_rank"]
                                  if entries[canonical]["detector_rank"] is not None
                                  else ordered.index(canonical) + 1),
                "role_rank": list(rank_by_category.get(canonical, (10_000, 10_000))),
            }
            for canonical in ordered
        ]

    def get_detector_prompts(self) -> list[str]:
        prompts = [entry["detector_label"] for entry in self.get_ranked_detector_vocabulary()]
        if len(prompts) != len(set(label.lower() for label in prompts)):
            raise ValueError("FM contract detector labels must be unique")
        return prompts

    def get_detector_label_to_canonical_map(self) -> dict[str, str]:
        return {
            entry["detector_label"].lower(): entry["canonical_label"]
            for entry in self.get_ranked_detector_vocabulary()
        }

    def get_alias_to_canonical_map(self) -> dict[str, str]:
        entries = self._vocabulary_entries()
        mapping: dict[str, str] = {}
        for canonical, entry in entries.items():
            mapping[canonical.lower()] = canonical
            mapping[entry["detector_label"].lower()] = canonical
            for alias in entry["aliases"]:
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

    def get_ranked_detector_vocabulary(self) -> list[dict[str, Any]]:
        raise FMBackendNotConfiguredError("Live FM backend not configured.")

    def get_detector_label_to_canonical_map(self) -> dict[str, str]:
        raise FMBackendNotConfiguredError("Live FM backend not configured.")

    def get_alias_to_canonical_map(self) -> dict[str, str]:
        raise FMBackendNotConfiguredError("Live FM backend not configured.")
