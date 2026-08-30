"""Functional requirement providers and manual future-FM contract for Workshop Phase 1."""

from __future__ import annotations

import abc
from pathlib import Path
import re
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
    def get_requirements(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> list[FunctionalRequirement]:
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

    def get_requirements(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> list[FunctionalRequirement]:
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


WORKSHOP_SEARCH_REGIONS = {
    "LEFT_DRAWER": "left storage drawer below workbench",
    "RIGHT_DRAWER": "right storage drawer below workbench",
    "TOOL_CABINET": "tall tool cabinet to the right of workbench",
}


class FMRequirementProvider(RequirementProvider):
    """One-shot VLM generation guarded by generic Workshop ontology mapping.

    Qwen may use natural phrases. Only phrases that map to known functions,
    semantic categories, and qualitative relations cross into production grounding.
    Unknown, incomplete, or ambiguous output fails closed.
    """

    def __init__(
        self,
        fm_adapter: FMAdapter | None = None,
        ontology_contract: ManualWorkshopFMContract | None = None,
    ) -> None:
        self.fm_adapter = fm_adapter or FMAdapter()
        self.ontology_contract = ontology_contract or ManualWorkshopFMContract()
        self.raw_decomposition: dict[str, Any] | None = None
        self.region_ranking: tuple[str, ...] = ()
        self.candidate_regions: tuple[str, ...] = ()
        self._requirements: list[FunctionalRequirement] | None = None
        self._category_rank: dict[str, int] = {}
        normalization = self.ontology_contract._contract_data.get(
            "fm_normalization", {}
        )
        self._function_aliases = self._normalized_alias_table(
            normalization.get("function_aliases", {})
        )
        self._relation_aliases = self._normalized_alias_table(
            normalization.get("relation_aliases", {})
        )

    @staticmethod
    def _phrase(value: object) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))

    @classmethod
    def _normalized_alias_table(
        cls, raw: object
    ) -> dict[str, tuple[str, ...]]:
        if not isinstance(raw, dict):
            raise ValueError("fm_normalization alias tables must be mappings")
        result: dict[str, tuple[str, ...]] = {}
        for canonical, aliases in raw.items():
            if not isinstance(aliases, list) or not aliases:
                raise ValueError(f"FM normalization aliases missing for {canonical}")
            normalized = tuple(cls._phrase(alias) for alias in aliases)
            if not all(normalized):
                raise ValueError(f"FM normalization contains an empty alias for {canonical}")
            result[str(canonical)] = normalized
        return result

    @staticmethod
    def _contains_phrase(text: str, alias: str) -> bool:
        return text == alias or f" {alias} " in f" {text} "

    def _map_category(self, phrase: object) -> str | None:
        normalized = self._phrase(phrase)
        aliases = self.ontology_contract.get_alias_to_canonical_map()
        exact = {self._phrase(alias): canonical for alias, canonical in aliases.items()}
        if normalized in exact:
            return exact[normalized]
        matches = {
            canonical
            for alias, canonical in exact.items()
            if self._contains_phrase(normalized, alias)
        }
        if len(matches) == 1:
            return next(iter(matches))
        words = set(normalized.replace("_", " ").split())
        if words & {"driver", "screwdriver", "drill"}:
            return "power_driver" if "power" in words or "electric" in words or "cordless" in words else "screwdriver"
        if words & {"screw", "fastener", "bolt"}:
            return "screw"
        return None

    def _map_function(
        self, raw: dict[str, Any], categories: list[str]
    ) -> str:
        text = self._phrase(f"{raw['function']} {raw['description']}")
        words = set(text.split())
        scores: dict[str, int] = {}
        for function_name, aliases in self._function_aliases.items():
            score = 3 * sum(
                1
                for alias in aliases
                if self._contains_phrase(text, alias)
            )
            if function_name == "CAN_DRIVE_SCREW":
                score += sum(1 for w in ("drive", "driver", "screwdriver", "torque", "tighten", "turn") if w in words)
                if any(c in {"screwdriver", "power_driver", "power_drill"} for c in categories):
                    score += 2
            elif function_name == "CAN_FASTEN":
                score += sum(1 for w in ("fasten", "fastener", "screw", "thread", "secure") if w in words)
                if any(c in {"screw", "phillips_screw"} for c in categories):
                    score += 2
            scores[function_name] = score
        best = max(scores.values(), default=0)
        winners = [name for name, score in scores.items() if score == best and score > 0]
        if len(winners) != 1:
            raise ValueError(
                f"VLM_SPEC_FAILED: VLM function phrase {raw['function']!r} cannot be mapped uniquely to ontology; "
                f"scores={scores}"
            )
        return winners[0]

    def _map_relations(self, properties: list[str]) -> set[str]:
        mapped: set[str] = set()
        for property_text in properties:
            normalized = self._phrase(property_text)
            for relation, aliases in self._relation_aliases.items():
                if any(self._contains_phrase(normalized, alias) for alias in aliases):
                    mapped.add(relation)
            words = set(normalized.split())
            if words & {"reach", "reaches", "access", "accessible"}:
                mapped.add("REACHES_TARGET")
            compatibility = words & {
                "fit", "fits", "match", "matches", "compatible", "compatibility",
                "thread", "threads", "suitable",
            }
            if compatibility and words & {"hole", "workbench", "target"}:
                mapped.add("COMPATIBLE_WITH_TARGET")
            if (
                compatibility or words & {"torque", "interface"}
            ) and words & {"screw", "recess", "head", "interface"}:
                mapped.add("COMPATIBLE_WITH")
        return mapped

    def _ensure_generated(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> None:
        if self._requirements is not None:
            return
        document = self.fm_adapter.generate_task_requirements(
            task_instruction, observation_images=observation_images or []
        )
        self.raw_decomposition = document
        if document.get("status") != "SUPPORTED":
            raise ValueError(
                "VLM_SPEC_FAILED: VLM marked the Workshop task unsupported: "
                f"{document.get('unsupported_reason', 'no reason')}"
            )

        self.region_ranking = tuple(WORKSHOP_SEARCH_REGIONS.keys())
        self.candidate_regions = tuple(WORKSHOP_SEARCH_REGIONS.keys())

        normalized: dict[str, FunctionalRequirement] = {}
        raw_requirements = document["functional_requirements"]
        for rank_idx, raw in enumerate(raw_requirements, start=1):
            if raw.get("entity_kind") != "OBJECT":
                continue
            categories: list[str] = []
            for candidate in raw.get("candidate_objects", []):
                for field in ("label", "visual_description", "suitability_reason"):
                    val = candidate.get(field)
                    if val:
                        canonical = self._map_category(val)
                        if canonical is not None and canonical not in categories:
                            categories.append(canonical)
                            self._category_rank.setdefault(canonical, len(self._category_rank) + 1)
                            break
            if not categories and not raw.get("candidate_objects"):
                for phrase in (raw.get("id"), raw.get("description"), raw.get("function")):
                    if phrase:
                        cat = self._map_category(phrase)
                        if cat and cat not in categories:
                            categories.append(cat)
                            self._category_rank.setdefault(cat, len(self._category_rank) + 1)
            if not categories:
                raise ValueError(
                    f"VLM_SPEC_FAILED: VLM candidates for role {raw['id']!r} could not be mapped to supported categories"
                )
            function_name = self._map_function(raw, categories)
            if function_name in normalized:
                raise ValueError(f"VLM_SPEC_FAILED: VLM emitted duplicate role {function_name}")
            if raw.get("required_count", 1) != 1:
                raise ValueError(
                    f"VLM_SPEC_FAILED: VLM role {function_name} required_count must be 1 for Workshop"
                )
            mapped_relations = self._map_relations(raw.get("required_properties", []))

            normalized[function_name] = FunctionalRequirement(
                requirement_id=raw["id"],
                entity_type=EntityType.OBJECT,
                function_name=function_name,
                description=raw.get("description", ""),
                rank=rank_idx,
                source=RequirementSource.FM,
                accepted_categories=list(categories),
                semantic_hints=[
                    candidate["label"] for candidate in raw.get("candidate_objects", [])
                ],
                geometric_constraints={},
                required_relations=list(mapped_relations),
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )

        if not normalized:
            raise ValueError("VLM_SPEC_FAILED: No functional requirements produced by VLM")

        self._requirements = sorted(normalized.values(), key=lambda item: item.rank)

    def generate_inspection_policy(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> tuple[str, ...]:
        images = list(observation_images or [])
        if not images:
            raise ValueError("VLM_SPEC_FAILED: Observation images required for VLM inspection policy")
        priors = self.fm_adapter.generate_inspection_priors(
            task_instruction,
            WORKSHOP_SEARCH_REGIONS,
            observation_images=images,
        )
        order = tuple(item["region_id"] for item in priors.get("inspection_order", []))
        if not order or set(order) != set(WORKSHOP_SEARCH_REGIONS.keys()):
            raise ValueError(f"VLM_SPEC_FAILED: VLM returned invalid inspection order: {order}")
        self.region_ranking = order
        return self.region_ranking

    def get_requirements(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> list[FunctionalRequirement]:
        self._ensure_generated(
            task_instruction, observation_images=observation_images
        )
        return list(self._requirements or [])

    def get_semantic_vocabulary(self) -> dict[str, list[str]]:
        self._ensure_generated()
        return self.ontology_contract.get_semantic_vocabulary()

    def get_ranked_detector_vocabulary(self) -> list[dict[str, Any]]:
        self._ensure_generated()
        entries = self.ontology_contract.get_ranked_detector_vocabulary()
        # Relevant VLM alternatives retain their generated order. Benchmark-
        # observable negative controls remain after them so detector evaluation
        # is stable and does not silently remove the hammer distractor.
        return sorted(
            entries,
            key=lambda entry: (
                self._category_rank.get(entry["canonical_label"], 10_000),
                entry["detector_rank"],
            ),
        )

    def get_detector_prompts(self) -> list[str]:
        return [
            entry["detector_label"]
            for entry in self.get_ranked_detector_vocabulary()
        ]

    def get_detector_label_to_canonical_map(self) -> dict[str, str]:
        self._ensure_generated()
        return self.ontology_contract.get_detector_label_to_canonical_map()

    def get_alias_to_canonical_map(self) -> dict[str, str]:
        self._ensure_generated()
        return self.ontology_contract.get_alias_to_canonical_map()
