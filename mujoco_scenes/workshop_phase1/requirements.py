"""Functional requirement providers and manual future-FM contract for Workshop Phase 1."""

from __future__ import annotations

import abc
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import yaml

from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
    VLMSpecificationError,
)
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


@dataclass(frozen=True)
class NormalizedWorkshopRole:
    raw_role_id: str
    canonical_role_id: str
    entity_kind: str  # "OBJECT", "FIXED_TARGET"
    raw_function: str
    canonical_function: str
    required_count: int
    binding_policy: str  # "DISTINCT", "REUSABLE", "SHARED"
    candidate_categories: tuple[str, ...]
    run_local_categories: tuple[str, ...]
    visible_candidates: tuple[dict[str, Any], ...]
    unary_predicates: tuple[str, ...]
    description: str
    semantic_hints: tuple[str, ...]
    provenance: str


@dataclass(frozen=True)
class NormalizedWorkshopRelation:
    raw_subject_role_id: str
    canonical_subject_role_id: str
    raw_relation_text: str
    canonical_predicate: str
    raw_object_role_id: str
    canonical_object_role_id: str


WORKSHOP_SEARCH_REGIONS = {
    "LEFT_DRAWER": "left storage drawer below workbench",
    "RIGHT_DRAWER": "right storage drawer below workbench",
    "TOOL_CABINET": "tall tool cabinet to the right of workbench",
}


WORKSHOP_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "LEFT_DRAWER": (
        "left storage drawer", "left drawer", "left workbench drawer",
        "drawer on the left", "left storage drawer below workbench",
        "left desk drawer", "left lower drawer", "left table drawer", "left_drawer",
    ),
    "RIGHT_DRAWER": (
        "right storage drawer", "right drawer", "right workbench drawer",
        "drawer on the right", "right storage drawer below workbench",
        "right desk drawer", "right lower drawer", "right table drawer", "right_drawer",
    ),
    "TOOL_CABINET": (
        "tool cabinet", "cabinet", "wall cabinet", "upper cabinet",
        "tool storage cabinet", "storage cabinet", "overhead tool cabinet",
        "upper tool storage", "wall tool cabinet", "tool_cabinet",
    ),
}


def map_workshop_fixed_target_role(raw: dict[str, Any]) -> str | None:
    """Deterministic concept matching for Workshop contextual FIXED_TARGET roles."""
    if raw.get("entity_kind") != "FIXED_TARGET":
        return None
    fn_desc = f"{raw.get('function', '')} {raw.get('description', '')}"
    norm = " ".join(re.findall(r"[a-z0-9]+", fn_desc.casefold()))
    if not norm:
        return None
    has_target_concept = any(
        k in norm for k in (
            "repair hole", "repair target", "target hole", "workbench hole",
            "frame joint", "joint hole", "workpiece hole", "threaded fastener target",
            "fastener insertion point", "accept threaded fastener", "accept screw",
            "insertion hole", "pre drilled hole", "hole in frame", "hole in workpiece",
        )
    )
    if has_target_concept:
        return "repair_target"
    return None


def resolve_workshop_region_proposal(proposal: dict[str, Any] | str) -> str | None:
    """Resolve visual region proposal to canonical region ID using ONLY label and visual_description.

    VLM-local IDs (e.g. 'id', 'region_id') are NOT semantic evidence and must be ignored.
    """
    if isinstance(proposal, str):
        text = proposal.strip().lower()
    else:
        text = f"{proposal.get('label', '')} {proposal.get('visual_description', '')}".strip().lower()
    norm = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    if not norm:
        return None
    matches = set()
    for reg_id, aliases in WORKSHOP_REGION_ALIASES.items():
        for alias in aliases:
            a_norm = " ".join(re.findall(r"[a-z0-9]+", alias.casefold()))
            if a_norm == norm or f" {a_norm} " in f" {norm} ":
                matches.add(reg_id)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(f"Ambiguous workshop region proposal {proposal!r} matches multiple regions: {sorted(matches)}")
    return None


def map_workshop_role_function(function_text: str) -> str | None:
    """Deterministic compositional concept matching for Workshop functional roles."""
    norm = " ".join(re.findall(r"[a-z0-9]+", str(function_text).casefold()))
    words = set(norm.split())

    has_driver_action = any(w in words or w in norm for w in ("drive", "tighten", "turn", "torque", "driving", "tightening", "turning", "screwing", "driver", "drill"))
    has_fastener_concept = any(w in words or w in norm for w in ("fasten", "secure", "join", "thread", "anchoring", "fastening", "securing", "joining", "anchor", "fastener", "threaded fastener"))

    if has_driver_action and not any(phrase in norm for phrase in ("fastener to", "fastener that", "to be driven", "threaded fastener to", "parts together")):
        return "CAN_DRIVE_SCREW"
    if has_fastener_concept and not has_driver_action:
        return "CAN_FASTEN"
    if "fastener" in norm and not has_driver_action:
        return "CAN_FASTEN"
    return None


def map_workshop_relation(relation_text: str) -> str | None:
    """Deterministic concept matching for Workshop relations."""
    norm = " ".join(re.findall(r"[a-z0-9]+", str(relation_text).casefold()))
    matches = set()
    if any(k in norm for k in ("engage", "fit screw", "fits driver", "fit driver", "driver bit", "fit fastener", "match bit", "compatible with fastener", "compatible with screw", "torque to screw", "compatible with the fastener", "compatible with")):
        matches.add("COMPATIBLE_WITH")
    if any(k in norm for k in ("reaches target", "reach target", "reaches into", "reach into", "reaches hole", "reach hole", "reaches repair", "reach repair", "access target", "length to reach", "reaches workpiece", "reach workpiece")):
        matches.add("REACHES_TARGET")
    if any(k in norm for k in ("thread into", "threads into", "fit hole", "fits hole", "fit target", "fits target", "anchor in", "anchors in", "compatible with hole", "compatible with target", "fits workpiece", "fit inside", "fits inside")):
        matches.add("COMPATIBLE_WITH_TARGET")
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(f"Ambiguous workshop relation {relation_text!r} matches multiple relations: {sorted(matches)}")
    return None


def map_workshop_unary_property(property_text: str) -> str | None:
    """Deterministic concept matching for Workshop unary properties."""
    norm = " ".join(re.findall(r"[a-z0-9]+", str(property_text).casefold()))
    if not norm:
        return None
    if any(k in norm for k in ("planar support", "planar_support", "flat surface")):
        return "PLANAR_SUPPORT"
    if any(k in norm for k in ("open cavity", "open_cavity", "container", "hollow")):
        return "OPEN_CAVITY"
    return None


class FMRequirementProvider(RequirementProvider):
    """Dynamic VLM requirement provider for Workshop domain."""

    def __init__(
        self,
        fm_adapter: FMAdapter | None = None,
        ontology_contract: ManualWorkshopFMContract | None = None,
    ) -> None:
        self.fm_adapter = fm_adapter or FMAdapter()
        self.ontology_contract = ontology_contract or ManualWorkshopFMContract()
        self._requirements: list[FunctionalRequirement] | None = None
        self._category_rank: dict[str, int] = {}
        self.raw_vlm_response: dict[str, Any] | None = None
        self.validated_vlm_specification: dict[str, Any] | None = None
        self.raw_decomposition: dict[str, Any] | None = None
        self.region_ranking: tuple[str, ...] = ()
        self.candidate_regions: tuple[str, ...] = ()
        self.transformation_trace: list[dict[str, Any]] = []
        self.normalized_roles: list[NormalizedWorkshopRole] = []
        self.normalized_relations: list[NormalizedWorkshopRelation] = []
        self.vlm_derived_detector_prompts: tuple[str, ...] = ()
        self.evaluation_negative_control_prompts: tuple[str, ...] = (
            "claw hammer", "ball peen hammer", "sledgehammer"
        )

        normalization = self.ontology_contract._contract_data.get(
            "fm_normalization", {}
        )
        self._function_aliases = self._normalized_alias_table(
            normalization.get("function_aliases", {})
        )
        self._relation_aliases = self._normalized_alias_table(
            normalization.get("relation_aliases", {})
        )

    @classmethod
    def _phrase(cls, value: object) -> str:
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
        if not normalized:
            return None
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
        if len(matches) > 1:
            raise VLMSpecificationError(f"Ambiguous category match for {phrase!r}: {sorted(matches)}")
        return None

    def _map_function(
        self, raw: dict[str, Any]
    ) -> str:
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
        comp = map_workshop_role_function(text)
        if comp is not None:
            return comp
        norm_text = self._phrase(text)
        matches = set()
        for function_name, aliases in self._function_aliases.items():
            for alias in aliases:
                if self._contains_phrase(norm_text, alias):
                    matches.add(function_name)
                    break
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) == 0:
            raise UnmappedFunctionalConceptError(
                f"VLM_SPEC_FAILED: VLM function phrase {raw.get('function')!r} cannot be mapped to any reviewed ontology function"
            )
        raise AmbiguousCanonicalizationError(
            f"VLM_SPEC_FAILED: VLM function phrase {raw.get('function')!r} is ambiguous across functions: {sorted(matches)}"
        )

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
        self.raw_vlm_response = deepcopy(
            getattr(self.fm_adapter, "last_raw_requirement_response", None) or document
        )
        self.validated_vlm_specification = deepcopy(document)
        self.raw_decomposition = document  # legacy alias
        if document.get("status") != "SUPPORTED":
            raise VLMSpecificationError(
                "VLM_SPEC_FAILED: VLM marked the Workshop task unsupported: "
                f"{document.get('unsupported_reason', 'no reason')}"
            )

        raw_requirements = document.get("functional_roles", [])
        self.transformation_trace = []
        raw_id_to_canon: dict[str, str] = {}
        normalized_roles: list[NormalizedWorkshopRole] = []
        legacy_requirements: dict[str, FunctionalRequirement] = {}

        for rank_idx, raw in enumerate(raw_requirements, start=1):
            raw_kind = raw.get("entity_kind")
            raw_id = raw.get("id")
            required_count = int(raw.get("required_count", 1))
            binding_policy = str(raw.get("binding_policy", "DISTINCT"))

            if raw_kind == "FIXED_TARGET":
                canonical_target = map_workshop_fixed_target_role(raw)
                if canonical_target is None:
                    fn_text = f"{raw.get('function', '')} {raw.get('description', '')}"
                    raise VLMSpecificationError(f"VLM_SPEC_FAILED: Unsupported FIXED_TARGET role {raw_id!r}: {fn_text}")
                raw_id_to_canon[raw_id] = canonical_target
                cand_cats = tuple(str(c).strip() for c in raw.get("candidate_categories", []) if str(c).strip())
                run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats)) or ("repair_target",)
                cand_objs = tuple(raw.get("visible_candidates", []))
                hints = tuple(candidate["label"] for candidate in cand_objs if candidate.get("label"))

                normalized_role = NormalizedWorkshopRole(
                    raw_role_id=raw_id,
                    canonical_role_id=canonical_target,
                    entity_kind="FIXED_TARGET",
                    raw_function=str(raw.get("function", "")),
                    canonical_function="FIXED_TARGET",
                    required_count=required_count,
                    binding_policy=binding_policy,
                    candidate_categories=cand_cats,
                    run_local_categories=run_local_cats,
                    visible_candidates=cand_objs,
                    unary_predicates=(),
                    description=str(raw.get("description", "")),
                    semantic_hints=hints,
                    provenance="vlm_explicit_fixed_target",
                )
                normalized_roles.append(normalized_role)
                self.transformation_trace.append({
                    "raw_role": raw_id,
                    "raw_entity_kind": raw_kind,
                    "transformation": "SYSTEM_OWNED_FIXED_TARGET_REPRESENTATION",
                    "canonical_role": canonical_target,
                })
                continue
            elif raw_kind == "REGION":
                raise VLMSpecificationError(f"VLM_SPEC_FAILED: Unsupported REGION role {raw_id!r} in Workshop specification")
            elif raw_kind != "OBJECT":
                raise VLMSpecificationError(f"VLM_SPEC_FAILED: Invalid entity_kind {raw_kind!r} for role {raw_id!r}")

            raw_cand_cats = [str(c).strip() for c in raw.get("candidate_categories", []) if str(c).strip()]
            if not raw_cand_cats:
                raise VLMSpecificationError(f"VLM_SPEC_FAILED: Role {raw_id!r} has no candidate categories")
            cand_cats = tuple(raw_cand_cats)
            run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats))

            for cat in cand_cats:
                canon_c = self._map_category(cat)
                if canon_c is not None:
                    self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)

            cand_objs = tuple(raw.get("visible_candidates", []))
            for candidate in cand_objs:
                for field in ("label", "visual_description"):
                    val = candidate.get(field)
                    if val:
                        canon_c = self._map_category(val)
                        if canon_c is not None:
                            self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
                            break
            hints = tuple(candidate["label"] for candidate in cand_objs if candidate.get("label"))

            function_name = self._map_function(raw)
            canonical_role_id = "driver" if function_name == "CAN_DRIVE_SCREW" else ("fastener" if function_name == "CAN_FASTEN" else raw_id)
            if canonical_role_id in raw_id_to_canon.values():
                raise VLMSpecificationError(f"VLM_SPEC_FAILED: VLM emitted duplicate role {canonical_role_id}")
            raw_id_to_canon[raw_id] = canonical_role_id

            raw_props = raw.get("required_properties", [])
            unary_predicates: list[str] = []
            for prop in raw_props:
                mapped_u = map_workshop_unary_property(prop)
                if mapped_u is None:
                    raise UnsupportedCheckerCapabilityError(
                        f"VLM_SPEC_FAILED: VLM unary property {prop!r} for role {raw_id!r} is not supported in Workshop"
                    )
                if mapped_u not in unary_predicates:
                    unary_predicates.append(mapped_u)

            normalized_role = NormalizedWorkshopRole(
                raw_role_id=raw_id,
                canonical_role_id=canonical_role_id,
                entity_kind="OBJECT",
                raw_function=str(raw.get("function", "")),
                canonical_function=function_name,
                required_count=required_count,
                binding_policy=binding_policy,
                candidate_categories=cand_cats,
                run_local_categories=run_local_cats,
                visible_candidates=cand_objs,
                unary_predicates=tuple(unary_predicates),
                description=str(raw.get("description", "")),
                semantic_hints=hints,
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
            normalized_roles.append(normalized_role)

            legacy_requirements[function_name] = FunctionalRequirement(
                requirement_id=raw["id"],
                entity_type=EntityType.OBJECT,
                function_name=function_name,
                description=raw.get("description", ""),
                rank=rank_idx,
                source=RequirementSource.FM,
                accepted_categories=list(cand_cats),
                semantic_hints=list(hints),
                geometric_constraints={},
                required_relations=list(unary_predicates),
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )

        normalized_relations: list[NormalizedWorkshopRelation] = []
        for rel in document.get("functional_relations", []):
            s = rel.get("subject_role")
            r = rel.get("relation")
            o = rel.get("object_role")
            if s not in raw_id_to_canon:
                raise MalformedVLMSpecificationError(f"VLM_SPEC_FAILED: Relation subject role {s!r} not declared")
            if o not in raw_id_to_canon:
                raise MalformedVLMSpecificationError(f"VLM_SPEC_FAILED: Relation object role {o!r} not declared")
            s_canon = raw_id_to_canon[s]
            o_canon = raw_id_to_canon[o]
            mapped_rel = map_workshop_relation(r)
            if mapped_rel is None:
                norm_r = self._phrase(r)
                prop_matches = set()
                for relation_name, aliases in self._relation_aliases.items():
                    for alias in aliases:
                        if self._contains_phrase(norm_r, alias):
                            prop_matches.add(relation_name)
                            break
                if len(prop_matches) == 1:
                    mapped_rel = next(iter(prop_matches))
                elif len(prop_matches) > 1:
                    raise AmbiguousCanonicalizationError(f"VLM_SPEC_FAILED: Ambiguous relation {r!r}: {sorted(prop_matches)}")
                else:
                    raise UnmappedFunctionalConceptError(f"VLM_SPEC_FAILED: Relation {r!r} cannot be mapped to any Workshop relation")

            normalized_relations.append(NormalizedWorkshopRelation(
                raw_subject_role_id=s,
                canonical_subject_role_id=s_canon,
                raw_relation_text=r,
                canonical_predicate=mapped_rel,
                raw_object_role_id=o,
                canonical_object_role_id=o_canon,
            ))

            func_key = "CAN_DRIVE_SCREW" if s_canon == "driver" else ("CAN_FASTEN" if s_canon == "fastener" else s_canon)
            if func_key in legacy_requirements:
                if mapped_rel not in legacy_requirements[func_key].required_relations:
                    legacy_requirements[func_key].required_relations.append(mapped_rel)

        if not normalized_roles:
            raise VLMSpecificationError("VLM_SPEC_FAILED: No functional requirements produced by VLM")

        self.normalized_roles = normalized_roles
        self.normalized_relations = normalized_relations
        self._requirements = sorted(legacy_requirements.values(), key=lambda item: item.rank)

        # Resolve regions from the single initial response
        resolved_map: dict[str, str] = {}
        for item in document.get("inspectable_regions", []):
            if isinstance(item, dict):
                prop_id = item.get("id") or item.get("region_id") or ""
                canon_reg = resolve_workshop_region_proposal(item)
                if canon_reg is not None and canon_reg in WORKSHOP_SEARCH_REGIONS:
                    resolved_map[prop_id] = canon_reg
            elif isinstance(item, str):
                canon_reg = resolve_workshop_region_proposal(item)
                if canon_reg is not None and canon_reg in WORKSHOP_SEARCH_REGIONS:
                    resolved_map[item] = canon_reg

        raw_order = document.get("inspection_order", [])
        order = []
        for raw_item in raw_order:
            if isinstance(raw_item, dict):
                raw_id = raw_item.get("id") or raw_item.get("region_id") or ""
                canon = resolved_map.get(raw_id)
            else:
                raw_id = str(raw_item)
                canon = resolved_map.get(raw_id)
            if canon is not None and canon in WORKSHOP_SEARCH_REGIONS and canon not in order:
                order.append(canon)

        self.region_ranking = tuple(order)
        self.candidate_regions = tuple(dict.fromkeys(resolved_map.values()))

        # VLM derived detector vocabulary
        vlm_prompts = list(dict.fromkeys(
            cat
            for r in self.normalized_roles
            if r.entity_kind == "OBJECT"
            for cat in r.candidate_categories
        ))
        self.vlm_derived_detector_prompts = tuple(vlm_prompts)

    def generate_inspection_policy(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
    ) -> tuple[str, ...]:
        if self.region_ranking:
            return self.region_ranking
        self._ensure_generated(
            task_instruction, observation_images=observation_images
        )
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
        mapping = dict(self.ontology_contract.get_detector_label_to_canonical_map())
        for role in getattr(self, "normalized_roles", []):
            for prompt, token in zip(role.candidate_categories, role.run_local_categories):
                mapping[prompt.lower()] = token
        return mapping

    def get_alias_to_canonical_map(self) -> dict[str, str]:
        self._ensure_generated()
        mapping = dict(self.ontology_contract.get_alias_to_canonical_map())
        for role in getattr(self, "normalized_roles", []):
            for prompt, token in zip(role.candidate_categories, role.run_local_categories):
                mapping[prompt.lower()] = token
                mapping[token.lower()] = token
        return mapping

