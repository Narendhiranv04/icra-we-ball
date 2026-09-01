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
from mujoco_scenes.functional_tamp_pipeline.models import OperationGroup
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter, FMBackendNotConfiguredError
from mujoco_scenes.workshop_phase1.types import (
    EntityType,
    FunctionalRequirement,
    RequirementSource,
)

DEFAULT_FM_CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "workshop_phase1_fm_contract.yaml"
)

WORKSHOP_VLM_CANONICALIZATION_VERSION = "phase3_p3g_3_v1"

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


def _phrase(value: object) -> str:
    """Normalize arbitrary string to lower-case alphanumeric single-spaced words."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _contains_phrase(text: str, alias: str) -> bool:
    """Check if alias appears as an exact string or word-bounded substring in text."""
    return text == alias or f" {alias} " in f" {text} "


def _is_explicit_alternative_role(function_text: str, description_text: str) -> bool:
    """Detect whether a raw role explicitly declares itself as an alternative option."""
    norm = _phrase(f"{function_text} {description_text}")
    return any(
        phrase in norm
        for phrase in (
            "alternative driver option",
            "alternative fastener option",
            "alternative option",
            "interchangeable tool candidate",
            "interchangeable fastener candidate",
            "interchangeable candidate",
            "either manual or powered",
            "either manual or power",
            "alternative tool",
            "alternative fastener",
            "substitute tool",
            "substitute fastener",
        )
    )


def map_workshop_context_region_role(raw: dict[str, Any] | str) -> str | None:
    """Deterministic concept matching for Workshop contextual REGION roles using ONLY function and description."""
    if isinstance(raw, dict):
        if raw.get("entity_kind") != "REGION":
            return None
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None
    if any(
        k in norm
        for k in (
            "workbench", "main workbench zone", "table", "desk", "bench",
            "support workpiece", "work surface", "support surface",
            "work table", "table surface", "workbench context",
            "bench surface", "bench support",
        )
    ):
        return "MAIN_WORKBENCH_ZONE"
    return None


def map_workshop_fixed_target_role(raw: dict[str, Any] | str) -> str | None:
    """Deterministic concept matching for Workshop contextual FIXED_TARGET roles using ONLY function and description."""
    if isinstance(raw, dict):
        if raw.get("entity_kind") != "FIXED_TARGET":
            return None
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None
    has_target_concept = any(
        k in norm
        for k in (
            "repair hole", "repair target", "target hole", "workbench hole",
            "frame joint", "joint hole", "workpiece hole", "threaded fastener target",
            "fastener insertion point", "accept threaded fastener", "accept screw",
            "insertion hole", "pre drilled hole", "hole in frame", "hole in workpiece",
            "accept screw insertion", "accept_screw_insertion", "mounting point",
            "target joint hole", "target joint", "loose frame joint", "repair the frame",
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
        raise AmbiguousCanonicalizationError(
            f"Ambiguous workshop region proposal {proposal!r} matches multiple regions: {sorted(matches)}"
        )
    return None


def map_workshop_role_function(raw: dict[str, Any] | str) -> str | None:
    """Deterministic multi-signal concept matching for Workshop functional roles using ONLY function and description."""
    if isinstance(raw, dict):
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None
    words = set(norm.split())

    driver_phrases = (
        "drive screw", "tighten screw", "turn threaded fastener", "turn screw",
        "rotate screw", "apply torque", "driving tool", "tightening tool",
        "screwdriver", "driver tool", "power driver", "cordless power drill",
        "manual driver", "torque tool", "power drill", "hex driver",
        "slotted driver", "phillips driver", "phillips screwdriver",
        "fastener driving tool", "screw driving tool", "tool capable of driving",
        "device that rotates the screw", "rotates the screw", "rotates screw",
        "tool capable of driving a screw", "tool to tighten screws",
    )
    driver_tokens = (
        "screwdriver", "screwdrivers", "drill", "drills", "driver", "drivers",
        "tool", "tools", "bit", "bits", "wrench", "wrenches",
    )

    fastener_phrases = (
        "fasten joint", "secure joint with threaded fastener", "secure the joint",
        "secure joint", "screw to join parts", "threaded fastener",
        "insert screw into target", "screw to secure", "fastener to secure",
        "securing fastener", "threaded component", "threaded component that holds",
        "threaded screw", "machine screw", "wood screw", "phillips screw",
        "fasten the frame", "fasten frame", "threaded fastener capable of",
        "fastener that joins parts", "threaded fastener to hold parts",
        "secure joint and anchor", "screw inserted into",
    )
    fastener_tokens = (
        "screw", "screws", "fastener", "fasteners", "bolt", "bolts", "hardware", "joiner",
    )

    has_driver_phrase = any(p in norm for p in driver_phrases)
    has_fastener_phrase = any(p in norm for p in fastener_phrases)

    # Check for passive/object fastener indicators
    is_fastener_target = any(
        p in norm for p in (
            "to be driven", "screw to be driven", "fastener to be driven",
            "threaded fastener to hold", "threaded fastener capable",
            "fastener capable of", "threaded component", "fastener that joins",
            "fastener that holds", "secure joint and anchor", "screw inserted into",
            "to hold parts", "to join parts", "threads into",
        )
    )

    # Action verb analysis
    has_driver_action = any(
        w in words for w in ("tighten", "tightening", "torque", "torquing", "turning", "screwing", "drive", "driving")
    )
    has_fastener_action = any(
        w in words for w in ("fasten", "fastening", "anchor", "anchoring")
    )

    if has_driver_action and not is_fastener_target:
        return "CAN_DRIVE_SCREW"

    if is_fastener_target and not any(p in norm for p in ("tool to", "tool for", "tool capable", "driver", "screwdriver", "drill")):
        return "CAN_FASTEN"

    if has_driver_phrase and not is_fastener_target:
        return "CAN_DRIVE_SCREW"

    if has_fastener_phrase and not any(w in words for w in ("driver", "screwdriver", "drill", "wrench")):
        return "CAN_FASTEN"

    if has_fastener_action or any(w in words for w in fastener_tokens):
        if not any(w in words for w in driver_tokens) and not has_driver_action:
            return "CAN_FASTEN"

    return None


def map_workshop_relation(relation_text: str) -> str | None:
    """Deterministic concept matching for Workshop relations."""
    norm = _phrase(relation_text)
    if not norm:
        return None
    matches = set()
    if any(_contains_phrase(norm, k) for k in (
        "engage", "engages", "engage screw", "engages screw",
        "driver engages screw", "driver engages", "fit screw", "fits screw",
        "fits driver", "fit driver", "driver bit", "fit fastener", "fits fastener",
        "match bit", "compatible with fastener", "compatible with screw", "torque to screw",
        "compatible with the fastener", "compatible with", "drives", "drives screw",
        "transmits torque", "transmit torque", "fit the screw head and transmit torque",
        "tip must fit the screw head and transmit torque", "fit screw head", "fits screw head",
        "driver bit matches fastener",
        "is driven by", "driven by", "is engaged by", "engaged by",
        "receives torque from", "driven by tool", "is driven by tool",
        "is turned by", "turned by", "receives drive from",
    )):
        matches.add("COMPATIBLE_WITH")
    if any(_contains_phrase(norm, k) for k in (
        "reaches target", "reach target", "reaches into", "reach into",
        "reaches hole", "reach hole", "reaches repair", "reach repair",
        "access target", "accesses target", "length to reach", "reaches workpiece", "reach workpiece",
        "must reach the workpiece hole recess", "long enough to reach workpiece hole recess",
        "long enough to reach hole", "long enough to reach",
        "reaches workpiece hole", "reach workpiece hole", "reach workpiece hole recess",
        "is reached by", "reached by", "target reached by", "accessed by", "target accessed by", "is accessed by",
    )):
        matches.add("REACHES_TARGET")
    if any(_contains_phrase(norm, k) for k in (
        "thread into", "threads into", "fit hole", "fits hole", "fit target",
        "fits target", "anchor in", "anchors in", "compatible with hole",
        "compatible with target", "fits workpiece", "fit workpiece", "fit inside", "fits inside",
        "fits into", "fit into", "inserted into", "insert into", "inserts into",
        "screw inserted into", "screw fits into",
        "must fit the workbench target hole and thread into the hole",
        "threads into target repair hole", "thread into target repair hole",
        "threads into hole", "thread into hole", "fits the hole", "fit the hole",
        "receives fastener", "receives screw", "is fastened by", "fastened by",
        "is threaded by", "threaded by", "fastened with",
    )):
        matches.add("COMPATIBLE_WITH_TARGET")
    if any(_contains_phrase(norm, k) for k in (
        "located in", "located on", "located on workbench", "placed on", "placed on workbench",
        "on surface", "on workbench", "supported by workbench", "rests on workbench",
        "supports repair target", "supports target", "holds repair target", "provides support for repair target",
        "supports", "held by",
    )):
        matches.add("LOCATED_ON")

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        if "COMPATIBLE_WITH" in matches and any(_contains_phrase(norm, k) for k in ("engage", "engages", "driver", "bit", "torque", "driven", "drives")):
            return "COMPATIBLE_WITH"
        if "COMPATIBLE_WITH_TARGET" in matches and any(_contains_phrase(norm, k) for k in ("hole", "target", "thread", "threads", "insert", "fastener", "screw")):
            return "COMPATIBLE_WITH_TARGET"
        raise AmbiguousCanonicalizationError(f"Ambiguous workshop relation {relation_text!r} matches multiple relations: {sorted(matches)}")
    return None


def map_workshop_unary_property(property_text: str) -> str | None:
    """Deterministic concept matching for Workshop unary properties."""
    norm = _phrase(property_text)
    if not norm:
        return None
    if any(_contains_phrase(norm, k) for k in ("planar support", "planar_support", "flat surface", "horizontal surface", "is_flat", "is_horizontal", "planar horizontal support", "horizontal planar support")):
        return "PLANAR_SUPPORT"
    if any(_contains_phrase(norm, k) for k in ("open cavity", "open_cavity", "container", "hollow", "capable of holding liquid")):
        return "OPEN_CAVITY"
    if any(_contains_phrase(norm, k) for k in ("elongated", "elongated_object", "slender", "shank", "elongated shape")):
        return "ELONGATED_OBJECT"
    return None


def canonicalize_workshop_relation(
    raw_subject_id: str,
    raw_subject_canon: str,
    raw_relation_text: str,
    raw_object_id: str,
    raw_object_canon: str,
) -> tuple[str, str, str, str, str]:
    """Deterministic signature-aware relation canonicalization for Workshop domain.

    Returns:
        (canonical_subject, canonical_predicate, canonical_object, direction_status, structural_destination)
    """
    if raw_subject_id == raw_object_id or (raw_subject_canon == raw_object_canon and raw_subject_canon not in ("UNKNOWN", "MAIN_WORKBENCH_ZONE")):
        raise MalformedVLMSpecificationError(
            f"Self-relations are not supported in Workshop domain: {raw_subject_id} -[{raw_relation_text}]-> {raw_object_id}"
        )

    norm_rel = _phrase(raw_relation_text)
    if not norm_rel:
        raise MalformedVLMSpecificationError(
            f"Empty relation text between {raw_subject_id} and {raw_object_id}"
        )

    # Check for unsupported functional LOCATED_ON
    is_loc_phrase = any(_contains_phrase(norm_rel, k) for k in (
        "located on", "located on workbench", "located in", "placed on", "placed on workbench",
        "on surface", "on workbench", "supported by workbench", "rests on workbench",
    ))
    if is_loc_phrase:
        if raw_subject_canon in ("driver", "fastener"):
            raise UnsupportedCheckerCapabilityError(
                f"Functional relation LOCATED_ON on role {raw_subject_id!r} is not supported in canonical Workshop G_F"
            )
        if raw_subject_canon == "repair_target" and raw_object_canon == "MAIN_WORKBENCH_ZONE":
            return ("repair_target", "LOCATED_ON", "MAIN_WORKBENCH_ZONE", "PRESERVED", "ABSORBED_INTO_PLANNER_CONTEXT")

    # Explicit reverse support grammar from workbench to repair_target
    if raw_subject_canon == "MAIN_WORKBENCH_ZONE" and raw_object_canon == "repair_target":
        if any(_contains_phrase(norm_rel, k) for k in (
            "supports repair target", "supports target", "holds repair target", "provides support for repair target",
        )):
            return ("repair_target", "LOCATED_ON", "MAIN_WORKBENCH_ZONE", "NORMALIZED_TO_CANONICAL_SIGNATURE", "ABSORBED_INTO_PLANNER_CONTEXT")

    # (driver, fastener) -> COMPATIBLE_WITH
    if raw_subject_canon == "driver" and raw_object_canon == "fastener":
        if any(_contains_phrase(norm_rel, k) for k in (
            "compatible with", "compatible with fastener", "compatible with screw",
            "compatible with the fastener", "engage", "engages", "engage screw", "engages screw",
            "driver engages screw", "driver engages", "fit screw", "fits screw",
            "fit driver", "fits driver", "driver bit", "fit fastener", "fits fastener",
            "match bit", "torque to screw", "drives", "drives screw", "driver engages screw",
            "transmits torque", "transmit torque",
            "fit the screw head and transmit torque", "tip must fit the screw head and transmit torque",
            "fit screw head", "fits screw head", "driver bit matches fastener",
        )):
            return ("driver", "COMPATIBLE_WITH", "fastener", "PRESERVED", "GRAPH_RELATION")

    if raw_subject_canon == "fastener" and raw_object_canon == "driver":
        if any(_contains_phrase(norm_rel, k) for k in (
            "is driven by", "driven by", "is engaged by", "engaged by",
            "receives torque from", "driven by tool", "is driven by tool",
            "is turned by", "turned by", "receives drive from",
        )):
            return ("driver", "COMPATIBLE_WITH", "fastener", "NORMALIZED_TO_CANONICAL_SIGNATURE", "GRAPH_RELATION")

    # (driver, repair_target) -> REACHES_TARGET
    if raw_subject_canon == "driver" and raw_object_canon == "repair_target":
        if any(_contains_phrase(norm_rel, k) for k in (
            "reaches target", "reach target", "reaches into", "reach into",
            "reaches hole", "reach hole", "reaches repair", "reach repair",
            "access target", "accesses target", "length to reach", "reaches workpiece", "reach workpiece",
            "must reach the workpiece hole recess", "reaches workpiece hole",
            "reach workpiece hole", "reach workpiece hole recess",
            "long enough to reach workpiece hole recess", "long enough to reach hole",
            "long enough to reach",
        )):
            return ("driver", "REACHES_TARGET", "repair_target", "PRESERVED", "GRAPH_RELATION")

    if raw_subject_canon == "repair_target" and raw_object_canon == "driver":
        if any(_contains_phrase(norm_rel, k) for k in (
            "is reached by", "reached by", "target reached by", "accessed by", "target accessed by", "is accessed by",
        )):
            return ("driver", "REACHES_TARGET", "repair_target", "NORMALIZED_TO_CANONICAL_SIGNATURE", "GRAPH_RELATION")

    # (fastener, repair_target) -> COMPATIBLE_WITH_TARGET
    if raw_subject_canon == "fastener" and raw_object_canon == "repair_target":
        if any(_contains_phrase(norm_rel, k) for k in (
            "compatible with target", "compatible with", "compatible with hole",
            "thread into", "threads into", "fit hole", "fits hole", "fit target",
            "fits target", "anchor in", "anchors in", "fits workpiece", "fit workpiece",
            "fit inside", "fits inside", "fits into", "fit into", "inserted into",
            "insert into", "inserts into", "screw inserted into", "screw fits into",
            "must fit the workbench target hole and thread into the hole",
            "threads into target repair hole", "thread into target repair hole",
            "threads into hole", "thread into hole", "fits the hole", "fit the hole",
        )):
            return ("fastener", "COMPATIBLE_WITH_TARGET", "repair_target", "PRESERVED", "GRAPH_RELATION")

    if raw_subject_canon == "repair_target" and raw_object_canon == "fastener":
        if any(_contains_phrase(norm_rel, k) for k in (
            "receives fastener", "receives screw", "is fastened by", "fastened by", "is threaded by", "threaded by", "fastened with",
        )):
            return ("fastener", "COMPATIBLE_WITH_TARGET", "repair_target", "NORMALIZED_TO_CANONICAL_SIGNATURE", "GRAPH_RELATION")

    # If none matched, check if relation phrase maps to a known relation but with incompatible endpoints
    mapped_rel = map_workshop_relation(raw_relation_text)
    if mapped_rel is not None:
        raise MalformedVLMSpecificationError(
            f"Relation {raw_relation_text!r} mapped to {mapped_rel} but endpoints ({raw_subject_canon}, {raw_object_canon}) violate signature constraints"
        )

    raise UnmappedFunctionalConceptError(
        f"VLM Workshop relation {raw_relation_text!r} between {raw_subject_canon!r} and {raw_object_canon!r} cannot be mapped to any active canonical relation"
    )


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
        self.normalized_operation_groups: list[OperationGroup] = []
        self.vlm_derived_detector_prompts: tuple[str, ...] = ()
        self.canonicalization_trace: dict[str, Any] = {}
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
        return _phrase(value)

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
        return _contains_phrase(text, alias)

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
        comp = map_workshop_role_function(raw)
        if comp is not None:
            return comp
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
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

    def generate_canonical(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
        raw_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Strict fail-closed canonicalization of Workshop raw VLM specification."""
        if raw_document is not None:
            document = raw_document
            self.raw_vlm_response = deepcopy(document)
            self.validated_vlm_specification = deepcopy(document)
            self.raw_decomposition = document
        else:
            document = self.fm_adapter.generate_task_requirements(
                task_instruction, observation_images=observation_images or []
            )
            self.raw_vlm_response = deepcopy(
                getattr(self.fm_adapter, "last_raw_requirement_response", None) or document
            )
            self.validated_vlm_specification = deepcopy(document)
            self.raw_decomposition = document

        if not isinstance(document, dict):
            raise MalformedVLMSpecificationError("Workshop VLM specification must be a dictionary")

        if document.get("status") != "SUPPORTED":
            raise VLMSpecificationError(
                "VLM_SPEC_FAILED: VLM marked the Workshop task unsupported: "
                f"{document.get('unsupported_reason', 'no reason')}"
            )

        # Strict schema validation of top-level sections
        if "functional_roles" not in document or not isinstance(document["functional_roles"], list):
            raise MalformedVLMSpecificationError("Workshop VLM specification missing required list section 'functional_roles'")
        if "functional_relations" not in document or not isinstance(document["functional_relations"], list):
            raise MalformedVLMSpecificationError("Workshop VLM specification missing required list section 'functional_relations'")

        raw_groups_list = document.get("interaction_groups", [])
        if not isinstance(raw_groups_list, list):
            raise MalformedVLMSpecificationError("Workshop VLM specification 'interaction_groups' must be a list")

        raw_inspectable_list = document.get("inspectable_regions", [])
        if not isinstance(raw_inspectable_list, list):
            raise MalformedVLMSpecificationError("Workshop VLM specification 'inspectable_regions' must be a list")

        raw_requirements = document["functional_roles"]
        if not raw_requirements:
            raise MalformedVLMSpecificationError("Workshop VLM specification has empty functional_roles")

        seen_raw_role_ids: set[str] = set()
        for raw in raw_requirements:
            if not isinstance(raw, dict):
                raise MalformedVLMSpecificationError(f"Raw role entry must be a dictionary, got {type(raw).__name__}")
            for req_field in (
                "id", "entity_kind", "function", "description",
                "required_count", "binding_policy", "candidate_categories",
                "visible_candidates", "required_properties",
            ):
                if req_field not in raw:
                    raise MalformedVLMSpecificationError(
                        f"Raw role {raw.get('id', '<unknown>')!r} missing required schema field {req_field!r}"
                    )
            raw_id = raw["id"]
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise MalformedVLMSpecificationError(f"Raw role id must be a non-empty string, got {raw_id!r}")
            if raw_id in seen_raw_role_ids:
                raise MalformedVLMSpecificationError(f"Duplicate raw role ID {raw_id!r} declared in functional_roles")
            seen_raw_role_ids.add(raw_id)

            raw_kind = raw["entity_kind"]
            if raw_kind not in {"OBJECT", "REGION", "FIXED_TARGET"}:
                raise MalformedVLMSpecificationError(
                    f"Invalid entity_kind {raw_kind!r} for role {raw_id!r}"
                )

            if not isinstance(raw["function"], str) or not raw["function"].strip():
                raise MalformedVLMSpecificationError(f"Role {raw_id!r} function must be a non-empty string")
            if not isinstance(raw["description"], str):
                raise MalformedVLMSpecificationError(f"Role {raw_id!r} description must be a string")

            if not isinstance(raw["required_count"], int) or isinstance(raw["required_count"], bool) or raw["required_count"] < 1:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_id!r} required_count must be an integer >= 1, got {raw['required_count']!r}"
                )

            if raw["binding_policy"] not in {"DISTINCT", "REUSABLE", "SHARED"}:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_id!r} invalid binding_policy {raw['binding_policy']!r}"
                )

            if not isinstance(raw["candidate_categories"], list):
                raise MalformedVLMSpecificationError(f"Role {raw_id!r} candidate_categories must be a list")
            if raw_kind == "OBJECT":
                cand_cats_valid = [str(c).strip() for c in raw["candidate_categories"] if str(c).strip()]
                if not cand_cats_valid:
                    raise MalformedVLMSpecificationError(
                        f"Workshop functional role {raw_id!r} must have non-empty candidate_categories"
                    )

            if not isinstance(raw["visible_candidates"], list):
                raise MalformedVLMSpecificationError(f"Role {raw_id!r} visible_candidates must be a list")
            if not isinstance(raw["required_properties"], list):
                raise MalformedVLMSpecificationError(f"Role {raw_id!r} required_properties must be a list")

        # Validate raw relations schema
        for rel in document["functional_relations"]:
            if not isinstance(rel, dict):
                raise MalformedVLMSpecificationError("Raw relation entry must be a dictionary")
            for req_field in ("subject_role", "relation", "object_role"):
                if req_field not in rel or not isinstance(rel[req_field], str) or not rel[req_field].strip():
                    raise MalformedVLMSpecificationError(
                        f"Raw relation missing or non-string required field {req_field!r}: {rel}"
                    )
            if rel["subject_role"] not in seen_raw_role_ids:
                raise MalformedVLMSpecificationError(
                    f"Relation subject role {rel['subject_role']!r} not declared in functional_roles"
                )
            if rel["object_role"] not in seen_raw_role_ids:
                raise MalformedVLMSpecificationError(
                    f"Relation object role {rel['object_role']!r} not declared in functional_roles"
                )

        # Validate raw interaction groups schema
        for grp in raw_groups_list:
            if not isinstance(grp, dict):
                raise MalformedVLMSpecificationError("Interaction group entry must be a dictionary")
            for req_field in (
                "id", "function", "tool_role", "target_role",
                "required_target_count", "usage_policy", "required_relations",
                "context_role", "context_relations",
            ):
                if req_field not in grp:
                    raise MalformedVLMSpecificationError(
                        f"Interaction group missing required field {req_field!r}: {grp}"
                    )
            grp_id = grp["id"]
            if grp["tool_role"] not in seen_raw_role_ids:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} tool role {grp['tool_role']!r} not declared in functional_roles"
                )
            if grp["target_role"] not in seen_raw_role_ids:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} target role {grp['target_role']!r} not declared in functional_roles"
                )
            if grp["context_role"] not in seen_raw_role_ids:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} context_role {grp['context_role']!r} not declared in functional_roles"
                )
            if not isinstance(grp["required_target_count"], int) or isinstance(grp["required_target_count"], bool) or grp["required_target_count"] < 1:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} required_target_count must be an integer >= 1, got {grp['required_target_count']!r}"
                )
            if grp["usage_policy"] != "DEDICATED_PER_TARGET":
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} has invalid usage_policy {grp['usage_policy']!r}, expected 'DEDICATED_PER_TARGET'"
                )
            if not isinstance(grp["required_relations"], list) or len(grp["required_relations"]) != 1:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} required_relations must contain exactly 1 relation phrase, got {len(grp['required_relations']) if isinstance(grp['required_relations'], list) else grp['required_relations']!r}"
                )
            if not isinstance(grp["context_relations"], list) or len(grp["context_relations"]) != 1:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {grp_id!r} context_relations must contain exactly 1 relation phrase, got {len(grp['context_relations']) if isinstance(grp['context_relations'], list) else grp['context_relations']!r}"
                )

        concept_accounting: dict[str, Any] = {
            "roles": {},
            "properties": [],
            "relations": [],
            "operation_groups": [],
        }
        self.transformation_trace = []
        raw_id_to_canon: dict[str, str] = {}

        fixed_target_roles: list[dict[str, Any]] = []
        context_region_roles: list[dict[str, Any]] = []
        classified_object_roles: dict[str, list[dict[str, Any]]] = {
            "driver": [],
            "fastener": [],
        }

        # Step 1: Classify all raw roles strictly by function and description
        for raw in raw_requirements:
            raw_id = raw["id"]
            raw_kind = raw["entity_kind"]

            if raw_kind == "FIXED_TARGET":
                canon_target = map_workshop_fixed_target_role(raw)
                if canon_target is None:
                    raise UnmappedFunctionalConceptError(
                        f"VLM FIXED_TARGET role {raw_id!r} with function {raw['function']!r} cannot be mapped to any Workshop fixed target"
                    )
                raw_id_to_canon[raw_id] = "repair_target"
                fixed_target_roles.append(raw)

            elif raw_kind == "REGION":
                canon_region = map_workshop_context_region_role(raw)
                if canon_region is None:
                    raise UnmappedFunctionalConceptError(
                        f"VLM REGION role {raw_id!r} with function {raw['function']!r} cannot be mapped to any Workshop context region"
                    )
                raw_id_to_canon[raw_id] = "MAIN_WORKBENCH_ZONE"
                context_region_roles.append(raw)

            elif raw_kind == "OBJECT":
                func_name = map_workshop_role_function(raw)
                if func_name is None:
                    raise UnmappedFunctionalConceptError(
                        f"VLM function phrase {raw['function']!r} on role {raw_id!r} cannot be mapped to any Workshop role"
                    )
                canon_role = "driver" if func_name == "CAN_DRIVE_SCREW" else "fastener"
                raw_id_to_canon[raw_id] = canon_role
                classified_object_roles[canon_role].append(raw)

        # Step 2: Validate Unary Properties fail-closed
        for raw in raw_requirements:
            raw_id = raw["id"]
            for prop in raw["required_properties"]:
                if not isinstance(prop, str) or not prop.strip():
                    raise MalformedVLMSpecificationError(
                        f"Required property on role {raw_id!r} must be a non-empty string, got {prop!r}"
                    )
                mapped_u = map_workshop_unary_property(prop)
                if mapped_u is None:
                    raise UnmappedFunctionalConceptError(
                        f"Required property {prop!r} on role {raw_id!r} cannot be mapped to any Workshop unary property"
                    )
                if raw["entity_kind"] == "REGION" and mapped_u == "PLANAR_SUPPORT":
                    concept_accounting["properties"].append({
                        "raw_role_id": raw_id,
                        "raw_phrase": prop,
                        "canonical_role": "MAIN_WORKBENCH_ZONE",
                        "canonical_predicate": "PLANAR_SUPPORT",
                        "status": "ABSORBED_INTO_PLANNER_CONTEXT",
                        "destination": "PLANNER_CONTEXT",
                    })
                else:
                    raise UnsupportedCheckerCapabilityError(
                        f"Unary predicate {mapped_u!r} on role {raw_id!r} is not supported in canonical Workshop G_F"
                    )

        # Step 3: Populate canonical roles
        normalized_roles: list[NormalizedWorkshopRole] = []
        legacy_requirements: dict[str, FunctionalRequirement] = {}

        # 3a: FIXED_TARGET
        if len(fixed_target_roles) > 1:
            raise AmbiguousCanonicalizationError(
                f"Multiple FIXED_TARGET roles mapping to repair_target: {[r['id'] for r in fixed_target_roles]}"
            )
        if fixed_target_roles:
            r = fixed_target_roles[0]
            cand_cats = tuple(str(c).strip() for c in r["candidate_categories"] if str(c).strip())
            run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats)) or ("repair_target",)
            cand_objs = tuple(r["visible_candidates"])
            hints = tuple(candidate["label"] for candidate in cand_objs if candidate.get("label"))

            normalized_role = NormalizedWorkshopRole(
                raw_role_id=r["id"],
                canonical_role_id="repair_target",
                entity_kind="FIXED_TARGET",
                raw_function=r["function"],
                canonical_function="FIXED_TARGET",
                required_count=r["required_count"],
                binding_policy=r["binding_policy"],
                candidate_categories=cand_cats,
                run_local_categories=run_local_cats,
                visible_candidates=cand_objs,
                unary_predicates=(),
                description=r["description"],
                semantic_hints=hints,
                provenance="vlm_explicit_fixed_target",
            )
            normalized_roles.append(normalized_role)
            concept_accounting["roles"][r["id"]] = {
                "canonical_role": "repair_target",
                "entity_kind": "FIXED_TARGET",
                "raw_count": r["required_count"],
                "canonical_count": r["required_count"],
                "binding_policy": r["binding_policy"],
                "unary_predicates": [],
                "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                "candidate_categories_used_for_role_identity": False,
                "status": "PRESERVED",
            }
            self.transformation_trace.append({
                "raw_role": r["id"],
                "raw_entity_kind": "FIXED_TARGET",
                "transformation": "SYSTEM_OWNED_FIXED_TARGET_REPRESENTATION",
                "canonical_role": "repair_target",
            })

        # 3b: CONTEXT REGIONS (Absorbed into planner context, not in normalized_roles)
        for r in context_region_roles:
            concept_accounting["roles"][r["id"]] = {
                "canonical_role": "MAIN_WORKBENCH_ZONE",
                "entity_kind": "REGION",
                "raw_count": r["required_count"],
                "canonical_count": 0,
                "binding_policy": r["binding_policy"],
                "unary_predicates": ["PLANAR_SUPPORT"] if any(map_workshop_unary_property(p) == "PLANAR_SUPPORT" for p in r["required_properties"]) else [],
                "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                "candidate_categories_used_for_role_identity": False,
                "status": "ABSORBED_INTO_PLANNER_CONTEXT",
                "planner_context_constant": "MAIN_WORKBENCH_ZONE",
            }
            self.transformation_trace.append({
                "raw_role": r["id"],
                "raw_entity_kind": "REGION",
                "transformation": "ABSORBED_INTO_PLANNER_CONTEXT",
                "canonical_role": "MAIN_WORKBENCH_ZONE",
            })

        # 3c: DRIVER OBJECT ROLES
        driver_list = classified_object_roles["driver"]
        if not driver_list:
            raise MalformedVLMSpecificationError("Missing required functional driver role in Workshop task")
        if len(driver_list) == 1:
            r = driver_list[0]
            cand_cats = tuple(str(c).strip() for c in r["candidate_categories"] if str(c).strip())
            run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats))
            for cat in cand_cats:
                canon_c = self._map_category(cat)
                if canon_c is not None:
                    self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
            cand_objs = tuple(r["visible_candidates"])
            for candidate in cand_objs:
                for field in ("label", "visual_description"):
                    val = candidate.get(field)
                    if val:
                        canon_c = self._map_category(val)
                        if canon_c is not None:
                            self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
                            break
            hints = tuple(candidate["label"] for candidate in cand_objs if candidate.get("label"))

            normalized_role = NormalizedWorkshopRole(
                raw_role_id=r["id"],
                canonical_role_id="driver",
                entity_kind="OBJECT",
                raw_function=r["function"],
                canonical_function="CAN_DRIVE_SCREW",
                required_count=r["required_count"],
                binding_policy=r["binding_policy"],
                candidate_categories=cand_cats,
                run_local_categories=run_local_cats,
                visible_candidates=cand_objs,
                unary_predicates=(),
                description=r["description"],
                semantic_hints=hints,
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
            normalized_roles.append(normalized_role)
            concept_accounting["roles"][r["id"]] = {
                "canonical_role": "driver",
                "entity_kind": "OBJECT",
                "raw_count": r["required_count"],
                "canonical_count": r["required_count"],
                "binding_policy": r["binding_policy"],
                "unary_predicates": [],
                "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                "candidate_categories_used_for_role_identity": False,
                "status": "PRESERVED",
            }
            self.transformation_trace.append({
                "raw_role": r["id"],
                "raw_entity_kind": "OBJECT",
                "transformation": "CANONICAL_DRIVER_ROLE_MAPPING",
                "canonical_role": "driver",
            })
            legacy_requirements["CAN_DRIVE_SCREW"] = FunctionalRequirement(
                requirement_id=r["id"],
                entity_type=EntityType.OBJECT,
                function_name="CAN_DRIVE_SCREW",
                description=r["description"],
                rank=1,
                source=RequirementSource.FM,
                accepted_categories=list(cand_cats),
                semantic_hints=list(hints),
                geometric_constraints={},
                required_relations=[],
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
        else:
            all_alternatives = all(_is_explicit_alternative_role(r["function"], r["description"]) for r in driver_list)
            all_count_1 = all(r["required_count"] == 1 for r in driver_list)
            all_distinct = all(r["binding_policy"] == "DISTINCT" for r in driver_list)
            if not (all_alternatives and all_count_1 and all_distinct):
                raise AmbiguousCanonicalizationError(
                    f"Multiple raw driver roles without explicit alternative evidence: {[r['id'] for r in driver_list]}"
                )
            combined_cats_list: list[str] = []
            combined_hints_list: list[str] = []
            for r in driver_list:
                for c in r["candidate_categories"]:
                    c_str = str(c).strip()
                    if c_str and c_str not in combined_cats_list:
                        combined_cats_list.append(c_str)
                for cand in r["visible_candidates"]:
                    if cand.get("label") and cand["label"] not in combined_hints_list:
                        combined_hints_list.append(cand["label"])
            cand_cats_tuple = tuple(combined_cats_list)
            run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats_tuple))
            for cat in cand_cats_tuple:
                canon_c = self._map_category(cat)
                if canon_c is not None:
                    self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
            r0 = driver_list[0]
            normalized_role = NormalizedWorkshopRole(
                raw_role_id=r0["id"],
                canonical_role_id="driver",
                entity_kind="OBJECT",
                raw_function=r0["function"],
                canonical_function="CAN_DRIVE_SCREW",
                required_count=1,
                binding_policy="DISTINCT",
                candidate_categories=cand_cats_tuple,
                run_local_categories=run_local_cats,
                visible_candidates=(),
                unary_predicates=(),
                description=f"Merged alternative driver candidates: {', '.join(r['id'] for r in driver_list)}",
                semantic_hints=tuple(combined_hints_list),
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
            normalized_roles.append(normalized_role)
            for r in driver_list:
                concept_accounting["roles"][r["id"]] = {
                    "canonical_role": "driver",
                    "entity_kind": "OBJECT",
                    "raw_count": r["required_count"],
                    "canonical_count": 1,
                    "binding_policy": r["binding_policy"],
                    "unary_predicates": [],
                    "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                    "candidate_categories_used_for_role_identity": False,
                    "status": "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE",
                    "reason": "Explicit alternative driver candidate merged into canonical driver role",
                }
            self.transformation_trace.append({
                "raw_roles": [r["id"] for r in driver_list],
                "raw_entity_kind": "OBJECT",
                "transformation": "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE",
                "canonical_role": "driver",
            })
            legacy_requirements["CAN_DRIVE_SCREW"] = FunctionalRequirement(
                requirement_id=r0["id"],
                entity_type=EntityType.OBJECT,
                function_name="CAN_DRIVE_SCREW",
                description=r0["description"],
                rank=1,
                source=RequirementSource.FM,
                accepted_categories=list(cand_cats_tuple),
                semantic_hints=list(combined_hints_list),
                geometric_constraints={},
                required_relations=[],
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )

        # 3d: FASTENER OBJECT ROLES
        fastener_list = classified_object_roles["fastener"]
        if not fastener_list:
            raise MalformedVLMSpecificationError("Missing required functional fastener role in Workshop task")
        if len(fastener_list) == 1:
            r = fastener_list[0]
            cand_cats = tuple(str(c).strip() for c in r["candidate_categories"] if str(c).strip())
            run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats))
            for cat in cand_cats:
                canon_c = self._map_category(cat)
                if canon_c is not None:
                    self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
            cand_objs = tuple(r["visible_candidates"])
            for candidate in cand_objs:
                for field in ("label", "visual_description"):
                    val = candidate.get(field)
                    if val:
                        canon_c = self._map_category(val)
                        if canon_c is not None:
                            self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
                            break
            hints = tuple(candidate["label"] for candidate in cand_objs if candidate.get("label"))

            normalized_role = NormalizedWorkshopRole(
                raw_role_id=r["id"],
                canonical_role_id="fastener",
                entity_kind="OBJECT",
                raw_function=r["function"],
                canonical_function="CAN_FASTEN",
                required_count=r["required_count"],
                binding_policy=r["binding_policy"],
                candidate_categories=cand_cats,
                run_local_categories=run_local_cats,
                visible_candidates=cand_objs,
                unary_predicates=(),
                description=r["description"],
                semantic_hints=hints,
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
            normalized_roles.append(normalized_role)
            concept_accounting["roles"][r["id"]] = {
                "canonical_role": "fastener",
                "entity_kind": "OBJECT",
                "raw_count": r["required_count"],
                "canonical_count": r["required_count"],
                "binding_policy": r["binding_policy"],
                "unary_predicates": [],
                "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                "candidate_categories_used_for_role_identity": False,
                "status": "PRESERVED",
            }
            self.transformation_trace.append({
                "raw_role": r["id"],
                "raw_entity_kind": "OBJECT",
                "transformation": "CANONICAL_FASTENER_ROLE_MAPPING",
                "canonical_role": "fastener",
            })
            legacy_requirements["CAN_FASTEN"] = FunctionalRequirement(
                requirement_id=r["id"],
                entity_type=EntityType.OBJECT,
                function_name="CAN_FASTEN",
                description=r["description"],
                rank=2,
                source=RequirementSource.FM,
                accepted_categories=list(cand_cats),
                semantic_hints=list(hints),
                geometric_constraints={},
                required_relations=[],
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
        else:
            all_alternatives = all(_is_explicit_alternative_role(r["function"], r["description"]) for r in fastener_list)
            all_count_1 = all(r["required_count"] == 1 for r in fastener_list)
            all_distinct = all(r["binding_policy"] == "DISTINCT" for r in fastener_list)
            if not (all_alternatives and all_count_1 and all_distinct):
                raise AmbiguousCanonicalizationError(
                    f"Multiple raw fastener roles without explicit alternative evidence: {[r['id'] for r in fastener_list]}"
                )
            combined_cats_list: list[str] = []
            combined_hints_list: list[str] = []
            for r in fastener_list:
                for c in r["candidate_categories"]:
                    c_str = str(c).strip()
                    if c_str and c_str not in combined_cats_list:
                        combined_cats_list.append(c_str)
                for cand in r["visible_candidates"]:
                    if cand.get("label") and cand["label"] not in combined_hints_list:
                        combined_hints_list.append(cand["label"])
            cand_cats_tuple = tuple(combined_cats_list)
            run_local_cats = tuple(dict.fromkeys(self._phrase(c).replace(" ", "_") for c in cand_cats_tuple))
            for cat in cand_cats_tuple:
                canon_c = self._map_category(cat)
                if canon_c is not None:
                    self._category_rank.setdefault(canon_c, len(self._category_rank) + 1)
            r0 = fastener_list[0]
            normalized_role = NormalizedWorkshopRole(
                raw_role_id=r0["id"],
                canonical_role_id="fastener",
                entity_kind="OBJECT",
                raw_function=r0["function"],
                canonical_function="CAN_FASTEN",
                required_count=1,
                binding_policy="DISTINCT",
                candidate_categories=cand_cats_tuple,
                run_local_categories=run_local_cats,
                visible_candidates=(),
                unary_predicates=(),
                description=f"Merged alternative fastener candidates: {', '.join(r['id'] for r in fastener_list)}",
                semantic_hints=tuple(combined_hints_list),
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )
            normalized_roles.append(normalized_role)
            for r in fastener_list:
                concept_accounting["roles"][r["id"]] = {
                    "canonical_role": "fastener",
                    "entity_kind": "OBJECT",
                    "raw_count": r["required_count"],
                    "canonical_count": 1,
                    "binding_policy": r["binding_policy"],
                    "unary_predicates": [],
                    "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                    "candidate_categories_used_for_role_identity": False,
                    "status": "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE",
                    "reason": "Explicit alternative fastener candidate merged into canonical fastener role",
                }
            self.transformation_trace.append({
                "raw_roles": [r["id"] for r in fastener_list],
                "raw_entity_kind": "OBJECT",
                "transformation": "MERGED_BY_EXPLICIT_ALTERNATIVE_RULE",
                "canonical_role": "fastener",
            })
            legacy_requirements["CAN_FASTEN"] = FunctionalRequirement(
                requirement_id=r0["id"],
                entity_type=EntityType.OBJECT,
                function_name="CAN_FASTEN",
                description=r0["description"],
                rank=2,
                source=RequirementSource.FM,
                accepted_categories=list(cand_cats_tuple),
                semantic_hints=list(combined_hints_list),
                geometric_constraints={},
                required_relations=[],
                provenance="qwen_vlm_normalized_by_workshop_ontology",
            )

        # Step 4: Canonicalize Relations
        normalized_relations: list[NormalizedWorkshopRelation] = []
        seen_canonical_triples: set[tuple[str, str, str]] = set()

        for rel in document.get("functional_relations", []):
            s = rel["subject_role"]
            r = rel["relation"]
            o = rel["object_role"]
            s_canon = raw_id_to_canon[s]
            o_canon = raw_id_to_canon[o]

            c_subj, c_pred, c_obj, dir_status, dest = canonicalize_workshop_relation(
                s, s_canon, r, o, o_canon
            )

            if dest == "GRAPH_RELATION":
                triple = (c_subj, c_pred, c_obj)
                if triple not in seen_canonical_triples:
                    seen_canonical_triples.add(triple)
                    normalized_relations.append(
                        NormalizedWorkshopRelation(
                            raw_subject_role_id=s,
                            canonical_subject_role_id=c_subj,
                            raw_relation_text=r,
                            canonical_predicate=c_pred,
                            raw_object_role_id=o,
                            canonical_object_role_id=c_obj,
                        )
                    )
                    concept_accounting["relations"].append({
                        "raw_subject_role_id": s,
                        "raw_relation_text": r,
                        "raw_object_role_id": o,
                        "canonical_subject_role_id": c_subj,
                        "canonical_predicate": c_pred,
                        "canonical_object_role_id": c_obj,
                        "direction_status": dir_status,
                        "structural_destination": "GRAPH_RELATION",
                        "status": "PRESERVED" if dir_status == "PRESERVED" else "NORMALIZED_TO_CANONICAL_SIGNATURE",
                    })
                else:
                    concept_accounting["relations"].append({
                        "raw_subject_role_id": s,
                        "raw_relation_text": r,
                        "raw_object_role_id": o,
                        "canonical_subject_role_id": c_subj,
                        "canonical_predicate": c_pred,
                        "canonical_object_role_id": c_obj,
                        "direction_status": dir_status,
                        "structural_destination": "GRAPH_RELATION",
                        "status": "MERGED_BY_EXPLICIT_RULE",
                        "reason": "Duplicate canonical relation triple merged",
                    })

                # Update legacy required_relations
                if c_subj == "driver" and "CAN_DRIVE_SCREW" in legacy_requirements:
                    if c_pred not in legacy_requirements["CAN_DRIVE_SCREW"].required_relations:
                        legacy_requirements["CAN_DRIVE_SCREW"].required_relations.append(c_pred)
                elif c_subj == "fastener" and "CAN_FASTEN" in legacy_requirements:
                    if c_pred not in legacy_requirements["CAN_FASTEN"].required_relations:
                        legacy_requirements["CAN_FASTEN"].required_relations.append(c_pred)

            elif dest == "ABSORBED_INTO_PLANNER_CONTEXT":
                concept_accounting["relations"].append({
                    "raw_subject_role_id": s,
                    "raw_relation_text": r,
                    "raw_object_role_id": o,
                    "canonical_subject_role_id": c_subj,
                    "canonical_predicate": c_pred,
                    "canonical_object_role_id": c_obj,
                    "direction_status": dir_status,
                    "structural_destination": "ABSORBED_INTO_PLANNER_CONTEXT",
                    "status": "ABSORBED_INTO_PLANNER_CONTEXT",
                    "planner_context_constant": "MAIN_WORKBENCH_ZONE",
                })

        # Step 5: Validate and absorb Interaction Groups (Zero runtime operation groups emitted)
        raw_groups = document.get("interaction_groups", [])
        if len(raw_groups) > 1:
            raise AmbiguousCanonicalizationError(
                f"Multiple interaction groups produced for Workshop task: {[g['id'] for g in raw_groups]}"
            )
        for grp in raw_groups:
            g_id = grp["id"]
            func_desc = grp["function"]
            tool_raw = grp["tool_role"]
            target_raw = grp["target_role"]
            target_count = grp["required_target_count"]
            policy = grp["usage_policy"]
            req_rels_raw = grp["required_relations"]
            ctx_raw = grp.get("context_role")
            ctx_rels_raw = grp.get("context_relations", [])

            if policy != "DEDICATED_PER_TARGET":
                raise MalformedVLMSpecificationError(
                    f"Interaction group {g_id!r} has invalid usage_policy {policy!r}, expected 'DEDICATED_PER_TARGET'"
                )

            if not ctx_raw:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {g_id!r} is missing required context_role"
                )

            tool_canon = raw_id_to_canon[tool_raw]
            target_canon = raw_id_to_canon[target_raw]
            ctx_canon = raw_id_to_canon[ctx_raw]

            if tool_canon != "driver":
                raise MalformedVLMSpecificationError(
                    f"Interaction group tool role {tool_raw!r} mapped to {tool_canon!r}, expected 'driver'"
                )
            if target_canon != "fastener":
                raise MalformedVLMSpecificationError(
                    f"Interaction group target role {target_raw!r} mapped to {target_canon!r}, expected 'fastener'"
                )
            if ctx_canon != "repair_target":
                raise MalformedVLMSpecificationError(
                    f"Interaction group context role {ctx_raw!r} mapped to {ctx_canon!r}, expected 'repair_target'"
                )
            if target_count != 1:
                raise MalformedVLMSpecificationError(
                    f"Invalid required_target_count {target_count} in interaction group {g_id!r}, expected 1"
                )

            norm_fn = self._phrase(func_desc)
            if not any(k in norm_fn for k in (
                "drive screw", "drive fastener", "fasten screw", "secure joint",
                "tighten screw", "turn screw", "repair joint", "fasten joint", "drive",
            )):
                raise UnmappedFunctionalConceptError(
                    f"Interaction group function {func_desc!r} cannot be mapped to reviewed driving action"
                )

            if len(req_rels_raw) != 1:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {g_id!r} required_relations must contain exactly 1 relation phrase, got {len(req_rels_raw)}"
                )

            raw_req_rel = req_rels_raw[0]
            m_sub, m_pred, m_obj, _, _ = canonicalize_workshop_relation(
                tool_raw, tool_canon, raw_req_rel, target_raw, target_canon
            )
            if m_pred != "COMPATIBLE_WITH":
                raise MalformedVLMSpecificationError(
                    f"Interaction group required_relation {raw_req_rel!r} mapped to {m_pred!r}, expected 'COMPATIBLE_WITH'"
                )

            if len(ctx_rels_raw) != 1:
                raise MalformedVLMSpecificationError(
                    f"Interaction group {g_id!r} context_relations must contain exactly 1 relation phrase, got {len(ctx_rels_raw)}"
                )

            raw_ctx_rel = ctx_rels_raw[0]
            m_sub, m_pred, m_obj, _, _ = canonicalize_workshop_relation(
                tool_raw, tool_canon, raw_ctx_rel, ctx_raw, ctx_canon
            )
            if m_pred != "REACHES_TARGET":
                raise MalformedVLMSpecificationError(
                    f"Interaction group context_relation {raw_ctx_rel!r} mapped to {m_pred!r}, expected 'REACHES_TARGET'"
                )

            # Redundancy Proof Rule:
            # Construct represented group triples and prove they exist in top-level relations
            seen_canonical_triples = {
                (r.canonical_subject_role_id, r.canonical_predicate, r.canonical_object_role_id)
                for r in normalized_relations
            }
            represented_group_triples = {
                ("driver", "COMPATIBLE_WITH", "fastener"),
                ("driver", "REACHES_TARGET", "repair_target"),
            }
            if not (represented_group_triples <= seen_canonical_triples):
                missing_triples = sorted(represented_group_triples - seen_canonical_triples)
                raise MalformedVLMSpecificationError(
                    f"Interaction group {g_id!r} claims redundancy but top-level canonical graph relations are missing required triple(s): {missing_triples}"
                )

            concept_accounting["operation_groups"].append({
                "raw_group_id": g_id,
                "raw_function": func_desc,
                "canonical_function": "DRIVE_FASTENER_INTO_TARGET",
                "tool_role": f"{tool_raw} -> driver",
                "target_role": f"{target_raw} -> fastener",
                "context_role": f"{ctx_raw} -> repair_target",
                "usage_policy": policy,
                "raw_required_relation": raw_req_rel,
                "canonical_required_relation": "COMPATIBLE_WITH",
                "raw_context_relation": raw_ctx_rel,
                "canonical_context_relation": "REACHES_TARGET",
                "status": "MERGED_BY_EXPLICIT_RULE",
                "structural_destination": "REDUNDANT_WITH_CANONICAL_GRAPH_RELATIONS",
                "represented_relations": ["COMPATIBLE_WITH", "REACHES_TARGET"],
                "represented_relation_triples": [
                    ["driver", "COMPATIBLE_WITH", "fastener"],
                    ["driver", "REACHES_TARGET", "repair_target"],
                ],
            })
            self.transformation_trace.append({
                "raw_group": g_id,
                "transformation": "VALIDATED_REDUNDANT_WITH_GRAPH_RELATIONS",
                "tool_role": f"{tool_raw} -> driver",
                "target_role": f"{target_raw} -> fastener",
                "context_role": f"{ctx_raw} -> repair_target",
            })

        # Workshop runtime G_F has zero operation groups
        self.normalized_operation_groups = []

        self.normalized_roles = normalized_roles
        self.normalized_relations = normalized_relations
        self._requirements = sorted(legacy_requirements.values(), key=lambda item: item.rank)

        # Resolve regions
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

        # Detector vocabulary
        vlm_prompts = list(dict.fromkeys(
            cat
            for r in self.normalized_roles
            if r.entity_kind == "OBJECT"
            for cat in r.candidate_categories
        ))
        self.vlm_derived_detector_prompts = tuple(vlm_prompts)

        self.canonicalization_trace = {
            "version": WORKSHOP_VLM_CANONICALIZATION_VERSION,
            "vlm_canonicalization_version": WORKSHOP_VLM_CANONICALIZATION_VERSION,
            "concept_accounting": concept_accounting,
            "raw_roles_count": len(raw_requirements),
            "raw_relations_count": len(document.get("functional_relations", [])),
            "raw_operation_groups_count": len(raw_groups),
            "transformation_trace": self.transformation_trace,
        }

        return {
            "status": "CANONICALIZED",
            "ready_for_grounding": True,
            "normalized_requirements": self.normalized_roles,
            "normalized_relations": self.normalized_relations,
            "normalized_operation_groups": self.normalized_operation_groups,
            "canonicalization_trace": self.canonicalization_trace,
        }

    def _ensure_generated(
        self,
        task_instruction: str = CANONICAL_WORKSHOP_INSTRUCTION,
        *,
        observation_images: list[str | Path] | None = None,
        raw_document: dict[str, Any] | None = None,
    ) -> None:
        if raw_document is not None or self._requirements is None:
            self.generate_canonical(
                task_instruction,
                observation_images=observation_images,
                raw_document=raw_document,
            )

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

