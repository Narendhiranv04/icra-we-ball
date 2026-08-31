"""Image-conditioned Qwen requirement generation for Kitchen and Living Room.

The model independently produces roles, properties, counts, and visible
candidate objects. This module validates and deterministically canonicalizes
the response against reviewed task-independent normalization contracts after
the call and stops before grounding, allocation, planning, or execution.
"""

from __future__ import annotations

from copy import deepcopy
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
from .workshop_phase1.fm_adapter import FMAdapter


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NORMALIZATION = (
    Path(__file__).resolve().parent
    / "configs"
    / "kitchen_living_room_vlm_normalization.yaml"
)
SUPPORTED_ENVIRONMENTS = ("kitchen", "living_room")
KITCHEN_SEARCH_REGIONS = {
    "D1": "upper kitchen drawer",
    "D2": "lower kitchen drawer",
    "C2": "upper wall cupboard",
    "B1": "countertop storage box",
    "C1": "lower kitchen cupboard",
}

VLM_CANONICALIZATION_VERSION = "phase3_6a7_2_1_v1"
LIVING_ROOM_VLM_CANONICALIZATION_VERSION = "phase3_p3f_2_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return document


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    words = text.split()
    p_words = phrase.split()
    n, k = len(words), len(p_words)
    return any(words[i:i + k] == p_words for i in range(n - k + 1))


def _phrase_score(text: str, alias: str) -> float:
    text_words = set(_phrase(text).split())
    alias_words = set(_phrase(alias).split())
    if not alias_words:
        return 0.0
    return len(text_words & alias_words) / len(alias_words)


LIVING_SUPPORTED_UNARY_PREDICATES = {"PLANAR_SUPPORT"}
KITCHEN_SUPPORTED_UNARY_PREDICATES = {"OPEN_CAVITY", "ELONGATED_OBJECT", "PLANAR_SUPPORT"}

LIVING_ROLE_EXPECTED_ENTITY_KINDS = {
    "PERSONAL_CUP_SAUCER_REGION": "REGION",
    "SHARED_REMOTE_REGION": "REGION",
    "CUP_SAUCER_SET": "OBJECT",
    "REMOTE": "OBJECT",
    "SEATING_POSITION": "FIXED_TARGET",
    "SEATING_PAIR": "FIXED_TARGET",
}

LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES = {
    "CUP_SAUCER_SET": ("cup_saucer_set", "cup", "saucer"),
    "REMOTE": ("tv_remote", "remote_control"),
    "SEATING_POSITION": ("seating_position", "armchair", "chair", "sofa"),
    "SEATING_PAIR": ("seating_pair", "armchair", "chair", "sofa"),
}

LIVING_REGION_ROLE_ALIASES = {
    "PERSONAL_CUP_SAUCER_REGION": (
        "personal cup and saucer support",
        "support a cup and saucer near a seat",
        "personal serving surface",
        "hold one viewer drinkware",
        "hold viewer drinkware",
        "support cup and saucer",
        "support personal cup and saucer",
        "personal drinkware support",
        "cup and saucer support",
        "personal support for cup and saucer",
        "support cup and saucer near armchair",
        "support for cup and saucer set",
        "support personal drink",
        "support drinks",
        "support a cup and saucer",
        "hold items for viewer",
        "fixed individual side table surface beside each viewer seating position for supporting drinkware",
        "side table surface beside viewer",
        "personal table surface for drinkware",
        "individual drink surface",
        "personal drinkware surface",
        "personal surface",
        "side table surface",
        "viewer 1 drinkware",
        "viewer 2 drinkware",
        "left viewer individual side table",
        "right viewer individual side table",
    ),
    "SHARED_REMOTE_REGION": (
        "shared remote support",
        "support the television remote",
        "shared control surface",
        "hold the remote for both viewers",
        "support the tv remote",
        "shared media remote support",
        "support television remote",
        "central support for television remote",
        "shared support for tv remote",
        "television remote support",
        "support remote control",
        "support the remote control",
        "support remote",
        "hold items for viewers",
        "fixed central coffee table surface accessible to both seated viewers for supporting the shared tv remote",
        "central coffee table surface",
        "coffee table surface for remote",
        "central control surface",
        "shared remote surface",
        "central coffee table",
        "television remote for both viewers",
    ),
}

LIVING_OBJECT_ROLE_ALIASES = {
    "CUP_SAUCER_SET": (
        "contain hot beverage and saucer",
        "individual cup and saucer drinkware set for each person",
        "cup and saucer set",
        "cup and saucer",
        "drinkware set",
        "cup and saucer drinkware set",
        "individual cup and saucer set",
        "contain hot beverage and saucer set",
        "hot beverage and saucer",
        "cup and saucer set for each person",
        "individual cup and saucer",
        "drinkware set for viewers",
    ),
    "CUP_COMPONENT": (
        "contain hot beverage",
        "contain coffee",
        "contain tea",
        "hold hot beverage",
        "hold coffee",
        "hold tea",
        "drinking cup",
        "hot beverage cup",
        "coffee cup",
        "tea cup",
        "cup component",
        "drink vessel",
        "beverage cup",
        "individual cup",
    ),
    "SAUCER_COMPONENT": (
        "saucer component",
        "cup saucer",
        "saucer plate",
        "drink saucer",
        "support cup",
        "under cup saucer",
        "saucer for cup",
        "saucer for drinkware",
        "saucer for cup component",
        "individual saucer",
    ),
    "REMOTE": (
        "control television",
        "handheld television remote control device",
        "handheld television remote control",
        "television remote control",
        "tv remote control",
        "remote control",
        "control tv",
        "tv remote",
        "television remote",
        "handheld remote",
        "shared remote control",
    ),
}

LIVING_FIXED_TARGET_ROLE_ALIASES = {
    "SEATING_POSITION": (
        "viewer seating position",
        "individual viewer seating position",
        "viewer seat",
        "seating position",
        "seat position",
        "viewer armchair position",
        "armchair position",
        "seating position for one viewer",
        "individual seat",
        "seated viewer position",
        "viewer chair",
    ),
    "SEATING_PAIR": (
        "paired viewer seating area",
        "both viewer seating positions collectively",
        "pair of seats",
        "paired seats",
        "both viewer seats",
        "seating pair",
        "pair of armchairs",
        "both seating positions",
        "paired seating positions",
        "both seats",
    ),
}

LIVING_INTERACTION_GROUP_ALIASES = {
    "personal_support_group": (
        "support drinkware set beside seat",
        "support drinkware beside seat",
        "place personal drinkware beside viewer",
        "provide personal drinkware support",
        "support cup and saucer set near seat",
        "personal cup and saucer support",
        "support drinkware",
        "support personal drinkware",
        "support cup and saucer",
        "support personal cup and saucer",
        "support drinkware set near seat",
    )
}

LIVING_BINARY_RELATION_ALIASES = {
    "FITS_SET_ON": (
        "can hold drinkware set",
        "supports drinkware set",
        "support drinkware set",
        "suitable for drinkware set",
        "cup and saucer set fits on",
        "drinkware set placed on",
        "drinkware set rests on",
        "hold drinkware set",
        "fit the complete set",
        "fit a cup and saucer together",
        "support the payload set",
        "enough usable area for the set",
        "surface area sufficient to hold both items",
        "hold a cup and saucer set",
        "fit cup and saucer",
        "fits cup and saucer",
        "fits cup and saucer set",
        "accommodate cup and saucer set",
        "support drinkware",
        "support personal drinkware",
        "fits personal drinkware",
        "fit personal drinkware",
        "fit set",
        "fits set",
        "hold cup and saucer",
        "hold the complete set",
        "support cup and saucer",
    ),
    "FITS_ON": (
        "can hold remote",
        "supports remote",
        "support remote",
        "suitable for remote",
        "remote placed on",
        "remote rests on",
        "hold remote",
        "fit the payload",
        "fit the remote",
        "fit remote",
        "fits remote",
        "support the remote",
        "enough usable area for the object",
        "large enough to accommodate the remote",
        "fit television remote",
        "support television remote",
        "accommodate television remote",
        "fits television remote",
        "accommodate remote",
    ),
    "NEAR_SEAT": (
        "near seat",
        "beside seat",
        "adjacent to seating position",
        "near the assigned seat",
        "within reach of one seated person",
        "within reach of one seated viewer armchair",
        "within reach of one seated viewer",
        "within reach of seat",
        "within reach of armchair",
        "within reach of the armchair",
        "reach of one seated",
        "reach of seat",
        "reach of armchair",
        "personal access from the seat",
        "adjacent to the viewer",
        "adjacent to viewer",
        "adjacent to seat",
        "adjacent to armchair",
        "position relative to the armchair",
        "accessibility during viewing",
        "near the seat",
        "near armchair",
        "near assigned seat",
        "near assigned seating",
        "near seating area",
        "near seating",
        "near seats",
        "near seat position",
    ),
    "ACCESSIBLE_FROM_BOTH_SEATS": (
        "accessible from both seats",
        "accessible from both",
        "accessible to both",
        "accessible to both viewers",
        "accessible from both seating positions",
        "reachable by both",
        "reachable by both viewers",
        "shared access",
        "shared access from both",
        "shared access from both seating positions",
        "both viewers",
        "centrally accessible",
        "central access",
        "between seats",
        "between seating positions",
        "between armchairs",
        "within easy reach of both viewers",
    ),
}


def map_living_room_role_function(raw: dict[str, Any] | str) -> str | None:
    """Deterministic concept matching for Living Room functional REGION roles using function and description only.

    Candidate categories are strictly excluded from role semantic authority.
    """
    if isinstance(raw, dict):
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None

    # 1. Exact or forward phrase match against reviewed aliases
    for role_name, aliases in LIVING_REGION_ROLE_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                return role_name

    # 2. Semantic analysis on function + description tokens
    words = set(norm.split())
    has_personal = any(
        w in words or _contains_phrase(norm, w)
        for w in (
            "personal", "individual", "viewer", "occupant",
            "beside seat", "near seat", "viewer 1", "viewer 2",
        )
    )
    has_shared = any(
        w in words or _contains_phrase(norm, w)
        for w in (
            "shared", "central", "both", "common", "mutual", "viewers", "accessible to both",
        )
    )
    has_drink = any(
        w in words or _contains_phrase(norm, w)
        for w in ("cup", "saucer", "drink", "drinkware", "beverage", "tea", "coffee")
    )
    has_remote = any(
        w in words or _contains_phrase(norm, w)
        for w in ("remote", "controller", "tv", "television")
    )

    if (has_personal or has_drink) and not (has_shared or has_remote):
        return "PERSONAL_CUP_SAUCER_REGION"
    if (has_shared or has_remote) and not (has_personal and has_drink):
        return "SHARED_REMOTE_REGION"
    return None


def map_living_room_object_payload_role(raw: dict[str, Any] | str) -> str | None:
    """Deterministic concept matching for Living Room task-explicit payload OBJECT roles.

    Uses function and description only. Candidate categories are strictly excluded.
    Returns 'CUP_SAUCER_SET', 'CUP_COMPONENT', 'SAUCER_COMPONENT', 'REMOTE', or None.
    """
    if isinstance(raw, dict):
        if raw.get("entity_kind") not in (None, "OBJECT"):
            return None
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None

    # 1. Forward match against reviewed aliases
    for role_name, aliases in LIVING_OBJECT_ROLE_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                return role_name

    # 2. Semantic analysis on function + description tokens
    words = set(norm.split())
    has_cup = any(
        w in words or _contains_phrase(norm, w)
        for w in ("cup", "drinking cup", "coffee cup", "tea cup", "drink vessel", "hold liquid", "contain liquid", "liquid vessel")
    )
    has_saucer = any(
        w in words or _contains_phrase(norm, w)
        for w in ("saucer", "saucer plate", "under cup", "support drink", "support cup", "flat dish")
    )
    has_drinkware = any(
        w in words or _contains_phrase(norm, w)
        for w in ("drinkware", "beverage set", "cup and saucer")
    )
    has_remote = any(
        w in words or _contains_phrase(norm, w)
        for w in ("remote", "tv remote", "remote control", "television remote", "control television", "control tv")
    )

    if has_remote and not (has_cup or has_saucer or has_drinkware):
        return "REMOTE"
    if (has_cup and has_saucer) or has_drinkware:
        return "CUP_SAUCER_SET"
    if has_cup and not has_saucer:
        return "CUP_COMPONENT"
    if has_saucer and not has_cup:
        return "SAUCER_COMPONENT"
    return None


def map_living_room_fixed_target_role(raw: dict[str, Any] | str) -> str | None:
    """Deterministic concept matching for Living Room contextual FIXED_TARGET roles.

    Uses function and description only. Candidate categories are strictly excluded.
    Returns 'SEATING_POSITION', 'SEATING_PAIR', or None.
    """
    if isinstance(raw, dict):
        if raw.get("entity_kind") not in (None, "FIXED_TARGET"):
            return None
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None

    # 1. Forward match against reviewed aliases
    for role_name, aliases in LIVING_FIXED_TARGET_ROLE_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                return role_name

    # 2. Semantic analysis on function + description tokens
    words = set(norm.split())
    has_pair = any(
        w in words or _contains_phrase(norm, w)
        for w in ("pair", "both", "paired", "collectively", "both seats", "seating area")
    )
    has_seat = any(
        w in words or _contains_phrase(norm, w)
        for w in ("seat", "seating", "viewer", "armchair", "position", "chair", "seated")
    )

    if has_pair and has_seat:
        return "SEATING_PAIR"
    if has_seat:
        return "SEATING_POSITION"
    return None


def map_living_room_operation_group_function(function_text: str) -> str | None:
    """Map raw interaction group function phrase to canonical Living Room group name."""
    norm = _phrase(function_text)
    if not norm:
        return None
    for group_name, aliases in LIVING_INTERACTION_GROUP_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                return group_name
    return None


def _extract_disjoint_slot_identity(function_text: str, description_text: str) -> str | None:
    """Extract explicit disjoint slot identity (VIEWER_1 vs VIEWER_2) from function and description ONLY."""
    text = f"{function_text} {description_text}".lower()

    v1_indicators = (
        "viewer 1", "viewer_1", "viewer 1's", "viewer 1s", "first viewer", "viewer one",
        "left viewer", "left seat", "left side", "left chair", "left armchair",
        "first seat", "seat 1", "seat_1", "seat one",
    )
    v2_indicators = (
        "viewer 2", "viewer_2", "viewer 2's", "viewer 2s", "second viewer", "viewer two",
        "right viewer", "right seat", "right side", "right chair", "right armchair",
        "second seat", "seat 2", "seat_2", "seat two",
    )

    has_v1 = any(_contains_phrase(text, ind) for ind in v1_indicators)
    has_v2 = any(_contains_phrase(text, ind) for ind in v2_indicators)

    if has_v1 and not has_v2:
        return "VIEWER_1"
    if has_v2 and not has_v1:
        return "VIEWER_2"
    return None


def canonicalize_living_room_relation(
    relation_text: str,
    subject_role: str,
    object_role: str,
    relation_aliases: dict[str, list[str]] | None = None,
) -> tuple[str, str, str, str]:
    """Signature-aware relation canonicalizer returning (canonical_subject, canonical_predicate, canonical_object, direction_status).

    Allowed canonical signatures:
    1. (PERSONAL_CUP_SAUCER_REGION, FITS_SET_ON, CUP_SAUCER_SET)
    2. (PERSONAL_CUP_SAUCER_REGION, NEAR_SEAT, SEATING_POSITION)
    3. (SHARED_REMOTE_REGION, FITS_ON, REMOTE)
    4. (SHARED_REMOTE_REGION, ACCESSIBLE_FROM_BOTH_SEATS, SEATING_PAIR)
    """
    norm = _phrase(relation_text)
    if not norm:
        raise UnmappedFunctionalConceptError("Empty relation text cannot be mapped")

    # Generic fragments that alone cannot establish a relation
    if norm in {"on", "placed", "holds", "accessible", "near", "support", "fit"}:
        raise UnmappedFunctionalConceptError(
            f"Generic relation fragment {relation_text!r} is insufficient to establish a reviewed Living Room relation"
        )

    alias_table = LIVING_BINARY_RELATION_ALIASES if relation_aliases is None else relation_aliases
    matched_predicates = set()
    for pred, aliases in alias_table.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matched_predicates.add(pred)
                break

    # If not matched, check contextual passive/placement verbs with exact pairs
    if not matched_predicates:
        if any(_contains_phrase(norm, p) for p in ("placed on", "placed upon", "rests on", "rest on", "sits on", "set on")):
            if {subject_role, object_role} == {"PERSONAL_CUP_SAUCER_REGION", "CUP_SAUCER_SET"}:
                matched_predicates.add("FITS_SET_ON")
            elif {subject_role, object_role} == {"SHARED_REMOTE_REGION", "REMOTE"}:
                matched_predicates.add("FITS_ON")

    if not matched_predicates:
        raise UnmappedFunctionalConceptError(
            f"VLM living room relation {relation_text!r} cannot be mapped to any reviewed relation "
            f"(available: {sorted(LIVING_BINARY_RELATION_ALIASES.keys())})"
        )
    if len(matched_predicates) > 1:
        raise AmbiguousCanonicalizationError(
            f"Ambiguous living room relation {relation_text!r} matches multiple predicates: {sorted(matched_predicates)}"
        )

    predicate = next(iter(matched_predicates))

    # Define expected signatures: (expected_subject, expected_object)
    expected_signatures = {
        "FITS_SET_ON": ("PERSONAL_CUP_SAUCER_REGION", "CUP_SAUCER_SET"),
        "FITS_ON": ("SHARED_REMOTE_REGION", "REMOTE"),
        "NEAR_SEAT": ("PERSONAL_CUP_SAUCER_REGION", "SEATING_POSITION"),
        "ACCESSIBLE_FROM_BOTH_SEATS": ("SHARED_REMOTE_REGION", "SEATING_PAIR"),
    }

    if predicate not in expected_signatures:
        raise MalformedVLMSpecificationError(f"Unknown predicate {predicate!r} in Living Room domain")

    exp_s, exp_o = expected_signatures[predicate]

    if subject_role == exp_s and object_role == exp_o:
        return (exp_s, predicate, exp_o, "PRESERVED")
    elif subject_role == exp_o and object_role == exp_s:
        return (exp_s, predicate, exp_o, "NORMALIZED_TO_CANONICAL_SIGNATURE")
    else:
        raise MalformedVLMSpecificationError(
            f"Relation {relation_text!r} mapped to predicate {predicate!r} expects endpoints ({exp_s}, {exp_o}), "
            f"but got ({subject_role}, {object_role})"
        )


def map_living_room_relation(
    relation_text: str,
    relation_aliases: dict[str, list[str]] | None = None,
    subject_role: str | None = None,
    object_role: str | None = None,
    *,
    fail_closed: bool = True,
) -> str | None:
    """Deterministic concept matching for Living Room binary relations via reviewed alias table."""
    try:
        if subject_role is not None and object_role is not None:
            _, pred, _, _ = canonicalize_living_room_relation(
                relation_text,
                subject_role,
                object_role,
                relation_aliases=relation_aliases,
            )
            return pred

        norm = _phrase(relation_text)
        if not norm:
            if fail_closed:
                raise UnmappedFunctionalConceptError("Empty relation text cannot be mapped")
            return None
        if norm in {"on", "placed", "holds", "accessible", "near", "support", "fit"}:
            if fail_closed:
                raise UnmappedFunctionalConceptError(
                    f"Generic relation fragment {relation_text!r} is insufficient to establish a reviewed Living Room relation"
                )
            return None
        alias_table = LIVING_BINARY_RELATION_ALIASES if relation_aliases is None else relation_aliases
        matches = set()
        for pred, aliases in alias_table.items():
            for alias in aliases:
                a_norm = _phrase(alias)
                if a_norm == norm or _contains_phrase(norm, a_norm):
                    matches.add(pred)
                    break
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            if fail_closed:
                raise AmbiguousCanonicalizationError(
                    f"Ambiguous living room relation {relation_text!r} matches: {sorted(matches)}"
                )
            return None
        if fail_closed:
            raise UnmappedFunctionalConceptError(
                f"VLM living room relation {relation_text!r} cannot be mapped to any reviewed relation"
            )
        return None
    except Exception:
        if fail_closed:
            raise
        return None


class EnvironmentVLMRequirementProvider:
    """Generate once, then audit against a frozen environment contract."""

    def __init__(
        self,
        environment: str,
        *,
        fm_adapter: FMAdapter | None = None,
        normalization_path: str | Path = DEFAULT_NORMALIZATION,
    ) -> None:
        if environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"environment must be one of {', '.join(SUPPORTED_ENVIRONMENTS)}"
            )
        self.environment = environment
        self.fm_adapter = fm_adapter or FMAdapter()
        normalization = _load_yaml(_resolve(normalization_path))
        if normalization.get("schema_version") != 1:
            raise ValueError("Unsupported Kitchen/Living-Room VLM normalization schema")
        self.unary_property_aliases = normalization.get("unary_property_aliases", {})
        self.binary_relation_aliases = normalization.get("binary_relation_aliases", {})
        self.relation_aliases = normalization.get("relation_aliases", {}) or {
            **self.unary_property_aliases,
            **self.binary_relation_aliases,
        }
        self.environment_config = normalization["environments"][environment]
        if environment == "living_room":
            self.supported_unary_predicates = set(LIVING_SUPPORTED_UNARY_PREDICATES)
        elif environment == "kitchen":
            self.supported_unary_predicates = set(KITCHEN_SUPPORTED_UNARY_PREDICATES)
        else:
            self.supported_unary_predicates = set(self.unary_property_aliases.keys())
        self.task_path = _resolve(self.environment_config["task_path"])
        self.vocabulary_path = _resolve(self.environment_config["vocabulary_path"])
        self.manual_task = _load_yaml(self.task_path)
        self.vocabulary = _load_yaml(self.vocabulary_path)
        self.instruction = str(
            self.manual_task[self.environment_config["instruction_field"]]
        )
        self.task_instruction = self.instruction
        self.raw_decomposition: dict[str, Any] | None = None
        self.normalized_task: dict[str, Any] | None = None
        self.normalized_requirements: list[dict[str, Any]] | None = None
        self.normalized_relations: list[dict[str, Any]] = []
        self.vlm_derived_role_vocabulary: tuple[str, ...] = ()
        self.task_explicit_context_vocabulary: tuple[str, ...] = ()
        self.ranked_detector_vocabulary: list[dict[str, Any]] | None = None
        self.normalization_issues: list[str] = []
        self.ready_for_grounding = False
        self.inspection_policy: dict[str, Any] | None = None

    def _vocabulary_aliases(self) -> dict[str, list[str]]:
        raw = self.vocabulary.get("canonical_labels", {})
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"Semantic vocabulary is empty: {self.vocabulary_path}")
        return {
            str(canonical): [str(alias) for alias in aliases]
            for canonical, aliases in raw.items()
        }

    def _role_specs(self) -> list[dict[str, Any]]:
        language_roles = self.environment_config["roles"]
        if self.environment == "kitchen":
            relations_by_subject: dict[str, list[str]] = {}
            for relation in self.manual_task.get("relations", []):
                relations_by_subject.setdefault(relation["subject_role"], []).append(
                    relation["predicate"]
                )
            specifications = []
            for role_id, role in self.manual_task["roles"].items():
                properties = [
                    item["predicate"] for item in role.get("unary_geometry", [])
                ]
                specifications.append(
                    {
                        "role_id": role_id,
                        "canonical_function": role_id.upper(),
                        "entity_kind": "OBJECT",
                        "purpose": language_roles[role_id]["function_hint"],
                        "function_aliases": language_roles[role_id]["function_aliases"],
                        "required_count": int(
                            role.get("count", role.get("binding_cardinality", {}).get(
                                "minimum_distinct_physical_objects", 1
                            ))
                        ),
                        "categories": [
                            preference["canonical_label"]
                            for preference in role["semantic_preferences"]
                        ],
                        "properties": list(dict.fromkeys(properties)),
                    }
                )
            for role_id, role in self.manual_task.get("symbolic_task", {}).get(
                "source_roles", {}
            ).items():
                specifications.append(
                    {
                        "role_id": role_id,
                        "canonical_function": f"PROVIDE_{str(role['provides']).upper()}",
                        "entity_kind": "OBJECT",
                        "purpose": language_roles[role_id]["function_hint"],
                        "function_aliases": language_roles[role_id]["function_aliases"],
                        "required_count": int(role.get("count", 1)),
                        "categories": list(role["accepted_semantic_labels"]),
                        "properties": [],
                    }
                )
            return specifications

        specifications = []
        region_roles = self.manual_task["semantic_requirements"]["region_roles"]
        for role_id, group in self.manual_task["function_groups"].items():
            role_name = group["region_role"]
            specifications.append(
                {
                    "role_id": role_id,
                    "canonical_function": group["function_id"],
                    "entity_kind": "REGION",
                    "purpose": language_roles[role_id]["function_hint"],
                    "function_aliases": language_roles[role_id]["function_aliases"],
                    "required_count": int(group.get("required_target_count", 1)),
                    "categories": list(region_roles[role_name]["accepted_categories"]),
                    "properties": list(group["required_relations"]),
                }
            )
        return specifications

    def _category_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for canonical, aliases in self._vocabulary_aliases().items():
            result[_phrase(canonical)] = canonical
            for alias in aliases:
                result[_phrase(alias)] = canonical
        return result

    def _map_category(self, value: object) -> str | None:
        normalized = _phrase(value)
        if not normalized:
            return None
        aliases = self._category_map()
        if normalized in aliases:
            return aliases[normalized]
        matches = {
            canonical
            for alias, canonical in aliases.items()
            if f" {alias} " in f" {normalized} "
        }
        if len(matches) == 1:
            return next(iter(matches))
        if len(matches) > 1:
            raise AmbiguousCanonicalizationError(f"Ambiguous category match for {value!r}: {sorted(matches)}")
        return None

    def _map_properties(self, values: list[str], *, fail_closed: bool = False) -> set[str]:
        mapped: set[str] = set()
        for value in values:
            norm = _phrase(value)
            prop_matches = set()
            for predicate, aliases in self.unary_property_aliases.items():
                for alias in aliases:
                    a_norm = _phrase(alias)
                    if a_norm == norm or f" {a_norm} " in f" {norm} ":
                        prop_matches.add(predicate)
                        break
            if len(prop_matches) == 1:
                pred = next(iter(prop_matches))
                if pred in self.supported_unary_predicates:
                    mapped.add(pred)
                elif fail_closed:
                    raise UnsupportedCheckerCapabilityError(
                        f"Unary predicate {pred!r} (mapped from {value!r}) is not supported by checkers in domain {self.environment}"
                    )
            elif len(prop_matches) > 1:
                if "PLANAR_SUPPORT" in prop_matches and self.environment == "living_room":
                    mapped.add("PLANAR_SUPPORT")
                elif fail_closed:
                    raise AmbiguousCanonicalizationError(
                        f"Ambiguous property {value!r} matches multiple unary predicates: {sorted(prop_matches)}"
                    )
            elif fail_closed:
                raise UnsupportedCheckerCapabilityError(
                    f"VLM required property {value!r} is not supported by any available checker in {self.environment}"
                )
        return mapped

    def _detector_label(self, canonical: str) -> str:
        aliases = self._vocabulary_aliases()[canonical]
        return aliases[0] if aliases else canonical.replace("_", " ")

    def _candidate_categories(self, raw: dict[str, Any]) -> list[str]:
        categories: list[str] = []
        for candidate in raw.get("visible_candidates", []):
            canonical = self._map_category(candidate.get("label", ""))
            if canonical is not None and canonical not in categories:
                categories.append(canonical)
        for cat in raw.get("candidate_categories", []):
            canonical = self._map_category(cat)
            if canonical is not None and canonical not in categories:
                categories.append(canonical)
        return categories

    def _assign_roles(
        self,
        raw_requirements: list[dict[str, Any]],
        specs: list[dict[str, Any]],
    ) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[str]]:
        issues: list[str] = []
        matched: dict[str, list[dict[str, Any]]] = {}
        for raw in raw_requirements:
            func_text = _phrase(f"{raw.get('function', '')} {raw.get('description', '')}")
            matching_specs = []
            for spec in specs:
                if raw.get("entity_kind") != spec.get("entity_kind"):
                    continue
                for alias in spec.get("function_aliases", []):
                    a_norm = _phrase(alias)
                    if a_norm == func_text or f" {a_norm} " in f" {func_text} ":
                        matching_specs.append(spec)
                        break
            if len(matching_specs) != 1:
                issues.append(
                    f"raw role {raw['id']!r} is unmapped or ambiguous; "
                    f"matches={[s['role_id'] for s in matching_specs]}"
                )
                continue
            spec = matching_specs[0]
            role_id = spec["role_id"]
            matched.setdefault(role_id, []).append(raw)
        for spec in specs:
            if spec["role_id"] not in matched:
                issues.append(f"reviewed role {spec['role_id']!r} was not recovered")
        return [
            (spec, matched[spec["role_id"]])
            for spec in specs
            if spec["role_id"] in matched
        ], issues

    def generate_canonical(
        self,
        instruction: str | None = None,
        *,
        observation_images: list[str | Path] | None = None,
        raw_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.generate(
            instruction,
            observation_images=observation_images,
            raw_document=raw_document,
            canonical=True,
        )

    def generate(
        self,
        instruction: str | None = None,
        *,
        observation_images: list[str | Path] | None = None,
        raw_document: dict[str, Any] | None = None,
        require_reviewed_contract: bool = False,
        include_inspection_policy: bool = False,
        canonical: bool = False,
    ) -> dict[str, Any]:
        if raw_document is not None:
            self.raw_decomposition = raw_document
        elif self.raw_decomposition is not None:
            if require_reviewed_contract and not self.ready_for_grounding:
                raise ValueError(
                    "VLM output is not ready for grounding: "
                    + "; ".join(self.normalization_issues)
                )
            return self.result()
        task_instruction = instruction or self.instruction
        self.task_instruction = task_instruction
        if raw_document is None:
            document = self.fm_adapter.generate_task_requirements(
                task_instruction, observation_images=observation_images or []
            )
            self.raw_decomposition = document
        else:
            document = raw_document
        if include_inspection_policy:
            if self.environment != "kitchen":
                raise ValueError("VLM inspection policy is currently Kitchen-only")
            self.inspection_policy = self.fm_adapter.generate_inspection_priors(
                task_instruction,
                KITCHEN_SEARCH_REGIONS,
                observation_images=observation_images or [],
            )
        if document.get("status") != "SUPPORTED":
            raise VLMSpecificationError(
                f"VLM marked {self.environment} unsupported: "
                f"{document.get('unsupported_reason', 'no reason')}"
            )
        raw_requirements = document.get("functional_roles", [])
        normalized_records = []
        category_rank: dict[str, int] = {}
        issues: list[str] = []

        if not canonical:
            specs = self._role_specs()
            assigned, issues = self._assign_roles(raw_requirements, specs)
            for spec, raw_group in assigned:
                raw = {
                    "id": raw_group[0]["id"],
                    "entity_kind": spec["entity_kind"],
                    "function": " / ".join(item["function"] for item in raw_group),
                    "description": " ".join(item["description"] for item in raw_group),
                    "required_count": sum(item["required_count"] for item in raw_group),
                    "visible_candidates": [
                        candidate
                        for item in raw_group
                        for candidate in item.get("visible_candidates", [])
                    ],
                    "required_properties": list(dict.fromkeys(
                        property_text
                        for item in raw_group
                        for property_text in item.get("required_properties", [])
                    )),
                }
                categories = self._candidate_categories(raw)
                for canonical_cat in categories:
                    if canonical_cat in spec["categories"]:
                        category_rank.setdefault(canonical_cat, len(category_rank) + 1)
                properties = self._map_properties(raw["required_properties"], fail_closed=False)
                missing_properties = set(spec["properties"]) - properties
                if missing_properties:
                    issues.append(
                        f"VLM role {spec['role_id']} omitted required properties: "
                        f"{sorted(missing_properties)}"
                    )
                count_matches = raw["required_count"] == spec["required_count"]
                if not count_matches:
                    issues.append(
                        f"VLM role {spec['role_id']} required_count={raw['required_count']} "
                        f"but reviewed minimum is {spec['required_count']}"
                    )
                normalized_records.append(
                    {
                        "role_id": spec["role_id"],
                        "raw_vlm_role_id": raw["id"],
                        "raw_vlm_role_ids": [item["id"] for item in raw_group],
                        "entity_kind": spec["entity_kind"],
                        "function": spec["canonical_function"],
                        "raw_function": raw["function"],
                        "vlm_required_count": raw["required_count"],
                        "reviewed_required_count": spec["required_count"],
                        "description": raw["description"],
                        "accepted_categories": list(spec["categories"]),
                        "visible_candidates": raw["visible_candidates"],
                        "required_properties": list(spec["properties"]),
                        "semantic_hints": [
                            candidate["label"] for candidate in raw["visible_candidates"] if candidate.get("label")
                        ],
                        "source": "FM",
                        "provenance": "qwen_vlm_normalized_by_reviewed_ontology",
                        "normalization_status": (
                            "COMPLETE"
                            if not missing_properties and count_matches
                            else "REVIEW_REQUIRED"
                        ),
                        "missing_reviewed_properties": sorted(missing_properties),
                    }
                )
            self.normalization_issues = issues
            self.ready_for_grounding = not issues
            if self.ready_for_grounding:
                normalized_task = deepcopy(self.manual_task)
                normalized_task["specification_source"] = (
                    "qwen_vlm_normalized_by_reviewed_ontology"
                )
                normalized_task["generated_from_foundation_model"] = True
                self.normalized_task = normalized_task
            else:
                self.normalized_task = None
        else:
            # Validate schema-required role fields before semantic compilation (fail-closed, no silent defaults)
            raw_ids_seen = set()
            for raw in raw_requirements:
                if not isinstance(raw, dict):
                    raise MalformedVLMSpecificationError(f"Role specification must be a dictionary, got {type(raw)}")
                raw_id = raw.get("id")
                if not raw_id or not isinstance(raw_id, str):
                    raise MalformedVLMSpecificationError(f"Duplicate or missing raw role id: {raw_id!r}")
                if raw_id in raw_ids_seen:
                    raise MalformedVLMSpecificationError(f"Duplicate or missing raw role id: {raw_id!r}")
                raw_ids_seen.add(raw_id)

                if raw.get("entity_kind") is None or not isinstance(raw.get("entity_kind"), str):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing required 'entity_kind'")
                if raw.get("function") is None or not isinstance(raw.get("function"), str):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing required 'function'")
                if raw.get("required_count") is None or not isinstance(raw.get("required_count"), int):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing required 'required_count'")
                if raw.get("binding_policy") is None or not isinstance(raw.get("binding_policy"), str):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing required 'binding_policy'")
                if raw.get("candidate_categories") is None or not isinstance(raw.get("candidate_categories"), list):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing candidate_categories list")
                if raw.get("visible_candidates") is None or not isinstance(raw.get("visible_candidates"), list):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing visible_candidates list")
                if raw.get("required_properties") is None or not isinstance(raw.get("required_properties"), list):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} is missing required_properties list")

                raw_kind = raw["entity_kind"]
                raw_count = raw["required_count"]
                if raw_count < 1:
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} required_count must be >= 1, got {raw_count}")

                raw_policy = raw["binding_policy"]
                if raw_policy not in ("DISTINCT", "SHARED", "REUSABLE"):
                    raise MalformedVLMSpecificationError(f"Role {raw_id!r} has invalid binding_policy: {raw_policy!r}")

            # Validate schema-required interaction group fields before semantic compilation
            for grp in self.raw_decomposition.get("interaction_groups", []):
                if not isinstance(grp, dict):
                    raise MalformedVLMSpecificationError(f"Interaction group must be a dict, got {type(grp)}")
                gid = grp.get("id")
                if not gid or not isinstance(gid, str):
                    raise MalformedVLMSpecificationError(f"Interaction group is missing required 'id': {grp!r}")
                if grp.get("function") is None or not isinstance(grp.get("function"), str):
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} is missing required 'function'")
                if grp.get("tool_role") is None or not isinstance(grp.get("tool_role"), str):
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} is missing required 'tool_role'")
                if grp.get("target_role") is None or not isinstance(grp.get("target_role"), str):
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} is missing required 'target_role'")
                if grp.get("required_target_count") is None or not isinstance(grp.get("required_target_count"), int):
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} is missing required 'required_target_count'")
                if grp["required_target_count"] < 1:
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} required_target_count must be >= 1, got {grp['required_target_count']}")
                if grp.get("usage_policy") is None or not isinstance(grp.get("usage_policy"), str):
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} is missing required 'usage_policy'")
                if grp.get("required_relations") is None or not isinstance(grp.get("required_relations"), list) or not grp["required_relations"]:
                    raise MalformedVLMSpecificationError(f"Interaction group {gid!r} is missing non-empty required_relations")

            # Validate schema-required functional relation fields before semantic compilation
            for rel in self.raw_decomposition.get("functional_relations", []):
                if not isinstance(rel, dict):
                    raise MalformedVLMSpecificationError(f"Functional relation must be a dict, got {type(rel)}")
                if not rel.get("subject_role") or not isinstance(rel.get("subject_role"), str):
                    raise MalformedVLMSpecificationError(f"Functional relation is missing required 'subject_role': {rel!r}")
                if not rel.get("relation") or not isinstance(rel.get("relation"), str):
                    raise MalformedVLMSpecificationError(f"Functional relation is missing required 'relation': {rel!r}")
                if not rel.get("object_role") or not isinstance(rel.get("object_role"), str):
                    raise MalformedVLMSpecificationError(f"Functional relation is missing required 'object_role': {rel!r}")

            raw_id_to_canon: dict[str, str] = {}
            classified_roles: dict[str, list[dict[str, Any]]] = {}

            for raw in raw_requirements:
                raw_id = raw["id"]
                raw_kind = raw["entity_kind"]

                if raw_kind == "REGION":
                    mapped = map_living_room_role_function(raw)
                    if mapped is None:
                        raise UnmappedFunctionalConceptError(
                            f"VLM REGION role {raw_id!r} with function {raw.get('function')!r} "
                            "cannot be mapped to any canonical Living Room region role"
                        )
                    canon_name = "PERSONAL_CUP_SAUCER_REGION" if "personal" in mapped.lower() else "SHARED_REMOTE_REGION"
                    classified_roles.setdefault(canon_name, []).append({"raw": raw, "component": canon_name})
                    raw_id_to_canon[raw_id] = canon_name

                elif raw_kind == "OBJECT":
                    mapped = map_living_room_object_payload_role(raw)
                    if mapped is None:
                        raise UnmappedFunctionalConceptError(
                            f"VLM OBJECT role {raw_id!r} with function {raw.get('function')!r} "
                            "cannot be mapped to any canonical Living Room object role"
                        )
                    if mapped in ("CUP_SAUCER_SET", "CUP_COMPONENT", "SAUCER_COMPONENT"):
                        classified_roles.setdefault("CUP_SAUCER_SET", []).append({"raw": raw, "component": mapped})
                        raw_id_to_canon[raw_id] = "CUP_SAUCER_SET"
                    elif mapped == "REMOTE":
                        classified_roles.setdefault("REMOTE", []).append({"raw": raw, "component": "REMOTE"})
                        raw_id_to_canon[raw_id] = "REMOTE"

                elif raw_kind == "FIXED_TARGET":
                    mapped = map_living_room_fixed_target_role(raw)
                    if mapped is None:
                        raise UnmappedFunctionalConceptError(
                            f"VLM FIXED_TARGET role {raw_id!r} with function {raw.get('function')!r} "
                            "cannot be mapped to any canonical Living Room fixed target role"
                        )
                    classified_roles.setdefault(mapped, []).append({"raw": raw, "component": mapped})
                    raw_id_to_canon[raw_id] = mapped

                else:
                    raise MalformedVLMSpecificationError(f"Unknown entity_kind {raw_kind!r} for role {raw_id!r}")

            concept_accounting: dict[str, Any] = {
                "roles": {},
                "properties": [],
                "relations": [],
                "operation_groups": [],
            }
            role_properties_map: dict[str, list[str]] = {}
            raw_role_properties_map: dict[str, list[str]] = {}

            for raw in raw_requirements:
                raw_id = raw["id"]
                raw_kind = raw["entity_kind"]
                seen_props: set[str] = set()
                for prop in raw.get("required_properties", []):
                    if not isinstance(prop, str):
                        raise MalformedVLMSpecificationError(f"Property must be string, got {prop!r}")
                    norm_p = _phrase(prop)
                    mapped_p = None
                    if any(a == norm_p or _contains_phrase(norm_p, a) for a in (
                        "planar support", "planar horizontal support", "horizontal planar support",
                        "planar surface", "flat support", "flat surface", "horizontal surface",
                    )):
                        mapped_p = "PLANAR_SUPPORT"
                    elif any(a == norm_p or _contains_phrase(norm_p, a) for a in (
                        "open cavity", "capable of holding liquid", "cavity",
                    )):
                        mapped_p = "OPEN_CAVITY"
                    elif any(a == norm_p or _contains_phrase(norm_p, a) for a in (
                        "elongated object", "elongated shape", "slender", "elongated",
                    )):
                        mapped_p = "ELONGATED_OBJECT"

                    if mapped_p is None:
                        raise UnmappedFunctionalConceptError(
                            f"Required property {prop!r} on role {raw_id!r} cannot be mapped to any Living Room unary property"
                        )
                    if mapped_p == "PLANAR_SUPPORT" and raw_kind != "REGION":
                        raise MalformedVLMSpecificationError(
                            f"PLANAR_SUPPORT requested on non-REGION role {raw_id!r} ({raw_kind})"
                        )
                    if mapped_p in ("OPEN_CAVITY", "ELONGATED_OBJECT"):
                        raise UnsupportedCheckerCapabilityError(
                            f"Unary predicate {mapped_p!r} is not supported in Living Room domain"
                        )

                    canon_role = raw_id_to_canon[raw_id]
                    raw_role_properties_map.setdefault(raw_id, [])
                    if mapped_p not in raw_role_properties_map[raw_id]:
                        raw_role_properties_map[raw_id].append(mapped_p)

                    if mapped_p not in seen_props:
                        seen_props.add(mapped_p)
                        role_properties_map.setdefault(canon_role, [])
                        if mapped_p not in role_properties_map[canon_role]:
                            role_properties_map[canon_role].append(mapped_p)
                        concept_accounting["properties"].append({
                            "raw_role_id": raw_id,
                            "raw_phrase": prop,
                            "canonical_predicate": mapped_p,
                            "status": "PRESERVED",
                        })
                    else:
                        concept_accounting["properties"].append({
                            "raw_role_id": raw_id,
                            "raw_phrase": prop,
                            "canonical_predicate": mapped_p,
                            "status": "MERGED_BY_EXPLICIT_RULE",
                            "reason": f"Duplicate alias for predicate {mapped_p} on same role",
                        })

            vlm_role_vocab: list[str] = []
            normalized_records: list[dict[str, Any]] = []

            # 1. Synthesis: CUP_SAUCER_SET
            if "CUP_SAUCER_SET" in classified_roles:
                items = classified_roles["CUP_SAUCER_SET"]
                has_composite = any(it["component"] == "CUP_SAUCER_SET" for it in items)
                if has_composite:
                    if len(items) > 1:
                        raise AmbiguousCanonicalizationError(
                            "Multiple composite drinkware roles or mixed composite+component roles in Living Room specification"
                        )
                    raw_item = items[0]["raw"]
                    c_count = int(raw_item["required_count"])
                    c_policy = raw_item["binding_policy"]
                    concept_accounting["roles"][raw_item["id"]] = {
                        "canonical_role": "CUP_SAUCER_SET",
                        "entity_kind": "OBJECT",
                        "raw_count": c_count,
                        "canonical_count": c_count,
                        "binding_policy": c_policy,
                        "unary_predicates": [],
                        "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                        "candidate_categories_used_for_role_identity": False,
                        "status": "PRESERVED",
                    }
                else:
                    cups = [it["raw"] for it in items if it["component"] == "CUP_COMPONENT"]
                    saucers = [it["raw"] for it in items if it["component"] == "SAUCER_COMPONENT"]
                    if len(cups) != 1 or len(saucers) != 1:
                        raise AmbiguousCanonicalizationError(
                            "Component decomposition must have exactly one cup role and one saucer role"
                        )
                    cup_raw = cups[0]
                    saucer_raw = saucers[0]
                    cup_cnt = int(cup_raw["required_count"])
                    saucer_cnt = int(saucer_raw["required_count"])
                    if cup_cnt != saucer_cnt:
                        raise MalformedVLMSpecificationError(
                            f"Mismatched component counts for cup ({cup_cnt}) and saucer ({saucer_cnt})"
                        )
                    cup_pol = cup_raw["binding_policy"]
                    saucer_pol = saucer_raw["binding_policy"]
                    if cup_pol != saucer_pol:
                        raise MalformedVLMSpecificationError(
                            f"Conflicting binding policies between cup ({cup_pol}) and saucer ({saucer_pol})"
                        )
                    c_count = cup_cnt
                    c_policy = cup_pol
                    concept_accounting["roles"][cup_raw["id"]] = {
                        "canonical_role": "CUP_SAUCER_SET",
                        "entity_kind": "OBJECT",
                        "raw_count": cup_cnt,
                        "canonical_count": c_count,
                        "binding_policy": c_policy,
                        "unary_predicates": [],
                        "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                        "candidate_categories_used_for_role_identity": False,
                        "status": "COMPOSED_FROM_COMPONENT_ROLES",
                        "composition_details": {
                            "partner_role_id": saucer_raw["id"],
                            "component_type": "cup",
                            "raw_count": cup_cnt,
                            "composite_count": c_count,
                            "merge_rule": "ONE_SET_PER_CUP_AND_SAUCER_PAIR",
                        },
                    }
                    concept_accounting["roles"][saucer_raw["id"]] = {
                        "canonical_role": "CUP_SAUCER_SET",
                        "entity_kind": "OBJECT",
                        "raw_count": saucer_cnt,
                        "canonical_count": c_count,
                        "binding_policy": c_policy,
                        "unary_predicates": [],
                        "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                        "candidate_categories_used_for_role_identity": False,
                        "status": "COMPOSED_FROM_COMPONENT_ROLES",
                        "composition_details": {
                            "partner_role_id": cup_raw["id"],
                            "component_type": "saucer",
                            "raw_count": saucer_cnt,
                            "composite_count": c_count,
                            "merge_rule": "ONE_SET_PER_CUP_AND_SAUCER_PAIR",
                        },
                    }

                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for it in items
                    for cat in it["raw"].get("candidate_categories", [])
                    if str(cat).strip()
                ))
                vlm_role_vocab.extend(cand_cats)
                run_local_cats = [_phrase(c).replace(" ", "_") for c in cand_cats if c]
                accepted_cats = list(dict.fromkeys(list(LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["CUP_SAUCER_SET"]) + run_local_cats))
                hints = list(dict.fromkeys(
                    candidate["label"]
                    for it in items
                    for candidate in it["raw"].get("visible_candidates", [])
                    if candidate.get("label")
                ))

                normalized_records.append({
                    "role_id": "cup_saucer_set",
                    "raw_vlm_role_ids": [it["raw"]["id"] for it in items],
                    "entity_kind": "OBJECT",
                    "binding_policy": c_policy,
                    "function": "CUP_SAUCER_SET",
                    "raw_function": " / ".join(dict.fromkeys(it["raw"].get("function", "") for it in items)),
                    "vlm_required_count": c_count,
                    "description": " ".join(dict.fromkeys(it["raw"].get("description", "") for it in items)),
                    "candidate_categories": cand_cats,
                    "raw_candidate_categories": cand_cats,
                    "canonical_graph_category": LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["CUP_SAUCER_SET"],
                    "accepted_categories": accepted_cats,
                    "required_properties": [],
                    "visible_candidates": [
                        candidate
                        for it in items
                        for candidate in it["raw"].get("visible_candidates", [])
                    ],
                    "semantic_hints": hints,
                    "source": "FM",
                    "provenance": "qwen_vlm_normalized_by_generic_ontology",
                    "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                    "normalization_status": "COMPLETE",
                })

            # 2. Synthesis: PERSONAL_CUP_SAUCER_REGION
            if "PERSONAL_CUP_SAUCER_REGION" in classified_roles:
                items = classified_roles["PERSONAL_CUP_SAUCER_REGION"]
                raw_list = [it["raw"] for it in items]

                # Validate each contributing raw support role independently for PLANAR_SUPPORT evidence
                for r in raw_list:
                    r_props = raw_role_properties_map.get(r["id"], [])
                    if "PLANAR_SUPPORT" not in r_props:
                        raise MalformedVLMSpecificationError(
                            f"Living Room support role {r['id']!r} omitted required executable property 'PLANAR_SUPPORT'"
                        )

                if len(raw_list) == 1:
                    r = raw_list[0]
                    cnt = int(r["required_count"])
                    pol = r["binding_policy"]
                    concept_accounting["roles"][r["id"]] = {
                        "canonical_role": "PERSONAL_CUP_SAUCER_REGION",
                        "entity_kind": "REGION",
                        "raw_count": cnt,
                        "canonical_count": cnt,
                        "binding_policy": pol,
                        "unary_predicates": list(raw_role_properties_map.get(r["id"], [])),
                        "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                        "candidate_categories_used_for_role_identity": False,
                        "status": "PRESERVED",
                    }
                elif len(raw_list) == 2:
                    slot_0 = _extract_disjoint_slot_identity(raw_list[0].get("function", ""), raw_list[0].get("description", ""))
                    slot_1 = _extract_disjoint_slot_identity(raw_list[1].get("function", ""), raw_list[1].get("description", ""))
                    if slot_0 is None or slot_1 is None:
                        raise AmbiguousCanonicalizationError(
                            f"Multiple personal cup/saucer region roles lack explicit disjoint slot identities (e.g. viewer 1 / viewer 2 or left / right): {[r['id'] for r in raw_list]}"
                        )
                    if slot_0 == slot_1:
                        raise AmbiguousCanonicalizationError(
                            f"Multiple personal cup/saucer region roles declare duplicate slot identity {slot_0!r}: {[r['id'] for r in raw_list]}"
                        )
                    cnt_0 = int(raw_list[0]["required_count"])
                    cnt_1 = int(raw_list[1]["required_count"])
                    if cnt_0 != 1 or cnt_1 != 1:
                        raise MalformedVLMSpecificationError(
                            f"Disjoint personal cup/saucer region roles must each have required_count=1, got counts ({cnt_0}, {cnt_1})"
                        )
                    pol_0 = raw_list[0]["binding_policy"]
                    pol_1 = raw_list[1]["binding_policy"]
                    if pol_0 != "DISTINCT" or pol_1 != "DISTINCT":
                        raise MalformedVLMSpecificationError(
                            f"Disjoint personal cup/saucer region roles must have binding_policy DISTINCT, got ({pol_0}, {pol_1})"
                        )
                    cnt = 2
                    pol = "DISTINCT"
                    for r in raw_list:
                        r_slot = _extract_disjoint_slot_identity(r.get("function", ""), r.get("description", ""))
                        concept_accounting["roles"][r["id"]] = {
                            "canonical_role": "PERSONAL_CUP_SAUCER_REGION",
                            "entity_kind": "REGION",
                            "raw_count": 1,
                            "canonical_count": 2,
                            "binding_policy": "DISTINCT",
                            "unary_predicates": list(raw_role_properties_map.get(r["id"], [])),
                            "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                            "candidate_categories_used_for_role_identity": False,
                            "status": "MERGED_BY_EXPLICIT_RULE",
                            "reason": f"Disjoint personal slot role ({r_slot}) composed into canonical role",
                        }
                else:
                    raise AmbiguousCanonicalizationError(
                        f"Unsupported number of personal cup/saucer region roles: {len(raw_list)} (expected 1 aggregate or 2 disjoint slots)"
                    )

                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for r in raw_list
                    for cat in r.get("candidate_categories", [])
                    if str(cat).strip()
                ))
                vlm_role_vocab.extend(cand_cats)
                accepted_cats = list(dict.fromkeys(_phrase(c).replace(" ", "_") for c in cand_cats if c))
                hints = list(dict.fromkeys(
                    candidate["label"]
                    for r in raw_list
                    for candidate in r.get("visible_candidates", [])
                    if candidate.get("label")
                ))
                props = role_properties_map.get("PERSONAL_CUP_SAUCER_REGION", [])

                normalized_records.append({
                    "role_id": "personal_cup_saucer",
                    "raw_vlm_role_ids": [r["id"] for r in raw_list],
                    "entity_kind": "REGION",
                    "binding_policy": pol,
                    "function": "PERSONAL_CUP_SAUCER_REGION",
                    "raw_function": " / ".join(dict.fromkeys(r.get("function", "") for r in raw_list)),
                    "vlm_required_count": cnt,
                    "description": " ".join(dict.fromkeys(r.get("description", "") for r in raw_list)),
                    "candidate_categories": cand_cats,
                    "raw_candidate_categories": cand_cats,
                    "canonical_graph_category": None,
                    "accepted_categories": accepted_cats,
                    "required_properties": props,
                    "visible_candidates": [
                        candidate
                        for r in raw_list
                        for candidate in r.get("visible_candidates", [])
                    ],
                    "semantic_hints": hints,
                    "source": "FM",
                    "provenance": "qwen_vlm_normalized_by_generic_ontology",
                    "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                    "normalization_status": "COMPLETE",
                })

            # 3. Synthesis: SHARED_REMOTE_REGION
            if "SHARED_REMOTE_REGION" in classified_roles:
                items = classified_roles["SHARED_REMOTE_REGION"]
                raw_list = [it["raw"] for it in items]
                if len(raw_list) > 1:
                    raise AmbiguousCanonicalizationError(
                        f"Multiple raw roles mapping to SHARED_REMOTE_REGION: {[r['id'] for r in raw_list]}"
                    )
                r = raw_list[0]
                r_props = raw_role_properties_map.get(r["id"], [])
                if "PLANAR_SUPPORT" not in r_props:
                    raise MalformedVLMSpecificationError(
                        f"Living Room support role {r['id']!r} omitted required executable property 'PLANAR_SUPPORT'"
                    )

                cnt = int(r["required_count"])
                pol = r["binding_policy"]
                concept_accounting["roles"][r["id"]] = {
                    "canonical_role": "SHARED_REMOTE_REGION",
                    "entity_kind": "REGION",
                    "raw_count": cnt,
                    "canonical_count": cnt,
                    "binding_policy": pol,
                    "unary_predicates": list(raw_role_properties_map.get(r["id"], [])),
                    "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                    "candidate_categories_used_for_role_identity": False,
                    "status": "PRESERVED",
                }
                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for cat in r.get("candidate_categories", [])
                    if str(cat).strip()
                ))
                vlm_role_vocab.extend(cand_cats)
                accepted_cats = list(dict.fromkeys(_phrase(c).replace(" ", "_") for c in cand_cats if c))
                hints = list(dict.fromkeys(
                    candidate["label"]
                    for candidate in r.get("visible_candidates", [])
                    if candidate.get("label")
                ))
                props = role_properties_map.get("SHARED_REMOTE_REGION", [])

                normalized_records.append({
                    "role_id": "shared_remote",
                    "raw_vlm_role_ids": [r["id"]],
                    "entity_kind": "REGION",
                    "binding_policy": pol,
                    "function": "SHARED_REMOTE_REGION",
                    "raw_function": r.get("function", ""),
                    "vlm_required_count": cnt,
                    "description": r.get("description", ""),
                    "candidate_categories": cand_cats,
                    "raw_candidate_categories": cand_cats,
                    "canonical_graph_category": None,
                    "accepted_categories": accepted_cats,
                    "required_properties": props,
                    "visible_candidates": list(r.get("visible_candidates", [])),
                    "semantic_hints": hints,
                    "source": "FM",
                    "provenance": "qwen_vlm_normalized_by_generic_ontology",
                    "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                    "normalization_status": "COMPLETE",
                })

            # 4. Synthesis: REMOTE
            if "REMOTE" in classified_roles:
                items = classified_roles["REMOTE"]
                raw_list = [it["raw"] for it in items]
                if len(raw_list) > 1:
                    raise AmbiguousCanonicalizationError(
                        f"Multiple raw roles mapping to REMOTE: {[r['id'] for r in raw_list]}"
                    )
                r = raw_list[0]
                cnt = int(r["required_count"])
                pol = r["binding_policy"]
                concept_accounting["roles"][r["id"]] = {
                    "canonical_role": "REMOTE",
                    "entity_kind": "OBJECT",
                    "raw_count": cnt,
                    "canonical_count": cnt,
                    "binding_policy": pol,
                    "unary_predicates": [],
                    "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                    "candidate_categories_used_for_role_identity": False,
                    "status": "PRESERVED",
                }
                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for cat in r.get("candidate_categories", [])
                    if str(cat).strip()
                ))
                vlm_role_vocab.extend(cand_cats)
                run_local_cats = [_phrase(c).replace(" ", "_") for c in cand_cats if c]
                accepted_cats = list(dict.fromkeys(list(LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["REMOTE"]) + run_local_cats))
                hints = list(dict.fromkeys(
                    candidate["label"]
                    for candidate in r.get("visible_candidates", [])
                    if candidate.get("label")
                ))

                normalized_records.append({
                    "role_id": "remote",
                    "raw_vlm_role_ids": [r["id"]],
                    "entity_kind": "OBJECT",
                    "binding_policy": pol,
                    "function": "REMOTE",
                    "raw_function": r.get("function", ""),
                    "vlm_required_count": cnt,
                    "description": r.get("description", ""),
                    "candidate_categories": cand_cats,
                    "raw_candidate_categories": cand_cats,
                    "canonical_graph_category": LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["REMOTE"],
                    "accepted_categories": accepted_cats,
                    "required_properties": [],
                    "visible_candidates": list(r.get("visible_candidates", [])),
                    "semantic_hints": hints,
                    "source": "FM",
                    "provenance": "qwen_vlm_normalized_by_generic_ontology",
                    "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                    "normalization_status": "COMPLETE",
                })

            # 5. Synthesis: SEATING_POSITION
            if "SEATING_POSITION" in classified_roles:
                items = classified_roles["SEATING_POSITION"]
                raw_list = [it["raw"] for it in items]
                if len(raw_list) == 1:
                    r = raw_list[0]
                    cnt = int(r["required_count"])
                    pol = r["binding_policy"]
                    concept_accounting["roles"][r["id"]] = {
                        "canonical_role": "SEATING_POSITION",
                        "entity_kind": "FIXED_TARGET",
                        "raw_count": cnt,
                        "canonical_count": cnt,
                        "binding_policy": pol,
                        "unary_predicates": [],
                        "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                        "candidate_categories_used_for_role_identity": False,
                        "status": "PRESERVED",
                    }
                elif len(raw_list) == 2:
                    slot_0 = _extract_disjoint_slot_identity(raw_list[0].get("function", ""), raw_list[0].get("description", ""))
                    slot_1 = _extract_disjoint_slot_identity(raw_list[1].get("function", ""), raw_list[1].get("description", ""))
                    if slot_0 is None or slot_1 is None:
                        raise AmbiguousCanonicalizationError(
                            f"Multiple seating position roles lack explicit disjoint slot identities (e.g. viewer 1 / viewer 2 or left / right): {[r['id'] for r in raw_list]}"
                        )
                    if slot_0 == slot_1:
                        raise AmbiguousCanonicalizationError(
                            f"Multiple seating position roles declare duplicate slot identity {slot_0!r}: {[r['id'] for r in raw_list]}"
                        )
                    cnt_0 = int(raw_list[0]["required_count"])
                    cnt_1 = int(raw_list[1]["required_count"])
                    if cnt_0 != 1 or cnt_1 != 1:
                        raise MalformedVLMSpecificationError(
                            f"Disjoint seating position roles must each have required_count=1, got counts ({cnt_0}, {cnt_1})"
                        )
                    pol_0 = raw_list[0]["binding_policy"]
                    pol_1 = raw_list[1]["binding_policy"]
                    if pol_0 != "DISTINCT" or pol_1 != "DISTINCT":
                        raise MalformedVLMSpecificationError(
                            f"Disjoint seating position roles must have binding_policy DISTINCT, got ({pol_0}, {pol_1})"
                        )
                    cnt = 2
                    pol = "DISTINCT"
                    for r in raw_list:
                        r_slot = _extract_disjoint_slot_identity(r.get("function", ""), r.get("description", ""))
                        concept_accounting["roles"][r["id"]] = {
                            "canonical_role": "SEATING_POSITION",
                            "entity_kind": "FIXED_TARGET",
                            "raw_count": 1,
                            "canonical_count": 2,
                            "binding_policy": "DISTINCT",
                            "unary_predicates": [],
                            "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                            "candidate_categories_used_for_role_identity": False,
                            "status": "MERGED_BY_EXPLICIT_RULE",
                            "reason": f"Disjoint personal seating position role ({r_slot}) composed into canonical role",
                        }
                else:
                    raise AmbiguousCanonicalizationError(
                        f"Unsupported number of seating position roles: {len(raw_list)} (expected 1 aggregate or 2 disjoint slots)"
                    )

                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for r in raw_list
                    for cat in r.get("candidate_categories", [])
                    if str(cat).strip()
                ))
                run_local_cats = [_phrase(c).replace(" ", "_") for c in cand_cats if c]
                accepted_cats = list(dict.fromkeys(list(LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["SEATING_POSITION"]) + run_local_cats))
                hints = list(dict.fromkeys(
                    candidate["label"]
                    for r in raw_list
                    for candidate in r.get("visible_candidates", [])
                    if candidate.get("label")
                ))

                normalized_records.append({
                    "role_id": "seating_position",
                    "raw_vlm_role_ids": [r["id"] for r in raw_list],
                    "entity_kind": "FIXED_TARGET",
                    "binding_policy": pol,
                    "function": "SEATING_POSITION",
                    "raw_function": " / ".join(dict.fromkeys(r.get("function", "") for r in raw_list)),
                    "vlm_required_count": cnt,
                    "description": " ".join(dict.fromkeys(r.get("description", "") for r in raw_list)),
                    "candidate_categories": cand_cats,
                    "raw_candidate_categories": cand_cats,
                    "canonical_graph_category": LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["SEATING_POSITION"],
                    "accepted_categories": accepted_cats,
                    "required_properties": [],
                    "visible_candidates": [
                        candidate
                        for r in raw_list
                        for candidate in r.get("visible_candidates", [])
                    ],
                    "semantic_hints": hints,
                    "source": "FM",
                    "provenance": "qwen_vlm_normalized_by_generic_ontology",
                    "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                    "normalization_status": "COMPLETE",
                })

            # 6. Synthesis: SEATING_PAIR
            if "SEATING_PAIR" in classified_roles:
                items = classified_roles["SEATING_PAIR"]
                raw_list = [it["raw"] for it in items]
                if len(raw_list) > 1:
                    raise AmbiguousCanonicalizationError(
                        f"Multiple raw roles mapping to SEATING_PAIR: {[r['id'] for r in raw_list]}"
                    )
                r = raw_list[0]
                cnt = int(r["required_count"])
                pol = r["binding_policy"]
                concept_accounting["roles"][r["id"]] = {
                    "canonical_role": "SEATING_PAIR",
                    "entity_kind": "FIXED_TARGET",
                    "raw_count": cnt,
                    "canonical_count": cnt,
                    "binding_policy": pol,
                    "unary_predicates": [],
                    "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
                    "candidate_categories_used_for_role_identity": False,
                    "status": "PRESERVED",
                }
                cand_cats = list(dict.fromkeys(
                    cat.strip()
                    for cat in r.get("candidate_categories", [])
                    if str(cat).strip()
                ))
                run_local_cats = [_phrase(c).replace(" ", "_") for c in cand_cats if c]
                accepted_cats = list(dict.fromkeys(list(LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["SEATING_PAIR"]) + run_local_cats))
                hints = list(dict.fromkeys(
                    candidate["label"]
                    for candidate in r.get("visible_candidates", [])
                    if candidate.get("label")
                ))

                normalized_records.append({
                    "role_id": "seating_pair",
                    "raw_vlm_role_ids": [r["id"]],
                    "entity_kind": "FIXED_TARGET",
                    "binding_policy": pol,
                    "function": "SEATING_PAIR",
                    "raw_function": r.get("function", ""),
                    "vlm_required_count": cnt,
                    "description": r.get("description", ""),
                    "candidate_categories": cand_cats,
                    "raw_candidate_categories": cand_cats,
                    "canonical_graph_category": LIVING_TASK_ANCHOR_CANONICAL_CATEGORIES["SEATING_PAIR"],
                    "accepted_categories": accepted_cats,
                    "required_properties": [],
                    "visible_candidates": list(r.get("visible_candidates", [])),
                    "semantic_hints": hints,
                    "source": "FM",
                    "provenance": "qwen_vlm_normalized_by_generic_ontology",
                    "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                    "normalization_status": "COMPLETE",
                })

            # Canonicalize interaction groups losslessly into OperationGroup objects
            canonical_operation_groups: list[dict[str, Any]] = []
            seen_group_canonical_ids: set[str] = set()
            for grp in self.raw_decomposition.get("interaction_groups", []):
                gid = grp["id"]
                t_role = grp["tool_role"]
                tgt_role = grp["target_role"]
                if t_role not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(f"Interaction group tool role {t_role!r} not declared in roles")
                if tgt_role not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(f"Interaction group target role {tgt_role!r} not declared in roles")

                t_canon = raw_id_to_canon[t_role]
                tgt_canon = raw_id_to_canon[tgt_role]

                ctx_role = grp.get("context_role")
                if not ctx_role or ctx_role not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(f"Interaction group context role {ctx_role!r} not declared in roles")
                ctx_canon = raw_id_to_canon[ctx_role]

                fn_canon = map_living_room_operation_group_function(grp["function"])
                if fn_canon is None:
                    raise UnmappedFunctionalConceptError(
                        f"Interaction group function {grp.get('function')!r} cannot be mapped to any Living Room operation group"
                    )
                if fn_canon != "personal_support_group" or (t_canon, tgt_canon, ctx_canon) != (
                    "PERSONAL_CUP_SAUCER_REGION", "CUP_SAUCER_SET", "SEATING_POSITION"
                ):
                    raise MalformedVLMSpecificationError(
                        f"Group endpoints ({t_canon}, {tgt_canon}, {ctx_canon}) contradict function {grp.get('function')!r}"
                    )
                if fn_canon in seen_group_canonical_ids:
                    raise AmbiguousCanonicalizationError(f"Duplicate interaction group mapping to canonical group {fn_canon!r}")
                seen_group_canonical_ids.add(fn_canon)

                req_count = int(grp["required_target_count"])
                target_rec = next((r for r in normalized_records if r["function"] == "CUP_SAUCER_SET"), None)
                if target_rec is None or req_count != target_rec["vlm_required_count"]:
                    raise MalformedVLMSpecificationError(
                        f"Group required_target_count={req_count} does not match target role count={target_rec['vlm_required_count'] if target_rec else None}"
                    )

                usage_policy = grp["usage_policy"]
                if usage_policy != "DEDICATED_PER_TARGET":
                    raise MalformedVLMSpecificationError(
                        f"Living Room group requires usage_policy DEDICATED_PER_TARGET, got {usage_policy!r}"
                    )

                req_rels: list[str] = []
                for r in grp["required_relations"]:
                    s, p, o, _ = canonicalize_living_room_relation(r, t_canon, tgt_canon)
                    if p != "FITS_SET_ON":
                        raise MalformedVLMSpecificationError(f"Group required_relations must be FITS_SET_ON, got {p}")
                    req_rels.append(p)
                if not req_rels:
                    raise MalformedVLMSpecificationError("Group required_relations cannot be empty")

                ctx_rels: list[str] = []
                for r in grp.get("context_relations", []):
                    s, p, o, _ = canonicalize_living_room_relation(r, t_canon, ctx_canon)
                    if p != "NEAR_SEAT":
                        raise MalformedVLMSpecificationError(f"Group context_relations must be NEAR_SEAT, got {p}")
                    ctx_rels.append(p)
                if not ctx_rels:
                    raise MalformedVLMSpecificationError("Group context_relations cannot be empty")

                canonical_operation_groups.append({
                    "id": fn_canon,
                    "function": "SUPPORT_DRINKWARE",
                    "tool_role": t_canon,
                    "target_role": tgt_canon,
                    "required_target_count": req_count,
                    "usage_policy": usage_policy,
                    "required_relations": tuple(req_rels),
                    "context_role": ctx_canon,
                    "context_relations": tuple(ctx_rels),
                    "distinct_within_group": True,
                    "same_tool_must_cover_all_targets": False,
                })
                concept_accounting["operation_groups"].append({
                    "raw_group_id": str(gid),
                    "canonical_group_id": fn_canon,
                    "raw_function": str(grp["function"]),
                    "canonical_function": "SUPPORT_DRINKWARE",
                    "tool_role": t_canon,
                    "target_role": tgt_canon,
                    "context_role": ctx_canon,
                    "required_target_count": req_count,
                    "usage_policy": usage_policy,
                    "required_relations": list(req_rels),
                    "context_relations": list(ctx_rels),
                    "function_mapping_status": "PRESERVED",
                    "status": "PRESERVED",
                })

            # Canonicalize relations losslessly and distribute structurally
            canonical_relations: list[dict[str, Any]] = []
            for rel_item in self.raw_decomposition.get("functional_relations", []):
                s = rel_item["subject_role"]
                r = rel_item["relation"]
                o = rel_item["object_role"]
                if s not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(
                        f"VLM relation subject role {s!r} not declared in living room roles"
                    )
                if o not in raw_id_to_canon:
                    raise MalformedVLMSpecificationError(
                        f"VLM relation object role {o!r} not declared in living room roles"
                    )
                s_canon = raw_id_to_canon[s]
                o_canon = raw_id_to_canon[o]
                canon_s, canon_p, canon_o, dir_status = canonicalize_living_room_relation(
                    r, s_canon, o_canon
                )

                if (canon_s, canon_p, canon_o) == ("PERSONAL_CUP_SAUCER_REGION", "FITS_SET_ON", "CUP_SAUCER_SET") and canonical_operation_groups:
                    dest = "OPERATION_REQUIRED_RELATION"
                elif (canon_s, canon_p, canon_o) == ("PERSONAL_CUP_SAUCER_REGION", "NEAR_SEAT", "SEATING_POSITION") and canonical_operation_groups:
                    dest = "OPERATION_CONTEXT_RELATION"
                else:
                    dest = "GRAPH_RELATION"
                    canonical_relations.append({
                        "raw_subject_role_id": str(s),
                        "canonical_subject_role_id": canon_s,
                        "raw_relation_text": str(r),
                        "canonical_predicate": canon_p,
                        "raw_object_role_id": str(o),
                        "canonical_object_role_id": canon_o,
                        "direction_status": dir_status,
                    })

                concept_accounting["relations"].append({
                    "raw_subject_role_id": str(s),
                    "raw_relation_text": str(r),
                    "raw_object_role_id": str(o),
                    "canonical_subject_role_id": canon_s,
                    "canonical_predicate": canon_p,
                    "canonical_object_role_id": canon_o,
                    "direction_status": dir_status,
                    "structural_destination": dest,
                    "status": "PRESERVED",
                })

            canonicalization_trace = {
                "version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                "vlm_canonicalization_version": LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
                "transformation": "DETERMINISTIC_NATURAL_LANGUAGE_CANONICALIZATION",
                "raw_id_to_canonical": dict(raw_id_to_canon),
                "roles": [
                    {
                        "canonical_func": r["function"],
                        "raw_role_ids": r["raw_vlm_role_ids"],
                        "raw_categories": r["raw_candidate_categories"],
                        "canonical_graph_category": r["canonical_graph_category"],
                        "accepted_categories": r["accepted_categories"],
                        "required_properties": r["required_properties"],
                    }
                    for r in normalized_records
                ],
                "relations": list(canonical_relations),
                "operation_groups": list(canonical_operation_groups),
                "concept_accounting": concept_accounting,
            }
            self.canonicalization_trace = canonicalization_trace
            self.normalized_relations = canonical_relations
            self.normalized_operation_groups = canonical_operation_groups
            self.vlm_derived_role_vocabulary = tuple(dict.fromkeys(vlm_role_vocab))
            self.task_explicit_context_vocabulary = ("armchair", "chair", "sofa")
            self.normalization_issues = []
            self.ready_for_grounding = True
            self.normalized_task = None

        self.normalized_requirements = normalized_records

        vocabulary_entries = []
        aliases = self._vocabulary_aliases()
        ordered_categories = sorted(
            aliases,
            key=lambda category: (
                category_rank.get(category, 10_000),
                list(aliases).index(category),
            ),
        )
        for rank, canonical in enumerate(ordered_categories, 1):
            vocabulary_entries.append(
                {
                    "canonical_label": canonical,
                    "aliases": list(aliases[canonical]),
                    "rank": rank,
                    "source": (
                        "VLM_MATCHED_ROLE_CATEGORY"
                        if canonical in category_rank
                        else "REVIEWED_NEGATIVE_OR_CONTEXT_CATEGORY"
                    ),
                }
            )
        self.ranked_detector_vocabulary = vocabulary_entries
        if require_reviewed_contract and not self.ready_for_grounding:
            raise ValueError(
                "VLM output is not ready for grounding: "
                + "; ".join(self.normalization_issues)
            )
        return self.result()

    def result(self) -> dict[str, Any]:
        if self.raw_decomposition is None:
            raise RuntimeError("generate() must be called before result()")
        raw_vlm = deepcopy(
            getattr(self.fm_adapter, "last_raw_response", None)
            or self.fm_adapter.last_raw_requirement_response
        )
        validated_vlm = deepcopy(self.raw_decomposition)
        canon_trace = deepcopy(getattr(self, "canonicalization_trace", {}))
        return {
            "schema_version": 1,
            "environment": self.environment,
            "scope": "VLM_REQUIREMENT_DECOMPOSITION_ONLY",
            "task_instruction": self.task_instruction,
            "initial_observation_images": self.fm_adapter.last_observation_images,
            "raw_vlm_response": raw_vlm,
            "validated_vlm_specification": validated_vlm,
            "canonicalization_trace": canon_trace,
            "raw_vlm_decomposition": self.raw_decomposition,  # legacy compatibility alias
            "raw_vlm_requirement_response": deepcopy(
                self.fm_adapter.last_raw_requirement_response
            ),
            "raw_vlm_inspection_response": deepcopy(
                self.fm_adapter.last_raw_inspection_response
            ),
            "normalized_requirements": self.normalized_requirements,
            "normalized_relations": self.normalized_relations,
            "normalized_operation_groups": getattr(self, "normalized_operation_groups", []),
            "normalized_task_contract": self.normalized_task,
            "ready_for_grounding": self.ready_for_grounding,
            "reviewed_ontology_audit": {
                "status": "PASS" if self.ready_for_grounding else "REVIEW_REQUIRED",
                "issues": list(self.normalization_issues),
                "note": (
                    "The reviewed ontology was used only after the VLM response; "
                    "it was not included in the model prompt."
                ),
            },
            "ranked_detector_vocabulary": self.ranked_detector_vocabulary,
            "vlm_inspection_policy": deepcopy(self.inspection_policy),
            "fm_calls": self.fm_adapter.metrics.total_calls,
            "observation_search_started": False,
            "semantic_grounding_started": False,
            "allocation_started": False,
            "geometry_verification_started": False,
            "planning_started": False,
            "execution_started": False,
        }
