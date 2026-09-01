"""Validate and compile raw natural-language Kitchen functional specification.

Performs deterministic, semantic-preserving, fail-closed canonicalization of
natural-language properties, qualitative relations, and visually proposed
inspectable regions into executable G_F and task contracts.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    VLMSpecificationError,
)
from .workshop_phase1.fm_adapter import validate_kitchen_functional_specification
from .task_witness import load_task_requirements

KITCHEN_VLM_CANONICALIZATION_VERSION = "phase3_p3e_1_v1"
VLM_CANONICALIZATION_VERSION = KITCHEN_VLM_CANONICALIZATION_VERSION


# Stable Kitchen Canonical Role Registry
KITCHEN_ROLE_REGISTRY: dict[str, tuple[str, ...]] = {
    "coffee_container": (
        "contain coffee", "hold coffee", "coffee container", "coffee receptacle",
        "coffee vessel", "coffee cup", "coffee mug", "coffee serving",
        "contain an individual serving of coffee", "contain one coffee serving",
        "hold coffee serving", "receptacle for coffee", "vessel for coffee",
        "individual serving of coffee", "cup for coffee", "mug for coffee",
        "hold an individual serving of coffee", "receptacle to hold coffee",
        "contain coffee serving",
    ),
    "soup_container": (
        "contain soup", "hold soup", "soup container", "soup receptacle",
        "soup vessel", "soup bowl", "soup serving",
        "contain an individual serving of soup", "contain one soup serving",
        "hold soup serving", "receptacle for soup", "vessel for soup",
        "individual serving of soup", "bowl for soup",
        "hold an individual serving of soup", "receptacle to hold soup",
        "contain soup serving",
    ),
    "coffee_stirrer": (
        "stir coffee", "mix coffee", "coffee stirrer", "coffee stirring",
        "stir", "mix", "stir beverage", "agitate coffee", "stirring utensil",
        "stirring implement", "stir both coffees", "coffee implement",
        "mixing implement", "stir drink", "mix beverage", "stir beverage in cups",
    ),
    "soup_eating_utensil": (
        "serve soup", "soup utensil", "eat soup", "consume soup", "soup spoon",
        "soup implement", "provide utensil for soup", "provide a suitable eating utensil for each soup bowl",
        "eating utensil for soup", "serve with soup", "soup utensil provision",
        "eating utensil", "soup eating utensil", "soup eating implement",
        "provide utensil", "provide eating utensil", "provide eating utensil for each soup bowl",
        "eating utensil for soup bowl", "provide a suitable utensil for each soup bowl",
    ),
    "coffee_source": (
        "provide coffee", "coffee source", "coffee material", "coffee ingredient",
        "coffee supply", "coffee jar", "source of coffee", "coffee grounds",
        "provide coffee material", "coffee beans", "instant coffee",
        "instant coffee jar", "package of coffee", "coffee container jar",
    ),
    "water_source": (
        "provide water", "water source", "pour water", "hot water", "kettle",
        "water supply", "source of water", "provide water for coffee",
        "water container", "water pitcher", "water jug", "compact kettle",
    ),
}

# Strictly physical / capability language only (no object nouns like cup/spoon)
UNARY_PROPERTY_ALIASES: dict[str, tuple[str, ...]] = {
    "OPEN_CAVITY": (
        "open cavity", "open_cavity", "cavity", "container", "hollow",
        "deep container", "liquid container", "holds liquid",
        "hold liquid", "contain liquid", "capable of containing", "receptacle",
        "hollow receptacle", "capable of holding liquid", "capable of containing liquid",
        "open top cavity", "open top", "open container",
    ),
    "ELONGATED_OBJECT": (
        "elongated", "elongated_object", "elongated object", "long thin",
        "long", "slender", "rod", "shank", "stick", "extended shape",
        "long and thin", "elongated shape", "slender shape", "long slender",
    ),
}

BINARY_RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "INSERTABLE_IN": (
        "insertable in", "insertable_in", "insertable", "fits inside", "fit inside",
        "must fit inside", "can enter", "enter opening", "fits into", "fit into",
        "goes inside", "go inside", "placed inside", "place inside",
        "inserted into", "insert into", "fits through opening", "fit through opening",
        "insert into container", "fits in", "fit in", "goes in", "goes into",
    ),
    "REACHES_BOTTOM": (
        "reaches bottom", "reaches_bottom", "reaches the bottom", "reach the bottom",
        "reach bottom", "long enough to reach bottom", "access bottom", "extends to bottom",
        "extend to bottom", "contacts bottom", "contact bottom", "touches bottom", "touch bottom",
        "reaches container bottom", "reach container bottom", "reaches bottom of container",
        "reach bottom of container",
    ),
}

# Reviewed semantic aliases for Kitchen interaction group functions
KITCHEN_INTERACTION_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "coffee_stirring": (
        "coffee stirring", "coffee_stirring", "stir coffee", "mix coffee",
        "stir beverage", "stir drinks", "stir beverage in cups", "mix beverage",
        "stir", "mix", "agitate coffee", "stirring", "mixing", "beverage stirring",
        "coffee preparation", "prepare coffee",
    ),
    "soup_serving": (
        "soup serving", "soup_serving", "serve soup", "provide utensil",
        "provide eating utensil", "provide utensil for soup",
        "provide a suitable eating utensil for each soup bowl",
        "provide a suitable utensil for each soup bowl",
        "provide eating utensil for each soup bowl",
        "soup utensil provision", "eating utensil", "soup eating utensil",
        "equip soup", "serve with soup", "soup utensil", "eating utensil provision",
        "soup consumption",
    ),
}

KITCHEN_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "D1": (
        "upper kitchen drawer", "upper drawer", "top kitchen drawer", "top drawer",
        "drawer above lower drawer", "first drawer", "topmost drawer", "upper storage drawer",
        "top drawer below counter",
    ),
    "D2": (
        "lower kitchen drawer", "lower drawer", "bottom kitchen drawer", "bottom drawer",
        "second drawer", "drawer below upper drawer", "bottom storage drawer",
        "lower drawer below counter",
    ),
    "C2": (
        "upper wall cupboard", "upper cupboard", "wall cupboard", "upper cabinet",
        "top cabinet", "wall cabinet", "overhead cupboard", "overhead cabinet",
        "cupboard above counter", "cabinet above counter", "upper storage",
        "wall mounted cupboard",
    ),
    "B1": (
        "countertop storage box", "countertop box", "storage box", "wooden box",
        "counter box", "box on counter", "storage bin on counter", "tabletop box",
    ),
    "C1": (
        "lower kitchen cupboard", "lower cupboard", "bottom cupboard", "base cupboard",
        "lower cabinet", "base cabinet", "under counter cupboard", "cupboard below counter",
        "cabinet below counter", "lower storage",
    ),
}

KITCHEN_OBSERVABLE_REGIONS = {
    "D1": "upper kitchen drawer",
    "D2": "lower kitchen drawer",
    "C2": "upper wall cupboard",
    "B1": "countertop storage box",
    "C1": "lower kitchen cupboard",
}


def _phrase(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    words = text.split()
    p_words = phrase.split()
    n, k = len(words), len(p_words)
    return any(words[i:i + k] == p_words for i in range(n - k + 1))


def map_kitchen_role_function(raw: dict[str, Any] | str) -> str | None:
    """Map natural language role to unique canonical Kitchen role using function and description only.

    Candidate categories are strictly excluded from role semantic authority and are
    reserved for detector vocabulary / semantic preferences.
    """
    if isinstance(raw, dict):
        text = f"{raw.get('function', '')} {raw.get('description', '')}"
    else:
        text = str(raw)
    norm = _phrase(text)
    if not norm:
        return None
    words = set(norm.split())

    has_coffee = (
        "coffee" in norm
        or any(w in words for w in ("coffee", "coffee_cup", "coffee_mug", "coffee_spoon", "instant_coffee"))
    )
    has_soup = (
        "soup" in norm
        or any(w in words for w in ("soup", "soup_bowl", "soup_spoon", "soup_utensil", "soup_fork"))
    )

    has_stir = any(
        w in words or _contains_phrase(norm, w)
        for w in ("stir", "mix", "agitate", "stirrer", "stirring", "stir beverage", "mixing implement")
    )
    has_utensil = any(
        w in words or _contains_phrase(norm, w)
        for w in (
            "utensil", "eat", "consume", "eating", "tablespoon", "fork",
            "soup utensil", "soup spoon", "eating utensil", "serve with soup",
            "provide utensil",
        )
    )
    has_spoon = any(
        w in words or _contains_phrase(norm, w)
        for w in ("spoon", "teaspoon", "soup spoon", "coffee spoon")
    )

    has_cup = any(
        w in words or _contains_phrase(norm, w)
        for w in ("cup", "mug", "tumbler", "glass", "beaker", "coffee cup", "coffee mug")
    )
    has_bowl = any(
        w in words or _contains_phrase(norm, w)
        for w in ("bowl", "dish", "soup bowl", "deep bowl", "shallow bowl")
    )
    has_contain = (
        has_cup or has_bowl
        or any(
            w in words or _contains_phrase(norm, w)
            for w in ("contain", "hold", "receptacle", "vessel", "serving", "container", "individual serving")
        )
    )

    has_source = any(
        w in words or _contains_phrase(norm, w)
        for w in ("source", "provide", "supply", "pour", "ingredient", "material", "grounds", "beans", "supply of", "source of")
    )
    has_water = any(
        w in words or _contains_phrase(norm, w)
        for w in ("water", "kettle", "hot water", "pour water", "pitcher", "water container", "water pitcher", "water jug", "water supply", "source of water")
    )
    has_jar = any(
        w in words or _contains_phrase(norm, w)
        for w in ("jar", "coffee jar", "can", "box", "package", "instant coffee jar")
    )

    # 1. Water source
    if has_water and not has_coffee and not has_soup:
        return "water_source"
    if has_water and (has_source or "kettle" in norm or "pitcher" in norm):
        return "water_source"

    # 2. Coffee source
    if (
        (has_coffee and has_source and not has_contain and not has_stir)
        or (has_coffee and has_jar)
        or ("source of coffee" in norm)
        or ("coffee grounds" in norm)
        or ("coffee jar" in norm)
    ):
        return "coffee_source"

    # 3. Stirrer vs Eating utensil
    if has_stir or (has_spoon and has_coffee and not has_contain and not has_cup and not has_soup):
        if not has_soup:
            return "coffee_stirrer"
    if (has_soup and (has_utensil or has_spoon)) and not has_bowl and not (has_contain and not has_spoon and not has_utensil):
        return "soup_eating_utensil"

    # 4. Containers
    if (has_coffee or has_cup) and has_contain and not has_stir and not has_source and not has_spoon:
        return "coffee_container"
    if (has_soup or has_bowl) and has_contain and not has_utensil and not has_source and not has_spoon:
        return "soup_container"

    # 5. Registry dictionary match (exact alias or alias contained as a phrase in norm, forward-only)
    matches = set()
    for role_name, aliases in KITCHEN_ROLE_REGISTRY.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(role_name)
                break
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        if has_coffee and "coffee_container" in matches and not has_stir and not has_source:
            return "coffee_container"
        if has_soup and "soup_container" in matches and not has_utensil and not has_source:
            return "soup_container"
        if has_coffee and "coffee_stirrer" in matches and has_stir:
            return "coffee_stirrer"
        if has_soup and "soup_eating_utensil" in matches and has_utensil:
            return "soup_eating_utensil"
        raise AmbiguousCanonicalizationError(f"Ambiguous kitchen role function {text!r} matches multiple roles: {sorted(matches)}")
    return None


def map_unary_property(text: str) -> str | None:
    """Map natural language property to unique canonical predicate or None.

    Requires exact reviewed alias or reviewed alias occurring as a full phrase
    inside the raw text. Reverse short-fragment containment is strictly disallowed.
    """
    norm = _phrase(text)
    if not norm:
        return None
    matches = set()
    for pred, aliases in UNARY_PROPERTY_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(pred)
                break
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(f"Ambiguous unary property {text!r} matches multiple checkers: {sorted(matches)}")
    return None


def map_binary_relation(text: str) -> str | None:
    """Map natural language relation to unique canonical relation or None.

    Requires exact reviewed alias or reviewed alias occurring as a full phrase
    inside the raw text. Reverse short-fragment containment is strictly disallowed.
    """
    norm = _phrase(text)
    if not norm:
        return None
    matches = set()
    for pred, aliases in BINARY_RELATION_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(pred)
                break
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(f"Ambiguous binary relation {text!r} matches multiple relations: {sorted(matches)}")
    return None


def map_kitchen_interaction_group_function(text: str) -> str | None:
    """Map natural language operation group function to unique canonical group ID or None.

    Requires exact reviewed alias or reviewed alias occurring as a full phrase
    inside the raw text. Reverse short-fragment containment is strictly disallowed.
    """
    norm = _phrase(text)
    if not norm:
        return None
    matches = set()
    for group_id, aliases in KITCHEN_INTERACTION_GROUP_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(group_id)
                break
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(
            f"Ambiguous kitchen interaction group function {text!r} matches multiple groups: {sorted(matches)}"
        )
    return None


def resolve_kitchen_region_proposal(proposal: dict[str, Any] | str) -> str | None:
    """Resolve visual region proposal to canonical region ID using ONLY label and visual_description.

    VLM-local IDs (e.g. 'c2', 'region_1') are NOT semantic evidence and must be ignored.
    """
    if isinstance(proposal, str):
        text = proposal.strip().lower()
    else:
        text = f"{proposal.get('label', '')} {proposal.get('visual_description', '')}".strip().lower()
    norm = _phrase(text)
    if not norm:
        return None
    matches = set()
    for reg_id, aliases in KITCHEN_REGION_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm) or a_norm in norm:
                matches.add(reg_id)
                break
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise AmbiguousCanonicalizationError(f"Ambiguous kitchen region proposal {proposal!r} matches multiple regions: {sorted(matches)}")
    return None


def _identifier(value: Any, context: str) -> str:
    val_str = str(value).strip().lower().replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", val_str):
        raise MalformedVLMSpecificationError(f"{context} must be a valid lower snake_case identifier, got {value!r}")
    return val_str


def compile_vlm_functional_graph(
    document: dict[str, Any],
    *,
    task_instruction: str,
    observable_regions: tuple[str, ...] = tuple(KITCHEN_OBSERVABLE_REGIONS.keys()),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Deterministically compile raw Kitchen specification into canonical G_F and task contract."""
    valid_doc = validate_kitchen_functional_specification(document)
    if valid_doc.get("status") != "SUPPORTED":
        raise VLMSpecificationError(
            f"VLM marked task unsupported: {valid_doc.get('unsupported_reason')}",
            category="UNSUPPORTED_TASK",
        )

    roles_raw = valid_doc.get("functional_roles", [])
    if not isinstance(roles_raw, list) or not roles_raw:
        raise MalformedVLMSpecificationError("VLM functional specification has no roles")

    raw_role_ids = [_identifier(row.get("id"), "role id") for row in roles_raw]
    if len(set(raw_role_ids)) != len(raw_role_ids):
        raise MalformedVLMSpecificationError("VLM functional specification contains duplicate role IDs")

    roles: dict[str, Any] = {}
    vocabulary: dict[str, list[str]] = {}
    canonical_predicates: dict[str, list[Any]] = {"unary": [], "binary": []}
    raw_role_to_canonical: dict[str, str] = {}
    concept_accounting: dict[str, Any] = {
        "roles": {},
        "properties": [],
        "relations": [],
        "operation_groups": [],
    }

    for order, row in enumerate(roles_raw):
        raw_role_id = _identifier(row["id"], "role id")
        raw_entity_kind = str(row.get("entity_kind", "OBJECT"))
        if raw_entity_kind != "OBJECT":
            raise MalformedVLMSpecificationError(
                f"Kitchen functional role {raw_role_id!r} must have entity_kind 'OBJECT', got {raw_entity_kind!r}"
            )

        if "binding_policy" not in row:
            raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} is missing binding_policy")
        binding_policy = str(row["binding_policy"])
        if binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
            raise MalformedVLMSpecificationError(
                f"Unknown binding_policy {binding_policy!r} for role {raw_role_id!r}"
            )

        if "required_count" not in row:
            raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} is missing required_count")
        try:
            required_count = int(row["required_count"])
        except (ValueError, TypeError):
            raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} has non-integer required_count: {row.get('required_count')!r}")
        if required_count < 1:
            raise MalformedVLMSpecificationError(
                f"Role {raw_role_id!r} has invalid required_count {required_count} (must be >= 1)"
            )

        canon_role_name = map_kitchen_role_function(row)
        if canon_role_name is None:
            raise UnmappedFunctionalConceptError(
                f"Raw functional role {raw_role_id!r} (function={row.get('function')!r}, "
                f"description={row.get('description')!r}, categories={row.get('candidate_categories')!r}) "
                f"cannot be mapped to any canonical Kitchen role"
            )

        if canon_role_name in roles:
            existing_raw_id = roles[canon_role_name]["raw_vlm_role_id"]
            existing_count = roles[canon_role_name]["count"]
            existing_fn = roles[canon_role_name]["vlm_function"]
            raise AmbiguousCanonicalizationError(
                f"Multiple distinct raw roles ({existing_raw_id!r} and {raw_role_id!r}) map to the same "
                f"canonical role {canon_role_name!r}. Raw counts: {existing_count} vs {required_count}. "
                f"Raw functions: {existing_fn!r} vs {row.get('function')!r}. "
                f"Duplicate canonical role collision fails closed without heuristics."
            )

        raw_role_to_canonical[raw_role_id] = canon_role_name

        unary = []
        raw_props = row.get("required_properties", [])
        if not isinstance(raw_props, list):
            raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} required_properties must be a list")

        seen_predicates_on_role: set[str] = set()
        for prop in raw_props:
            if not isinstance(prop, str):
                raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} required_property must be a string, got {type(prop)}")
            mapped = map_unary_property(prop)
            if mapped is None:
                raise UnmappedFunctionalConceptError(
                    f"Required property {prop!r} on role {raw_role_id!r} (canonical {canon_role_name!r}) "
                    f"cannot be mapped to any active physical unary property (available: {list(UNARY_PROPERTY_ALIASES.keys())})"
                )
            if mapped not in seen_predicates_on_role:
                seen_predicates_on_role.add(mapped)
                unary.append({"predicate": mapped, "expected": True})
                canonical_predicates["unary"].append({"role": canon_role_name, "predicate": mapped})
                concept_accounting["properties"].append({
                    "raw_role_id": raw_role_id,
                    "raw_phrase": prop,
                    "canonical_predicate": mapped,
                    "status": "PRESERVED",
                })
            else:
                concept_accounting["properties"].append({
                    "raw_role_id": raw_role_id,
                    "raw_phrase": prop,
                    "canonical_predicate": mapped,
                    "status": "MERGED_BY_EXPLICIT_RULE",
                    "reason": f"Duplicate alias for predicate {mapped} on same role",
                })

        cand_cats = row.get("candidate_categories", [])
        if not isinstance(cand_cats, list) or not cand_cats:
            raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} must specify non-empty candidate_categories")
        preferences = []
        seen_canon = set()
        for cat in cand_cats:
            if isinstance(cat, dict):
                canon_label = _identifier(cat.get("canonical_label", "").lower().replace(" ", "_"), f"canonical category for {raw_role_id}")
                phrases = [canon_label.replace("_", " "), *map(lambda s: str(s).lower(), cat.get("detector_phrases", []))]
            else:
                c_str = str(cat).lower()
                canon_label = _identifier(c_str.replace(" ", "_"), f"canonical category for {raw_role_id}")
                phrases = [canon_label.replace("_", " "), c_str]
            phrases = list(dict.fromkeys([p for p in phrases if p]))
            if canon_label in seen_canon:
                continue
            seen_canon.add(canon_label)
            vocabulary.setdefault(canon_label, [])
            vocabulary[canon_label] = list(dict.fromkeys([*vocabulary[canon_label], *phrases]))
            preferences.append({
                "rank": len(preferences) + 1,
                "canonical_label": canon_label,
                "detector_aliases": phrases,
            })

        raw_card = row.get("binding_cardinality")
        direct_min = row.get("min_count")
        direct_max = row.get("max_count")
        direct_pref = row.get("preference")

        if raw_card is not None:
            if not isinstance(raw_card, dict):
                raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} binding_cardinality must be a dict")
            allowed_card_keys = {
                "minimum_distinct_physical_objects",
                "maximum_distinct_physical_objects",
                "preferred",
                "preference",
                "mode",
            }
            unexpected_keys = set(raw_card) - allowed_card_keys
            if unexpected_keys:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality has unexpected keys: {sorted(unexpected_keys)}"
                )
            if "minimum_distinct_physical_objects" not in raw_card or "maximum_distinct_physical_objects" not in raw_card:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality missing required minimum/maximum fields"
                )
            c_min = raw_card["minimum_distinct_physical_objects"]
            c_max = raw_card["maximum_distinct_physical_objects"]
            if isinstance(c_min, bool) or not isinstance(c_min, int) or c_min < 1:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality minimum_distinct_physical_objects must be integer >= 1"
                )
            if isinstance(c_max, bool) or not isinstance(c_max, int) or c_max < 1:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality maximum_distinct_physical_objects must be integer >= 1"
                )
            if c_min > c_max:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality minimum ({c_min}) > maximum ({c_max})"
                )
            if c_max > required_count:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality maximum ({c_max}) > required_count ({required_count})"
                )
            if "preferred" in raw_card and "preference" in raw_card:
                if raw_card["preferred"] != raw_card["preference"]:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} binding_cardinality conflicting preferred ({raw_card['preferred']!r}) and preference ({raw_card['preference']!r})"
                    )
            c_pref = raw_card.get("preferred") if "preferred" in raw_card else raw_card.get("preference")
            if c_pref is not None and c_pref not in {"minimize_distinct", "maximize_distinct", "deterministic_rank"}:
                raise MalformedVLMSpecificationError(
                    f"Role {raw_role_id!r} binding_cardinality invalid preference: {c_pref!r}"
                )
            if binding_policy == "DISTINCT":
                if c_min != required_count or c_max != required_count:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} has DISTINCT binding_policy but cardinality range [{c_min}, {c_max}] != required_count {required_count}"
                    )
            if direct_min is not None:
                if isinstance(direct_min, bool) or not isinstance(direct_min, int) or direct_min != c_min:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} min_count ({direct_min}) contradicts binding_cardinality minimum ({c_min})"
                    )
            if direct_max is not None:
                if isinstance(direct_max, bool) or not isinstance(direct_max, int) or direct_max != c_max:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} max_count ({direct_max}) contradicts binding_cardinality maximum ({c_max})"
                    )
            if direct_pref is not None:
                if direct_pref not in {"minimize_distinct", "maximize_distinct", "deterministic_rank"}:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} preference invalid: {direct_pref!r}"
                    )
                if c_pref is not None and direct_pref != c_pref:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} preference ({direct_pref!r}) contradicts binding_cardinality preference ({c_pref!r})"
                    )
            cardinality_data: dict[str, Any] | None = dict(raw_card)
            cardinality_data.setdefault("mode", "assignment_driven")
            min_count = c_min
            max_count = c_max
            preferred = c_pref
        else:
            cardinality_data = None
            if direct_min is not None or direct_max is not None:
                if direct_min is None or direct_max is None:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} must provide both min_count and max_count"
                    )
                if isinstance(direct_min, bool) or not isinstance(direct_min, int) or direct_min < 1:
                    raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} min_count must be integer >= 1")
                if isinstance(direct_max, bool) or not isinstance(direct_max, int) or direct_max < 1:
                    raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} max_count must be integer >= 1")
                if direct_min > direct_max:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} min_count ({direct_min}) > max_count ({direct_max})"
                    )
                if direct_max > required_count:
                    raise MalformedVLMSpecificationError(
                        f"Role {raw_role_id!r} max_count ({direct_max}) > required_count ({required_count})"
                    )
                if binding_policy == "DISTINCT":
                    if direct_min != required_count or direct_max != required_count:
                        raise MalformedVLMSpecificationError(
                            f"Role {raw_role_id!r} has DISTINCT binding_policy but cardinality range [{direct_min}, {direct_max}] != required_count {required_count}"
                        )
                min_count = direct_min
                max_count = direct_max
            else:
                min_count = None
                max_count = None

            if direct_pref is not None:
                if direct_pref not in {"minimize_distinct", "maximize_distinct", "deterministic_rank"}:
                    raise MalformedVLMSpecificationError(f"Role {raw_role_id!r} preference invalid: {direct_pref!r}")
                preferred = direct_pref
            else:
                preferred = None

        if preferred is None and binding_policy == "REUSABLE":
            preferred = "minimize_distinct"

        role_entry: dict[str, Any] = {
            "raw_vlm_role_id": raw_role_id,
            "entity_kind": "OBJECT",
            "count": required_count,
            "min_count": min_count,
            "max_count": max_count,
            "preference": preferred,
            "assignment_order": order,
            "vlm_function": str(row.get("function", "")),
            "vlm_binding_policy": binding_policy,
            "vlm_verification_mode": "SEMANTIC_AND_GEOMETRIC" if unary else "SEMANTIC_ONLY",
            "allow_empty_geometry": not bool(unary),
            "semantic_preferences": preferences,
            "unary_geometry": unary,
            "visible_candidates": list(row.get("visible_candidates", [])),
        }
        if cardinality_data is not None:
            role_entry["binding_cardinality"] = cardinality_data
        roles[canon_role_name] = role_entry
        concept_accounting["roles"][raw_role_id] = {
            "canonical_role": canon_role_name,
            "count": required_count,
            "min_count": min_count,
            "max_count": max_count,
            "preference": preferred,
            "binding_policy": binding_policy,
            "unary_predicates": list(seen_predicates_on_role),
            "role_semantic_source": "FUNCTION_AND_DESCRIPTION",
            "candidate_categories_used_for_role_identity": False,
            "status": "PRESERVED",
        }

    if not roles:
        raise MalformedVLMSpecificationError("No executable kitchen functional roles could be mapped from VLM specification")

    relations = []
    relation_index = set()
    raw_relations = valid_doc.get("functional_relations", [])
    if not isinstance(raw_relations, list):
        raise MalformedVLMSpecificationError("functional_relations must be a list")

    for rel_row in raw_relations:
        if not isinstance(rel_row, dict):
            raise MalformedVLMSpecificationError("Each item in functional_relations must be a dict")
        raw_subj_id = rel_row.get("subject_role")
        raw_obj_id = rel_row.get("object_role")
        if not raw_subj_id or not raw_obj_id:
            raise MalformedVLMSpecificationError(f"functional_relation missing subject_role or object_role: {rel_row}")

        raw_subj = _identifier(raw_subj_id, "subject role")
        raw_obj = _identifier(raw_obj_id, "object role")

        if raw_subj not in raw_role_to_canonical:
            raise MalformedVLMSpecificationError(f"functional_relation specifies undeclared or unmapped subject role: {raw_subj!r}")
        if raw_obj not in raw_role_to_canonical:
            raise MalformedVLMSpecificationError(f"functional_relation specifies undeclared or unmapped object role: {raw_obj!r}")

        subj = raw_role_to_canonical[raw_subj]
        obj = raw_role_to_canonical[raw_obj]
        rel_str = rel_row.get("relation") or rel_row.get("predicate")
        if not rel_str:
            raise MalformedVLMSpecificationError(f"functional_relation missing relation phrase: {rel_row}")

        mapped_rel = map_binary_relation(str(rel_str))
        if mapped_rel is None:
            raise UnmappedFunctionalConceptError(
                f"Functional relation {rel_str!r} between {subj!r} (raw {raw_subj!r}) and {obj!r} (raw {raw_obj!r}) "
                f"cannot be mapped to any active Kitchen binary predicate (available: {list(BINARY_RELATION_ALIASES.keys())})"
            )

        key = (mapped_rel, subj, obj)
        if key not in relation_index:
            relation_index.add(key)
            relations.append({
                "predicate": mapped_rel,
                "subject_role": subj,
                "object_role": obj,
                "expected": True,
            })
            canonical_predicates["binary"].append({
                "subject_role": subj, "predicate": mapped_rel, "object_role": obj,
            })
            concept_accounting["relations"].append({
                "raw_subject": raw_subj,
                "raw_phrase": str(rel_str),
                "raw_object": raw_obj,
                "canonical_subject": subj,
                "canonical_predicate": mapped_rel,
                "canonical_object": obj,
                "status": "PRESERVED",
            })
        else:
            concept_accounting["relations"].append({
                "raw_subject": raw_subj,
                "raw_phrase": str(rel_str),
                "raw_object": raw_obj,
                "canonical_subject": subj,
                "canonical_predicate": mapped_rel,
                "canonical_object": obj,
                "status": "MERGED_BY_EXPLICIT_RULE",
                "reason": f"Duplicate canonical relation {mapped_rel}({subj}, {obj})",
            })

    operations = {}
    raw_group_to_canonical: dict[str, str] = {}
    raw_ops = valid_doc.get("interaction_groups", [])
    if not isinstance(raw_ops, list):
        raise MalformedVLMSpecificationError("interaction_groups must be a list")

    for row in raw_ops:
        if not isinstance(row, dict):
            raise MalformedVLMSpecificationError("Each item in interaction_groups must be a dict")
        raw_group_id = _identifier(row["id"], "operation group id")
        raw_tool_id = row.get("tool_role")
        raw_target_id = row.get("target_role")
        if not raw_tool_id or not raw_target_id:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} missing tool_role or target_role")

        raw_tool_role = _identifier(raw_tool_id, "tool role")
        raw_target_role = _identifier(raw_target_id, "target role")

        if raw_tool_role not in raw_role_to_canonical:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} references undeclared tool_role: {raw_tool_role!r}")
        if raw_target_role not in raw_role_to_canonical:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} references undeclared target_role: {raw_target_role!r}")

        tool_role = raw_role_to_canonical[raw_tool_role]
        target_role = raw_role_to_canonical[raw_target_role]

        raw_fn = row.get("function")
        if not raw_fn or not isinstance(raw_fn, str):
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} is missing non-empty 'function' field")

        fn_group = map_kitchen_interaction_group_function(raw_fn)
        if fn_group is None:
            raise UnmappedFunctionalConceptError(
                f"Operation group {raw_group_id!r} function {raw_fn!r} cannot be mapped to any active Kitchen "
                f"operation group (available: {list(KITCHEN_INTERACTION_GROUP_ALIASES.keys())})"
            )

        if tool_role == "coffee_stirrer" and target_role == "coffee_container":
            endpoint_group = "coffee_stirring"
        elif tool_role == "soup_eating_utensil" and target_role == "soup_container":
            endpoint_group = "soup_serving"
        else:
            raise MalformedVLMSpecificationError(
                f"Unsupported Kitchen operation group tool/target pair: {tool_role!r} (raw {raw_tool_role!r}) -> {target_role!r} (raw {raw_target_role!r})"
            )

        if fn_group != endpoint_group:
            raise MalformedVLMSpecificationError(
                f"Operation group {raw_group_id!r} function semantics {raw_fn!r} (maps to {fn_group!r}) "
                f"contradicts tool/target endpoint pair ({tool_role!r} -> {target_role!r}, maps to {endpoint_group!r})"
            )

        canon_group_id = endpoint_group

        if canon_group_id in operations:
            existing_raw_gid = operations[canon_group_id]["raw_vlm_group_id"]
            raise AmbiguousCanonicalizationError(
                f"Multiple raw operation groups ({existing_raw_gid!r} and {raw_group_id!r}) map to the same "
                f"canonical operation group {canon_group_id!r}. Duplicate operation group collision fails closed."
            )

        raw_group_to_canonical[raw_group_id] = canon_group_id

        if "required_target_count" not in row:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} missing required_target_count")
        try:
            req_target_count = int(row["required_target_count"])
        except (ValueError, TypeError):
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} has non-integer required_target_count: {row.get('required_target_count')!r}")
        if req_target_count < 1:
            raise MalformedVLMSpecificationError(
                f"Operation group {raw_group_id!r} has invalid required_target_count: {req_target_count} (must be >= 1)"
            )

        target_role_count = int(roles[target_role]["count"])
        if req_target_count > target_role_count:
            raise MalformedVLMSpecificationError(
                f"Operation group {raw_group_id!r} has required_count {target_role_count}, but group requires {req_target_count}"
            )

        req_rels = row.get("required_relations", [])
        if not isinstance(req_rels, list) or not req_rels:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} must specify non-empty required_relations")

        mapped_op_rels = []
        for rel_item in req_rels:
            if not isinstance(rel_item, str):
                raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} required_relation must be a string, got {type(rel_item)}")
            mapped_op_rel = map_binary_relation(rel_item)
            if mapped_op_rel is None:
                raise UnmappedFunctionalConceptError(
                    f"Operation group {raw_group_id!r} required relation {rel_item!r} cannot be mapped "
                    f"to any active Kitchen predicate (available: {list(BINARY_RELATION_ALIASES.keys())})"
                )
            if mapped_op_rel not in mapped_op_rels:
                mapped_op_rels.append(mapped_op_rel)
            key = (mapped_op_rel, tool_role, target_role)
            if key not in relation_index:
                relation_index.add(key)
                relations.append({
                    "predicate": mapped_op_rel,
                    "subject_role": tool_role,
                    "object_role": target_role,
                    "expected": True,
                })
                canonical_predicates["binary"].append({
                    "subject_role": tool_role, "predicate": mapped_op_rel, "object_role": target_role
                })

        if "usage_policy" not in row:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} missing usage_policy")
        policy = str(row["usage_policy"])
        if policy not in {"SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"}:
            raise MalformedVLMSpecificationError(f"Operation group {raw_group_id!r} has unknown usage_policy: {policy!r}")

        canon_fn = "STIR_COFFEE" if fn_group == "coffee_stirring" else "PROVIDE_SOUP_EATING_UTENSIL"
        operations[canon_group_id] = {
            "raw_vlm_group_id": raw_group_id,
            "function": canon_fn,
            "canonical_function": canon_fn,
            "raw_function": str(raw_fn),
            "tool_role": tool_role,
            "target_role": target_role,
            "required_target_count": req_target_count,
            "usage_policy": {
                "mode": policy.lower(),
                "distinct_within_group": policy == "DEDICATED_PER_TARGET",
                "same_tool_must_cover_all_targets": False,
                "selection_preference": (
                    "minimize_distinct_tools"
                    if policy == "SEQUENTIAL_REUSE_ALLOWED" else "deterministic_rank"
                ),
            },
            "relations": mapped_op_rels,
        }
        concept_accounting["operation_groups"].append({
            "raw_group_id": raw_group_id,
            "canonical_group": canon_group_id,
            "raw_function": str(raw_fn),
            "canonical_function": canon_fn,
            "function_mapping_status": "PRESERVED",
            "tool_role": tool_role,
            "target_role": target_role,
            "required_target_count": req_target_count,
            "usage_policy": policy,
            "required_relations": mapped_op_rels,
            "status": "PRESERVED",
        })

    # Reconcile role cardinality with operation groups if not explicitly set
    for canon_gid, op_info in operations.items():
        t_role = op_info["tool_role"]
        if t_role in roles:
            r_entry = roles[t_role]
            t_req_count = op_info["required_target_count"]
            if op_info["usage_policy"]["mode"] == "sequential_reuse_allowed":
                if r_entry.get("min_count") is None:
                    r_entry["min_count"] = 1
                if r_entry.get("max_count") is None:
                    r_entry["max_count"] = t_req_count
                if r_entry.get("preference") is None:
                    r_entry["preference"] = "minimize_distinct"
            elif op_info["usage_policy"]["mode"] == "dedicated_per_target":
                if r_entry.get("min_count") is None:
                    r_entry["min_count"] = t_req_count
                if r_entry.get("max_count") is None:
                    r_entry["max_count"] = t_req_count

    # Deterministic Region Resolution: resolve ONLY through label/visual_description (ignore VLM local ID)
    local_id_to_canonical: dict[str, str] = {}
    region_proposal_trace: list[dict[str, Any]] = []
    canonical_to_raw_ids: dict[str, str] = {}
    raw_regions = valid_doc.get("inspectable_regions", [])
    for idx, prop in enumerate(raw_regions):
        if isinstance(prop, dict):
            prop_id = str(prop.get("id") or "")
            raw_label = str(prop.get("label") or "")
            raw_desc = str(prop.get("visual_description") or "")
        else:
            prop_id = str(prop)
            raw_label = str(prop)
            raw_desc = ""

        canon_reg = resolve_kitchen_region_proposal(prop)
        if canon_reg is None or canon_reg not in observable_regions:
            raise UnmappedFunctionalConceptError(
                f"Kitchen inspectable region proposal {prop_id!r} (label={raw_label!r}, "
                f"visual_description={raw_desc!r}) cannot be mapped to any known system search region "
                f"(available: {sorted(observable_regions)})"
            )

        if canon_reg in canonical_to_raw_ids:
            prev_raw_id = canonical_to_raw_ids[canon_reg]
            raise AmbiguousCanonicalizationError(
                f"Multiple raw region proposals ({prev_raw_id!r} and {prop_id!r}) map to the same "
                f"canonical search region {canon_reg!r}. Duplicate search region proposal collision fails closed."
            )

        canonical_to_raw_ids[canon_reg] = prop_id
        local_id_to_canonical[prop_id] = canon_reg
        region_proposal_trace.append({
            "raw_index": idx,
            "raw_id": prop_id,
            "raw_label": raw_label,
            "raw_visual_description": raw_desc,
            "canonical_region_id": canon_reg,
            "resolution_status": "RESOLVED",
            "reason": "Deterministic label/visual_description match",
        })

    resolved_candidate_regions = tuple(dict.fromkeys(local_id_to_canonical.values()))

    # Resolve inspection order strictly through local_id_to_canonical lookup only
    vlm_order = list(map(str, valid_doc.get("inspection_order", [])))
    resolved_order = []
    for local_id in vlm_order:
        canon_id = local_id_to_canonical.get(local_id)
        if canon_id is not None and canon_id in resolved_candidate_regions and canon_id not in resolved_order:
            resolved_order.append(canon_id)

    contract = {
        "schema_version": 2,
        "task_id": "qwen_kitchen_functional_graph",
        "specification_source": "qwen_vlm_natural_language_specification",
        "generated_from_foundation_model": True,
        "goal_instruction": task_instruction,
        "target_assignment_ablation": True,
        "pairing": {"strategy": "semantic_role_scoped"},
        "constraints": {
            "distinct_role_assignments": not bool(
                valid_doc.get("cross_group_reuse_allowed", False)
            )
        },
        "roles": roles,
        "relations": relations,
        "cross_group_reuse": {
            "allowed": bool(valid_doc.get("cross_group_reuse_allowed", False)),
            "selection_preference": "neutral",
        },
        "selection": {
            "policy": "ranked_valid_candidate",
            "semantic_confidence_is_only_a_gate": True,
        },
        "symbolic_task": {},
    }
    if operations:
        contract["operation_groups"] = operations

    loaded = load_task_requirements(deepcopy(contract))
    detector_vocabulary = {
        "schema_version": 1,
        "canonical_labels": vocabulary,
    }
    trace = {
        "schema_version": 2,
        "vlm_canonicalization_version": KITCHEN_VLM_CANONICALIZATION_VERSION,
        "transformation": "DETERMINISTIC_NATURAL_LANGUAGE_CANONICALIZATION",
        "raw_roles_preserved": raw_role_ids,
        "raw_role_to_canonical": raw_role_to_canonical,
        "raw_group_to_canonical": raw_group_to_canonical,
        "canonical_predicates_dispatched": canonical_predicates,
        "concept_accounting": concept_accounting,
        "resolved_regions": local_id_to_canonical,
        "region_proposal_trace": region_proposal_trace,
        "candidate_regions": list(resolved_candidate_regions),
        "inspection_order": resolved_order,
        "task_contract": {
            key: value for key, value in loaded.items() if not key.startswith("_")
        },
        "detector_vocabulary": detector_vocabulary,
    }
    return contract, {"object": detector_vocabulary}, trace

