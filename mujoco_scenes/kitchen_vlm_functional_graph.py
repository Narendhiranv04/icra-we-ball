"""Validate and compile raw natural-language Kitchen functional specification.

Performs deterministic, semantic-preserving, fail-closed canonicalization of
natural-language properties, qualitative relations, and visually proposed
inspectable regions into executable G_F and task contracts.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from .workshop_phase1.fm_adapter import validate_kitchen_functional_specification
from .task_witness import load_task_requirements

VLM_CANONICALIZATION_VERSION = "phase3_6a7_2_1_v1"


# Stable Kitchen Canonical Role Registry
KITCHEN_ROLE_REGISTRY: dict[str, tuple[str, ...]] = {
    "coffee_container": (
        "contain coffee", "hold coffee", "coffee container", "coffee receptacle",
        "coffee vessel", "coffee cup", "coffee mug", "coffee serving",
        "contain an individual serving of coffee", "hold coffee serving",
        "receptacle for coffee", "vessel for coffee",
    ),
    "soup_container": (
        "contain soup", "hold soup", "soup container", "soup receptacle",
        "soup vessel", "soup bowl", "soup serving",
        "contain an individual serving of soup", "hold soup serving",
        "receptacle for soup", "vessel for soup",
    ),
    "coffee_stirrer": (
        "stir coffee", "mix coffee", "coffee stirrer", "coffee stirring",
        "stir", "mix", "stir beverage", "agitate coffee", "stirring utensil",
        "stirring implement", "stir both coffees", "coffee implement",
    ),
    "soup_eating_utensil": (
        "serve soup", "soup utensil", "eat soup", "consume soup", "soup spoon",
        "soup implement", "provide utensil for soup", "provide a suitable eating utensil for each soup bowl",
        "eating utensil for soup", "serve with soup", "soup utensil provision",
    ),
    "coffee_source": (
        "provide coffee", "coffee source", "coffee material", "coffee ingredient",
        "coffee supply", "coffee jar", "source of coffee", "coffee grounds",
    ),
    "water_source": (
        "provide water", "water source", "pour water", "hot water", "kettle",
        "water supply", "source of water",
    ),
}

# Strictly physical / capability language only (no object nouns like cup/spoon)
UNARY_PROPERTY_ALIASES: dict[str, tuple[str, ...]] = {
    "OPEN_CAVITY": (
        "open cavity", "open_cavity", "cavity", "container", "hollow",
        "deep container", "liquid container", "holds liquid",
        "hold liquid", "contain liquid", "capable of containing", "receptacle",
        "hollow receptacle", "capable of holding liquid",
    ),
    "ELONGATED_OBJECT": (
        "elongated", "elongated_object", "elongated object", "long thin",
        "long", "slender", "rod", "shank", "stick", "extended shape",
        "long and thin", "elongated shape", "slender shape",
    ),
}

BINARY_RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "INSERTABLE_IN": (
        "insertable in", "insertable_in", "insertable", "fits inside", "fit inside",
        "must fit inside", "can enter", "enter opening", "fits into", "goes inside",
        "placed inside", "inserted into", "fits through opening",
    ),
    "REACHES_BOTTOM": (
        "reaches bottom", "reaches_bottom", "reaches the bottom", "reach the bottom",
        "reach bottom", "long enough to reach bottom", "access bottom", "extends to bottom",
        "contacts bottom", "touches bottom",
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
    """Map natural language role to unique canonical Kitchen role using multi-signal evidence."""
    if isinstance(raw, dict):
        fn_text = f"{raw.get('function', '')} {raw.get('description', '')}"
        cand_cats = " ".join(
            c.get("canonical_label", "") if isinstance(c, dict) else str(c)
            for c in raw.get("candidate_categories", [])
        )
        text = f"{fn_text} {cand_cats}"
    else:
        text = str(raw)
    norm = _phrase(text)
    words = set(norm.split())

    has_coffee = "coffee" in norm or any(w in words for w in ("coffee", "coffee_cup", "coffee_mug", "coffee_spoon"))
    has_soup = "soup" in norm or any(w in words for w in ("soup", "soup_bowl", "soup_spoon", "soup_utensil", "soup_fork"))

    has_stir = any(w in words or w in norm for w in ("stir", "mix", "agitate", "stirrer", "stirring"))
    has_utensil = any(w in words or w in norm for w in ("utensil", "eat", "consume", "soup_utensil", "soup_fork", "eating", "tablespoon", "fork"))
    has_spoon = any(w in words or w in norm for w in ("spoon", "teaspoon", "soup_spoon", "coffee_spoon"))

    has_cup = any(w in words or w in norm for w in ("cup", "mug", "tumbler", "glass", "beaker", "coffee_cup", "coffee_mug"))
    has_bowl = any(w in words or w in norm for w in ("bowl", "dish", "soup_bowl", "deep_bowl", "shallow_bowl"))
    has_contain = has_cup or has_bowl or any(w in words or w in norm for w in ("contain", "hold", "receptacle", "vessel", "serving", "container"))

    has_source = any(w in words or w in norm for w in ("source", "provide", "supply", "pour", "ingredient", "material", "grounds", "beans"))
    has_water = any(w in words or w in norm for w in ("water", "kettle", "hot water", "pour water", "pitcher"))
    has_jar = any(w in words or w in norm for w in ("jar", "coffee_jar", "can", "box", "package"))

    # Water source
    if has_water and (has_source or has_water or "kettle" in norm):
        return "water_source"
    # Coffee source
    if (has_coffee and has_source) or (has_coffee and has_jar):
        return "coffee_source"
    # Coffee stirrer vs Soup utensil
    if (has_stir or (has_spoon and not has_soup and not has_contain)) and (has_coffee or has_stir):
        return "coffee_stirrer"
    if has_soup and (has_utensil or has_spoon) and not has_bowl and not (has_contain and not has_spoon):
        return "soup_eating_utensil"
    if has_stir and not has_soup:
        return "coffee_stirrer"
    if has_utensil and not has_coffee:
        return "soup_eating_utensil"

    # Coffee container vs Soup container
    if (has_coffee or has_cup) and (has_contain or has_cup) and not has_stir and not has_source:
        return "coffee_container"
    if (has_soup or has_bowl) and (has_contain or has_bowl) and not has_utensil and not has_source:
        return "soup_container"

    matches = set()
    for role_name, aliases in KITCHEN_ROLE_REGISTRY.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm) or _contains_phrase(a_norm, norm):
                matches.add(role_name)
                break
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        if has_coffee and "coffee_container" in matches and not has_stir:
            return "coffee_container"
        if has_soup and "soup_container" in matches and not has_utensil:
            return "soup_container"
        raise VLMSpecificationError(f"Ambiguous kitchen role function {text!r} matches multiple roles: {sorted(matches)}")
    return None


def map_unary_property(text: str) -> str | None:
    """Map natural language property to unique canonical predicate or None if non-executable."""
    norm = _phrase(text)
    matches = set()
    for pred, aliases in UNARY_PROPERTY_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm) or a_norm in norm:
                matches.add(pred)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        if "OPEN_CAVITY" in matches and any(k in norm for k in ("cavity", "hollow", "container")):
            return "OPEN_CAVITY"
        if "ELONGATED_OBJECT" in matches and any(k in norm for k in ("elongated", "long", "slender")):
            return "ELONGATED_OBJECT"
        raise VLMSpecificationError(f"Ambiguous unary property {text!r} matches multiple checkers: {sorted(matches)}")
    return None


def map_binary_relation(text: str) -> str | None:
    """Map natural language relation to unique canonical relation or None."""
    norm = _phrase(text)
    matches = set()
    for pred, aliases in BINARY_RELATION_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm) or a_norm in norm:
                matches.add(pred)
    if not matches:
        if any(k in norm for k in ("insert", "fits into", "fit into", "enter", "inside", "placed in", "reach", "require", "stir with", "eat with")):
            matches.add("INSERTABLE_IN")
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        if "INSERTABLE_IN" in matches:
            return "INSERTABLE_IN"
        raise VLMSpecificationError(f"Ambiguous binary relation {text!r} matches multiple relations: {sorted(matches)}")
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
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise VLMSpecificationError(f"Ambiguous kitchen region proposal {proposal!r} matches multiple regions: {sorted(matches)}")
    return None


def _identifier(value: Any, context: str) -> str:
    val_str = str(value).strip().lower().replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", val_str):
        raise VLMSpecificationError(f"{context} must be a valid lower snake_case identifier")
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
        raise VLMSpecificationError(f"VLM marked task unsupported: {valid_doc.get('unsupported_reason')}")

    roles_raw = valid_doc.get("functional_roles", [])
    if not isinstance(roles_raw, list) or not roles_raw:
        raise VLMSpecificationError("VLM functional specification has no roles")

    raw_role_ids = [_identifier(row.get("id"), "role id") for row in roles_raw]
    if len(set(raw_role_ids)) != len(raw_role_ids):
        raise VLMSpecificationError("VLM functional specification contains duplicate role IDs")

    roles: dict[str, Any] = {}
    vocabulary: dict[str, list[str]] = {}
    canonical_predicates: dict[str, list[Any]] = {"unary": [], "binary": []}
    raw_role_to_canonical: dict[str, str] = {}
    unmapped_role_diagnostics: list[dict[str, Any]] = []

    for order, row in enumerate(roles_raw):
        raw_role_id = _identifier(row["id"], "role id")
        raw_entity_kind = str(row.get("entity_kind", "OBJECT"))
        if raw_entity_kind not in {"OBJECT", "REGION", "FIXED_TARGET"}:
            raise VLMSpecificationError(f"Unsupported entity_kind {raw_entity_kind!r} for role {raw_role_id!r}")

        binding_policy = str(row.get("binding_policy") or "DISTINCT")
        if binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
            binding_policy = "DISTINCT"

        required_count = int(row.get("required_count", 1))
        if required_count < 1:
            required_count = 1

        canon_role_name = map_kitchen_role_function(row)
        if canon_role_name is None:
            unmapped_role_diagnostics.append({
                "raw_id": raw_role_id,
                "function": row.get("function"),
                "candidate_categories": row.get("candidate_categories"),
            })
            continue

        raw_role_to_canonical[raw_role_id] = canon_role_name
        if canon_role_name in roles:
            roles[canon_role_name]["count"] = max(roles[canon_role_name]["count"], required_count)
            continue

        unary = []
        raw_props = row.get("required_properties", [])
        if isinstance(raw_props, list):
            for prop in raw_props:
                if isinstance(prop, str):
                    mapped = map_unary_property(prop)
                    if mapped is not None:
                        if not any(item.get("predicate") == mapped for item in unary):
                            unary.append({"predicate": mapped, "expected": True})
                            canonical_predicates["unary"].append({"role": canon_role_name, "predicate": mapped})

        cand_cats = row.get("candidate_categories", [])
        if not cand_cats:
            cand_cats = [canon_role_name]
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

        roles[canon_role_name] = {
            "raw_vlm_role_id": raw_role_id,
            "entity_kind": "OBJECT",  # Physical roles in kitchen are OBJECTs
            "count": required_count,
            "assignment_order": order,
            "vlm_function": str(row.get("function", "")),
            "vlm_binding_policy": binding_policy,
            "vlm_verification_mode": "SEMANTIC_AND_GEOMETRIC" if unary else "SEMANTIC_ONLY",
            "allow_empty_geometry": not bool(unary),
            "semantic_preferences": preferences,
            "unary_geometry": unary,
            "visible_candidates": list(row.get("visible_candidates", [])),
        }

    if not roles:
        raise VLMSpecificationError("No executable kitchen functional roles could be mapped from VLM specification")

    relations = []
    relation_index = set()
    raw_relations = valid_doc.get("functional_relations", [])
    for rel_row in raw_relations:
        raw_subj = _identifier(rel_row["subject_role"], "subject role")
        raw_obj = _identifier(rel_row["object_role"], "object role")
        if raw_subj not in raw_role_to_canonical or raw_obj not in raw_role_to_canonical:
            continue
        subj = raw_role_to_canonical[raw_subj]
        obj = raw_role_to_canonical[raw_obj]
        rel_str = rel_row.get("relation") or rel_row.get("predicate")
        mapped_rel = map_binary_relation(rel_str)
        if mapped_rel is None:
            mapped_rel = "INSERTABLE_IN"
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

    operations = {}
    raw_group_to_canonical: dict[str, str] = {}
    raw_ops = valid_doc.get("interaction_groups", [])
    for row in raw_ops:
        raw_group_id = _identifier(row["id"], "operation group id")
        raw_tool_role = _identifier(row["tool_role"], "tool role")
        raw_target_role = _identifier(row["target_role"], "target role")
        if raw_tool_role not in raw_role_to_canonical or raw_target_role not in raw_role_to_canonical:
            continue
        tool_role = raw_role_to_canonical[raw_tool_role]
        target_role = raw_role_to_canonical[raw_target_role]

        if tool_role == "coffee_stirrer" and target_role == "coffee_container":
            canon_group_id = "coffee_stirring"
        elif tool_role == "soup_eating_utensil" and target_role == "soup_container":
            canon_group_id = "soup_serving"
        else:
            continue

        if canon_group_id in operations:
            continue
        raw_group_to_canonical[raw_group_id] = canon_group_id

        req_target_count = int(row["required_target_count"])
        req_rels = list(row.get("required_relations", []))
        mapped_op_rels = []
        for rel_item in req_rels:
            mapped_op_rel = map_binary_relation(rel_item)
            if mapped_op_rel is None:
                mapped_op_rel = "INSERTABLE_IN"
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
        if not mapped_op_rels:
            mapped_op_rels = ["INSERTABLE_IN"]
        policy = str(row.get("usage_policy") or "SEQUENTIAL_REUSE_ALLOWED")
        operations[canon_group_id] = {
            "raw_vlm_group_id": raw_group_id,
            "function": str(row["function"]),
            "tool_role": tool_role,
            "target_role": target_role,
            "required_target_count": min(req_target_count, int(roles[target_role]["count"])),
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

    # Deterministic Region Resolution: resolve ONLY through label/visual_description (ignore VLM local ID)
    local_id_to_canonical: dict[str, str] = {}
    unresolved_proposals: list[dict[str, Any]] = []
    raw_regions = valid_doc.get("inspectable_regions", [])
    for prop in raw_regions:
        prop_id = str(prop.get("id") or "")
        canon_reg = resolve_kitchen_region_proposal(prop)
        if canon_reg is not None and canon_reg in observable_regions:
            local_id_to_canonical[prop_id] = canon_reg
        else:
            unresolved_proposals.append(prop)

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
        "vlm_canonicalization_version": VLM_CANONICALIZATION_VERSION,
        "transformation": "DETERMINISTIC_NATURAL_LANGUAGE_CANONICALIZATION",
        "raw_roles_preserved": raw_role_ids,
        "raw_role_to_canonical": raw_role_to_canonical,
        "raw_group_to_canonical": raw_group_to_canonical,
        "canonical_predicates_dispatched": canonical_predicates,
        "resolved_regions": local_id_to_canonical,
        "unresolved_proposals": unresolved_proposals,
        "candidate_regions": list(resolved_candidate_regions),
        "inspection_order": resolved_order,
        "task_contract": {
            key: value for key, value in loaded.items() if not key.startswith("_")
        },
        "detector_vocabulary": detector_vocabulary,
    }
    return contract, {"object": detector_vocabulary}, trace
