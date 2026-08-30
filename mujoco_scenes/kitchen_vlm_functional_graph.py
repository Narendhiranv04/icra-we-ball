"""Validate and compile raw natural-language Kitchen functional specification.

Performs deterministic, semantic-preserving, fail-closed canonicalization of
natural-language properties, qualitative relations, and visually proposed
inspectable regions into executable G_F and task contracts.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .workshop_phase1.fm_adapter import validate_kitchen_functional_specification
from .task_witness import load_task_requirements


# Strictly physical / capability language only (no object nouns like cup/spoon)
UNARY_PROPERTY_ALIASES: dict[str, tuple[str, ...]] = {
    "OPEN_CAVITY": (
        "open cavity", "open_cavity", "cavity", "container", "hollow",
        "deep container", "liquid container", "holds liquid",
        "hold liquid", "contain liquid", "capable of containing", "receptacle",
        "contain coffee", "contain soup", "contain liquid serving",
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


def map_unary_property(text: str) -> str | None:
    """Map natural language property to unique canonical predicate or fail if ambiguous/unmapped."""
    norm = _phrase(text)
    matches = set()
    for pred, aliases in UNARY_PROPERTY_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(pred)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous unary property {text!r} matches multiple checkers: {sorted(matches)}")
    return None


def map_binary_relation(text: str) -> str | None:
    """Map natural language relation to unique canonical relation or fail if ambiguous/unmapped."""
    norm = _phrase(text)
    matches = set()
    for pred, aliases in BINARY_RELATION_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(pred)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous binary relation {text!r} matches multiple relations: {sorted(matches)}")
    return None


def resolve_kitchen_region_proposal(proposal: dict[str, Any] | str) -> str | None:
    """Resolve visual region proposal to canonical region ID using ONLY label and visual_description.

    VLM-local IDs (e.g. 'c2', 'region_1') are NOT semantic evidence and must be ignored.
    """
    if isinstance(proposal, str):
        text = proposal.strip().lower()
    else:
        # Strictly ignore proposal.get('id') or proposal.get('region_id')
        text = f"{proposal.get('label', '')} {proposal.get('visual_description', '')}".strip().lower()
    norm = _phrase(text)
    if not norm:
        return None
    matches = set()
    for reg_id, aliases in KITCHEN_REGION_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                matches.add(reg_id)
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous kitchen region proposal {proposal!r} matches multiple regions: {sorted(matches)}")
    return None


def _identifier(value: Any, context: str) -> str:
    val_str = str(value).strip().lower().replace(" ", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", val_str):
        raise ValueError(f"{context} must be a valid lower snake_case identifier")
    return val_str


def compile_vlm_functional_graph(
    document: dict[str, Any],
    *,
    task_instruction: str,
    observable_regions: tuple[str, ...] = tuple(KITCHEN_OBSERVABLE_REGIONS.keys()),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Deterministically compile raw Kitchen specification into canonical G_F and task contract."""
    # First strictly validate raw response schema locally
    valid_doc = validate_kitchen_functional_specification(document)
    if valid_doc.get("status") != "SUPPORTED":
        raise ValueError(f"VLM marked task unsupported: {valid_doc.get('unsupported_reason')}")

    roles_raw = valid_doc.get("functional_roles", [])
    if not isinstance(roles_raw, list) or not roles_raw:
        raise ValueError("VLM functional specification has no roles")

    role_ids = [_identifier(row.get("id"), "role id") for row in roles_raw]
    if len(set(role_ids)) != len(role_ids):
        raise ValueError("VLM functional specification contains duplicate role IDs")

    roles: dict[str, Any] = {}
    vocabulary: dict[str, list[str]] = {}
    canonical_predicates: dict[str, list[Any]] = {"unary": [], "binary": []}

    for order, row in enumerate(roles_raw):
        role_id = _identifier(row["id"], "role id")
        raw_entity_kind = str(row.get("entity_kind", "OBJECT"))
        if raw_entity_kind not in {"OBJECT", "REGION", "FIXED_TARGET"}:
            raise ValueError(f"Unsupported entity_kind {raw_entity_kind!r} for role {role_id!r}")

        binding_policy = str(row.get("binding_policy") or "")
        if binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
            raise ValueError(f"Role {role_id!r} missing required binding_policy (got {binding_policy!r})")

        required_count = int(row["required_count"])
        if required_count < 1:
            raise ValueError(f"Role {role_id!r} required_count must be >= 1")

        unary = []
        raw_props = row.get("required_properties", [])
        if isinstance(raw_props, list):
            for prop in raw_props:
                if isinstance(prop, str):
                    mapped = map_unary_property(prop)
                    if mapped is None:
                        raise ValueError(
                            f"VLM emitted unary property {prop!r} for role {role_id!r}, "
                            "but no exact or alias checker mapping exists"
                        )
                    if not any(item.get("predicate") == mapped for item in unary):
                        unary.append({"predicate": mapped, "expected": True})
                        canonical_predicates["unary"].append({"role": role_id, "predicate": mapped})

        cand_cats = row.get("candidate_categories", [])
        if not cand_cats:
            raise ValueError(f"Role {role_id!r} has no candidate categories")
        preferences = []
        seen_canon = set()
        for cat in cand_cats:
            if isinstance(cat, dict):
                canon_label = _identifier(cat.get("canonical_label", "").lower().replace(" ", "_"), f"canonical category for {role_id}")
                phrases = [canon_label.replace("_", " "), *map(lambda s: str(s).lower(), cat.get("detector_phrases", []))]
            else:
                c_str = str(cat).lower()
                canon_label = _identifier(c_str.replace(" ", "_"), f"canonical category for {role_id}")
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

        roles[role_id] = {
            "entity_kind": raw_entity_kind,
            "count": required_count,
            "assignment_order": order,
            "vlm_function": str(row.get("function", "")),
            "vlm_binding_policy": binding_policy,
            "vlm_verification_mode": "SEMANTIC_AND_GEOMETRIC" if unary else "SEMANTIC_ONLY",
            "allow_empty_geometry": not bool(unary),
            "semantic_preferences": preferences,
            "unary_geometry": unary,
        }

    relations = []
    relation_index = set()
    raw_relations = valid_doc.get("functional_relations", [])
    for rel_row in raw_relations:
        subj = _identifier(rel_row["subject_role"], "subject role")
        obj = _identifier(rel_row["object_role"], "object role")
        if subj not in roles or obj not in roles:
            raise ValueError(f"VLM relation references unknown roles: {subj}, {obj}")
        rel_str = rel_row.get("relation") or rel_row.get("predicate")
        mapped_rel = map_binary_relation(rel_str)
        if mapped_rel is None:
            raise ValueError(
                f"VLM emitted binary relation {rel_str!r}, but no exact or alias checker mapping exists"
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

    operations = {}
    raw_ops = valid_doc.get("interaction_groups", [])
    for row in raw_ops:
        group_id = _identifier(row["id"], "operation group id")
        tool_role = _identifier(row["tool_role"], "tool role")
        target_role = _identifier(row["target_role"], "target role")
        if group_id in operations or tool_role not in roles or target_role not in roles:
            raise ValueError("VLM operation group has duplicate ID or unknown role")
        req_target_count = int(row["required_target_count"])
        if int(roles[target_role]["count"]) != req_target_count:
            raise ValueError(
                f"VLM specification inconsistency: role {target_role} has required_count "
                f"{roles[target_role]['count']}, but operation group {group_id} requires {req_target_count}"
            )
        req_rels = list(row.get("required_relations", []))
        mapped_op_rels = []
        for rel_item in req_rels:
            mapped_op_rel = map_binary_relation(rel_item)
            if mapped_op_rel is None:
                raise ValueError(
                    f"VLM operation group emitted binary relation {rel_item!r}, but no exact checker exists"
                )
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
        policy = str(row.get("usage_policy") or "SEQUENTIAL_REUSE_ALLOWED")
        operations[group_id] = {
            "function": str(row["function"]),
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
        "operation_groups": operations,
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

    loaded = load_task_requirements(deepcopy(contract))
    detector_vocabulary = {
        "schema_version": 1,
        "canonical_labels": vocabulary,
    }
    trace = {
        "schema_version": 2,
        "transformation": "DETERMINISTIC_NATURAL_LANGUAGE_CANONICALIZATION",
        "raw_roles_preserved": role_ids,
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
