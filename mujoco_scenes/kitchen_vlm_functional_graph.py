"""Validate and compile raw natural-language Kitchen functional specification.

Performs deterministic, fail-closed canonicalization of natural-language properties,
qualitative relations, and visually proposed inspectable regions into executable G_F
and task contracts.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .task_witness import load_task_requirements


UNARY_PROPERTY_ALIASES: dict[str, tuple[str, ...]] = {
    "OPEN_CAVITY": (
        "open cavity", "open_cavity", "cavity", "container", "hollow", "cup",
        "mug", "bowl", "deep container", "liquid container", "holds liquid",
        "hold liquid", "contain liquid", "capable of containing", "receptacle",
        "contain coffee", "contain soup", "contain liquid serving",
    ),
    "ELONGATED_OBJECT": (
        "elongated", "elongated_object", "elongated object", "long thin",
        "long", "slender", "utensil", "rod", "spoon", "stirrer", "long enough",
        "shank", "stick", "soup spoon", "stirring utensil", "metal spoon",
        "serving utensil",
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
        "drawer above lower drawer", "first drawer", "topmost drawer", "d1",
    ),
    "D2": (
        "lower kitchen drawer", "lower drawer", "bottom kitchen drawer", "bottom drawer",
        "second drawer", "drawer below upper drawer", "d2",
    ),
    "C2": (
        "upper wall cupboard", "upper cupboard", "wall cupboard", "upper cabinet",
        "top cabinet", "wall cabinet", "overhead cupboard", "overhead cabinet",
        "cupboard above counter", "cabinet above counter", "upper storage", "c2",
    ),
    "B1": (
        "countertop storage box", "countertop box", "storage box", "wooden box",
        "counter box", "box on counter", "storage bin on counter", "tabletop box", "b1",
    ),
    "C1": (
        "lower kitchen cupboard", "lower cupboard", "bottom cupboard", "base cupboard",
        "lower cabinet", "base cabinet", "under counter cupboard", "cupboard below counter",
        "cabinet below counter", "lower storage", "c1",
    ),
}

SUPPORTED_NUMERIC_PROPERTIES = {
    "total_length_m": "m",
    "usable_length_m": "m",
    "maximum_cross_section_m": "m",
    "elongation_ratio": "ratio",
    "flatness_ratio": "ratio",
    "planarity_score": "ratio",
    "support_length_m": "m",
    "support_width_m": "m",
    "support_thickness_m": "m",
    "support_area_m2": "m2",
    "opening_width_m": "m",
    "opening_length_m": "m",
    "cavity_depth_m": "m",
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
    norm = _phrase(text)
    for pred, aliases in UNARY_PROPERTY_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                return pred
    return None


def map_binary_relation(text: str) -> str | None:
    norm = _phrase(text)
    for pred, aliases in BINARY_RELATION_ALIASES.items():
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm):
                return pred
    return None


def resolve_kitchen_region_proposal(proposal: dict[str, Any] | str) -> str | None:
    if isinstance(proposal, str):
        text = proposal.strip().lower()
    else:
        text = f"{proposal.get('id', '')} {proposal.get('region_id', '')} {proposal.get('label', '')} {proposal.get('visual_description', '')}".strip().lower()
    norm = _phrase(text)
    for reg_id, aliases in KITCHEN_REGION_ALIASES.items():
        if norm == reg_id.lower():
            return reg_id
        for alias in aliases:
            a_norm = _phrase(alias)
            if a_norm == norm or _contains_phrase(norm, a_norm) or _contains_phrase(a_norm, norm):
                return reg_id
    words = set(norm.split())
    if "upper" in words and "cupboard" in words:
        return "C2"
    if "upper" in words and "drawer" in words:
        return "D1"
    if "lower" in words and "drawer" in words:
        return "D2"
    if "lower" in words and "cupboard" in words:
        return "C1"
    if "countertop" in words and "box" in words:
        return "B1"
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
    """Return task contract, detector vocabulary, and deterministic trace."""
    if document.get("status") != "SUPPORTED":
        raise ValueError(f"VLM marked task unsupported: {document.get('unsupported_reason')}")

    roles_raw = document.get("functional_roles") or document.get("roles")
    if not isinstance(roles_raw, list) or not roles_raw:
        raise ValueError("VLM functional specification has no roles")

    role_ids = [_identifier(row.get("id"), "role id") for row in roles_raw]
    if len(set(role_ids)) != len(role_ids):
        raise ValueError("VLM functional specification contains duplicate role IDs")

    roles: dict[str, Any] = {}
    vocabulary: dict[str, list[str]] = {}
    canonical_predicates: dict[str, list[Any]] = {"unary": [], "numeric": [], "binary": []}

    for order, row in enumerate(roles_raw):
        role_id = _identifier(row["id"], "role id")
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
                elif isinstance(prop, dict) and "predicate" in prop:
                    pred = str(prop["predicate"])
                    mapped = map_unary_property(pred) or pred
                    if mapped not in UNARY_PROPERTY_ALIASES:
                        raise ValueError(
                            f"VLM emitted unary predicate {pred!r}, but no exact checker exists"
                        )
                    unary.append({"predicate": mapped, "expected": True})
                    canonical_predicates["unary"].append({"role": role_id, "predicate": mapped})

        for prop_item in row.get("unary_properties", []):
            if isinstance(prop_item, dict) and "predicate" in prop_item:
                pred = str(prop_item["predicate"])
                mapped = map_unary_property(pred) or pred
                if mapped not in UNARY_PROPERTY_ALIASES:
                    raise ValueError(
                        f"VLM emitted unary predicate {pred!r}, but no exact checker exists"
                    )
                if not any(item.get("predicate") == mapped for item in unary):
                    unary.append({"predicate": mapped, "expected": True})
                    canonical_predicates["unary"].append({"role": role_id, "predicate": mapped})

        for req in row.get("numeric_properties", []):
            p_name = str(req["property"])
            exp_unit = SUPPORTED_NUMERIC_PROPERTIES.get(p_name)
            if exp_unit is None:
                raise ValueError(f"VLM emitted numeric property {p_name!r}, but no exact checker exists")
            u = str(req["unit"])
            if u != exp_unit:
                raise ValueError(f"VLM emitted unit {u!r} for {p_name}; expected {exp_unit!r}")
            op = str(req["operator"])
            num_req = {
                "property": p_name,
                "operator": op,
                "value": float(req["value"]),
                "unit": u,
            }
            unary.append(num_req)
            canonical_predicates["numeric"].append({"role": role_id, **num_req})

        v_mode = str(row.get("verification_mode", "SEMANTIC_AND_GEOMETRIC" if unary else "SEMANTIC_ONLY"))
        if v_mode == "SEMANTIC_AND_GEOMETRIC" and not unary:
            raise ValueError(f"VLM geometric role {role_id} has no verifiable property")
        if v_mode == "SEMANTIC_ONLY" and unary:
            raise ValueError(f"VLM semantic-only role {role_id} declared geometric properties")

        cand_cats = row.get("candidate_categories", [])
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

        binding_policy = str(row.get("reuse_policy") or row.get("binding_policy") or "DISTINCT")
        roles[role_id] = {
            "count": int(row["required_count"]),
            "assignment_order": order,
            "vlm_function": str(row.get("function", "")),
            "vlm_binding_policy": binding_policy,
            "vlm_verification_mode": v_mode,
            "allow_empty_geometry": v_mode == "SEMANTIC_ONLY",
            "semantic_preferences": preferences,
            "unary_geometry": unary,
        }
        if "binding_cardinality" in row:
            roles[role_id]["binding_cardinality"] = dict(row["binding_cardinality"])
        elif "min_count" in row or "max_count" in row or "preference" in row:
            roles[role_id]["binding_cardinality"] = {
                "minimum_distinct_physical_objects": row.get("min_count", row.get("required_count", 1)),
                "maximum_distinct_physical_objects": row.get("max_count", row.get("required_count", 1)),
                "preferred": row.get("preference"),
            }

    relations = []
    relation_index = set()
    raw_relations = document.get("functional_relations") or document.get("relations") or []
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
    raw_ops = document.get("interaction_groups") or document.get("operation_groups") or []
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
        policy = str(row.get("usage_policy") or row.get("reuse_policy") or "SEQUENTIAL_REUSE_ALLOWED")
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

    # Deterministic Region Resolution
    resolved_regions_map: dict[str, str] = {}
    unresolved_proposals: list[dict[str, Any]] = []
    raw_regions = document.get("inspectable_regions") or document.get("candidate_regions") or []
    for prop in raw_regions:
        prop_id = str(prop.get("id") or prop.get("region_id") or "")
        canon_reg = resolve_kitchen_region_proposal(prop)
        if canon_reg is not None and canon_reg in observable_regions:
            resolved_regions_map[prop_id] = canon_reg
        else:
            unresolved_proposals.append(prop)

    resolved_candidate_regions = tuple(dict.fromkeys(resolved_regions_map.values()))

    # Resolve inspection order over VLM proposed region IDs
    vlm_order = list(map(str, document.get("inspection_order", [])))
    resolved_order = []
    for item in vlm_order:
        canon_id = resolved_regions_map.get(item) or resolve_kitchen_region_proposal(item)
        if canon_id is not None and canon_id in resolved_candidate_regions and canon_id not in resolved_order:
            resolved_order.append(canon_id)

    # Deterministically derive symbolic planning task if not directly supplied
    planning_raw = document.get("planning")
    if planning_raw is not None:
        source_roles = {}
        for source in planning_raw.get("source_roles", []):
            source_id = _identifier(source["id"], "source role id")
            witness_role = _identifier(source["witness_role"], "source witness role")
            if witness_role not in roles:
                raise ValueError(f"VLM source role {source_id} references an unknown role")
            source_roles[source_id] = {
                "witness_role": witness_role,
                "provides": str(source["provides"]),
                "count": 1,
            }
        targets = {}
        for target in planning_raw.get("target_requirements", []):
            content = _identifier(target["content"], "target content")
            witness_role = _identifier(target["witness_role"], "target witness role")
            op_group = _identifier(target["operation_group"], "target operation group")
            if witness_role not in roles or op_group not in operations:
                raise ValueError("VLM planning target references an unknown role or operation")
            targets[content] = {
                "witness_role": witness_role,
                "required_contents": list(map(str, target["required_contents"])),
                "initial_contents": list(map(str, target.get("initial_contents", []))),
                "requires_operation_group": op_group,
                "final_goal": str(target.get("final_goal", "served")),
            }
        symbolic_task = {
            "schema_version": 1,
            "home_region": "countertop",
            "initial_observation_region": "countertop",
            "contents": list(map(str, planning_raw.get("contents", ["coffee", "water", "soup"]))),
            "source_roles": source_roles,
            "target_requirements": targets,
        }
    else:
        # Deterministically derive symbolic task from canonical G_F roles and interaction groups
        source_roles = {}
        targets = {}
        for r_id, r_info in roles.items():
            fn = r_info["vlm_function"].lower()
            cats = [p["canonical_label"] for p in r_info["semantic_preferences"]]
            if "water" in r_id or "kettle" in cats:
                source_roles["water_source"] = {"witness_role": r_id, "provides": "water", "count": 1}
            elif "coffee" in r_id or "coffee_jar" in cats:
                source_roles["coffee_source"] = {"witness_role": r_id, "provides": "coffee", "count": 1}
            elif "water" in fn:
                source_roles["water_source"] = {"witness_role": r_id, "provides": "water", "count": 1}
            elif "coffee" in fn:
                source_roles["coffee_source"] = {"witness_role": r_id, "provides": "coffee", "count": 1}

        for gid, grp in operations.items():
            fn = grp["function"].lower()
            tgt_role = grp["target_role"]
            if "stir" in fn or "coffee" in fn or "drink" in tgt_role:
                targets["coffee"] = {
                    "witness_role": tgt_role,
                    "required_contents": ["water", "coffee"],
                    "initial_contents": [],
                    "requires_operation_group": gid,
                    "final_goal": "served",
                }
            elif "soup" in fn or "soup" in tgt_role or "utensil" in fn:
                targets["soup"] = {
                    "witness_role": tgt_role,
                    "required_contents": ["soup"],
                    "initial_contents": ["soup"],
                    "requires_operation_group": gid,
                    "final_goal": "served",
                }

        symbolic_task = {
            "schema_version": 1,
            "home_region": "countertop",
            "initial_observation_region": "countertop",
            "contents": ["coffee", "water", "soup"],
            "source_roles": source_roles,
            "target_requirements": targets,
        }

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
                document.get("cross_group_reuse_allowed", False)
            )
        },
        "roles": roles,
        "relations": relations,
        "operation_groups": operations,
        "cross_group_reuse": {
            "allowed": bool(document.get("cross_group_reuse_allowed", False)),
            "selection_preference": "neutral",
        },
        "selection": {
            "policy": "ranked_valid_candidate",
            "semantic_confidence_is_only_a_gate": True,
        },
        "symbolic_task": symbolic_task,
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
        "resolved_regions": resolved_regions_map,
        "unresolved_proposals": unresolved_proposals,
        "candidate_regions": list(resolved_candidate_regions),
        "inspection_order": resolved_order,
        "task_contract": {
            key: value for key, value in loaded.items() if not key.startswith("_")
        },
        "detector_vocabulary": detector_vocabulary,
    }
    return contract, {"object": detector_vocabulary}, trace

