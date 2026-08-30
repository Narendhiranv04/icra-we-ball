"""Validate and compile one raw Qwen Kitchen graph without role/property mapping."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .task_witness import load_task_requirements


SUPPORTED_UNARY_CHECKERS = {"OPEN_CAVITY", "ELONGATED_OBJECT"}
SUPPORTED_BINARY_CHECKERS = {"INSERTABLE_IN", "REACHES_BOTTOM"}
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


def _identifier(value: Any, context: str) -> str:
    val_str = str(value).strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", val_str):
        raise ValueError(f"{context} must be a valid lower snake_case identifier")
    return val_str


def compile_vlm_functional_graph(
    document: dict[str, Any],
    *,
    task_instruction: str,
    observable_regions: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return task contract, detector vocabulary, and a literal mapping trace."""
    if document.get("status") != "SUPPORTED" and not document.get("roles"):
        raise ValueError(f"Qwen marked task unsupported: {document.get('unsupported_reason')}")
    roles_raw = document.get("roles")
    if not isinstance(roles_raw, list) or not roles_raw:
        raise ValueError("Qwen functional graph has no roles")
    role_ids = [_identifier(row.get("id"), "role id") for row in roles_raw]
    if len(set(role_ids)) != len(role_ids):
        raise ValueError("Qwen functional graph repeats role IDs")

    roles: dict[str, Any] = {}
    vocabulary: dict[str, list[str]] = {}
    exact_predicates = {"unary": [], "numeric": [], "binary": []}
    for order, row in enumerate(roles_raw):
        role_id = _identifier(row["id"], "role id")
        unary = []
        for requirement in row["unary_properties"]:
            predicate = str(requirement["predicate"])
            if predicate not in SUPPORTED_UNARY_CHECKERS:
                raise ValueError(
                    f"VLM emitted unary predicate {predicate!r}, but no exact checker exists"
                )
            unary.append({"predicate": predicate, "expected": True})
            exact_predicates["unary"].append({"role": role_id, "predicate": predicate})
        for requirement in row["numeric_properties"]:
            property_name = str(requirement["property"])
            expected_unit = SUPPORTED_NUMERIC_PROPERTIES.get(property_name)
            if expected_unit is None:
                raise ValueError(
                    f"VLM emitted numeric property {property_name!r}, but no exact checker exists"
                )
            unit = str(requirement["unit"])
            if unit != expected_unit:
                raise ValueError(
                    f"VLM emitted unit {unit!r} for {property_name}; expected {expected_unit!r}"
                )
            operator = str(requirement["operator"])
            numeric_requirement = {
                "property": property_name,
                "operator": operator,
                "value": float(requirement["value"]),
                "unit": unit,
            }
            unary.append(numeric_requirement)
            exact_predicates["numeric"].append({
                "role": role_id, **numeric_requirement,
            })
        verification_mode = str(row["verification_mode"])
        if verification_mode == "SEMANTIC_AND_GEOMETRIC" and not unary:
            raise ValueError(
                f"VLM geometric role {role_id} has no verifiable property"
            )
        if verification_mode == "SEMANTIC_ONLY" and unary:
            raise ValueError(
                f"VLM semantic-only role {role_id} declared geometric properties"
            )
        preferences = []
        for rank, category in enumerate(row["candidate_categories"], 1):
            canonical = _identifier(
                str(category["canonical_label"]).strip().lower().replace(" ", "_"),
                f"canonical category for {role_id}",
            )
            phrases = list(dict.fromkeys(
                [canonical.replace("_", " "), *map(str, category["detector_phrases"])]
            ))
            vocabulary.setdefault(canonical, [])
            vocabulary[canonical] = list(dict.fromkeys([*vocabulary[canonical], *phrases]))
            preferences.append({
                "rank": rank,
                "canonical_label": canonical,
                "detector_aliases": phrases,
            })
        roles[role_id] = {
            "count": int(row["required_count"]),
            "assignment_order": order,
            "vlm_function": str(row["function"]),
            "vlm_binding_policy": str(row["binding_policy"]),
            "vlm_verification_mode": verification_mode,
            "allow_empty_geometry": verification_mode == "SEMANTIC_ONLY",
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
    for row in document["relations"]:
        predicate = str(row["predicate"])
        if predicate not in SUPPORTED_BINARY_CHECKERS:
            raise ValueError(
                f"VLM emitted binary predicate {predicate!r}, but no exact checker exists"
            )
        subject, target = _identifier(row["subject_role"], "subject role"), _identifier(row["object_role"], "object role")
        if subject not in roles or target not in roles:
            raise ValueError("VLM relation references an unknown role")
        key = (predicate, subject, target)
        if key in relation_index:
            raise ValueError(f"VLM repeats relation {key}")
        relation_index.add(key)
        relations.append({
            "predicate": predicate,
            "subject_role": subject,
            "object_role": target,
            "expected": True,
        })
        exact_predicates["binary"].append({
            "subject_role": subject, "predicate": predicate, "object_role": target
        })

    operations = {}
    for row in document["operation_groups"]:
        group_id = _identifier(row["id"], "operation group id")
        tool_role, target_role = _identifier(row["tool_role"], "tool role"), _identifier(row["target_role"], "target role")
        if group_id in operations or tool_role not in roles or target_role not in roles:
            raise ValueError("VLM operation group has duplicate ID or unknown role")
        required_relations = list(map(str, row["required_relations"]))
        for predicate in required_relations:
            if (predicate, tool_role, target_role) not in relation_index:
                raise ValueError(
                    f"Operation {group_id} requires undeclared exact relation {predicate}"
                )
        policy = str(row["usage_policy"])
        req_target_count = int(row["required_target_count"])
        if target_role in roles:
            roles[target_role]["count"] = max(int(roles[target_role]["count"]), req_target_count)
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
            "relations": required_relations,
        }

    planning = document["planning"]
    source_roles = {}
    for source in planning["source_roles"]:
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
    for target in planning["target_requirements"]:
        content = _identifier(target["content"], "target content")
        witness_role = _identifier(target["witness_role"], "target witness role")
        op_group = _identifier(target["operation_group"], "target operation group")
        if witness_role not in roles or op_group not in operations:
            raise ValueError("VLM planning target references an unknown role or operation")
        targets[content] = {
            "witness_role": witness_role,
            "required_contents": list(map(str, target["required_contents"])),
            "initial_contents": list(map(str, target["initial_contents"])),
            "requires_operation_group": op_group,
            "final_goal": str(target["final_goal"]),
        }

    order = list(map(str, document["inspection_order"]))
    if len(order) != len(observable_regions) or set(order) != set(observable_regions):
        raise ValueError("VLM inspection_order must contain every observable region once")
    candidate_region_ids = {row["region_id"] for row in document["candidate_regions"]}
    if not candidate_region_ids.issubset(set(observable_regions)):
        raise ValueError("VLM candidate_regions contains an unknown region")

    contract = {
        "schema_version": 2,
        "task_id": "qwen_kitchen_functional_graph",
        "specification_source": "qwen_vlm_single_call_exact_graph",
        "generated_from_foundation_model": True,
        "goal_instruction": task_instruction,
        "target_assignment_ablation": True,
        "pairing": {"strategy": "semantic_role_scoped"},
        "constraints": {
            "distinct_role_assignments": not bool(
                document["cross_group_reuse_allowed"]
            )
        },
        "roles": roles,
        "relations": relations,
        "operation_groups": operations,
        "cross_group_reuse": {
            "allowed": bool(document["cross_group_reuse_allowed"]),
            "selection_preference": "neutral",
        },
        "selection": {
            "policy": "ranked_valid_candidate",
            "semantic_confidence_is_only_a_gate": True,
        },
        "symbolic_task": {
            "schema_version": 1,
            "home_region": "countertop",
            "initial_observation_region": "countertop",
            "contents": list(map(str, planning["contents"])),
            "source_roles": source_roles,
            "target_requirements": targets,
        },
    }
    # The existing loader is the structural API validator, not a reviewed task
    # oracle. It checks references/cardinalities without adding requirements.
    loaded = load_task_requirements(deepcopy(contract))
    detector_vocabulary = {
        "schema_version": 1,
        "canonical_labels": vocabulary,
    }
    trace = {
        "schema_version": 2,
        "transformation": "STRUCTURAL_ONLY_NO_ROLE_OR_PROPERTY_ALIAS_MAPPING",
        "raw_roles_preserved": role_ids,
        "exact_vlm_predicates_dispatched": exact_predicates,
        "added_task_requirements": [],
        "checker_api_validation_only": True,
        "compiler_runtime_defaults": {
            "home_region": "countertop",
            "initial_observation_region": "countertop",
            "semantic_assignment_policy": "ranked_valid_candidate",
        },
        "task_contract": {
            key: value for key, value in loaded.items() if not key.startswith("_")
        },
        "detector_vocabulary": detector_vocabulary,
        "inspection_order": order,
    }
    return contract, {"object": detector_vocabulary}, trace
