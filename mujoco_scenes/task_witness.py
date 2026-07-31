"""Deterministic geometry-only task witness evaluation."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_TASK_REQUIREMENTS_PATH = (
    CONFIG_DIR / "serve_two_person_breakfast.yaml"
)
WITNESS_SCHEMA_VERSION = 2
RELATION_STATUSES = {"TRUE", "FALSE", "UNKNOWN"}
PROPERTY_STATUSES = {"MEASURED", "DERIVED", "UNKNOWN"}
GROUNDING_MODES = {"joint", "geometry-only", "semantic-only"}


def resolve_task_requirements_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    package_relative = Path(__file__).resolve().parent / candidate
    if package_relative.exists():
        return package_relative.resolve()
    config_relative = CONFIG_DIR / candidate.name
    if config_relative.exists():
        return config_relative.resolve()
    raise FileNotFoundError(f"Task-requirement configuration not found: {path}")


def _normalize_true(value: Any, context: str) -> str:
    if value is True:
        return "TRUE"
    if isinstance(value, str) and value.upper() == "TRUE":
        return "TRUE"
    raise ValueError(f"{context} must require status TRUE")


def _find_forbidden_keys(value: Any) -> set[str]:
    forbidden = {
        "category",
        "categories",
        "candidate_function",
        "category_functions",
        "object_family",
    }
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                found.add(key)
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def _normalize_joint_unary_requirement(
    requirement: dict[str, Any],
    *,
    role_name: str,
    index: int,
) -> dict[str, Any]:
    if not isinstance(requirement, dict):
        raise ValueError(
            f"Role '{role_name}' unary_geometry {index} must be a mapping"
        )
    if "predicate" in requirement:
        expected = requirement.get(
            "expected", requirement.get("required_status", True)
        )
        if expected not in {True, "TRUE"}:
            raise ValueError(
                "Joint unary predicate requirements currently support "
                "expected: true only"
            )
        return {
            "predicate": str(requirement["predicate"]),
            "required_status": "TRUE",
        }
    if "property" not in requirement:
        raise ValueError(
            f"Role '{role_name}' unary_geometry {index} needs predicate "
            "or property"
        )
    operator = requirement.get("operator")
    value = requirement.get("value")
    if operator not in {">=", "<=", "=="}:
        raise ValueError(
            f"Role '{role_name}' property operator must be >=, <=, or =="
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Joint unary property value must be numeric")
    normalized = {
        "property": str(requirement["property"]),
        "unit": requirement.get("unit", "m"),
        "allowed_statuses": requirement.get(
            "allowed_statuses", ["MEASURED", "DERIVED"]
        ),
    }
    normalized[{">=": "minimum", "<=": "maximum", "==": "equals"}[operator]] = (
        float(value)
    )
    return normalized


def _validate_joint_task_requirements(
    config: dict[str, Any],
) -> dict[str, Any]:
    task_id = config.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Joint task requirements need a non-empty task_id")
    roles = config.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("Joint task requirements need roles")
    assignment_orders = set()
    for role_name, role in roles.items():
        if not isinstance(role, dict):
            raise ValueError(f"Role '{role_name}' must be a mapping")
        role.setdefault("count", 1)
        if (
            isinstance(role["count"], bool)
            or not isinstance(role["count"], int)
            or role["count"] < 1
        ):
            raise ValueError(f"Role '{role_name}' count must be positive")
        assignment_order = role.setdefault(
            "assignment_order", len(assignment_orders)
        )
        if (
            isinstance(assignment_order, bool)
            or not isinstance(assignment_order, int)
            or assignment_order < 0
            or assignment_order in assignment_orders
        ):
            raise ValueError("Role assignment_order values must be unique")
        assignment_orders.add(assignment_order)
        preferences = role.get("semantic_preferences")
        if not isinstance(preferences, list) or not preferences:
            raise ValueError(
                f"Role '{role_name}' needs semantic_preferences"
            )
        ranks = set()
        aliases_seen = set()
        for preference in preferences:
            if not isinstance(preference, dict):
                raise ValueError("Semantic preferences must be mappings")
            rank = preference.get("rank")
            canonical = preference.get("canonical_label")
            aliases = preference.get("detector_aliases", [])
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
                or rank in ranks
            ):
                raise ValueError(
                    f"Role '{role_name}' semantic ranks must be unique"
                )
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("canonical_label must be non-empty")
            if not isinstance(aliases, list):
                raise ValueError("detector_aliases must be a list")
            ranks.add(rank)
            preference["canonical_label"] = canonical.strip().lower()
            preference["detector_aliases"] = sorted(
                {
                    canonical.strip().lower(),
                    *(str(alias).strip().lower() for alias in aliases),
                }
            )
            overlap = aliases_seen & set(preference["detector_aliases"])
            if overlap:
                raise ValueError(
                    f"Role '{role_name}' repeats aliases: {sorted(overlap)}"
                )
            aliases_seen.update(preference["detector_aliases"])
        unary = role.get("unary_geometry")
        if not isinstance(unary, list) or not unary:
            raise ValueError(f"Role '{role_name}' needs unary_geometry")
        role["geometric_requirements"] = [
            _normalize_joint_unary_requirement(
                requirement,
                role_name=role_name,
                index=index,
            )
            for index, requirement in enumerate(unary)
        ]

    relations = config.setdefault("relations", [])
    if not isinstance(relations, list):
        raise ValueError("relations must be a list")
    normalized_pairwise = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError(f"Relation {index} must be a mapping")
        predicate = relation.get("predicate")
        subject = relation.get("subject_role")
        target = relation.get("object_role")
        expected = relation.get("expected", True)
        if not isinstance(predicate, str) or not predicate:
            raise ValueError(f"Relation {index} needs predicate")
        if subject not in roles or target not in roles:
            raise ValueError(f"Relation {index} references an unknown role")
        if expected not in {True, "TRUE"}:
            raise ValueError("Joint relations currently require expected: true")
        normalized_pairwise.append(
            {
                "relation": predicate,
                "from_role": subject,
                "to_role": target,
                "required_status": "TRUE",
                "apply_to": "all_selected_targets",
            }
        )
    constraints = config.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a mapping")
    distinct = constraints.get(
        "distinct_role_assignments",
        constraints.get("distinct_objects", True),
    )
    if not isinstance(distinct, bool):
        raise ValueError("distinct_role_assignments must be boolean")
    constraints["distinct_role_assignments"] = distinct
    constraints["distinct_objects"] = distinct
    constraints["pairwise"] = normalized_pairwise
    selection = config.setdefault("selection", {})
    if selection.setdefault("policy", "ranked_valid_candidate") != (
        "ranked_valid_candidate"
    ):
        raise ValueError("Only ranked_valid_candidate selection is supported")
    selection.setdefault("semantic_confidence_is_only_a_gate", True)
    diagnostics = config.setdefault("diagnostics", {})
    diagnostics.setdefault("max_representative_rejections", 50)
    config["_task_schema"] = "JOINT_ROLE_GROUNDING"
    return config


def load_task_requirements(
    source: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a category-free geometric task requirement mapping."""
    if source is None:
        source = DEFAULT_TASK_REQUIREMENTS_PATH
    if isinstance(source, dict):
        config = deepcopy(source)
    else:
        path = resolve_task_requirements_path(source)
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Task requirements must be a YAML/JSON mapping")
    is_joint = any(
        isinstance(role, dict)
        and (
            "semantic_preferences" in role
            or "unary_geometry" in role
        )
        for role in config.get("roles", {}).values()
    )
    if is_joint:
        return _validate_joint_task_requirements(config)

    forbidden = _find_forbidden_keys(config)
    if forbidden:
        raise ValueError(
            "Geometry-only tasks cannot use semantic keys: "
            + ", ".join(sorted(forbidden))
        )

    task_id = config.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("Task requirements need a non-empty task_id")
    roles = config.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("Task requirements need at least one role")
    for role_name, role in roles.items():
        if not isinstance(role_name, str) or not role_name:
            raise ValueError("Task role names must be non-empty strings")
        if not isinstance(role, dict):
            raise ValueError(f"Role '{role_name}' must be a mapping")
        count = role.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Role '{role_name}' count must be positive")
        requirements = role.get("geometric_requirements")
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(
                f"Role '{role_name}' needs geometric_requirements"
            )
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, dict):
                raise ValueError(
                    f"Role '{role_name}' requirement {index} must be a mapping"
                )
            has_predicate = "predicate" in requirement
            has_property = "property" in requirement
            if has_predicate == has_property:
                raise ValueError(
                    f"Role '{role_name}' requirement {index} must define "
                    "exactly one of predicate or property"
                )
            if has_predicate:
                predicate = requirement["predicate"]
                if not isinstance(predicate, str) or not predicate:
                    raise ValueError("Geometric predicate names must be strings")
                requirement["required_status"] = _normalize_true(
                    requirement.get("required_status", "TRUE"),
                    f"Role '{role_name}' predicate '{predicate}'",
                )
                continue
            property_name = requirement["property"]
            if not isinstance(property_name, str) or not property_name:
                raise ValueError("Geometric property names must be strings")
            bounds = [
                key for key in ("minimum", "maximum", "equals")
                if key in requirement
            ]
            if not bounds:
                raise ValueError(
                    f"Property '{property_name}' needs a numeric bound"
                )
            for key in bounds:
                value = requirement[key]
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(
                        f"Property '{property_name}' {key} must be numeric"
                    )
            allowed = requirement.setdefault(
                "allowed_statuses", ["MEASURED", "DERIVED"]
            )
            if not isinstance(allowed, list) or not allowed:
                raise ValueError("allowed_statuses must be a non-empty list")
            if not set(allowed).issubset(PROPERTY_STATUSES - {"UNKNOWN"}):
                raise ValueError(
                    "allowed_statuses may contain MEASURED and/or DERIVED"
                )

    constraints = config.setdefault("constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a mapping")
    distinct = constraints.setdefault("distinct_objects", True)
    if not isinstance(distinct, bool):
        raise ValueError("constraints.distinct_objects must be boolean")
    pairwise = constraints.setdefault("pairwise", [])
    if not isinstance(pairwise, list):
        raise ValueError("constraints.pairwise must be a list")
    for index, constraint in enumerate(pairwise):
        if not isinstance(constraint, dict):
            raise ValueError(f"Pairwise constraint {index} must be a mapping")
        relation = constraint.get("relation")
        if not isinstance(relation, str) or not relation:
            raise ValueError(f"Pairwise constraint {index} needs a relation")
        for key in ("from_role", "to_role"):
            if constraint.get(key) not in roles:
                raise ValueError(
                    f"Pairwise constraint {index} references unknown "
                    f"{key} '{constraint.get(key)}'"
                )
        if constraint.setdefault(
            "apply_to", "all_selected_targets"
        ) != "all_selected_targets":
            raise ValueError(
                "Only apply_to: all_selected_targets is supported"
            )
        constraint["required_status"] = _normalize_true(
            constraint.get("required_status", "TRUE"),
            f"Pairwise constraint {index}",
        )

    diagnostics = config.setdefault("diagnostics", {})
    maximum = diagnostics.setdefault("max_representative_rejections", 5)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError(
            "diagnostics.max_representative_rejections must be non-negative"
        )
    config["_task_schema"] = "GEOMETRY_ONLY"
    return config


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def evaluate_geometric_requirements(
    object_record: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one object against explicit geometric requirements."""
    properties = object_record.get("geometric_properties", {})
    predicates = object_record.get("geometric_predicates", {})
    checks: list[dict[str, Any]] = []
    for requirement in requirements:
        if "predicate" in requirement:
            name = requirement["predicate"]
            predicate = predicates.get(name, {})
            status = predicate.get("status", "UNKNOWN")
            if status not in RELATION_STATUSES:
                status = "UNKNOWN"
            checks.append(
                {
                    "kind": "predicate",
                    "name": name,
                    "status": status,
                    "required_status": "TRUE",
                    "method": predicate.get("method"),
                    "evidence": predicate.get("evidence", {}),
                    "reason": predicate.get("reason"),
                }
            )
            continue

        name = requirement["property"]
        record = properties.get(name, {})
        evidence_status = record.get("status", "UNKNOWN")
        value = _finite_number(record.get("value"))
        expected_unit = requirement.get("unit")
        actual_unit = record.get("unit")
        reason = None
        if value is None or evidence_status == "UNKNOWN":
            status = "UNKNOWN"
            reason = "PROPERTY_UNAVAILABLE"
        elif evidence_status not in requirement["allowed_statuses"]:
            status = "UNKNOWN"
            reason = "EVIDENCE_STATUS_NOT_ALLOWED"
        elif expected_unit is not None and actual_unit != expected_unit:
            status = "UNKNOWN"
            reason = "UNIT_MISMATCH"
        else:
            passed = True
            if "minimum" in requirement:
                passed &= value >= float(requirement["minimum"])
            if "maximum" in requirement:
                passed &= value <= float(requirement["maximum"])
            if "equals" in requirement:
                passed &= value == float(requirement["equals"])
            status = "TRUE" if passed else "FALSE"
        checks.append(
            {
                "kind": "property",
                "name": name,
                "status": status,
                "value": record.get("value"),
                "unit": actual_unit,
                "evidence_status": evidence_status,
                "bounds": {
                    key: requirement[key]
                    for key in ("minimum", "maximum", "equals")
                    if key in requirement
                },
                "method": record.get("method"),
                "reason": reason,
            }
        )
    statuses = {check["status"] for check in checks}
    aggregate = (
        "FALSE"
        if "FALSE" in statuses
        else "UNKNOWN"
        if "UNKNOWN" in statuses
        else "TRUE"
    )
    return {"status": aggregate, "checks": checks}


def _graph_index(
    graph: dict[str, Any],
) -> tuple[
    dict[str, str],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str, str], str],
]:
    object_nodes = {
        node["id"]: node.get("attributes", {}).get("object_id")
        for node in graph.get("nodes", [])
        if node.get("type") == "object"
    }
    object_nodes = {
        node_id: object_id
        for node_id, object_id in object_nodes.items()
        if isinstance(object_id, str)
    }
    role_candidates: dict[str, list[dict[str, Any]]] = {}
    relation_status: dict[tuple[str, str, str], str] = {}
    for edge in graph.get("edges", []):
        relation = edge.get("relation")
        source_id = object_nodes.get(edge.get("source"))
        target = edge.get("target")
        if relation == "SATISFIES_GEOMETRY":
            if source_id is None or not isinstance(target, str):
                continue
            if not target.startswith("role:"):
                continue
            status = edge.get("status", "UNKNOWN")
            if status == "FALSE":
                continue
            if status not in {"TRUE", "UNKNOWN"}:
                status = "UNKNOWN"
            role_candidates.setdefault(
                target.removeprefix("role:"), []
            ).append(
                {
                    "object_id": source_id,
                    "status": status,
                    "checks": edge.get("evidence", {}).get("checks", []),
                }
            )
            continue
        target_id = object_nodes.get(target)
        if source_id is None or target_id is None:
            continue
        status = edge.get("status", "UNKNOWN")
        if status not in RELATION_STATUSES:
            status = "UNKNOWN"
        relation_status[(str(relation), source_id, target_id)] = status
    for candidates in role_candidates.values():
        candidates.sort(key=lambda record: record["object_id"])
    return object_nodes, role_candidates, relation_status


def _selected_mapping(
    assignment: dict[tuple[str, int], dict[str, Any]],
    role_requirements: dict[str, int],
) -> dict[str, list[str]]:
    return {
        role: [
            assignment[(role, index)]["object_id"]
            for index in range(role_requirements[role])
            if (role, index) in assignment
        ]
        for role in sorted(role_requirements)
    }


def _relation_instances(
    assignment: dict[tuple[str, int], dict[str, Any]],
    role_requirements: dict[str, int],
    constraints: list[dict[str, Any]],
    relation_status: dict[tuple[str, str, str], str],
) -> list[dict[str, Any]]:
    selected = _selected_mapping(assignment, role_requirements)
    checks = []
    for constraint_index, constraint in enumerate(constraints):
        for source_id in selected[constraint["from_role"]]:
            for target_id in selected[constraint["to_role"]]:
                if source_id == target_id:
                    continue
                relation = constraint["relation"]
                checks.append(
                    {
                        "constraint_index": constraint_index,
                        "relation": relation,
                        "from_role": constraint["from_role"],
                        "to_role": constraint["to_role"],
                        "from_object": source_id,
                        "to_object": target_id,
                        "status": relation_status.get(
                            (relation, source_id, target_id), "UNKNOWN"
                        ),
                        "required_status": "TRUE",
                    }
                )
    return checks


def evaluate_task_witness(
    graph: dict[str, Any],
    requirements: dict[str, Any] | str | Path | None,
    *,
    stage: int | None = None,
) -> dict[str, Any]:
    """Return the deterministic geometry-only witness status and assignment."""
    config = load_task_requirements(requirements)
    if config.get("_task_schema") == "JOINT_ROLE_GROUNDING":
        return evaluate_joint_task_witness(
            graph,
            config,
            stage=stage,
            grounding_mode=str(
                graph.get("grounding_mode", "joint")
            ),
        )
    roles = config["roles"]
    pairwise = config["constraints"]["pairwise"]
    distinct = bool(config["constraints"]["distinct_objects"])
    max_examples = int(
        config["diagnostics"]["max_representative_rejections"]
    )
    if stage is None:
        stage = int(graph.get("stage", -1))
    _objects, candidates_by_role, relation_status = _graph_index(graph)
    role_requirements = {
        role: int(roles[role]["count"]) for role in sorted(roles)
    }
    candidates = {
        role: list(candidates_by_role.get(role, []))
        for role in sorted(roles)
    }
    observed_candidates = {
        role: [record["object_id"] for record in candidates[role]]
        for role in sorted(roles)
    }
    confirmed_candidates = {
        role: [
            record["object_id"]
            for record in candidates[role]
            if record["status"] == "TRUE"
        ]
        for role in sorted(roles)
    }
    indeterminate_candidates = {
        role: [
            record["object_id"]
            for record in candidates[role]
            if record["status"] == "UNKNOWN"
        ]
        for role in sorted(roles)
    }
    satisfied_counts = {
        role: min(
            len(confirmed_candidates[role]), role_requirements[role]
        )
        for role in sorted(roles)
    }
    missing_counts = {
        role: role_requirements[role] - len(observed_candidates[role])
        for role in sorted(roles)
        if len(observed_candidates[role]) < role_requirements[role]
    }
    result: dict[str, Any] = {
        "schema_version": WITNESS_SCHEMA_VERSION,
        "inference_basis": "GEOMETRY_ONLY",
        "task_id": config["task_id"],
        "stage": stage,
        "status": "INCOMPLETE",
        "selected_witness": None,
        "role_requirements": role_requirements,
        "observed_candidates": observed_candidates,
        "confirmed_candidates": confirmed_candidates,
        "indeterminate_candidates": indeterminate_candidates,
        "satisfied_candidate_counts": satisfied_counts,
        "missing_counts": missing_counts,
        "unknown_requirements": [],
        "unknown_relations": [],
        "false_relations": [],
        "selected_candidate_edges": [],
        "selected_pairwise_relations": [],
        "reason_codes": [],
        "search_diagnostics": {
            "assignments_completed": 0,
            "distinctness_rejections": 0,
            "false_relation_prunes": 0,
            "indeterminate_assignments": 0,
            "representative_rejections": [],
        },
    }
    if missing_counts:
        result["reason_codes"] = ["INSUFFICIENT_GEOMETRIC_CANDIDATES"]
        return result

    slots = [
        (role, index)
        for role in sorted(role_requirements)
        for index in range(role_requirements[role])
    ]
    assignment: dict[tuple[str, int], dict[str, Any]] = {}
    used: set[str] = set()
    first_complete = None
    first_indeterminate = None
    representative_false = []
    diagnostics = result["search_diagnostics"]

    def remember_example(example: dict[str, Any]) -> None:
        examples = diagnostics["representative_rejections"]
        if len(examples) < max_examples and example not in examples:
            examples.append(example)

    def search(index: int) -> bool:
        nonlocal first_complete, first_indeterminate
        if index == len(slots):
            diagnostics["assignments_completed"] += 1
            selected = _selected_mapping(assignment, role_requirements)
            relation_checks = _relation_instances(
                assignment, role_requirements, pairwise, relation_status
            )
            false_checks = [
                check for check in relation_checks
                if check["status"] == "FALSE"
            ]
            if false_checks:
                diagnostics["false_relation_prunes"] += 1
                for check in false_checks:
                    if (
                        len(representative_false) < max_examples
                        and check not in representative_false
                    ):
                        representative_false.append(check)
                    remember_example(
                        {"reason": "REQUIRED_RELATION_FALSE", **check}
                    )
                return False
            unknown_roles = [
                {
                    "relation": "SATISFIES_GEOMETRY",
                    "role": role,
                    "object_id": assignment[(role, slot)]["object_id"],
                    "status": "UNKNOWN",
                    "checks": assignment[(role, slot)]["checks"],
                }
                for role, slot in slots
                if assignment[(role, slot)]["status"] == "UNKNOWN"
            ]
            unknown_relations = [
                check for check in relation_checks
                if check["status"] == "UNKNOWN"
            ]
            if unknown_roles or unknown_relations:
                diagnostics["indeterminate_assignments"] += 1
                if first_indeterminate is None:
                    first_indeterminate = (
                        deepcopy(selected),
                        unknown_roles,
                        unknown_relations,
                    )
                return False
            first_complete = (deepcopy(selected), relation_checks)
            return True

        role, slot = slots[index]
        for candidate in candidates[role]:
            object_id = candidate["object_id"]
            if distinct and object_id in used:
                diagnostics["distinctness_rejections"] += 1
                remember_example(
                    {
                        "reason": "DISTINCT_OBJECT_REUSE",
                        "object_id": object_id,
                        "role": role,
                        "slot": slot,
                    }
                )
                continue
            assignment[(role, slot)] = candidate
            if distinct:
                used.add(object_id)
            grounded = _relation_instances(
                assignment, role_requirements, pairwise, relation_status
            )
            false_checks = [
                check for check in grounded if check["status"] == "FALSE"
            ]
            found = False
            if false_checks:
                diagnostics["false_relation_prunes"] += 1
                for check in false_checks:
                    if (
                        len(representative_false) < max_examples
                        and check not in representative_false
                    ):
                        representative_false.append(check)
                    remember_example(
                        {"reason": "REQUIRED_RELATION_FALSE", **check}
                    )
            else:
                found = search(index + 1)
            if distinct:
                used.remove(object_id)
            del assignment[(role, slot)]
            if found:
                return True
        return False

    search(0)
    if first_complete is not None:
        selected, relation_checks = first_complete
        result.update(
            {
                "status": "COMPLETE",
                "selected_witness": selected,
                "selected_candidate_edges": [
                    {"object_id": object_id, "role": role}
                    for role in sorted(selected)
                    for object_id in selected[role]
                ],
                "selected_pairwise_relations": relation_checks,
                "reason_codes": ["COMPLETE_GEOMETRIC_WITNESS_FOUND"],
            }
        )
        return result
    if first_indeterminate is not None:
        selected, unknown_roles, unknown_relations = first_indeterminate
        result.update(
            {
                "status": "INDETERMINATE",
                "indeterminate_assignment": selected,
                "unknown_requirements": unknown_roles,
                "unknown_relations": unknown_relations,
                "reason_codes": ["INSUFFICIENT_GEOMETRIC_EVIDENCE"],
            }
        )
        return result

    reasons = []
    if diagnostics["false_relation_prunes"]:
        reasons.append("REQUIRED_RELATION_FALSE")
    if diagnostics["distinctness_rejections"]:
        reasons.append("GLOBAL_DISTINCTNESS_UNSATISFIABLE")
    result["false_relations"] = representative_false
    result["reason_codes"] = reasons or ["NO_VALID_GEOMETRIC_ASSIGNMENT"]
    return result


def _joint_graph_index(
    graph: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    objects = {
        str(node.get("attributes", {}).get("object_id")): node.get(
            "attributes", {}
        )
        for node in graph.get("nodes", [])
        if node.get("type") == "object"
        and isinstance(node.get("attributes", {}).get("object_id"), str)
    }
    node_to_object = {
        node["id"]: str(node.get("attributes", {}).get("object_id"))
        for node in graph.get("nodes", [])
        if node.get("type") == "object"
        and isinstance(node.get("attributes", {}).get("object_id"), str)
    }
    geometry: dict[tuple[str, str], dict[str, Any]] = {}
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in graph.get("edges", []):
        source = node_to_object.get(edge.get("source"))
        if source is None:
            continue
        relation = str(edge.get("relation", ""))
        target = edge.get("target")
        if (
            relation == "SATISFIES_GEOMETRY"
            and isinstance(target, str)
            and target.startswith("role:")
        ):
            geometry[(source, target.removeprefix("role:"))] = edge
            continue
        target_object = node_to_object.get(target)
        if target_object is not None:
            relations[(relation, source, target_object)] = edge
    return objects, geometry, relations


def evaluate_semantic_compatibility(
    object_attributes: dict[str, Any],
    role: dict[str, Any],
) -> dict[str, Any]:
    """Return tri-state compatibility from cached detector evidence."""
    semantics = object_attributes.get("semantics", {})
    validated = semantics.get("validated")
    if (
        not isinstance(validated, dict)
        or validated.get("status") != "SUPPORTED"
        or not isinstance(validated.get("canonical_label"), str)
    ):
        return {
            "status": "UNKNOWN",
            "canonical_label": None,
            "semantic_rank": None,
            "confidence": None,
            "reason": "NO_VALIDATED_SEMANTIC_EVIDENCE",
            "provenance": validated,
        }
    label = validated["canonical_label"].strip().lower()
    for preference in sorted(
        role["semantic_preferences"], key=lambda item: item["rank"]
    ):
        accepted = {
            preference["canonical_label"],
            *preference.get("detector_aliases", []),
        }
        if label in accepted:
            return {
                "status": "TRUE",
                "canonical_label": label,
                "semantic_rank": int(preference["rank"]),
                "confidence": validated.get("mean_confidence"),
                "reason": "SUPPORTED_ACCEPTABLE_LABEL",
                "preference": deepcopy(preference),
                "provenance": validated,
            }
    return {
        "status": "FALSE",
        "canonical_label": label,
        "semantic_rank": None,
        "confidence": validated.get("mean_confidence"),
        "reason": "SUPPORTED_EXCLUDED_LABEL",
        "acceptable_labels": [
            preference["canonical_label"]
            for preference in sorted(
                role["semantic_preferences"],
                key=lambda item: item["rank"],
            )
        ],
        "provenance": validated,
    }


def _combined_required_status(statuses: list[str]) -> str:
    if "FALSE" in statuses:
        return "FALSE"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "TRUE"


def evaluate_joint_task_witness(
    graph: dict[str, Any],
    requirements: dict[str, Any] | str | Path,
    *,
    stage: int | None = None,
    grounding_mode: str = "joint",
) -> dict[str, Any]:
    """Resolve ranked distinct task roles from semantic and geometry evidence."""
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(
            f"grounding_mode must be one of {sorted(GROUNDING_MODES)}"
        )
    config = load_task_requirements(requirements)
    if config.get("_task_schema") != "JOINT_ROLE_GROUNDING":
        if grounding_mode != "geometry-only":
            raise ValueError(
                "Semantic grounding modes require a joint role specification"
            )
        legacy_graph = deepcopy(graph)
        legacy_graph["grounding_mode"] = "geometry-only"
        legacy_config = deepcopy(config)
        legacy_config["_task_schema"] = "GEOMETRY_ONLY"
        return evaluate_task_witness(
            legacy_graph, legacy_config, stage=stage
        )
    if stage is None:
        stage = int(graph.get("stage", -1))
    objects, geometry_edges, relation_edges = _joint_graph_index(graph)
    role_order = sorted(
        config["roles"],
        key=lambda role_name: (
            config["roles"][role_name]["assignment_order"],
            role_name,
        ),
    )
    slots = [
        (role_name, index)
        for role_name in role_order
        for index in range(config["roles"][role_name]["count"])
    ]
    candidate_index: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_evaluations = []
    for role_name in role_order:
        role = config["roles"][role_name]
        for object_id, attributes in sorted(objects.items()):
            semantic = evaluate_semantic_compatibility(attributes, role)
            geometry_edge = geometry_edges.get((object_id, role_name), {})
            geometry = {
                "status": geometry_edge.get("status", "UNKNOWN"),
                "checks": geometry_edge.get("evidence", {}).get(
                    "checks", []
                ),
            }
            semantic_gate = (
                "TRUE"
                if grounding_mode == "geometry-only"
                else semantic["status"]
            )
            geometry_gate = (
                "TRUE"
                if grounding_mode == "semantic-only"
                else geometry["status"]
            )
            status = _combined_required_status(
                [semantic_gate, geometry_gate]
            )
            if semantic_gate == "FALSE":
                decision = "REJECTED_SEMANTIC"
            elif geometry_gate == "FALSE":
                decision = "REJECTED_GEOMETRY"
            elif status == "UNKNOWN":
                decision = "INDETERMINATE"
            else:
                decision = "ROLE_CANDIDATE"
            evaluation = {
                "object_id": object_id,
                "role": role_name,
                "grounding_mode": grounding_mode,
                "semantic": semantic,
                "unary_geometry": geometry,
                "semantic_gate_status": semantic_gate,
                "geometry_gate_status": geometry_gate,
                "status": status,
                "decision": decision,
            }
            candidate_index[(role_name, object_id)] = evaluation
            candidate_evaluations.append(evaluation)

    pairwise_constraints = config["constraints"]["pairwise"]
    assignment_evaluations = []
    valid_assignments = []
    indeterminate_assignments = []
    distinct = bool(config["constraints"]["distinct_objects"])
    object_ids = sorted(objects)
    for choices in product(object_ids, repeat=len(slots)):
        mapping_by_slot = dict(zip(slots, choices))
        selected = {
            role_name: [
                mapping_by_slot[(role_name, index)]
                for index in range(config["roles"][role_name]["count"])
            ]
            for role_name in role_order
        }
        if distinct and len(set(choices)) != len(choices):
            assignment_evaluations.append(
                {
                    "selected_objects": selected,
                    "status": "FALSE",
                    "decision": "REJECTED_DISTINCTNESS",
                    "relation_checks": [],
                }
            )
            continue
        candidate_checks = [
            candidate_index[(role_name, object_id)]
            for (role_name, _index), object_id in mapping_by_slot.items()
        ]
        relation_checks = []
        for constraint in pairwise_constraints:
            for source_id in selected[constraint["from_role"]]:
                for target_id in selected[constraint["to_role"]]:
                    edge = relation_edges.get(
                        (constraint["relation"], source_id, target_id), {}
                    )
                    measured_status = edge.get("status", "UNKNOWN")
                    gate_status = (
                        "TRUE"
                        if grounding_mode == "semantic-only"
                        else measured_status
                    )
                    relation_checks.append(
                        {
                            "relation": constraint["relation"],
                            "from_role": constraint["from_role"],
                            "to_role": constraint["to_role"],
                            "from_object": source_id,
                            "to_object": target_id,
                            "measured_status": measured_status,
                            "status": gate_status,
                            "required_status": "TRUE",
                            "evidence": edge.get("evidence", {}),
                        }
                    )
        statuses = [
            check["status"] for check in candidate_checks
        ] + [check["status"] for check in relation_checks]
        assignment_status = _combined_required_status(statuses)
        failed_semantic = [
            check
            for check in candidate_checks
            if check["semantic_gate_status"] == "FALSE"
        ]
        failed_geometry = [
            check
            for check in candidate_checks
            if check["geometry_gate_status"] == "FALSE"
        ]
        failed_relations = [
            check for check in relation_checks if check["status"] == "FALSE"
        ]
        if failed_semantic:
            decision = "REJECTED_SEMANTIC"
        elif failed_geometry or failed_relations:
            decision = "REJECTED_GEOMETRY"
        elif assignment_status == "UNKNOWN":
            decision = "INDETERMINATE"
        else:
            decision = "VALID"
        assignment = {
            "selected_objects": selected,
            "candidate_checks": candidate_checks,
            "relation_checks": relation_checks,
            "status": assignment_status,
            "decision": decision,
        }
        assignment_evaluations.append(assignment)
        if assignment_status == "TRUE":
            valid_assignments.append(assignment)
        elif assignment_status == "UNKNOWN":
            indeterminate_assignments.append(assignment)

    def selection_key(assignment: dict[str, Any]) -> tuple:
        ranks = []
        confidences = []
        ids = []
        for role_name in role_order:
            for object_id in assignment["selected_objects"][role_name]:
                evaluation = candidate_index[(role_name, object_id)]
                semantic = evaluation["semantic"]
                if grounding_mode != "geometry-only":
                    ranks.append(int(semantic["semantic_rank"]))
                    confidence = _finite_number(semantic.get("confidence"))
                    confidences.append(
                        -confidence if confidence is not None else 0.0
                    )
                ids.append(object_id)
        return (*ranks, *confidences, *ids)

    valid_assignments.sort(key=selection_key)
    selected_assignment = valid_assignments[0] if valid_assignments else None
    status = (
        "COMPLETE"
        if selected_assignment is not None
        else "INDETERMINATE"
        if indeterminate_assignments
        else "INCOMPLETE"
    )
    selected_witness = (
        deepcopy(selected_assignment["selected_objects"])
        if selected_assignment is not None
        else None
    )
    selected_edges = []
    if selected_witness is not None:
        for role_name in role_order:
            for object_id in selected_witness[role_name]:
                semantic = candidate_index[(role_name, object_id)][
                    "semantic"
                ]
                selected_edges.append(
                    {
                        "object_id": object_id,
                        "role": role_name,
                        "semantic_rank": semantic.get("semantic_rank"),
                        "canonical_label": semantic.get(
                            "canonical_label"
                        ),
                        "confidence": semantic.get("confidence"),
                    }
                )
    reason_codes = (
        ["COMPLETE_JOINT_WITNESS_FOUND"]
        if status == "COMPLETE" and grounding_mode == "joint"
        else [f"COMPLETE_{grounding_mode.upper().replace('-', '_')}_ABLATION"]
        if status == "COMPLETE"
        else ["REQUIRED_EVIDENCE_UNKNOWN"]
        if status == "INDETERMINATE"
        else ["NO_VALID_ROLE_ASSIGNMENT"]
    )
    return {
        "schema_version": WITNESS_SCHEMA_VERSION + 1,
        "inference_basis": grounding_mode.upper().replace("-", "_"),
        "grounding_mode": grounding_mode,
        "diagnostic_ablation": grounding_mode != "joint",
        "task_id": config["task_id"],
        "specification_source": config.get("specification_source"),
        "stage": stage,
        "status": status,
        "selected_witness": selected_witness,
        "selected_candidate_edges": selected_edges,
        "selected_pairwise_relations": (
            selected_assignment["relation_checks"]
            if selected_assignment is not None
            else []
        ),
        "role_requirements": {
            role_name: config["roles"][role_name]["count"]
            for role_name in role_order
        },
        "candidate_evaluations": candidate_evaluations,
        "assignment_evaluations": assignment_evaluations,
        "valid_assignment_count": len(valid_assignments),
        "indeterminate_assignment_count": len(
            indeterminate_assignments
        ),
        "reason_codes": reason_codes,
        "selection_policy": config["selection"],
    }
