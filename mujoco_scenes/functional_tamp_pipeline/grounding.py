"""Canonical constraint-aware graph grounding boundary phi : G_F -> G_O."""

from __future__ import annotations

from itertools import combinations, product
from typing import Any, Sequence

from .models import FunctionalRequirementGraph, FunctionalRole, GraphGroundingResult, NumericConstraint
from .scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph


def _extract_property_value(node: ObservedNode, property_name: str) -> float | None:
    """Extract a numerical property value from an observed node."""
    val = node.unary_properties.get(property_name)
    if val is not None:
        if isinstance(val, dict) and "value" in val:
            val = val["value"]
        if isinstance(val, (int, float)):
            return float(val)

    geom = node.geometry
    if property_name in geom:
        val = geom[property_name]
        if isinstance(val, dict) and "value" in val:
            val = val["value"]
        if isinstance(val, (int, float)):
            return float(val)

    props = geom.get("properties", {})
    if property_name in props:
        val = props[property_name]
        if isinstance(val, dict) and "value" in val:
            val = val["value"]
        if isinstance(val, (int, float)):
            return float(val)

    return None


def _check_unary_predicate(node: ObservedNode, predicate_name: str) -> str:
    """Check if an observed node satisfies a unary predicate. Returns 'TRUE', 'FALSE', or 'UNKNOWN'."""
    record = node.unary_predicates.get(predicate_name)
    if record is not None:
        if isinstance(record, dict):
            status = record.get("status", "UNKNOWN")
            if status in {"TRUE", "FALSE", "UNKNOWN"}:
                return status
            if "value" in record and record["value"] is not None:
                return "TRUE" if bool(record["value"]) else "FALSE"
        elif isinstance(record, bool):
            return "TRUE" if record else "FALSE"
        elif isinstance(record, str) and record.upper() in {"TRUE", "FALSE", "UNKNOWN"}:
            return record.upper()

    record = node.unary_properties.get(predicate_name)
    if record is not None:
        if isinstance(record, dict):
            status = record.get("status", "UNKNOWN")
            if status in {"TRUE", "FALSE", "UNKNOWN"}:
                return status
            if "value" in record and record["value"] is not None:
                return "TRUE" if bool(record["value"]) else "FALSE"
        elif isinstance(record, bool):
            return "TRUE" if record else "FALSE"
        elif isinstance(record, str) and record.upper() in {"TRUE", "FALSE", "UNKNOWN"}:
            return record.upper()

    geom_predicates = node.geometry.get("predicates", {})
    if predicate_name in geom_predicates:
        record = geom_predicates[predicate_name]
        if isinstance(record, dict):
            status = record.get("status", "UNKNOWN")
            if status in {"TRUE", "FALSE", "UNKNOWN"}:
                return status
        elif isinstance(record, bool):
            return "TRUE" if record else "FALSE"
        elif isinstance(record, str) and record.upper() in {"TRUE", "FALSE", "UNKNOWN"}:
            return record.upper()

    return "UNKNOWN"


def _check_semantic_category(node: ObservedNode, accepted_categories: Sequence[str]) -> tuple[str, str | None]:
    """Check if an observed node matches accepted semantic categories.
    Returns (status, matched_category) where status in {'TRUE', 'FALSE', 'UNKNOWN'}.
    """
    if not accepted_categories:
        return "TRUE", node.canonical_category

    accepted_set = {cat.strip().lower().replace(" ", "_") for cat in accepted_categories}

    if node.canonical_category:
        norm_canonical = node.canonical_category.strip().lower().replace(" ", "_")
        if norm_canonical in accepted_set:
            return "TRUE", node.canonical_category

    belief = node.semantic_labels
    if belief:
        status = belief.get("status")
        if status == "UNKNOWN":
            return "UNKNOWN", None

        for key in ("canonical_label", "evaluated_label", "predicted_label", "label"):
            label = belief.get(key)
            if label:
                norm_label = str(label).strip().lower().replace(" ", "_")
                if norm_label in accepted_set:
                    return "TRUE", str(label)

        support = belief.get("label_supporting_view_count", {})
        for label, count in support.items():
            if count > 0:
                norm_label = str(label).strip().lower().replace(" ", "_")
                if norm_label in accepted_set:
                    return "TRUE", str(label)

        if status == "SUPPORTED" or belief.get("canonical_label"):
            return "FALSE", None

    return "UNKNOWN", None


def evaluate_node_for_role(node: ObservedNode, role: FunctionalRole) -> tuple[str, dict[str, Any]]:
    """Evaluate whether an observed node satisfies a single functional role's local requirements.
    Returns (status, diagnostic_details).
    """
    details: dict[str, Any] = {
        "instance_id": node.instance_id,
        "role_name": role.name,
        "checks": {},
    }

    # 1. Entity kind check
    if role.entity_kind and node.entity_kind != role.entity_kind:
        details["checks"]["entity_kind"] = {
            "expected": role.entity_kind,
            "actual": node.entity_kind,
            "status": "FALSE",
        }
        return "FALSE", details

    # 2. Semantic category check
    if role.semantic_categories:
        sem_status, matched_cat = _check_semantic_category(node, role.semantic_categories)
        details["checks"]["semantic_categories"] = {
            "expected": list(role.semantic_categories),
            "matched": matched_cat,
            "status": sem_status,
        }
        if sem_status == "FALSE":
            return "FALSE", details

    # 3. Unary predicates check
    for pred in role.unary_predicates:
        pred_status = _check_unary_predicate(node, pred)
        details["checks"][f"unary_predicate_{pred}"] = {
            "predicate": pred,
            "status": pred_status,
        }
        if pred_status == "FALSE":
            return "FALSE", details

    # 4. Numeric constraints check
    for constraint in role.numeric_constraints:
        val = _extract_property_value(node, constraint.property_name)
        if val is None:
            details["checks"][f"numeric_{constraint.property_name}"] = {
                "constraint": constraint.to_dict(),
                "actual_value": None,
                "status": "UNKNOWN",
            }
        else:
            passed = constraint.matches(val)
            status = "TRUE" if passed else "FALSE"
            details["checks"][f"numeric_{constraint.property_name}"] = {
                "constraint": constraint.to_dict(),
                "actual_value": val,
                "status": status,
            }
            if not passed:
                return "FALSE", details

    statuses = [c["status"] for c in details["checks"].values()]
    if "FALSE" in statuses:
        return "FALSE", details
    if "UNKNOWN" in statuses:
        return "UNKNOWN", details
    return "TRUE", details


def ground_graph(
    graph_f: FunctionalRequirementGraph,
    graph_o: ObservedSceneGraph,
    domain_context: dict[str, Any] | None = None,
) -> GraphGroundingResult:
    """Perform constraint-aware graph grounding phi : G_F -> G_O."""
    context = domain_context or {}
    roles = graph_f.nodes

    # Step 1: Find candidate nodes for each role
    role_candidates: dict[str, list[str]] = {}
    unresolved_role_candidates: dict[str, list[str]] = {}
    evaluations: dict[tuple[str, str], dict[str, Any]] = {}

    for role_name, role in roles.items():
        candidates: list[str] = []
        unresolved: list[str] = []
        for instance_id, node in sorted(graph_o.nodes.items()):
            status, details = evaluate_node_for_role(node, role)
            evaluations[(role_name, instance_id)] = details
            if status == "TRUE":
                candidates.append(instance_id)
            elif status == "UNKNOWN":
                unresolved.append(instance_id)

        role_candidates[role_name] = candidates
        unresolved_role_candidates[role_name] = unresolved

    missing_roles = tuple(
        role_name for role_name, cands in role_candidates.items()
        if len(cands) < roles[role_name].count
    )

    role_names = sorted(roles.keys())
    role_combos = []
    for r_name in role_names:
        role = roles[r_name]
        cands = role_candidates[r_name]
        if len(cands) < role.count:
            role_combos.append([])
        else:
            role_combos.append(list(combinations(cands, role.count)))

    if any(len(combos) == 0 for combos in role_combos):
        has_unresolved = any(
            len(unresolved_role_candidates[r]) > 0 for r in missing_roles
        )
        status = "INCOMPLETE" if has_unresolved or missing_roles else "INFEASIBLE"
        return GraphGroundingResult(
            status=status,
            complete=False,
            assignment=None,
            missing_roles=missing_roles,
            unsatisfied_relations=(),
            unresolved_constraints=missing_roles,
            evidence={"candidate_evaluations": {f"{k[0]}:{k[1]}": v for k, v in evaluations.items()}},
        )

    unsatisfied_relations_recorded: list[dict[str, Any]] = []
    valid_assignments: list[dict[str, Any]] = []

    for combo in product(*role_combos):
        assignment_map: dict[str, Any] = {}
        used_instances: set[str] = set()
        conflict = False

        for r_idx, r_name in enumerate(role_names):
            role = roles[r_name]
            selected_for_role = combo[r_idx]

            if not role.reusable and not role.shared:
                for inst_id in selected_for_role:
                    if inst_id in used_instances:
                        conflict = True
                        break
                    used_instances.add(inst_id)
            if conflict:
                break

            assignment_map[r_name] = (
                selected_for_role[0] if role.count == 1 else list(selected_for_role)
            )

        if conflict:
            continue

        all_relations_satisfied = True
        for relation in graph_f.relations:
            subj_role = relation.subject_role
            obj_role = relation.object_role
            predicate = relation.predicate

            if subj_role not in assignment_map or obj_role not in assignment_map:
                continue

            subj_instances = (
                [assignment_map[subj_role]]
                if isinstance(assignment_map[subj_role], str)
                else assignment_map[subj_role]
            )
            obj_instances = (
                [assignment_map[obj_role]]
                if isinstance(assignment_map[obj_role], str)
                else assignment_map[obj_role]
            )

            for s_id in subj_instances:
                for o_id in obj_instances:
                    if s_id == o_id:
                        continue
                    obs_rel = graph_o.get_relation(predicate, s_id, o_id)
                    rel_status = obs_rel.status if obs_rel else "UNKNOWN"
                    if rel_status != "TRUE":
                        all_relations_satisfied = False
                        unsatisfied_relations_recorded.append({
                            "subject_role": subj_role,
                            "subject_id": s_id,
                            "predicate": predicate,
                            "object_role": obj_role,
                            "object_id": o_id,
                            "status": rel_status,
                            "expected": relation.expected,
                        })
                        break
                if not all_relations_satisfied:
                    break
            if not all_relations_satisfied:
                break

        if all_relations_satisfied:
            valid_assignments.append(assignment_map)

    if valid_assignments:
        valid_assignments.sort(key=lambda a: sorted(str(v) for v in a.values()))
        chosen = valid_assignments[0]
        return GraphGroundingResult(
            status="COMPLETE",
            complete=True,
            assignment=chosen,
            missing_roles=(),
            unsatisfied_relations=(),
            unresolved_constraints=(),
            evidence={"valid_assignment_count": len(valid_assignments)},
        )

    return GraphGroundingResult(
        status="INFEASIBLE" if unsatisfied_relations_recorded else "INCOMPLETE",
        complete=False,
        assignment=None,
        missing_roles=(),
        unsatisfied_relations=tuple(unsatisfied_relations_recorded),
        unresolved_constraints=tuple(
            dict.fromkeys(r["predicate"] for r in unsatisfied_relations_recorded)
        ),
        evidence={"unsatisfied_relations": unsatisfied_relations_recorded},
    )
