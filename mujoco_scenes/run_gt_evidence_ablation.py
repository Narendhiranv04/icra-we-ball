"""Privileged GT graph evidence ablation with three-level plan evaluation."""
from __future__ import annotations

import argparse, csv, json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import KitchenPlanningCompiler
from mujoco_scenes.functional_tamp_pipeline.domains.workshop import SURFACE, TARGET, WorkshopPlanningCompiler
from mujoco_scenes.functional_tamp_pipeline.grounding import check_semantic_role_compatibility, evaluate_node_for_role, ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import GraphGroundingResult
from mujoco_scenes.functional_tamp_pipeline.oracle_evidence import build_oracle_graph, intended_outcome, kitchen_variants, living_room_variants, workshop_variants
from mujoco_scenes.functional_tamp_pipeline.planning import plan_with_common_astar
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedSceneGraph
from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem

DOMAINS = ("kitchen", "living_room", "workshop")
COMPONENT_MASKS = {
    "semantic_only": ("semantic",), "unary_only": ("unary",),
    "binary_only": ("binary",), "no_binary": ("semantic", "unary"),
    "no_unary": ("semantic", "binary"), "no_semantic": ("unary", "binary"),
    "full": ("semantic", "unary", "binary"),
}
KITCHEN_FIXED_SOURCE_ROLES = ("coffee_source", "water_source")

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _variants(domain: str) -> tuple[str, ...]:
    return tuple({"kitchen": kitchen_variants, "living_room": living_room_variants, "workshop": workshop_variants}[domain]())

def _as_list(value: Any) -> list[str]:
    return [] if value is None else ([str(value)] if isinstance(value, str) else [str(x) for x in value])

def _fixed_kitchen_sources(specification, graph: ObservedSceneGraph) -> dict[str, str] | None:
    fixed = {}
    for name in KITCHEN_FIXED_SOURCE_ROLES:
        role = specification.nodes[name]
        candidates = [node_id for node_id, node in graph.nodes.items()
                      if node.entity_kind == role.entity_kind and check_semantic_role_compatibility(node, role.semantic_categories)[0] == "TRUE"]
        if len(candidates) != 1:
            return None
        fixed[name] = candidates[0]
    return fixed

def _ground(domain: str, specification, graph: ObservedSceneGraph, components: tuple[str, ...]) -> GraphGroundingResult:
    context = {"search_exhausted": True, "evidence_mode": "joint", "evidence_components": components}
    if domain != "kitchen":
        return ground_graph(specification, graph, context)
    fixed = _fixed_kitchen_sources(specification, graph)
    if fixed is None:
        return GraphGroundingResult(status="INFEASIBLE", complete=False, assignment=None,
                                    missing_roles=KITCHEN_FIXED_SOURCE_ROLES,
                                    unresolved_constraints=("BENCHMARK_SOURCE_IDENTITY_MISSING",))
    nodes = {name: role for name, role in specification.nodes.items() if name not in fixed}
    ordinary_spec = replace(specification, nodes=nodes,
        relations=tuple(r for r in specification.relations if r.subject_role in nodes and r.object_role in nodes))
    source_ids = set(fixed.values())
    ordinary_graph = ObservedSceneGraph.from_dict(graph.to_dict())
    for source_id in source_ids:
        ordinary_graph.nodes.pop(source_id, None)
    ordinary_graph.relations = {key: relation for key, relation in ordinary_graph.relations.items()
                                if relation.subject_id not in source_ids and relation.object_id not in source_ids}
    result = ground_graph(ordinary_spec, ordinary_graph, context)
    if not result.complete:
        return result
    return replace(result, assignment={**(result.assignment or {}), **fixed}, evidence={
        **result.evidence, "benchmark_fixed_source_assignments": fixed,
        "benchmark_fixed_source_identity_is_not_masked_evidence": True,
    })

def _validate_roles(specification, result, graph: ObservedSceneGraph) -> dict[str, Any]:
    total = sum(role.minimum_count for role in specification.nodes.values())
    selected = sem_valid = unary_valid = joint_valid = 0
    details = []
    for name, role in specification.nodes.items():
        ids = _as_list((result.assignment or {}).get(name))
        for index in range(role.minimum_count):
            object_id = ids[index] if index < len(ids) else None
            sem = unary = joint = "FALSE"
            node = graph.get_node(object_id) if object_id else None
            if node:
                selected += 1
                sem = check_semantic_role_compatibility(node, role.semantic_categories)[0]
                unary = evaluate_node_for_role(node, role, evidence_components=("unary",))[0]
                joint = evaluate_node_for_role(node, role, evidence_components=("semantic", "unary"))[0]
            sem_valid += sem == "TRUE"; unary_valid += unary == "TRUE"; joint_valid += joint == "TRUE"
            details.append({"role": name, "slot_index": index, "selected_object_id": object_id,
                            "semantic_status": sem, "unary_geometry_status": unary,
                            "gt_role_status": joint, "gt_valid": joint == "TRUE"})
    exact = bool(result.complete and selected == total == joint_valid)
    return {"per_role_gt_validation": details, "role_slots_total": total,
            "role_slots_selected": selected, "semantic_valid_role_slots": sem_valid,
            "geometric_valid_role_slots": unary_valid, "joint_valid_role_slots": joint_valid,
            "role_slots_gt_valid": joint_valid,
            "role_slot_validity_pct": round(100 * joint_valid / total, 2) if total else 100.0,
            "exact_role_grounding_success": exact}

def _validate_bindings(specification, result, graph: ObservedSceneGraph) -> dict[str, Any]:
    managed = {
        (group.tool_role, predicate, group.target_role)
        for group in specification.operation_groups
        for predicate in group.required_relations
    } | {
        (group.tool_role, predicate, group.context_role)
        for group in specification.operation_groups if group.context_role
        for predicate in group.context_relations
    }
    explicit = [relation for relation in specification.relations
                if (relation.subject_role, relation.predicate, relation.object_role) not in managed]
    total = sum(g.required_target_count for g in specification.operation_groups) + sum(
        specification.nodes[r.subject_role].minimum_count * specification.nodes[r.object_role].minimum_count
        for r in explicit
    )
    selected = valid = 0; details = []
    for group in specification.operation_groups:
        bindings = list(result.operation_bindings.get(group.id, []))
        for index in range(group.required_target_count):
            binding = bindings[index] if index < len(bindings) else None
            checks = []
            if binding:
                selected += 1
                for predicate in group.required_relations:
                    relation = graph.get_relation(predicate, binding.get("tool_id"), binding.get("target_id"))
                    checks.append({"predicate": predicate, "subject_id": binding.get("tool_id"),
                                   "object_id": binding.get("target_id"), "status": relation.status if relation else "UNKNOWN"})
                if group.context_role:
                    context_id = binding.get("context", {}).get(group.context_role)
                    for predicate in group.context_relations:
                        relation = graph.get_relation(predicate, binding.get("tool_id"), context_id)
                        checks.append({"predicate": predicate, "subject_id": binding.get("tool_id"),
                                       "object_id": context_id, "status": relation.status if relation else "UNKNOWN"})
            ok = bool(checks) and all(check["status"] == "TRUE" for check in checks)
            valid += ok
            details.append({"group": group.id, "binding_index": index, "binding": binding,
                            "relation_checks": checks, "gt_valid": ok})
    for relation in explicit:
        subjects = _as_list((result.assignment or {}).get(relation.subject_role))
        objects = _as_list((result.assignment or {}).get(relation.object_role))
        required = (specification.nodes[relation.subject_role].minimum_count
                    * specification.nodes[relation.object_role].minimum_count)
        pairs = [(subject, object_id) for subject in subjects for object_id in objects if subject != object_id]
        for index in range(required):
            pair = pairs[index] if index < len(pairs) else None
            status = "UNKNOWN"
            if pair:
                selected += 1
                observed = graph.get_relation(relation.predicate, pair[0], pair[1])
                status = observed.status if observed else "UNKNOWN"
            ok = status == ("TRUE" if relation.expected else "FALSE")
            valid += ok
            details.append({
                "group": f"explicit:{relation.predicate}", "binding_index": index,
                "binding": None if pair is None else {
                    "tool_id": pair[0], "target_id": pair[1], "context": {},
                },
                "relation_checks": [{
                    "predicate": relation.predicate,
                    "subject_id": pair[0] if pair else None,
                    "object_id": pair[1] if pair else None,
                    "status": status, "expected": relation.expected,
                }],
                "gt_valid": ok,
            })
    exact = bool(result.complete and selected == total == valid)
    return {"per_binding_gt_validation": details, "operation_bindings_total": total,
            "operation_bindings_selected": selected, "operation_bindings_gt_valid": valid,
            "ground_truth_valid_operation_bindings": valid,
            "operation_binding_validity_pct": round(100 * valid / total, 2) if total else 100.0,
            "exact_operation_binding_success": exact}

def _action(name, arguments, positive, add, delete) -> SymbolicAction:
    return SymbolicAction(name, arguments, frozenset(positive), frozenset(), frozenset(add), frozenset(delete))

class LivingRoomAblationPlanningCompiler:
    def compile_problem(self, assignment: dict[str, Any], context: dict[str, Any]) -> SymbolicProblem:
        destinations = {str(b["target_id"]): str(b["tool_id"])
                        for b in context["operation_bindings"].get("personal_support_group", [])}
        shared, remote = _as_list(assignment.get("SHARED_REMOTE_REGION")), _as_list(assignment.get("REMOTE"))
        if shared and remote: destinations[remote[0]] = shared[0]
        initial = {("hand_empty",)} | {("available", obj) for obj in destinations}
        actions = []
        for obj, region in sorted(destinations.items()):
            actions += [_action("PICK", (obj,), {("hand_empty",), ("available", obj)}, {("holding", obj)}, {("hand_empty",), ("available", obj)}),
                        _action("PLACE", (obj, region), {("holding", obj)}, {("hand_empty",), ("on", obj, region)}, {("holding", obj)})]
        return SymbolicProblem(frozenset(initial), frozenset(("on", o, r) for o, r in destinations.items()), tuple(actions))

def _kitchen_state(specification, result, graph: ObservedSceneGraph) -> dict[str, Any]:
    assignment = result.assignment or {}; coffee = _as_list(assignment.get("coffee_container")); soup = _as_list(assignment.get("soup_container"))
    cb = result.operation_bindings.get("coffee_stirring", []); sb = result.operation_bindings.get("soup_serving", [])
    sources = {name: str(assignment[name]) for name in KITCHEN_FIXED_SOURCE_ROLES}
    objects = set(coffee + soup + list(sources.values())) | {str(b["tool_id"]) for b in cb + sb}
    records = {}
    for object_id in objects:
        node = graph.get_node(object_id); region = node.source_region if node else "INITIAL"
        records[object_id] = {"location": {"region_id": "countertop" if region == "INITIAL" else str(region)}}
    return {"objects": records, "regions": {"countertop": {"open": True}},
            "role_assignments": {"coffee_targets": coffee, "soup_targets": soup, "source_roles": sources,
                                 "coffee_stirring": cb, "soup_serving": sb},
            "capabilities": {"source_contains": [[sources["coffee_source"], "coffee"], [sources["water_source"], "water"]],
                             "initial_target_contents": [[target, "soup"] for target in soup],
                             "can_stir": [[b["tool_id"], b["target_id"]] for b in cb],
                             "assigned_soup_utensil": [[b["tool_id"], b["target_id"]] for b in sb]},
            "requirements": specification.metadata.get("symbolic_task", {})}

def _plan(domain: str, specification, result, graph: ObservedSceneGraph) -> dict[str, Any]:
    if not result.complete or not result.assignment:
        return {"plan_generated": False, "plan_generation_failure_reason": "GROUNDING_INCOMPLETE",
                "generated_action_sequence": [], "plan_replay_valid": False, "planner_status": "NOT_RUN", "planner_runtime_ms": 0.0}
    start = perf_counter()
    try:
        if domain == "kitchen":
            compiler, assignment, context = KitchenPlanningCompiler(), result.assignment, {"compiled_observed_state": _kitchen_state(specification, result, graph)}
        elif domain == "living_room":
            compiler, assignment, context = LivingRoomAblationPlanningCompiler(), result.assignment, {"operation_bindings": result.operation_bindings}
        else:
            compiler, assignment = WorkshopPlanningCompiler(), dict(result.assignment)
            for role in ("driver", "fastener"):
                assignment[f"{role}_source"] = str(graph.get_node(str(result.assignment[role])).source_region or SURFACE)
            context = {"opened_regions": [assignment["driver_source"], assignment["fastener_source"]]}
        planned = plan_with_common_astar(compiler, assignment, context)
        return {"plan_generated": True, "plan_generation_failure_reason": None,
                "generated_action_sequence": list(planned.actions), "plan_replay_valid": planned.validation.get("status") == "VALID",
                "plan_replay_result": planned.validation, "planner_status": "SUCCESS",
                "planner_statistics": planned.search.statistics, "planner_runtime_ms": round((perf_counter()-start)*1000, 3)}
    except Exception as exc:
        return {"plan_generated": False, "plan_generation_failure_reason": f"{type(exc).__name__}: {exc}",
                "generated_action_sequence": [], "plan_replay_valid": False, "planner_status": "FAILED",
                "planner_runtime_ms": round((perf_counter()-start)*1000, 3)}

def _task_actions_valid(domain: str, result, actions: list[dict[str, Any]]) -> bool:
    actual = {(a["operator"], tuple(a["arguments"])) for a in actions}; assignment = result.assignment or {}
    if domain == "kitchen":
        expected = [("STIR", (b["tool_id"], b["target_id"])) for b in result.operation_bindings.get("coffee_stirring", [])]
        expected += [("PLACE", (b["tool_id"], b["target_id"])) for b in result.operation_bindings.get("soup_serving", [])]
        expected += [("PLACE", (target, "serving_area")) for target in _as_list(assignment.get("coffee_container")) + _as_list(assignment.get("soup_container"))]
    elif domain == "living_room":
        expected = [("PLACE", (b["target_id"], b["tool_id"])) for b in result.operation_bindings.get("personal_support_group", [])]
        shared, remote = _as_list(assignment.get("SHARED_REMOTE_REGION")), _as_list(assignment.get("REMOTE"))
        if shared and remote: expected.append(("PLACE", (remote[0], shared[0])))
    else:
        expected = [("SCREW", (assignment.get("driver"), assignment.get("fastener"), TARGET))]
    return bool(expected) and all(item in actual for item in expected)

def _failure(row: dict[str, Any]) -> tuple[str | None, str | None]:
    if row["intended_outcome"] != "FEASIBLE": return None, None
    checks = [(not row["grounding_complete"], "GROUNDING_INCOMPLETE", row["grounding_status"]),
              (not row["exact_role_grounding_success"], "ROLE_ASSIGNMENT_INVALID", "Selected role fails full GT semantic/unary validation"),
              (not row["exact_operation_binding_success"], "PAIR_BINDING_INVALID", "Selected binding fails full GT binary validation"),
              (not row["plan_generated"], "PLANNING_FAILED", row["plan_generation_failure_reason"]),
              (not row["plan_replay_valid"], "PLAN_REPLAY_INVALID", "Independent replay rejected plan"),
              (not row["gt_task_plan_valid"], "GT_TASK_PLAN_INVALID", "Plan does not satisfy GT-grounded task")]
    return next(((level, reason) for failed, level, reason in checks if failed), (None, None))

def evaluate_one(domain: str, variant: str, condition: str, *, specification=None, graph=None) -> dict[str, Any]:
    components = COMPONENT_MASKS[condition]; specification = specification or GTSpecProvider().provide(domain, "")
    graph = graph or build_oracle_graph(domain, variant, specification); start = perf_counter()
    result = _ground(domain, specification, graph, components); grounding_ms = (perf_counter()-start)*1000
    expected = intended_outcome(domain, variant); predicted = "FEASIBLE" if result.complete else "INFEASIBLE"
    roles = _validate_roles(specification, result, graph); bindings = _validate_bindings(specification, result, graph)
    planning = _plan(domain, specification, result, graph)
    structural = _task_actions_valid(domain, result, planning["generated_action_sequence"]) if planning["plan_generated"] else False
    gt_valid = bool(planning["plan_replay_valid"] and roles["exact_role_grounding_success"] and bindings["exact_operation_binding_success"] and structural)
    exact = bool(expected == "FEASIBLE" and result.complete and gt_valid)
    selected_bindings = {name: list(values) for name, values in result.operation_bindings.items()}
    explicit_bindings = [
        {"relation": item["relation_checks"][0]["predicate"], **item["binding"]}
        for item in bindings["per_binding_gt_validation"]
        if item["group"].startswith("explicit:") and item["binding"] is not None
    ]
    if explicit_bindings:
        selected_bindings["explicit_relations"] = explicit_bindings
    row = {"domain": domain, "variant": variant, "condition": condition,
           "enabled_evidence_components": list(components), "evidence_components": list(components),
           "intended_outcome": expected, "predicted_grounding_outcome": predicted, "predicted_outcome": predicted,
           "outcome_correct": predicted == expected, "grounding_status": result.status, "grounding_complete": result.complete,
           "grounding_runtime_ms": round(grounding_ms, 3), "runtime_ms": round(grounding_ms, 3),
           **roles, **bindings, **planning, "gt_task_plan_valid": gt_valid,
           "gt_task_plan_validation": {"selected_task_actions_present": structural,
                                       "oracle_roles_valid": roles["exact_role_grounding_success"],
                                       "oracle_bindings_valid": bindings["exact_operation_binding_success"]},
           "exact_symbolic_task_success": exact, "selected_assignment": result.assignment, "assignment": result.assignment,
           "selected_operation_bindings": selected_bindings, "operation_bindings": result.operation_bindings,
           "missing_roles": list(result.missing_roles), "unsatisfied_relations": list(result.unsatisfied_relations),
           "unresolved_constraints": list(result.unresolved_constraints), "grounding_result": result.to_dict(), "graph": graph.to_dict()}
    row["first_failure_level"], row["failure_reason"] = _failure(row)
    return row

def _rate(rows: Iterable[dict[str, Any]], predicate) -> float | None:
    rows = list(rows); return round(100 * sum(bool(predicate(r)) for r in rows) / len(rows), 2) if rows else None

def _summary(domain: str, condition: str, group: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [r for r in group if r["intended_outcome"] == "FEASIBLE"]; infeasible = [r for r in group if r["intended_outcome"] == "INFEASIBLE"]
    role_total = sum(r["role_slots_total"] for r in feasible); binding_total = sum(r["operation_bindings_total"] for r in feasible)
    return {"domain": domain, "condition": condition, "enabled_evidence_components": "+".join(group[0]["enabled_evidence_components"]),
            "variants": len(group), "feasible_variants": len(feasible), "infeasible_variants": len(infeasible),
            "grounding_completion_pct": _rate(feasible, lambda r:r["grounding_complete"]),
            "role_slot_validity_pct": round(100*sum(r["role_slots_gt_valid"] for r in feasible)/role_total,2) if role_total else None,
            "exact_role_grounding_success_pct": _rate(feasible, lambda r:r["exact_role_grounding_success"]),
            "operation_binding_validity_pct": round(100*sum(r["operation_bindings_gt_valid"] for r in feasible)/binding_total,2) if binding_total else None,
            "exact_operation_binding_success_pct": _rate(feasible, lambda r:r["exact_operation_binding_success"]),
            "plan_generation_pct": _rate(feasible, lambda r:r["plan_generated"]),
            "plan_replay_valid_pct": _rate(feasible, lambda r:r["plan_replay_valid"]),
            "gt_task_valid_plan_pct": _rate(feasible, lambda r:r["gt_task_plan_valid"]),
            "exact_symbolic_task_success_pct": _rate(feasible, lambda r:r["exact_symbolic_task_success"]),
            "infeasible_rejection_pct": _rate(infeasible, lambda r:not r["grounding_complete"]),
            "false_completion_pct": _rate(infeasible, lambda r:r["grounding_complete"]),
            "outcome_agreement_pct": _rate(group, lambda r:r["outcome_correct"]),
            "mean_grounding_ms": round(sum(r["grounding_runtime_ms"] for r in group)/len(group),3),
            "mean_planning_ms": round(sum(r["planner_runtime_ms"] for r in group)/len(group),3)}

def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output=[]
    for domain in (*DOMAINS, "all"):
        source = rows if domain == "all" else [r for r in rows if r["domain"] == domain]
        for condition in COMPONENT_MASKS:
            group=[r for r in source if r["condition"] == condition]
            if group: output.append(_summary(domain, condition, group))
    return output

def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer=csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--domains",default=",".join(DOMAINS)); parser.add_argument("--variants")
    parser.add_argument("--component-masks",default="all"); parser.add_argument("--output-root",type=Path); args=parser.parse_args()
    domains=tuple(x.strip() for x in args.domains.split(",") if x.strip())
    if any(x not in DOMAINS for x in domains): raise ValueError(f"domains must be selected from {DOMAINS}")
    requested=None if args.variants is None else {x.strip() for x in args.variants.split(",") if x.strip()}
    masks=tuple(COMPONENT_MASKS) if args.component_masks=="all" else tuple(x.strip() for x in args.component_masks.split(",") if x.strip())
    if set(masks)-set(COMPONENT_MASKS): raise ValueError(f"unknown component masks: {sorted(set(masks)-set(COMPONENT_MASKS))}")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); output=args.output_root or Path("runs/gt_evidence_ablation")/f"three_level_{stamp}"
    output.mkdir(parents=True,exist_ok=False); rows=[]
    for domain in domains:
        variants=[v for v in _variants(domain) if requested is None or v in requested]
        if not variants: raise ValueError(f"No {domain} variants matched --variants")
        specification=GTSpecProvider().provide(domain,"")
        for variant in variants:
            graph=build_oracle_graph(domain,variant,specification)
            for condition in masks:
                print(f"[gt-evidence] {domain} {variant} {condition}",flush=True)
                row=evaluate_one(domain,variant,condition,specification=specification,graph=graph); rows.append(row)
                _write_json(output/domain/variant/f"{condition}.json",row)
    compact=[{k:v for k,v in row.items() if k not in {"graph","grounding_result"}} for row in rows]; summary=summarize(rows)
    failures=[{k:r[k] for k in ("domain","variant","condition","first_failure_level","failure_reason","selected_assignment","selected_operation_bindings","per_role_gt_validation","per_binding_gt_validation")}
              for r in rows if r["intended_outcome"]=="FEASIBLE" and not r["exact_symbolic_task_success"]]
    _write_json(output/"results.json",compact); _write_json(output/"summary.json",{"schema_version":3,"kind":"PRIVILEGED_GT_THREE_LEVEL_EVIDENCE_ABLATION","rows":summary})
    _write_csv(output/"summary.csv",summary); _write_json(output/"failure_breakdown.json",failures); print(json.dumps(summary,indent=2),flush=True); return 0

if __name__ == "__main__": raise SystemExit(main())
