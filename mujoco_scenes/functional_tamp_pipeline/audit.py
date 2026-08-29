"""Evaluation audit for grounding and plan consistency."""

from __future__ import annotations

from typing import Any

from .models import FunctionalSpecification
from .scene_graph import ObservedSceneGraph


def audit_plan_grounding(
    specification: FunctionalSpecification,
    graph_o: ObservedSceneGraph,
    ground_result: Any,
    plan: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    home_region: str = "countertop",
) -> dict[str, Any]:
    """Audit that A* plan adheres to phi* grounding and causal accessibility."""
    violations: list[str] = []
    grounding_complete = bool(getattr(ground_result, "complete", False))
    if not grounding_complete:
        violations.append("Grounding result is not complete")

    assignment = getattr(ground_result, "assignment", {}) or {}
    operation_bindings = getattr(ground_result, "operation_bindings", {}) or {}

    assigned_object_ids: set[str] = set()
    for role_name, val in assignment.items():
        if isinstance(val, str):
            assigned_object_ids.add(val)
        elif isinstance(val, (list, tuple, set)):
            assigned_object_ids.update(val)

    # 1. Verify all assigned nodes exist in G_O
    all_assignment_nodes_observed = True
    for obj_id in sorted(assigned_object_ids):
        if obj_id not in graph_o.nodes:
            all_assignment_nodes_observed = False
            violations.append(f"Assigned object '{obj_id}' not found in observed scene graph G_O")

    # 2. Verify all required relations in operation bindings are TRUE
    all_required_relations_true = True
    for group_id, bindings in operation_bindings.items():
        # Find group definition if present
        matching_group = next((g for g in specification.operation_groups if g.id == group_id), None)
        required_rels = matching_group.required_relations if matching_group else ("INSERTABLE_IN", "REACHES_BOTTOM")
        for binding in bindings:
            tool_id = binding.get("tool_id")
            target_id = binding.get("target_id")
            for rel in required_rels:
                obs_rel = graph_o.get_relation(rel, tool_id, target_id)
                if obs_rel is None or obs_rel.status != "TRUE":
                    all_required_relations_true = False
                    status_str = obs_rel.status if obs_rel else "MISSING"
                    violations.append(
                        f"Operation binding ({tool_id}, {target_id}) for group '{group_id}' has relation '{rel}' with status '{status_str}' (expected TRUE)"
                    )

    # 3. Plan argument consistency: ensure task objects used in plan come from assigned objects
    plan_uses_only_grounded_task_objects = True
    # Known region/surface identifiers that are not physical manipulable objects
    known_regions = {
        home_region, "serving_area", "countertop", "shared_table", "personal_table_left",
        "personal_table_right", "staging_tray", "work_surface", "D1", "D2", "C1", "C2", "B1",
        "TOOL_CABINET", "WORKBENCH_DRAWER", "DRILL_PRESS_CABINET",
    }
    for action in plan:
        op = action.get("operator", "")
        args = action.get("arguments", [])
        for arg in args:
            if arg not in known_regions and arg not in assigned_object_ids and not arg.startswith("pos_") and not arg.startswith("slot_"):
                plan_uses_only_grounded_task_objects = False
                violations.append(f"Action {op}({', '.join(args)}) uses ungrounded object '{arg}'")

    # 4. Preparation accessibility check: POUR and STIR targets must be at home_region
    preparation_accessibility_valid = True
    object_locations: dict[str, str] = {}
    for node_id, node in graph_o.nodes.items():
        if node.source_region:
            object_locations[node_id] = node.source_region
        elif node.region:
            object_locations[node_id] = node.region

    held_obj: str | None = None
    for action in plan:
        op = action.get("operator", "")
        args = action.get("arguments", [])
        if op == "PICK":
            obj = args[0]
            held_obj = obj
            object_locations.pop(obj, None)
        elif op == "PLACE":
            obj, dest = args[0], args[1]
            held_obj = None
            object_locations[obj] = dest
        elif op == "POUR":
            _src, tgt = args[0], args[1]
            tgt_loc = object_locations.get(tgt)
            if tgt_loc != home_region:
                preparation_accessibility_valid = False
                violations.append(f"POUR into target '{tgt}' while target is at '{tgt_loc}' (expected '{home_region}')")
        elif op == "STIR":
            _tool, tgt = args[0], args[1]
            tgt_loc = object_locations.get(tgt)
            if tgt_loc != home_region:
                preparation_accessibility_valid = False
                violations.append(f"STIR target '{tgt}' while target is at '{tgt_loc}' (expected '{home_region}')")

    return {
        "grounding_complete": grounding_complete,
        "role_assignments": assignment,
        "operation_bindings": operation_bindings,
        "all_assignment_nodes_observed": all_assignment_nodes_observed,
        "all_required_relations_true": all_required_relations_true,
        "plan_uses_only_grounded_task_objects": plan_uses_only_grounded_task_objects,
        "preparation_accessibility_valid": preparation_accessibility_valid,
        "plan_replay_valid": bool(plan) and len(violations) == 0,
        "violations": violations,
    }

