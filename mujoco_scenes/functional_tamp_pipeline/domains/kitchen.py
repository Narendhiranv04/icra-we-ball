"""Kitchen adapter around existing observed-state grounding and planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from mujoco_scenes.scene_loader import KitchenScene
from mujoco_scenes.sequential_inspection import run_sequential_inspection
from mujoco_scenes.symbolic_planning import (
    KitchenSymbolicProblem, compile_observed_symbolic_state,
)
from mujoco_scenes.symbolic_planning_core import SymbolicAction, SymbolicProblem

from ..models import FunctionalSpecification, PipelineResult
from ..outcome_classifier import classify_pipeline_outcome, complete_planning_contract
from ..planning import plan_with_common_astar


TASK = (
    "Prepare and serve one coffee and one soup for each of two people. "
    "Make each coffee using coffee and water and stir it before serving. "
    "Serve each soup bowl with its own suitable eating utensil."
)
LOCAL_YOLO_WORLD = (
    Path(__file__).resolve().parents[3]
    / "semantic_model_cache" / "yolov8m-worldv2.pt"
)


def _action(
    name: str,
    arguments: tuple[str, ...],
    positive: set[tuple[str, ...]],
    add: set[tuple[str, ...]],
    delete: set[tuple[str, ...]],
) -> SymbolicAction:
    return SymbolicAction(
        name=name, arguments=arguments,
        positive_preconditions=frozenset(positive),
        negative_preconditions=frozenset(),
        add_effects=frozenset(add), delete_effects=frozenset(delete),
    )


class KitchenPlanningCompiler:
    """Translate the existing observed Kitchen witness into common STRIPS."""

    def compile_problem(
        self, assignment: dict[str, Any], context: dict[str, Any]
    ) -> SymbolicProblem:
        if "compiled_observed_state" in context:
            legacy = KitchenSymbolicProblem(context["compiled_observed_state"])
            initial: set[tuple[str, ...]] = {("hand_empty",)}
            initial.update(("at", obj, region) for obj, region in legacy.initial.locations)
            initial.update(("contains", target, content) for target, content in legacy.initial.contents)
            initial.update(("stirred", target) for target in legacy.initial.stirred)

            actions: list[SymbolicAction] = []
            for obj in sorted(legacy.manipulable):
                destinations = set(legacy._allowed_destinations(obj))
                locations = destinations | {
                    region for candidate, region in legacy.initial.locations if candidate == obj
                }
                for region in sorted(locations):
                    actions.append(_action(
                        "PICK", (obj,),
                        {("hand_empty",), ("at", obj, region)},
                        {("holding", obj)},
                        {("hand_empty",), ("at", obj, region)},
                    ))
                initial_locations = dict(legacy.initial.locations)
                for destination in sorted(destinations):
                    preconditions = {("holding", obj)}
                    if (obj, destination) in legacy.soup_assignments:
                        preconditions.add(("contains", destination, "soup"))
                        if initial_locations.get(destination) == "B1":
                            preconditions.add(("at", destination, legacy.serving_destination))
                    if destination == legacy.serving_destination:
                        if obj in legacy.coffee_targets:
                            preconditions.add(("contains", obj, "coffee"))
                            preconditions.add(("contains", obj, "water"))
                            preconditions.add(("stirred", obj))
                        elif obj in legacy.soup_targets:
                            preconditions.add(("contains", obj, "soup"))
                            if initial_locations.get(obj) != "B1":
                                for tool, assigned_target in legacy.soup_assignments:
                                    if assigned_target == obj:
                                        preconditions.add(("at", tool, obj))
                    actions.append(_action(
                        "PLACE", (obj, destination), preconditions,
                        {("hand_empty",), ("at", obj, destination)},
                        {("holding", obj)},
                    ))

            for source, content in sorted(legacy.source_contents.items()):
                targets = legacy.soup_targets if content == "soup" else legacy.coffee_targets
                for target in sorted(targets):
                    actions.append(_action(
                        "POUR", (source, target),
                        {("holding", source), ("at", target, legacy.home)},
                        {("contains", target, content)}, set(),
                    ))
            for tool, target in sorted(legacy.can_stir):
                actions.append(_action(
                    "STIR", (tool, target),
                    {
                        ("holding", tool),
                        ("at", target, legacy.home),
                        ("contains", target, "coffee"),
                        ("contains", target, "water"),
                    },
                    {("stirred", target)}, set(),
                ))
            actions.sort(key=lambda item: (
                item.name, item.arguments, tuple(sorted(item.positive_preconditions)),
            ))
            return SymbolicProblem(
                initial_atoms=frozenset(initial),
                goal_atoms=frozenset(legacy.goal_facts()),
                actions=tuple(actions),
            )

        # VLM Candidate Planning Mode:
        # Build candidate problem from G_F^VLM, G_O, and phi^VLM without injecting hidden GT recipe
        graph_o = context.get("graph_o")

        def _extract_cands(role_name: str) -> list[str]:
            items = []
            for k, v in (assignment or {}).items():
                if k == role_name or k.startswith(role_name + "_"):
                    if isinstance(v, str):
                        items.append(v)
                    elif isinstance(v, (list, tuple, set)):
                        items.extend(str(x) for x in v)
            return sorted(list(dict.fromkeys(items)))

        coffee_targets = _extract_cands("coffee_container")
        soup_targets = _extract_cands("soup_container")
        coffee_stirrers = _extract_cands("coffee_stirrer")
        soup_utensils = _extract_cands("soup_eating_utensil") or _extract_cands("soup_utensil")
        coffee_sources = _extract_cands("coffee_source")
        water_sources = _extract_cands("water_source")

        specification = context.get("specification")
        grounded = context.get("ground_result")
        bindings = context.get("operation_bindings") or (getattr(grounded, "operation_bindings", {}) if grounded else {})
        stir_pairs: set[tuple[str, str]] = set()
        soup_pairs: set[tuple[str, str]] = set()
        transfer_pairs: set[tuple[str, str, str]] = set()  # (source, target, content)

        if specification is not None:
            for group in specification.operation_groups:
                func_lower = group.function.lower()
                cap_id = getattr(group, "capability_id", "") or ""

                # 1. Material transfer / pour operation
                if cap_id == "TRANSFER_CONTENT_TO_CONTAINER" or any(w in func_lower for w in ("pour", "transfer", "fill", "dispense")):
                    group_bindings = bindings.get(group.id, [])
                    if group_bindings:
                        for b in group_bindings:
                            s = b.get("tool_id")
                            t = b.get("target_id")
                            if s and t:
                                content = "coffee" if (s in coffee_sources or "coffee" in func_lower or "coffee" in group.tool_role.lower()) else "water"
                                transfer_pairs.add((s, t, content))
                    else:
                        sources = _extract_cands(group.tool_role)
                        targets = _extract_cands(group.target_role)
                        for s in sources:
                            for t in targets:
                                content = "coffee" if (s in coffee_sources or "coffee" in func_lower or "coffee" in group.tool_role.lower()) else "water"
                                transfer_pairs.add((s, t, content))

                # 2. Stirring operation
                elif cap_id == "MIX_BEVERAGE_CONTENTS" or any(w in func_lower for w in ("stir", "mix")):
                    group_bindings = bindings.get(group.id, [])
                    if group_bindings:
                        for b in group_bindings:
                            pair = (b.get("tool_id"), b.get("target_id"))
                            if pair[0] and pair[1]:
                                stir_pairs.add(pair)
                    else:
                        tools = _extract_cands(group.tool_role)
                        targets = _extract_cands(group.target_role)
                        for tool in tools:
                            for target in targets:
                                stir_pairs.add((tool, target))

                # 3. Soup eating utensil provision operation
                elif cap_id == "PROVIDE_SOUP_EATING_UTENSIL" or any(w in func_lower for w in ("utensil", "spoon")) or (group.tool_role in ("soup_eating_utensil", "soup_utensil")):
                    group_bindings = bindings.get(group.id, [])
                    if group_bindings:
                        for b in group_bindings:
                            pair = (b.get("tool_id"), b.get("target_id"))
                            if pair[0] and pair[1]:
                                soup_pairs.add(pair)
                    else:
                        tools = _extract_cands(group.tool_role)
                        targets = _extract_cands(group.target_role)
                        for tool in tools:
                            for target in targets:
                                soup_pairs.add((tool, target))

            has_serving = (
                any("serve" in g.function.lower() or "dining" in g.function.lower() or g.target_role in {"serving_location", "dining_table"} for g in specification.operation_groups)
                or any(r.object_role in {"serving_location", "dining_table"} for r in specification.relations)
                or any(r.object_role in {"serving_location", "dining_table"} for r in getattr(specification, "task_causal_relations", ()))
                or any(k in specification.task_instruction.lower() for k in ("serve", "dining", "table", "place coffee", "place soup"))
            )
        else:
            # Standalone legacy invocation when specification is not provided
            stir_pairs = {(t, c) for t in coffee_stirrers for c in coffee_targets}
            soup_pairs = set(zip(soup_utensils, soup_targets))
            for s in coffee_sources:
                for t in coffee_targets:
                    transfer_pairs.add((s, t, "coffee"))
            for s in water_sources:
                for t in coffee_targets:
                    transfer_pairs.add((s, t, "water"))
            has_serving = True

        home = "countertop"
        serving_destination = "dining_table"
        all_objs = sorted(list(dict.fromkeys(
            coffee_targets + soup_targets + coffee_stirrers + soup_utensils + coffee_sources + water_sources
        )))
        if not all_objs:
            raise RuntimeError("No grounded candidate objects for Kitchen planning")

        initial = {("hand_empty",)}
        initial_locations: dict[str, str] = {}
        for obj in all_objs:
            node = graph_o.nodes.get(obj) if graph_o else None
            loc = (node.source_region if node else None) or home
            initial_locations[obj] = loc
            initial.add(("at", obj, loc))

        for s in soup_targets:
            initial.add(("contains", s, "soup"))

        actions: list[SymbolicAction] = []
        for obj in all_objs:
            init_loc = initial_locations.get(obj, home)
            destinations = {home}
            if obj in coffee_targets or obj in soup_targets:
                destinations.add(serving_destination)
            elif obj in soup_utensils:
                destinations.update(t for u, t in soup_pairs if u == obj)

            locations = destinations | {init_loc}
            for loc in sorted(locations):
                actions.append(_action(
                    "PICK", (obj,),
                    {("hand_empty",), ("at", obj, loc)},
                    {("holding", obj)},
                    {("hand_empty",), ("at", obj, loc)},
                ))

            for destination in sorted(destinations):
                preconditions = {("holding", obj)}
                if destination == serving_destination:
                    if obj in coffee_targets:
                        if specification is None:
                            preconditions.add(("contains", obj, "coffee"))
                            preconditions.add(("contains", obj, "water"))
                            preconditions.add(("stirred", obj))
                        else:
                            if any(t == obj and c == "coffee" for _, t, c in transfer_pairs):
                                preconditions.add(("contains", obj, "coffee"))
                            if any(t == obj and c == "water" for _, t, c in transfer_pairs):
                                preconditions.add(("contains", obj, "water"))
                            if any(t == obj for _, t in stir_pairs):
                                preconditions.add(("stirred", obj))
                    elif obj in soup_targets:
                        preconditions.add(("contains", obj, "soup"))
                        if soup_utensils and any(t == obj for _, t in soup_pairs):
                            for u in soup_utensils:
                                if (u, obj) not in soup_pairs:
                                    continue
                                actions.append(_action(
                                    "PLACE", (obj, destination),
                                    preconditions | {("at", u, obj)},
                                    {("hand_empty",), ("at", obj, destination)},
                                    {("holding", obj)},
                                ))
                            continue
                actions.append(_action(
                    "PLACE", (obj, destination), preconditions,
                    {("hand_empty",), ("at", obj, destination)},
                    {("holding", obj)},
                ))

        for source, target, content in sorted(transfer_pairs):
            actions.append(_action(
                "POUR", (source, target),
                {("holding", source), ("at", target, home)},
                {("contains", target, content)}, set(),
            ))

        for tool, target in sorted(stir_pairs):
            preconditions = {("holding", tool), ("at", target, home)}
            if specification is None:
                preconditions.add(("contains", target, "coffee"))
                preconditions.add(("contains", target, "water"))
            else:
                if any(t == target and c == "coffee" for _, t, c in transfer_pairs):
                    preconditions.add(("contains", target, "coffee"))
                if any(t == target and c == "water" for _, t, c in transfer_pairs):
                    preconditions.add(("contains", target, "water"))
            actions.append(_action(
                "STIR", (tool, target),
                preconditions,
                {("stirred", target)}, set(),
            ))

        goal_atoms = set()
        if specification is None:
            for c in coffee_targets:
                goal_atoms.add(("at", c, serving_destination))
                goal_atoms.add(("stirred", c))
            for s in soup_targets:
                goal_atoms.add(("at", s, serving_destination))
            for u, s in soup_pairs:
                goal_atoms.add(("at", u, s))
        else:
            for _, target, content in sorted(transfer_pairs):
                goal_atoms.add(("contains", target, content))
            for _, target in sorted(stir_pairs):
                goal_atoms.add(("stirred", target))
            for u, s in sorted(soup_pairs):
                goal_atoms.add(("at", u, s))
            if has_serving:
                for c in coffee_targets:
                    goal_atoms.add(("at", c, serving_destination))
                for s in soup_targets:
                    goal_atoms.add(("at", s, serving_destination))
        goal_atoms.add(("hand_empty",))

        actions.sort(key=lambda item: (
            item.name, item.arguments, tuple(sorted(item.positive_preconditions)),
        ))
        return SymbolicProblem(
            initial_atoms=frozenset(initial),
            goal_atoms=frozenset(goal_atoms),
            actions=tuple(actions),
        )


def scene_for_variant(internal_variant: str) -> KitchenScene:
    code = internal_variant.split("_", 1)[0]
    return KitchenScene(
        f"S1_integrated_kitchen_object_function_feasibility_{code}",
        include_robot=False,
        robot="none",
    )


def compile_kitchen_contract_from_graph(graph: FunctionalRequirementGraph) -> dict[str, Any]:
    """Deterministically compile legacy kitchen contract from G_F without reading raw_requirements."""
    roles_dict = {}
    for name, node in graph.nodes.items():
        if name in {"coffee_source", "water_source", "soup_source"}:
            continue
        pref = [
            {"rank": i + 1, "canonical_label": cat, "detector_aliases": []}
            for i, cat in enumerate(node.semantic_categories)
        ]
        unary = []
        for p in node.unary_predicates:
            unary.append({"predicate": p, "expected": True})
        for c in node.numeric_constraints:
            unary.append({
                "property": c.property_name,
                "operator": c.operator,
                "value": c.threshold,
                "unit": c.unit,
            })
        min_c = node.minimum_count
        max_c = node.maximum_count
        cardinality = {
            "mode": "assignment_driven",
            "minimum_distinct_physical_objects": min_c,
            "maximum_distinct_physical_objects": max_c,
        }
        if node.preference:
            cardinality["preferred"] = node.preference
        elif node.reusable:
            cardinality["preferred"] = "minimize_distinct"
        roles_dict[name] = {
            "count": max_c,
            "binding_cardinality": cardinality,
            "semantic_preferences": pref,
            "unary_geometry": unary,
            "allow_empty_geometry": not bool(unary),
        }

    relations_list = [
        {
            "predicate": r.predicate,
            "subject_role": r.subject_role,
            "object_role": r.object_role,
            "expected": r.expected,
        }
        for r in graph.relations
        if r.subject_role in roles_dict and r.object_role in roles_dict
    ]

    op_groups_dict = {}
    for grp in graph.operation_groups:
        if grp.tool_role not in roles_dict or grp.target_role not in roles_dict:
            continue
        distinct = (
            grp.distinct_within_group
            if grp.distinct_within_group is not None
            else (grp.usage_policy == "DEDICATED_PER_TARGET")
        )
        same_tool = (
            grp.same_tool_must_cover_all_targets
            if grp.same_tool_must_cover_all_targets is not None
            else (grp.usage_policy == "SHARED_ACROSS_ALL_TARGETS")
        )
        pref = grp.selection_preference or (
            "minimize_distinct_tools"
            if grp.usage_policy in {"SEQUENTIAL_REUSE_ALLOWED", "SHARED_ACROSS_ALL_TARGETS"}
            else "deterministic_rank"
        )
        op_groups_dict[grp.id] = {
            "function": grp.function,
            "tool_role": grp.tool_role,
            "target_role": grp.target_role,
            "required_target_count": grp.required_target_count,
            "usage_policy": {
                "mode": grp.usage_policy.lower(),
                "distinct_within_group": distinct,
                "same_tool_must_cover_all_targets": same_tool,
                "selection_preference": pref,
            },
            "relations": list(grp.required_relations),
        }
        # The legacy joint-witness contract indexes pairwise verifiers from
        # the top-level relation list. Operation-group requirements already
        # carry the same expressed semantics, so project them directionally
        # into that index without inventing any additional relation.
        for predicate in grp.required_relations:
            projected = {
                "predicate": predicate,
                "subject_role": grp.tool_role,
                "object_role": grp.target_role,
                "expected": True,
            }
            if projected not in relations_list:
                relations_list.append(projected)

    symbolic_task = graph.metadata.get("symbolic_task")
    if not symbolic_task:
        source_roles = {}
        for name, node in graph.nodes.items():
            if "source" in name or "provider" in name:
                provides = "coffee" if "coffee" in name else ("water" if "water" in name else "soup")
                source_roles[name] = {
                    "accepted_semantic_labels": list(node.semantic_categories),
                    "provides": provides,
                    "count": getattr(node, "minimum_count", 1),
                }
        target_reqs = {}
        if "coffee_container" in graph.nodes:
            target_reqs["coffee"] = {
                "witness_role": "coffee_container",
                "required_contents": [p for p in ["coffee", "water"] if f"{p}_source" in source_roles],
                "requires_operation_group": "coffee_stirring",
                "final_goal": "served",
            }
        if "soup_container" in graph.nodes:
            target_reqs["soup"] = {
                "witness_role": "soup_container",
                "required_contents": ["soup"],
                "initial_contents": ["soup"],
                "requires_operation_group": "soup_serving",
                "final_goal": "served",
            }
        symbolic_task = {
            "schema_version": 1,
            "home_region": "countertop",
            "initial_observation_region": "countertop",
            "contents": ["soup"],
            "source_roles": source_roles,
            "target_requirements": target_reqs,
            "causal_dependencies": [],
        }

    contract_result = {
        "schema_version": 2,
        "task_id": "s1_integrated_prepare_and_serve_coffee_and_soup",
        "specification_source": graph.source,
        "goal_instruction": graph.task_instruction,
        "roles": roles_dict,
        "relations": relations_list,
        "cross_group_reuse": {"allowed": graph.cross_group_reuse_allowed},
        "symbolic_task": symbolic_task,
    }
    if op_groups_dict:
        contract_result["operation_groups"] = op_groups_dict
    return contract_result


def build_kitchen_observed_scene_graph(session: Any) -> ObservedSceneGraph:
    """Build canonical ObservedSceneGraph G_O from kitchen inspection session evidence."""
    from ..scene_graph import ObservedNode, ObservedObject, ObservedRelation, ObservedSceneGraph

    graph_o = ObservedSceneGraph()
    for obj_id, record in sorted(session.registry.get("objects", {}).items()):
        semantics = record.get("semantics", {}) or record.get("semantic_classification", {})
        canonical = (
            semantics.get("validated", {}).get("canonical_label")
            or semantics.get("latest_observation", {}).get("canonical_label")
            or semantics.get("canonical_label")
        )
        unary_preds = {
            k: v.get("status", "TRUE" if v.get("value") else "FALSE") if isinstance(v, dict) else ("TRUE" if v else "FALSE")
            for k, v in record.get("geometric_predicates", {}).items()
        }
        geom_props_raw = record.get("geometric_properties", {})
        geom_props = {
            k: v.get("value", v) if isinstance(v, dict) else v
            for k, v in geom_props_raw.items()
        }
        if "open_cavity" in geom_props_raw and "OPEN_CAVITY" not in unary_preds:
            unary_preds["OPEN_CAVITY"] = "TRUE" if geom_props_raw["open_cavity"].get("value") else "FALSE"
        if "elongated" in geom_props_raw and "ELONGATED_OBJECT" not in unary_preds:
            unary_preds["ELONGATED_OBJECT"] = "TRUE" if geom_props_raw["elongated"].get("value") else "FALSE"

        node = ObservedNode(
            instance_id=obj_id,
            entity_kind="OBJECT",
            canonical_category=canonical,
            semantic_labels=dict(semantics),
            source_region=record.get("source_region"),
            geometry=dict(geom_props),
            unary_properties=dict(geom_props),
            unary_predicates=unary_preds,
            first_seen_stage=int(record.get("first_seen_stage", 0)),
            last_seen_stage=int(record.get("last_seen_stage", 0)),
        )
        graph_o.add_node(node)

    # Load evaluated pairwise relations from stage artifacts if available
    run_dir = getattr(session, "run_dir", None)
    if run_dir is not None and hasattr(run_dir, "glob"):
        for stage_dir in sorted(run_dir.glob("stages/*")):
            pair_path = stage_dir / "pair_relation_evaluations.json"
            if pair_path.is_file():
                try:
                    pair_data = json.loads(pair_path.read_text(encoding="utf-8"))
                    for item in pair_data.get("relations", []):
                        graph_o.add_relation(ObservedRelation(
                            subject_id=item["source_object_id"],
                            predicate=item["relation"],
                            object_id=item["target_object_id"],
                            status=str(item.get("status", "UNKNOWN")),
                            evidence=dict(item.get("evidence", {})),
                        ))
                except Exception:
                    pass

    # Check remaining pairwise relations or evaluate them directly
    from mujoco_scenes.geometry_properties import pairwise_relation_evaluation

    obj_ids = sorted(session.registry.get("objects", {}).keys())
    for s_id in obj_ids:
        for t_id in obj_ids:
            if s_id == t_id:
                continue
            s_rec = session.registry["objects"][s_id]
            t_rec = session.registry["objects"][t_id]
            for rel_name in ("INSERTABLE_IN", "REACHES_BOTTOM"):
                if not graph_o.get_relation(rel_name, s_id, t_id):
                    eval_res = pairwise_relation_evaluation(rel_name, s_rec, t_rec, session.config)
                    graph_o.add_relation(ObservedRelation(
                        subject_id=s_id,
                        predicate=rel_name,
                        object_id=t_id,
                        status=str(eval_res.get("status", "UNKNOWN")),
                        evidence=dict(eval_res),
                    ))

    return graph_o


def build_canonical_kitchen_witness(
    specification: FunctionalRequirementGraph,
    ground_result: GraphGroundingResult,
    graph_o: ObservedSceneGraph,
) -> dict[str, Any]:
    """Build pure canonical compatibility witness dictionary from graph grounding and G_O evidence."""
    selected_witness: dict[str, list[str]] = {}
    for role_name, assigned_val in ground_result.assignment.items():
        if isinstance(assigned_val, str):
            selected_witness[role_name] = [assigned_val]
        elif isinstance(assigned_val, (list, tuple, set)):
            selected_witness[role_name] = list(assigned_val)
        else:
            selected_witness[role_name] = []

    operation_assignments: list[dict[str, Any]] = []
    for group in specification.operation_groups:
        bindings = ground_result.operation_bindings.get(group.id, [])
        for binding in bindings:
            tool_id = binding.get("tool_id")
            target_id = binding.get("target_id")
            checks = []
            all_checks_true = True
            for rel in group.required_relations:
                obs_rel = graph_o.get_relation(rel, tool_id, target_id)
                if obs_rel is None:
                    raise RuntimeError(
                        f"Architecture Error: ground_graph selected binding ({tool_id}, {target_id}) "
                        f"for group '{group.id}', but required relation '{rel}' was not found in G_O."
                    )
                status = str(obs_rel.status)
                if status != "TRUE":
                    all_checks_true = False
                    if ground_result.complete:
                        raise RuntimeError(
                            f"Architecture Error: ground_graph claimed COMPLETE, but selected binding "
                            f"({tool_id}, {target_id}) for group '{group.id}' has relation '{rel}' with status '{status}'."
                        )
                checks.append({
                    "relation": rel,
                    "status": status,
                    "evidence": dict(obs_rel.evidence),
                })
            assignment_status = "TRUE" if all_checks_true else "FALSE"
            operation_assignments.append({
                "function_group_id": group.id,
                "utensil_object_id": tool_id,
                "target_object_id": target_id,
                "assignment_status": assignment_status,
                "pair_geometry_status": assignment_status,
                "relation_checks": checks,
                "context": dict(binding.get("context", {})),
            })

    return {
        "status": "COMPLETE" if ground_result.complete else "INCOMPLETE",
        "inference_basis": "CANONICAL_GRAPH_GROUNDING_PLUS_OBSERVED_RELATION_EVIDENCE",
        "selected_witness": selected_witness,
        "operation_assignments": operation_assignments,
    }


def run_to_plan(
    *,
    variant_label: str,
    internal_variant: str,
    mode: str,
    specification: FunctionalSpecification,
    output_dir: Path,
    scene: KitchenScene | None = None,
    search_contract: SearchRegionContract | None = None,
    search_order: tuple[str, ...] | SearchRegionContract | None = None,
    observer: Any = None,
) -> PipelineResult:
    from ..grounding import ground_graph
    from ..task_interface_validator import validate_runtime_gf
    validate_runtime_gf(specification)

    scene = scene or scene_for_variant(internal_variant)
    contract = compile_kitchen_contract_from_graph(specification)

    phase1_dir = output_dir / "observed_search" / "phase1"
    # Dump task-scoped vocabulary for perception
    root = Path(__file__).resolve().parents[2]
    base_vocab_path = Path(specification.metadata.get("semantic_vocabulary_path", root / "configs" / "semantic_vocabulary.yaml"))
    base_vocab: dict[str, Any] = {}
    if base_vocab_path.is_file():
        base_vocab = yaml.safe_load(base_vocab_path.read_text(encoding="utf-8")) or {}

    all_system_role_cats: set[str] = set()
    for role in specification.nodes.values():
        all_system_role_cats.update(role.semantic_categories)

    raw_candidates = list(specification.detector_vocabulary)
    from ..role_semantic_ontology import build_task_detector_vocabulary
    canonical_labels = build_task_detector_vocabulary(
        system_role_categories=all_system_role_cats,
        raw_vlm_candidate_categories=raw_candidates,
        base_semantic_ontology=base_vocab,
    )
    vocab_dict = {
        "schema_version": 1,
        "canonical_labels": canonical_labels,
    }
    vocab_text = yaml.safe_dump(vocab_dict, sort_keys=False)
    vocabulary_path = phase1_dir / "yolo_world_dynamic_vocabulary.yaml"
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path.write_text(vocab_text, encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "kitchen_vocabulary.yaml").write_text(vocab_text, encoding="utf-8")
    from ..search_contract import SearchRegionContract, freeze_search_region_contract

    if search_contract is None:
        if isinstance(search_order, SearchRegionContract):
            search_contract = search_order
        else:
            search_contract = freeze_search_region_contract(
                specification,
                domain="kitchen",
                mode=mode,
                variant=variant_label,
            )

    from ..search import classify_search_state, compute_causal_search_recovery

    # Gate before search: if online_executable_contract_complete is False, zero pointless search
    contract_ok = getattr(
        specification,
        "online_executable_contract_complete",
        getattr(specification, "required_contract_complete", False),
    )
    if not contract_ok:
        order = ()
    else:
        order = tuple(search_contract.canonical_region_ids)


    grounding_snapshots: list[dict[str, Any]] = []

    def kitchen_completion_predicate(current: Any) -> bool:
        current_go = build_kitchen_observed_scene_graph(current)
        current_events = [
            json.loads(line)
            for line in Path(getattr(current, "events_path", "")).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if getattr(current, "events_path", None) and Path(getattr(current, "events_path", "")).is_file() else []
        current_opened = [
            event["region_id"] for event in current_events if event.get("event") == "REGION_OPENED"
        ]
        res = ground_graph(specification, current_go, {"search_exhausted": False})
        search_state = classify_search_state(specification, res, search_contract, current_opened)
        res_dict = res.to_dict()
        if "evidence" in res_dict and isinstance(res_dict["evidence"], dict):
            res_dict["evidence"] = {k: v for k, v in res_dict["evidence"].items() if k != "grounding_snapshots"}
        stage_label = f"after_{current_opened[-1]}" if current_opened else "initial"
        grounding_snapshots.append({
            "stage": stage_label,
            "inspected_regions": list(current_opened),
            "search_state": search_state,
            "grounding": res_dict,
        })
        if observer is not None:
            observer("grounding_updated", {
                "grounding": res.to_dict(),
                "satisfied": bool(res.complete),
                "status": res.status,
                "scene_graph": current_go.to_dict(),
                "search_state": search_state,
            })
        return bool(res.complete)

    if not contract.get("roles"):
        from ..scene_graph import ObservedSceneGraph
        graph_o = ObservedSceneGraph()
        opened = ()
        is_exhausted = True
        ground_result = ground_graph(specification, graph_o, {"search_exhausted": is_exhausted})
        final_search_state = classify_search_state(specification, ground_result, search_contract, opened)
        gr_dict = ground_result.to_dict()
        if "evidence" in gr_dict and isinstance(gr_dict["evidence"], dict):
            gr_dict["evidence"] = {k: v for k, v in gr_dict["evidence"].items() if k != "grounding_snapshots"}
        grounding_snapshots.append({
            "stage": "final",
            "inspected_regions": list(opened),
            "search_state": final_search_state,
            "grounding": gr_dict,
        })
        causal_search_recovery = False
    else:
        session = run_sequential_inspection(
            scene,
            order,
            runs_root=output_dir / "observed_search",
            run_id="phase1",
            width=1280,
            height=960,
            task_requirements=contract,
            stop_on_complete=True,
            semantic_backend="yolo_world",
            semantic_model=str(LOCAL_YOLO_WORLD),
            semantic_vocabulary_path=vocabulary_path,
            semantic_min_supporting_views=2,
            grounding_mode="joint",
            completion_predicate=kitchen_completion_predicate,
            record_oracle_diagnostics=False,
            observer=observer,
        )
        events = [
            json.loads(line) for line in session.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        opened = tuple(
            event["region_id"] for event in events if event.get("event") == "REGION_OPENED"
        )

        # Populate canonical ObservedSceneGraph from session evidence
        graph_o = build_kitchen_observed_scene_graph(session)
        for r in opened:
            graph_o.mark_region_inspected(r)

        is_exhausted = len(opened) >= len(order)
        ground_result = ground_graph(specification, graph_o, {"search_exhausted": is_exhausted})
        if mode == "vlm" and is_exhausted and not ground_result.complete and getattr(
            specification, "online_executable_contract_complete", getattr(specification, "required_contract_complete", True)
        ):
            from ..grounding import ground_verified_candidate_subgraph
            ground_result = ground_verified_candidate_subgraph(specification, graph_o)


        final_search_state = classify_search_state(specification, ground_result, search_contract, opened)
        gr_dict = ground_result.to_dict()
        if "evidence" in gr_dict and isinstance(gr_dict["evidence"], dict):
            gr_dict["evidence"] = {k: v for k, v in gr_dict["evidence"].items() if k != "grounding_snapshots"}
        grounding_snapshots.append({
            "stage": "final",
            "inspected_regions": list(opened),
            "search_state": final_search_state,
            "grounding": gr_dict,
        })
        initial_complete = bool(grounding_snapshots[0]["grounding"].get("complete", False)) if grounding_snapshots else False
        causal_search_recovery = compute_causal_search_recovery(
            initial_complete, opened, bool(ground_result.complete)
        )

    (output_dir / "grounding_snapshots.json").write_text(
        json.dumps(grounding_snapshots, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if isinstance(ground_result.evidence, dict):
        ground_result.evidence["grounding_snapshots"] = list(grounding_snapshots)
        ground_result.evidence["search_state"] = final_search_state
        ground_result.evidence["causal_search_recovery"] = causal_search_recovery

    if observer is not None:
        observer("grounding_updated", {
            "grounding": ground_result.to_dict(),
            "satisfied": bool(ground_result.complete),
            "status": ground_result.status,
            "scene_graph": graph_o.to_dict(),
            "search_state": final_search_state,
        })

    (output_dir / "observed_scene_graph.json").write_text(
        json.dumps(graph_o.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "graph_grounding_result.json").write_text(
        json.dumps(ground_result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if (not ground_result.complete and mode != "vlm") or not ground_result.assignment:
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status="NO_MEANINGFUL_CANDIDATE_PLAN" if mode == "vlm" else ground_result.status,
            inspected_regions=opened,
            canonicalization_succeeded=True,
            functional_spec_complete=False,
            failure_reason=str(ground_result.unsatisfied_relations or ground_result.missing_roles or "NO_COMPLETE_FUNCTIONAL_WITNESS"),
            outcome_category=classify_pipeline_outcome(
                task_specification_valid=True,
                graph_compiled=True,
                individual_candidates_sufficient=ground_result.failure_kind != "OBJECT_DISCOVERY_FAILURE",
                functional_assignment_complete=False,
            ).category,
        )

    from ..grounding import resolved_functional_graph
    planning_specification = resolved_functional_graph(specification, ground_result)

    # Compile observed symbolic state from graph grounding assignment
    # Build compatibility witness from canonical graph grounding result & actual G_O relations
    witness_payload = build_canonical_kitchen_witness(planning_specification, ground_result, graph_o)

    (session.run_dir / "latest_witness.json").write_text(
        json.dumps(witness_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "canonical_grounding_witness.json").write_text(
        json.dumps(witness_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        assignments = ground_result.assignment
        is_vlm_candidate = mode == "vlm"
        if is_vlm_candidate:
            planned = plan_with_common_astar(
                KitchenPlanningCompiler(), assignments,
                {
                    "specification": planning_specification,
                    "graph_o": graph_o,
                    "ground_result": ground_result,
                    "operation_bindings": getattr(ground_result, "operation_bindings", {}),
                    "is_vlm_candidate": True,
                },
                allow_partial=True,
            )
        else:
            compiled = compile_observed_symbolic_state(session.run_dir, contract)
            planned = plan_with_common_astar(
                KitchenPlanningCompiler(), assignments,
                {"compiled_observed_state": compiled},
                allow_partial=(mode == "vlm"),
            )
        plan_dir = output_dir / "action_sequence"
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "action_plan.json").write_text(
            json.dumps({
                "planner": planned.search.statistics,
                "actions": list(planned.actions),
                "validation": planned.validation,
                "exploratory_open_actions_excluded": True,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from ..audit import audit_plan_grounding

        audit = audit_plan_grounding(
            planning_specification, graph_o, ground_result, planned.actions, home_region=contract.get("symbolic_task", {}).get("home_region", "countertop")
        )
        (output_dir / "plan_grounding_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        is_partial = planned.search.statistics.get("is_partial", False)
        is_full_plan = complete_planning_contract(
            planning_specification, ground_result, planned.search.statistics, planned.validation
        )
        if is_full_plan:
            status = "ACTION_SEQUENCE_READY"
            spec_complete = True
        elif planned.actions:
            status = "PARTIAL_ACTION_SEQUENCE_READY"
            spec_complete = False
        else:
            status = "NO_MEANINGFUL_CANDIDATE_PLAN" if mode == "vlm" else "CANDIDATE_GRAPH_UNSATISFIABLE"
            spec_complete = False

        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status=status, inspected_regions=opened,
            assignment=assignments,
            plan=planned.actions if is_full_plan else (),
            candidate_plan=planned.actions,
            search_statistics=planned.search.statistics,
            candidate_search_statistics=planned.search.statistics,
            canonicalization_succeeded=True,
            functional_spec_complete=spec_complete,
            outcome_category=classify_pipeline_outcome(
                task_specification_valid=True, graph_compiled=True,
                individual_candidates_sufficient=ground_result.failure_kind != "OBJECT_DISCOVERY_FAILURE",
                functional_assignment_complete=ground_result.complete,
                planning_invoked=True, plan_complete=is_full_plan,
            ).category,
        )
    except Exception as exc:
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status="NO_MEANINGFUL_CANDIDATE_PLAN" if mode == "vlm" else "CANDIDATE_GRAPH_UNSATISFIABLE",
            inspected_regions=opened,
            assignment=ground_result.assignment,
            plan=(),
            candidate_plan=(),
            canonicalization_succeeded=True,
            functional_spec_complete=False,
            failure_reason=f"NO_MEANINGFUL_CANDIDATE_PLAN: {exc}" if mode == "vlm" else f"CANDIDATE_GRAPH_UNSATISFIABLE: {exc}",
            outcome_category=classify_pipeline_outcome(
                task_specification_valid=True,
                graph_compiled=True,
                individual_candidates_sufficient=True,
                functional_assignment_complete=ground_result.complete,
                planning_invoked=True,
                plan_complete=False,
            ).category,
        )
