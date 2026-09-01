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
from ..planning import plan_with_common_astar


TASK = (
    "Prepare and serve coffee and soup for two people using the available "
    "kitchenware. Stir both coffees and provide each soup bowl with a suitable "
    "utensil. Search closed kitchen storage for anything still required."
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
        del assignment
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
            for destination in sorted(destinations):
                preconditions = {("holding", obj)}
                if (obj, destination) in legacy.soup_assignments:
                    preconditions.add(("contains", destination, "soup"))
                if destination == legacy.serving_destination:
                    if obj in legacy.coffee_targets:
                        preconditions.add(("contains", obj, "coffee"))
                        preconditions.add(("contains", obj, "water"))
                        preconditions.add(("stirred", obj))
                    elif obj in legacy.soup_targets:
                        preconditions.add(("contains", obj, "soup"))
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
            if grp.usage_policy == "SHARED_ACROSS_ALL_TARGETS"
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

    symbolic_task = graph.metadata.get("symbolic_task")
    if not symbolic_task:
        symbolic_task = {
            "schema_version": 1,
            "home_region": "countertop",
            "initial_observation_region": "countertop",
            "contents": ["coffee", "water", "soup"],
            "source_roles": {
                "coffee_source": {"accepted_semantic_labels": ["coffee_source"], "provides": "coffee", "count": 1},
                "water_source": {"accepted_semantic_labels": ["kettle"], "provides": "water", "count": 1},
            },
            "target_requirements": {
                "coffee": {
                    "witness_role": "coffee_container",
                    "required_contents": ["coffee", "water"],
                    "requires_operation_group": "coffee_stirring",
                    "final_goal": "served",
                },
                "soup": {
                    "witness_role": "soup_container",
                    "required_contents": ["soup"],
                    "initial_contents": ["soup"],
                    "requires_operation_group": "soup_serving",
                    "final_goal": "served",
                },
            },
            "causal_dependencies": [
                ["coffee_contents_present", "coffee_stirred"],
                ["coffee_stirred", "coffee_served"],
                ["soup_content_present", "soup_utensil_placed"],
                ["soup_utensil_placed", "soup_served"],
            ],
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
    search_order: tuple[str, ...] | None = None,
    observer: Any = None,
) -> PipelineResult:
    from ..grounding import ground_graph
    from ..task_interface_validator import validate_runtime_gf
    validate_runtime_gf(specification)

    scene = scene or scene_for_variant(internal_variant)
    contract = compile_kitchen_contract_from_graph(specification)
    vocabulary_path = output_dir / "kitchen_vocabulary.yaml"
    canonical_labels: dict[str, list[str]] = {}
    root = Path(__file__).resolve().parents[2]
    base_vocab_path = Path(specification.metadata.get("semantic_vocabulary_path", root / "configs" / "semantic_vocabulary.yaml"))
    base_canon: dict[str, list[str]] = {}
    alias_to_base_canon: dict[str, str] = {}
    if base_vocab_path.is_file():
        base_vocab = yaml.safe_load(base_vocab_path.read_text(encoding="utf-8"))
        base_canon = dict(base_vocab.get("canonical_labels", {}))
        for canon_k, aliases in base_canon.items():
            for a in aliases:
                alias_to_base_canon[a.strip().lower()] = canon_k

    for role in specification.nodes.values():
        for cat in role.semantic_categories:
            norm_cat = cat.strip().lower()
            if norm_cat in base_canon:
                if norm_cat not in canonical_labels:
                    canonical_labels[norm_cat] = list(base_canon[norm_cat])
            elif norm_cat in alias_to_base_canon:
                resolved_canon = alias_to_base_canon[norm_cat]
                if resolved_canon not in canonical_labels:
                    canonical_labels[resolved_canon] = list(base_canon[resolved_canon])
            else:
                if norm_cat not in canonical_labels:
                    canonical_labels[norm_cat] = [norm_cat]
    vocab_dict = {
        "schema_version": 1,
        "canonical_labels": canonical_labels,
    }
    vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_path.write_text(yaml.safe_dump(vocab_dict, sort_keys=False), encoding="utf-8")

    phase1_dir = output_dir / "observed_search" / "phase1"
    from ..search_contract import freeze_search_region_contract

    if search_order is not None:
        order = tuple(search_order.canonical_region_ids if hasattr(search_order, "canonical_region_ids") else search_order)
    else:
        order = tuple(freeze_search_region_contract(specification).canonical_region_ids)

    def kitchen_completion_predicate(current: Any) -> bool:
        current_go = build_kitchen_observed_scene_graph(current)
        res = ground_graph(specification, current_go, {"search_exhausted": False})
        if observer is not None:
            observer("grounding_updated", {
                "grounding": res.to_dict(),
                "satisfied": bool(res.complete),
                "status": res.status,
                "scene_graph": current_go.to_dict(),
            })
        return bool(res.complete)

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
    # Canonical graph grounding decides the assignment authority
    ground_result = ground_graph(specification, graph_o, {"search_exhausted": is_exhausted})
    if observer is not None:
        observer("grounding_updated", {
            "grounding": ground_result.to_dict(),
            "satisfied": bool(ground_result.complete),
            "status": ground_result.status,
            "scene_graph": graph_o.to_dict(),
        })

    (output_dir / "observed_scene_graph.json").write_text(
        json.dumps(graph_o.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "graph_grounding_result.json").write_text(
        json.dumps(ground_result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not ground_result.complete or not ground_result.assignment:
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status=ground_result.status, inspected_regions=opened,
            failure_reason=str(ground_result.unsatisfied_relations or ground_result.missing_roles or "NO_COMPLETE_FUNCTIONAL_WITNESS"),
        )

    # Compile observed symbolic state from graph grounding assignment
    # Build compatibility witness from canonical graph grounding result & actual G_O relations
    witness_payload = build_canonical_kitchen_witness(specification, ground_result, graph_o)

    (session.run_dir / "latest_witness.json").write_text(
        json.dumps(witness_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "canonical_grounding_witness.json").write_text(
        json.dumps(witness_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        compiled = compile_observed_symbolic_state(session.run_dir, contract)
        assignments = ground_result.assignment

        planned = plan_with_common_astar(
            KitchenPlanningCompiler(), assignments,
            {"compiled_observed_state": compiled},
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
            specification, graph_o, ground_result, planned.actions, home_region=contract["symbolic_task"].get("home_region", "countertop")
        )
        (output_dir / "plan_grounding_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status="ACTION_SEQUENCE_READY", inspected_regions=opened,
            assignment=assignments, plan=planned.actions,
            search_statistics=planned.search.statistics,
        )
    except Exception as exc:
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status="INFEASIBLE", inspected_regions=opened,
            failure_reason=str(exc),
        )
