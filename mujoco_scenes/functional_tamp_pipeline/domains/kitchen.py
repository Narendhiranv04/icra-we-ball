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
                actions.append(_action(
                    "PLACE", (obj, destination), preconditions,
                    {("hand_empty",), ("at", obj, destination)},
                    {("holding", obj)},
                ))

        for source, content in sorted(legacy.source_contents.items()):
            targets = legacy.soup_targets if content == "soup" else legacy.coffee_targets
            for target in sorted(targets):
                actions.append(_action(
                    "POUR", (source, target), {("holding", source)},
                    {("contains", target, content)}, set(),
                ))
        for tool, target in sorted(legacy.can_stir):
            actions.append(_action(
                "STIR", (tool, target),
                {
                    ("holding", tool), ("contains", target, "coffee"),
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
        cardinality = {
            "mode": "assignment_driven",
            "minimum_distinct_physical_objects": 1 if node.reusable else node.count,
            "maximum_distinct_physical_objects": node.count,
        }
        if node.reusable:
            cardinality["preferred"] = "minimize_distinct"
        roles_dict[name] = {
            "count": node.count,
            "binding_cardinality": cardinality,
            "semantic_preferences": pref,
            "unary_geometry": unary,
        }

    relations_list = [
        {
            "predicate": r.predicate,
            "subject_role": r.subject_role,
            "object_role": r.object_role,
            "expected": r.expected,
        }
        for r in graph.relations
    ]

    op_groups_dict = {}
    for grp in graph.operation_groups:
        op_groups_dict[grp.id] = {
            "function": grp.function,
            "tool_role": grp.tool_role,
            "target_role": grp.target_role,
            "required_target_count": grp.required_target_count,
            "usage_policy": {
                "mode": grp.usage_policy.lower(),
                "distinct_within_group": grp.usage_policy == "DEDICATED_PER_TARGET",
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

    return {
        "schema_version": 2,
        "task_id": "s1_integrated_prepare_and_serve_coffee_and_soup",
        "specification_source": graph.source,
        "goal_instruction": graph.task_instruction,
        "roles": roles_dict,
        "relations": relations_list,
        "operation_groups": op_groups_dict,
        "cross_group_reuse": {"allowed": graph.cross_group_reuse_allowed},
        "symbolic_task": symbolic_task,
    }


def build_kitchen_observed_scene_graph(session: Any) -> ObservedSceneGraph:
    """Build canonical ObservedSceneGraph G_O from kitchen inspection session evidence."""
    graph_o = ObservedSceneGraph()
    for obj_id, record in sorted(session.registry.get("objects", {}).items()):
        canonical = record.get("semantic_classification", {}).get("canonical_label")
        unary_preds = {
            k: "TRUE" if v else "FALSE"
            for k, v in record.get("unary_predicates", {}).items()
        }
        # Populate standard kitchen geometric predicates if properties are present
        geom_props = record.get("geometric_properties", {})
        if "open_cavity" in geom_props:
            unary_preds["OPEN_CAVITY"] = "TRUE" if geom_props["open_cavity"].get("value") else "FALSE"
        if "elongated" in geom_props:
            unary_preds["ELONGATED_OBJECT"] = "TRUE" if geom_props["elongated"].get("value") else "FALSE"

        node = ObservedNode(
            instance_id=obj_id,
            entity_kind="OBJECT",
            canonical_category=canonical,
            semantic_labels=dict(record.get("semantic_classification", {})),
            source_region=record.get("source_region"),
            geometry=dict(geom_props),
            unary_properties=dict(geom_props),
            unary_predicates=unary_preds,
            first_seen_stage=int(record.get("first_seen_stage", 0)),
            last_seen_stage=int(record.get("last_seen_stage", 0)),
        )
        graph_o.add_node(node)

    # Check pairwise relations recorded in session stage runs or evaluate them
    from mujoco_scenes.geometry_properties import pairwise_relation_evaluation

    obj_ids = sorted(session.registry.get("objects", {}).keys())
    for s_id in obj_ids:
        for t_id in obj_ids:
            if s_id == t_id:
                continue
            s_rec = session.registry["objects"][s_id]
            t_rec = session.registry["objects"][t_id]
            for rel_name in ("INSERTABLE_IN", "REACHES_BOTTOM"):
                eval_res = pairwise_relation_evaluation(rel_name, s_rec, t_rec, session.config)
                graph_o.add_relation(ObservedRelation(
                    subject_id=s_id,
                    predicate=rel_name,
                    object_id=t_id,
                    status=str(eval_res.get("status", "UNKNOWN")),
                    evidence=dict(eval_res),
                ))

    return graph_o


def run_to_plan(
    *,
    variant_label: str,
    internal_variant: str,
    mode: str,
    specification: FunctionalSpecification,
    output_dir: Path,
    scene: KitchenScene | None = None,
) -> PipelineResult:
    from ..grounding import ground_graph

    scene = scene or scene_for_variant(internal_variant)
    contract = compile_kitchen_contract_from_graph(specification)

    vocabulary_path: Path
    if "object_vocabulary" in specification.metadata:
        vocabulary_path = output_dir / "object_vocabulary.yaml"
        vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        vocabulary_path.write_text(
            yaml.safe_dump(specification.metadata["object_vocabulary"], sort_keys=False),
            encoding="utf-8",
        )
    else:
        vocabulary_path = Path(specification.metadata["semantic_vocabulary_path"])

    session = run_sequential_inspection(
        scene,
        specification.region_ranking,
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
        completion_predicate=lambda current: (
            (current.latest_witness or {}).get("status") == "COMPLETE"
        ),
        record_oracle_diagnostics=False,
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

    # Canonical graph grounding decides the assignment authority
    ground_result = ground_graph(specification, graph_o)

    if not ground_result.complete or not ground_result.assignment:
        return PipelineResult(
            domain="kitchen", variant=variant_label, mode=mode,
            status=ground_result.status, inspected_regions=opened,
            failure_reason=str(ground_result.unsatisfied_relations or ground_result.missing_roles or "NO_COMPLETE_FUNCTIONAL_WITNESS"),
        )

    # Compile observed symbolic state from graph grounding assignment
    compiled = compile_observed_symbolic_state(session.run_dir, contract)
    assignments = ground_result.assignment
    compiled["role_assignments"] = assignments

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
    return PipelineResult(
        domain="kitchen", variant=variant_label, mode=mode,
        status="ACTION_SEQUENCE_READY", inspected_regions=opened,
        assignment=assignments, plan=planned.actions,
        search_statistics=planned.search.statistics,
    )
