"""Unit tests for Stage 2: Complete Operation -> Capability Bridge (Gate 2).

Verifies:
1. explicit FM "stir contents" -> STIR capability -> INSERTABLE_IN + REACHES_BOTTOM.
2. explicit FM "install/fasten component at target" -> FASTEN capability ->
   workshop physical preconditions (COMPATIBLE_WITH, REACHES_TARGET, COMPATIBLE_WITH_TARGET)
   instantiated in both V1 and V2 without requiring FM to separately name checker predicates.
3. explicit FM "transfer material into container" -> transfer capability ->
   planner can generate POUR realization.
4. roles only, no explicit operation -> no task operation synthesized.
5. physical precondition provenance is recorded on FunctionalRelation and metadata.
6. workshop return reusable equipment capability maps to PLACE operator.
"""
from __future__ import annotations

import json
import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    get_robot_capabilities,
    interpret_operation,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.symbolic_planning import GroundAction, KitchenSymbolicProblem, PlannerState


def test_explicit_fm_stir_contents_derives_stir_preconditions():
    """Explicit FM 'stir contents' maps to STIR_COFFEE and gains INSERTABLE_IN + REACHES_BOTTOM."""
    # 1. Direct interpreter check
    res = interpret_operation(
        domain="kitchen",
        raw_phrase="stir contents",
        source_role="coffee_stirrer",
        target_role="coffee_container",
    )
    assert res.succeeded is True
    assert res.capability is not None
    assert res.capability.capability_id == "STIR_COFFEE"
    assert set(res.required_relations) == {"INSERTABLE_IN", "REACHES_BOTTOM"}

    # 2. End-to-end compiler check without FM specifying checker predicates
    raw_fm = {
        "status": "SUPPORTED",
        "task_summary": "Stir beverage",
        "functional_roles": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon"],
                "required_properties": [],
            },
            {
                "id": "r2",
                "entity_kind": "OBJECT",
                "function": "contain coffee",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup"],
                "required_properties": [],
            },
        ],
        "functional_relations": [],
        "interaction_groups": [
            {
                "id": "g1",
                "function": "stir contents",
                "tool_role": "r1",
                "target_role": "r2",
                "required_target_count": 2,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": [],  # FM omits explicit low-level checker predicates!
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("kitchen", "Stir coffee", raw_fm)
    assert len(graph.operation_groups) == 1
    grp = graph.operation_groups[0]
    assert grp.capability_id == "STIR_COFFEE"
    assert grp.function == "STIR_COFFEE"
    assert set(grp.required_relations) == {"INSERTABLE_IN", "REACHES_BOTTOM"}

    # Precondition provenance verification
    prov = grp.preconditions_provenance
    assert len(prov) == 2
    preds = {p["predicate"] for p in prov}
    assert preds == {"INSERTABLE_IN", "REACHES_BOTTOM"}
    for p in prov:
        assert p["provenance"] == "ROBOT_CAPABILITY_PRECONDITION"
        assert p["capability_id"] == "STIR_COFFEE"


def test_explicit_fm_fasten_derives_workshop_preconditions_v1_and_v2():
    """Explicit FM 'install/fasten component at target' instantiates all 3 workshop physical preconditions in V1 & V2."""
    # 1. Direct interpreter check
    res = interpret_operation(
        domain="workshop",
        raw_phrase="install/fasten component at target",
        source_role="driver",
        target_role="fastener",
        anchor_role="repair_target",
    )
    assert res.succeeded is True
    assert res.capability is not None
    assert res.capability.capability_id == "FASTEN_JOINT"
    assert len(res.physical_preconditions) == 3

    # 2. End-to-end compiler check on V1 document (no explicit relations listed by FM)
    raw_v1 = {
        "status": "SUPPORTED",
        "task_summary": "Fasten joint",
        "functional_roles": [
            {"id": "r1", "entity_kind": "OBJECT", "function": "drive screw", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screwdriver"], "required_properties": []},
            {"id": "r2", "entity_kind": "OBJECT", "function": "fasten joint", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw"], "required_properties": []},
            {"id": "r3", "entity_kind": "FIXED_TARGET", "function": "frame joint repair target", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw hole"], "required_properties": []},
        ],
        "functional_relations": [],
        "interaction_groups": [
            {
                "id": "g1",
                "function": "install/fasten component at target",
                "tool_role": "r1",
                "target_role": "r2",
                "required_target_count": 1,
                "usage_policy": "DEDICATED_PER_TARGET",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    graph_v1 = compile_candidate_graph("workshop", "Fasten joint", raw_v1)
    v1_preds = {r.predicate for r in graph_v1.relations}
    assert v1_preds == {"COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET"}
    for rel in graph_v1.relations:
        assert rel.provenance == "ROBOT_CAPABILITY_PRECONDITION"
        assert rel.capability_id == "FASTEN_JOINT"

    # 3. End-to-end compiler check on V2 document (where FM does not provide checker relations in functional_relations)
    raw_v2 = {
        "status": "SUPPORTED",
        "task_summary": "Fasten joint in V2 schema",
        "task_contract": {
            "functional_roles": [
                {"id": "role_driver", "entity_kind": "OBJECT", "function": "drive screw", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screwdriver"], "required_properties": []},
                {"id": "role_fastener", "entity_kind": "OBJECT", "function": "fasten joint", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw"], "required_properties": []},
                {"id": "role_target", "entity_kind": "FIXED_TARGET", "function": "frame joint repair target", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw hole"], "required_properties": []},
            ],
            "functional_relations": [],  # Completely omitted by FM!
            "operation_pairings": [
                {
                    "id": "op_fasten",
                    "operation": "install component at target",
                    "source_role": "role_driver",
                    "target_role": "role_fastener",
                    "operation_count": 1,
                }
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }
    graph_v2 = compile_candidate_graph("workshop", "Fasten joint V2", raw_v2)
    v2_preds = {r.predicate for r in graph_v2.relations}
    assert v2_preds == {"COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET"}
    for rel in graph_v2.relations:
        assert rel.provenance == "ROBOT_CAPABILITY_PRECONDITION"
        assert rel.capability_id == "FASTEN_JOINT"

    # Verify provenance in graph metadata
    prov_meta = graph_v2.metadata.get("precondition_provenance", [])
    assert len(prov_meta) == 3
    assert {p["predicate"] for p in prov_meta} == {"COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET"}


def test_explicit_fm_transfer_material_maps_to_pour_realization():
    """Explicit FM 'transfer material into container' maps to TRANSFER capability and yields POUR realization."""
    # 1. Direct interpreter check
    res = interpret_operation(
        domain="kitchen",
        raw_phrase="transfer material into container",
        source_role="coffee_source",
        target_role="coffee_container",
    )
    assert res.succeeded is True
    assert res.capability is not None
    assert res.capability.capability_id == "TRANSFER_CONTENT_TO_CONTAINER"
    assert res.planner_operation == "POUR"

    # 2. Compiler check
    raw_fm = {
        "status": "SUPPORTED",
        "task_summary": "Transfer coffee grounds into cup",
        "functional_roles": [
            {"id": "r_src", "entity_kind": "OBJECT", "function": "source of coffee", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["jar"], "required_properties": []},
            {"id": "r_cup", "entity_kind": "OBJECT", "function": "contain coffee", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["cup"], "required_properties": []},
        ],
        "functional_relations": [],
        "interaction_groups": [
            {
                "id": "g_pour",
                "function": "transfer material into container",
                "tool_role": "r_src",
                "target_role": "r_cup",
                "required_target_count": 1,
                "usage_policy": "DEDICATED_PER_TARGET",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("kitchen", "Pour coffee", raw_fm)
    assert len(graph.operation_groups) == 1
    grp = graph.operation_groups[0]
    assert grp.capability_id == "TRANSFER_CONTENT_TO_CONTAINER"
    assert grp.function == "POUR"

    # 3. Planner realization check: verify symbolic kitchen problem applicable_actions produces pour
    problem = KitchenSymbolicProblem({
        "role_assignments": {
            "coffee_targets": ["cup_1"],
            "soup_targets": [],
            "source_roles": {},
            "coffee_stirring": [],
            "soup_serving": [],
        },
        "capabilities": {
            "source_contains": [["kettle_1", "coffee"]],
            "initial_target_contents": [],
            "can_stir": [],
            "assigned_soup_utensil": [],
        },
        "requirements": {
            "home_region": "countertop",
            "serving_destination": "serving_area",
        },
        "objects": {
            "kettle_1": {"location": {"region_id": "countertop"}},
            "cup_1": {"location": {"region_id": "countertop"}},
        },
    })
    state = PlannerState(
        locations=(("kettle_1", "hand"), ("cup_1", "countertop")),
        held="kettle_1",
        contents=frozenset(),
        stirred=frozenset(),
    )
    actions = problem.applicable_actions(state)
    pour_actions = [a for a in actions if a.name == "pour"]
    assert len(pour_actions) == 1
    assert pour_actions[0].arguments == ("kettle_1", "cup_1")


def test_roles_only_no_explicit_operation_synthesizes_no_operation():
    """Gate 2 invariant: Roles alone NEVER create task operations or capability preconditions."""
    # 1. Kitchen pair: stirrer + container, no operations specified
    raw_kitchen_roles_only = {
        "status": "SUPPORTED",
        "task_summary": "Just roles",
        "functional_roles": [
            {"id": "r1", "entity_kind": "OBJECT", "function": "stir coffee", "required_count": 1, "binding_policy": "REUSABLE", "candidate_categories": ["spoon"], "required_properties": []},
            {"id": "r2", "entity_kind": "OBJECT", "function": "contain coffee", "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["cup"], "required_properties": []},
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    g_k = compile_candidate_graph("kitchen", "Roles only", raw_kitchen_roles_only)
    assert len(g_k.operation_groups) == 0
    assert len(g_k.relations) == 0

    # 2. Workshop pair: driver + fastener + target, no operations specified
    raw_workshop_roles_only = {
        "status": "SUPPORTED",
        "task_summary": "Just roles",
        "functional_roles": [
            {"id": "r1", "entity_kind": "OBJECT", "function": "drive screw", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screwdriver"], "required_properties": []},
            {"id": "r2", "entity_kind": "OBJECT", "function": "fasten joint", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw"], "required_properties": []},
            {"id": "r3", "entity_kind": "FIXED_TARGET", "function": "frame joint repair target", "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["screw hole"], "required_properties": []},
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    g_w = compile_candidate_graph("workshop", "Roles only", raw_workshop_roles_only)
    assert len(g_w.operation_groups) == 0
    assert len(g_w.relations) == 0


def test_workshop_equipment_return_capability():
    """Workshop equipment return capability maps to PLACE operator."""
    res = interpret_operation(
        domain="workshop",
        raw_phrase="return reusable equipment to workbench",
        source_role="driver",
        target_role="MAIN_WORKBENCH_ZONE",
    )
    assert res.succeeded is True
    assert res.capability is not None
    assert res.capability.capability_id == "RETURN_REUSABLE_ITEM_TO_SUPPORT"
    assert res.planner_operation == "PLACE"
