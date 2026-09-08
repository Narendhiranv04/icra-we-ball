"""Stage 4 Test Suite — Relation Handling: Task/Causal Semantics vs Physical Verifiers.

Validates Gate 4 requirements from CORRECTIVE_RECOVERY_PLAN_ANTIGRAVITY_FLASH38.md:
1. Explicit physical relation -> physical verifier (INSERTABLE_IN, REACHES_BOTTOM, etc.).
2. Explicit causal relation -> preserved semantic edge (PROVIDES_MATERIAL_TO, ACTS_ON, etc.).
3. Explicit operation -> capability-derived physical preconditions.
4. Unknown required relation -> fails closed (UNINTERPRETABLE_REQUIRED_RELATION, contract incomplete).
5. Endpoint signature alone -> never creates a relation without semantic evidence.
6. Structural consistency and serialization round-trip.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import (
    interpret_relation,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    compile_candidate_graph,
)
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import (
    validate_runtime_gf,
)


def test_explicit_physical_relation_to_verifier():
    """Explicit physical relation compiles to a physical verifier in graph.relations."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Stir coffee in mug",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_stirrer",
                    "entity_kind": "OBJECT",
                    "function": "Tool used to mix ingredients inside the coffee serving vessel.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["spoon", "stirrer"],
                    "required_properties": [],
                },
                {
                    "id": "role_mug",
                    "entity_kind": "OBJECT",
                    "function": "Receives coffee and water ingredients and is the target of stirring action.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["mug", "cup"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "subject_role": "role_stirrer",
                    "relation": "fits inside container opening",
                    "object_role": "role_mug",
                    "required": True,
                },
                {
                    "subject_role": "role_stirrer",
                    "relation": "touches bottom of the mug",
                    "object_role": "role_mug",
                    "required": True,
                },
            ],
            "interaction_groups": [],
        },
    }

    graph = compile_candidate_graph("kitchen", "Stir coffee in mug", raw)
    validate_runtime_gf(graph)

    # Both relations compiled as physical verifiers
    pred_names = {r.predicate for r in graph.relations}
    assert "INSERTABLE_IN" in pred_names
    assert "REACHES_BOTTOM" in pred_names

    for r in graph.relations:
        assert r.category == "PHYSICAL_VERIFIER"
        assert r.provenance == "EXPLICIT_REQUIREMENT"

    # Task causal relations empty
    assert len(graph.task_causal_relations) == 0


def test_explicit_causal_relation_to_preserved_semantic_edge():
    """Explicit causal relation compiles to graph.task_causal_relations, preserving narrative dependency."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Prepare and stir coffee",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_source",
                    "entity_kind": "OBJECT",
                    "function": "Dispenses hot water or brewed coffee into container.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["kettle", "pot"],
                    "required_properties": [],
                },
                {
                    "id": "role_mug",
                    "entity_kind": "OBJECT",
                    "function": "Receives coffee and water ingredients and is the target of stirring action.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["mug", "cup"],
                    "required_properties": [],
                },
                {
                    "id": "role_stirrer",
                    "entity_kind": "OBJECT",
                    "function": "Tool used to mix ingredients inside the coffee serving vessel.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["spoon", "stirrer"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "subject_role": "role_source",
                    "relation": "provides material to coffee container",
                    "object_role": "role_mug",
                    "required": True,
                },
                {
                    "subject_role": "role_stirrer",
                    "relation": "manipulates interior of coffee serving vessel",
                    "object_role": "role_mug",
                    "required": True,
                },
            ],
            "interaction_groups": [
                {
                    "id": "coffee_stirring",
                    "function": "STIR_COFFEE",
                    "tool_role": "role_stirrer",
                    "target_role": "role_mug",
                    "required_target_count": 1,
                    "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                },
            ],
        },
    }

    graph = compile_candidate_graph("kitchen", "Prepare and stir coffee", raw)
    validate_runtime_gf(graph)

    # Physical relations in graph.relations do not contain PROVIDES_MATERIAL_TO or ACTS_ON
    phys_preds = {r.predicate for r in graph.relations}
    assert "PROVIDES_MATERIAL_TO" not in phys_preds
    assert "ACTS_ON" not in phys_preds

    # Preserved in task_causal_relations
    causal_preds = {r.predicate for r in graph.task_causal_relations}
    assert "PROVIDES_MATERIAL_TO" in causal_preds
    assert "ACTS_ON" in causal_preds

    for r in graph.task_causal_relations:
        assert r.category == "TASK_CAUSAL_SEMANTICS"
        assert r.provenance == "TASK_CAUSAL_SEMANTICS"

    # Online executable contract remains complete
    assert graph.online_executable_contract_complete is True


def test_direction_normalized_causal_relation():
    """Inverse causal relation normalized in direction."""
    res = interpret_relation(
        domain="kitchen",
        raw_phrase="coffee container receives contents from coffee source",
        subject_role="coffee_container",
        object_role="coffee_source",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        required=True,
    )
    assert res.succeeded is True
    assert res.status == "TASK_CAUSAL_SEMANTIC_MATCH"
    assert res.category == "TASK_CAUSAL_SEMANTICS"
    assert res.direction_normalized is True
    assert len(res.interpreted_predicates) == 1
    p = res.interpreted_predicates[0]
    assert p.subject_role == "coffee_source"
    assert p.predicate_name == "PROVIDES_MATERIAL_TO"
    assert p.object_role == "coffee_container"


def test_workshop_causal_and_physical_separation():
    """Workshop separates tool acts on component (causal) from mechanical engagement (physical)."""
    # 1. Causal phrase
    res_causal = interpret_relation(
        domain="workshop",
        raw_phrase="tool acts on fastening component",
        subject_role="driver",
        object_role="fastener",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        required=True,
    )
    assert res_causal.succeeded is True
    assert res_causal.category == "TASK_CAUSAL_SEMANTICS"
    assert res_causal.interpreted_predicates[0].predicate_name == "ACTS_ON"

    # 2. Physical verifier phrase
    res_phys = interpret_relation(
        domain="workshop",
        raw_phrase="mechanically engages screw head",
        subject_role="driver",
        object_role="fastener",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        required=True,
    )
    assert res_phys.succeeded is True
    assert res_phys.category == "PHYSICAL_VERIFIER"
    assert res_phys.interpreted_predicates[0].predicate_name == "COMPATIBLE_WITH"


def test_explicit_operation_capability_derived_preconditions():
    """Operation capability provides physical preconditions without redundant FM relations."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Stir coffee",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_spoon",
                    "entity_kind": "OBJECT",
                    "function": "Tool used to mix ingredients inside the coffee serving vessel.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["spoon", "stirrer"],
                    "required_properties": [],
                },
                {
                    "id": "role_cup",
                    "entity_kind": "OBJECT",
                    "function": "Receives coffee and water ingredients and is the target of stirring action.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["cup", "mug"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [],  # FM omits explicit physical relations
            "interaction_groups": [
                {
                    "id": "stir_group",
                    "function": "STIR_COFFEE",
                    "tool_role": "role_spoon",
                    "target_role": "role_cup",
                    "required_target_count": 1,
                    "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                },
            ],
        },
    }

    graph = compile_candidate_graph("kitchen", "Stir coffee", raw)
    validate_runtime_gf(graph)

    # Operation group has capability preconditions
    assert len(graph.operation_groups) == 1
    grp = graph.operation_groups[0]
    assert "INSERTABLE_IN" in grp.required_relations
    assert "REACHES_BOTTOM" in grp.required_relations
    assert grp.capability_id == "STIR_COFFEE"

    # Preconditions provenance recorded
    assert len(grp.preconditions_provenance) >= 2
    for prov in grp.preconditions_provenance:
        assert prov["provenance"] == "ROBOT_CAPABILITY_PRECONDITION"
        assert prov["capability_id"] == "STIR_COFFEE"

    # Contract complete without redundant explicit relations
    assert graph.online_executable_contract_complete is True


def test_unknown_required_relation_fails_closed():
    """Unknown required relation triggers UNINTERPRETABLE_REQUIRED_RELATION and fails closed."""
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Drive fastener with tilt",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_tool",
                    "entity_kind": "OBJECT",
                    "function": "An implement used to manipulate or install the fastening component.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["screwdriver"],
                    "required_properties": [],
                },
                {
                    "id": "role_screw",
                    "entity_kind": "OBJECT",
                    "function": "A physical item capable of connecting or securing elements at the marked location.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["screw"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "subject_role": "role_tool",
                    "relation": "must maintain 45 degree tilt during operation",
                    "object_role": "role_screw",
                    "required": True,
                },
            ],
            "interaction_groups": [],
        },
    }

    graph = compile_candidate_graph("workshop", "Drive fastener with tilt", raw)

    # Fail closed: contract incomplete
    assert graph.online_executable_contract_complete is False
    assert graph.required_contract_complete is False

    # Check unresolved required relations trace
    unresolved_trace = graph.metadata["canonicalization_trace"]["unresolved_required_relations"]
    assert len(unresolved_trace) >= 1
    assert unresolved_trace[0]["status"] == "UNINTERPRETABLE_REQUIRED_RELATION"
    assert "tilt" in unresolved_trace[0]["raw_phrase"]


def test_endpoint_signature_alone_never_creates_relation():
    """Between driver and fastener, nonsensical text fails closed and does NOT guess COMPATIBLE_WITH."""
    res = interpret_relation(
        domain="workshop",
        raw_phrase="bananas on the moon",
        subject_role="driver",
        object_role="fastener",
        subject_kind="OBJECT",
        object_kind="OBJECT",
        required=True,
    )
    assert res.succeeded is False
    assert res.status == "UNINTERPRETABLE_REQUIRED_RELATION"
    assert len(res.interpreted_predicates) == 0

    # Test that compiler fails closed
    raw = {
        "status": "SUPPORTED",
        "task_summary": "Drive fastener",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_tool",
                    "entity_kind": "OBJECT",
                    "function": "An implement used to manipulate or install the fastening component.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["screwdriver"],
                    "required_properties": [],
                },
                {
                    "id": "role_screw",
                    "entity_kind": "OBJECT",
                    "function": "A physical item capable of connecting or securing elements at the marked location.",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["screw"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "subject_role": "role_tool",
                    "relation": "bananas on the moon",
                    "object_role": "role_screw",
                    "required": True,
                },
            ],
            "interaction_groups": [],
        },
    }

    graph = compile_candidate_graph("workshop", "Drive fastener", raw)
    assert graph.online_executable_contract_complete is False
    assert len(graph.relations) == 0  # No relation manufactured!


def test_graph_serialization_roundtrip_preserves_categories():
    """Serialization to_dict / from_dict preserves task_causal_relations and category."""
    role1 = FunctionalRole(name="driver", entity_kind="OBJECT", count=1)
    role2 = FunctionalRole(name="fastener", entity_kind="OBJECT", count=1)

    rel_phys = FunctionalRelation(
        subject_role="driver",
        predicate="COMPATIBLE_WITH",
        object_role="fastener",
        category="PHYSICAL_VERIFIER",
        provenance="EXPLICIT_REQUIREMENT",
    )
    rel_causal = FunctionalRelation(
        subject_role="driver",
        predicate="ACTS_ON",
        object_role="fastener",
        category="TASK_CAUSAL_SEMANTICS",
        provenance="TASK_CAUSAL_SEMANTICS",
    )

    graph = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Drive fastener",
        nodes={"driver": role1, "fastener": role2},
        relations=(rel_phys,),
        task_causal_relations=(rel_causal,),
    )
    graph.validate()
    validate_runtime_gf(graph)

    # Roundtrip
    data = graph.to_dict()
    assert "task_causal_relations" in data
    assert len(data["task_causal_relations"]) == 1
    assert data["task_causal_relations"][0]["category"] == "TASK_CAUSAL_SEMANTICS"
    assert data["task_causal_relations"][0]["predicate"] == "ACTS_ON"

    reconstructed = FunctionalRequirementGraph.from_dict(data)
    reconstructed.validate()
    validate_runtime_gf(reconstructed)

    assert len(reconstructed.relations) == 1
    assert reconstructed.relations[0].category == "PHYSICAL_VERIFIER"
    assert reconstructed.relations[0].predicate == "COMPATIBLE_WITH"

    assert len(reconstructed.task_causal_relations) == 1
    assert reconstructed.task_causal_relations[0].category == "TASK_CAUSAL_SEMANTICS"
    assert reconstructed.task_causal_relations[0].predicate == "ACTS_ON"
