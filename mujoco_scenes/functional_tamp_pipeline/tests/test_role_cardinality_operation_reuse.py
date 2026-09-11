from __future__ import annotations

import json
from pathlib import Path

from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode, ObservedSceneGraph
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph


ROOT = Path(__file__).resolve().parents[3]
K1_DIAGNOSTIC = (
    ROOT
    / "benchmark_reports/corrective_recovery_final_32x1_20260909T013814IST"
    / "kitchen/K1/vlm/fm_diagnostics/fm_call_001.json"
)


def _compiler_document(tool_policy: str = "DISTINCT", tool_count: int = 2) -> dict:
    return {
        "functional_roles": [
            {
                "id": "tool",
                "entity_kind": "OBJECT",
                "function": "tool for stirring coffee",
                "required_count": tool_count,
                "binding_policy": tool_policy,
                "candidate_categories": ["spoon"],
                "required_properties": [],
            },
            {
                "id": "target",
                "entity_kind": "OBJECT",
                "function": "container for coffee",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup"],
                "required_properties": [],
            },
        ],
        "functional_relations": [],
        "interaction_groups": [{
            "id": "stir_twice",
            "function": "stir coffee",
            "tool_role": "tool",
            "target_role": "target",
            "required_target_count": 2,
            "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
            "required_relations": [],
        }],
        "inspectable_regions": [],
        "inspection_order": [],
    }


def _grounding_graph(
    *, tool_count: int, target_count: int, tool_policy: str, usage_policy: str
) -> FunctionalRequirementGraph:
    return FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="apply tools to targets",
        nodes={
            "tool": FunctionalRole(
                name="tool", count=tool_count, binding_policy=tool_policy,
                semantic_categories=("tool",),
            ),
            "target": FunctionalRole(
                name="target", count=target_count, binding_policy="DISTINCT",
                semantic_categories=("target",),
            ),
        },
        operation_groups=(OperationGroup(
            id="apply",
            function="APPLY",
            tool_role="tool",
            target_role="target",
            required_target_count=target_count,
            usage_policy=usage_policy,
        ),),
    )


def _observed(tool_count: int, target_count: int) -> ObservedSceneGraph:
    graph = ObservedSceneGraph()
    for index in range(tool_count):
        graph.add_node(ObservedNode(instance_id=f"tool_{index + 1}", canonical_category="tool"))
    for index in range(target_count):
        graph.add_node(ObservedNode(instance_id=f"target_{index + 1}", canonical_category="target"))
    return graph


def test_compiler_resolves_an_isolated_distinct_field_on_a_reusable_implement():
    """A stirrer the model called DISTINCT purely because two coffees are stirred.

    The task uses it twice, so ``count`` stays two.  Nothing in the contract asks
    for two *separate* stirrers -- not the role's own words, not the operation's
    phrase, no second declaration, no one-for-one pairing -- and the runtime
    declares a stirrer reusable across applications, so two occasions come down
    to one physical spoon.
    """
    graph = compile_candidate_graph("kitchen", "stir two targets", _compiler_document())
    role = graph.nodes["coffee_stirrer"]
    group = graph.operation_groups[0]

    assert role.count == 2, "the number of task uses the model stated is preserved"
    assert role.binding_policy == "REUSABLE"
    assert role.minimum_count == 1
    override = [
        row for row in graph.metadata["canonicalization_trace"]["roles"]
        if row.get("status") == "GLOBAL_GRAPH_BINDING_POLICY_CONSISTENCY_OVERRIDE"
    ]
    assert [row["canonical_role"] for row in override] == ["coffee_stirrer"]
    assert override[0]["raw_binding_policy"] == "DISTINCT"
    assert override[0]["separate_identity_evidence"] == []
    assert group.usage_policy == "SEQUENTIAL_REUSE_ALLOWED"
    # No role/operation reuse tension is left to reconcile: the reconciliation
    # exists to record a role the model insisted was DISTINCT while its
    # operation allowed reuse, and that disagreement is what has just been
    # resolved.
    assert graph.metadata["canonicalization_trace"]["role_operation_reconciliations"] == []


def test_explicit_reusable_role_remains_single_without_conflict():
    graph = compile_candidate_graph(
        "kitchen", "reuse one tool", _compiler_document("REUSABLE", 1)
    )
    role = graph.nodes["coffee_stirrer"]

    assert role.binding_policy == "REUSABLE"
    assert role.minimum_count == role.maximum_count == 1
    assert not graph.metadata["canonicalization_trace"]["role_operation_reconciliations"]


def test_sequential_matching_uses_two_distinct_tools_for_two_targets():
    graph = _grounding_graph(
        tool_count=2, target_count=2, tool_policy="DISTINCT",
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
    )
    result = ground_graph(graph, _observed(2, 2))

    assert result.complete
    bindings = result.operation_bindings["apply"]
    assert len({binding["tool_id"] for binding in bindings}) == 2


def test_sequential_matching_covers_distinct_minimum_before_reuse():
    graph = _grounding_graph(
        tool_count=2, target_count=3, tool_policy="DISTINCT",
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
    )
    result = ground_graph(graph, _observed(2, 3))

    assert result.complete
    bindings = result.operation_bindings["apply"]
    assert len(bindings) == 3
    assert len({binding["tool_id"] for binding in bindings}) == 2


def test_reusable_single_tool_may_cover_multiple_targets():
    graph = _grounding_graph(
        tool_count=1, target_count=2, tool_policy="REUSABLE",
        usage_policy="SEQUENTIAL_REUSE_ALLOWED",
    )
    result = ground_graph(graph, _observed(1, 2))

    assert result.complete
    assert {binding["tool_id"] for binding in result.operation_bindings["apply"]} == {"tool_1"}


def test_dedicated_operation_rejects_one_tool_for_two_targets():
    graph = _grounding_graph(
        tool_count=1, target_count=2, tool_policy="REUSABLE",
        usage_policy="DEDICATED_PER_TARGET",
    )
    result = ground_graph(graph, _observed(1, 2))

    assert not result.complete
    assert result.operation_bindings == {}


def test_distinct_role_requires_distinct_observed_instances():
    graph = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="require two tools",
        nodes={
            "tool": FunctionalRole(
                name="tool", count=2, binding_policy="DISTINCT",
                semantic_categories=("tool",),
            )
        },
    )

    one = ground_graph(graph, _observed(1, 0))
    two = ground_graph(graph, _observed(2, 0))
    assert not one.complete
    assert two.complete
    assert set(two.assignment["tool"]) == {"tool_1", "tool_2"}


def test_archived_k1_preserves_every_fm_emitted_count_and_resolves_only_reuse():
    """The model's counts survive; only how they come down to objects is re-read.

    This contract declares four participants "2 DISTINCT".  ``required_count``
    is the number of task uses and is preserved for all four -- nothing the
    model stated is discarded.  What differs is the minimum number of separate
    objects each needs, and that follows the runtime's own reuse declaration
    once the contract turns out to say nothing about separate identity: a jar
    poured from twice and a stirrer used twice are one object each, while the
    eating utensil the runtime calls one-per-application stays two.
    """
    raw = json.loads(json.loads(K1_DIAGNOSTIC.read_text())["content"])
    graph = compile_candidate_graph("kitchen", raw["task_summary"], raw)

    for role_name in (
        "coffee_source", "water_source", "coffee_stirrer", "soup_eating_utensil"
    ):
        assert graph.nodes[role_name].count == 2, role_name

    for role_name in ("coffee_source", "water_source", "coffee_stirrer"):
        role = graph.nodes[role_name]
        assert role.binding_policy == "REUSABLE", role_name
        assert role.minimum_count == 1, role_name

    utensil = graph.nodes["soup_eating_utensil"]
    assert utensil.binding_policy == "DISTINCT"
    assert utensil.minimum_count == 2


def test_separate_identity_wording_keeps_a_reusable_implement_distinct():
    """The adversarial case: the model says the stirrers differ, so they differ."""
    document = _compiler_document()
    document["functional_roles"][0]["function"] = (
        "each cup is stirred with its own separate stirring spoon"
    )
    graph = compile_candidate_graph("kitchen", "stir two targets", document)
    role = graph.nodes["coffee_stirrer"]

    assert role.binding_policy == "DISTINCT"
    assert role.minimum_count == 2
    assert not [
        row for row in graph.metadata["canonicalization_trace"]["roles"]
        if row.get("status") == "GLOBAL_GRAPH_BINDING_POLICY_CONSISTENCY_OVERRIDE"
    ]


def test_scene_inventory_cannot_change_binding_policy_resolution():
    """The reading is the same whatever the scene turns out to hold.

    Cardinality is settled while compiling, before any candidate is looked at,
    so a scene short of spoons can never be the reason a requirement is re-read.
    """
    document = _compiler_document()
    first = compile_candidate_graph("kitchen", "stir two targets", document)
    second = compile_candidate_graph("kitchen", "stir two targets", document)
    assert first.nodes["coffee_stirrer"].binding_policy == "REUSABLE"
    assert second.nodes["coffee_stirrer"].binding_policy == "REUSABLE"
    # compile_candidate_graph takes no observed scene at all, which is what makes
    # the independence structural rather than a property of this fixture.
    import inspect
    assert "graph_o" not in inspect.signature(compile_candidate_graph).parameters
    assert "observed" not in inspect.signature(compile_candidate_graph).parameters
