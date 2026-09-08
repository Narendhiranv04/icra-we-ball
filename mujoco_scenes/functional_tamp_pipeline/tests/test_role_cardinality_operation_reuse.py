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


def test_compiler_preserves_distinct_role_and_operation_level_reuse():
    graph = compile_candidate_graph("kitchen", "stir two targets", _compiler_document())
    role = graph.nodes["coffee_stirrer"]
    group = graph.operation_groups[0]

    assert role.binding_policy == "DISTINCT"
    assert role.minimum_count == 2
    assert role.maximum_count == 2
    assert group.usage_policy == "SEQUENTIAL_REUSE_ALLOWED"
    assert graph.metadata["canonicalization_trace"]["role_operation_reconciliations"] == [{
        "code": "ROLE_OPERATION_REUSE_RECONCILED",
        "raw_role_id": "tool",
        "canonical_role": "coffee_stirrer",
        "role_count": 2,
        "role_minimum_count": 2,
        "role_maximum_count": 2,
        "role_binding_policy": "DISTINCT",
        "operation_id": "stir_twice",
        "operation_target_count": 2,
        "operation_usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
        "resolution": "PRESERVE_ROLE_DISTINCTNESS_REUSE_REMAINS_OPTIONAL",
    }]


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


def test_archived_k1_preserves_all_fm_emitted_source_tool_counts():
    raw = json.loads(json.loads(K1_DIAGNOSTIC.read_text())["content"])
    graph = compile_candidate_graph("kitchen", raw["task_summary"], raw)

    for role_name in (
        "coffee_source", "water_source", "coffee_stirrer", "soup_eating_utensil"
    ):
        role = graph.nodes[role_name]
        assert role.count == role.minimum_count == role.maximum_count == 2
        assert role.binding_policy == "DISTINCT"
        assert role.min_count is None
        assert role.preference is None
