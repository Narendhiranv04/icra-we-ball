from __future__ import annotations

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    project_functional_graph_to_roles,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
    TaskEffectRelation,
)


def _role(name: str) -> FunctionalRole:
    return FunctionalRole(name=name, count=2, binding_policy="DISTINCT")


def _graph() -> FunctionalRequirementGraph:
    return FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction="projection test",
        nodes={name: _role(name) for name in ("A", "B", "C")},
        relations=(FunctionalRelation("A", "P", "B"),),
        task_causal_relations=(FunctionalRelation(
            "A", "CAUSAL", "B", category="TASK_CAUSAL_SEMANTICS"
        ),),
        task_effect_relations=(
            TaskEffectRelation("A", "CONTAINS", "B"),
            TaskEffectRelation("A", "CONTAINS", "material", object_is_literal=True),
        ),
        operation_groups=(
            OperationGroup(
                id="with_context",
                function="OP",
                tool_role="A",
                target_role="B",
                context_role="C",
                required_target_count=1,
                usage_policy="DEDICATED_PER_TARGET",
            ),
            OperationGroup(
                id="without_context",
                function="OP",
                tool_role="A",
                target_role="B",
                required_target_count=1,
                usage_policy="DEDICATED_PER_TARGET",
            ),
        ),
        metadata={"online_executable_contract_complete": True},
    )


def test_physical_relation_projection_retains_only_complete_edges():
    graph = _graph()

    retained = project_functional_graph_to_roles(graph, {"A", "B"})
    removed = project_functional_graph_to_roles(graph, {"A", "C"})

    assert retained.relations == graph.relations
    assert removed.relations == ()
    retained.validate()
    removed.validate()


def test_task_causal_projection_retains_only_complete_edges():
    graph = _graph()

    assert project_functional_graph_to_roles(
        graph, {"A", "B"}
    ).task_causal_relations == graph.task_causal_relations
    projected = project_functional_graph_to_roles(graph, {"A"})
    assert projected.task_causal_relations == ()
    projected.validate()


def test_physical_role_task_effect_requires_both_endpoints():
    graph = _graph()

    both = project_functional_graph_to_roles(graph, {"A", "B"})
    carrier_only = project_functional_graph_to_roles(graph, {"A"})

    assert graph.task_effect_relations[0] in both.task_effect_relations
    assert graph.task_effect_relations[0] not in carrier_only.task_effect_relations
    both.validate()
    carrier_only.validate()


def test_literal_task_effect_depends_only_on_carrier():
    graph = _graph()

    carrier = project_functional_graph_to_roles(graph, {"A"})
    no_carrier = project_functional_graph_to_roles(graph, {"B"})

    literal = graph.task_effect_relations[1]
    assert literal in carrier.task_effect_relations
    assert literal not in no_carrier.task_effect_relations
    assert "material" not in carrier.nodes
    carrier.validate()
    no_carrier.validate()


def test_operation_group_projection_requires_all_role_endpoints():
    graph = _graph()

    all_roles = project_functional_graph_to_roles(graph, {"A", "B", "C"})
    no_context = project_functional_graph_to_roles(graph, {"A", "B"})
    no_target = project_functional_graph_to_roles(graph, {"A", "C"})

    assert {group.id for group in all_roles.operation_groups} == {
        "with_context", "without_context"
    }
    assert [group.id for group in no_context.operation_groups] == ["without_context"]
    assert no_target.operation_groups == ()


def test_projection_does_not_mutate_original_graph():
    graph = _graph()
    before = graph.to_dict()

    projected = project_functional_graph_to_roles(graph, {"A"})

    assert graph.to_dict() == before
    assert set(graph.nodes) == {"A", "B", "C"}
    assert len(graph.relations) == 1
    assert len(graph.task_causal_relations) == 1
    assert len(graph.task_effect_relations) == 2
    assert len(graph.operation_groups) == 2
    assert projected.metadata is graph.metadata
