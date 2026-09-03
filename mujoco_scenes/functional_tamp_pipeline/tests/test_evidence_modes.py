from mujoco_scenes.functional_tamp_pipeline.grounding import (
    evaluate_node_for_role,
    ground_graph,
    resolve_evidence_components,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)


def test_semantic_only_ignores_geometric_predicates() -> None:
    role = FunctionalRole(
        name="tool",
        semantic_categories=("spoon",),
        unary_predicates=("ELONGATED_OBJECT",),
    )
    node = ObservedNode(
        instance_id="tool_1",
        canonical_category="spoon",
        unary_predicates={"ELONGATED_OBJECT": {"status": "FALSE"}},
    )

    assert evaluate_node_for_role(node, role, evidence_mode="semantic_only")[0] == "TRUE"
    assert evaluate_node_for_role(node, role, evidence_mode="joint")[0] == "FALSE"


def test_geometric_only_ignores_semantic_labels() -> None:
    role = FunctionalRole(
        name="tool",
        semantic_categories=("spoon",),
        unary_predicates=("ELONGATED_OBJECT",),
    )
    node = ObservedNode(
        instance_id="tool_1",
        canonical_category="remote_control",
        unary_predicates={"ELONGATED_OBJECT": {"status": "TRUE"}},
    )

    assert evaluate_node_for_role(node, role, evidence_mode="geometric_only")[0] == "TRUE"
    assert evaluate_node_for_role(node, role, evidence_mode="joint")[0] == "FALSE"


def test_component_mask_can_remove_unary_without_removing_semantics() -> None:
    role = FunctionalRole(
        name="tool",
        semantic_categories=("spoon",),
        unary_predicates=("ELONGATED_OBJECT",),
    )
    node = ObservedNode(
        instance_id="tool_1",
        canonical_category="spoon",
        unary_predicates={"ELONGATED_OBJECT": {"status": "FALSE"}},
    )

    assert evaluate_node_for_role(
        node, role, evidence_components=("semantic", "binary")
    )[0] == "TRUE"
    assert evaluate_node_for_role(
        node, role, evidence_components=("semantic", "unary")
    )[0] == "FALSE"


def test_component_mask_can_remove_binary_without_removing_local_checks() -> None:
    roles = {
        "tool": FunctionalRole(name="tool", semantic_categories=("spoon",)),
        "target": FunctionalRole(name="target", semantic_categories=("cup",)),
    }
    graph_f = FunctionalRequirementGraph(
        domain="test",
        task_instruction="test",
        nodes=roles,
        relations=(FunctionalRelation("tool", "INSERTABLE_IN", "target"),),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(ObservedNode("spoon_1", canonical_category="spoon"))
    graph_o.add_node(ObservedNode("cup_1", canonical_category="cup"))
    graph_o.add_relation(ObservedRelation(
        "spoon_1", "INSERTABLE_IN", "cup_1", status="FALSE"
    ))

    assert not ground_graph(graph_f, graph_o).complete
    result = ground_graph(
        graph_f,
        graph_o,
        {"evidence_components": ("semantic", "unary"), "search_exhausted": True},
    )
    assert result.complete
    assert result.evidence["evidence_components"] == ["semantic", "unary"]


def test_component_mask_validation_rejects_empty_or_unknown_components() -> None:
    assert resolve_evidence_components(evidence_components="semantic,unary") == {
        "semantic", "unary"
    }
    for invalid in ("", "semantic,unknown"):
        try:
            resolve_evidence_components(evidence_components=invalid)
        except ValueError:
            continue
        raise AssertionError(f"Expected invalid component mask: {invalid!r}")
