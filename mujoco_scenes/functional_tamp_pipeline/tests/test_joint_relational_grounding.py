"""Joint semantic and relational grounding contract tests."""

from dataclasses import replace

from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation, FunctionalRequirementGraph, FunctionalRole, OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode, ObservedRelation, ObservedSceneGraph,
)


def _role(name, categories, count=1, policy="DISTINCT"):
    return FunctionalRole(
        name=name, entity_kind="OBJECT", count=count, binding_policy=policy,
        semantic_categories=tuple(categories), canonical_role_candidates=(name,),
    )


def _node(instance_id, category=None, *, unknown=False):
    belief = ({"status": "UNKNOWN", "reason_codes": ["NO_ASSOCIATED_DETECTION"]} if unknown else {})
    return ObservedNode(instance_id=instance_id, entity_kind="OBJECT",
                        canonical_category=category, semantic_labels=belief)


def _stir_graph(operation=True):
    groups = ()
    if operation:
        groups = (OperationGroup(
            id="stir", function="STIR_COFFEE", capability_id="STIR_COFFEE",
            tool_role="stirrer", target_role="cup", required_target_count=1,
            usage_policy="DEDICATED_PER_TARGET", required_relations=("INSERTABLE_IN",),
        ),)
    return FunctionalRequirementGraph(
        domain="kitchen", task_instruction="stir",
        nodes={"stirrer": _role("stirrer", ("spoon",)), "cup": _role("cup", ("cup",))},
        relations=(FunctionalRelation("stirrer", "INSERTABLE_IN", "cup"),),
        operation_groups=groups,
    )


def _scene(tool, relation_status="TRUE"):
    graph = ObservedSceneGraph()
    graph.add_node(tool)
    graph.add_node(_node("cup_1", "cup"))
    graph.add_relation(ObservedRelation("spoon_1", "INSERTABLE_IN", "cup_1", relation_status))
    return graph


def test_semantic_true_and_relations_true_complete_with_operation_binding():
    result = ground_graph(_stir_graph(), _scene(_node("spoon_1", "spoon")))
    assert result.complete
    assert result.operation_bindings["stir"][0]["capability_id"] == "STIR_COFFEE"
    assert result.evidence["operation_binding_complete"] is True


def test_semantic_false_is_absolute_veto_even_when_relation_true():
    result = ground_graph(_stir_graph(), _scene(_node("spoon_1", "remote_control")))
    assert not result.complete
    assert result.failure_kind == "OBJECT_DISCOVERY_FAILURE"


def test_semantic_unknown_is_accepted_only_when_required_relations_true():
    result = ground_graph(_stir_graph(), _scene(_node("spoon_1", unknown=True)))
    assert result.complete
    provenance = result.evidence["binding_provenance"]["stirrer:spoon_1"]
    assert provenance["grounding_mode"] == "RELATIONALLY_VERIFIED_GROUNDING"
    assert provenance["semantic_status"] == "UNKNOWN"


def test_semantic_unknown_with_false_relation_rejects_assignment():
    result = ground_graph(_stir_graph(), _scene(_node("spoon_1", unknown=True), "FALSE"))
    assert not result.complete
    assert result.failure_kind == "FUNCTIONAL_ASSIGNMENT_FAILURE"


def test_semantic_unknown_with_unknown_relation_requires_search_then_assignment_failure():
    scene = _scene(_node("spoon_1", unknown=True), "UNKNOWN")
    before = ground_graph(_stir_graph(), scene, {"search_exhausted": False})
    after = ground_graph(_stir_graph(), scene, {"search_exhausted": True})
    assert before.status == "INCOMPLETE" and before.failure_kind is None
    assert after.failure_kind == "FUNCTIONAL_ASSIGNMENT_FAILURE"


def test_missing_individuals_and_joint_mismatch_have_distinct_failures():
    assert ground_graph(_stir_graph(), ObservedSceneGraph()).failure_kind == "OBJECT_DISCOVERY_FAILURE"
    mismatch = _scene(_node("spoon_1", "spoon"), "FALSE")
    assert ground_graph(_stir_graph(), mismatch).failure_kind == "FUNCTIONAL_ASSIGNMENT_FAILURE"


def test_distinct_cardinality_never_reuses_one_observed_object():
    graph = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="two tools",
        nodes={"stirrer": _role("stirrer", ("spoon",), count=2)},
    )
    scene = ObservedSceneGraph()
    scene.add_node(_node("spoon_1", "spoon"))
    assert ground_graph(graph, scene).failure_kind == "OBJECT_DISCOVERY_FAILURE"


def test_reusable_role_remains_groundable():
    graph = _stir_graph()
    graph = replace(graph, nodes={**graph.nodes, "stirrer": replace(graph.nodes["stirrer"], binding_policy="REUSABLE")})
    assert ground_graph(graph, _scene(_node("spoon_1", "spoon"))).complete
