from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph, resolved_functional_graph
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    ProvisionalRelationConstraint,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider


def _unknown_node(instance_id):
    return ObservedNode(
        instance_id=instance_id, entity_kind="OBJECT", canonical_category=None,
        semantic_labels={"status": "UNKNOWN", "reason_codes": ["NO_ASSOCIATED_DETECTION"]},
    )


def _graph():
    tool = FunctionalRole(
        name="fm_role__implement", raw_role_id="implement", entity_kind="OBJECT",
        semantic_categories=("spoon",), canonical_role_candidates=("coffee_stirrer", "soup_eating_utensil"),
        role_resolution_status="AMBIGUOUS_ROLE_TYPE",
    )
    vessel = FunctionalRole(
        name="fm_role__vessel", raw_role_id="vessel", entity_kind="OBJECT",
        semantic_categories=("cup", "mug", "bowl"), canonical_role_candidates=("coffee_container", "soup_container"),
        role_resolution_status="AMBIGUOUS_ROLE_TYPE",
    )
    constraint = ProvisionalRelationConstraint(
        raw_subject_role="implement", raw_object_role="vessel",
        subject_node=tool.name, object_node=vessel.name,
        semantic_candidates=(),
        allowed_canonical_role_pairs=(
            ("coffee_stirrer", "coffee_container", "REACHES_BOTTOM", "PHYSICAL_VERIFIER"),
            ("soup_eating_utensil", "soup_container", "INSERTABLE_IN", "PHYSICAL_VERIFIER"),
        ),
    )
    return FunctionalRequirementGraph(
        domain="kitchen", task_instruction="use an elongated implement",
        nodes={tool.name: tool, vessel.name: vessel},
        provisional_relation_constraints=(constraint,),
    )


def _scene(*predicates, tool_category=None, vessel_category=None):
    graph = ObservedSceneGraph()
    graph.add_node(ObservedNode("implement_1", "OBJECT", canonical_category=tool_category,
                                semantic_labels={"status": "UNKNOWN"} if tool_category is None else {}))
    graph.add_node(ObservedNode("vessel_1", "OBJECT", canonical_category=vessel_category,
                                semantic_labels={"status": "UNKNOWN"} if vessel_category is None else {}))
    for predicate in predicates:
        graph.add_relation(ObservedRelation("implement_1", predicate, "vessel_1", "TRUE"))
    return graph


def test_go_relation_resolves_coffee_stirrer_type_and_object():
    result = ground_graph(_graph(), _scene("REACHES_BOTTOM"))
    assert result.complete
    assert result.resolved_role_types == {"implement": "coffee_stirrer", "vessel": "coffee_container"}
    assert result.assignment["coffee_stirrer"] == "implement_1"
    assert result.resolved_graph["provisional_relation_constraints"] == []
    resolved = resolved_functional_graph(_graph(), result)
    assert set(resolved.nodes) == {"coffee_stirrer", "coffee_container"}
    assert resolved.provisional_relation_constraints == ()


def test_go_relation_resolves_soup_utensil_type_and_object():
    result = ground_graph(_graph(), _scene("INSERTABLE_IN"))
    assert result.complete
    assert result.resolved_role_types["implement"] == "soup_eating_utensil"
    assert result.assignment["soup_eating_utensil"] == "implement_1"


def test_materially_distinct_valid_types_remain_assignment_failure():
    result = ground_graph(_graph(), _scene("REACHES_BOTTOM", "INSERTABLE_IN"))
    assert not result.complete
    assert result.failure_kind == "FUNCTIONAL_ASSIGNMENT_FAILURE"
    assert result.unresolved_constraints == ("AMBIGUOUS_FUNCTIONAL_ASSIGNMENT",)


def test_semantic_false_type_cannot_be_rescued_by_true_relation():
    # A cup is FALSE for soup_container, so the all-TRUE soup relation cannot
    # rescue that type; the coffee interpretation remains uniquely valid.
    result = ground_graph(
        _graph(), _scene("REACHES_BOTTOM", "INSERTABLE_IN", vessel_category="cup")
    )
    assert result.complete
    assert result.resolved_role_types["vessel"] == "coffee_container"


def test_semantic_unknown_type_is_resolved_by_true_relation():
    result = ground_graph(_graph(), _scene("REACHES_BOTTOM"))
    provenance = result.evidence["binding_provenance"]["coffee_stirrer:implement_1"]
    assert provenance["semantic_status"] == "UNKNOWN"
    assert provenance["grounding_mode"] == "RELATIONALLY_VERIFIED_GROUNDING"


def test_multiple_predicate_interpretations_are_resolved_by_go_not_sort_order():
    graph = _graph()
    constraint = ProvisionalRelationConstraint(
        raw_subject_role="implement", raw_object_role="vessel",
        subject_node="fm_role__implement", object_node="fm_role__vessel",
        semantic_candidates=(),
        allowed_canonical_role_pairs=(
            ("coffee_stirrer", "coffee_container", "INSERTABLE_IN", "PHYSICAL_VERIFIER"),
            ("coffee_stirrer", "coffee_container", "REACHES_BOTTOM", "PHYSICAL_VERIFIER"),
            ("soup_eating_utensil", "soup_container", "INSERTABLE_IN", "PHYSICAL_VERIFIER"),
        ),
    )
    graph = FunctionalRequirementGraph(
        domain=graph.domain, task_instruction=graph.task_instruction,
        nodes=graph.nodes, provisional_relation_constraints=(constraint,),
    )
    result = ground_graph(graph, _scene("REACHES_BOTTOM"))
    assert result.complete
    assert result.resolved_graph["relations"][0]["predicate"] == "REACHES_BOTTOM"


def test_provider_accepts_complete_provisional_graph(monkeypatch):
    graph = _graph()
    monkeypatch.setattr(
        "mujoco_scenes.functional_tamp_pipeline.semantic_compiler.compile_candidate_graph",
        lambda *_args, **_kwargs: graph,
    )
    result = VLMSpecProvider().provide(
        "kitchen", "use implement", raw_document={"legacy": True},
        validation_mode="LEGACY_FIXTURE",
    )
    assert result is graph
