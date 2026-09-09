"""Open-world semantic compatibility and failure-class separation.

Two architectural guarantees are covered here.

1. Grounding is open-world.  An observed label that the runtime ontology cannot
   resolve is UNKNOWN (undecided), never FALSE.  Only a *justified* mismatch --
   both sides resolvable inside the known vocabulary and disjoint there -- is a
   rejection.  Synonyms declared by the runtime ontology resolve to TRUE.

2. OBJECT_DISCOVERY_FAILURE and FUNCTIONAL_ASSIGNMENT_FAILURE are distinct.
   Discovery failure means some role lacks enough individually plausible
   observed candidates.  Assignment failure means plausible individuals exist
   for every role but no joint assignment satisfies the constraints.
"""

from mujoco_scenes.functional_tamp_pipeline.grounding import (
    check_semantic_role_compatibility,
    ground_graph,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation, FunctionalRequirementGraph, FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import classify_pipeline_outcome
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode, ObservedRelation, ObservedSceneGraph,
)


def _belief(label):
    return {"status": "SUPPORTED", "canonical_label": label}


def _role(name, categories, count=1):
    return FunctionalRole(
        name=name, entity_kind="OBJECT", count=count, binding_policy="DISTINCT",
        semantic_categories=tuple(categories), canonical_role_candidates=(name,),
    )


def _node(instance_id, label):
    return ObservedNode(instance_id=instance_id, entity_kind="OBJECT",
                        canonical_category=label, semantic_labels=_belief(label))


# --------------------------------------------------------------------------
# 1. Open-world semantic compatibility
# --------------------------------------------------------------------------

def test_open_vocabulary_mismatch_is_unknown_not_false():
    """A role category the runtime cannot resolve must not reject a real detection."""
    status, _ = check_semantic_role_compatibility(
        _belief("screwdriver"), ["fastening instrument"]
    )
    assert status == "UNKNOWN"


def test_unresolvable_observed_label_is_unknown_not_false():
    status, _ = check_semantic_role_compatibility(_belief("gizmo"), ["cup", "mug"])
    assert status == "UNKNOWN"


def test_known_incompatibility_is_still_false():
    """Both sides inside the known vocabulary and disjoint: a justified rejection."""
    status, _ = check_semantic_role_compatibility(_belief("spoon"), ["cup"])
    assert status == "FALSE"


def test_ontology_synonym_resolves_to_true():
    """Vocabulary variation alone must not cost a match."""
    assert check_semantic_role_compatibility(_belief("coffee mug"), ["cup"])[0] == "TRUE"
    assert check_semantic_role_compatibility(_belief("soup bowl"), ["bowl"])[0] == "TRUE"
    assert check_semantic_role_compatibility(_belief("Phillips screwdriver"),
                                             ["screwdriver"])[0] == "TRUE"


def test_unknown_is_not_promoted_to_true_without_evidence():
    """UNKNOWN candidates stay UNKNOWN; they never silently become matches."""
    graph_f = FunctionalRequirementGraph(
        domain="workshop", task_instruction="fasten",
        nodes={"driver": _role("driver", ("fastening instrument",))},
        relations=(), operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("tool_1", "screwdriver"))
    result = ground_graph(graph_f, graph_o)
    plaus = result.evidence["role_plausibility"]["driver"]
    assert plaus["true"] == 0
    assert plaus["unknown"] == 1


# --------------------------------------------------------------------------
# 2. Failure-class separation
# --------------------------------------------------------------------------

def test_object_discovery_failure_when_too_few_plausible_individuals():
    """Two distinct cups required, one cup observed plus a known-incompatible spoon."""
    graph_f = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="serve two coffees",
        nodes={"cup": _role("cup", ("cup",), count=2)},
        relations=(), operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("cup_1", "cup"))
    graph_o.add_node(_node("spoon_1", "spoon"))
    result = ground_graph(graph_f, graph_o, {"search_exhausted": True})

    assert not result.complete
    assert result.failure_kind == "OBJECT_DISCOVERY_FAILURE"
    assert result.evidence["individual_candidates_sufficient"] is False
    outcome = classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=True, search_exhausted=True,
        individual_candidates_sufficient=result.evidence["individual_candidates_sufficient"],
        functional_assignment_complete=False,
    )
    assert outcome.category == "OBJECT_DISCOVERY_FAILURE"


def test_functional_assignment_failure_when_individuals_exist_but_joint_fails():
    """Every role has a plausible individual, but the required relation cannot hold."""
    graph_f = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="stir the coffee",
        nodes={"stirrer": _role("stirrer", ("spoon",)), "cup": _role("cup", ("cup",))},
        relations=(FunctionalRelation("stirrer", "INSERTABLE_IN", "cup"),),
        operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("spoon_1", "spoon"))
    graph_o.add_node(_node("cup_1", "cup"))
    graph_o.add_relation(ObservedRelation("spoon_1", "INSERTABLE_IN", "cup_1", "FALSE"))
    result = ground_graph(graph_f, graph_o, {"search_exhausted": True})

    assert not result.complete
    assert result.evidence["individual_candidates_sufficient"] is True
    assert result.failure_kind == "FUNCTIONAL_ASSIGNMENT_FAILURE"
    outcome = classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=True, search_exhausted=True,
        individual_candidates_sufficient=result.evidence["individual_candidates_sufficient"],
        functional_assignment_complete=False,
    )
    assert outcome.category == "FUNCTIONAL_ASSIGNMENT_FAILURE"


def test_role_plausibility_evidence_is_always_emitted():
    graph_f = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="serve coffee",
        nodes={"cup": _role("cup", ("cup",))}, relations=(), operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("cup_1", "cup"))
    result = ground_graph(graph_f, graph_o, {"search_exhausted": True})
    assert result.complete
    assert result.evidence["role_plausibility"]["cup"]["true"] == 1
    assert result.evidence["individual_candidates_sufficient"] is True
