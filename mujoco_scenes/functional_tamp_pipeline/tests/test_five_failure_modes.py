"""All five terminal failure classes remain reachable and correctly separated.

These are synthetic. They exist so that each class stays a real, distinguishable
outcome of the architecture rather than a label nothing can produce. They
deliberately do not encode any benchmark variant.
"""

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import (
    MalformedVLMSpecificationError, TaskSpecificationValidationError,
)
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation, FunctionalRequirementGraph, FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import classify_pipeline_outcome
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    abstract_slots_for_operation,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode, ObservedRelation, ObservedSceneGraph,
)


def _role(name, categories, count=1):
    return FunctionalRole(
        name=name, entity_kind="OBJECT", count=count, binding_policy="DISTINCT",
        semantic_categories=tuple(categories), canonical_role_candidates=(name,),
    )


def _node(instance_id, label):
    return ObservedNode(instance_id=instance_id, entity_kind="OBJECT",
                        canonical_category=label,
                        semantic_labels={"status": "SUPPORTED", "canonical_label": label})


def _v3(roles, relations=(), operations=()):
    return {
        "schema_version": 3, "status": "SUPPORTED", "task_summary": "t",
        "task_contract": {"functional_roles": list(roles),
                          "functional_relations": list(relations),
                          "operation_pairings": list(operations)},
        "observation_guidance": {"visible_candidates_per_role": {},
                                 "inspectable_regions": [], "inspection_order": []},
        "unsupported_reason": "",
    }


def _raw_role(rid, function, categories=()):
    return {"id": rid, "entity_kind": "OBJECT", "function": function,
            "required_count": 1, "binding_policy": "DISTINCT",
            "candidate_categories": list(categories), "required_properties": []}


# ---------------------------------------------------------------- 1 of 5
def test_task_specification_failure_on_internally_contradictory_contract():
    """An operation naming a participant the contract never declared."""
    doc = _v3(
        roles=[_raw_role("cup", "a container for the served drink", ["cup"])],
        operations=[{"id": "op1", "operation": "pour into cup",
                     "participant_roles": ["cup", "undeclared_source"],
                     "operation_count": 1}],
    )
    with pytest.raises((MalformedVLMSpecificationError, TaskSpecificationValidationError)):
        normalize_and_validate_v3_contract(doc, domain="kitchen")

    assert classify_pipeline_outcome(
        task_specification_valid=False, graph_compiled=False,
    ).category == "TASK_SPECIFICATION_FAILURE"


# ---------------------------------------------------------------- 2 of 5
def test_graph_compilation_failure_for_physical_operation_runtime_cannot_execute():
    """A coherent, physical operation the capability registry does not support."""
    assert abstract_slots_for_operation("kitchen", "caramelise the sugar with a blowtorch") == ()

    assert classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=False,
    ).category == "GRAPH_COMPILATION_FAILURE"


# ---------------------------------------------------------------- 3 of 5
def test_object_discovery_failure_when_plausible_individuals_are_too_few():
    graph_f = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="two servings",
        nodes={"cup": _role("cup", ("cup",), count=2)}, relations=(), operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("cup_1", "cup"))
    graph_o.add_node(_node("spoon_1", "spoon"))
    result = ground_graph(graph_f, graph_o, {"search_exhausted": True})

    assert result.evidence["individual_candidates_sufficient"] is False
    assert classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=True, search_exhausted=True,
        individual_candidates_sufficient=False, functional_assignment_complete=False,
    ).category == "OBJECT_DISCOVERY_FAILURE"


# ---------------------------------------------------------------- 4 of 5
def test_functional_assignment_failure_when_no_joint_choice_satisfies_constraints():
    graph_f = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="stir",
        nodes={"stirrer": _role("stirrer", ("spoon",)), "cup": _role("cup", ("cup",))},
        relations=(FunctionalRelation("stirrer", "INSERTABLE_IN", "cup"),),
        operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("spoon_1", "spoon"))
    graph_o.add_node(_node("cup_1", "cup"))
    graph_o.add_relation(ObservedRelation("spoon_1", "INSERTABLE_IN", "cup_1", "FALSE"))
    result = ground_graph(graph_f, graph_o, {"search_exhausted": True})

    assert result.evidence["individual_candidates_sufficient"] is True
    assert not result.complete
    assert classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=True, search_exhausted=True,
        individual_candidates_sufficient=True, functional_assignment_complete=False,
    ).category == "FUNCTIONAL_ASSIGNMENT_FAILURE"


# ---------------------------------------------------------------- 5 of 5
def test_planning_failure_when_complete_grounding_yields_no_validated_plan():
    graph_f = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="serve",
        nodes={"cup": _role("cup", ("cup",))}, relations=(), operation_groups=(),
    )
    graph_o = ObservedSceneGraph()
    graph_o.add_node(_node("cup_1", "cup"))
    result = ground_graph(graph_f, graph_o, {"search_exhausted": True})
    assert result.complete, "precondition: grounding is complete"

    assert classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=True, search_exhausted=True,
        individual_candidates_sufficient=True, functional_assignment_complete=True,
        planning_invoked=True, plan_complete=False,
    ).category == "PLANNING_FAILURE"


def test_success_is_distinct_from_every_failure_class():
    assert classify_pipeline_outcome(
        task_specification_valid=True, graph_compiled=True, search_exhausted=True,
        individual_candidates_sufficient=True, functional_assignment_complete=True,
        planning_invoked=True, plan_complete=True,
    ).category == "SUCCESS"


def test_all_five_failure_classes_are_reachable_and_distinct():
    categories = {
        classify_pipeline_outcome(task_specification_valid=False, graph_compiled=False).category,
        classify_pipeline_outcome(task_specification_valid=True, graph_compiled=False).category,
        classify_pipeline_outcome(task_specification_valid=True, graph_compiled=True,
                                  search_exhausted=True, individual_candidates_sufficient=False,
                                  functional_assignment_complete=False).category,
        classify_pipeline_outcome(task_specification_valid=True, graph_compiled=True,
                                  search_exhausted=True, individual_candidates_sufficient=True,
                                  functional_assignment_complete=False).category,
        classify_pipeline_outcome(task_specification_valid=True, graph_compiled=True,
                                  search_exhausted=True, individual_candidates_sufficient=True,
                                  functional_assignment_complete=True, planning_invoked=True,
                                  plan_complete=False).category,
    }
    assert categories == {
        "TASK_SPECIFICATION_FAILURE", "GRAPH_COMPILATION_FAILURE",
        "OBJECT_DISCOVERY_FAILURE", "FUNCTIONAL_ASSIGNMENT_FAILURE", "PLANNING_FAILURE",
    }


# ---------------------------------------------------------------------------
# A response that never arrived is not a statement about the task
# ---------------------------------------------------------------------------


def test_a_transport_failure_is_not_a_specification_failure():
    from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
        FM_RESPONSE_FAILURE_CATEGORIES,
        classify_pipeline_outcome,
    )

    assert "TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE" in FM_RESPONSE_FAILURE_CATEGORIES
    outcome = classify_pipeline_outcome(
        task_specification_valid=False,
        graph_compiled=False,
        fm_response_usable=False,
        reason="connection reset before any content arrived",
    )
    assert outcome.category == "FM_RESPONSE_FAILURE"


def test_a_contract_that_arrived_and_is_wrong_is_still_a_specification_failure():
    """The adversarial half: the response was readable, the task was not."""
    from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
        classify_pipeline_outcome,
    )

    outcome = classify_pipeline_outcome(
        task_specification_valid=False,
        graph_compiled=False,
        fm_response_usable=True,
        reason="contract declares a relation over a role it never declared",
    )
    assert outcome.category == "TASK_SPECIFICATION_FAILURE"
