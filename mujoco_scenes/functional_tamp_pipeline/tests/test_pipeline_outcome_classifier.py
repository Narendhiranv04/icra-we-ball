from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
    PIPELINE_OUTCOMES, classify_pipeline_outcome,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph, FunctionalRole, GraphGroundingResult, OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import complete_planning_contract


def _classify(**overrides):
    args = dict(task_specification_valid=True, graph_compiled=True, search_exhausted=True,
                individual_candidates_sufficient=True, functional_assignment_complete=True,
                planning_invoked=True, plan_complete=True)
    args.update(overrides)
    return classify_pipeline_outcome(**args).category


def test_exact_frozen_taxonomy_has_no_generic_grounding_failure():
    assert PIPELINE_OUTCOMES == {
        "SUCCESS", "FM_RESPONSE_FAILURE",
        "TASK_SPECIFICATION_FAILURE", "GRAPH_COMPILATION_FAILURE",
        "OBJECT_DISCOVERY_FAILURE", "FUNCTIONAL_ASSIGNMENT_FAILURE", "PLANNING_FAILURE",
    }
    assert "GROUNDING_FAILURE" not in PIPELINE_OUTCOMES


def test_stage_failures_map_to_exact_categories():
    assert _classify(task_specification_valid=False) == "TASK_SPECIFICATION_FAILURE"
    assert _classify(graph_compiled=False) == "GRAPH_COMPILATION_FAILURE"
    assert _classify(individual_candidates_sufficient=False, functional_assignment_complete=False) == "OBJECT_DISCOVERY_FAILURE"
    assert _classify(functional_assignment_complete=False) == "FUNCTIONAL_ASSIGNMENT_FAILURE"
    assert _classify(plan_complete=False) == "PLANNING_FAILURE"
    assert _classify() == "SUCCESS"


def test_search_not_exhausted_does_not_claim_object_discovery_failure():
    assert _classify(search_exhausted=False, individual_candidates_sufficient=False,
                     functional_assignment_complete=False) == "FUNCTIONAL_ASSIGNMENT_FAILURE"


def test_projected_grounding_cannot_masquerade_as_complete_task_success():
    graph = FunctionalRequirementGraph(
        domain="kitchen", task_instruction="task",
        nodes={"tool": FunctionalRole("tool"), "target": FunctionalRole("target")},
        operation_groups=(OperationGroup(
            id="required_op", function="STIR_COFFEE", tool_role="tool", target_role="target",
            required_target_count=1, usage_policy="DEDICATED_PER_TARGET",
            required_relations=("INSERTABLE_IN",),
        ),),
    )
    projected = GraphGroundingResult(
        status="PARTIAL_VERIFIED_GROUNDING", complete=False,
        assignment={"tool": "tool_1"}, operation_bindings={},
    )
    assert not complete_planning_contract(
        graph, projected, {"is_partial": False}, {"status": "VALID", "goal_status": "GOAL_SATISFIED"}
    )


def test_complete_plan_requires_every_operation_binding():
    group = OperationGroup(
        id="required_op", function="STIR_COFFEE", tool_role="tool", target_role="target",
        required_target_count=1, usage_policy="DEDICATED_PER_TARGET", required_relations=("INSERTABLE_IN",),
    )
    graph = FunctionalRequirementGraph(domain="kitchen", task_instruction="task",
        nodes={"tool": FunctionalRole("tool"), "target": FunctionalRole("target")}, operation_groups=(group,))
    grounding = GraphGroundingResult(status="COMPLETE", complete=True,
        assignment={"tool": "tool_1", "target": "target_1"}, operation_bindings={})
    assert not complete_planning_contract(
        graph, grounding, {"is_partial": False}, {"status": "VALID", "goal_status": "GOAL_SATISFIED"}
    )
    grounding = GraphGroundingResult(status="COMPLETE", complete=True,
        assignment={"tool": "tool_1", "target": "target_1"},
        operation_bindings={"required_op": [{"tool_id": "tool_1", "target_id": "target_1"}]})
    assert complete_planning_contract(
        graph, grounding, {"is_partial": False}, {"status": "VALID", "goal_status": "GOAL_SATISFIED"}
    )
