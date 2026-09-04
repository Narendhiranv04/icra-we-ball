"""Focused contracts for the privileged three-level GT evidence ablation."""
from __future__ import annotations

from unittest.mock import patch
import pytest

from mujoco_scenes import run_gt_evidence_ablation as runner
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph, FunctionalRole, GraphGroundingResult, OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import ObservedNode, ObservedRelation, ObservedSceneGraph


@pytest.mark.parametrize("condition,expected", list(runner.COMPONENT_MASKS.items()))
def test_seven_masks_are_exact(condition, expected):
    assert runner.COMPONENT_MASKS[condition] == expected


def _fixture():
    spec = FunctionalRequirementGraph(
        domain="test", task_instruction="test",
        nodes={
            "tool": FunctionalRole("tool", semantic_categories=("spoon",), unary_predicates=("ELONGATED_OBJECT",)),
            "target": FunctionalRole("target", semantic_categories=("mug",), unary_predicates=("OPEN_CAVITY",)),
        },
        operation_groups=(OperationGroup("use", "STIR", "tool", "target", 1,
                                         "SEQUENTIAL_REUSE_ALLOWED", ("INSERTABLE_IN",)),),
    )
    graph = ObservedSceneGraph()
    graph.add_node(ObservedNode("wrong", canonical_category="bowl", unary_predicates={"ELONGATED_OBJECT": "TRUE"}))
    graph.add_node(ObservedNode("mug", canonical_category="mug", unary_predicates={"OPEN_CAVITY": "TRUE"}))
    graph.add_relation(ObservedRelation("wrong", "INSERTABLE_IN", "mug", "FALSE"))
    result = GraphGroundingResult(status="COMPLETE", complete=True,
        assignment={"tool": "wrong", "target": "mug"},
        operation_bindings={"use": [{"tool_id": "wrong", "target_id": "mug", "context": {}}]})
    return spec, graph, result


def test_complete_grounding_can_have_gt_invalid_role_and_pair():
    spec, graph, result = _fixture()
    roles = runner._validate_roles(spec, result, graph)
    pairs = runner._validate_bindings(spec, result, graph)
    assert result.complete
    assert not roles["exact_role_grounding_success"]
    assert not pairs["exact_operation_binding_success"]


def test_complete_invalid_grounding_is_not_exact_symbolic_success(monkeypatch):
    spec, graph, result = _fixture()
    monkeypatch.setattr(runner, "_ground", lambda *args: result)
    monkeypatch.setattr(runner, "intended_outcome", lambda *args: "FEASIBLE")
    monkeypatch.setattr(runner, "_plan", lambda *args: {
        "plan_generated": True, "plan_generation_failure_reason": None,
        "generated_action_sequence": [{"operator": "STIR", "arguments": ["wrong", "mug"]}],
        "plan_replay_valid": True, "planner_status": "SUCCESS", "planner_runtime_ms": 0.0,
    })
    monkeypatch.setattr(runner, "_task_actions_valid", lambda *args: True)
    row = runner.evaluate_one("kitchen", "fixture", "semantic_only", specification=spec, graph=graph)
    assert row["grounding_complete"] and row["plan_replay_valid"]
    assert not row["gt_task_plan_valid"]
    assert not row["exact_symbolic_task_success"]
    assert row["first_failure_level"] == "ROLE_ASSIGNMENT_INVALID"


def test_placeholder_binding_is_preserved_then_scored_against_gt():
    spec, graph, result = _fixture()
    scored = runner._validate_bindings(spec, result, graph)
    assert scored["per_binding_gt_validation"][0]["binding"] == result.operation_bindings["use"][0]
    assert scored["per_binding_gt_validation"][0]["relation_checks"][0]["status"] == "FALSE"


def test_planner_receives_ablation_selected_assignment_and_bindings():
    spec = runner.GTSpecProvider().provide("living_room", "")
    graph = ObservedSceneGraph()
    assignment = {"SHARED_REMOTE_REGION": "selected_region", "REMOTE": "selected_remote"}
    result = GraphGroundingResult(status="COMPLETE", complete=True, assignment=assignment,
                                  operation_bindings={"personal_support_group": []})
    with patch.object(runner, "plan_with_common_astar") as planning:
        planning.side_effect = RuntimeError("captured")
        runner._plan("living_room", spec, result, graph)
    assert planning.call_args.args[1] is assignment
    assert planning.call_args.args[2]["operation_bindings"] is result.operation_bindings


def test_full_evidence_known_fixture_has_gt_valid_roles_and_pairs():
    spec = runner.GTSpecProvider().provide("kitchen", "")
    graph = runner.build_oracle_graph("kitchen", "F0_ALL_VISIBLE", spec)
    row = runner.evaluate_one("kitchen", "F0_ALL_VISIBLE", "full", specification=spec, graph=graph)
    assert row["exact_role_grounding_success"]
    assert row["exact_operation_binding_success"]
    assert row["gt_task_plan_valid"]


def test_workshop_explicit_relations_are_level_two_bindings():
    spec = runner.GTSpecProvider().provide("workshop", "")
    graph = runner.build_oracle_graph("workshop", "F0_MANUAL_FIRST_ONE_REGION", spec)
    row = runner.evaluate_one("workshop", "F0_MANUAL_FIRST_ONE_REGION", "full", specification=spec, graph=graph)
    assert row["operation_bindings_total"] == 3
    assert row["operation_bindings_gt_valid"] == 3
    assert {binding["relation"] for binding in row["selected_operation_bindings"]["explicit_relations"]} == {
        "COMPATIBLE_WITH", "REACHES_TARGET", "COMPATIBLE_WITH_TARGET",
    }


def test_kitchen_source_identity_is_fixed_and_cannot_fill_other_roles():
    spec = runner.GTSpecProvider().provide("kitchen", "")
    graph = runner.build_oracle_graph("kitchen", "F0_ALL_VISIBLE", spec)
    expected = runner._fixed_kitchen_sources(spec, graph)
    for condition, components in runner.COMPONENT_MASKS.items():
        result = runner._ground("kitchen", spec, graph, components)
        assert {role: result.assignment[role] for role in runner.KITCHEN_FIXED_SOURCE_ROLES} == expected
        ordinary = {obj for role, value in result.assignment.items()
                    if role not in runner.KITCHEN_FIXED_SOURCE_ROLES for obj in runner._as_list(value)}
        assert ordinary.isdisjoint(expected.values()), condition
        assert result.evidence["benchmark_fixed_source_identity_is_not_masked_evidence"] is True


def _summary_row(outcome="FEASIBLE", **updates):
    row = {"domain": "kitchen", "condition": "full", "enabled_evidence_components": ["semantic", "unary", "binary"],
           "intended_outcome": outcome, "grounding_complete": True, "outcome_correct": True,
           "role_slots_total": 2, "role_slots_gt_valid": 2, "exact_role_grounding_success": True,
           "operation_bindings_total": 1, "operation_bindings_gt_valid": 1,
           "exact_operation_binding_success": True, "plan_generated": True, "plan_replay_valid": True,
           "gt_task_plan_valid": True, "exact_symbolic_task_success": True,
           "grounding_runtime_ms": 1.0, "planner_runtime_ms": 2.0}
    row.update(updates); return row


def test_summary_uses_all_feasible_variants_as_exact_metric_denominator():
    rows = [_summary_row(), _summary_row(exact_role_grounding_success=False,
            exact_operation_binding_success=False, exact_symbolic_task_success=False,
            gt_task_plan_valid=False, role_slots_gt_valid=1, operation_bindings_gt_valid=0)]
    summary = runner.summarize(rows)[0]
    assert summary["exact_role_grounding_success_pct"] == 50.0
    assert summary["exact_operation_binding_success_pct"] == 50.0
    assert summary["exact_symbolic_task_success_pct"] == 50.0
    assert summary["role_slot_validity_pct"] == 75.0


def test_infeasible_rejection_and_false_completion_accounting():
    rows = [_summary_row("INFEASIBLE", grounding_complete=False),
            _summary_row("INFEASIBLE", grounding_complete=True, outcome_correct=False)]
    summary = runner.summarize(rows)[0]
    assert summary["infeasible_rejection_pct"] == 50.0
    assert summary["false_completion_pct"] == 50.0
