"""Exactness checks for semantic-role-scoped witness pair pruning."""
from __future__ import annotations

import copy
import random

from mujoco_scenes.task_witness import evaluate_usage_policy_task_witness
from mujoco_scenes.tests.test_ablation3_target_assignment import TASK, _graph


def _result(graph):
    return evaluate_usage_policy_task_witness(graph, TASK, target_assignment_mode="joint-target-specific")


def _decision_projection(result):
    return (
        result["status"],
        tuple(
            (
                group["function_group_id"], group["status"],
                tuple((a["utensil_object_id"], a["target_object_id"]) for a in group.get("selected_assignments", [])),
            )
            for group in result["function_group_evaluations"]
        ),
    )


def _compare(graph):
    exhaustive = copy.deepcopy(graph); exhaustive["pairing"] = {"strategy": "exhaustive_all_pairs"}
    pruned = copy.deepcopy(graph); pruned["pairing"] = {"strategy": "semantic_role_scoped"}
    assert _decision_projection(_result(pruned)) == _decision_projection(_result(exhaustive))


def test_pruned_witness_matches_exhaustive_small_case():
    _compare(_graph(include_long=True))


def test_capacity_pruning_preserves_valid_distinct_assignment():
    _compare(_graph())


def test_reusable_role_pruning_preserves_valid_reuse_solution():
    _compare(_graph(include_long=True))


def test_pruning_never_changes_grounding_result_on_small_random_cases():
    rng = random.Random(20260908)
    for _ in range(20):
        graph = _graph(include_partial=rng.choice((False, True)), include_long=rng.choice((False, True)))
        for edge in graph["edges"]:
            if edge.get("relation") in {"INSERTABLE_IN", "REACHES_BOTTOM"} and rng.random() < 0.12:
                edge["status"] = rng.choice(("TRUE", "FALSE", "UNKNOWN"))
        _compare(graph)
