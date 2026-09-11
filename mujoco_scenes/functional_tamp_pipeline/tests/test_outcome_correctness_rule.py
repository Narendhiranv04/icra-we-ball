"""The one scoring rule, and every wrong definition it replaced.

Four definitions of these metrics existed across the repository and disagreed.
Each refutation below is pinned as a test so the mistake cannot be reintroduced
silently.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mujoco_scenes.evaluation_outcome import (
    COMPLETION_CLAIMED_STATUSES,
    INFEASIBILITY_CONCLUDED_STATUSES,
    completion_claimed,
    is_false_completion,
    outcome_is_correct,
    summarize,
)

REPO = Path(__file__).resolve().parents[3]


def _row(*, feasible, status, satisfied):
    return {"feasible": feasible, "pipeline_status": status,
            "gt_full_task_satisfied": satisfied}


# --- false completion -------------------------------------------------------

def test_the_adversarial_false_completion_is_caught():
    """Infeasible task, system announces a finished plan, GT says unfinished.

    The live evaluator defined this as `not feasible and full_task_satisfied`,
    which needs ground truth to certify an impossible task as done.  It could
    therefore never fire, and missed exactly this case.
    """
    assert is_false_completion(pipeline_status="ACTION_SEQUENCE_READY",
                               gt_full_task_satisfied=False)


def test_claiming_completion_of_an_unfinished_feasible_task_is_also_false():
    """The defect does not depend on the task being impossible."""
    assert is_false_completion(pipeline_status="ACTION_SEQUENCE_READY",
                               gt_full_task_satisfied=False)
    assert not is_false_completion(pipeline_status="ACTION_SEQUENCE_READY",
                                   gt_full_task_satisfied=True)


def test_not_claiming_completion_is_never_a_false_completion():
    for status in ("PARTIAL_ACTION_SEQUENCE_READY", "NO_MEANINGFUL_CANDIDATE_PLAN",
                   "INFEASIBLE", "VLM_SPEC_FAILED", None):
        assert not is_false_completion(pipeline_status=status,
                                       gt_full_task_satisfied=False), status


def test_only_one_status_counts_as_claiming_completion():
    assert COMPLETION_CLAIMED_STATUSES == {"ACTION_SEQUENCE_READY"}
    assert completion_claimed("ACTION_SEQUENCE_READY")
    assert not completion_claimed("PARTIAL_ACTION_SEQUENCE_READY")


# --- feasible correctness ---------------------------------------------------

def test_a_feasible_variant_needs_both_a_claim_and_confirmation():
    assert outcome_is_correct(gt_feasible=True, gt_full_task_satisfied=True,
                              pipeline_status="ACTION_SEQUENCE_READY")
    # Claimed but not actually done -- a false completion, never correct.
    assert not outcome_is_correct(gt_feasible=True, gt_full_task_satisfied=False,
                                  pipeline_status="ACTION_SEQUENCE_READY")


def test_stopping_short_of_the_full_task_is_incorrect_even_if_gt_is_satisfied():
    """The refutation of definition 4, and the source of the 41-vs-42 mismatch.

    Feasible correctness was scored from ground-truth satisfaction alone.  A run
    that stopped at a partial action sequence, whose leftover artifacts happened
    to satisfy every goal, scored as a correct outcome despite never delivering
    a plan -- so feasible-correct and feasible-success disagreed by one trial and
    the overall count stopped being the sum of its parts.
    """
    assert not outcome_is_correct(gt_feasible=True, gt_full_task_satisfied=True,
                                  pipeline_status="PARTIAL_ACTION_SEQUENCE_READY")


# --- infeasible correctness -------------------------------------------------

def test_an_infeasible_variant_needs_a_conclusion_not_merely_a_non_completion():
    """The refutation of definition 1: not finishing the impossible is not an
    achievement, it is arithmetic."""
    for status in sorted(INFEASIBILITY_CONCLUDED_STATUSES):
        assert outcome_is_correct(gt_feasible=False, gt_full_task_satisfied=False,
                                  pipeline_status=status), status
    for status in ("PARTIAL_ACTION_SEQUENCE_READY", "NO_MEANINGFUL_CANDIDATE_PLAN",
                   "VLM_SPEC_FAILED", None):
        assert not outcome_is_correct(gt_feasible=False, gt_full_task_satisfied=False,
                                      pipeline_status=status), status


def test_a_partial_plan_is_not_an_infeasibility_conclusion():
    """The refutation of definition 2, from the held-out matrix evaluator."""
    assert "NO_MEANINGFUL_CANDIDATE_PLAN" not in INFEASIBILITY_CONCLUDED_STATUSES
    assert "INFEASIBLE" in INFEASIBILITY_CONCLUDED_STATUSES


def test_a_false_completion_is_never_a_correct_outcome():
    assert not outcome_is_correct(gt_feasible=False, gt_full_task_satisfied=False,
                                  pipeline_status="ACTION_SEQUENCE_READY")


# --- arithmetic -------------------------------------------------------------

def test_the_decomposition_is_asserted_not_assumed():
    rows = [
        _row(feasible=True, status="ACTION_SEQUENCE_READY", satisfied=True),
        _row(feasible=True, status="PARTIAL_ACTION_SEQUENCE_READY", satisfied=True),
        _row(feasible=True, status="NO_MEANINGFUL_CANDIDATE_PLAN", satisfied=False),
        _row(feasible=False, status="INFEASIBLE", satisfied=False),
        _row(feasible=False, status="PARTIAL_ACTION_SEQUENCE_READY", satisfied=False),
    ]
    got = summarize(rows)
    assert got["feasible_success"] == 1
    assert got["feasible_outcome_correct"] == 1
    assert got["infeasible_outcome_correct"] == 1
    assert got["overall_outcome_correct"] == 2
    assert got["overall_outcome_correct"] == (
        got["feasible_outcome_correct"] + got["infeasible_outcome_correct"])
    assert got["false_completions"] == 0


def test_feasible_success_and_feasible_outcome_correct_are_one_measurement():
    """They disagreed by one trial before; summarize refuses to let them drift."""
    rows = [_row(feasible=True, status="ACTION_SEQUENCE_READY", satisfied=True),
            _row(feasible=True, status="PARTIAL_ACTION_SEQUENCE_READY", satisfied=True)]
    got = summarize(rows)
    assert got["feasible_success"] == got["feasible_outcome_correct"] == 1


# --- structural -------------------------------------------------------------

def test_every_scorer_uses_this_one_rule():
    for script in ("scripts/replay_frozen_trial.py",
                   "scripts/evaluate_vlm_functional_tamp.py",
                   "scripts/evaluate_heldout_matrix.py"):
        source = (REPO / script).read_text()
        assert "outcome_is_correct" in source, f"{script} does not use the shared rule"
        assert "is_false_completion" in source, f"{script} defines its own false completion"
        assert "evaluation_outcome" in source, (
            f"{script} must take the rule from the scoring layer, not the pipeline")
        assert "PLANNING_PROVEN_INFEASIBLE" not in source, (
            f"{script} still carries its own copy of the status list")
        assert "not is_feasible and full_task_sat" not in source, (
            f"{script} still carries the false-completion definition that never fires")
