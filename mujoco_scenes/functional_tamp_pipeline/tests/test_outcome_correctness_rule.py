"""The infeasible-variant outcome rule, and the tautology it replaced.

Two scorers disagreed on what `outcome_correct` means for an infeasible variant.
The live evaluator required the pipeline to reach an infeasibility conclusion;
the frozen-replay scorer asked only whether the task went unsatisfied.  On an
infeasible variant the task cannot be satisfied, so that condition is true by
construction: it credited every trial and measured nothing.  These tests pin the
shared rule and the refutation, so the tautology cannot come back.
"""
from __future__ import annotations

from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
    INFEASIBILITY_CONCLUDED_STATUSES,
    outcome_is_correct,
)


def test_a_feasible_variant_is_correct_only_when_the_task_is_finished():
    assert outcome_is_correct(gt_feasible=True, task_satisfied=True,
                              false_completion=False, pipeline_status="ACTION_SEQUENCE_READY")
    assert not outcome_is_correct(gt_feasible=True, task_satisfied=False,
                                  false_completion=False, pipeline_status="INFEASIBLE")


def test_an_infeasible_variant_needs_a_conclusion_not_merely_a_non_completion():
    """The refutation: not finishing an impossible task is not an achievement."""
    for status in sorted(INFEASIBILITY_CONCLUDED_STATUSES):
        assert outcome_is_correct(gt_feasible=False, task_satisfied=False,
                                  false_completion=False, pipeline_status=status), status
    # Same non-completion, no conclusion reached -- must not be credited.
    for status in ("PARTIAL_ACTION_SEQUENCE_READY", "NO_MEANINGFUL_CANDIDATE_PLAN",
                   "VLM_SPEC_FAILED", None):
        assert not outcome_is_correct(gt_feasible=False, task_satisfied=False,
                                      false_completion=False, pipeline_status=status), status


def test_a_false_completion_is_never_a_correct_outcome():
    assert not outcome_is_correct(gt_feasible=False, task_satisfied=False,
                                  false_completion=True, pipeline_status="INFEASIBLE")


def test_both_scorers_use_this_one_rule():
    """Offline and live must not score the same behaviour differently."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    for script in ("scripts/replay_frozen_trial.py",
                   "scripts/evaluate_vlm_functional_tamp.py",
                   "scripts/evaluate_heldout_matrix.py"):
        source = (repo / script).read_text()
        assert "outcome_is_correct" in source, f"{script} does not use the shared rule"
        assert "PLANNING_PROVEN_INFEASIBLE" not in source, (
            f"{script} still carries its own copy of the status list")
        assert '"NO_MEANINGFUL_CANDIDATE_PLAN",\n' not in source, (
            f"{script} still credits a partial plan as an infeasibility conclusion")


def test_a_partial_plan_is_not_an_infeasibility_conclusion():
    """The held-out scorer credited this and omitted INFEASIBLE -- both wrong."""
    assert not outcome_is_correct(gt_feasible=False, task_satisfied=False,
                                  false_completion=False,
                                  pipeline_status="NO_MEANINGFUL_CANDIDATE_PLAN")
    assert outcome_is_correct(gt_feasible=False, task_satisfied=False,
                              false_completion=False, pipeline_status="INFEASIBLE")
