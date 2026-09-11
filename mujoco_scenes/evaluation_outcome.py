"""How a reported outcome is scored against a variant's ground-truth feasibility.

This lives outside `functional_tamp_pipeline` on purpose.  Scoring is the only
layer allowed to know whether a variant is feasible; the pipeline must never see
that, and the no-GT-leakage audit enforces it by treating any `gt_` reference
inside the pipeline package as a finding.  The first version of this module was
placed there and the audit caught it immediately -- correctly.
"""
from __future__ import annotations

# The pipeline statuses that constitute *reaching* an infeasibility conclusion.
#
# This set exists because two scorers disagreed.  The live evaluator required one
# of these statuses; the frozen-replay scorer asked only whether the task went
# unsatisfied, which on an infeasible variant is true by construction -- the task
# cannot be satisfied, so nothing the system does can make it false.  That scored
# a tautology as a result and reported 36/36 for behaviour that had not been
# measured at all.
#
# Refusing to claim completion and concluding that the task cannot be completed
# are different achievements.  The first is a safety property and is reported
# separately as the false-completion count.  This set is the second one.
INFEASIBILITY_CONCLUDED_STATUSES = frozenset({
    "INFEASIBLE",
    "EXHAUSTED_NO_VALID_GROUNDING",
    "NO_VALID_COMPLETE_ASSIGNMENT",
    "PLANNING_PROVEN_INFEASIBLE",
})


def outcome_is_correct(*, gt_feasible: bool, task_satisfied: bool,
                       false_completion: bool, pipeline_status: str | None) -> bool:
    """Whether the system's reported outcome matches the variant's truth.

    On a feasible variant: it had to finish the task.  On an infeasible one: it
    had to avoid claiming completion *and* reach an infeasibility conclusion.
    """
    if gt_feasible:
        return bool(task_satisfied)
    return (not task_satisfied and not false_completion
            and str(pipeline_status) in INFEASIBILITY_CONCLUDED_STATUSES)
