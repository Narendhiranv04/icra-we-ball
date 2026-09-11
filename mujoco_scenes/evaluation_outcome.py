"""The single authoritative scoring rule for benchmark outcomes.

This lives outside `functional_tamp_pipeline` on purpose.  Scoring is the only
layer allowed to know whether a variant is feasible; the online pipeline must
never see that, and the no-GT-leakage audit enforces it by treating any `gt_`
reference inside the pipeline package as a finding.

Four different definitions of these metrics existed across the repository and
disagreed with one another.  Each is recorded here as a refutation, because the
same mistakes are easy to reintroduce:

1. The frozen-replay scorer asked only `not gt_full_task_satisfied` for an
   infeasible variant.  That is true by construction -- an impossible task
   cannot be satisfied -- so it credited every infeasible trial automatically
   and measured nothing.
2. The held-out matrix evaluator credited `NO_MEANINGFUL_CANDIDATE_PLAN`, a
   partial plan, as an infeasibility conclusion, while omitting `INFEASIBLE`
   entirely, so a real conclusion scored wrong and a partial plan scored right.
3. The live evaluator defined a false completion as `not feasible and
   full_task_satisfied` -- a condition that requires ground truth to certify an
   impossible task as done, so it effectively never fires and misses the actual
   adversarial case: the system announcing a finished plan for a task that is
   not finished.
4. Feasible-variant correctness was scored from ground-truth satisfaction alone,
   ignoring whether the system ever claimed to be finished.  A run that stopped
   at a partial action sequence therefore scored as a correct outcome, which
   also broke the arithmetic: feasible-correct and feasible-success disagreed by
   one trial, so the overall count was not the sum of its parts.

The definitions below are the ones this project reports.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# A run announces a finished task with exactly this status.  Anything else --
# a partial sequence, an exhausted search, a rejected contract -- is the system
# declining to claim completion, whatever ground truth later says about the
# artifacts it left behind.
COMPLETION_CLAIMED_STATUSES = frozenset({"ACTION_SEQUENCE_READY"})

# The statuses that constitute *reaching* an infeasibility conclusion.
# Declining to claim completion and concluding that a task cannot be done are
# different achievements: the first is a safety property, reported separately as
# the false-completion count, and the second is what this set measures.
#
# Five further names once appeared in the whitelists this replaced --
# NO_VALID_GROUNDING, TASK_REJECTED_UNSUPPORTED, NO_SEARCH_REGIONS_DECLARED and
# others -- that no code path emits.  Those that are genuine conclusions are
# retained here so a future path emitting them is scored correctly; the ones
# that describe a partial result (NO_MEANINGFUL_CANDIDATE_PLAN) are not.
INFEASIBILITY_CONCLUDED_STATUSES = frozenset({
    "INFEASIBLE",
    "EXHAUSTED_NO_VALID_GROUNDING",
    "NO_VALID_COMPLETE_ASSIGNMENT",
    "PLANNING_PROVEN_INFEASIBLE",
})


def completion_claimed(pipeline_status: str | None) -> bool:
    """Whether the run announced a finished task."""
    return str(pipeline_status) in COMPLETION_CLAIMED_STATUSES


def is_false_completion(*, pipeline_status: str | None,
                        gt_full_task_satisfied: bool) -> bool:
    """The system said it was finished and independent evaluation disagrees.

    This is the metric that has to be zero.  It is deliberately not conditioned
    on feasibility: announcing a finished plan for a task that is not finished
    is the same defect whether or not the task was possible, and on an
    infeasible variant it is the adversarial case by construction.
    """
    return completion_claimed(pipeline_status) and not bool(gt_full_task_satisfied)


def outcome_is_correct(*, gt_feasible: bool, gt_full_task_satisfied: bool,
                       pipeline_status: str | None,
                       runtime_contract_complete: bool | None = None) -> bool:
    """Whether the reported outcome matches the variant's truth.

    Feasible: the run had to announce completion *and* independent evaluation
    had to confirm the task is done.  Both halves are required -- claiming a
    finish that did not happen is a false completion, and finishing without
    saying so is not a delivered result.

    Infeasible: the run had to avoid claiming completion, reach an
    infeasibility conclusion, *and* have had a complete contract to conclude it
    from.  A run whose semantics never compiled has not proven the scene cannot
    satisfy the task; it has only failed to state the task.  Crediting that
    would score a parsing failure as a correct scientific conclusion.

    `runtime_contract_complete=None` means the caller does not track it and the
    condition is skipped.  That default exists only for unit tests; both
    production scorers pass it, because applying it in one and not the other is
    exactly how the live and offline numbers came to disagree.
    """
    claimed = completion_claimed(pipeline_status)
    if gt_feasible:
        return claimed and bool(gt_full_task_satisfied)
    if runtime_contract_complete is False:
        return False
    return (not claimed
            and not is_false_completion(pipeline_status=pipeline_status,
                                        gt_full_task_satisfied=gt_full_task_satisfied)
            and str(pipeline_status) in INFEASIBILITY_CONCLUDED_STATUSES)


def summarize(rows: Iterable[Mapping[str, Any]], *, feasible_key: str = "feasible",
              satisfied_key: str = "gt_full_task_satisfied",
              status_key: str = "pipeline_status") -> dict[str, Any]:
    """Mechanically decompose the headline metrics, and check the arithmetic.

    Headline numbers are computed here rather than written into a report by
    hand, and the decomposition is asserted rather than assumed: an overall
    count that is not the sum of its feasible and infeasible parts means the two
    halves are using different definitions, which is exactly the defect this
    module exists to prevent.
    """
    rows = list(rows)
    feasible = [r for r in rows if r.get(feasible_key)]
    infeasible = [r for r in rows if not r.get(feasible_key)]

    def correct(group):
        return sum(1 for r in group if outcome_is_correct(
            gt_feasible=bool(r.get(feasible_key)),
            gt_full_task_satisfied=bool(r.get(satisfied_key)),
            pipeline_status=r.get(status_key),
            runtime_contract_complete=r.get("executable_contract_complete")))

    feasible_correct = correct(feasible)
    infeasible_correct = correct(infeasible)
    overall_correct = correct(rows)
    if overall_correct != feasible_correct + infeasible_correct:
        raise AssertionError(
            f"outcome decomposition is inconsistent: overall {overall_correct} != "
            f"feasible {feasible_correct} + infeasible {infeasible_correct}")

    successes = sum(1 for r in feasible if completion_claimed(r.get(status_key))
                    and bool(r.get(satisfied_key)))
    if successes != feasible_correct:
        raise AssertionError(
            f"feasible success {successes} != feasible outcome correct "
            f"{feasible_correct}; the two must be the same measurement")

    false_completions = sum(1 for r in rows if is_false_completion(
        pipeline_status=r.get(status_key),
        gt_full_task_satisfied=bool(r.get(satisfied_key))))
    return {
        "trials": len(rows),
        "feasible_trials": len(feasible),
        "infeasible_trials": len(infeasible),
        "feasible_success": successes,
        "feasible_outcome_correct": feasible_correct,
        "infeasible_outcome_correct": infeasible_correct,
        "overall_outcome_correct": overall_correct,
        "false_completions": false_completions,
    }
