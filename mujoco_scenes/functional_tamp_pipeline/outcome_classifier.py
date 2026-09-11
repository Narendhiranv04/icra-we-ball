"""Authoritative top-level outcome taxonomy for the Functional-TAMP pipeline."""

from dataclasses import dataclass, field
from typing import Any


PIPELINE_OUTCOMES = frozenset({
    "SUCCESS",
    # A response that never arrived, or arrived unparseable, is not a statement
    # about the task.  Counting it as a specification failure attributed a
    # dropped connection or a truncated generation to the model's understanding,
    # which is the wrong finding twice over: it overstates specification error
    # and hides a plumbing fault that a rerun would clear.
    "FM_RESPONSE_FAILURE",
    "TASK_SPECIFICATION_FAILURE",
    "GRAPH_COMPILATION_FAILURE",
    "OBJECT_DISCOVERY_FAILURE",
    "FUNCTIONAL_ASSIGNMENT_FAILURE",
    "PLANNING_FAILURE",
})

# Failure categories that mean the response itself did not arrive in a usable
# form, as opposed to arriving and being wrong about the task.
FM_RESPONSE_FAILURE_CATEGORIES = frozenset({
    "TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE",
})


@dataclass(frozen=True)
class PipelineOutcome:
    category: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


def complete_planning_contract(specification, grounding, search_statistics, validation) -> bool:
    """Return whether the original task—not a projected subgraph—was fully planned.

    The check over operation bindings is vacuously true for a graph carrying no
    operations, which is how a workshop trial once reported the fastening done
    after picking a screwdriver up and putting it back down, and how a kitchen
    trial reported two coffees served having only ever laid out a spoon.  A task
    with nothing to bring about is not a task, and a task whose required
    operations the runtime could not represent was not the task that got planned.
    """
    operation_bindings = getattr(grounding, "operation_bindings", {})
    operation_groups = tuple(specification.operation_groups)
    operations_complete = bool(operation_groups) and all(
        group.id in operation_bindings
        and len(operation_bindings[group.id]) >= group.required_target_count
        for group in operation_groups
    )
    executable_contract = bool(getattr(
        specification, "online_executable_contract_complete",
        getattr(specification, "required_contract_complete", True),
    ))
    return bool(
        getattr(grounding, "complete", False)
        and not search_statistics.get("is_partial", False)
        # Legacy/custom planners may omit validation metadata; an explicit
        # non-valid value still vetoes completeness.
        and validation.get("status", "VALID") == "VALID"
        and validation.get("goal_status", "GOAL_SATISFIED") == "GOAL_SATISFIED"
        and operations_complete
        and executable_contract
    )


def executable_graph_compiled(specification) -> bool:
    """Whether the compiled graph represents the task's required semantics.

    The outcome taxonomy reserves GRAPH_COMPILATION_FAILURE for coherent FM
    semantics the runtime cannot represent.  Reporting "compiled" for any graph
    that merely held a node sent such trials on to grounding and let them be
    scored as discovery or assignment problems, which is the wrong attribution.
    """
    return bool(getattr(
        specification, "online_executable_contract_complete",
        getattr(specification, "required_contract_complete", True),
    ))


def classify_pipeline_outcome(
    *,
    task_specification_valid: bool,
    graph_compiled: bool,
    search_exhausted: bool = True,
    individual_candidates_sufficient: bool = False,
    functional_assignment_complete: bool = False,
    planning_invoked: bool = False,
    plan_complete: bool = False,
    reason: str = "",
    evidence: dict[str, Any] | None = None,
    fm_response_usable: bool = True,
) -> PipelineOutcome:
    """Map stage evidence to exactly one frozen terminal category."""
    facts = dict(evidence or {})
    facts.update({
        "fm_response_usable": fm_response_usable,
        "task_specification_valid": task_specification_valid,
        "graph_compiled": graph_compiled,
        "search_exhausted": search_exhausted,
        "individual_candidates_sufficient": individual_candidates_sufficient,
        "functional_assignment_complete": functional_assignment_complete,
        "planning_invoked": planning_invoked,
        "plan_complete": plan_complete,
    })
    if not fm_response_usable:
        category, default = (
            "FM_RESPONSE_FAILURE",
            "The FM response did not arrive in a form that could be read at all",
        )
    elif not task_specification_valid:
        category, default = "TASK_SPECIFICATION_FAILURE", "FM task contract is missing, malformed, or contradictory"
    elif not graph_compiled:
        category, default = "GRAPH_COMPILATION_FAILURE", "Coherent FM semantics are unsupported by the runtime representation"
    elif not functional_assignment_complete:
        if search_exhausted and not individual_candidates_sufficient:
            category, default = "OBJECT_DISCOVERY_FAILURE", "Search exhausted without enough role-compatible observed individuals"
        else:
            category, default = "FUNCTIONAL_ASSIGNMENT_FAILURE", "Observed individuals cannot satisfy the joint functional constraints"
    elif not planning_invoked or not plan_complete:
        category, default = "PLANNING_FAILURE", "Complete grounding did not produce a complete validated plan"
    else:
        category, default = "SUCCESS", "Complete grounded task plan independently validated"
    assert category in PIPELINE_OUTCOMES
    return PipelineOutcome(category, reason or default, facts)


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
