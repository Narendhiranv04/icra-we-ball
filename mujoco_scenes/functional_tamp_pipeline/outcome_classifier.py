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


def exhaustion_proves_no_valid_grounding(specification, satisfaction) -> bool:
    """Whether an exhausted search over a complete contract has proven a negative.

    A run that stated the task fully, enumerated every candidate it could
    observe and rejected all of them has shown -- over its own contract and its
    own observations -- that no valid complete assignment exists.  Reporting
    that as a partial plan makes it indistinguishable from a run that merely
    stopped early, which is why infeasibility was never validly concluded.

    All three conditions are required.  Without a complete contract the run
    never stated the task; without exhaustion it has not finished looking; and
    with a valid grounding nothing was shown to be impossible.

    This lives here, and not inline at a status branch, because the three
    domains decide their terminal status in three different places and an
    implementation added to one of them silently does nothing for the other two.
    """
    contract_complete = bool(getattr(
        specification, "online_executable_contract_complete",
        getattr(specification, "required_contract_complete", False)))
    search_exhausted = bool(
        (getattr(satisfaction, "evidence", None) or {}).get("search_exhausted"))
    grounded = bool(getattr(satisfaction, "complete", False))
    return contract_complete and search_exhausted and not grounded
