"""Authoritative top-level outcome taxonomy for the Functional-TAMP pipeline."""

from dataclasses import dataclass, field
from typing import Any


PIPELINE_OUTCOMES = frozenset({
    "SUCCESS",
    "TASK_SPECIFICATION_FAILURE",
    "GRAPH_COMPILATION_FAILURE",
    "OBJECT_DISCOVERY_FAILURE",
    "FUNCTIONAL_ASSIGNMENT_FAILURE",
    "PLANNING_FAILURE",
})


@dataclass(frozen=True)
class PipelineOutcome:
    category: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


def complete_planning_contract(specification, grounding, search_statistics, validation) -> bool:
    """Return whether the original task—not a projected subgraph—was fully planned."""
    operation_bindings = getattr(grounding, "operation_bindings", {})
    operations_complete = all(
        group.id in operation_bindings
        and len(operation_bindings[group.id]) >= group.required_target_count
        for group in specification.operation_groups
    )
    return bool(
        getattr(grounding, "complete", False)
        and not search_statistics.get("is_partial", False)
        # Legacy/custom planners may omit validation metadata; an explicit
        # non-valid value still vetoes completeness.
        and validation.get("status", "VALID") == "VALID"
        and validation.get("goal_status", "GOAL_SATISFIED") == "GOAL_SATISFIED"
        and operations_complete
    )


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
) -> PipelineOutcome:
    """Map stage evidence to exactly one frozen terminal category."""
    facts = dict(evidence or {})
    facts.update({
        "task_specification_valid": task_specification_valid,
        "graph_compiled": graph_compiled,
        "search_exhausted": search_exhausted,
        "individual_candidates_sufficient": individual_candidates_sufficient,
        "functional_assignment_complete": functional_assignment_complete,
        "planning_invoked": planning_invoked,
        "plan_complete": plan_complete,
    })
    if not task_specification_valid:
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
