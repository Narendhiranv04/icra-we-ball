"""Shared ranked evidence-acquisition loop."""

from __future__ import annotations

from typing import Any, Protocol

from .models import (
    FunctionalRequirementGraph,
    FunctionalSpecification,
    GraphGroundingResult,
    SatisfactionResult,
)


class SearchDomain(Protocol):
    def observe_initial(self) -> None: ...
    def evaluate_satisfaction(self) -> SatisfactionResult: ...
    def open_region(self, region: str) -> dict[str, Any]: ...
    def observe_after_open(self, region: str) -> None: ...


def search_until_satisfied(
    domain: SearchDomain,
    specification: FunctionalSpecification,
    *,
    emit=print,
) -> tuple[SatisfactionResult, tuple[str, ...]]:
    domain.observe_initial()
    result = domain.evaluate_satisfaction()
    emit(f"[SEARCH] Initial functional satisfaction: {result.status}")
    inspected: list[str] = []
    if result.satisfied:
        return result, tuple(inspected)
    for region in specification.region_ranking:
        emit(f"[SEARCH] Opening region: {region}")
        opened = domain.open_region(region)
        if not opened.get("success", False):
            raise RuntimeError(f"Physical OPEN({region}) failed: {opened}")
        inspected.append(region)
        domain.observe_after_open(region)
        result = domain.evaluate_satisfaction()
        emit(f"[SEARCH] Functional satisfaction: {result.status}")
        if result.satisfied:
            return result, tuple(inspected)
    try:
        final_result = domain.evaluate_satisfaction(search_exhausted=True)
    except TypeError:
        final_result = domain.evaluate_satisfaction()
    return GraphGroundingResult(
        status=final_result.status,
        complete=final_result.complete,
        assignment=final_result.assignment,
        unresolved_constraints=final_result.unresolved_constraints,
        evidence=final_result.evidence,
    ), tuple(inspected)
