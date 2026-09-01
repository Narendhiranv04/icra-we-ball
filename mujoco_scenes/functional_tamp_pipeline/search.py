"""Shared ranked evidence-acquisition loop."""

from __future__ import annotations

from typing import Any, Protocol

from .models import (
    FunctionalRequirementGraph,
    FunctionalSpecification,
    GraphGroundingResult,
    SatisfactionResult,
    SearchRegionContract,
    freeze_search_region_contract,
)


class SearchDomain(Protocol):
    def observe_initial(self) -> None: ...
    def evaluate_satisfaction(self, search_exhausted: bool = False) -> SatisfactionResult: ...
    def open_region(self, region: str) -> dict[str, Any]: ...
    def observe_after_open(self, region: str) -> None: ...


def _extract_scene_graph_dict(domain: SearchDomain) -> dict[str, Any] | None:
    graph = getattr(domain, "graph", None)
    if graph is not None and hasattr(graph, "to_dict"):
        return graph.to_dict()
    return None


def search_until_satisfied(
    domain: SearchDomain,
    specification: FunctionalSpecification,
    *,
    search_order: tuple[str, ...] | SearchRegionContract | None = None,
    search_contract: SearchRegionContract | None = None,
    observer: Any = None,
    emit=print,
) -> tuple[SatisfactionResult, tuple[str, ...]]:
    if search_contract is not None:
        order = tuple(search_contract.canonical_region_ids)
    elif isinstance(search_order, SearchRegionContract):
        order = tuple(search_order.canonical_region_ids)
    elif search_order is not None:
        order = tuple(search_order)
    elif hasattr(specification, "region_ranking") and specification.region_ranking:
        order = tuple(specification.region_ranking)
    else:
        order = tuple(freeze_search_region_contract(specification).canonical_region_ids)
    domain.observe_initial()
    if observer is not None:
        sg_dict = _extract_scene_graph_dict(domain)
        frame_rgb = getattr(domain, "latest_frame_rgb", None)
        observer("observation_updated", {
            "stage": "initial",
            "inspected_regions": [],
            "scene_graph": sg_dict,
            "frame_rgb": frame_rgb,
        })

    result = domain.evaluate_satisfaction()
    emit(f"[SEARCH] Initial functional satisfaction: {result.status}")
    if observer is not None:
        sg_dict = _extract_scene_graph_dict(domain)
        observer("grounding_updated", {
            "grounding": result.to_dict(),
            "satisfied": bool(result.satisfied),
            "status": result.status,
            "scene_graph": sg_dict,
        })

    inspected: list[str] = []
    if result.satisfied:
        return result, tuple(inspected)

    for idx, region in enumerate(order):
        emit(f"[SEARCH] Opening region: {region}")
        if observer is not None:
            observer("search_region_selected", {
                "region": region,
                "index": idx,
                "total_regions": len(order),
            })
        opened = domain.open_region(region)
        if not opened.get("success", False):
            raise RuntimeError(f"Physical OPEN({region}) failed: {opened}")
        if observer is not None:
            observer("search_region_opened", {
                "region": region,
                "success": True,
                "exploratory": True,
            })
        inspected.append(region)
        domain.observe_after_open(region)
        if observer is not None:
            sg_dict = _extract_scene_graph_dict(domain)
            frame_rgb = getattr(domain, "latest_frame_rgb", None)
            observer("observation_updated", {
                "stage": f"after_{region}",
                "inspected_regions": list(inspected),
                "scene_graph": sg_dict,
                "frame_rgb": frame_rgb,
            })
        result = domain.evaluate_satisfaction()
        emit(f"[SEARCH] Functional satisfaction: {result.status}")
        if observer is not None:
            sg_dict = _extract_scene_graph_dict(domain)
            observer("grounding_updated", {
                "grounding": result.to_dict(),
                "satisfied": bool(result.satisfied),
                "status": result.status,
                "scene_graph": sg_dict,
            })
        if result.satisfied:
            return result, tuple(inspected)

    try:
        final_result = domain.evaluate_satisfaction(search_exhausted=True)
    except TypeError:
        final_result = domain.evaluate_satisfaction()
    if observer is not None:
        sg_dict = _extract_scene_graph_dict(domain)
        observer("grounding_updated", {
            "grounding": final_result.to_dict(),
            "satisfied": bool(final_result.satisfied),
            "status": final_result.status,
            "scene_graph": sg_dict,
        })

    return GraphGroundingResult(
        status=final_result.status,
        complete=final_result.complete,
        assignment=final_result.assignment,
        operation_bindings=final_result.operation_bindings,
        missing_roles=final_result.missing_roles,
        unsatisfied_relations=final_result.unsatisfied_relations,
        unresolved_constraints=final_result.unresolved_constraints,
        evidence=final_result.evidence,
    ), tuple(inspected)
