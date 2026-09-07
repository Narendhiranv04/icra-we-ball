"""Shared ranked evidence-acquisition loop."""

from __future__ import annotations

from typing import Any, Iterable, Protocol

from .errors import SearchRegionContractError
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


def _clean_grounding_dict(result: SatisfactionResult) -> dict[str, Any]:
    gr = result.to_dict()
    if "evidence" in gr and isinstance(gr["evidence"], dict):
        gr["evidence"] = {k: v for k, v in gr["evidence"].items() if k != "grounding_snapshots"}
    return gr


def roles_in_missing_or_failed_bindings(
    graph_f: FunctionalRequirementGraph,
    grounding: GraphGroundingResult | None,
) -> set[str]:
    """Identify all functional roles implicated by ungrounded, under-cardinality, or failed bindings."""
    implicated: set[str] = set()
    if grounding is None:
        return set(graph_f.nodes.keys())

    # 1. Completely ungrounded roles
    implicated.update(grounding.missing_roles)

    # 2. Roles under cardinality
    if grounding.assignment is not None:
        for r_name, r_node in graph_f.nodes.items():
            if r_name not in grounding.assignment:
                implicated.add(r_name)
            else:
                assigned = grounding.assignment[r_name]
                cnt = (
                    len(assigned)
                    if isinstance(assigned, (list, tuple, set))
                    else (1 if assigned is not None else 0)
                )
                if cnt < r_node.minimum_count:
                    implicated.add(r_name)
    else:
        implicated.update(graph_f.nodes.keys())

    # 3. Endpoints of FALSE / unsatisfied relations
    for failure in grounding.unsatisfied_relations:
        if isinstance(failure, dict):
            if failure.get("subject_role"):
                implicated.add(failure["subject_role"])
            if failure.get("object_role"):
                implicated.add(failure["object_role"])
            if "group" in failure:
                grp = next((g for g in graph_f.operation_groups if g.id == failure["group"]), None)
                if grp:
                    implicated.add(grp.tool_role)
                    implicated.add(grp.target_role)
                    if grp.context_role:
                        implicated.add(grp.context_role)
        else:
            if hasattr(failure, "subject_role"):
                implicated.add(failure.subject_role)
            if hasattr(failure, "object_role"):
                implicated.add(failure.object_role)

    # 4. Endpoints of UNKNOWN required relations where new measurements/candidates may help
    if isinstance(grounding.evidence, dict):
        for unres in grounding.evidence.get("unresolved_relations", []):
            if isinstance(unres, dict):
                if unres.get("subject_role"):
                    implicated.add(unres["subject_role"])
                if unres.get("object_role"):
                    implicated.add(unres["object_role"])
                if "group" in unres:
                    grp = next((g for g in graph_f.operation_groups if g.id == unres["group"]), None)
                    if grp:
                        implicated.add(grp.tool_role)
                        implicated.add(grp.target_role)
                        if grp.context_role:
                            implicated.add(grp.context_role)
            else:
                if hasattr(unres, "subject_role"):
                    implicated.add(unres.subject_role)
                if hasattr(unres, "object_role"):
                    implicated.add(unres.object_role)

    # 5. Members of failed joint matching constraints
    for failure in grounding.unresolved_constraints:
        if hasattr(failure, "role_ids"):
            implicated.update(failure.role_ids)
        elif isinstance(failure, str) and failure in graph_f.nodes:
            implicated.add(failure)
        elif isinstance(failure, dict) and "role" in failure:
            implicated.add(failure["role"])

    # 6. Members of failed operation bindings
    for grp in graph_f.operation_groups:
        bindings = grounding.operation_bindings.get(grp.id, []) if grounding.operation_bindings else []
        if len(bindings) < grp.required_target_count:
            implicated.add(grp.tool_role)
            implicated.add(grp.target_role)
            if grp.context_role:
                implicated.add(grp.context_role)
        elif any(isinstance(b, dict) and b.get("status") not in (None, "TRUE", True) for b in bindings):
            implicated.add(grp.tool_role)
            implicated.add(grp.target_role)
            if grp.context_role:
                implicated.add(grp.context_role)

    return implicated


def classify_search_state(
    graph_f: FunctionalRequirementGraph,
    grounding: GraphGroundingResult | None,
    search_contract: SearchRegionContract | None,
    inspected_regions: Iterable[str] = (),
) -> str:
    """Classify current search eligibility and termination status."""
    contract_complete = graph_f.metadata.get("required_contract_complete")
    if contract_complete is None:
        contract_complete = getattr(graph_f, "required_contract_complete", True)
    if not contract_complete:
        return "CONTRACT_INCOMPLETE_NOT_SEARCHABLE"

    if grounding is not None and (grounding.complete or getattr(grounding, "satisfied", False)):
        return "SATISFIED"

    canonical_regions = (
        getattr(search_contract, "canonical_region_ids", ())
        if search_contract is not None
        else ()
    )
    inspected_set = set(inspected_regions)
    remaining_regions = [r for r in canonical_regions if r not in inspected_set]
    if not remaining_regions:
        return "SEARCH_EXHAUSTED"

    implicated = roles_in_missing_or_failed_bindings(graph_f, grounding)

    searchable = [
        graph_f.nodes[r]
        for r in implicated
        if r in graph_f.nodes
        and graph_f.nodes[r].entity_kind in {"OBJECT", "REGION"}
        and graph_f.nodes[r].semantic_categories
    ]

    if searchable:
        return "SEARCH_RECOVERABLE"

    return "GROUNDING_FAILURE_NOT_SEARCH_RECOVERABLE"


def compute_causal_search_recovery(
    initial_grounding_complete: bool,
    inspected_regions: Iterable[str],
    final_grounding_complete: bool,
) -> bool:
    """A search recovery counts only when:
    initial grounding was incomplete, at least one region was inspected, and final grounding is complete.
    """
    return bool(
        initial_grounding_complete is False
        and bool(tuple(inspected_regions))
        and final_grounding_complete is True
    )


def search_until_satisfied(
    domain: SearchDomain,
    specification: FunctionalSpecification,
    *,
    search_contract: SearchRegionContract,
    observer: Any = None,
    emit=print,
) -> tuple[SatisfactionResult, tuple[str, ...]]:
    if not isinstance(search_contract, SearchRegionContract):
        raise SearchRegionContractError(
            f"search_contract must be an instance of SearchRegionContract, got {type(search_contract).__name__}"
        )
    order = tuple(search_contract.canonical_region_ids)
    grounding_snapshots: list[dict[str, Any]] = []

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
    initial_grounding_complete = bool(result.satisfied)
    inspected: list[str] = []

    search_state = classify_search_state(specification, result, search_contract, inspected)
    emit(f"[SEARCH] Initial search state: {search_state}")
    grounding_snapshots.append({
        "stage": "initial",
        "inspected_regions": [],
        "search_state": search_state,
        "grounding": _clean_grounding_dict(result),
    })
    if observer is not None:
        sg_dict = _extract_scene_graph_dict(domain)
        observer("grounding_updated", {
            "grounding": result.to_dict(),
            "satisfied": bool(result.satisfied),
            "status": result.status,
            "scene_graph": sg_dict,
            "search_state": search_state,
        })

    if search_state != "SEARCH_RECOVERABLE":
        evidence = dict(result.evidence) if isinstance(result.evidence, dict) else {}
        evidence["grounding_snapshots"] = list(grounding_snapshots)
        evidence["search_state"] = search_state
        evidence["causal_search_recovery"] = compute_causal_search_recovery(
            initial_grounding_complete, inspected, bool(result.satisfied)
        )
        if hasattr(domain, "grounding_snapshots"):
            setattr(domain, "grounding_snapshots", list(grounding_snapshots))
        ret = GraphGroundingResult(
            status=result.status,
            complete=result.complete,
            assignment=result.assignment,
            operation_bindings=result.operation_bindings,
            missing_roles=result.missing_roles,
            unsatisfied_relations=result.unsatisfied_relations,
            unresolved_constraints=result.unresolved_constraints,
            evidence=evidence,
        )
        return ret, tuple(inspected)

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
        search_state = classify_search_state(specification, result, search_contract, inspected)
        emit(f"[SEARCH] Search state after {region}: {search_state}")
        grounding_snapshots.append({
            "stage": f"after_{region}",
            "inspected_regions": list(inspected),
            "region": region,
            "search_state": search_state,
            "grounding": _clean_grounding_dict(result),
        })
        if observer is not None:
            sg_dict = _extract_scene_graph_dict(domain)
            observer("grounding_updated", {
                "grounding": result.to_dict(),
                "satisfied": bool(result.satisfied),
                "status": result.status,
                "scene_graph": sg_dict,
                "search_state": search_state,
            })
        if search_state != "SEARCH_RECOVERABLE":
            if search_state == "SATISFIED":
                evidence = dict(result.evidence) if isinstance(result.evidence, dict) else {}
                evidence["grounding_snapshots"] = list(grounding_snapshots)
                evidence["search_state"] = search_state
                evidence["causal_search_recovery"] = compute_causal_search_recovery(
                    initial_grounding_complete, inspected, bool(result.satisfied)
                )
                if hasattr(domain, "grounding_snapshots"):
                    setattr(domain, "grounding_snapshots", list(grounding_snapshots))
                ret = GraphGroundingResult(
                    status=result.status,
                    complete=result.complete,
                    assignment=result.assignment,
                    operation_bindings=result.operation_bindings,
                    missing_roles=result.missing_roles,
                    unsatisfied_relations=result.unsatisfied_relations,
                    unresolved_constraints=result.unresolved_constraints,
                    evidence=evidence,
                )
                return ret, tuple(inspected)
            break

    try:
        final_result = domain.evaluate_satisfaction(search_exhausted=True)
    except TypeError:
        final_result = domain.evaluate_satisfaction()

    final_search_state = classify_search_state(specification, final_result, search_contract, inspected)
    grounding_snapshots.append({
        "stage": "final",
        "inspected_regions": list(inspected),
        "search_state": final_search_state,
        "grounding": _clean_grounding_dict(final_result),
    })
    if observer is not None:
        sg_dict = _extract_scene_graph_dict(domain)
        observer("grounding_updated", {
            "grounding": final_result.to_dict(),
            "satisfied": bool(final_result.satisfied),
            "status": final_result.status,
            "scene_graph": sg_dict,
            "search_state": final_search_state,
        })

    evidence = dict(final_result.evidence) if isinstance(final_result.evidence, dict) else {}
    evidence["grounding_snapshots"] = list(grounding_snapshots)
    evidence["search_state"] = final_search_state
    evidence["causal_search_recovery"] = compute_causal_search_recovery(
        initial_grounding_complete, inspected, bool(final_result.complete)
    )
    if hasattr(domain, "grounding_snapshots"):
        setattr(domain, "grounding_snapshots", list(grounding_snapshots))

    return GraphGroundingResult(
        status=final_result.status,
        complete=final_result.complete,
        assignment=final_result.assignment,
        operation_bindings=final_result.operation_bindings,
        missing_roles=final_result.missing_roles,
        unsatisfied_relations=final_result.unsatisfied_relations,
        unresolved_constraints=final_result.unresolved_constraints,
        evidence=evidence,
    ), tuple(inspected)

