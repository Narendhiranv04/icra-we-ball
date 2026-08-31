"""Deterministic Workshop planner-instance to simulator-body resolution."""

from __future__ import annotations

import math
from typing import Any, Iterable


class WorkshopEntityResolutionError(ValueError):
    """Raised when a frozen Phase-3 instance cannot be resolved uniquely."""


# Phase-3's Workshop association evaluator accepts a measured centroid as the
# same physical instance only within 0.16 m (workshop_phase1/evaluation.py).
# The execution boundary reuses that documented registration tolerance rather
# than accepting nearest-neighbour rank alone.
MAX_CENTROID_CORRESPONDENCE_ERROR_M = 0.16


def _objects_by_id(observed_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = observed_graph.get("objects", {})
    if isinstance(objects, dict):
        return {str(key): dict(value) for key, value in objects.items()}
    if isinstance(objects, list):
        return {
            str(row["instance_id"]): dict(row)
            for row in objects
            if isinstance(row, dict) and row.get("instance_id")
        }
    raise WorkshopEntityResolutionError("Observed graph has no object registry")


def _pick_sources(actions: Iterable[dict[str, Any]]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for action in actions:
        if str(action.get("operator", "")).upper() != "PICK":
            continue
        arguments = action.get("arguments", [])
        if len(arguments) < 2:
            continue
        planner_id, source = map(str, arguments[:2])
        previous = sources.setdefault(planner_id, source)
        if previous != source:
            raise WorkshopEntityResolutionError(
                f"Planner instance {planner_id} has conflicting PICK sources: "
                f"{previous} and {source}"
            )
    return sources


def resolve_workshop_entities(
    planner_ids: Iterable[str],
    observed_graph: dict[str, Any],
    actions: Iterable[dict[str, Any]],
    simulator_candidates: Iterable[dict[str, Any]],
    *,
    ambiguity_margin_m: float = 0.01,
    maximum_centroid_error_m: float = MAX_CENTROID_CORRESPONDENCE_ERROR_M,
) -> dict[str, Any]:
    """Resolve frozen G_O IDs without selecting new functional fillers.

    Source region is a hard identity gate.  When more than one simulator body
    occupies that region, the observed centroid must identify a unique nearest
    body.  Semantic labels are retained as audit evidence but never select or
    reject a body.
    """
    ordered_planner_ids = tuple(dict.fromkeys(map(str, planner_ids)))
    observed = _objects_by_id(observed_graph)
    sources = _pick_sources(actions)
    candidates = [dict(row) for row in simulator_candidates]
    rows: list[dict[str, Any]] = []
    for planner_id in ordered_planner_ids:
        node = observed.get(planner_id)
        if node is None:
            raise WorkshopEntityResolutionError(
                f"Frozen planner instance {planner_id} is absent from final G_O"
            )
        observed_source = str(node.get("source_region") or node.get("region") or "")
        plan_source = sources.get(planner_id)
        if not observed_source or not plan_source:
            raise WorkshopEntityResolutionError(
                f"Frozen planner instance {planner_id} has no source-region evidence"
            )
        if observed_source != plan_source:
            raise WorkshopEntityResolutionError(
                f"Frozen planner instance {planner_id} source mismatch: "
                f"final G_O={observed_source}, plan={plan_source}"
            )
        region_candidates = [
            row for row in candidates
            if str(row.get("source_region")) == observed_source
        ]
        if not region_candidates:
            raise WorkshopEntityResolutionError(
                f"No simulator body is present in source region {observed_source} "
                f"for {planner_id}"
            )
        geometry = node.get("geometry") or node.get("unary_properties") or {}
        centroid = geometry.get("centroid_world_m")
        if centroid is None:
            raise WorkshopEntityResolutionError(
                f"Frozen planner instance {planner_id} has no centroid evidence"
            )
        try:
            measured = tuple(map(float, centroid))
            if len(measured) != 3:
                raise ValueError
        except (TypeError, ValueError):
            raise WorkshopEntityResolutionError(
                f"Invalid observed centroid for {planner_id}: {centroid!r}"
            )
        ranked: list[tuple[float, dict[str, Any]]] = []
        for candidate in region_candidates:
            candidate_centroid = candidate.get("centroid_world_m")
            if candidate_centroid is None:
                raise WorkshopEntityResolutionError(
                    f"Simulator candidate {candidate.get('simulator_id')} has "
                    "no centroid evidence"
                )
            ranked.append((
                math.dist(measured, tuple(map(float, candidate_centroid))),
                candidate,
            ))
        ranked.sort(key=lambda item: (item[0], str(item[1]["simulator_id"])))
        distance, selected = ranked[0]
        if distance > maximum_centroid_error_m:
            raise WorkshopEntityResolutionError(
                f"Centroid correspondence for {planner_id} exceeds absolute "
                f"gate: {distance:.6f} m > {maximum_centroid_error_m:.6f} m"
            )
        if len(ranked) > 1:
            if ranked[1][0] - ranked[0][0] <= ambiguity_margin_m:
                raise WorkshopEntityResolutionError(
                    f"Ambiguous simulator mapping for {planner_id} in "
                    f"{observed_source}: nearest distances "
                    f"{ranked[0][0]:.6f} and {ranked[1][0]:.6f} m"
                )
        method = "SOURCE_REGION_ABSOLUTE_GATED_UNIQUE_NEAREST_CENTROID_V2"
        semantic = node.get("semantic_labels") or {}
        rows.append({
            "planner_id": planner_id,
            "simulator_id": str(selected["simulator_id"]),
            "entity_kind": "OBJECT",
            "source_region": observed_source,
            "resolution_method": method,
            "evidence": {
                "observed_centroid_world_m": centroid,
                "simulator_centroid_world_m": selected.get("centroid_world_m"),
                "centroid_distance_m": distance,
                "maximum_centroid_correspondence_error_m": (
                    maximum_centroid_error_m
                ),
                "correspondence_threshold_basis": (
                    "PHASE3_WORKSHOP_ASSOCIATION_EVALUATION_RADIUS"
                ),
                "observed_canonical_category": node.get("canonical_category"),
                "observed_semantic_labels": semantic,
                "semantic_evidence_used_for_selection": False,
                "candidate_distances_m": [
                    {
                        "simulator_id": str(candidate["simulator_id"]),
                        "distance_m": candidate_distance,
                    }
                    for candidate_distance, candidate in ranked
                ],
            },
        })
    simulator_ids = [row["simulator_id"] for row in rows]
    if len(set(simulator_ids)) != len(simulator_ids):
        raise WorkshopEntityResolutionError(
            "Workshop execution mapping is not one-to-one"
        )
    return {
        "schema_version": 2,
        "all_resolved": len(rows) == len(ordered_planner_ids),
        "one_to_one": True,
        "objects": rows,
    }
