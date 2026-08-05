"""Validate FM-ranked workshop alternatives against observed geometry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


METHOD_FUNCTIONS = {
    "nail": "can_hammer",
    "screw": "can_screw",
}
MAX_ALTERNATIVES = 3


def _positive_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Geometry field {key!r} must be numeric")
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"Geometry field {key!r} must be positive")
    return value


def _normalize_objects(
    observed_objects: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for raw in observed_objects:
        object_id = raw.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Every observed object needs an object_id")
        if object_id in objects:
            raise ValueError(f"Duplicate observed object_id: {object_id}")
        functions = raw.get("functions", ())
        geometry = raw.get("geometry", {})
        if not isinstance(functions, (list, tuple, set)) or not all(
            isinstance(function, str) for function in functions
        ):
            raise ValueError(f"Invalid functions for {object_id}")
        if not isinstance(geometry, Mapping):
            raise ValueError(f"Invalid geometry for {object_id}")
        objects[object_id] = {
            "object_id": object_id,
            "functions": frozenset(functions),
            "geometry": dict(geometry),
            "source_region": raw.get("source_region"),
        }
    return objects


def _check_tool_mates(
    method: str,
    tool_geometry: Mapping[str, Any],
    fastener_geometry: Mapping[str, Any],
) -> tuple[bool, dict[str, Any]]:
    if method == "nail":
        face_width = _positive_number(tool_geometry, "face_width_m")
        head_width = _positive_number(fastener_geometry, "head_width_m")
        return face_width >= head_width, {
            "tool_face_width_m": face_width,
            "fastener_head_width_m": head_width,
        }

    tip_profile = tool_geometry.get("tip_profile")
    recess_profile = fastener_geometry.get("recess_profile")
    tip_width = _positive_number(tool_geometry, "tip_width_m")
    recess_width = _positive_number(fastener_geometry, "recess_width_m")
    passed = (
        isinstance(tip_profile, str)
        and tip_profile == recess_profile
        and tip_width <= recess_width
    )
    return passed, {
        "tool_tip_profile": tip_profile,
        "fastener_recess_profile": recess_profile,
        "tool_tip_width_m": tip_width,
        "fastener_recess_width_m": recess_width,
    }


def _evaluate_proposal(
    proposal: Mapping[str, Any],
    objects: Mapping[str, dict[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    method = proposal.get("method")
    if method not in METHOD_FUNCTIONS:
        raise ValueError(f"Unknown fastening method: {method!r}")
    tool_id = proposal.get("tool_object_id")
    fastener_id = proposal.get("fastener_object_id")
    if tool_id not in objects or fastener_id not in objects:
        raise ValueError("A proposal referenced an object outside visible state")
    if tool_id == fastener_id:
        raise ValueError("Tool and fastener must be distinct observed objects")

    tool = objects[tool_id]
    fastener = objects[fastener_id]
    tool_function = METHOD_FUNCTIONS[method]
    semantic_checks = {
        tool_function: tool_function in tool["functions"],
        "can_fasten": "can_fasten" in fastener["functions"],
    }

    hole_diameter = _positive_number(target, "hole_diameter_m")
    joint_depth = _positive_number(target, "joint_depth_m")
    clearance = _positive_number(target, "radial_clearance_m")
    fastener_diameter = _positive_number(
        fastener["geometry"], "diameter_m"
    )
    fastener_length = _positive_number(fastener["geometry"], "length_m")
    tool_mates, mating_evidence = _check_tool_mates(
        method, tool["geometry"], fastener["geometry"]
    )
    geometry_checks = {
        "fits_hole": fastener_diameter + 2.0 * clearance <= hole_diameter,
        "reaches_joint": fastener_length >= joint_depth,
        "tool_mates": tool_mates,
    }
    accepted = all(semantic_checks.values()) and all(geometry_checks.values())
    return {
        "rank": int(proposal["rank"]),
        "method": method,
        "tool_object_id": tool_id,
        "fastener_object_id": fastener_id,
        "source_regions": sorted(
            {
                region
                for region in (
                    tool.get("source_region"),
                    fastener.get("source_region"),
                )
                if region is not None
            }
        ),
        "semantic_checks": semantic_checks,
        "geometry_checks": geometry_checks,
        "geometry_evidence": {
            "hole_diameter_m": hole_diameter,
            "joint_depth_m": joint_depth,
            "radial_clearance_m": clearance,
            "fastener_diameter_m": fastener_diameter,
            "fastener_length_m": fastener_length,
            **mating_evidence,
        },
        "status": "VALID" if accepted else "REJECTED",
    }


def evaluate_ranked_alternatives(
    *,
    observed_objects: Iterable[Mapping[str, Any]],
    target_geometry: Mapping[str, Any],
    ranked_proposals: Iterable[Mapping[str, Any]],
    max_alternatives: int = MAX_ALTERNATIVES,
) -> dict[str, Any]:
    """Validate visible FM proposals and stop at the first geometric witness.

    This function does not rank candidates and has no scene inventory. The FM
    supplies a ranking over visible object IDs; this stage only checks that the
    proposed objects are observed and satisfy the functional geometry.
    """
    if not 1 <= max_alternatives <= MAX_ALTERNATIVES:
        raise ValueError(f"max_alternatives must be in [1, {MAX_ALTERNATIVES}]")
    objects = _normalize_objects(observed_objects)
    proposals = list(ranked_proposals)
    if len(proposals) > max_alternatives:
        raise ValueError(
            f"Received {len(proposals)} proposals; maximum is {max_alternatives}"
        )
    ranks = [proposal.get("rank") for proposal in proposals]
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks):
        raise ValueError("Every proposal needs an integer rank")
    if sorted(ranks) != list(range(1, len(proposals) + 1)):
        raise ValueError("Proposal ranks must be unique and contiguous from 1")

    evaluated = []
    selected = None
    for proposal in sorted(proposals, key=lambda item: item["rank"]):
        result = _evaluate_proposal(proposal, objects, target_geometry)
        evaluated.append(result)
        if result["status"] == "VALID":
            selected = result
            break
    return {
        "status": "COMPLETE" if selected is not None else "INCOMPLETE",
        "selected": selected,
        "evaluated": evaluated,
        "early_terminated": selected is not None and len(evaluated) < len(proposals),
        "visible_object_ids": sorted(objects),
    }
