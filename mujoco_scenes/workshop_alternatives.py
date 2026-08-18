"""Validate FM-ranked workshop alternatives against observed object and region geometry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


METHOD_FUNCTIONS = {
    "nail": "can_hammer",
    "screw": "can_drive_screw",
    "can_screw": "can_drive_screw",
    "can_drive_screw": "can_drive_screw",
}
MAX_ALTERNATIVES = 5


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

        # Support canonical functions with backward compatibility
        fn_set = set(functions)
        if "can_screw" in fn_set:
            fn_set.add("can_drive_screw")
        if "can_drive_screw" in fn_set:
            fn_set.add("can_screw")

        objects[object_id] = {
            "object_id": object_id,
            "functions": frozenset(fn_set),
            "geometry": dict(geometry),
            "source_region": raw.get("source_region"),
        }
    return objects


def _normalize_regions(
    observed_regions: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    regions: dict[str, dict[str, Any]] = {}
    for raw in observed_regions:
        region_id = raw.get("region_id")
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("Every observed region needs a region_id")
        if region_id in regions:
            raise ValueError(f"Duplicate observed region_id: {region_id}")
        regions[region_id] = dict(raw)
    return regions


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
    tool_reach = tool_geometry.get("reach_m", 0.10)
    min_required_reach = fastener_geometry.get("required_tool_reach_m", 0.05)

    profile_match = (
        isinstance(tip_profile, str)
        and isinstance(recess_profile, str)
        and tip_profile.upper() == recess_profile.upper()
    )
    width_match = tip_width <= recess_width * 1.05  # slight tolerance
    reach_match = tool_reach >= min_required_reach

    passed = profile_match and width_match and reach_match
    return passed, {
        "tool_tip_profile": tip_profile,
        "fastener_recess_profile": recess_profile,
        "tool_tip_width_m": tip_width,
        "fastener_recess_width_m": recess_width,
        "tool_reach_m": tool_reach,
        "required_tool_reach_m": min_required_reach,
    }


def _evaluate_proposal(
    proposal: Mapping[str, Any],
    objects: Mapping[str, dict[str, Any]],
    target: Mapping[str, Any],
    regions: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    method = proposal.get("method", "screw")
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
        tool_function: (
            tool_function in tool["functions"] or "can_screw" in tool["functions"]
        ),
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

    region_checks: dict[str, bool] = {}
    region_evidence: dict[str, Any] = {}

    # Check relational region coupling if proposal includes work_surface or parts_container
    if regions is not None:
        work_surface_id = proposal.get("work_surface_id")
        if work_surface_id is not None:
            if work_surface_id not in regions:
                region_checks["valid_work_surface"] = False
            else:
                surf = regions[work_surface_id]
                usable_area = surf.get("usable_area_m2", 0.0)
                tool_area = tool["geometry"].get("bounding_area_m2", 0.01)
                fastener_area = fastener["geometry"].get("bounding_area_m2", 0.001)
                fits_set = usable_area >= (tool_area + fastener_area) * 1.2
                region_checks["fits_work_surface"] = fits_set
                region_evidence["work_surface_usable_area_m2"] = usable_area
                region_evidence["required_set_area_m2"] = tool_area + fastener_area

        container_id = proposal.get("parts_container_id")
        if container_id is not None:
            if container_id not in regions:
                region_checks["valid_parts_container"] = False
            else:
                cont = regions[container_id]
                cavity_vol = cont.get("cavity_volume_m3", 0.0)
                is_open = cont.get("is_open", True)
                region_checks["fits_parts_container"] = is_open and cavity_vol > 0.0
                region_evidence["container_cavity_volume_m3"] = cavity_vol

    accepted = (
        all(semantic_checks.values())
        and all(geometry_checks.values())
        and all(region_checks.values())
    )

    return {
        "rank": int(proposal["rank"]),
        "method": method,
        "tool_object_id": tool_id,
        "fastener_object_id": fastener_id,
        "work_surface_id": proposal.get("work_surface_id"),
        "parts_container_id": proposal.get("parts_container_id"),
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
        "region_checks": region_checks,
        "geometry_evidence": {
            "hole_diameter_m": hole_diameter,
            "joint_depth_m": joint_depth,
            "radial_clearance_m": clearance,
            "fastener_diameter_m": fastener_diameter,
            "fastener_length_m": fastener_length,
            **mating_evidence,
            **region_evidence,
        },
        "status": "VALID" if accepted else "REJECTED",
    }


def evaluate_ranked_alternatives(
    *,
    observed_objects: Iterable[Mapping[str, Any]],
    target_geometry: Mapping[str, Any],
    ranked_proposals: Iterable[Mapping[str, Any]],
    observed_regions: Iterable[Mapping[str, Any]] | None = None,
    max_alternatives: int = MAX_ALTERNATIVES,
) -> dict[str, Any]:
    """Validate visible FM proposals and stop at the first geometric witness."""
    if not 1 <= max_alternatives <= MAX_ALTERNATIVES:
        raise ValueError(f"max_alternatives must be in [1, {MAX_ALTERNATIVES}]")
    objects = _normalize_objects(observed_objects)
    regions = _normalize_regions(observed_regions) if observed_regions is not None else None
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
        result = _evaluate_proposal(proposal, objects, target_geometry, regions=regions)
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
