"""Build planner state from observed registries without simulator identity leaks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .state import (
    ObjectObservation,
    ObservedState,
    RegionObservation,
    RobotObservation,
)


_OPEN_REGIONS = frozenset(("INITIAL", "countertop", "table", "room"))


def observed_registry_state(
    inventory: Mapping[str, Any],
    region_states: Mapping[str, Mapping[str, Any]],
    *,
    robot_location: str,
    held_object: str | None,
    revision: int,
    visible_object_ids: set[str] | frozenset[str] | None = None,
    live_locations: Mapping[str, str | None] | None = None,
) -> ObservedState:
    """Convert the execution inventory into the common bounded state.

    The inventory must come from point-cloud/semantic observations. Backend
    resolution records are deliberately not accepted by this function.
    """
    regions = {
        region_id: RegionObservation(
            region_id,
            str(row.get("category", "region")),
            True,
            inspected=bool(row.get("inspected", False)),
            open=(bool(row["open"]) if "open" in row else None),
        )
        for region_id, row in region_states.items()
    }
    objects: dict[str, ObjectObservation] = {}
    location_overrides = {} if live_locations is None else dict(live_locations)
    for row in inventory.get("objects", ()):
        object_id = str(row["generic_object_id"])
        context = row.get("source_context", {})
        discovered_source = str(
            context.get("observed_source_region") or "countertop"
        )
        source = location_overrides.get(object_id, discovered_source)
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError(
                f"Live location for {object_id} must be a non-empty string or null"
            )
        region = regions.get(source) if source is not None else None
        inferred_visible = bool(
            object_id == held_object
            or source in _OPEN_REGIONS
            or (region is not None and region.open is True)
        )
        visible = (
            object_id in visible_object_ids
            if visible_object_ids is not None
            else inferred_visible
        )
        facts = {
            "dimensions_m": dict(row.get("observed_dimensions_m", {})),
            "source_region": source,
            "discovered_source_region": discovered_source,
            "semantic_provenance": row.get("semantic_label_source"),
            "selected_functions": list(row.get("selected_functions", ())),
        }
        objects[object_id] = ObjectObservation(
            object_id,
            str(row.get("semantic_label") or "unknown_object"),
            visible,
            source,
            facts,
        )
    return ObservedState(
        objects,
        regions,
        RobotObservation(robot_location, held_object),
        revision=revision,
    )
