"""Hidden terminal requirements for the kitchen benchmark."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import (
    HiddenBenchmarkContext,
    TerminalStateSnapshot,
    check,
    effect_exists,
    physical_on,
    required_sequence,
    required_string,
)


def evaluate_kitchen_requirements(
    terminal_state: TerminalStateSnapshot,
    effect_ledger: Sequence[Mapping[str, Any]],
    hidden_context: HiddenBenchmarkContext,
) -> tuple[Mapping[str, Any], ...]:
    """Check physical serving plus certified abstract liquid effects."""
    requirements = hidden_context.requirements
    coffee_vessels = required_sequence(requirements, "coffee_vessels")
    soup_vessels = required_sequence(requirements, "soup_vessels")
    water_sources = required_sequence(requirements, "water_sources")
    coffee_sources = required_sequence(requirements, "coffee_sources")
    stirrers = required_sequence(requirements, "suitable_stirrers")
    soup_utensils = required_sequence(requirements, "suitable_soup_utensils")
    serving_support = required_string(requirements, "serving_support")
    water_content = required_string(requirements, "water_content")
    coffee_content = required_string(requirements, "coffee_content")
    contained = terminal_state.relations.get("contained_in", {})
    if not isinstance(contained, Mapping):
        contained = {}

    coffee_distinct = len(set(coffee_vessels)) == len(coffee_vessels) == 2
    soup_distinct = len(set(soup_vessels)) == len(soup_vessels) == 2
    vessel_groups_disjoint = set(coffee_vessels).isdisjoint(soup_vessels)
    served_coffee = all(
        physical_on(terminal_state.objects.get(vessel, {}), serving_support)
        for vessel in coffee_vessels
    )
    served_soup = all(
        physical_on(terminal_state.objects.get(vessel, {}), serving_support)
        for vessel in soup_vessels
    )
    coffee_prepared = all(
        any(
            effect_exists(
                effect_ledger, "POUR_COMPLETED", (source, vessel, water_content)
            )
            for source in water_sources
        )
        and any(
            effect_exists(
                effect_ledger, "POUR_COMPLETED", (source, vessel, coffee_content)
            )
            for source in coffee_sources
        )
        and any(
            effect_exists(effect_ledger, "STIR_COMPLETED", (tool, vessel))
            for tool in stirrers
        )
        for vessel in coffee_vessels
    )
    utensils_by_bowl = {
        vessel: tuple(str(item) for item in contained.get(vessel, ()))
        for vessel in soup_vessels
    }
    chosen_utensils = [
        next(
            (tool for tool in utensils_by_bowl[vessel] if tool in soup_utensils),
            None,
        )
        for vessel in soup_vessels
    ]
    soup_equipped = all(tool is not None for tool in chosen_utensils)
    soup_tool_distinct = soup_equipped and len(set(chosen_utensils)) == 2
    contained_physically = all(
        terminal_state.objects.get(str(tool), {}).get("contained_stably") is True
        for tool in chosen_utensils
        if tool is not None
    )
    nothing_held = not terminal_state.held_objects

    return (
        check("two_distinct_coffee_vessels", coffee_distinct),
        check("coffee_vessels_physically_served", served_coffee),
        check("coffee_ingredients_and_stirring_verified", coffee_prepared),
        check("two_distinct_soup_vessels", soup_distinct),
        check("coffee_and_soup_vessels_distinct", vessel_groups_disjoint),
        check("soup_vessels_physically_served", served_soup),
        check(
            "distinct_suitable_soup_utensils_contained",
            soup_tool_distinct and contained_physically,
            utensils=chosen_utensils,
        ),
        check("no_required_object_held", nothing_held),
    )
