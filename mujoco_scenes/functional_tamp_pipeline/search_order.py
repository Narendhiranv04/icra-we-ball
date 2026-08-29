"""Search-order resolution and validation for Phase 3."""

from __future__ import annotations

from typing import Final, Mapping, Tuple

from .models import FunctionalRequirementGraph

FIXED_SEARCH_ORDERS: Final[Mapping[str, Tuple[str, ...]]] = {
    "kitchen": ("D1", "D2", "C2", "B1", "C1"),
    "workshop": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
}


def resolve_search_order(
    specification: FunctionalRequirementGraph,
    domain: str,
    source: str = "provider",
) -> tuple[str, ...]:
    """
    Resolve the search order for a run from specification and requested source.
    Pure helper: does not access scenes, G_O, or hidden truth.
    Does NOT mutate specification.
    """
    if source not in {"provider", "fixed"}:
        raise ValueError(f"Unknown search-order source: {source!r}. Must be 'provider' or 'fixed'.")

    if domain == "living_room":
        return ()

    if source == "provider":
        order = specification.region_ranking
    elif source == "fixed":
        if domain not in FIXED_SEARCH_ORDERS:
            raise ValueError(f"No fixed search order defined for domain {domain!r}")
        order = FIXED_SEARCH_ORDERS[domain]
    else:
        raise ValueError(f"Unknown search-order source: {source!r}")

    # Validation
    if len(order) != len(set(order)):
        raise ValueError(f"Search order has duplicate regions: {order}")

    candidates = set(specification.candidate_regions)
    order_set = set(order)

    if order_set != candidates:
        missing = candidates - order_set
        extra = order_set - candidates
        err_parts = []
        if missing:
            err_parts.append(f"missing candidate regions {sorted(missing)}")
        if extra:
            err_parts.append(f"extra unknown regions {sorted(extra)}")
        raise ValueError(
            f"Resolved search order {order} does not match candidate regions "
            f"{specification.candidate_regions}: {', '.join(err_parts)}"
        )

    return tuple(order)
