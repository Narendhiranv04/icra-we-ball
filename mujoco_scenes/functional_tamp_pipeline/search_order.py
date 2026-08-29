"""Search-order resolution and validation for Phase 3."""

from __future__ import annotations

import random
from typing import Final, Mapping, Tuple

from mujoco_scenes.final_paper_variant_labels import paper_variant_label
from .models import FunctionalRequirementGraph

# Canonical fixed/default search orders
FIXED_SEARCH_ORDERS: Final[Mapping[str, Tuple[str, ...]]] = {
    "kitchen": ("D1", "D2", "C2", "B1", "C1"),
    "workshop": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
}

# Ground-truth verified oracle search orders
ORACLE_SEARCH_ORDERS: Final[Mapping[str, Mapping[str, Tuple[str, ...]]]] = {
    "kitchen": {
        "K1": ("D1", "D2", "C2", "B1", "C1"),
        "K2": ("C2", "D1", "D2", "B1", "C1"),
        "K3": ("B1", "D1", "D2", "C2", "C1"),
        "K4": ("C2", "B1", "D1", "D2", "C1"),
        "K5": ("D1", "D2", "C2", "B1", "C1"),
        "K6": ("D1", "D2", "C2", "B1", "C1"),
        "K7": ("D1", "D2", "C2", "B1", "C1"),
        "K8": ("D1", "D2", "C2", "B1", "C1"),
        "K9": ("D1", "D2", "C2", "B1", "C1"),
        "K10": ("D1", "D2", "C2", "B1", "C1"),
        "K11": ("D1", "D2", "C2", "B1", "C1"),
        "K12": ("D1", "D2", "C2", "B1", "C1"),
    },
    "workshop": {
        "W1": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        "W2": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        "W3": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        "W4": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        "W5": ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER"),
        "W6": ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER"),
        "W7": ("RIGHT_DRAWER", "TOOL_CABINET", "LEFT_DRAWER"),
        "W8": ("RIGHT_DRAWER", "TOOL_CABINET", "LEFT_DRAWER"),
        "W9": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        "W10": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    },
}


def validate_search_order_preflight(
    domain: str,
    source: str = "auto",
    *,
    mode: str = "gt",
    seed: int | None = None,
) -> None:
    """
    Validate search order CLI configuration before doing any expensive rendering or provider calls.
    Raises ValueError on invalid configurations.
    """
    if source not in {"auto", "oracle", "provider", "random", "fixed"}:
        raise ValueError(
            f"Unknown search-order source: {source!r}. "
            "Must be 'auto', 'oracle', 'provider', 'random', or 'fixed'."
        )

    # Normalize deprecated alias 'fixed' to 'oracle'
    if source == "fixed":
        source = "oracle"

    if domain == "living_room":
        if source in {"oracle", "random"}:
            raise ValueError(f"Search order {source!r} is not applicable for living_room")
        if seed is not None:
            raise ValueError("--search-seed is not applicable for living_room")
        return

    effective_source = ("oracle" if mode == "gt" else "provider") if source == "auto" else source

    if effective_source != "random" and seed is not None:
        raise ValueError(f"--search-seed is only valid for random search order, got seed={seed} with source={effective_source!r}")

    if effective_source == "oracle" and mode == "vlm":
        raise ValueError("oracle search is privileged and only valid with GT mode")

    if effective_source == "random":
        if seed is None:
            raise ValueError("random search requires --search-seed")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"random search seed must be a non-negative integer, got {seed!r}")


def resolve_search_order(
    specification: FunctionalRequirementGraph,
    domain: str,
    source: str = "auto",
    *,
    mode: str = "gt",
    variant: str | None = None,
    seed: int | None = None,
) -> tuple[tuple[str, ...], str, int | None]:
    """
    Resolve the search order for a run.
    Pure helper: does not access scenes, G_O, or hidden simulation state.
    Does NOT mutate specification.

    Returns: (resolved_order, effective_source, effective_seed)
    """
    validate_search_order_preflight(domain, source, mode=mode, seed=seed)

    # Normalize deprecated alias 'fixed' to 'oracle'
    if source == "fixed":
        source = "oracle"

    if domain == "living_room":
        return (), "not_applicable", None

    # Resolve effective source for 'auto'
    if source == "auto":
        effective_source = "oracle" if mode == "gt" else "provider"
    else:
        effective_source = source

    if effective_source == "oracle":
        if variant is None:
            raise ValueError(f"variant is required to resolve oracle search order for domain {domain!r}")
        normalized_variant = paper_variant_label(domain, variant)
        domain_oracle = ORACLE_SEARCH_ORDERS.get(domain, {})
        if normalized_variant not in domain_oracle:
            raise ValueError(f"No oracle search order defined for {domain} variant {variant!r} ({normalized_variant})")
        order = domain_oracle[normalized_variant]
        effective_seed = None

    elif effective_source == "provider":
        order = specification.region_ranking
        effective_seed = None

    elif effective_source == "random":
        assert seed is not None
        base = list(specification.candidate_regions)
        rng = random.Random(seed)
        rng.shuffle(base)
        order = tuple(base)
        effective_seed = seed

    else:
        raise ValueError(f"Unhandled search-order source: {effective_source!r}")

    # Validate exact candidate set permutation
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

    return tuple(order), effective_source, effective_seed
