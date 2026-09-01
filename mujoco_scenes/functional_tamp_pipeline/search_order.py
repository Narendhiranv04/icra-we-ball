"""Search-order resolution and validation for Phase 3."""

from __future__ import annotations

from typing import Final, Mapping, Tuple

from .models import FunctionalRequirementGraph
from .search_contract import (
    CANONICAL_SEARCH_REGIONS,
    FIXED_SEARCH_ORDERS,
    ORACLE_SEARCH_ORDERS,
    PHASE3_SEARCH_REGION_POLICY_VERSION,
    SearchRegionContract,
    SearchRegionContractError,
    freeze_search_region_contract,
    validate_search_order_preflight,
)


def resolve_search_order(
    specification: FunctionalRequirementGraph,
    domain: str,
    source: str = "auto",
    *,
    mode: str = "gt",
    variant: str | None = None,
    seed: int | None = None,
) -> tuple[tuple[str, ...], str, int | None]:
    """Resolve the search order for a run.
    Pure helper: does not access scenes, G_O, or hidden simulation state.
    Does NOT mutate specification.

    Returns: (resolved_order, effective_source, effective_seed)
    """
    contract = freeze_search_region_contract(
        specification,
        domain=domain,
        source=source,
        mode=mode,
        variant=variant,
        seed=seed,
    )
    if contract.no_search_required:
        return (), "not_applicable", None

    if source in {"oracle", "provider", "random", "fixed"}:
        eff_source = "oracle" if source == "fixed" else source
    else:
        eff_source = "oracle" if mode == "gt" else "provider"

    return contract.canonical_region_ids, eff_source, contract.search_seed
