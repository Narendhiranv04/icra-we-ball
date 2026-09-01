"""Immutable search-region contract and deterministic policy handoff for Phase 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random
from typing import Any, Final, Mapping, Sequence, Tuple

from mujoco_scenes.final_paper_variant_labels import paper_variant_label
from .errors import SearchRegionContractError

PHASE3_SEARCH_REGION_POLICY_VERSION: Final[str] = "phase3_p3h_v1"

# Domain canonical search region registries
CANONICAL_SEARCH_REGIONS: Final[Mapping[str, Tuple[str, ...]]] = {
    "kitchen": ("D1", "D2", "C2", "B1", "C1"),
    "workshop": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    "living_room": (),
}

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


@dataclass(frozen=True)
class SearchRegionContract:
    """Immutable canonical search-region contract for Phase 3 runtime execution."""

    domain: str
    canonical_region_ids: tuple[str, ...] = field(default_factory=tuple)
    source: str = "UNKNOWN"
    policy_version: str = PHASE3_SEARCH_REGION_POLICY_VERSION
    proposal_trace: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    no_search_required: bool = False
    search_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain:
            raise SearchRegionContractError(
                f"SearchRegionContract domain must be a non-empty string, got {self.domain!r}"
            )
        if not isinstance(self.canonical_region_ids, tuple):
            object.__setattr__(self, "canonical_region_ids", tuple(map(str, self.canonical_region_ids)))
        if not isinstance(self.proposal_trace, tuple):
            frozen_trace = tuple(
                copy.deepcopy(item) if isinstance(item, dict) else item
                for item in self.proposal_trace
            )
            object.__setattr__(self, "proposal_trace", frozen_trace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "canonical_region_ids": list(self.canonical_region_ids),
            "source": self.source,
            "policy_version": self.policy_version,
            "proposal_trace": [copy.deepcopy(t) for t in self.proposal_trace],
            "no_search_required": self.no_search_required,
            "search_seed": self.search_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchRegionContract:
        return cls(
            domain=str(data["domain"]),
            canonical_region_ids=tuple(map(str, data.get("canonical_region_ids", ()))),
            source=str(data.get("source", "UNKNOWN")),
            policy_version=str(data.get("policy_version", PHASE3_SEARCH_REGION_POLICY_VERSION)),
            proposal_trace=tuple(copy.deepcopy(t) for t in data.get("proposal_trace", ())),
            no_search_required=bool(data.get("no_search_required", False)),
            search_seed=int(data["search_seed"]) if data.get("search_seed") is not None else None,
        )


def validate_search_order_preflight(
    domain: str,
    source: str = "auto",
    *,
    mode: str = "gt",
    seed: int | None = None,
) -> None:
    """Validate search order CLI configuration before doing any expensive rendering or provider calls."""
    if source not in {"auto", "oracle", "provider", "random", "fixed"}:
        raise SearchRegionContractError(
            f"Unknown search-order source: {source!r}. "
            "Must be 'auto', 'oracle', 'provider', 'random', or 'fixed'."
        )

    # Normalize deprecated alias 'fixed' to 'oracle'
    if source == "fixed":
        source = "oracle"

    if domain == "living_room":
        if source in {"oracle", "random"}:
            raise SearchRegionContractError(f"Search order {source!r} is not applicable for living_room")
        if seed is not None:
            raise SearchRegionContractError("--search-seed is not applicable for living_room")
        return

    effective_source = ("oracle" if mode == "gt" else "provider") if source == "auto" else source

    if effective_source != "random" and seed is not None:
        raise SearchRegionContractError(
            f"--search-seed is only valid for random search order, got seed={seed} with source={effective_source!r}"
        )

    if effective_source == "oracle" and mode == "vlm":
        raise SearchRegionContractError("oracle search is privileged and only valid with GT mode")

    if effective_source == "random":
        if seed is None:
            raise SearchRegionContractError("random search requires --search-seed")
        if not isinstance(seed, int) or seed < 0:
            raise SearchRegionContractError(f"random search seed must be a non-negative integer, got {seed!r}")


def freeze_search_region_contract(
    specification: Any,
    domain: str | None = None,
    source: str = "auto",
    *,
    mode: str = "gt",
    variant: str | None = None,
    seed: int | None = None,
) -> SearchRegionContract:
    """Resolve and freeze an immutable SearchRegionContract from G_F specification.

    This represents the authoritative one-time handoff boundary between specification
    and runtime perception/inspection.
    """
    spec_domain = getattr(specification, "domain", None) or domain
    eff_domain = domain or spec_domain
    if eff_domain is None:
        raise SearchRegionContractError("Domain must be specified either in specification or as argument")

    if spec_domain is not None and domain is not None and domain != spec_domain:
        raise SearchRegionContractError(f"Domain mismatch: requested {domain!r} but specification has {spec_domain!r}")

    if eff_domain not in CANONICAL_SEARCH_REGIONS:
        raise SearchRegionContractError(f"Unknown domain {eff_domain!r} for search region resolution")

    validate_search_order_preflight(eff_domain, source, mode=mode, seed=seed)

    # Living Room: strictly zero hidden search regions
    if eff_domain == "living_room":
        spec_candidates = getattr(specification, "candidate_regions", ())
        spec_ranking = getattr(specification, "region_ranking", ())
        if spec_candidates or spec_ranking:
            raise SearchRegionContractError(
                f"living_room has no inspectable search regions, but received: candidate_regions={spec_candidates}, region_ranking={spec_ranking}"
            )
        return SearchRegionContract(
            domain="living_room",
            canonical_region_ids=(),
            source="SYSTEM_DECLARED_NO_SEARCH",
            no_search_required=True,
            policy_version=PHASE3_SEARCH_REGION_POLICY_VERSION,
            proposal_trace=(),
        )

    # For search-requiring domains (kitchen, workshop), candidate_regions must be provided
    spec_candidates = getattr(specification, "candidate_regions", ())
    spec_ranking = getattr(specification, "region_ranking", ())
    if not spec_candidates and not spec_ranking:
        raise SearchRegionContractError(f"Missing search region metadata for domain {eff_domain!r}")

    # Resolve effective source
    if source == "auto":
        eff_source = "oracle" if (mode == "gt" and variant is not None) else "provider"
    else:
        eff_source = source

    if eff_source == "fixed":
        eff_source = "oracle"

    effective_seed: int | None = None
    if eff_source == "oracle":
        if variant is None:
            raise SearchRegionContractError(f"variant is required to resolve oracle search order for domain {eff_domain!r}")
        normalized_variant = paper_variant_label(eff_domain, variant)
        domain_oracle = ORACLE_SEARCH_ORDERS.get(eff_domain, {})
        if normalized_variant not in domain_oracle:
            raise SearchRegionContractError(f"No oracle search order defined for {eff_domain} variant {variant!r} ({normalized_variant})")
        order = tuple(domain_oracle[normalized_variant])
        source_label = "GT_ORACLE_SEARCH_POLICY"

    elif eff_source == "provider":
        order = tuple(spec_ranking if spec_ranking else spec_candidates)
        source_label = "VLM_CANONICALIZED_SEARCH_POLICY" if mode == "vlm" else "GT_EXPLICIT_SEARCH_POLICY"

    elif eff_source == "random":
        assert seed is not None
        base = list(spec_candidates)
        rng = random.Random(seed)
        rng.shuffle(base)
        order = tuple(base)
        effective_seed = seed
        source_label = "SEEDED_RANDOM_SEARCH_POLICY"

    else:
        raise SearchRegionContractError(f"Unhandled search-order source: {eff_source!r}")

    # Validate against allowed canonical domain vocabulary
    allowed_regions = set(CANONICAL_SEARCH_REGIONS[eff_domain])
    for reg in order:
        if reg not in allowed_regions:
            raise SearchRegionContractError(
                f"Unknown canonical search region {reg!r} for domain {eff_domain!r} (allowed: {sorted(allowed_regions)})"
            )

    # Validate duplicate canonical region IDs
    if len(order) != len(set(order)):
        raise SearchRegionContractError(f"Search order has duplicate regions: {order}")

    # Validate exact candidate set permutation
    candidates = set(spec_candidates)
    order_set = set(order)
    if order_set != candidates:
        missing = candidates - order_set
        extra = order_set - candidates
        err_parts = []
        if missing:
            err_parts.append(f"missing candidate regions {sorted(missing)}")
        if extra:
            err_parts.append(f"extra unknown regions {sorted(extra)}")
        raise SearchRegionContractError(
            f"Resolved search order {order} does not match candidate regions "
            f"{spec_candidates}: {', '.join(err_parts)}"
        )

    # Extract proposal trace from metadata if present
    proposal_trace_list = []
    metadata = getattr(specification, "metadata", {}) or {}
    trace = metadata.get("canonicalization_trace", {}) or {}
    if "proposal_trace" in trace and isinstance(trace["proposal_trace"], (list, tuple)):
        proposal_trace_list = list(trace["proposal_trace"])
    elif "inspectable_regions" in metadata and isinstance(metadata["inspectable_regions"], (list, tuple)):
        for idx, item in enumerate(metadata["inspectable_regions"]):
            if isinstance(item, dict):
                proposal_trace_list.append({
                    "source_index": idx,
                    "raw_label": item.get("label", ""),
                    "raw_description": item.get("visual_description", ""),
                    "status": "CANONICALIZED",
                })

    return SearchRegionContract(
        domain=eff_domain,
        canonical_region_ids=order,
        source=source_label,
        policy_version=PHASE3_SEARCH_REGION_POLICY_VERSION,
        proposal_trace=tuple(copy.deepcopy(proposal_trace_list)),
        no_search_required=False,
        search_seed=effective_seed,
    )
