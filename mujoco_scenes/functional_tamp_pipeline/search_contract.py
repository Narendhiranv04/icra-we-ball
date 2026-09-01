"""Immutable search-region contract and deterministic policy handoff for Phase 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random
from typing import Any, Final, Mapping, Sequence, Tuple

from mujoco_scenes.final_paper_variant_labels import paper_variant_label
from .errors import SearchRegionContractError
from .system_context_registry import get_domain_search_regions

PHASE3_SEARCH_REGION_POLICY_VERSION: Final[str] = "phase3_p3h_1_v1"

# Base canonical deterministic ordering per domain (system-owned)
DOMAIN_CANONICAL_SEARCH_BASE_ORDERS: Final[Mapping[str, Tuple[str, ...]]] = {
    "kitchen": ("D1", "D2", "C2", "B1", "C1"),
    "workshop": ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    "living_room": (),
}

# Programmatically derived domain search regions to prevent drift from system_context_registry
CANONICAL_SEARCH_REGIONS: Final[Mapping[str, Tuple[str, ...]]] = {
    "kitchen": tuple(
        r for r in DOMAIN_CANONICAL_SEARCH_BASE_ORDERS["kitchen"]
        if r in get_domain_search_regions("kitchen")
    ),
    "workshop": tuple(
        r for r in DOMAIN_CANONICAL_SEARCH_BASE_ORDERS["workshop"]
        if r in get_domain_search_regions("workshop")
    ),
    "living_room": (),
}

# Canonical fixed/default search orders
FIXED_SEARCH_ORDERS: Final[Mapping[str, Tuple[str, ...]]] = DOMAIN_CANONICAL_SEARCH_BASE_ORDERS

# Ground-truth verified privileged diagnostic oracle search orders
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
class RegionProposalTraceEntry:
    """Immutable trace entry recording raw inspectable-region proposal resolution."""

    raw_index: int
    raw_id: str
    raw_label: str
    raw_visual_description: str
    canonical_region_id: str | None
    resolution_status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_index": self.raw_index,
            "raw_id": self.raw_id,
            "raw_label": self.raw_label,
            "raw_visual_description": self.raw_visual_description,
            "canonical_region_id": self.canonical_region_id,
            "resolution_status": self.resolution_status,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RegionProposalTraceEntry:
        return cls(
            raw_index=int(data.get("raw_index", 0)),
            raw_id=str(data.get("raw_id", "")),
            raw_label=str(data.get("raw_label", "")),
            raw_visual_description=str(data.get("raw_visual_description", "")),
            canonical_region_id=str(data["canonical_region_id"]) if data.get("canonical_region_id") is not None else None,
            resolution_status=str(data.get("resolution_status", "RESOLVED")),
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class SearchPolicyTraceEntry:
    """Immutable trace entry recording final search region rank and policy origin."""

    region_id: str
    final_rank: int
    provider_rank: int | None
    origin: str  # "PROVIDER_RANKED", "SYSTEM_COMPLETION", "SEEDED_RANDOM", "PRIVILEGED_GT_ORACLE", "GT_SYSTEM"

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "final_rank": self.final_rank,
            "provider_rank": self.provider_rank,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchPolicyTraceEntry:
        return cls(
            region_id=str(data["region_id"]),
            final_rank=int(data["final_rank"]),
            provider_rank=int(data["provider_rank"]) if data.get("provider_rank") is not None else None,
            origin=str(data.get("origin", "SYSTEM_COMPLETION")),
        )


@dataclass(frozen=True)
class SearchRegionContract:
    """Immutable canonical search-region contract for Phase 3 runtime execution."""

    domain: str
    canonical_region_ids: tuple[str, ...] = field(default_factory=tuple)
    source: str = "UNKNOWN"
    policy_version: str = PHASE3_SEARCH_REGION_POLICY_VERSION
    region_proposal_trace: tuple[RegionProposalTraceEntry, ...] = field(default_factory=tuple)
    search_policy_trace: tuple[SearchPolicyTraceEntry, ...] = field(default_factory=tuple)
    no_search_required: bool = False
    search_seed: int | None = None

    @property
    def proposal_trace(self) -> tuple[dict[str, Any], ...]:
        """Backwards-compatibility property returning detached proposal trace dictionaries."""
        return tuple(entry.to_dict() for entry in self.region_proposal_trace)

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain:
            raise SearchRegionContractError(
                f"SearchRegionContract domain must be a non-empty string, got {self.domain!r}"
            )
        if not isinstance(self.canonical_region_ids, tuple):
            object.__setattr__(self, "canonical_region_ids", tuple(map(str, self.canonical_region_ids)))
        if not isinstance(self.region_proposal_trace, tuple):
            prop_tuple = tuple(
                RegionProposalTraceEntry.from_dict(item) if isinstance(item, (dict, Mapping))
                else (item if isinstance(item, RegionProposalTraceEntry) else RegionProposalTraceEntry.from_dict(item.__dict__))
                for item in self.region_proposal_trace
            )
            object.__setattr__(self, "region_proposal_trace", prop_tuple)
        if not isinstance(self.search_policy_trace, tuple):
            policy_tuple = tuple(
                SearchPolicyTraceEntry.from_dict(item) if isinstance(item, (dict, Mapping))
                else (item if isinstance(item, SearchPolicyTraceEntry) else SearchPolicyTraceEntry.from_dict(item.__dict__))
                for item in self.search_policy_trace
            )
            object.__setattr__(self, "search_policy_trace", policy_tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "canonical_region_ids": list(self.canonical_region_ids),
            "source": self.source,
            "policy_version": self.policy_version,
            "region_proposal_trace": [t.to_dict() for t in self.region_proposal_trace],
            "search_policy_trace": [t.to_dict() for t in self.search_policy_trace],
            "proposal_trace": [t.to_dict() for t in self.region_proposal_trace],
            "no_search_required": self.no_search_required,
            "search_seed": self.search_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchRegionContract:
        raw_prop_trace = data.get("region_proposal_trace") or data.get("proposal_trace", ())
        prop_entries = tuple(
            RegionProposalTraceEntry.from_dict(t) if isinstance(t, (dict, Mapping))
            else (t if isinstance(t, RegionProposalTraceEntry) else RegionProposalTraceEntry.from_dict(t.__dict__))
            for t in raw_prop_trace
        )
        raw_policy_trace = data.get("search_policy_trace", ())
        policy_entries = tuple(
            SearchPolicyTraceEntry.from_dict(t) if isinstance(t, (dict, Mapping))
            else (t if isinstance(t, SearchPolicyTraceEntry) else SearchPolicyTraceEntry.from_dict(t.__dict__))
            for t in raw_policy_trace
        )
        return cls(
            domain=str(data["domain"]),
            canonical_region_ids=tuple(map(str, data.get("canonical_region_ids", ()))),
            source=str(data.get("source", "UNKNOWN")),
            policy_version=str(data.get("policy_version", PHASE3_SEARCH_REGION_POLICY_VERSION)),
            region_proposal_trace=prop_entries,
            search_policy_trace=policy_entries,
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

    effective_source = source

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

    if eff_domain not in DOMAIN_CANONICAL_SEARCH_BASE_ORDERS:
        raise SearchRegionContractError(f"Unknown domain {eff_domain!r} for search region resolution")

    system_regions_set = get_domain_search_regions(eff_domain)

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
            region_proposal_trace=(),
            search_policy_trace=(),
        )

    base_canonical_order = DOMAIN_CANONICAL_SEARCH_BASE_ORDERS[eff_domain]
    spec_candidates = tuple(getattr(specification, "candidate_regions", ()) or ())
    spec_ranking = tuple(getattr(specification, "region_ranking", ()) or ())

    # Reject duplicate candidate regions or rankings if passed
    if len(spec_ranking) != len(set(spec_ranking)):
        raise SearchRegionContractError(f"Search order has duplicate regions: {spec_ranking}")

    eff_source = source
    if eff_source == "fixed":
        eff_source = "oracle"

    effective_seed: int | None = None
    policy_trace_list: list[SearchPolicyTraceEntry] = []

    # 1. EXPLICIT PRIVILEGED ORACLE (Diagnostic only; GT only; requires explicit user selection)
    if eff_source == "oracle":
        if mode == "vlm":
            raise SearchRegionContractError("oracle search is privileged and only valid with GT mode")
        if variant is None:
            raise SearchRegionContractError(f"variant is required to resolve oracle search order for domain {eff_domain!r}")
        normalized_variant = paper_variant_label(eff_domain, variant)
        domain_oracle = ORACLE_SEARCH_ORDERS.get(eff_domain, {})
        if normalized_variant not in domain_oracle:
            raise SearchRegionContractError(f"No oracle search order defined for {eff_domain} variant {variant!r} ({normalized_variant})")
        order = tuple(domain_oracle[normalized_variant])
        source_label = "PRIVILEGED_GT_ORACLE_DIAGNOSTIC"
        for rank, reg in enumerate(order):
            policy_trace_list.append(SearchPolicyTraceEntry(
                region_id=reg,
                final_rank=rank,
                provider_rank=None,
                origin="PRIVILEGED_GT_ORACLE",
            ))

        # Check candidate regions if provided for oracle
        if spec_candidates:
            cand_set = set(spec_candidates)
            order_set = set(order)
            missing = cand_set - order_set
            extra = order_set - cand_set
            if missing or extra:
                err_parts = []
                if missing:
                    err_parts.append(f"missing candidate regions {sorted(missing)}")
                if extra:
                    err_parts.append(f"extra unknown regions {sorted(extra)}")
                raise SearchRegionContractError(
                    f"Oracle search order does not match candidate regions: {', '.join(err_parts)}"
                )

    # 2. SEEDED RANDOM SEARCH POLICY (Shuffles entire system region universe)
    elif eff_source == "random":
        assert seed is not None
        base = list(base_canonical_order)
        rng = random.Random(seed)
        rng.shuffle(base)
        order = tuple(base)
        effective_seed = seed
        source_label = "SEEDED_RANDOM_SYSTEM_SEARCH_POLICY"
        for rank, reg in enumerate(order):
            policy_trace_list.append(SearchPolicyTraceEntry(
                region_id=reg,
                final_rank=rank,
                provider_rank=None,
                origin="SEEDED_RANDOM",
            ))

    # 3. GT AUTO NORMAL SEARCH POLICY (Variant-independent system canonical search policy)
    elif mode == "gt" and eff_source == "auto":
        order = tuple(base_canonical_order)
        source_label = "GT_SYSTEM_SEARCH_POLICY"
        for rank, reg in enumerate(order):
            policy_trace_list.append(SearchPolicyTraceEntry(
                region_id=reg,
                final_rank=rank,
                provider_rank=None,
                origin="GT_SYSTEM",
            ))

    # 4. VLM PROVIDER-RANKED OR EXPLICIT PROVIDER WITH SYSTEM COMPLETION
    elif eff_source == "provider" or mode == "vlm":
        raw_ranking = spec_ranking if spec_ranking else spec_candidates
        # Validate that any proposed regions are known to the domain
        for reg in raw_ranking:
            if reg not in system_regions_set:
                raise SearchRegionContractError(
                    f"Unknown canonical search region {reg!r} for domain {eff_domain!r} (allowed: {sorted(system_regions_set)})"
                )
        if len(raw_ranking) != len(set(raw_ranking)):
            raise SearchRegionContractError(f"Search order has duplicate regions: {raw_ranking}")

        ranked_order: list[str] = []
        # Add provider-ranked regions
        for p_rank, reg in enumerate(raw_ranking):
            ranked_order.append(reg)
            policy_trace_list.append(SearchPolicyTraceEntry(
                region_id=reg,
                final_rank=len(ranked_order) - 1,
                provider_rank=p_rank,
                origin="PROVIDER_RANKED",
            ))

        # Deterministic system completion: append any omitted system search regions
        for reg in base_canonical_order:
            if reg not in ranked_order:
                ranked_order.append(reg)
                policy_trace_list.append(SearchPolicyTraceEntry(
                    region_id=reg,
                    final_rank=len(ranked_order) - 1,
                    provider_rank=None,
                    origin="SYSTEM_COMPLETION",
                ))

        order = tuple(ranked_order)
        source_label = "VLM_PROVIDER_RANKED_SYSTEM_COMPLETED" if mode == "vlm" else "GT_EXPLICIT_SEARCH_POLICY"

    else:
        raise SearchRegionContractError(f"Unhandled search-order source: {eff_source!r}")

    # Validate against allowed canonical domain vocabulary
    for reg in order:
        if reg not in system_regions_set:
            raise SearchRegionContractError(
                f"Unknown canonical search region {reg!r} for domain {eff_domain!r} (allowed: {sorted(system_regions_set)})"
            )

    # Validate duplicate canonical region IDs
    if len(order) != len(set(order)):
        raise SearchRegionContractError(f"Search order has duplicate regions: {order}")

    # Validate that complete system region universe is present
    if set(order) != system_regions_set:
        missing = system_regions_set - set(order)
        extra = set(order) - system_regions_set
        raise SearchRegionContractError(
            f"Search order does not contain complete domain search universe: missing={sorted(missing)}, extra={sorted(extra)}"
        )

    # Extract proposal trace from metadata if present
    proposal_trace_list: list[RegionProposalTraceEntry] = []
    metadata = getattr(specification, "metadata", {}) or {}
    trace = metadata.get("canonicalization_trace", {}) or {}
    raw_props = trace.get("region_proposal_trace") or metadata.get("region_proposal_trace")
    if raw_props and isinstance(raw_props, (list, tuple)):
        for item in raw_props:
            if isinstance(item, (dict, Mapping)):
                proposal_trace_list.append(RegionProposalTraceEntry.from_dict(item))
            elif isinstance(item, RegionProposalTraceEntry):
                proposal_trace_list.append(item)
    elif "proposal_trace" in trace and isinstance(trace["proposal_trace"], (list, tuple)):
        for item in trace["proposal_trace"]:
            if isinstance(item, (dict, Mapping)):
                proposal_trace_list.append(RegionProposalTraceEntry.from_dict(item))
            elif isinstance(item, RegionProposalTraceEntry):
                proposal_trace_list.append(item)

    return SearchRegionContract(
        domain=eff_domain,
        canonical_region_ids=order,
        source=source_label,
        policy_version=PHASE3_SEARCH_REGION_POLICY_VERSION,
        region_proposal_trace=tuple(proposal_trace_list),
        search_policy_trace=tuple(policy_trace_list),
        no_search_required=False,
        search_seed=effective_seed,
    )
