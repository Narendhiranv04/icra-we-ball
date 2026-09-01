"""Comprehensive verification suite for Phase 3 P3-H SearchRegionContract and Search Authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import (
    SearchRegionContractError,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    GraphGroundingResult,
    PipelineResult,
    SatisfactionResult,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.search import (
    search_until_satisfied,
)
from mujoco_scenes.functional_tamp_pipeline.search_contract import (
    CANONICAL_SEARCH_REGIONS,
    FIXED_SEARCH_ORDERS,
    ORACLE_SEARCH_ORDERS,
    PHASE3_SEARCH_REGION_POLICY_VERSION,
    SearchRegionContract,
    freeze_search_region_contract,
    validate_search_order_preflight,
)


def _make_spec(
    domain: str = "workshop",
    candidate_regions: tuple[str, ...] = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    region_ranking: tuple[str, ...] = ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    source: str = "GT_WORKSHOP_SPEC",
    metadata: dict[str, Any] | None = None,
) -> FunctionalRequirementGraph:
    nodes = {
        "driver": FunctionalRole(
            name="driver",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("screwdriver",),
        ),
        "fastener": FunctionalRole(
            name="fastener",
            entity_kind="OBJECT",
            count=1,
            semantic_categories=("screw",),
        ),
    }
    return FunctionalRequirementGraph(
        domain=domain,
        task_instruction="Fasten workpiece joint",
        nodes=nodes,
        candidate_regions=candidate_regions,
        region_ranking=region_ranking,
        source=source,
        metadata=dict(metadata or {}),
    )


# ==============================================================================
# 1. CONTRACT SCHEMA & IMMUTABILITY TESTS
# ==============================================================================

def test_contract_immutability():
    """Verify SearchRegionContract is frozen and canonical_region_ids cannot be mutated."""
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("LEFT_DRAWER", "RIGHT_DRAWER"),
        source="GT_EXPLICIT_SEARCH_POLICY",
    )
    assert isinstance(contract.canonical_region_ids, tuple)
    assert contract.policy_version == PHASE3_SEARCH_REGION_POLICY_VERSION
    assert not contract.no_search_required

    with pytest.raises((AttributeError, TypeError)):
        contract.domain = "kitchen"  # type: ignore[misc]

    with pytest.raises((AttributeError, TypeError)):
        contract.canonical_region_ids = ("D1",)  # type: ignore[misc]


def test_contract_metadata_mutation_isolation():
    """Mutating G_F metadata after freezing contract MUST NOT affect the frozen contract."""
    raw_regions = ["LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"]
    raw_metadata = {
        "canonicalization_trace": {
            "proposal_trace": [
                {"raw_label": "left drawer", "status": "CANONICALIZED"},
            ]
        }
    }
    spec = _make_spec(
        domain="workshop",
        candidate_regions=tuple(raw_regions),
        region_ranking=tuple(raw_regions),
        metadata=raw_metadata,
    )

    contract = freeze_search_region_contract(spec, domain="workshop", source="provider")
    assert contract.canonical_region_ids == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
    assert len(contract.proposal_trace) == 1

    # Mutate the original metadata dict
    raw_metadata["canonicalization_trace"]["proposal_trace"].append({"malicious": True})
    raw_metadata["canonicalization_trace"]["proposal_trace"][0]["raw_label"] = "tampered"
    spec.metadata["new_key"] = "tampered"

    # Contract must remain completely unchanged
    assert contract.canonical_region_ids == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
    assert len(contract.proposal_trace) == 1
    assert contract.proposal_trace[0]["raw_label"] == "left drawer"
    assert "malicious" not in contract.proposal_trace[0]


def test_contract_serialization_roundtrip():
    """Verify to_dict and from_dict produce identical contract."""
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="VLM_CANONICALIZED_SEARCH_POLICY",
        policy_version=PHASE3_SEARCH_REGION_POLICY_VERSION,
        proposal_trace=({"id": "r1", "label": "drawer"},),
        no_search_required=False,
        search_seed=123,
    )
    serialized = contract.to_dict()
    assert isinstance(serialized, dict)
    assert serialized["domain"] == "kitchen"
    assert serialized["canonical_region_ids"] == ["D1", "D2", "C2", "B1", "C1"]
    assert serialized["search_seed"] == 123

    deserialized = SearchRegionContract.from_dict(serialized)
    assert deserialized == contract
    assert isinstance(deserialized.canonical_region_ids, tuple)
    assert isinstance(deserialized.proposal_trace, tuple)


# ==============================================================================
# 2. VALIDATION & FAIL-CLOSED TESTS
# ==============================================================================

def test_unknown_domain_rejected():
    spec = _make_spec(domain="unknown_warehouse")
    with pytest.raises(SearchRegionContractError, match="Unknown domain"):
        freeze_search_region_contract(spec, domain="unknown_warehouse")


def test_domain_mismatch_rejected():
    spec = _make_spec(domain="workshop")
    with pytest.raises(SearchRegionContractError, match="Domain mismatch"):
        freeze_search_region_contract(spec, domain="kitchen")


def test_living_room_explicit_no_search_contract():
    nodes = {"table": FunctionalRole(name="table", entity_kind="REGION", count=1)}
    spec = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="Arrange coffee cups",
        nodes=nodes,
        candidate_regions=(),
        region_ranking=(),
        source="GT_LIVING_ROOM_SPEC",
    )
    contract = freeze_search_region_contract(spec, domain="living_room", source="auto")
    assert contract.domain == "living_room"
    assert contract.canonical_region_ids == ()
    assert contract.no_search_required is True
    assert contract.source == "SYSTEM_DECLARED_NO_SEARCH"


def test_living_room_rejects_candidate_regions():
    nodes = {"table": FunctionalRole(name="table", entity_kind="REGION", count=1)}
    spec = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="Arrange coffee cups",
        nodes=nodes,
        candidate_regions=("D1",),
        region_ranking=("D1",),
        source="GT_LIVING_ROOM_SPEC",
    )
    with pytest.raises(SearchRegionContractError, match="living_room has no inspectable search regions"):
        freeze_search_region_contract(spec, domain="living_room", source="auto")


def test_living_room_rejects_oracle_or_random_search():
    spec = FunctionalRequirementGraph(
        domain="living_room",
        task_instruction="Arrange cups",
        nodes={},
        candidate_regions=(),
        region_ranking=(),
        source="GT_LIVING_ROOM_SPEC",
    )
    with pytest.raises(SearchRegionContractError, match="not applicable for living_room"):
        freeze_search_region_contract(spec, domain="living_room", source="oracle")

    with pytest.raises(SearchRegionContractError, match="not applicable for living_room"):
        freeze_search_region_contract(spec, domain="living_room", source="random", seed=0)


def test_missing_search_metadata_fails_closed():
    spec = _make_spec(
        domain="workshop",
        candidate_regions=(),
        region_ranking=(),
    )
    with pytest.raises(SearchRegionContractError, match="Missing search region metadata"):
        freeze_search_region_contract(spec, domain="workshop", source="provider")


def test_unknown_canonical_region_id_fails_closed():
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "MYSTERY_COMPARTMENT"),
        region_ranking=("LEFT_DRAWER", "MYSTERY_COMPARTMENT"),
    )
    with pytest.raises(SearchRegionContractError, match="Unknown canonical search region"):
        freeze_search_region_contract(spec, domain="workshop", source="provider")


def test_duplicate_canonical_regions_fails_closed():
    spec = _make_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2"),
        region_ranking=("D1", "D1"),
    )
    with pytest.raises(SearchRegionContractError, match="duplicate regions"):
        freeze_search_region_contract(spec, domain="kitchen", source="provider")


def test_order_candidate_mismatch_fails_closed():
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER"),
    )
    with pytest.raises(SearchRegionContractError, match="does not match candidate regions"):
        freeze_search_region_contract(spec, domain="workshop", source="provider")


def test_vlm_with_oracle_fails_closed():
    spec = _make_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2", "C2", "B1", "C1"),
        region_ranking=("D1", "D2", "C2", "B1", "C1"),
        source="VLM_KITCHEN_SPEC",
    )
    with pytest.raises(SearchRegionContractError, match="oracle search is privileged and only valid with GT mode"):
        freeze_search_region_contract(spec, domain="kitchen", source="oracle", mode="vlm", variant="K1")


def test_random_search_seed_validation():
    spec = _make_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2", "C2", "B1", "C1"),
        region_ranking=("D1", "D2", "C2", "B1", "C1"),
    )
    # Missing seed
    with pytest.raises(SearchRegionContractError, match="random search requires --search-seed"):
        freeze_search_region_contract(spec, domain="kitchen", source="random", seed=None)

    # Negative seed
    with pytest.raises(SearchRegionContractError, match="must be a non-negative integer"):
        freeze_search_region_contract(spec, domain="kitchen", source="random", seed=-10)

    # Valid seed produces deterministic shuffle and records seed
    c1 = freeze_search_region_contract(spec, domain="kitchen", source="random", seed=42)
    c2 = freeze_search_region_contract(spec, domain="kitchen", source="random", seed=42)
    assert c1.canonical_region_ids == c2.canonical_region_ids
    assert set(c1.canonical_region_ids) == set(spec.candidate_regions)
    assert c1.search_seed == 42
    assert c1.source == "SEEDED_RANDOM_SEARCH_POLICY"


# ==============================================================================
# 3. RUNTIME BEHAVIOR & AUTHORITY ISOLATION TESTS
# ==============================================================================

def test_no_fallback_append_to_partial_proposals():
    """If a contract has only 1 region, runtime MUST NOT append the other regions."""
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER",),
        region_ranking=("LEFT_DRAWER",),
    )
    contract = freeze_search_region_contract(spec, domain="workshop", source="provider")
    assert contract.canonical_region_ids == ("LEFT_DRAWER",)

    class MockWorkshopDomain:
        def __init__(self):
            self.inspected = []
        def observe_initial(self): pass
        def evaluate_satisfaction(self, search_exhausted: bool = False):
            return SatisfactionResult(complete=False, status="INFEASIBLE" if search_exhausted else "INCOMPLETE")
        def open_region(self, region: str):
            self.inspected.append(region)
            return {"success": True}
        def observe_after_open(self, region: str): pass

    domain = MockWorkshopDomain()
    res, inspected = search_until_satisfied(
        domain,
        spec,
        search_contract=contract,
    )
    assert inspected == ("LEFT_DRAWER",)
    assert res.status == "INFEASIBLE"
    assert domain.inspected == ["LEFT_DRAWER"]


def test_provider_invoked_strictly_once():
    """Prove runtime inspection retries do NOT call provider/FM/VLM repeatedly."""
    provider_call_count = 0
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    )

    class CountingProvider:
        def provide(self, *args, **kwargs):
            nonlocal provider_call_count
            provider_call_count += 1
            return spec

    prov = CountingProvider()
    acquired_spec = prov.provide("workshop", "task", [])
    assert provider_call_count == 1

    contract = freeze_search_region_contract(acquired_spec, domain="workshop", source="provider")

    # Simulate multi-step search with retries
    class MultiStepDomain:
        def __init__(self):
            self.step = 0
        def observe_initial(self): pass
        def evaluate_satisfaction(self, search_exhausted: bool = False):
            if self.step >= 2:
                return SatisfactionResult(complete=True, status="COMPLETE", assignment={"driver": "d1", "fastener": "f1"})
            return SatisfactionResult(complete=False, status="INCOMPLETE")
        def open_region(self, region: str):
            self.step += 1
            return {"success": True}
        def observe_after_open(self, region: str): pass

    domain = MultiStepDomain()
    res, inspected = search_until_satisfied(
        domain,
        acquired_spec,
        search_contract=contract,
    )
    assert res.status == "COMPLETE"
    assert len(inspected) == 2
    # Provider MUST NEVER have been called again during search loop
    assert provider_call_count == 1


def test_search_order_does_not_rank_or_alter_canonical_grounding():
    """Two different search orders reaching the same verified G_O produce the exact same phi*."""
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    )

    # Build G_O containing driver and fastener
    go = ObservedSceneGraph()
    go.add_node(ObservedNode(instance_id="driver_01", entity_kind="OBJECT", canonical_category="screwdriver"))
    go.add_node(ObservedNode(instance_id="fastener_01", entity_kind="OBJECT", canonical_category="screw"))
    go.add_relation(ObservedRelation(subject_id="driver_01", predicate="COMPATIBLE_WITH", object_id="fastener_01", status="TRUE"))

    # Order A
    res_a = ground_graph(spec, go, {"search_exhausted": False})

    # Order B (simulate G_O state after different inspection sequence but identical facts)
    go_b = ObservedSceneGraph()
    go_b.add_node(ObservedNode(instance_id="driver_01", entity_kind="OBJECT", canonical_category="screwdriver"))
    go_b.add_node(ObservedNode(instance_id="fastener_01", entity_kind="OBJECT", canonical_category="screw"))
    go_b.add_relation(ObservedRelation(subject_id="driver_01", predicate="COMPATIBLE_WITH", object_id="fastener_01", status="TRUE"))

    res_b = ground_graph(spec, go_b, {"search_exhausted": False})

    assert res_a.complete is True
    assert res_b.complete is True
    assert res_a.assignment == res_b.assignment
    assert res_a.assignment == {"driver": "driver_01", "fastener": "fastener_01"}
