"""Comprehensive verification suite for Phase 3 P3-H.1 SearchRegionContract and Search Authority."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    SearchRegionContractError,
    UnmappedFunctionalConceptError,
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
    DOMAIN_CANONICAL_SEARCH_BASE_ORDERS,
    FIXED_SEARCH_ORDERS,
    ORACLE_SEARCH_ORDERS,
    PHASE3_SEARCH_REGION_POLICY_VERSION,
    RegionProposalTraceEntry,
    SearchPolicyTraceEntry,
    SearchRegionContract,
    freeze_search_region_contract,
    validate_search_order_preflight,
)
from mujoco_scenes.functional_tamp_pipeline.system_context_registry import (
    get_domain_search_regions,
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
# 1. GT AUTO SEARCH VS PRIVILEGED ORACLE TESTS (Defect 1: A1-A3)
# ==============================================================================

def test_01_gt_auto_never_uses_oracle():
    """GT auto search MUST use variant-independent system canonical search policy."""
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt", variant="W5")
    assert contract.source == "GT_SYSTEM_SEARCH_POLICY"
    assert contract.canonical_region_ids == DOMAIN_CANONICAL_SEARCH_BASE_ORDERS["workshop"]
    # Ensure it did NOT use W5 oracle ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER")
    assert contract.canonical_region_ids[0] == "LEFT_DRAWER"


def test_02_w5_gt_auto_starts_from_system_canonical_order():
    """W5 GT auto search must start from LEFT_DRAWER (canonical base order), NOT TOOL_CABINET."""
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt", variant="W5")
    assert contract.canonical_region_ids == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


def test_03_w7_gt_auto_does_not_use_right_drawer_first_because_of_variant():
    """W7 GT auto search must NOT start with RIGHT_DRAWER due to variant."""
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt", variant="W7")
    assert contract.canonical_region_ids == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


def test_04_k2_gt_auto_does_not_use_c2_first_because_of_variant():
    """K2 GT auto search must NOT start with C2 due to variant."""
    spec = _make_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2", "C2", "B1", "C1"),
        region_ranking=("D1", "D2", "C2", "B1", "C1"),
    )
    contract = freeze_search_region_contract(spec, domain="kitchen", source="auto", mode="gt", variant="K2")
    assert contract.source == "GT_SYSTEM_SEARCH_POLICY"
    assert contract.canonical_region_ids == ("D1", "D2", "C2", "B1", "C1")
    assert contract.canonical_region_ids[0] == "D1"


def test_05_explicit_oracle_still_yields_variant_specific_diagnostic_order():
    """Explicitly requesting source='oracle' on GT yields privileged diagnostic oracle order."""
    spec = _make_spec(domain="workshop")
    contract_w5 = freeze_search_region_contract(spec, domain="workshop", source="oracle", mode="gt", variant="W5")
    assert contract_w5.source == "PRIVILEGED_GT_ORACLE_DIAGNOSTIC"
    assert contract_w5.canonical_region_ids == ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER")

    contract_w7 = freeze_search_region_contract(spec, domain="workshop", source="oracle", mode="gt", variant="W7")
    assert contract_w7.source == "PRIVILEGED_GT_ORACLE_DIAGNOSTIC"
    assert contract_w7.canonical_region_ids == ("RIGHT_DRAWER", "TOOL_CABINET", "LEFT_DRAWER")


def test_06_vlm_mode_rejects_oracle():
    """VLM mode must reject privileged oracle search source with SearchRegionContractError."""
    spec = _make_spec(domain="workshop", source="VLM_CANONICAL_G_F")
    with pytest.raises(SearchRegionContractError, match="privileged"):
        freeze_search_region_contract(spec, domain="workshop", source="oracle", mode="vlm", variant="W1")


# ==============================================================================
# 2. SYSTEM SEARCH COMPLETENESS & DETERMINISTIC COMPLETION (Defect 2 & 3: B1-C4)
# ==============================================================================

def test_07_vlm_partial_ranking_gets_deterministic_system_completion():
    """VLM proposing only a subset of search regions has the remaining regions appended deterministically."""
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("TOOL_CABINET",),
        region_ranking=("TOOL_CABINET",),
        source="VLM_CANONICAL_G_F",
    )
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="vlm")
    assert contract.source == "VLM_PROVIDER_RANKED_SYSTEM_COMPLETED"
    assert contract.canonical_region_ids == ("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER")
    assert len(contract.search_policy_trace) == 3
    assert contract.search_policy_trace[0].origin == "PROVIDER_RANKED"
    assert contract.search_policy_trace[0].provider_rank == 0
    assert contract.search_policy_trace[1].origin == "SYSTEM_COMPLETION"
    assert contract.search_policy_trace[1].provider_rank is None
    assert contract.search_policy_trace[2].origin == "SYSTEM_COMPLETION"
    assert contract.search_policy_trace[2].provider_rank is None


def test_08_vlm_omission_cannot_erase_valid_physical_search_region():
    """VLM omitting all or part of search regions cannot prevent system search exhaustion."""
    spec = _make_spec(
        domain="kitchen",
        candidate_regions=("C2",),
        region_ranking=("C2",),
        source="VLM_CANONICAL_G_F",
    )
    contract = freeze_search_region_contract(spec, domain="kitchen", source="auto", mode="vlm")
    assert set(contract.canonical_region_ids) == get_domain_search_regions("kitchen")
    assert contract.canonical_region_ids == ("C2", "D1", "D2", "B1", "C1")


def test_09_random_uses_complete_system_universe():
    """Random search policy shuffles the complete system universe."""
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="random", mode="gt", seed=42)
    assert contract.source == "SEEDED_RANDOM_SYSTEM_SEARCH_POLICY"
    assert set(contract.canonical_region_ids) == get_domain_search_regions("workshop")
    assert contract.search_seed == 42


def test_10_system_context_registry_is_sole_universe_authority():
    """CANONICAL_SEARCH_REGIONS must match get_domain_search_regions without drift."""
    for domain in ("kitchen", "workshop", "living_room"):
        assert set(CANONICAL_SEARCH_REGIONS[domain]) == get_domain_search_regions(domain)


# ==============================================================================
# 3. REGION PROPOSAL RESOLUTION & FAIL-CLOSED CHECKS (Defect 4: D1-D4)
# ==============================================================================

def test_11_raw_kitchen_unknown_proposal_fails():
    """Kitchen VLM proposal with unmapped label/description fails closed."""
    from mujoco_scenes.kitchen_vlm_functional_graph import compile_vlm_functional_graph
    doc = {
        "status": "SUPPORTED",
        "task_summary": "make coffee",
        "cross_group_reuse_allowed": False,
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "stirrer",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "description": "spoon",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee_spoon"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "container",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "description": "mug",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee_cup"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "stirrer", "relation": "reaches bottom", "object_role": "container"}
        ],
        "interaction_groups": [
            {
                "id": "stir_group",
                "function": "coffee stirring",
                "tool_role": "stirrer",
                "target_role": "container",
                "required_target_count": 1,
                "required_relations": ["reaches bottom"],
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
            }
        ],
        "inspectable_regions": [
            {"id": "reg1", "label": "mysterious secret compartment", "visual_description": "hidden vault", "reason": "search"}
        ],
        "inspection_order": ["reg1"],
    }
    with pytest.raises(UnmappedFunctionalConceptError):
        compile_vlm_functional_graph(doc, task_instruction="make coffee")


def test_12_raw_workshop_unknown_proposal_fails():
    """Workshop VLM proposal with unmapped label/description fails closed."""
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    provider = FMRequirementProvider()
    doc = {
        "status": "SUPPORTED",
        "functional_roles": [
            {
                "id": "driver",
                "entity_kind": "OBJECT",
                "function": "turn screw",
                "description": "screwdriver",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "fastener",
                "entity_kind": "OBJECT",
                "function": "secure part",
                "description": "screw",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "driver", "relation": "compatible with fastener", "object_role": "fastener"}
        ],
        "interaction_groups": [],
        "inspectable_regions": [
            {"id": "r1", "label": "mysterious secret bin", "visual_description": "hidden cavity"}
        ],
        "inspection_order": ["r1"],
    }
    with pytest.raises(UnmappedFunctionalConceptError):
        provider.generate_canonical(task_instruction="drive screw", raw_document=doc)


def test_13_ambiguous_proposal_fails():
    """Region proposal matching multiple distinct physical regions fails closed."""
    from mujoco_scenes.kitchen_vlm_functional_graph import resolve_kitchen_region_proposal
    with pytest.raises(AmbiguousCanonicalizationError):
        resolve_kitchen_region_proposal({"label": "upper drawer lower drawer", "visual_description": "upper drawer and lower drawer"})


def test_14_duplicate_canonical_proposal_fails():
    """Two distinct raw proposals mapping to the same canonical search region fail closed."""
    from mujoco_scenes.kitchen_vlm_functional_graph import compile_vlm_functional_graph
    doc = {
        "status": "SUPPORTED",
        "task_summary": "make coffee",
        "cross_group_reuse_allowed": False,
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "stirrer",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "description": "spoon",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee_spoon"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "container",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "description": "mug",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee_cup"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "stirrer", "relation": "reaches bottom", "object_role": "container"}
        ],
        "interaction_groups": [
            {
                "id": "stir_group",
                "function": "coffee stirring",
                "tool_role": "stirrer",
                "target_role": "container",
                "required_target_count": 1,
                "required_relations": ["reaches bottom"],
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
            }
        ],
        "inspectable_regions": [
            {"id": "reg1", "label": "top drawer", "visual_description": "d1 drawer", "reason": "search"},
            {"id": "reg2", "label": "upper drawer", "visual_description": "d1 compartment", "reason": "search"},
        ],
        "inspection_order": ["reg1", "reg2"],
    }
    with pytest.raises(AmbiguousCanonicalizationError):
        compile_vlm_functional_graph(doc, task_instruction="make coffee")


def test_15_vlm_local_id_does_not_provide_semantic_resolution():
    """VLM local ID matching a canonical name (e.g. id='C2') does NOT resolve if label is unknown."""
    from mujoco_scenes.kitchen_vlm_functional_graph import resolve_kitchen_region_proposal
    # ID is C2, but label is unknown
    res = resolve_kitchen_region_proposal({"id": "C2", "label": "unknown box", "visual_description": "mystery"})
    assert res is None


# ==============================================================================
# 4. PROPOSAL & POLICY TRACE POPULATION (Defect 5: E1-E3)
# ==============================================================================

def test_16_production_proposal_trace_populated_in_kitchen():
    """Kitchen canonicalizer populates region_proposal_trace in canonicalization_trace."""
    from mujoco_scenes.kitchen_vlm_functional_graph import compile_vlm_functional_graph
    doc = {
        "status": "SUPPORTED",
        "task_summary": "make coffee",
        "cross_group_reuse_allowed": False,
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "stirrer",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "description": "spoon",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee_spoon"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "container",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "description": "mug",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee_cup"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "stirrer", "relation": "reaches bottom", "object_role": "container"}
        ],
        "interaction_groups": [
            {
                "id": "stir_group",
                "function": "coffee stirring",
                "tool_role": "stirrer",
                "target_role": "container",
                "required_target_count": 1,
                "required_relations": ["reaches bottom"],
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
            }
        ],
        "inspectable_regions": [
            {"id": "r_d1", "label": "top drawer", "visual_description": "d1", "reason": "search"}
        ],
        "inspection_order": ["r_d1"],
    }
    contract, vocab, trace = compile_vlm_functional_graph(doc, task_instruction="make coffee")
    assert "region_proposal_trace" in trace
    assert len(trace["region_proposal_trace"]) == 1
    assert trace["region_proposal_trace"][0]["canonical_region_id"] == "D1"


def test_17_production_proposal_trace_populated_in_workshop():
    """Workshop canonicalizer populates region_proposal_trace in canonicalization_trace."""
    from mujoco_scenes.workshop_phase1.requirements import FMRequirementProvider
    provider = FMRequirementProvider()
    doc = {
        "status": "SUPPORTED",
        "functional_roles": [
            {
                "id": "driver",
                "entity_kind": "OBJECT",
                "function": "turn screw",
                "description": "screwdriver",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "fastener",
                "entity_kind": "OBJECT",
                "function": "secure part",
                "description": "screw",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {"subject_role": "driver", "relation": "compatible with fastener", "object_role": "fastener"}
        ],
        "interaction_groups": [],
        "inspectable_regions": [
            {"id": "r_left", "label": "left drawer", "visual_description": "drawer on the left"}
        ],
        "inspection_order": ["r_left"],
    }
    provider.generate_canonical(task_instruction="drive screw", raw_document=doc)
    assert "region_proposal_trace" in provider.canonicalization_trace
    assert len(provider.canonicalization_trace["region_proposal_trace"]) == 1
    assert provider.canonicalization_trace["region_proposal_trace"][0]["canonical_region_id"] == "LEFT_DRAWER"


def test_18_final_search_policy_trace_shows_provider_ranked_vs_system_completed():
    """Search policy trace accurately distinguishes provider-ranked regions from system completion."""
    spec = _make_spec(
        domain="workshop",
        candidate_regions=("TOOL_CABINET",),
        region_ranking=("TOOL_CABINET",),
        source="VLM_CANONICAL_G_F",
    )
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="vlm")
    trace = contract.search_policy_trace
    assert len(trace) == 3
    assert trace[0].region_id == "TOOL_CABINET"
    assert trace[0].origin == "PROVIDER_RANKED"
    assert trace[0].provider_rank == 0
    assert trace[0].final_rank == 0

    assert trace[1].region_id == "LEFT_DRAWER"
    assert trace[1].origin == "SYSTEM_COMPLETION"
    assert trace[1].provider_rank is None
    assert trace[1].final_rank == 1

    assert trace[2].region_id == "RIGHT_DRAWER"
    assert trace[2].origin == "SYSTEM_COMPLETION"
    assert trace[2].provider_rank is None
    assert trace[2].final_rank == 2


# ==============================================================================
# 5. DEEP IMMUTABILITY & SERIALIZATION (Defect 6: F1-F3)
# ==============================================================================

def test_19_contract_deeply_immutable():
    """Attempting to mutate contract or its trace entries raises FrozenInstanceError."""
    entry = RegionProposalTraceEntry(
        raw_index=0,
        raw_id="r1",
        raw_label="top drawer",
        raw_visual_description="d1",
        canonical_region_id="D1",
        resolution_status="RESOLVED",
    )
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="GT_SYSTEM_SEARCH_POLICY",
        region_proposal_trace=(entry,),
    )
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        contract.region_proposal_trace[0].raw_label = "tampered"  # type: ignore[misc]

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        contract.domain = "workshop"  # type: ignore[misc]


def test_20_serialization_roundtrip_preserves_typed_trace():
    """to_dict() and from_dict() roundtrip cleanly preserving typed dataclass entries."""
    entry = RegionProposalTraceEntry(
        raw_index=0,
        raw_id="r1",
        raw_label="top drawer",
        raw_visual_description="d1",
        canonical_region_id="D1",
        resolution_status="RESOLVED",
        reason="Exact match",
    )
    policy_entry = SearchPolicyTraceEntry(
        region_id="D1",
        final_rank=0,
        provider_rank=0,
        origin="PROVIDER_RANKED",
    )
    contract = SearchRegionContract(
        domain="kitchen",
        canonical_region_ids=("D1", "D2", "C2", "B1", "C1"),
        source="VLM_PROVIDER_RANKED_SYSTEM_COMPLETED",
        region_proposal_trace=(entry,),
        search_policy_trace=(policy_entry,),
    )
    dict_repr = contract.to_dict()
    roundtrip = SearchRegionContract.from_dict(dict_repr)
    assert roundtrip == contract
    assert isinstance(roundtrip.region_proposal_trace[0], RegionProposalTraceEntry)
    assert isinstance(roundtrip.search_policy_trace[0], SearchPolicyTraceEntry)


def test_21_mutating_serialized_dict_does_not_mutate_contract():
    """Mutating the dictionary returned by to_dict() does not affect the contract."""
    contract = SearchRegionContract(
        domain="workshop",
        canonical_region_ids=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
        source="GT_SYSTEM_SEARCH_POLICY",
    )
    d = contract.to_dict()
    d["canonical_region_ids"].append("EXTRA")
    d["domain"] = "tampered"
    assert contract.canonical_region_ids == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")
    assert contract.domain == "workshop"


# ==============================================================================
# 6. RUNTIME CONTRACT AUTHORITY & EXECUTION (Defect 7 & 8: G1-G5, Defect 8)
# ==============================================================================

def test_22_runtime_search_requires_search_region_contract():
    """search_until_satisfied requires search_contract to be a SearchRegionContract."""
    class DummyDomain:
        def observe_initial(self): pass
        def evaluate_satisfaction(self, **kw): return SatisfactionResult(status="COMPLETE", complete=True)
        def open_region(self, r): return {}
        def observe_after_open(self, r): pass

    domain = DummyDomain()
    spec = _make_spec(domain="workshop")

    # Omitting search_contract raises TypeError (missing keyword-only arg)
    with pytest.raises(TypeError):
        search_until_satisfied(domain, spec)  # type: ignore[call-arg]

    # Passing invalid object raises SearchRegionContractError
    with pytest.raises(SearchRegionContractError):
        search_until_satisfied(domain, spec, search_contract="not_a_contract")  # type: ignore[arg-type]


def test_23_raw_tuple_cannot_bypass_canonical_runtime():
    """Passing a raw tuple as search_contract raises SearchRegionContractError."""
    class DummyDomain:
        def observe_initial(self): pass
        def evaluate_satisfaction(self, **kw): return SatisfactionResult(status="COMPLETE", complete=True)
        def open_region(self, r): return {}
        def observe_after_open(self, r): pass

    domain = DummyDomain()
    spec = _make_spec(domain="workshop")
    with pytest.raises(SearchRegionContractError):
        search_until_satisfied(domain, spec, search_contract=("LEFT_DRAWER",))  # type: ignore[arg-type]


def test_24_kitchen_receives_typed_contract(tmp_path: Path):
    """Kitchen run_to_plan accepts and respects SearchRegionContract."""
    from mujoco_scenes.functional_tamp_pipeline.domains.kitchen import run_to_plan
    spec = _make_spec(
        domain="kitchen",
        candidate_regions=("D1", "D2", "C2", "B1", "C1"),
        region_ranking=("D1", "D2", "C2", "B1", "C1"),
    )
    contract = freeze_search_region_contract(spec, domain="kitchen", source="auto", mode="gt")
    assert isinstance(contract, SearchRegionContract)


def test_25_workshop_receives_typed_contract():
    """Workshop runner passes SearchRegionContract directly into search_until_satisfied."""
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt")
    assert isinstance(contract, SearchRegionContract)
    assert contract.canonical_region_ids == ("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET")


def test_26_internal_typeerror_propagates_no_fallback_second_run():
    """Any internal TypeError in domain adapter must propagate without silent catch."""
    class FailingDomain:
        def observe_initial(self):
            raise TypeError("adapter internal bug")
        def evaluate_satisfaction(self, **kw): pass
        def open_region(self, r): pass
        def observe_after_open(self, r): pass

    domain = FailingDomain()
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt")

    with pytest.raises(TypeError, match="adapter internal bug"):
        search_until_satisfied(domain, spec, search_contract=contract)


def test_27_no_provider_vlm_recall_during_inspection():
    """Inspection execution never re-queries provider or modifies G_F."""
    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt")
    nodes_before = dict(spec.nodes)

    class DummyDomain:
        def __init__(self):
            self.inspected = []
        def observe_initial(self): pass
        def evaluate_satisfaction(self, search_exhausted=False):
            if "LEFT_DRAWER" in self.inspected:
                return SatisfactionResult(status="COMPLETE", complete=True, assignment={"driver": "d1", "fastener": "f1"})
            return SatisfactionResult(status="INCOMPLETE", complete=False)
        def open_region(self, r):
            self.inspected.append(r)
            return {"success": True}
        def observe_after_open(self, r): pass

    domain = DummyDomain()
    res, inspected = search_until_satisfied(domain, spec, search_contract=contract)
    assert res.complete is True
    assert spec.nodes == nodes_before


def test_28_identical_go_plus_different_search_order_identical_grounding():
    """Identical G_O produces identical grounding regardless of which search order was used."""
    go = ObservedSceneGraph()
    go.add_node(ObservedNode(
        instance_id="d1", canonical_category="screwdriver",
        geometry={"tool_tip": "phillips"},
    ))
    go.add_node(ObservedNode(
        instance_id="f1", canonical_category="screw",
        geometry={"screw_head": "phillips"},
    ))
    go.add_relation(ObservedRelation(
        subject_id="d1", predicate="COMPATIBLE_WITH", object_id="f1",
        status="TRUE", evidence={"match": True},
    ))

    spec1 = _make_spec(
        domain="workshop",
        region_ranking=("LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"),
    )
    spec2 = _make_spec(
        domain="workshop",
        region_ranking=("TOOL_CABINET", "LEFT_DRAWER", "RIGHT_DRAWER"),
    )

    res1 = ground_graph(spec1, go, {"search_exhausted": False})
    res2 = ground_graph(spec2, go, {"search_exhausted": False})

    assert res1.complete == res2.complete
    assert res1.assignment == res2.assignment
    assert res1.status == res2.status


# ==============================================================================
# 7. MANIFEST INTEGRITY & LIVING ROOM (Defect 9 & 10: I1-I3)
# ==============================================================================

def test_29_manifest_version_is_null_if_contract_freeze_fails(tmp_path: Path):
    """Manifest search_policy_version and search_contract are null if state.search_contract is None."""
    from mujoco_scenes.functional_tamp_pipeline.run import _RunState, _write_run_manifest

    state = _RunState(
        domain="workshop",
        variant="W1",
        internal_variant="workshop_test_v1",
        mode="gt",
        run_dir=tmp_path,
        search_contract=None,
    )
    _write_run_manifest(state)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["search_policy_version"] is None
    assert manifest["search_contract"] is None


def test_30_successful_manifest_records_exact_p3h1_policy_version(tmp_path: Path):
    """Manifest records exact P3-H.1 policy version when contract is frozen."""
    from mujoco_scenes.functional_tamp_pipeline.run import _RunState, _write_run_manifest

    spec = _make_spec(domain="workshop")
    contract = freeze_search_region_contract(spec, domain="workshop", source="auto", mode="gt")

    state = _RunState(
        domain="workshop",
        variant="W1",
        internal_variant="workshop_test_v1",
        mode="gt",
        run_dir=tmp_path,
        specification=spec,
        search_contract=contract,
        resolved_search_order=contract.canonical_region_ids,
        search_order_source_effective="gt_system",
    )
    _write_run_manifest(state)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["search_policy_version"] == "phase3_p3h_1_v1"
    assert manifest["search_contract"]["policy_version"] == "phase3_p3h_1_v1"
    assert manifest["search_contract"]["source"] == "GT_SYSTEM_SEARCH_POLICY"


def test_31_living_explicit_no_search_remains():
    """Living room produces explicit SYSTEM_DECLARED_NO_SEARCH contract with empty regions."""
    spec = _make_spec(domain="living_room", candidate_regions=(), region_ranking=())
    contract = freeze_search_region_contract(spec, domain="living_room", source="auto", mode="gt")
    assert contract.domain == "living_room"
    assert contract.canonical_region_ids == ()
    assert contract.no_search_required is True
    assert contract.source == "SYSTEM_DECLARED_NO_SEARCH"
    assert contract.policy_version == PHASE3_SEARCH_REGION_POLICY_VERSION
