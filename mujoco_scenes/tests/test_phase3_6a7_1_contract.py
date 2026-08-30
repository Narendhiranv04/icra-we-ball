"""Pass 3.6A.7.1 Contract Test Suite.

Verifies:
1. Living production-G_O integration & task-anchor categories alignment.
2. Domain-scoped unary checker capability enforcement.
3. Strict interaction-group schema validation.
4. Mode-safe task-interface completeness for Kitchen, Living Room, and Workshop.
5. Entity-kind invariant enforcement without type coercion.
6. Genuine provenance separation: raw_vlm_response vs validated_vlm_specification vs canonicalization_trace.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import pytest

from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    map_living_room_relation,
)
from mujoco_scenes.functional_tamp_pipeline.domains.living_room import (
    build_living_room_observed_scene_graph,
)
from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import (
    validate_canonical_task_interface,
    validate_runtime_gf,
)
from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import (
    evaluate_gf_against_reference,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLM_CANONICALIZATION_VERSION,
    VLMSpecProvider,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    validate_requirement_response,
)


class MockFMAdapter:
    def __init__(self, doc: dict[str, Any], raw_doc: dict[str, Any] | None = None):
        self.doc = doc
        self.raw_doc = raw_doc or doc
        self.last_observation_images: list[Any] = []
        self.last_raw_requirement_response = self.raw_doc
        self.last_raw_response = self.raw_doc
        self.last_raw_inspection_response: dict[str, Any] = {}
        self.metrics = MagicMock(total_calls=1)

    def generate_task_requirements(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.doc)


def _valid_living_vlm_doc() -> dict[str, Any]:
    return {
        "status": "SUPPORTED",
        "task_summary": "Two person tea serving",
        "functional_roles": [
            {
                "id": "cup_set",
                "entity_kind": "OBJECT",
                "function": "cup saucer set beverage payload",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup", "saucer"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "seat",
                "entity_kind": "FIXED_TARGET",
                "function": "seating position armchair",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["armchair"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "personal_surface",
                "entity_kind": "REGION",
                "function": "personal beverage support surface",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side_table"],
                "visible_candidates": [],
                "required_properties": ["planar support surface"],
            },
            {
                "id": "remote_obj",
                "entity_kind": "OBJECT",
                "function": "television remote control",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["remote_control"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "seat_pair",
                "entity_kind": "FIXED_TARGET",
                "function": "seating pair both seats",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["armchair"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "shared_surface",
                "entity_kind": "REGION",
                "function": "shared central remote placement table",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee_table"],
                "visible_candidates": [],
                "required_properties": ["planar support surface"],
            },
        ],
        "functional_relations": [
            {"subject_role": "personal_surface", "relation": "fits cup and saucer set", "object_role": "cup_set"},
            {"subject_role": "personal_surface", "relation": "near seat", "object_role": "seat"},
            {"subject_role": "shared_surface", "relation": "fit remote", "object_role": "remote_obj"},
            {"subject_role": "shared_surface", "relation": "accessible from both", "object_role": "seat_pair"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support_group",
                "function": "support drinkware",
                "tool_role": "personal_surface",
                "target_role": "cup_set",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["fits cup and saucer set"],
                "context_role": "seat",
                "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


def _make_fake_production_run(invalid_two_seat_pairing: bool = False) -> SimpleNamespace:
    region_registry = {
        "table_left": {
            "semantics": {"canonical_label": "side_table"},
            "geometry": {"PLANAR_SUPPORT": {"value": True, "status": "TRUE"}},
        },
        "table_right": {
            "semantics": {"canonical_label": "side_table"},
            "geometry": {"PLANAR_SUPPORT": {"value": True, "status": "TRUE"}},
        },
        "table_center": {
            "semantics": {"canonical_label": "coffee_table"},
            "geometry": {"PLANAR_SUPPORT": {"value": True, "status": "TRUE"}},
        },
    }
    seat_2_target = "armchair_left" if invalid_two_seat_pairing else "armchair_right"
    personal_rows = [
        {
            "region_id": "table_left",
            "slot_id": "slot_1",
            "seating_target_id": "armchair_left",
            "payload_ids": ["cup_1", "saucer_1"],
            "FITS_SET_ON": "TRUE",
            "NEAR_SEAT": "TRUE",
        },
        {
            "region_id": "table_right",
            "slot_id": "slot_2",
            "seating_target_id": seat_2_target,
            "payload_ids": ["cup_2", "saucer_2"],
            "FITS_SET_ON": "TRUE",
            "NEAR_SEAT": "TRUE",
        },
    ]
    seating_registry = {
        "armchair_left": {"geometry": {}},
        "armchair_right": {"geometry": {}},
    }
    shared_rows = [
        {
            "region_id": "table_center",
            "payload_ids": ["remote_1"],
            "FITS_ON": "TRUE",
            "ACCESSIBLE_FROM_BOTH_SEATS": "TRUE",
        }
    ]
    return SimpleNamespace(
        region_registry=region_registry,
        personal_rows=personal_rows,
        seating_registry=seating_registry,
        shared_rows=shared_rows,
    )


# ---------------------------------------------------------------------------
# Section 29: Living Production Compatibility
# ---------------------------------------------------------------------------
def test_a_living_production_go_positive_grounding():
    vlm_doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    fake_run = _make_fake_production_run(invalid_two_seat_pairing=False)
    go = build_living_room_observed_scene_graph(fake_run)

    ground_res = ground_graph(vlm_gf, go, {"search_exhausted": True})
    assert ground_res.satisfied is True
    assert ground_res.status == "COMPLETE"
    assert ground_res.complete is True


def test_b_living_production_go_invalid_pairing_rejected():
    vlm_doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    # Both personal tables bound to the SAME seat -> cannot satisfy distinct seating positions
    fake_run = _make_fake_production_run(invalid_two_seat_pairing=True)
    go = build_living_room_observed_scene_graph(fake_run)

    ground_res = ground_graph(vlm_gf, go, {"search_exhausted": True})
    assert ground_res.satisfied is False
    assert ground_res.status != "COMPLETE"


def test_c_task_anchor_semantic_categories_match_production_go():
    vlm_doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    # Verify task anchors include production G_O canonical categories
    assert "cup_saucer_set" in vlm_gf.nodes["CUP_SAUCER_SET"].semantic_categories
    assert "tv_remote" in vlm_gf.nodes["REMOTE"].semantic_categories
    assert "seating_position" in vlm_gf.nodes["SEATING_POSITION"].semantic_categories
    assert "seating_pair" in vlm_gf.nodes["SEATING_PAIR"].semantic_categories


def test_d_discoverable_support_region_categories_remain_vlm_derived():
    vlm_doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    # Discoverable region categories come from VLM output
    assert "side_table" in vlm_gf.nodes["PERSONAL_CUP_SAUCER_REGION"].semantic_categories
    assert "coffee_table" in vlm_gf.nodes["SHARED_REMOTE_REGION"].semantic_categories


# ---------------------------------------------------------------------------
# Section 30: Living Checker Scope
# ---------------------------------------------------------------------------
def test_e_living_planar_support_accepted():
    provider = EnvironmentVLMRequirementProvider("living_room")
    res = provider._map_properties(["planar support surface"])
    assert res == {"PLANAR_SUPPORT"}


def test_f_living_open_cavity_raises_unsupported_checker():
    provider = EnvironmentVLMRequirementProvider("living_room")
    with pytest.raises(UnsupportedCheckerCapabilityError) as exc_info:
        provider._map_properties(["open cavity"])
    assert "not supported by checkers in domain living_room" in str(exc_info.value)


def test_g_living_elongated_object_raises_unsupported_checker():
    provider = EnvironmentVLMRequirementProvider("living_room")
    with pytest.raises(UnsupportedCheckerCapabilityError) as exc_info:
        provider._map_properties(["elongated object"])
    assert "not supported by checkers in domain living_room" in str(exc_info.value)


def test_h_required_properties_near_seat_raises_unsupported_checker():
    provider = EnvironmentVLMRequirementProvider("living_room")
    with pytest.raises(UnsupportedCheckerCapabilityError):
        provider._map_properties(["near seat"])


def test_i_functional_relation_near_seat_maps_to_binary_relation():
    provider = EnvironmentVLMRequirementProvider("living_room")
    assert map_living_room_relation("near seat", provider.binary_relation_aliases) == "NEAR_SEAT"


# ---------------------------------------------------------------------------
# Section 31: Interaction Group Schema Validation
# ---------------------------------------------------------------------------
def test_j_interaction_group_missing_required_relations_fails():
    doc = _valid_living_vlm_doc()
    del doc["interaction_groups"][0]["required_relations"]
    with pytest.raises(FMResponseValidationError):
        validate_requirement_response(doc)


def test_k_interaction_group_empty_required_relations_fails():
    doc = _valid_living_vlm_doc()
    doc["interaction_groups"][0]["required_relations"] = []
    with pytest.raises(FMResponseValidationError):
        validate_requirement_response(doc)


def test_l_context_role_without_context_relations_fails():
    doc = _valid_living_vlm_doc()
    doc["interaction_groups"][0]["context_role"] = "seat"
    doc["interaction_groups"][0]["context_relations"] = []
    with pytest.raises(FMResponseValidationError):
        validate_requirement_response(doc)


def test_m_context_relations_without_context_role_fails():
    doc = _valid_living_vlm_doc()
    doc["interaction_groups"][0]["context_role"] = None
    doc["interaction_groups"][0]["context_relations"] = ["near seat"]
    with pytest.raises(FMResponseValidationError):
        validate_requirement_response(doc)


def test_n_non_string_relation_item_fails():
    doc = _valid_living_vlm_doc()
    doc["interaction_groups"][0]["required_relations"] = [123]
    with pytest.raises(FMResponseValidationError):
        validate_requirement_response(doc)


def test_o_valid_non_empty_interaction_group_accepted():
    doc = _valid_living_vlm_doc()
    validated = validate_requirement_response(doc)
    assert len(validated["interaction_groups"]) == 1
    grp = validated["interaction_groups"][0]
    assert grp["required_relations"] == ["fits cup and saucer set"]
    assert grp["context_role"] == "seat"
    assert grp["context_relations"] == ["near seat"]


# ---------------------------------------------------------------------------
# Section 32: Task Interface Completeness
# ---------------------------------------------------------------------------
def test_p_gt_kitchen_completeness_passes():
    gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    validate_canonical_task_interface(gf)


def test_q_gt_living_completeness_passes():
    gf = GTSpecProvider().provide("living_room", "serve tea for two")
    validate_canonical_task_interface(gf)


def test_r_gt_workshop_completeness_passes():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    validate_canonical_task_interface(gf)


def test_s_valid_synthetic_vlm_kitchen_passes():
    # Load GT kitchen graph to simulate a canonical valid kitchen G_F
    gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    vlm_gf = FunctionalRequirementGraph(
        domain="kitchen",
        task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes),
        relations=gf.relations,
        operation_groups=gf.operation_groups,
        source="VLM_CANONICAL_G_F",
    )
    validate_canonical_task_interface(vlm_gf)


def test_t_valid_synthetic_vlm_living_passes():
    doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)
    validate_canonical_task_interface(vlm_gf)


def test_u_valid_synthetic_vlm_workshop_passes():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    vlm_gf = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes),
        relations=gf.relations,
        source="VLM_CANONICAL_G_F",
    )
    validate_canonical_task_interface(vlm_gf)


def test_v_kitchen_graph_missing_required_role_fails():
    gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    nodes = dict(gf.nodes)
    del nodes["soup_eating_utensil"]
    bad_gf = FunctionalRequirementGraph(
        domain="kitchen", task_instruction=gf.task_instruction,
        nodes=nodes, relations=gf.relations, operation_groups=gf.operation_groups,
    )
    result = evaluate_gf_against_reference(bad_gf, gf)
    assert not result.structurally_complete
    assert "soup_eating_utensil" in result.missing_roles


def test_w_kitchen_graph_missing_operation_group_fails():
    gf = GTSpecProvider().provide("kitchen", "prepare coffee and soup")
    bad_gf = FunctionalRequirementGraph(
        domain="kitchen", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=gf.relations, operation_groups=(),
    )
    result = evaluate_gf_against_reference(bad_gf, gf)
    assert not result.structurally_complete
    assert len(result.missing_operation_groups) > 0


def test_x_living_graph_missing_cup_saucer_set_fails():
    doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    gf = spec_provider._living_room("serve tea for two", [], provider=provider)
    nodes = dict(gf.nodes)
    del nodes["CUP_SAUCER_SET"]
    bad_gf = FunctionalRequirementGraph(
        domain="living_room", task_instruction=gf.task_instruction,
        nodes=nodes, relations=gf.relations, operation_groups=gf.operation_groups,
    )
    ref_gf = GTSpecProvider().provide("living_room", "serve tea for two")
    result = evaluate_gf_against_reference(bad_gf, ref_gf)
    assert not result.structurally_complete
    assert "CUP_SAUCER_SET" in result.missing_roles


def test_y_living_graph_missing_personal_operation_group_fails():
    doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    gf = spec_provider._living_room("serve tea for two", [], provider=provider)
    bad_gf = FunctionalRequirementGraph(
        domain="living_room", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=gf.relations, operation_groups=(),
    )
    ref_gf = GTSpecProvider().provide("living_room", "serve tea for two")
    result = evaluate_gf_against_reference(bad_gf, ref_gf)
    assert not result.structurally_complete
    assert len(result.missing_operation_groups) > 0


def test_z_living_graph_missing_shared_relation_fails():
    doc = _valid_living_vlm_doc()
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    gf = spec_provider._living_room("serve tea for two", [], provider=provider)
    # Remove shared FITS_ON relation
    filtered_rels = tuple(r for r in gf.relations if r.predicate != "FITS_ON")
    bad_gf = FunctionalRequirementGraph(
        domain="living_room", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=filtered_rels, operation_groups=gf.operation_groups,
    )
    ref_gf = GTSpecProvider().provide("living_room", "serve tea for two")
    result = evaluate_gf_against_reference(bad_gf, ref_gf)
    assert not result.structurally_complete
    assert any(r[1] == "FITS_ON" for r in result.missing_relations)


def test_aa_workshop_graph_missing_repair_target_fails():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    nodes = dict(gf.nodes)
    del nodes["repair_target"]
    bad_gf = FunctionalRequirementGraph(
        domain="workshop", task_instruction=gf.task_instruction,
        nodes=nodes, relations=gf.relations,
    )
    result = evaluate_gf_against_reference(bad_gf, gf)
    assert not result.structurally_complete
    assert "repair_target" in result.missing_roles


def test_ab_workshop_graph_missing_compatible_with_fails():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    filtered_rels = tuple(r for r in gf.relations if r.predicate != "COMPATIBLE_WITH")
    bad_gf = FunctionalRequirementGraph(
        domain="workshop", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=filtered_rels,
    )
    result = evaluate_gf_against_reference(bad_gf, gf)
    assert not result.structurally_complete
    assert ("driver", "COMPATIBLE_WITH", "fastener", True) in result.missing_relations


def test_ac_workshop_graph_missing_reaches_target_fails():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    filtered_rels = tuple(r for r in gf.relations if r.predicate != "REACHES_TARGET")
    bad_gf = FunctionalRequirementGraph(
        domain="workshop", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=filtered_rels,
    )
    result = evaluate_gf_against_reference(bad_gf, gf)
    assert not result.structurally_complete
    assert ("driver", "REACHES_TARGET", "repair_target", True) in result.missing_relations


def test_ad_workshop_graph_missing_compatible_with_target_fails():
    gf = GTSpecProvider().provide("workshop", "fasten joint")
    filtered_rels = tuple(r for r in gf.relations if r.predicate != "COMPATIBLE_WITH_TARGET")
    bad_gf = FunctionalRequirementGraph(
        domain="workshop", task_instruction=gf.task_instruction,
        nodes=dict(gf.nodes), relations=filtered_rels,
    )
    result = evaluate_gf_against_reference(bad_gf, gf)
    assert not result.structurally_complete
    assert ("fastener", "COMPATIBLE_WITH_TARGET", "repair_target", True) in result.missing_relations


# ---------------------------------------------------------------------------
# Section 33: Entity Kind Invariants
# ---------------------------------------------------------------------------
def test_ae_living_support_role_as_object_fails():
    doc = _valid_living_vlm_doc()
    for r in doc["functional_roles"]:
        if r["id"] == "personal_surface":
            r["entity_kind"] = "OBJECT"
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    with pytest.raises(MalformedVLMSpecificationError) as exc_info:
        provider.generate_canonical(instruction="serve tea")
    assert "must have entity_kind 'REGION', got 'OBJECT'" in str(exc_info.value)


def test_af_living_payload_role_as_region_fails():
    doc = _valid_living_vlm_doc()
    for r in doc["functional_roles"]:
        if r["id"] == "cup_set":
            r["entity_kind"] = "REGION"
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    with pytest.raises(MalformedVLMSpecificationError) as exc_info:
        provider.generate_canonical(instruction="serve tea")
    assert "must have entity_kind 'OBJECT', got 'REGION'" in str(exc_info.value)


def test_ag_living_seating_role_as_object_fails():
    doc = _valid_living_vlm_doc()
    for r in doc["functional_roles"]:
        if r["id"] == "seat":
            r["entity_kind"] = "OBJECT"
    adapter = MockFMAdapter(doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    with pytest.raises(MalformedVLMSpecificationError) as exc_info:
        provider.generate_canonical(instruction="serve tea")
    assert "must have entity_kind 'FIXED_TARGET', got 'OBJECT'" in str(exc_info.value)


def test_ah_workshop_driver_wrong_entity_kind_fails():
    nodes = {
        "driver": FunctionalRole(name="driver", entity_kind="REGION", count=1, semantic_categories=("screwdriver",), binding_policy="DISTINCT"),
        "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1, semantic_categories=("screw",), binding_policy="DISTINCT"),
        "repair_target": FunctionalRole(name="repair_target", entity_kind="FIXED_TARGET", count=1, semantic_categories=("hole",), binding_policy="DISTINCT"),
    }
    rels = (
        FunctionalRelation(subject_role="driver", predicate="COMPATIBLE_WITH", object_role="fastener", expected=True),
        FunctionalRelation(subject_role="driver", predicate="REACHES_TARGET", object_role="repair_target", expected=True),
        FunctionalRelation(subject_role="fastener", predicate="COMPATIBLE_WITH_TARGET", object_role="repair_target", expected=True),
    )
    bad_gf = FunctionalRequirementGraph(domain="workshop", task_instruction="fasten", nodes=nodes, relations=rels)
    ref_gf = GTSpecProvider().provide("workshop", "fasten")
    result = evaluate_gf_against_reference(bad_gf, ref_gf)
    assert not result.structurally_complete
    assert "driver" in result.role_attribute_mismatches
    assert result.role_attribute_mismatches["driver"]["entity_kind"]["candidate"] == "REGION"


# ---------------------------------------------------------------------------
# Section 34: Provenance
# ---------------------------------------------------------------------------
def test_ai_to_am_provenance_separation():
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "  Two person tea serving with extra whitespace  ",
        "functional_roles": [
            {
                "id": "cup_set", "entity_kind": "OBJECT", "function": "cup saucer set beverage payload",
                "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": [" cup ", " saucer "],
                "visible_candidates": [], "required_properties": [],
            },
            {
                "id": "seat", "entity_kind": "FIXED_TARGET", "function": "seating position armchair",
                "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["armchair"],
                "visible_candidates": [], "required_properties": [],
            },
            {
                "id": "personal_surface", "entity_kind": "REGION", "function": "personal beverage support surface",
                "required_count": 2, "binding_policy": "DISTINCT", "candidate_categories": ["side_table"],
                "visible_candidates": [], "required_properties": ["planar support surface"],
            },
            {
                "id": "remote_obj", "entity_kind": "OBJECT", "function": "television remote control",
                "required_count": 1, "binding_policy": "DISTINCT", "candidate_categories": ["remote_control"],
                "visible_candidates": [], "required_properties": [],
            },
            {
                "id": "seat_pair", "entity_kind": "FIXED_TARGET", "function": "seating pair both seats",
                "required_count": 1, "binding_policy": "SHARED", "candidate_categories": ["armchair"],
                "visible_candidates": [], "required_properties": [],
            },
            {
                "id": "shared_surface", "entity_kind": "REGION", "function": "shared central remote placement table",
                "required_count": 1, "binding_policy": "SHARED", "candidate_categories": ["coffee_table"],
                "visible_candidates": [], "required_properties": ["planar support surface"],
            },
        ],
        "functional_relations": [
            {"subject_role": "personal_surface", "relation": "fits cup and saucer set", "object_role": "cup_set"},
            {"subject_role": "personal_surface", "relation": "near seat", "object_role": "seat"},
            {"subject_role": "shared_surface", "relation": "fit remote", "object_role": "remote_obj"},
            {"subject_role": "shared_surface", "relation": "accessible from both", "object_role": "seat_pair"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support_group", "function": "support drinkware", "tool_role": "personal_surface",
                "target_role": "cup_set", "required_target_count": 2, "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["fits cup and saucer set"], "context_role": "seat", "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [], "inspection_order": [], "unsupported_reason": "",
    }
    validated_doc = validate_requirement_response(raw_doc)
    assert raw_doc["task_summary"] != validated_doc["task_summary"]  # Strip cleaned

    adapter = MockFMAdapter(validated_doc, raw_doc=raw_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    spec_provider = VLMSpecProvider()
    vlm_gf = spec_provider._living_room("serve tea for two", [], provider=provider)

    # AI: raw_vlm_response separately preserved
    assert vlm_gf.metadata["raw_vlm_response"] == raw_doc
    # AJ: validated_vlm_specification separately preserved
    assert vlm_gf.metadata["validated_vlm_specification"] == validated_doc
    assert vlm_gf.metadata["raw_vlm_response"] != vlm_gf.metadata["validated_vlm_specification"]
    # AK: canonicalization_trace separately preserved
    assert "canonicalization_trace" in vlm_gf.metadata
    assert vlm_gf.metadata["canonicalization_trace"]["version"] == VLM_CANONICALIZATION_VERSION
    assert len(vlm_gf.metadata["canonicalization_trace"]["roles"]) == 6
    # AL: final G_F metadata exposes correct provenance
    assert vlm_gf.metadata["vlm_canonicalization_version"] == VLM_CANONICALIZATION_VERSION
    # AM: legacy raw_decomposition retained as legacy alias
    assert vlm_gf.metadata["raw_decomposition"] == validated_doc
