"""Tests for Living Room VLM Canonicalization Compiler (Pass P3-F).

Verifies fail-closed deterministic natural language compilation, composite cup-saucer
semantics, role semantic authority, property validation, direction normalization,
and concept accounting trace for the Living Room domain.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    LIVING_ROOM_VLM_CANONICALIZATION_VERSION,
    canonicalize_living_room_relation,
    map_living_room_fixed_target_role,
    map_living_room_object_payload_role,
    map_living_room_operation_group_function,
    map_living_room_relation,
    map_living_room_role_function,
)
from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
    VLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.task_interface_validator import validate_runtime_gf
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider


def create_ideal_living_room_doc() -> dict[str, Any]:
    """Returns a valid 6-role ideal Living Room VLM specification document."""
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare living room seating surfaces for drinks and remote.",
        "functional_roles": [
            {
                "id": "role_1",
                "entity_kind": "REGION",
                "function": "personal cup and saucer support",
                "description": "fixed individual side table surface beside each viewer seating position for supporting drinkware",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table", "end table"],
                "visible_candidates": [
                    {"label": "side table", "visual_description": "table next to armchair 1"},
                    {"label": "end table", "visual_description": "table next to armchair 2"},
                ],
                "required_properties": ["planar horizontal support"],
            },
            {
                "id": "role_2",
                "entity_kind": "REGION",
                "function": "shared remote support",
                "description": "fixed central coffee table surface accessible to both seated viewers for supporting the shared tv remote",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee table"],
                "visible_candidates": [
                    {"label": "coffee table", "visual_description": "central low coffee table"},
                ],
                "required_properties": ["planar horizontal support"],
            },
            {
                "id": "role_3",
                "entity_kind": "OBJECT",
                "function": "contain hot beverage and saucer",
                "description": "individual cup and saucer drinkware set for each person",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup", "saucer", "drinkware"],
                "visible_candidates": [
                    {"label": "cup", "visual_description": "ceramic mug on tray"},
                    {"label": "saucer", "visual_description": "matching small saucer plate"},
                ],
                "required_properties": [],
            },
            {
                "id": "role_4",
                "entity_kind": "OBJECT",
                "function": "control television",
                "description": "handheld television remote control device",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["remote control", "tv remote"],
                "visible_candidates": [
                    {"label": "remote control", "visual_description": "black remote control device on coffee table"},
                ],
                "required_properties": [],
            },
            {
                "id": "role_5",
                "entity_kind": "FIXED_TARGET",
                "function": "viewer seating position",
                "description": "individual seated viewer location for personal drink access",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["armchair", "chair"],
                "visible_candidates": [
                    {"label": "armchair", "visual_description": "viewer armchair 1"},
                    {"label": "armchair", "visual_description": "viewer armchair 2"},
                ],
                "required_properties": [],
            },
            {
                "id": "role_6",
                "entity_kind": "FIXED_TARGET",
                "function": "paired viewer seating area",
                "description": "both viewer seating positions collectively for shared item accessibility",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["armchairs", "seating area"],
                "visible_candidates": [
                    {"label": "armchairs", "visual_description": "both armchairs together"},
                ],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "role_1",
                "relation": "can hold drinkware set",
                "object_role": "role_3",
            },
            {
                "subject_role": "role_1",
                "relation": "near seat",
                "object_role": "role_5",
            },
            {
                "subject_role": "role_2",
                "relation": "can hold remote",
                "object_role": "role_4",
            },
            {
                "subject_role": "role_2",
                "relation": "accessible from both seats",
                "object_role": "role_6",
            },
        ],
        "interaction_groups": [
            {
                "id": "group_1",
                "function": "support drinkware set beside seat",
                "tool_role": "role_1",
                "target_role": "role_3",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["can hold drinkware set"],
                "context_role": "role_5",
                "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }


def test_ideal_living_room_canonicalization():
    """Verify ideal Living Room document compiles to a valid executable G_F."""
    doc = create_ideal_living_room_doc()
    provider = EnvironmentVLMRequirementProvider("living_room")
    res = provider.generate_canonical(
        "Prepare living room for two people",
        observation_images=[Path("/tmp/obs.png")],
        raw_document=doc,
    )
    assert res["ready_for_grounding"] is True
    assert len(res["normalized_requirements"]) == 6

    # Verify canonical trace and version
    trace = res["canonicalization_trace"]
    assert trace["version"] == LIVING_ROOM_VLM_CANONICALIZATION_VERSION
    assert trace["vlm_canonicalization_version"] == "phase3_p3f_v1"

    # Build G_F and validate runtime executable constraints
    gf = VLMSpecProvider._living_room(
        "Prepare living room for two people",
        [Path("/tmp/obs.png")],
        provider=provider,
    )
    gf.validate()
    validate_runtime_gf(gf)

    assert set(gf.nodes.keys()) == {
        "PERSONAL_CUP_SAUCER_REGION",
        "SHARED_REMOTE_REGION",
        "CUP_SAUCER_SET",
        "REMOTE",
        "SEATING_POSITION",
        "SEATING_PAIR",
    }
    assert gf.nodes["CUP_SAUCER_SET"].count == 2
    assert gf.nodes["PERSONAL_CUP_SAUCER_REGION"].count == 2
    assert gf.nodes["SHARED_REMOTE_REGION"].count == 1
    assert gf.nodes["REMOTE"].count == 1
    assert gf.nodes["SEATING_POSITION"].count == 2
    assert gf.nodes["SEATING_PAIR"].count == 1

    # Operation group relations are represented inside OperationGroup, not duplicated at top-level
    assert len(gf.relations) == 2
    assert any(
        r.subject_role == "SHARED_REMOTE_REGION"
        and r.predicate == "FITS_ON"
        and r.object_role == "REMOTE"
        for r in gf.relations
    )
    assert any(
        r.subject_role == "SHARED_REMOTE_REGION"
        and r.predicate == "ACCESSIBLE_FROM_BOTH_SEATS"
        and r.object_role == "SEATING_PAIR"
        for r in gf.relations
    )

    assert len(gf.operation_groups) == 1
    og = gf.operation_groups[0]
    assert og.id == "personal_support_group"
    assert og.tool_role == "PERSONAL_CUP_SAUCER_REGION"
    assert og.target_role == "CUP_SAUCER_SET"
    assert og.required_relations == ("FITS_SET_ON",)
    assert og.context_role == "SEATING_POSITION"
    assert og.context_relations == ("NEAR_SEAT",)
    assert og.required_target_count == 2


def test_composite_cup_saucer_cardinality_cases():
    """Verify composite cup-saucer semantics across Cases A to F."""
    # Case A: 1 raw role for cup+saucer set with count 2 -> canonical count = 2 (PRESERVED)
    doc_a = create_ideal_living_room_doc()
    provider = EnvironmentVLMRequirementProvider("living_room")
    res_a = provider.generate_canonical(raw_document=doc_a)
    cup_saucer_rec = next(r for r in res_a["normalized_requirements"] if r["function"] == "CUP_SAUCER_SET")
    assert cup_saucer_rec["vlm_required_count"] == 2
    assert res_a["canonicalization_trace"]["concept_accounting"]["roles"]["role_3"]["status"] == "PRESERVED"

    # Case B: 2 component roles (cup count 2 + saucer count 2) -> canonical count = 2 (COMPOSED_FROM_COMPONENT_ROLES)
    doc_b = deepcopy(doc_a)
    doc_b["functional_roles"] = [r for r in doc_b["functional_roles"] if r["id"] != "role_3"]
    doc_b["functional_roles"].extend([
        {
            "id": "cup_comp",
            "entity_kind": "OBJECT",
            "function": "contain hot beverage cup",
            "description": "individual hot beverage cups for each person",
            "required_count": 2,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["cup", "mug"],
            "visible_candidates": [],
            "required_properties": [],
        },
        {
            "id": "saucer_comp",
            "entity_kind": "OBJECT",
            "function": "saucer plate for cup",
            "description": "matching saucer plates for each cup",
            "required_count": 2,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["saucer"],
            "visible_candidates": [],
            "required_properties": [],
        },
    ])
    # Update relations/group to reference cup_comp
    for rel in doc_b["functional_relations"]:
        if rel["object_role"] == "role_3":
            rel["object_role"] = "cup_comp"
    doc_b["interaction_groups"][0]["target_role"] = "cup_comp"

    res_b = provider.generate_canonical(raw_document=doc_b)
    cup_saucer_b = next(r for r in res_b["normalized_requirements"] if r["function"] == "CUP_SAUCER_SET")
    assert cup_saucer_b["vlm_required_count"] == 2  # NOT 4!
    acct_b = res_b["canonicalization_trace"]["concept_accounting"]["roles"]
    assert acct_b["cup_comp"]["status"] == "COMPOSED_FROM_COMPONENT_ROLES"
    assert acct_b["saucer_comp"]["status"] == "COMPOSED_FROM_COMPONENT_ROLES"
    assert acct_b["cup_comp"]["canonical_count"] == 2

    # Case C: Mismatched counts (cup count 2 vs saucer count 1) -> MalformedVLMSpecificationError
    doc_c = deepcopy(doc_b)
    next(r for r in doc_c["functional_roles"] if r["id"] == "saucer_comp")["required_count"] = 1
    with pytest.raises(MalformedVLMSpecificationError, match="Mismatched component counts"):
        provider.generate_canonical(raw_document=doc_c)

    # Case D: Multiple ambiguous cup roles (2 cup roles + 1 saucer role) -> AmbiguousCanonicalizationError
    doc_d = deepcopy(doc_b)
    doc_d["functional_roles"].append({
        "id": "cup_comp_extra",
        "entity_kind": "OBJECT",
        "function": "drinking cup",
        "description": "extra cup",
        "required_count": 2,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["cup"],
        "visible_candidates": [],
        "required_properties": [],
    })
    with pytest.raises(AmbiguousCanonicalizationError, match="exactly one cup role and one saucer role"):
        provider.generate_canonical(raw_document=doc_d)

    # Case E: Missing component (only cup without saucer) -> AmbiguousCanonicalizationError
    doc_e = deepcopy(doc_b)
    doc_e["functional_roles"] = [r for r in doc_e["functional_roles"] if r["id"] != "saucer_comp"]
    with pytest.raises(AmbiguousCanonicalizationError, match="exactly one cup role and one saucer role"):
        provider.generate_canonical(raw_document=doc_e)

    # Case F: Conflicting binding policies (cup DISTINCT vs saucer SHARED) -> MalformedVLMSpecificationError
    doc_f = deepcopy(doc_b)
    next(r for r in doc_f["functional_roles"] if r["id"] == "saucer_comp")["binding_policy"] = "SHARED"
    with pytest.raises(MalformedVLMSpecificationError, match="Conflicting binding policies"):
        provider.generate_canonical(raw_document=doc_f)


def test_role_authority_function_and_description_only():
    """Verify that candidate categories NEVER manufacture roles or alter role resolution."""
    doc = create_ideal_living_room_doc()
    # Inject contradictory candidate categories
    doc["functional_roles"][2]["candidate_categories"] = ["hammer", "screwdriver", "pliers"]
    provider = EnvironmentVLMRequirementProvider("living_room")
    res = provider.generate_canonical(raw_document=doc)
    cup_rec = next(r for r in res["normalized_requirements"] if r["function"] == "CUP_SAUCER_SET")
    assert cup_rec["function"] == "CUP_SAUCER_SET"
    role_acct = res["canonicalization_trace"]["concept_accounting"]["roles"]["role_3"]
    assert role_acct["role_semantic_source"] == "FUNCTION_AND_DESCRIPTION"
    assert role_acct["candidate_categories_used_for_role_identity"] is False

    # Nonsense function fails closed
    doc_bad = create_ideal_living_room_doc()
    doc_bad["functional_roles"][2]["function"] = "paint the living room wall"
    doc_bad["functional_roles"][2]["description"] = "wall painting task"
    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped"):
        provider.generate_canonical(raw_document=doc_bad)


def test_required_properties_fail_closed():
    """Verify fail-closed handling for unary properties in Living Room."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # 1. PLANAR_SUPPORT on REGION -> PRESERVED
    doc = create_ideal_living_room_doc()
    res = provider.generate_canonical(raw_document=doc)
    prop_acct = res["canonicalization_trace"]["concept_accounting"]["properties"]
    assert any(p["canonical_predicate"] == "PLANAR_SUPPORT" and p["status"] == "PRESERVED" for p in prop_acct)

    # 2. Duplicate PLANAR_SUPPORT on same role -> MERGED_BY_EXPLICIT_RULE
    doc_dup = create_ideal_living_room_doc()
    doc_dup["functional_roles"][0]["required_properties"] = [
        "planar horizontal support",
        "horizontal planar support",
    ]
    res_dup = provider.generate_canonical(raw_document=doc_dup)
    dup_acct = res_dup["canonicalization_trace"]["concept_accounting"]["properties"]
    assert any(p["status"] == "MERGED_BY_EXPLICIT_RULE" for p in dup_acct)

    # 3. PLANAR_SUPPORT on OBJECT -> MalformedVLMSpecificationError
    doc_obj_prop = create_ideal_living_room_doc()
    doc_obj_prop["functional_roles"][2]["required_properties"] = ["planar support"]
    with pytest.raises(MalformedVLMSpecificationError, match="PLANAR_SUPPORT requested on non-REGION role"):
        provider.generate_canonical(raw_document=doc_obj_prop)

    # 4. Unsupported checker properties (OPEN_CAVITY / ELONGATED_OBJECT) -> UnsupportedCheckerCapabilityError
    doc_unsupported = create_ideal_living_room_doc()
    doc_unsupported["functional_roles"][2]["required_properties"] = ["open cavity"]
    with pytest.raises(UnsupportedCheckerCapabilityError, match="is not supported in Living Room domain"):
        provider.generate_canonical(raw_document=doc_unsupported)

    # 5. Unknown property -> UnmappedFunctionalConceptError
    doc_unknown = create_ideal_living_room_doc()
    doc_unknown["functional_roles"][0]["required_properties"] = ["magnetic levitation"]
    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped"):
        provider.generate_canonical(raw_document=doc_unknown)


def test_relation_direction_normalization_and_fail_closed():
    """Verify direction normalization for reverse placement phrasing and fail-closed behavior."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # Reverse relation: SET -- "placed on" --> REGION -> NORMALIZED_TO_CANONICAL_SIGNATURE
    doc_rev = create_ideal_living_room_doc()
    doc_rev["functional_relations"][0] = {
        "subject_role": "role_3",  # CUP_SAUCER_SET
        "relation": "placed on",
        "object_role": "role_1",   # PERSONAL_CUP_SAUCER_REGION
    }
    # Update group required relation to forward phrasing so group succeeds
    doc_rev["interaction_groups"][0]["required_relations"] = ["can hold drinkware set"]

    res_rev = provider.generate_canonical(raw_document=doc_rev)
    rel_acct = res_rev["canonicalization_trace"]["concept_accounting"]["relations"]
    norm_entry = next(r for r in rel_acct if r["raw_subject_role_id"] == "role_3" and r["raw_relation_text"] == "placed on")
    assert norm_entry["canonical_subject_role_id"] == "PERSONAL_CUP_SAUCER_REGION"
    assert norm_entry["canonical_predicate"] == "FITS_SET_ON"
    assert norm_entry["canonical_object_role_id"] == "CUP_SAUCER_SET"
    assert norm_entry["direction_status"] == "NORMALIZED_TO_CANONICAL_SIGNATURE"

    # Incompatible endpoints: SET -- "near seat" --> SEATING_POSITION -> MalformedVLMSpecificationError
    with pytest.raises(MalformedVLMSpecificationError, match="expects endpoints"):
        canonicalize_living_room_relation("near seat", "CUP_SAUCER_SET", "SEATING_POSITION")

    # Generic fragment alone: "on" -> UnmappedFunctionalConceptError
    with pytest.raises(UnmappedFunctionalConceptError, match="Generic relation fragment"):
        canonicalize_living_room_relation("on", "PERSONAL_CUP_SAUCER_REGION", "CUP_SAUCER_SET")


def test_operation_group_validation():
    """Verify fail-closed validation on Living Room operation groups."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # 1. Unknown group function -> UnmappedFunctionalConceptError
    doc_bad_fn = create_ideal_living_room_doc()
    doc_bad_fn["interaction_groups"][0]["function"] = "make coffee in the living room"
    with pytest.raises(UnmappedFunctionalConceptError, match="cannot be mapped"):
        provider.generate_canonical(raw_document=doc_bad_fn)

    # 2. Mismatched required_target_count -> MalformedVLMSpecificationError
    doc_count = create_ideal_living_room_doc()
    doc_count["interaction_groups"][0]["required_target_count"] = 1  # but CUP_SAUCER_SET is 2
    with pytest.raises(MalformedVLMSpecificationError, match="does not match target role count"):
        provider.generate_canonical(raw_document=doc_count)

    # 3. Contradictory endpoints -> MalformedVLMSpecificationError
    doc_bad_ep = create_ideal_living_room_doc()
    doc_bad_ep["interaction_groups"][0]["tool_role"] = "role_2"  # SHARED_REMOTE_REGION instead of PERSONAL
    with pytest.raises(MalformedVLMSpecificationError, match="contradict function"):
        provider.generate_canonical(raw_document=doc_bad_ep)


def test_missing_fixed_target_roles_fail_closed():
    """Verify that missing raw FIXED_TARGET roles are never fabricated and fail closed."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # A: Remove raw SEATING_POSITION role, keep personal group / NEAR_SEAT -> MalformedVLMSpecificationError
    doc_a = create_ideal_living_room_doc()
    doc_a["functional_roles"] = [r for r in doc_a["functional_roles"] if r["id"] != "role_5"]
    with pytest.raises(MalformedVLMSpecificationError, match="role 'role_5' not declared"):
        provider.generate_canonical(raw_document=doc_a)

    # B: Remove raw SEATING_PAIR role, keep ACCESSIBLE_FROM_BOTH_SEATS -> MalformedVLMSpecificationError
    doc_b = create_ideal_living_room_doc()
    doc_b["functional_roles"] = [r for r in doc_b["functional_roles"] if r["id"] != "role_6"]
    with pytest.raises(MalformedVLMSpecificationError, match="role 'role_6' not declared"):
        provider.generate_canonical(raw_document=doc_b)

    # C: Remove both fixed target roles -> MalformedVLMSpecificationError
    doc_c = create_ideal_living_room_doc()
    doc_c["functional_roles"] = [r for r in doc_c["functional_roles"] if r["id"] not in ("role_5", "role_6")]
    with pytest.raises(MalformedVLMSpecificationError):
        provider.generate_canonical(raw_document=doc_c)

    # Verify no canonical trace contains fabricated IDs like system_context_seating_*
    doc_valid = create_ideal_living_room_doc()
    res = provider.generate_canonical(raw_document=doc_valid)
    trace_json = str(res["canonicalization_trace"])
    assert "system_context_seating_position" not in trace_json
    assert "system_context_seating_pair" not in trace_json


def test_self_relations_fail_closed():
    """Verify that self-relations cannot invent fixed-target endpoints and fail closed."""
    # 1. PERSONAL_REGION -- near seat --> PERSONAL_REGION -> MalformedVLMSpecificationError
    with pytest.raises(MalformedVLMSpecificationError, match="expects endpoints"):
        canonicalize_living_room_relation(
            "near seat",
            "PERSONAL_CUP_SAUCER_REGION",
            "PERSONAL_CUP_SAUCER_REGION",
        )

    # 2. SHARED_REGION -- accessible from both seats --> SHARED_REGION -> MalformedVLMSpecificationError
    with pytest.raises(MalformedVLMSpecificationError, match="expects endpoints"):
        canonicalize_living_room_relation(
            "accessible from both seats",
            "SHARED_REMOTE_REGION",
            "SHARED_REMOTE_REGION",
        )

    # 3. PERSONAL_REGION -- can hold drinkware set --> PERSONAL_REGION -> MalformedVLMSpecificationError
    with pytest.raises(MalformedVLMSpecificationError, match="expects endpoints"):
        canonicalize_living_room_relation(
            "can hold drinkware set",
            "PERSONAL_CUP_SAUCER_REGION",
            "PERSONAL_CUP_SAUCER_REGION",
        )

    # 4. SHARED_REGION -- can hold remote --> SHARED_REGION -> MalformedVLMSpecificationError
    with pytest.raises(MalformedVLMSpecificationError, match="expects endpoints"):
        canonicalize_living_room_relation(
            "can hold remote",
            "SHARED_REMOTE_REGION",
            "SHARED_REMOTE_REGION",
        )


def test_disjoint_slot_composition_rules():
    """Verify deterministic disjoint slot composition and rejection of generic duplicates."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # A: Two generic personal-region roles count 1 each (no disjoint slot identity) -> AmbiguousCanonicalizationError
    doc_a = create_ideal_living_room_doc()
    doc_a["functional_roles"] = [r for r in doc_a["functional_roles"] if r["id"] != "role_1"]
    doc_a["functional_roles"].extend([
        {
            "id": "drink_surface_a",
            "entity_kind": "REGION",
            "function": "personal cup and saucer support",
            "description": "side table surface for drinks",
            "required_count": 1,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["side table"],
            "visible_candidates": [],
            "required_properties": ["planar horizontal support"],
        },
        {
            "id": "drink_surface_b",
            "entity_kind": "REGION",
            "function": "personal cup and saucer support",
            "description": "side table surface for drinks",
            "required_count": 1,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["side table"],
            "visible_candidates": [],
            "required_properties": ["planar horizontal support"],
        },
    ])
    for rel in doc_a["functional_relations"]:
        if rel["subject_role"] == "role_1":
            rel["subject_role"] = "drink_surface_a"
    doc_a["interaction_groups"][0]["tool_role"] = "drink_surface_a"
    with pytest.raises(AmbiguousCanonicalizationError, match="lack explicit disjoint slot identities"):
        provider.generate_canonical(raw_document=doc_a)

    # B: Explicit left + right personal-region roles count 1 each -> count 2 (MERGED_BY_EXPLICIT_RULE)
    doc_b = deepcopy(doc_a)
    doc_b["functional_roles"][-2]["description"] = "left viewer drink table"
    doc_b["functional_roles"][-1]["description"] = "right viewer drink table"
    res_b = provider.generate_canonical(raw_document=doc_b)
    p_req = next(r for r in res_b["normalized_requirements"] if r["function"] == "PERSONAL_CUP_SAUCER_REGION")
    assert p_req["vlm_required_count"] == 2
    acct_b = res_b["canonicalization_trace"]["concept_accounting"]["roles"]
    assert acct_b["drink_surface_a"]["status"] == "MERGED_BY_EXPLICIT_RULE"
    assert acct_b["drink_surface_b"]["status"] == "MERGED_BY_EXPLICIT_RULE"

    # C: Duplicate slot identity: left + left -> AmbiguousCanonicalizationError
    doc_c = deepcopy(doc_a)
    doc_c["functional_roles"][-2]["description"] = "left viewer drink table"
    doc_c["functional_roles"][-1]["description"] = "left seat drink table"
    with pytest.raises(AmbiguousCanonicalizationError, match="declare duplicate slot identity"):
        provider.generate_canonical(raw_document=doc_c)

    # D: Two generic SEATING_POSITION roles count 1 each -> AmbiguousCanonicalizationError
    doc_d = create_ideal_living_room_doc()
    doc_d["functional_roles"] = [r for r in doc_d["functional_roles"] if r["id"] != "role_5"]
    doc_d["functional_roles"].extend([
        {
            "id": "seat_a",
            "entity_kind": "FIXED_TARGET",
            "function": "viewer seating position",
            "description": "seated viewer position",
            "required_count": 1,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["armchair"],
            "visible_candidates": [],
            "required_properties": [],
        },
        {
            "id": "seat_b",
            "entity_kind": "FIXED_TARGET",
            "function": "viewer seating position",
            "description": "seated viewer position",
            "required_count": 1,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["armchair"],
            "visible_candidates": [],
            "required_properties": [],
        },
    ])
    for rel in doc_d["functional_relations"]:
        if rel["object_role"] == "role_5":
            rel["object_role"] = "seat_a"
    doc_d["interaction_groups"][0]["context_role"] = "seat_a"
    with pytest.raises(AmbiguousCanonicalizationError, match="lack explicit disjoint slot identities"):
        provider.generate_canonical(raw_document=doc_d)

    # E: Explicit left/right seating positions -> count 2 (MERGED_BY_EXPLICIT_RULE)
    doc_e = deepcopy(doc_d)
    doc_e["functional_roles"][-2]["description"] = "first viewer seat 1"
    doc_e["functional_roles"][-1]["description"] = "second viewer seat 2"
    res_e = provider.generate_canonical(raw_document=doc_e)
    seat_req = next(r for r in res_e["normalized_requirements"] if r["function"] == "SEATING_POSITION")
    assert seat_req["vlm_required_count"] == 2
    acct_e = res_e["canonicalization_trace"]["concept_accounting"]["roles"]
    assert acct_e["seat_a"]["status"] == "MERGED_BY_EXPLICIT_RULE"
    assert acct_e["seat_b"]["status"] == "MERGED_BY_EXPLICIT_RULE"

    # F: Aggregate SEATING_POSITION count 2 -> preserved count 2 directly
    doc_f = create_ideal_living_room_doc()
    res_f = provider.generate_canonical(raw_document=doc_f)
    seat_f = next(r for r in res_f["normalized_requirements"] if r["function"] == "SEATING_POSITION")
    assert seat_f["vlm_required_count"] == 2
    assert res_f["canonicalization_trace"]["concept_accounting"]["roles"]["role_5"]["status"] == "PRESERVED"


def test_planar_support_omission_fails_closed():
    """Verify that support REGION roles missing PLANAR_SUPPORT fail closed."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # 1. Support REGION with required_properties=[] -> MalformedVLMSpecificationError
    doc_empty_prop = create_ideal_living_room_doc()
    doc_empty_prop["functional_roles"][0]["required_properties"] = []
    with pytest.raises(MalformedVLMSpecificationError, match="omitted required executable property 'PLANAR_SUPPORT'"):
        provider.generate_canonical(raw_document=doc_empty_prop)

    # 2. Support REGION with PLANAR_SUPPORT -> PRESERVED
    doc_valid = create_ideal_living_room_doc()
    res_valid = provider.generate_canonical(raw_document=doc_valid)
    assert res_valid["ready_for_grounding"] is True

    # 3. Two synonymous planar properties -> one canonical property + MERGED_BY_EXPLICIT_RULE trace
    doc_syn = create_ideal_living_room_doc()
    doc_syn["functional_roles"][0]["required_properties"] = ["planar support", "horizontal planar support"]
    res_syn = provider.generate_canonical(raw_document=doc_syn)
    prop_acct = res_syn["canonicalization_trace"]["concept_accounting"]["properties"]
    assert any(p["status"] == "MERGED_BY_EXPLICIT_RULE" for p in prop_acct)


def test_missing_schema_required_fields_fail_closed():
    """Verify that schema-required role and group fields without values fail closed (no silent defaults)."""
    provider = EnvironmentVLMRequirementProvider("living_room")

    # 1. Missing role.required_count -> MalformedVLMSpecificationError
    doc_no_count = create_ideal_living_room_doc()
    del doc_no_count["functional_roles"][0]["required_count"]
    with pytest.raises(MalformedVLMSpecificationError, match="missing required 'required_count'"):
        provider.generate_canonical(raw_document=doc_no_count)

    # 2. Missing role.binding_policy -> MalformedVLMSpecificationError
    doc_no_pol = create_ideal_living_room_doc()
    del doc_no_pol["functional_roles"][0]["binding_policy"]
    with pytest.raises(MalformedVLMSpecificationError, match="missing required 'binding_policy'"):
        provider.generate_canonical(raw_document=doc_no_pol)

    # 3. Missing group.required_target_count -> MalformedVLMSpecificationError
    doc_no_tgt_cnt = create_ideal_living_room_doc()
    del doc_no_tgt_cnt["interaction_groups"][0]["required_target_count"]
    with pytest.raises(MalformedVLMSpecificationError, match="missing required 'required_target_count'"):
        provider.generate_canonical(raw_document=doc_no_tgt_cnt)

    # 4. Missing group.usage_policy -> MalformedVLMSpecificationError
    doc_no_grp_pol = create_ideal_living_room_doc()
    del doc_no_grp_pol["interaction_groups"][0]["usage_policy"]
    with pytest.raises(MalformedVLMSpecificationError, match="missing required 'usage_policy'"):
        provider.generate_canonical(raw_document=doc_no_grp_pol)

    # 5. Missing group.required_relations -> MalformedVLMSpecificationError
    doc_no_req_rels = create_ideal_living_room_doc()
    del doc_no_req_rels["interaction_groups"][0]["required_relations"]
    with pytest.raises(MalformedVLMSpecificationError, match="missing non-empty required_relations"):
        provider.generate_canonical(raw_document=doc_no_req_rels)

