"""Reverse compiler tests for joint function/relation/operation role typing."""

from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import (
    extract_relation_semantic_candidates,
)
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    extract_operation_semantic_candidates,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    resolve_role_type_hypotheses,
)


def _role(role_id, kind, function, policy="DISTINCT"):
    return {
        "id": role_id, "entity_kind": kind, "function": function,
        "description": "", "required_count": 1, "binding_policy": policy,
        "required_properties": [], "candidate_categories": [],
    }


def _doc(roles, relations=(), operations=()):
    return {
        "functional_roles": list(roles),
        "functional_relations": list(relations),
        "interaction_groups": list(operations),
    }


def _resolved(domain, document, role_id):
    return resolve_role_type_hypotheses(domain, document)[role_id]


def test_candidate_extractors_are_endpoint_independent_and_text_nominated():
    assert {c.predicate_name for c in extract_relation_semantic_candidates("kitchen", "fits inside vessel")} == {"INSERTABLE_IN"}
    assert extract_relation_semantic_candidates("kitchen", "semantically unrelated wording") == ()
    assert {c.capability_id for c in extract_operation_semantic_candidates("kitchen", "stir beverage")} == {"STIR_COFFEE"}
    assert extract_operation_semantic_candidates("kitchen", "inspect beverage") == ()


def test_direct_function_match_remains_authoritative_when_evidence_agrees():
    doc = _doc(
        [_role("tool", "OBJECT", "coffee stirring implement"), _role("cup", "OBJECT", "beverage vessel")],
        operations=[{"source_role": "tool", "target_role": "cup", "operation": "stir beverage"}],
    )
    hypothesis = _resolved("kitchen", doc, "tool")
    assert hypothesis.resolved_role == "coffee_stirrer"
    assert hypothesis.status == "DIRECT_FUNCTION_MATCH"


def test_unknown_function_relation_and_operation_uniquely_resolve_stirrer():
    doc = _doc(
        [_role("tool", "OBJECT", "elongated preparation implement"), _role("cup", "OBJECT", "beverage vessel")],
        [{"subject_role": "tool", "relation": "fits inside beverage vessel", "object_role": "cup"}],
        [{"source_role": "tool", "target_role": "cup", "operation": "stir beverage"}],
    )
    hypothesis = _resolved("kitchen", doc, "tool")
    assert hypothesis.resolved_role == "coffee_stirrer"
    assert hypothesis.status == "JOINT_SEMANTIC_RESOLUTION"


def test_relation_only_ambiguity_is_retained_not_arbitrarily_selected():
    doc = _doc(
        [_role("tool", "OBJECT", "elongated preparation implement"), _role("vessel", "OBJECT", "container")],
        [{"subject_role": "tool", "relation": "fits inside container", "object_role": "vessel"}],
    )
    hypothesis = _resolved("kitchen", doc, "tool")
    assert hypothesis.canonical_role_candidates == ("coffee_stirrer", "soup_eating_utensil")
    assert hypothesis.status == "AMBIGUOUS_ROLE_TYPE"


def test_operation_disambiguates_relation_hypothesis():
    doc = _doc(
        [_role("tool", "OBJECT", "elongated preparation implement"), _role("vessel", "OBJECT", "container")],
        [{"subject_role": "tool", "relation": "fits inside container", "object_role": "vessel"}],
        [{"source_role": "tool", "target_role": "vessel", "operation": "stir beverage"}],
    )
    assert _resolved("kitchen", doc, "tool").resolved_role == "coffee_stirrer"


def test_direct_function_contradiction_fails_closed():
    doc = _doc(
        [_role("display", "OBJECT", "television display"), _role("surface", "REGION", "central shared support")],
        operations=[{"source_role": "display", "target_role": "surface", "operation": "place entertainment control"}],
    )
    assert _resolved("living_room", doc, "display").status == "CONTRADICTORY_ROLE_TYPE"


def test_workshop_operation_and_relations_reverse_type_fastener():
    doc = _doc(
        [_role("tool", "OBJECT", "driving equipment"), _role("part", "OBJECT", "retained element"),
         _role("anchor", "FIXED_TARGET", "marked fastening target")],
        [{"subject_role": "tool", "relation": "compatible with", "object_role": "part"},
         {"subject_role": "part", "relation": "compatible with fastening target", "object_role": "anchor"}],
        [{"source_role": "tool", "target_role": "part", "anchor_role": "anchor", "operation": "fasten component at target"}],
    )
    assert _resolved("workshop", doc, "part").resolved_role == "fastener"


def test_living_support_relations_reverse_type_personal_region():
    doc = _doc(
        [_role("support", "REGION", "bounded horizontal locus"), _role("seat", "FIXED_TARGET", "individual seating"),
         _role("payload", "OBJECT", "handled payload")],
        [{"subject_role": "support", "relation": "near individual seating", "object_role": "seat"},
         {"subject_role": "support", "relation": "supports refreshment setting", "object_role": "payload"}],
    )
    assert _resolved("living_room", doc, "support").resolved_role == "PERSONAL_CUP_SAUCER_REGION"


def test_living_support_relations_reverse_type_shared_region():
    doc = _doc(
        [_role("support", "REGION", "bounded horizontal locus", "SHARED"), _role("seat", "FIXED_TARGET", "seating pair", "SHARED"),
         _role("payload", "OBJECT", "handled payload")],
        [{"subject_role": "support", "relation": "accessible from seating pair", "object_role": "seat"},
         {"subject_role": "support", "relation": "supports entertainment control", "object_role": "payload"}],
    )
    assert _resolved("living_room", doc, "support").resolved_role == "SHARED_REMOTE_REGION"
