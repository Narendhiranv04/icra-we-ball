from __future__ import annotations

from copy import deepcopy

import jsonschema
import pytest

from mujoco_scenes.environment_vlm_requirements import (
    map_living_room_fixed_target_role,
    map_living_room_object_payload_role,
    map_living_room_role_function,
)
from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.executability import analyze_executability
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    LIVE_RESPONSE_SCHEMA_V2,
    compute_v2_prompt_and_schema_hash,
    normalize_v2_live_document,
    validate_v2_live_contract,
)
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
)
from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import interpret_relation
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import _map_role, compile_candidate_graph
from mujoco_scenes.kitchen_vlm_functional_graph import map_kitchen_role_function
from mujoco_scenes.workshop_phase1.requirements import map_workshop_role_function


def _role(role_id: str, function: str, count: int = 1, binding: str = "REUSABLE") -> dict:
    return {
        "id": role_id,
        "entity_kind": "OBJECT",
        "function": function,
        "required_count": count,
        "binding_policy": binding,
        "candidate_categories": [],
        "required_properties": [],
    }


def _doc(anchor: str | None = None, *, source: str = "source", target: str = "target",
         count: int = 1, reuse: str = "REUSABLE_ACROSS_TARGETS") -> dict:
    operation = {
        "id": "op",
        "operation": "transfer material",
        "source_role": source,
        "target_role": target,
        "operation_count": count,
        "reuse_policy": reuse,
    }
    if anchor is not None:
        operation["anchor_role"] = anchor
    return {
        "status": "SUPPORTED",
        "task_summary": "transfer",
        "task_contract": {
            "functional_roles": [_role("source", "material source"), _role("target", "receiver")],
            "functional_relations": [],
            "operation_pairings": [operation],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }


@pytest.mark.parametrize(
    ("anchor", "code"),
    [("", "EMPTY_OPTIONAL_ANCHOR_REMOVED"), ("   ", "EMPTY_OPTIONAL_ANCHOR_REMOVED"),
     ("source", "REDUNDANT_ANCHOR_REMOVED"), ("target", "REDUNDANT_ANCHOR_REMOVED")],
)
def test_safe_optional_anchor_normalization(anchor: str, code: str):
    raw = _doc(anchor)
    normalized, trace = normalize_v2_live_document(raw)
    assert "anchor_role" not in normalized["task_contract"]["operation_pairings"][0]
    assert trace[0]["code"] == code
    assert raw["task_contract"]["operation_pairings"][0]["anchor_role"] == anchor


def test_undeclared_anchor_is_not_repaired_and_self_pair_still_fails():
    normalized, trace = normalize_v2_live_document(_doc("missing"))
    assert not trace and normalized["task_contract"]["operation_pairings"][0]["anchor_role"] == "missing"
    with pytest.raises(MalformedVLMSpecificationError, match="anchor_role"):
        validate_v2_live_contract(normalized)
    with pytest.raises(MalformedVLMSpecificationError, match="SELF_PAIRING"):
        validate_v2_live_contract(_doc(source="source", target="source"))


def test_live_schema_disallows_empty_anchor_when_present():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_doc(""), LIVE_RESPONSE_SCHEMA_V2)


def test_operation_reuse_cardinality_consistency():
    reusable = _doc(count=2)
    reusable["task_contract"]["functional_roles"][1]["required_count"] = 2
    validate_v2_live_contract(reusable)

    contradictory = _doc(count=2, reuse="DEDICATED_PER_TARGET")
    contradictory["task_contract"]["functional_roles"][1]["required_count"] = 2
    with pytest.raises(MalformedVLMSpecificationError, match="INCONSISTENT_OPERATION_REUSE_CARDINALITY"):
        validate_v2_live_contract(contradictory)

    distinct = _doc(count=2, reuse="DEDICATED_PER_TARGET")
    distinct["task_contract"]["functional_roles"][0].update(required_count=2, binding_policy="DISTINCT")
    distinct["task_contract"]["functional_roles"][1]["required_count"] = 2
    validate_v2_live_contract(distinct)


def test_compiler_also_fails_closed_on_dedicated_source_shortage():
    raw = {
        "functional_roles": [
            _role("source", "coffee material source", 1, "REUSABLE"),
            _role("target", "coffee serving vessel", 2, "DISTINCT"),
        ],
        "functional_relations": [],
        "interaction_groups": [{
            "id": "transfer",
            "function": "transfer coffee to cup",
            "tool_role": "source",
            "target_role": "target",
            "required_target_count": 2,
            "usage_policy": "DEDICATED_PER_TARGET",
            "required_relations": [],
        }],
        "inspectable_regions": [],
        "inspection_order": [],
    }
    graph = compile_candidate_graph("kitchen", "transfer twice", raw)
    assert graph.metadata["canonicalization_status"] == "PARTIAL"
    assert graph.metadata["canonicalization_trace"]["disabled_groups"][0]["status"] == (
        "INCONSISTENT_OPERATION_REUSE_CARDINALITY"
    )


def test_generic_domain_role_semantics_and_no_cross_mapping():
    assert map_kitchen_role_function(_role("x", "coffee material source")) == "coffee_source"
    assert map_kitchen_role_function(_role("x", "utensil for eating soup")) == "soup_eating_utensil"
    assert map_living_room_role_function({**_role("x", "personal refreshment support surface"), "entity_kind": "REGION"}) == "PERSONAL_CUP_SAUCER_REGION"
    assert map_living_room_object_payload_role(_role("x", "refreshment cup and dish set")) == "CUP_SAUCER_SET"
    assert map_living_room_role_function({**_role("x", "shared central support for media control"), "entity_kind": "REGION", "binding_policy": "SHARED"}) == "SHARED_REMOTE_REGION"
    assert map_living_room_object_payload_role(_role("x", "television controller")) == "REMOTE"
    assert map_living_room_object_payload_role(_role("x", "television display")) is None
    assert map_living_room_fixed_target_role(_role("x", "television display")) is None
    assert map_workshop_role_function(_role("x", "mechanical joining part")) == "CAN_FASTEN"
    assert map_workshop_role_function(_role("x", "driving tool instrument")) == "CAN_DRIVE_SCREW"
    assert _map_role("workshop", _role("x", "object receiving fastening"), {"interaction_groups": [], "functional_relations": []})[0] == "repair_target"


def test_missing_domain_participants_are_not_synthesized():
    kitchen_doc = {"functional_roles": [_role("bowl", "soup serving vessel")], "functional_relations": [], "interaction_groups": []}
    workshop_doc = {"functional_roles": [_role("tool", "driving tool instrument")], "functional_relations": [], "interaction_groups": []}
    assert {_map_role("kitchen", r, kitchen_doc)[0] for r in kitchen_doc["functional_roles"]} == {"soup_container"}
    assert {_map_role("workshop", r, workshop_doc)[0] for r in workshop_doc["functional_roles"]} == {"driver"}


def test_machine_style_relation_aliases_and_direction():
    transfer = interpret_relation("kitchen", "receives_transfer_from", "coffee_container", "coffee_source", "OBJECT", "OBJECT")
    assert transfer.category == "TASK_CAUSAL_SEMANTICS"
    assert (transfer.interpreted_predicates[0].subject_role, transfer.interpreted_predicates[0].predicate_name,
            transfer.interpreted_predicates[0].object_role) == ("coffee_source", "PROVIDES_MATERIAL_TO", "coffee_container")
    near = interpret_relation("living_room", "located_adjacent", "SEATING_POSITION", "PERSONAL_CUP_SAUCER_REGION", "FIXED_TARGET", "REGION")
    assert near.direction_normalized and near.interpreted_predicates[0].predicate_name == "NEAR_SEAT"
    acts = interpret_relation("workshop", "acts_upon_during_fastening", "driver", "fastener", "OBJECT", "OBJECT")
    assert acts.category == "TASK_CAUSAL_SEMANTICS" and acts.interpreted_predicates[0].predicate_name == "ACTS_ON"
    unrelated = interpret_relation("kitchen", "associated with", "soup_eating_utensil", "soup_container", "OBJECT", "OBJECT")
    assert not unrelated.succeeded
    access = interpret_relation("living_room", "accessible_from", "SHARED_REMOTE_REGION", "SEATING_PAIR", "REGION", "FIXED_TARGET")
    assert access.interpreted_predicates[0].predicate_name == "ACCESSIBLE_FROM_BOTH_SEATS"
    invalid_access = interpret_relation("living_room", "near", "REMOTE", "CUP_SAUCER_SET", "OBJECT", "OBJECT")
    assert not invalid_access.succeeded


def test_relationless_capability_requires_registered_zero_precondition_capability():
    nodes = {
        "coffee_source": FunctionalRole("coffee_source", semantic_categories=("coffee",)),
        "coffee_container": FunctionalRole("coffee_container", semantic_categories=("cup",)),
    }
    group = OperationGroup(
        "transfer", "POUR", "coffee_source", "coffee_container", 1,
        "SEQUENTIAL_REUSE_ALLOWED", capability_id="TRANSFER_CONTENT_TO_CONTAINER"
    )
    graph = FunctionalRequirementGraph("kitchen", "transfer", nodes, operation_groups=(group,))
    statuses = analyze_executability(graph, {"coffee_source": "jar", "coffee_container": "cup"})
    assert statuses[-1]["status"] == "EXECUTABLE"
    absent = FunctionalRequirementGraph("kitchen", "transfer", nodes, operation_groups=(OperationGroup(
        "unknown", "UNKNOWN", "coffee_source", "coffee_container", 1,
        "SEQUENTIAL_REUSE_ALLOWED", capability_id="NOT_REGISTERED"
    ),))
    assert analyze_executability(absent, {"coffee_source": "jar", "coffee_container": "cup"})[-1]["status"] == "UNINSTANTIABLE_MISSING_RELATION"


def test_user_request_is_part_of_v2_hash(monkeypatch):
    before = compute_v2_prompt_and_schema_hash()
    monkeypatch.setattr(
        "mujoco_scenes.functional_tamp_pipeline.fm_schema_v2.USER_REQUEST_V2",
        "changed canonical request",
    )
    assert compute_v2_prompt_and_schema_hash() != before
