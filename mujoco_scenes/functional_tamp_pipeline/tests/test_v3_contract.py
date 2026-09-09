import copy
import json
from pathlib import Path

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import TaskSpecificationValidationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    LIVE_RESPONSE_SCHEMA_V3,
    SYSTEM_PROMPT_V3,
    USER_REQUEST_V3,
    compute_v3_prompt_and_schema_hash,
    convert_v3_to_canonical_document,
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import SYSTEM_PROMPT_V2, USER_REQUEST_V2


def role(role_id, function, *, kind="OBJECT", count=1, policy="REUSABLE"):
    return {
        "id": role_id, "entity_kind": kind, "function": function,
        "required_count": count, "binding_policy": policy,
        "candidate_categories": [], "required_properties": [],
    }


def document(roles, relations=(), operations=(), visible=None):
    return {
        "schema_version": 3, "status": "SUPPORTED", "task_summary": "task",
        "task_contract": {
            "functional_roles": list(roles),
            "functional_relations": list(relations),
            "operation_pairings": list(operations),
        },
        "observation_guidance": {
            "visible_candidates_per_role": visible or {},
            "inspectable_regions": [], "inspection_order": [],
        },
        "unsupported_reason": "",
    }


def operation(op_id, text, participants, count=1):
    return {"id": op_id, "operation": text, "participant_roles": participants,
            "operation_count": count}


def relation(rel_id, text, participants):
    return {"id": rel_id, "relation": text, "participant_roles": participants,
            "required": True}


def test_v3_operation_schema_has_participant_set_and_no_reuse_or_slots():
    props = LIVE_RESPONSE_SCHEMA_V3["properties"]["task_contract"]["properties"]["operation_pairings"]["items"]["properties"]
    assert "participant_roles" in props
    assert not {"source_role", "target_role", "anchor_role", "reuse_policy"} & set(props)


def test_v3_relation_schema_accepts_two_to_four_manual_unique_participants():
    props = LIVE_RESPONSE_SCHEMA_V3["properties"]["task_contract"]["properties"]["functional_relations"]["items"]["properties"]
    assert props["participant_roles"]["minItems"] == 2
    assert props["participant_roles"]["maxItems"] == 4
    assert "uniqueItems" not in props["participant_roles"]


def test_reusable_transfer_count_is_derived_without_redundant_policy():
    raw = document(
        [role("water", "water material source"), role("cup", "receiving coffee container", count=2, policy="DISTINCT")],
        operations=[operation("fill", "transfer water into container", ["cup", "water"], 2)],
    )
    validated, _ = normalize_and_validate_v3_contract(raw, domain="kitchen")
    group = convert_v3_to_canonical_document(validated, domain="kitchen")["interaction_groups"][0]
    assert group["tool_role"] == "water"
    assert group["target_role"] == "cup"
    assert group["usage_policy"] == "SEQUENTIAL_REUSE_ALLOWED"


def test_unordered_relation_is_oriented_from_semantics_and_types():
    raw = document(
        [role("cup", "receiving coffee container"), role("source", "coffee material source")],
        relations=[relation("supply", "supplies content", ["cup", "source"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="kitchen")
    edge = canonical["functional_relations"][0]
    assert (edge["subject_role"], edge["object_role"]) == ("source", "cup")


def test_kitchen_service_surface_resolves_to_existing_context():
    raw = document([role("service", "surface to hold items during service", kind="REGION", policy="SHARED")])
    canonical = convert_v3_to_canonical_document(raw, domain="kitchen")
    from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses
    assert build_role_type_hypotheses("kitchen", canonical)["service"].canonical_role_candidates == ("serving_area",)


def test_living_two_participant_relocation_remains_a_capability_candidate():
    raw = document(
        [role("payload", "personal refreshment drinkware set"),
         role("support", "personal side table support", kind="REGION", policy="SHARED")],
        operations=[operation("place", "place drinkware", ["payload", "support"], 2)],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    group = canonical["interaction_groups"][0]
    assert (group["tool_role"], group["target_role"]) == ("support", "payload")
    assert group["context_role"] is None


def test_living_personal_and_shared_slots_resolve():
    personal = document(
        [role("payload", "personal refreshment drinkware set", count=2, policy="DISTINCT"),
         role("support", "personal side table support", kind="REGION", count=2, policy="DISTINCT"),
         role("seat", "seating fixture", kind="FIXED_TARGET", policy="SHARED")],
        operations=[operation("place", "place drinkware", ["payload", "seat", "support"], 2)],
    )
    shared = document(
        [role("remote", "entertainment remote control"),
         role("support", "shared central support surface", kind="REGION", policy="SHARED"),
         role("seats", "shared seating context", kind="FIXED_TARGET", policy="SHARED")],
        operations=[operation("place", "place remote control", ["seats", "remote", "support"])],
    )
    p_group = convert_v3_to_canonical_document(personal, domain="living_room")["interaction_groups"][0]
    s_group = convert_v3_to_canonical_document(shared, domain="living_room")["interaction_groups"][0]
    assert (p_group["tool_role"], p_group["target_role"], p_group["context_role"]) == ("support", "payload", "seat")
    assert (s_group["tool_role"], s_group["target_role"], s_group["context_role"]) == ("support", "remote", "seats")


def test_workshop_missing_fastener_fails_but_complete_slots_resolve():
    missing = document(
        [role("tool", "reusable fastening tool"),
         role("target", "fixed assembly receiving installed component", kind="FIXED_TARGET", policy="SHARED")],
        operations=[operation("fasten", "fasten component at target", ["tool", "target"])],
    )
    with pytest.raises(TaskSpecificationValidationError):
        normalize_and_validate_v3_contract(missing, domain="workshop")
    complete = copy.deepcopy(missing)
    complete["task_contract"]["functional_roles"].append(role("fastener", "manipulated joining component"))
    complete["task_contract"]["operation_pairings"][0]["participant_roles"].append("fastener")
    group = convert_v3_to_canonical_document(complete, domain="workshop")["interaction_groups"][0]
    assert (group["tool_role"], group["target_role"], group["context_role"]) == ("tool", "fastener", "target")


def test_robot_self_observation_candidate_is_removed_unless_explicit_role():
    raw = document(
        [role("tool", "reusable fastening tool")],
        visible={"tool": [
            {"label": "end-effector attachment", "visual_description": "tool fixed to robot arm"},
            {"label": "hand driver", "visual_description": "loose reusable driver"},
        ]},
    )
    normalized, trace = normalize_and_validate_v3_contract(raw)
    assert [x["label"] for x in normalized["observation_guidance"]["visible_candidates_per_role"]["tool"]] == ["hand driver"]
    assert trace[0]["code"] == "ROBOT_SELF_CANDIDATE_REMOVED"


def test_v3_prompt_is_materially_shorter_and_hashes_actual_wire_contract():
    assert len(SYSTEM_PROMPT_V3.split()) + len(USER_REQUEST_V3.split()) < (
        len(SYSTEM_PROMPT_V2.split()) + len(USER_REQUEST_V2.split())
    ) / 2
    assert len(compute_v3_prompt_and_schema_hash()) == 64


def test_adapter_selects_v3_schema_and_preserves_one_call(monkeypatch, tmp_path):
    from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter

    raw = document([role("tool", "reusable fastening tool")])

    class Transport:
        def __init__(self):
            self.calls = []

        def complete(self, payload):
            self.calls.append(payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(raw)}}]}

    image = tmp_path / "view.png"
    image.write_bytes(b"view")
    transport = Transport()
    monkeypatch.setenv("TAMP_FM_SCHEMA_VERSION", "3")
    adapter = FMAdapter(base_url="http://local/v1", model="test", transport=transport)
    result = adapter.generate_kitchen_functional_graph("task", observation_images=[image])
    assert result["schema_version"] == 3
    assert len(transport.calls) == 1
    payload = transport.calls[0]
    assert payload["response_format"]["json_schema"]["schema"] is LIVE_RESPONSE_SCHEMA_V3
    assert "reuse_policy" not in payload["response_format"]["json_schema"]["schema"]["properties"]["task_contract"]["properties"]["operation_pairings"]["items"]["properties"]


def test_v3_compiles_to_existing_planner_facing_ir():
    from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph

    raw = document(
        [role("tool", "reusable fastening tool"),
         role("fastener", "manipulated joining component"),
         role("target", "fixed assembly receiving installed component", kind="FIXED_TARGET", policy="SHARED")],
        operations=[operation("fasten", "fasten component at target", ["target", "tool", "fastener"])],
    )
    graph = compile_candidate_graph("workshop", "task", raw)
    assert graph.online_executable_contract_complete
    assert graph.operation_groups[0].capability_id == "FASTEN_JOINT"
    assert graph.metadata["is_v3_specification"] is True


_FRESH_V2 = Path(__file__).resolve().parents[3] / "benchmark_reports" / "live_semantic_three_domain_gate_20260909T200613IST"


def _fresh_raw(domain, variant):
    return json.loads((_FRESH_V2 / domain / variant / "raw_v2.json").read_text())


def test_fresh_k1_reuse_contradiction_is_absent_in_v3_equivalent():
    old = _fresh_raw("kitchen", "K1")["task_contract"]
    transfers = []
    for old_op in old["operation_pairings"][:2]:
        transfers.append(operation(
            old_op["id"], old_op["operation"],
            [old_op["source_role"], old_op["target_role"]], old_op["operation_count"],
        ))
    raw = document(old["functional_roles"], operations=transfers)
    canonical = convert_v3_to_canonical_document(raw, domain="kitchen")
    assert [group["usage_policy"] for group in canonical["interaction_groups"]] == [
        "SEQUENTIAL_REUSE_ALLOWED", "SEQUENTIAL_REUSE_ALLOWED",
    ]


def test_fresh_l1_display_cannot_replace_missing_seating_context_in_v3():
    old = _fresh_raw("living_room", "L1")["task_contract"]
    old_op = old["operation_pairings"][0]
    raw = document(
        old["functional_roles"],
        operations=[operation(old_op["id"], old_op["operation"], [
            old_op["source_role"], old_op["target_role"], old_op["anchor_role"],
        ], old_op["operation_count"])],
    )
    with pytest.raises(TaskSpecificationValidationError):
        normalize_and_validate_v3_contract(raw, domain="living_room")


def test_fresh_w3_participant_set_still_fails_when_fastener_is_omitted():
    old = _fresh_raw("workshop", "W3")["task_contract"]
    old_op = old["operation_pairings"][0]
    raw = document(
        old["functional_roles"],
        operations=[operation(old_op["id"], old_op["operation"], [
            old_op["source_role"], old_op["target_role"], old_op["anchor_role"],
        ], old_op["operation_count"])],
    )
    with pytest.raises(TaskSpecificationValidationError):
        normalize_and_validate_v3_contract(raw, domain="workshop")
