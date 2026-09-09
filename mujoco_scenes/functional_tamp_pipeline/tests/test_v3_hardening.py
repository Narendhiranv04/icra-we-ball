import copy

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses


def role(role_id, function, *, kind="OBJECT", count=1, policy="REUSABLE", categories=(), description=""):
    return {"id": role_id, "entity_kind": kind, "function": function, "description": description,
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": []}


def operation(op_id, text, participants, count=1):
    return {"id": op_id, "operation": text, "participant_roles": list(participants), "operation_count": count}


def relation(rel_id, text, participants):
    return {"id": rel_id, "relation": text, "participant_roles": list(participants), "required": True}


def document(roles=(), relations=(), operations=(), *, status="SUPPORTED", reason="", regions=(), order=()):
    return {"schema_version": 3, "status": status, "task_summary": "task",
            "task_contract": {"functional_roles": list(roles), "functional_relations": list(relations),
                              "operation_pairings": list(operations)},
            "observation_guidance": {"visible_candidates_per_role": {},
                                     "inspectable_regions": list(regions), "inspection_order": list(order)},
            "unsupported_reason": reason}


def test_robot_agent_is_removed_but_physical_fastening_participants_remain():
    raw = document([
        role("robot", "active agent performing manipulation", kind="REGION", policy="SHARED"),
        role("driver", "reusable fastening implement"), role("fastener", "manipulated joining component"),
        role("target", "fixed repair target", kind="FIXED_TARGET", policy="SHARED"),
    ], operations=[operation("secure", "secure joint", ["robot", "driver", "fastener", "target"])])
    normalized, trace = normalize_and_validate_v3_contract(raw, domain="workshop", task_instruction="Secure the joint.")
    assert [r["id"] for r in normalized["task_contract"]["functional_roles"]] == ["driver", "fastener", "target"]
    assert normalized["task_contract"]["operation_pairings"][0]["participant_roles"] == ["driver", "fastener", "target"]
    assert any(row["code"] == "IMPLICIT_ROBOT_EXECUTOR_REMOVED" for row in trace)


def test_robot_component_target_is_not_removed_when_instruction_targets_it():
    raw = document([role("gripper", "robot gripper component"), role("tool", "repair implement")],
                   operations=[operation("repair", "repair component", ["tool", "gripper"])])
    normalized, _ = normalize_and_validate_v3_contract(raw, task_instruction="Inspect and repair the robot gripper.")
    assert {r["id"] for r in normalized["task_contract"]["functional_roles"]} == {"gripper", "tool"}


def test_function_semantics_dominate_incidental_description_neighbors():
    personal = document([role("support", "personal support", kind="REGION", description="table beside armchair")])
    seating = document([role("seat", "seating position", kind="FIXED_TARGET", description="chair beside table")])
    p = build_role_type_hypotheses("living_room", convert_v3_to_canonical_document(personal, domain="living_room"))["support"]
    s = build_role_type_hypotheses("living_room", convert_v3_to_canonical_document(seating, domain="living_room"))["seat"]
    assert p.canonical_role_candidates == ("PERSONAL_CUP_SAUCER_REGION",)
    assert s.canonical_role_candidates == ("SEATING_POSITION",)
    assert p.evidence[0]["source"] == "FUNCTION_TEXT"


def test_initial_location_source_is_context_not_material_source():
    raw = document([role("desk", "initial refreshment source", kind="FIXED_TARGET", policy="SHARED", categories=["TABLE", "DESK"])])
    from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import _map_role
    assert _map_role("living_room", raw["task_contract"]["functional_roles"][0], raw) == (None, "CURRENT_STATE_CONTEXT_ONLY")


def test_current_location_relation_is_preserved_outside_required_gf_relations():
    raw = document([role("payload", "refreshment payload", categories=["CUP"]),
                    role("desk", "initial storage surface", kind="REGION", policy="SHARED")],
                   relations=[relation("initial", "LOCATED_AT_INITIALLY", ["payload", "desk"])])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert canonical["functional_relations"] == []
    assert canonical["current_state_relations"][0]["category"] == "CURRENT_STATE_CONTEXT"


def test_equivalent_payload_instances_consolidate_as_two_distinct_objects():
    raw = document([
        role("cup_1", "refreshment payload", categories=["CUP"]),
        role("cup_2", "refreshment payload", categories=["CUP"]),
        role("support", "personal support", kind="REGION", policy="SHARED"),
    ], operations=[operation("a", "move", ["cup_1", "support"]), operation("b", "move", ["cup_2", "support"])])
    graph = compile_candidate_graph("living_room", "place refreshments", raw)
    payload = graph.nodes["CUP_SAUCER_SET"]
    assert (payload.count, payload.binding_policy) == (2, "DISTINCT")
    assert graph.metadata["canonicalization_trace"]["merged_roles"][0]["code"] == "CONSOLIDATED_EQUIVALENT_FM_INSTANCES"


def test_non_equivalent_spoon_functions_never_merge():
    raw = document([role("stirrer", "coffee stirring implement"), role("utensil", "soup eating utensil")])
    graph = compile_candidate_graph("kitchen", "stir coffee and provide soup utensil", raw)
    assert {"coffee_stirrer", "soup_eating_utensil"} <= set(graph.nodes)


def test_unordered_operation_resolves_identically_when_reversed():
    roles = [role("tool", "reusable fastening implement"), role("part", "joining component"),
             role("target", "fixed repair target", kind="FIXED_TARGET", policy="SHARED")]
    first = convert_v3_to_canonical_document(document(roles, operations=[operation("x", "fasten", ["tool", "part", "target"])]), domain="workshop")
    second = convert_v3_to_canonical_document(document(roles, operations=[operation("x", "fasten", ["target", "part", "tool"])]), domain="workshop")
    slots = lambda d: [(x["source_role"], x["target_role"], x["anchor_role"]) for x in d["interaction_groups"][0]["v3_slot_assignments"]]
    assert slots(first) == slots(second) == [("tool", "part", "target")]


def test_unknown_operation_never_falls_back_to_participant_order():
    raw = document([role("a", "object"), role("b", "object")], operations=[operation("x", "levitate ceremonially", ["a", "b"])])
    group = convert_v3_to_canonical_document(raw, domain="workshop")["interaction_groups"][0]
    assert group["tool_role"] is group["target_role"] is None
    assert group["v3_participant_roles"] == ["a", "b"]


def test_multiple_slot_assignments_are_not_committed_to_first_option():
    raw = document([
        role("support", "shared support surface", kind="REGION", policy="SHARED"),
        role("payload", "entertainment remote control"),
        role("context", "contextual reference", kind="FIXED_TARGET", policy="SHARED"),
    ], operations=[operation("x", "place remote control", ["payload", "support", "context"])])
    group = convert_v3_to_canonical_document(raw, domain="living_room")["interaction_groups"][0]
    assert len(group["v3_slot_assignments"]) > 1
    assert group["tool_role"] is group["target_role"] is None


@pytest.mark.parametrize("text,participants,capability", [
    ("move", ("remote", "shared"), "SUPPORT_ENTERTAINMENT_CONTROL"),
    ("transfer", ("refreshment", "personal"), "SUPPORT_DRINKWARE"),
])
def test_broad_living_relocation_family_is_filtered_by_participants(text, participants, capability):
    roles = [role("remote", "entertainment remote control"), role("shared", "shared support", kind="REGION", policy="SHARED"),
             role("refreshment", "refreshment payload", categories=["CUP"]), role("personal", "personal support", kind="REGION", policy="SHARED")]
    group = convert_v3_to_canonical_document(document(roles, operations=[operation("x", text, participants)]), domain="living_room")["interaction_groups"][0]
    assert {row["capability_id"] for row in group["v3_slot_assignments"]} == {capability}


def test_broad_workshop_fasten_and_return_families():
    roles = [role("tool", "reusable fastening implement"), role("part", "joining component"),
             role("target", "fixed repair target", kind="FIXED_TARGET", policy="SHARED"),
             role("bench", "workbench support", kind="REGION", policy="SHARED")]
    canonical = convert_v3_to_canonical_document(document(roles, operations=[
        operation("a", "secure", ["tool", "part", "target"]), operation("b", "deposit", ["tool", "bench"]),
    ]), domain="workshop")
    assert [{row["capability_id"] for row in g["v3_slot_assignments"]} for g in canonical["interaction_groups"]] == [
        {"FASTEN_JOINT"}, {"RETURN_REUSABLE_ITEM_TO_SUPPORT"},
    ]


@pytest.mark.parametrize("kind", ["relation", "operation"])
def test_duplicate_participants_fail_manual_structural_validation(kind):
    raw = document([role("tool", "tool")],
                   relations=[relation("x", "compatible", ["tool", "tool"])] if kind == "relation" else (),
                   operations=[operation("x", "move", ["tool", "tool"])] if kind == "operation" else ())
    with pytest.raises(MalformedVLMSpecificationError, match="DUPLICATE_.*_PARTICIPANT"):
        normalize_and_validate_v3_contract(raw)


def test_inspection_order_is_repaired_from_region_declarations():
    regions = [{"id": "drawer_a", "label": "drawer", "visual_description": "closed drawer", "reason": "contains missing tool"},
               {"id": "cabinet_b", "label": "cabinet", "visual_description": "closed cabinet", "reason": "inspect"}]
    raw = document([role("tool", "tool")], regions=regions, order=["look in drawers", "drawer_a"])
    normalized, trace = normalize_and_validate_v3_contract(raw)
    assert normalized["observation_guidance"]["inspection_order"] == ["drawer_a", "cabinet_b"]
    assert any(row["code"] == "INSPECTION_ORDER_NORMALIZED" for row in trace)


def test_observability_only_unsupported_status_is_normalized():
    raw = document([role("tool", "tool")], status="UNSUPPORTED", reason="Hidden inventory is not visible in the initial RGB views.")
    normalized, trace = normalize_and_validate_v3_contract(raw)
    assert normalized["status"] == "SUPPORTED" and normalized["unsupported_reason"] == ""
    assert trace[0]["code"] == "OBSERVABILITY_UNSUPPORTED_NORMALIZED_TO_SUPPORTED"


def test_genuine_unsupported_empty_contract_remains_unsupported():
    raw = document(status="UNSUPPORTED", reason="The task cannot be represented by roles, relations, and operations.")
    normalized, _ = normalize_and_validate_v3_contract(raw)
    assert normalized["status"] == "UNSUPPORTED"


def test_entity_kind_payload_recovery_is_narrow_and_traced():
    raw = document([role("cup", "refreshment payload", kind="REGION", categories=["CUP"]),
                    role("table", "personal support", kind="REGION", policy="SHARED")],
                   operations=[operation("x", "move", ["cup", "table"])])
    normalized, trace = normalize_and_validate_v3_contract(raw, domain="living_room")
    assert normalized["task_contract"]["functional_roles"][0]["entity_kind"] == "OBJECT"
    assert any(row["code"] == "ENTITY_KIND_SEMANTIC_NORMALIZATION" for row in trace)
