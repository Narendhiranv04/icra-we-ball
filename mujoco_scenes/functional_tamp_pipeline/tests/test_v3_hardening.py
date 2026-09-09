import copy

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import (
    MalformedVLMSpecificationError,
    TaskSpecificationValidationError,
)
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


def test_plain_is_on_without_destination_operation_is_current_state_context():
    raw = document([
        role("tool", "fastening instrument"),
        role("tray", "current source container", kind="REGION", policy="SHARED"),
    ], relations=[relation("initial", "is_on", ["tool", "tray"])], operations=[
        operation("inspect", "identify components", ["tool", "tray"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="workshop")
    assert canonical["functional_relations"] == []
    assert canonical["current_state_relations"][0]["id"] == "initial"


def test_plain_is_on_with_explicit_placement_remains_task_effect():
    raw = document([
        role("payload", "personal refreshment payload", categories=["CUP"]),
        role("support", "personal support", kind="REGION", policy="SHARED"),
    ], relations=[relation("goal", "is_on", ["payload", "support"])], operations=[
        operation("place", "place payload", ["payload", "support"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert canonical["current_state_relations"] == []
    assert canonical["functional_relations"][0]["id"] == "goal"


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


def test_fixed_receiving_location_is_a_repair_target_family():
    raw = document([role("target", "fixed receiving location for component", kind="REGION", policy="SHARED")])
    hypothesis = build_role_type_hypotheses("workshop", convert_v3_to_canonical_document(raw, domain="workshop"))["target"]
    assert hypothesis.canonical_role_candidates == ("repair_target",)


def test_object_receiving_fastening_attachment_is_fixed_target_family():
    raw = document([role(
        "workpiece", "object receiving the fastening attachment",
        kind="FIXED_TARGET", policy="SHARED", categories=["ASSEMBLY"],
    )])
    hypothesis = build_role_type_hypotheses(
        "workshop", convert_v3_to_canonical_document(raw, domain="workshop")
    )["workpiece"]
    assert hypothesis.canonical_role_candidates == ("repair_target",)


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


def test_undeclared_observation_hint_is_removed_without_altering_task_contract():
    raw = document([role("payload", "refreshment payload", categories=["CUP"])])
    raw["observation_guidance"]["visible_candidates_per_role"] = {
        "payload": [], "undeclared_table": [{"label": "table", "visual_description": "visible table"}],
    }
    normalized, trace = normalize_and_validate_v3_contract(raw, domain="living_room")
    assert set(normalized["observation_guidance"]["visible_candidates_per_role"]) == {"payload"}
    assert any(row["code"] == "UNDECLARED_OBSERVATION_GUIDANCE_REMOVED" for row in trace)


def _living_context_roles(*, include_remote=True, include_extra=False):
    roles = [
        role("shared", "shared support accessible to both seats", kind="REGION", policy="SHARED"),
        role("seat_left", "left seating position", kind="FIXED_TARGET", policy="SHARED"),
        role("seat_right", "right seating position", kind="FIXED_TARGET", policy="SHARED"),
    ]
    if include_remote:
        roles.append(role("remote", "entertainment control device", categories=["REMOTE"]))
    if include_extra:
        roles.append(role("display", "television display context", kind="FIXED_TARGET", policy="SHARED"))
    return roles


def test_quantified_relation_builds_explicit_seating_pair_context():
    raw = document(
        _living_context_roles(include_remote=False),
        relations=[relation("access", "accessible to both", ["shared", "seat_left", "seat_right"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    edge = canonical["functional_relations"][0]
    assert edge["subject_role"] == "shared"
    assert edge["object_role"].startswith("fm_context_set__")
    context = canonical["explicit_context_sets"][0]
    assert context["runtime_role"] == "SEATING_PAIR"
    assert context["member_raw_roles"] == ["seat_left", "seat_right"]
    assert context["code"] == "EXPLICIT_CONTEXT_SET_CANONICALIZATION"


def test_two_explicit_per_seat_access_edges_conjoin_via_explicit_move_target():
    roles = _living_context_roles()
    for item in roles:
        if item["id"].startswith("seat_"):
            item["binding_policy"] = "DISTINCT"
            item["function"] = "seating reference"
    raw = document(
        roles,
        relations=[
            relation("left_access", "accessible to", ["remote", "seat_left"]),
            relation("right_access", "accessible to", ["remote", "seat_right"]),
        ],
        operations=[operation("move", "move to", ["remote", "shared"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert len(canonical["functional_relations"]) == 1
    edge = canonical["functional_relations"][0]
    assert edge["subject_role"] == "shared"
    assert edge["object_role"].startswith("fm_context_set__")
    trace = canonical["functional_constraint_interpretation"]
    conjunction = next(row for row in trace if row["code"] == "EXPLICIT_BINARY_RELATION_SET_CONJUNCTION")
    assert conjunction["source_relation_ids"] == ["left_access", "right_access"]
    graph = compile_candidate_graph("living_room", "move remote where both seats can access it", raw)
    assert graph.online_executable_contract_complete
    assert graph.operation_groups[0].capability_id == "SUPPORT_ENTERTAINMENT_CONTROL"
    accounting = {row["raw_id"]: row["disposition"] for row in canonical["fm_semantic_accounting"]}
    assert accounting["left_access"] == accounting["right_access"] == "GROUNDED_TASK_RELATION"


def test_single_per_seat_access_edge_does_not_invent_missing_pair_member():
    raw = document(
        _living_context_roles(),
        relations=[relation("left_access", "accessible to", ["remote", "seat_left"])],
        operations=[operation("move", "move to", ["remote", "shared"])],
    )
    with pytest.raises(TaskSpecificationValidationError, match="FM_INTERNAL_RELATION_PARTICIPANT_CONTRADICTION"):
        convert_v3_to_canonical_document(raw, domain="living_room")


def test_per_seat_access_edges_without_explicit_shared_move_target_do_not_create_pair():
    raw = document(
        _living_context_roles(),
        relations=[
            relation("left_access", "accessible to", ["remote", "seat_left"]),
            relation("right_access", "accessible to", ["remote", "seat_right"]),
        ],
    )
    with pytest.raises(TaskSpecificationValidationError, match="FM_INTERNAL_RELATION_PARTICIPANT_CONTRADICTION"):
        convert_v3_to_canonical_document(raw, domain="living_room")


def test_between_relation_builds_explicit_seating_pair_context():
    raw = document(
        _living_context_roles(include_remote=False),
        relations=[relation("between", "between", ["shared", "seat_left", "seat_right"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    edge = canonical["functional_relations"][0]
    assert edge["relation"] == "situated between"
    assert edge["subject_role"] == "shared"
    assert edge["object_role"].startswith("fm_context_set__")
    graph = compile_candidate_graph("living_room", "place shared support between both seats", raw)
    assert graph.task_causal_relations[0].predicate == "SITUATED_BETWEEN"
    assert graph.task_causal_relations[0].object_role == "SEATING_PAIR"


def test_quantified_relation_missing_member_fails_closed():
    raw = document(
        _living_context_roles(include_remote=False)[:2],
        relations=[relation("access", "accessible to both", ["shared", "seat_left"])],
    )
    with pytest.raises(TaskSpecificationValidationError, match="FM_INTERNAL_RELATION_PARTICIPANT_CONTRADICTION"):
        convert_v3_to_canonical_document(raw, domain="living_room")


def test_quantified_relation_never_discards_unrelated_extra_participant():
    raw = document(
        _living_context_roles(include_remote=False, include_extra=True),
        relations=[relation("access", "accessible to both", ["shared", "seat_left", "seat_right", "display"])],
    )
    with pytest.raises(TaskSpecificationValidationError, match="QUANTIFIED_RELATION_PARTICIPANT_CONTRADICTION"):
        convert_v3_to_canonical_document(raw, domain="living_room")


def test_seats_in_graph_without_quantified_text_do_not_create_pair():
    raw = document(_living_context_roles(include_remote=False))
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert not canonical.get("explicit_context_sets")
    assert all(r["id"] != "SEATING_PAIR" for r in canonical["functional_roles"])


def test_quantified_operation_uses_explicit_seating_pair_anchor_and_capability_preconditions():
    raw = document(
        _living_context_roles(),
        operations=[operation(
            "place_remote", "place entertainment control where accessible from both",
            ["remote", "shared", "seat_left", "seat_right"],
        )],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    group = canonical["interaction_groups"][0]
    assert len(group["v3_slot_assignments"]) == 1
    assert group["v3_slot_assignments"][0]["anchor_type"] == "SEATING_PAIR"
    graph = compile_candidate_graph("living_room", "place the remote for both seats", raw)
    compiled = graph.operation_groups[0]
    assert compiled.capability_id == "SUPPORT_ENTERTAINMENT_CONTROL"
    assert compiled.context_role == "SEATING_PAIR"
    assert {p[1] for p in compiled.physical_preconditions} == {"FITS_ON", "ACCESSIBLE_FROM_BOTH_SEATS"}
    assert graph.metadata["explicit_context_sets"][0]["member_raw_roles"] == ["seat_left", "seat_right"]
    assert graph.nodes["SEATING_PAIR"].role_resolution_status == "EXPLICIT_CONTEXT_SET_CANONICALIZATION"


def test_personal_placement_operation_supplies_robot_owned_preconditions():
    raw = document([
        role("payload", "personal refreshment setting", categories=["CUP", "PLATE"]),
        role("support", "personal support beside seat", kind="REGION", policy="SHARED"),
        role("seat", "seating position", kind="FIXED_TARGET", policy="SHARED"),
    ], operations=[operation("place", "place refreshment setting near seat", ["payload", "support", "seat"])])
    graph = compile_candidate_graph("living_room", "place the setting near the seat", raw)
    group = graph.operation_groups[0]
    assert group.capability_id == "SUPPORT_DRINKWARE"
    assert {p[1] for p in group.physical_preconditions} == {"FITS_SET_ON", "NEAR_SEAT"}
    assert all(row["provenance"] == "ROBOT_CAPABILITY_PRECONDITION" for row in group.preconditions_provenance)


def test_current_storage_context_is_elided_and_explicit_anchor_is_graph_joined():
    raw = document([
        role("storage", "current storage location", kind="REGION", policy="SHARED", categories=["TABLE"]),
        role("payload", "personal refreshment payload", categories=["CUP"]),
        role("support", "personal support", kind="REGION", policy="SHARED"),
        role("seat", "seating reference", kind="FIXED_TARGET", policy="SHARED"),
    ], relations=[relation("near", "near", ["support", "seat"])], operations=[
        operation("move", "transfer", ["storage", "payload", "support"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    group = canonical["interaction_groups"][0]
    assert group["v3_participant_roles"] == ["payload", "support", "seat"]
    assert group["v3_explicit_participant_roles"] == ["payload", "support", "seat"]
    assert group["v3_raw_fm_participant_roles"] == ["storage", "payload", "support"]
    codes = {row["code"] for row in canonical["functional_constraint_interpretation"]}
    assert codes == {"CURRENT_STATE_OPERATION_CONTEXT_ELIDED", "EXPLICIT_RELATION_OPERATION_CONTEXT_JOIN"}
    graph = compile_candidate_graph("living_room", "move refreshment from storage beside seat", raw)
    assert graph.operation_groups[0].capability_id == "SUPPORT_DRINKWARE"
    assert graph.operation_groups[0].context_role == "SEATING_POSITION"


def test_payload_near_seat_effect_joins_unique_explicit_placement_support():
    raw = document([
        role("payload", "personal refreshment payload", categories=["CUP"]),
        role("support", "personal support", kind="REGION", policy="SHARED"),
        role("seat", "seating reference", kind="FIXED_TARGET", policy="SHARED"),
    ], relations=[relation("near", "nearby", ["payload", "seat"])], operations=[
        operation("move", "place on", ["payload", "support"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    edge = canonical["functional_relations"][0]
    assert {edge["subject_role"], edge["object_role"]} == {"support", "seat"}
    normalized = next(
        row for row in canonical["functional_constraint_interpretation"]
        if row["code"] == "OPERATION_MEDIATED_RELATION_TARGET_NORMALIZATION"
    )
    assert normalized["raw_participant_roles"] == ["payload", "seat"]
    assert normalized["normalized_participant_roles"] == ["support", "seat"]
    graph = compile_candidate_graph("living_room", "place refreshment near the seat", raw)
    assert graph.online_executable_contract_complete
    assert graph.operation_groups[0].capability_id == "SUPPORT_DRINKWARE"


def test_explicit_relation_completes_missing_shared_operation_context():
    raw = document(
        _living_context_roles(),
        relations=[relation("between", "between", ["shared", "seat_left", "seat_right"])],
        operations=[operation("move", "move remote", ["remote", "shared"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    group = canonical["interaction_groups"][0]
    assert group["v3_slot_assignments"][0]["capability_id"] == "SUPPORT_ENTERTAINMENT_CONTROL"
    assert group["v3_slot_assignments"][0]["anchor_type"] == "SEATING_PAIR"


def test_missing_explicit_shared_context_is_not_invented():
    raw = document(_living_context_roles()[:1] + _living_context_roles()[3:], operations=[
        operation("move", "move remote", ["remote", "shared"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert canonical["interaction_groups"][0]["v3_slot_assignments"]
    assert canonical["interaction_groups"][0]["context_role"] is None
    assert not canonical.get("explicit_context_sets")
    graph = compile_candidate_graph("living_room", "move remote to shared support", raw)
    assert not graph.online_executable_contract_complete
    assert "SEATING_PAIR" not in graph.nodes


def test_beverage_macro_has_unique_two_transfer_lowering_and_accounting():
    raw = document([
        role("source_a", "source of coffee material", policy="SHARED"),
        role("source_b", "source of water", policy="SHARED"),
        role("container", "receiving vessel for coffee", count=2, policy="DISTINCT"),
    ], operations=[operation("prepare", "prepare coffee beverage", ["source_a", "source_b", "container"], count=2)])
    canonical = convert_v3_to_canonical_document(raw, domain="kitchen")
    assert len(canonical["interaction_groups"]) == 2
    assert {g["v3_slot_assignments"][0]["capability_id"] for g in canonical["interaction_groups"]} == {"TRANSFER_CONTENT_TO_CONTAINER"}
    assert canonical["fm_semantic_accounting"][0]["disposition"] == "DETERMINISTIC_COMPOSITE_OPERATION_LOWERING"


def test_fill_and_mix_macro_is_lowered_from_participant_functions():
    raw = document([
        role("source_a", "source of coffee material", policy="SHARED"),
        role("source_b", "source of water", policy="SHARED"),
        role("container", "receiving vessel for coffee"),
    ], operations=[operation("macro", "fill_and_mix", ["source_a", "source_b", "container"])])
    canonical = convert_v3_to_canonical_document(raw, domain="kitchen")
    assert len(canonical["interaction_groups"]) == 2
    assert all(group["v3_slot_assignments"][0]["capability_id"] == "TRANSFER_CONTENT_TO_CONTAINER"
               for group in canonical["interaction_groups"])


def test_object_used_only_as_current_state_container_is_context_not_groundable_role():
    raw = document([
        role("tool", "fastening instrument"),
        role("tray", "source container", categories=["TRAY"]),
    ], relations=[relation("initial", "is_on", ["tool", "tray"])], operations=[
        operation("inspect", "identify components", ["tool", "tray"]),
    ])
    graph = compile_candidate_graph("workshop", "identify components", raw)
    assert "tray" not in graph.nodes
    assert graph.metadata["current_state_operation_context_roles"] == ["tray"]


def test_beverage_macro_with_incompatible_participant_does_not_lower():
    raw = document([
        role("source", "source of coffee material", policy="SHARED"),
        role("utensil", "soup eating utensil"),
        role("container", "receiving vessel for coffee", count=2, policy="DISTINCT"),
    ], operations=[operation("prepare", "prepare coffee beverage", ["source", "utensil", "container"], count=2)])
    canonical = convert_v3_to_canonical_document(raw, domain="kitchen")
    assert len(canonical["interaction_groups"]) == 1
    assert canonical["interaction_groups"][0]["v3_slot_assignments"] == []
    assert canonical["fm_semantic_accounting"][0]["disposition"] == "UNRESOLVED_REQUIRED_OPERATION"


def test_every_raw_fm_element_has_semantic_accounting_disposition():
    raw = document([
        role("payload", "personal refreshment payload", categories=["CUP"]),
        role("support", "personal support", kind="REGION", policy="SHARED"),
    ], relations=[relation("on", "on", ["payload", "support"])], operations=[
        operation("move", "move", ["payload", "support"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    keys = {(row["element_kind"], row["raw_id"]) for row in canonical["fm_semantic_accounting"]}
    assert keys == {("role", "payload"), ("role", "support"), ("relation", "on"), ("operation", "move")}


def test_unique_multi_channel_remote_evidence_overrides_isolated_bad_function():
    raw = document([
        role("remote_control", "joining_component", categories=["electronic", "controller"],
             description="entertainment control device"),
        role("shared", "shared central support", kind="REGION", policy="SHARED"),
    ], relations=[relation("on", "on", ["remote_control", "shared"])], operations=[
        operation("move", "move entertainment control", ["remote_control", "shared"]),
    ])
    hypothesis = build_role_type_hypotheses(
        "living_room", convert_v3_to_canonical_document(raw, domain="living_room")
    )["remote_control"]
    assert hypothesis.canonical_role_candidates == ("REMOTE",)
    assert hypothesis.status == "GLOBAL_GRAPH_CONSISTENCY_OVERRIDE"
    assert any(row.get("status") == "UNIQUE_MINIMAL_REPAIR" for row in hypothesis.evidence)


def test_explicit_raw_personal_pairings_survive_count_consolidation():
    raw = document([
        role("payload_1", "personal refreshment payload", categories=["CUP"]),
        role("payload_2", "personal refreshment payload", categories=["CUP"]),
        role("support_1", "personal support", kind="REGION", policy="SHARED"),
        role("support_2", "personal support", kind="REGION", policy="SHARED"),
        role("seat_1", "seating position", kind="FIXED_TARGET", policy="SHARED"),
        role("seat_2", "seating position", kind="FIXED_TARGET", policy="SHARED"),
    ], operations=[
        operation("pair_1", "place refreshment setting near seat", ["payload_1", "support_1", "seat_1"]),
        operation("pair_2", "place refreshment setting near seat", ["payload_2", "support_2", "seat_2"]),
    ])
    graph = compile_candidate_graph("living_room", "place each setting by its seat", raw)
    assert graph.nodes["CUP_SAUCER_SET"].count == 2
    assert graph.nodes["PERSONAL_CUP_SAUCER_REGION"].count == 2
    assert graph.nodes["SEATING_POSITION"].count == 2
    assert graph.nodes["SEATING_POSITION"].binding_policy == "DISTINCT"
    pairings = graph.metadata["explicit_role_pairings"]
    assert [row["raw_participant_roles"] for row in pairings] == [
        ["payload_1", "support_1", "seat_1"],
        ["payload_2", "support_2", "seat_2"],
    ]
    assert len(graph.operation_groups) == 1
    assert graph.operation_groups[0].required_target_count == 2
    assert any(row.get("status") == "CONSOLIDATED_EQUIVALENT_OPERATION_INSTANCES"
               for row in graph.metadata["canonicalization_trace"]["groups"])


def test_kitchen_operation_rejects_meal_utensil_as_beverage_stirrer():
    invalid = document([
        role("container", "receiving vessel for beverage"),
        role("meal_utensil", "eating utensil for consuming soup", categories=["SPOON"]),
    ], operations=[operation("stir", "stir beverage", ["meal_utensil", "container"])])
    with pytest.raises(TaskSpecificationValidationError, match="MISSING_OR_CONTRADICTORY_OPERATION_PARTICIPANTS"):
        normalize_and_validate_v3_contract(invalid, domain="kitchen")

    valid = document([
        role("container", "receiving vessel for coffee beverage"),
        role("stirrer", "instrument for stirring coffee beverage", categories=["SPOON"]),
    ], operations=[operation("stir", "stir beverage", ["stirrer", "container"])])
    canonical = convert_v3_to_canonical_document(valid, domain="kitchen")
    assert canonical["interaction_groups"][0]["v3_slot_assignments"][0]["capability_id"] == "STIR_COFFEE"
