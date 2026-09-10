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
    # The support resolved from its own wording, and the armchair its description
    # mentions is recorded as a neighbour rather than as a competing claim.
    assert "PURPOSE_TEXT" in p.evidence[0]["source"]
    assert p.evidence[0]["family_precedence_rules"], "a precedence rule settled support vs seating"
    assert any(
        rule.endswith(("OVER_AMBIGUOUS_SEATING_CATEGORY", "RELATIVE_TO_SEATING"))
        for rule in p.evidence[0]["family_precedence_rules"]
    )


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
    # Two roles the model described identically are each plausible in the same
    # slot, so the operation has more than one legal reading.
    # Two participants the model described in no way that distinguishes them
    # are each plausible in either remaining slot, so the fastening has more
    # than one legal reading.
    raw = document([
        role("implement", "reusable fastening implement", categories=["screwdriver"]),
        role("thing_one", "component of the assembly"),
        role("thing_two", "component of the assembly"),
    ], operations=[operation("x", "fasten joint", ["thing_one", "thing_two", "implement"])])
    canonical = convert_v3_to_canonical_document(raw, domain="workshop")
    group = canonical["interaction_groups"][0]
    assert len(group["v3_slot_assignments"]) > 1
    # Every assignment travels with the group, so nothing is chosen by position.
    # A slot is filled only where all assignments agree on it; a slot they
    # disagree about stays open for grounding to settle.
    for field, key in (("tool_role", "source_role"), ("target_role", "target_role"),
                       ("context_role", "anchor_role")):
        distinct = {row[key] for row in group["v3_slot_assignments"]}
        assert group[field] == (next(iter(distinct)) if len(distinct) == 1 else None)


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
def test_exact_duplicate_participant_is_repaired_and_recorded(kind):
    """A repeated participant says nothing twice, so the repetition is removed.

    Here nothing is left to state a relation with, so the element is carried out
    of the contract into the runtime's own accounting rather than dropped: the
    stage that decides whether required semantics survived still sees it.
    """
    raw = document([role("tool", "tool")],
                   relations=[relation("x", "compatible", ["tool", "tool"])] if kind == "relation" else (),
                   operations=[operation("x", "move", ["tool", "tool"])] if kind == "operation" else ())
    normalized, trace = normalize_and_validate_v3_contract(raw)
    assert any(row["code"] == "DUPLICATE_PARTICIPANT_DEDUPLICATED" for row in trace)
    assert any(row["code"] == "STRUCTURALLY_UNUSABLE_ELEMENT_CARRIED_OUT" for row in trace)
    carried = normalized["structurally_unusable_elements"]
    assert [item["id"] for item in carried] == ["x"]
    assert carried[0]["reason"] == "TOO_FEW_DISTINCT_PARTICIPANTS_TO_STATE_A_RELATION"


@pytest.mark.parametrize("kind", ["relation", "operation"])
def test_deduplicated_element_keeps_its_remaining_participants(kind):
    """Deduplication leaves a usable statement when enough participants remain."""
    roles = [role("driver", "reusable fastening implement"),
             role("fastener", "manipulated joining component")]
    parts = ["driver", "fastener", "fastener"]
    raw = document(roles,
                   relations=[relation("x", "compatible with", parts)] if kind == "relation" else (),
                   operations=[operation("x", "fasten joint", parts)] if kind == "operation" else ())
    normalized, trace = normalize_and_validate_v3_contract(raw, domain="workshop")
    assert any(row["code"] == "DUPLICATE_PARTICIPANT_DEDUPLICATED" for row in trace)
    assert "structurally_unusable_elements" not in normalized
    collection = "functional_relations" if kind == "relation" else "operation_pairings"
    assert normalized["task_contract"][collection][0]["participant_roles"] == ["driver", "fastener"]


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


def _no_pair_invented(canonical):
    """No synthetic context-set role and no explicit context set were created."""
    invented = [r["id"] for r in canonical["functional_roles"]
                if str(r["id"]).startswith("fm_context_set__")]
    return not invented and not canonical.get("explicit_context_sets")


def test_single_seat_and_no_shared_wording_does_not_invent_a_seat_pair():
    """Reaching one named seat is not a claim about reaching both of them."""
    raw = document(
        [role("remote", "entertainment control device", categories=["REMOTE"]),
         role("low_table", "low table", kind="REGION", policy="REUSABLE", categories=["TABLE"]),
         role("seat_left", "left seating position", kind="FIXED_TARGET", policy="REUSABLE")],
        relations=[relation("left_access", "accessible to", ["remote", "seat_left"])],
        operations=[operation("move", "move to", ["remote", "low_table"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert _no_pair_invented(canonical)
    assert not canonical.get("operation_induced_slot_completions")
    graph = compile_candidate_graph("living_room", "move the remote", raw)
    assert "SEATING_PAIR" not in graph.nodes
    assert not graph.online_executable_contract_complete


def test_shared_access_wording_with_one_named_seat_realizes_the_pair():
    """The model said "accessible from both seats"; that is the requirement.

    Which seats those are is the scene's business, and the runtime holds them as
    its registered pair.  The wording is what licenses it -- the previous test
    shows the same graph refusing without it.
    """
    raw = document(
        [role("remote", "entertainment control device", categories=["REMOTE"]),
         role("low_table", "low table", kind="REGION", policy="REUSABLE", categories=["TABLE"]),
         role("seat_left", "left seating position", kind="FIXED_TARGET", policy="REUSABLE")],
        relations=[relation("both_access", "accessible from both seats", ["remote", "seat_left"])],
        operations=[operation("move", "move to", ["remote", "low_table"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    completion = canonical["operation_induced_slot_completions"][0]
    assert completion["slot_resolution"]["anchor"]["canonical_role"] == "SEATING_PAIR"
    assert completion["fm_semantic_witnesses"]["anchor"]


def test_per_seat_access_edges_without_explicit_shared_move_target_do_not_create_pair():
    """Two per-seat edges with no shared destination still do not form a pair."""
    raw = document(
        _living_context_roles(),
        relations=[
            relation("left_access", "accessible to", ["remote", "seat_left"]),
            relation("right_access", "accessible to", ["remote", "seat_right"]),
        ],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert _no_pair_invented(canonical)
    assert {u["id"] for u in canonical["unresolved_relation_semantics"]} == {"left_access", "right_access"}


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
    """A "both" relation naming one member contributes nothing and invents nothing.

    It fails closed in the sense that matters: no pair is synthesised and no
    relation is emitted from it. It is recorded as an unresolved required
    semantic rather than aborting the whole contract.
    """
    raw = document(
        _living_context_roles(include_remote=False)[:2],
        relations=[relation("access", "accessible to both", ["shared", "seat_left"])],
    )
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert _no_pair_invented(canonical)
    assert canonical["functional_relations"] == []
    assert [u["id"] for u in canonical["unresolved_relation_semantics"]] == ["access"]
    accounting = {row["raw_id"]: row["disposition"] for row in canonical["fm_semantic_accounting"]}
    assert accounting["access"] == "UNRESOLVED_REQUIRED_SEMANTIC"


def test_quantified_relation_never_discards_unrelated_extra_participant():
    """An extra participant is not quietly dropped to make a phrase fit.

    The seat-set adapter declines the relation rather than assigning three of
    its four participants, and the relation is then recorded -- previously the
    conversion aborted, taking every other coherent semantic with it.
    """
    raw = document(
        _living_context_roles(include_remote=False, include_extra=True),
        relations=[relation("access", "accessible to both", ["shared", "seat_left", "seat_right", "display"])],
    )
    instruction = (
        "place the entertainment control where it is accessible to both people"
    )
    canonical = convert_v3_to_canonical_document(
        raw, domain="living_room", task_instruction=instruction)
    assert not canonical.get("explicit_context_sets")
    accounted = {
        row["raw_id"]: row["disposition"] for row in canonical["fm_semantic_accounting"]
        if row["element_kind"] == "relation"
    }
    assert accounted["access"] != "GROUNDED_TASK_RELATION"
    graph = compile_candidate_graph("living_room", instruction, canonical)
    assert not graph.online_executable_contract_complete


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


def test_shared_access_stated_anywhere_realizes_the_seating_pair_anchor():
    """The support's own wording states the requirement, so the anchor stands for it."""
    raw = document(_living_context_roles()[:1] + _living_context_roles()[3:], operations=[
        operation("move", "move remote", ["remote", "shared"]),
    ])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    group = canonical["interaction_groups"][0]
    assert group["v3_slot_assignments"]
    completion = canonical["operation_induced_slot_completions"][0]
    assert completion["slot_resolution"]["anchor"]["canonical_role"] == "SEATING_PAIR"
    assert completion["fm_semantic_witnesses"]["anchor"], "the FM said it, and the trace says where"
    graph = compile_candidate_graph("living_room", "move remote to shared support", raw)
    assert "SEATING_PAIR" in graph.nodes


def test_missing_shared_access_semantics_is_not_invented():
    """Placing something on a surface does not by itself demand shared access."""
    raw = document([
        role("remote", "entertainment control device", categories=["REMOTE"]),
        role("table", "low table", kind="REGION", policy="SHARED", categories=["TABLE"]),
    ], operations=[operation("move", "move remote", ["remote", "table"])])
    canonical = convert_v3_to_canonical_document(raw, domain="living_room")
    assert not canonical.get("operation_induced_slot_completions")
    assert canonical["interaction_groups"] == []
    assert canonical["unresolved_operation_semantics"]
    graph = compile_candidate_graph("living_room", "move remote to table", raw)
    assert not graph.online_executable_contract_complete
    assert "SEATING_PAIR" not in graph.nodes
    assert "SEATING_POSITION" not in graph.nodes


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
    # No lowering, and no group either: an operation the runtime can neither
    # seat nor decompose is recorded as unrepresented rather than emitted as a
    # group with empty slots for a later stage to discover.
    assert canonical["interaction_groups"] == []
    assert canonical["fm_semantic_accounting"][0]["disposition"] == "UNRESOLVED_REQUIRED_SEMANTIC"
    assert [item["id"] for item in canonical["unresolved_operation_semantics"]] == ["prepare"]


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
    # The stray function phrase names no family this domain realizes, while the
    # categories and the description independently name the remote.  The
    # resolution therefore rests on more of the role than its function sentence.
    sources = {row.get("source") or "" for row in hypothesis.evidence}
    assert any("IDENTITY_TEXT" in source and "PURPOSE_TEXT" in source for source in sources)


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
    # A soup eating utensil is still refused as a beverage stirrer: no group is
    # created for the operation and the contract does not complete.  What changed
    # is that this is an unexecutable operation rather than a malformed task.
    normalized, _ = normalize_and_validate_v3_contract(invalid, domain="kitchen")
    canonical = convert_v3_to_canonical_document(
        normalized, domain="kitchen", task_instruction="stir the beverage")
    assert canonical["interaction_groups"] == []
    assert [u["id"] for u in canonical["unresolved_operation_semantics"]] == ["stir"]
    accounting = {row["raw_id"]: row["disposition"] for row in canonical["fm_semantic_accounting"]}
    assert accounting["stir"] == "UNRESOLVED_REQUIRED_SEMANTIC"

    valid = document([
        role("container", "receiving vessel for coffee beverage"),
        role("stirrer", "instrument for stirring coffee beverage", categories=["SPOON"]),
    ], operations=[operation("stir", "stir beverage", ["stirrer", "container"])])
    canonical = convert_v3_to_canonical_document(valid, domain="kitchen")
    assert canonical["interaction_groups"][0]["v3_slot_assignments"][0]["capability_id"] == "STIR_COFFEE"
