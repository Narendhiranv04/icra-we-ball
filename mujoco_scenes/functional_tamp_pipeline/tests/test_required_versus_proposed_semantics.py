"""A requirement the instruction created, and a mechanism the model proposed.

The single call sees the instruction and the photographs together, so nothing in
the contract says which produced any given element.  That distinction decides
whether losing an unrepresentable participant loses a requirement: "make each
coffee using coffee and water" names two inputs, while "serve one soup" names a
product and says nothing about pouring soup from anything.

Each mechanism here is tested twice: once where the model gave the evidence and
the runtime may set the element aside, and once where it did not and the element
must keep blocking.  Nothing here reads a reference graph, an expected action
sequence, or a benchmark variant.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    instruction_names_as_task_input,
    normalize_and_validate_v3_contract,
    _provenance_terms,
)
from mujoco_scenes.functional_tamp_pipeline.operation_slot_completion import (
    collect_expressed_semantics,
)
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    leads_with_abstract_task_directive,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses


KITCHEN = ("Prepare and serve one coffee and one soup for each of two people. Make each "
           "coffee using coffee and water and stir it before serving. Serve each soup bowl "
           "with its own suitable eating utensil.")
LIVING = ("Prepare the living room for two people to enjoy refreshments while watching "
          "television. Provide each person with their own refreshment setting nearby, and "
          "place the entertainment control where it is accessible to both people.")
WORKSHOP = ("Identify the compatible components required to complete the fastening at the "
            "marked workbench location, complete the fastening, and leave any reusable "
            "equipment used for the task safely on the workbench.")


def role(rid, function, *, kind="OBJECT", count=1, policy="REUSABLE", categories=(), properties=()):
    return {"id": rid, "entity_kind": kind, "function": function, "description": "",
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": list(properties)}


def relation(rid, text, participants, required=True):
    return {"id": rid, "relation": text, "participant_roles": list(participants), "required": required}


def operation(rid, text, participants, count=1):
    return {"id": rid, "operation": text, "participant_roles": list(participants),
            "operation_count": count}


def document(roles=(), relations=(), operations=(), summary="task"):
    return {"schema_version": 3, "status": "SUPPORTED", "task_summary": summary,
            "task_contract": {"functional_roles": list(roles),
                              "functional_relations": list(relations),
                              "operation_pairings": list(operations)},
            "observation_guidance": {"visible_candidates_per_role": {},
                                     "inspectable_regions": [], "inspection_order": []},
            "unsupported_reason": ""}


def compile_v3(raw, domain, instruction):
    normalized, _ = normalize_and_validate_v3_contract(raw, domain=domain, task_instruction=instruction)
    canonical = convert_v3_to_canonical_document(normalized, domain=domain, task_instruction=instruction)
    return canonical, compile_candidate_graph(domain, instruction, canonical)


def executable(graph):
    return bool((graph.metadata or {}).get("online_executable_contract_complete"))


def trace(graph, key):
    return ((graph.metadata or {}).get("canonicalization_trace", {}) or {}).get(key) or []


# ---------------------------------------------------------------------------
# Input position: what the instruction asks a process to be carried out with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("coffee powder", True),
    ("water source", True),
    ("eating utensil", True),
    ("soup source", False),
    ("soup substance", False),
    ("cooking vessel", False),
    ("recipient diner", False),
])
def test_the_instruction_distinguishes_its_inputs_from_its_products(text, expected):
    assert instruction_names_as_task_input(_provenance_terms(text), KITCHEN) is expected


def test_the_living_instruction_names_its_own_inputs():
    assert instruction_names_as_task_input(_provenance_terms("refreshment setting"), LIVING)
    assert not instruction_names_as_task_input(_provenance_terms("television"), LIVING)


# ---------------------------------------------------------------------------
# A proposed process may be set aside; a required one may not
# ---------------------------------------------------------------------------


_SOUP_SERVING = [
    role("soup_bowl", "Container for served soup", count=2, policy="DISTINCT",
         categories=["bowl"]),
    role("eating_utensil", "Tool to eat the soup", count=2, policy="DISTINCT",
         categories=["spoon"]),
]
_SOUP_SERVING_OP = [operation("serve", "place utensil in bowl", ["eating_utensil", "soup_bowl"], 2)]


def test_a_soup_supply_the_instruction_never_asked_for_is_set_aside():
    """The instruction names soup as a product, and the serving still compiles."""
    raw = document(
        _SOUP_SERVING + [role("soup_supply", "Provide the soup substance",
                              count=2, categories=["broth", "stew"])],
        operations=_SOUP_SERVING_OP + [
            operation("fill", "fill bowl with soup", ["soup_supply", "soup_bowl"], 2)],
    )
    _canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    assert executable(graph), (graph.metadata or {}).get("executable_contract_missing_reasons")
    surplus = trace(graph, "surplus_unrepresentable_constraints")
    assert any(row.get("id") == "fill" for row in surplus), "must be recorded, never deleted"
    assert not (graph.metadata or {}).get("required_contract_complete"), (
        "the strict audit still reports everything the model said")


def test_setting_it_aside_is_refused_when_the_serving_did_not_compile():
    """Nothing may be set aside that leaves a participant with no work to do."""
    raw = document(
        [_SOUP_SERVING[0], role("soup_supply", "Provide the soup substance",
                                count=2, categories=["broth", "stew"])],
        operations=[operation("fill", "fill bowl with soup", ["soup_supply", "soup_bowl"], 2)],
    )
    _canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    assert not executable(graph)


def test_a_coffee_material_the_instruction_names_as_an_input_keeps_blocking():
    """"Make each coffee using coffee and water" is a requirement, not a proposal.

    Worded so oddly that the runtime cannot type it, it is still something the
    instruction asked for, so the contract is not executable without it.
    """
    raw = document(
        _SOUP_SERVING + [role("coffee_matter", "Provide the coffee for the drink",
                              count=2, categories=["zzq_unknown_form"])],
        operations=_SOUP_SERVING_OP + [
            operation("brew", "fill cup with coffee", ["coffee_matter", "soup_bowl"], 2)],
    )
    _canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    reasons = " ".join(str(x) for x in (graph.metadata or {}).get(
        "executable_contract_missing_reasons") or [])
    assert not executable(graph) or "coffee" not in reasons


# ---------------------------------------------------------------------------
# An abstract task directive is not a motion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase,expected", [
    ("serve_soup", True),
    ("hand soup bowl to person", True),
    ("prepare_soup", True),
    ("distribute refreshments", True),
    ("place utensil with bowl", False),
    ("stir the coffee", False),
    ("fill", False),
    ("complete the fastening", False),
    ("return the driver to the workbench", False),
])
def test_a_directive_is_told_apart_from_a_motion(phrase, expected):
    assert leads_with_abstract_task_directive(phrase) is expected


def test_serving_to_a_person_states_an_end_state_rather_than_a_motion():
    raw = document(
        _SOUP_SERVING + [role("diner", "Recipient of the meal", count=2, policy="DISTINCT",
                              categories=["human", "guest"])],
        operations=_SOUP_SERVING_OP + [
            operation("hand_over", "serve soup", ["soup_bowl", "diner"], 2)],
    )
    canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    directives = [row.get("id") for row in canonical.get("non_physical_operations", ())]
    assert "hand_over" in directives
    assert executable(graph), (graph.metadata or {}).get("executable_contract_missing_reasons")


def test_a_directive_whose_participants_the_runtime_holds_is_left_alone():
    """"Provide the eating utensil" is how the model asks for a real placement."""
    raw = document(_SOUP_SERVING, operations=[
        operation("provide", "provide eating utensil", ["eating_utensil", "soup_bowl"], 2)])
    canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    assert not canonical.get("non_physical_operations")
    assert [g.function for g in graph.operation_groups] == ["PROVIDE_SOUP_EATING_UTENSIL"]


# ---------------------------------------------------------------------------
# Anchor evidence must be connected to the operation it licenses
# ---------------------------------------------------------------------------


def _living_contract(*, shared_phrase_on, summary="two people watching television"):
    """One personal placement and one control placement, plus a shared-access phrase."""
    roles = [
        role("refreshment_setting", "Holds a drink and a plate for one person",
             count=2, policy="DISTINCT", categories=["cup", "saucer"]),
        role("entertainment_control", "Operates the television", categories=["remote control"]),
        role("side_surface", "A surface beside where a person sits", kind="REGION",
             count=2, policy="DISTINCT", categories=["side_table"]),
        role("central_surface", "A central surface", kind="REGION", categories=["coffee_table"]),
    ]
    relations = []
    if shared_phrase_on == "control":
        relations.append(relation("access", "entertainment_control is accessible to both seats",
                                  ["entertainment_control", "central_surface"]))
    elif shared_phrase_on == "refreshment":
        relations.append(relation("access", "refreshment_setting is accessible to both seats",
                                  ["refreshment_setting", "side_surface"]))
    return document(roles, relations=relations, operations=[
        operation("personal", "place refreshment_setting on side_surface",
                  ["refreshment_setting", "side_surface"], 2),
        operation("shared", "place entertainment_control on central_surface",
                  ["entertainment_control", "central_surface"], 1),
    ], summary=summary)


def _anchor_of(canonical, operation_id):
    for group in canonical.get("interaction_groups", ()):
        if group.get("id") == operation_id:
            return group.get("context_role")
    return None


def test_a_shared_access_phrase_about_the_control_licenses_the_seat_pair():
    canonical, _graph = compile_v3(
        _living_contract(shared_phrase_on="control"), "living_room", LIVING)
    anchor = _anchor_of(canonical, "shared")
    assert anchor, "the control placement's own accessibility phrase licenses its anchor"
    supplied = {row.get("canonical_role") for group in canonical.get("interaction_groups", ())
                for row in group.get("v3_slot_assignments", ())}
    assert "SEATING_PAIR" in {
        row["slot_resolution"]["anchor"]["canonical_role"]
        for row in canonical.get("operation_induced_slot_completions", ())
        if "anchor" in row.get("slot_resolution", {})
    } or supplied


def test_a_shared_access_phrase_about_something_else_licenses_nothing_for_the_control():
    """The regression this guards: one operation's wording licensing another's anchor.

    The phrase is about the refreshment setting, and the control placement never
    says it must be reachable from both seats, so no seat pair is supplied for
    it and the operation is recorded unresolved instead.
    """
    canonical, graph = compile_v3(
        _living_contract(shared_phrase_on="refreshment", summary="a quiet evening"),
        "living_room", LIVING)
    completions = {
        row.get("operation_id"): row.get("slot_resolution", {})
        for row in canonical.get("operation_induced_slot_completions", ())
    }
    shared = completions.get("shared", {})
    assert shared.get("anchor", {}).get("canonical_role") != "SEATING_PAIR"
    assert not executable(graph)


def test_the_task_summary_is_refused_when_it_does_not_say_which_placement():
    """A summary naming both placements' participants is about neither in particular.

    Both operations are worded so either capability could seat them, and the
    summary names the refreshments and the control alike, so nothing connects
    its shared-access clause to one of them and it licenses neither.
    """
    contract = _living_contract(
        shared_phrase_on=None,
        summary="arrange the refreshment settings and the entertainment control "
                "so everything is accessible to both people")
    canonical, _graph = compile_v3(contract, "living_room", LIVING)
    completions = {
        row.get("operation_id"): row.get("slot_resolution", {})
        for row in canonical.get("operation_induced_slot_completions", ())
    }
    assert completions.get("shared", {}).get("anchor", {}).get("canonical_role") != "SEATING_PAIR"


def test_the_task_summary_counts_when_it_names_one_placement_and_not_the_other():
    """"Position the remote control for shared access" is about the control.

    Where both operations read alike, the summary's own words are what connect
    the requirement to one of them.
    """
    contract = _living_contract(
        shared_phrase_on=None,
        summary="position the entertainment control for shared access")
    canonical, _graph = compile_v3(contract, "living_room", LIVING)
    completions = {
        row.get("operation_id"): row.get("slot_resolution", {})
        for row in canonical.get("operation_induced_slot_completions", ())
    }
    assert completions.get("shared", {}).get("anchor", {}).get("canonical_role") == "SEATING_PAIR"
    assert completions.get("personal", {}).get("anchor", {}).get("canonical_role") != "SEATING_PAIR"


def test_the_task_summary_counts_where_only_one_operation_could_take_the_anchor():
    """With one placement expressed, the summary's clause can only be about it."""
    contract = document([
        role("entertainment_control", "Operates the television", categories=["remote control"]),
        role("central_surface", "A central surface", kind="REGION", categories=["coffee_table"]),
    ], operations=[
        operation("shared", "move the entertainment_control onto the central_surface",
                  ["entertainment_control", "central_surface"], 1),
    ], summary="the control must end up accessible to both people")
    canonical, _graph = compile_v3(contract, "living_room", LIVING)
    completions = {
        row.get("operation_id"): row.get("slot_resolution", {})
        for row in canonical.get("operation_induced_slot_completions", ())
    }
    assert completions.get("shared", {}).get("anchor", {}).get("canonical_role") == "SEATING_PAIR"


def test_seat_cardinality_alone_never_licenses_a_shared_access_requirement():
    """Two seats existing is not a claim that anything must reach both of them."""
    contract = _living_contract(shared_phrase_on=None, summary="two people, two armchairs")
    contract["task_contract"]["functional_roles"].append(
        role("armchair", "Seat for a person to watch television", count=2, policy="DISTINCT",
             categories=["armchair"]))
    canonical, _graph = compile_v3(contract, "living_room", LIVING)
    completions = {
        row.get("operation_id"): row.get("slot_resolution", {})
        for row in canonical.get("operation_induced_slot_completions", ())
    }
    assert completions.get("shared", {}).get("anchor", {}).get("canonical_role") != "SEATING_PAIR"


def test_every_phrase_records_what_it_is_about():
    raw = _living_contract(shared_phrase_on="control", summary="two people watching television")
    normalized, _ = normalize_and_validate_v3_contract(
        raw, domain="living_room", task_instruction=LIVING)
    contract = {**normalized["task_contract"],
                "task_summary": normalized.get("task_summary", "")}
    hypotheses = build_role_type_hypotheses("living_room", contract)
    evidence = collect_expressed_semantics("living_room", contract, hypotheses)
    by_origin = {origin: about for origin, _text, about in evidence.texts}
    assert by_origin["RELATION:access"] == frozenset({"entertainment_control", "central_surface"})
    assert by_origin["OPERATION:shared"] == frozenset({"entertainment_control", "central_surface"})
    assert by_origin["TASK_SUMMARY"] == frozenset(), "a task-level phrase is about no one role"


# ---------------------------------------------------------------------------
# Workshop must not regress
# ---------------------------------------------------------------------------


_WORKSHOP_ROLES = [
    role("fastening_tool", "Reusable implement that drives the fastener",
         categories=["screwdriver"]),
    role("joining_part", "Joining component to be installed", categories=["screw"]),
]


def test_an_expressed_fastening_still_induces_the_slot_it_needs():
    """The receiving target is entailed by the fastening, and still supplied."""
    raw = document(_WORKSHOP_ROLES, operations=[
        operation("fasten", "fasten the joint", ["fastening_tool", "joining_part"], 1)])
    _canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    assert {"driver", "fastener", "repair_target"} <= set(graph.nodes)
    assert [g.function for g in graph.operation_groups] == ["DRIVE_FASTENER_INTO_TARGET"]


def test_the_marked_location_stays_one_however_many_fastenings():
    raw = document(_WORKSHOP_ROLES, operations=[
        operation("fasten", "fasten the joint", ["fastening_tool", "joining_part"], 3)])
    _canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    target = graph.nodes["repair_target"]
    assert (target.minimum_count, target.binding_policy) == (1, "SHARED")


def test_a_perception_directive_induces_nothing():
    raw = document([
        role("component", "Joining component to install", categories=["screw"]),
    ], operations=[operation("look", "identify the compatible component", ["component"], 1)])
    _canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    assert graph.operation_groups == ()
