"""Information has to be able to flow back from operations to role typing.

Roles, relations and operations were solved as one-way stages, and each of the
failures below is a place where a later stage knew better and could not say so:
an alias table calling a manipulated target a piece of planner furniture, a
participant one operation set aside disappearing from the next one, a wording
whose reading the endpoints refuse discarding an operation the participants
identify exactly.

Each mechanism is tested twice -- once where the evidence licenses it, once
where it does not and it must refuse.  Nothing here reads a reference graph, an
expected action sequence, or a benchmark variant.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    compile_candidate_graph,
    resolve_role_type_hypotheses,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses


WORKSHOP = ("Identify the compatible components required to complete the fastening at the "
            "marked workbench location, complete the fastening, and leave any reusable "
            "equipment used for the task safely on the workbench.")
KITCHEN = ("Prepare and serve one coffee and one soup for each of two people. Make each "
           "coffee using coffee and water and stir it before serving. Serve each soup bowl "
           "with its own suitable eating utensil.")


def role(rid, function, *, kind="OBJECT", count=1, policy="REUSABLE", categories=(), description=""):
    return {"id": rid, "entity_kind": kind, "function": function, "description": description,
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": []}


def relation(rid, text, participants, required=True):
    return {"id": rid, "relation": text, "participant_roles": list(participants),
            "required": required}


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


def trace_of(graph):
    return (graph.metadata or {})["canonicalization_trace"]


# ---------------------------------------------------------------------------
# The alias table does not outrank a role type the graph already settled
# ---------------------------------------------------------------------------


def test_a_site_the_operation_anchors_is_not_rewritten_to_planner_furniture():
    raw = document([
        role("fastening_tool", "instrument used to drive the fastener",
             categories=["screwdriver"]),
        role("attachment_component", "the part to be secured at the marked place",
             categories=["screw"], policy="DISTINCT"),
        role("marked_fastening_site", "the marked place on the bench where the fastening must occur",
             kind="REGION", policy="SHARED", categories=["marked spot"]),
    ], operations=[operation(
        "assemble", "fasten the component at the marked site",
        ["fastening_tool", "attachment_component", "marked_fastening_site"])])
    canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    hypotheses = resolve_role_type_hypotheses("workshop", canonical)
    assert hypotheses["marked_fastening_site"].canonical_role_candidates == ("repair_target",)
    assert "repair_target" in graph.nodes, sorted(graph.nodes)
    assert graph.operation_groups, trace_of(graph)["disabled_groups"]


def test_a_bench_no_operation_acts_on_is_still_planner_context():
    """The adversarial half: nothing here makes the bench a fastening site."""
    raw = document([
        role("fastening_tool", "instrument used to drive the fastener",
             categories=["screwdriver"]),
        role("attachment_component", "the part to be secured at the joint",
             categories=["screw"], policy="DISTINCT"),
        role("bench", "flat surface the work happens above", kind="REGION",
             policy="SHARED", categories=["workbench"]),
    ], operations=[operation(
        "assemble", "fasten the component into the joint",
        ["fastening_tool", "attachment_component"])])
    _, graph = compile_v3(raw, "workshop", WORKSHOP)
    assert "MAIN_WORKBENCH_ZONE" not in graph.nodes


# ---------------------------------------------------------------------------
# One role stating two functions keeps both readings
# ---------------------------------------------------------------------------


def test_a_surface_that_states_a_tool_return_keeps_its_support_reading():
    document_with_both = {"functional_roles": [
        role("target_location", "specific area on the workbench requiring fastening",
             kind="REGION", policy="DISTINCT", categories=["workbench surface"]),
        role("workbench", "surface where the fastening occurs and the tool is returned",
             kind="REGION", policy="SHARED", categories=["wooden table"]),
    ], "functional_relations": [], "interaction_groups": []}
    hypotheses = build_role_type_hypotheses("workshop", document_with_both)
    assert hypotheses["target_location"].canonical_role_candidates == ("repair_target",)
    assert "MAIN_WORKBENCH_ZONE" in hypotheses["workbench"].canonical_role_candidates
    assert hypotheses["target_location"].canonical_role_candidates != (
        hypotheses["workbench"].canonical_role_candidates)


def test_a_surface_that_states_only_the_fastening_is_still_only_the_site():
    """The adversarial half: no tool return, no support reading."""
    document_site_only = {"functional_roles": [
        role("marked_site", "the place on the workbench where the fastening must occur",
             kind="REGION", policy="SHARED", categories=["marked spot"]),
    ], "functional_relations": [], "interaction_groups": []}
    hypotheses = build_role_type_hypotheses("workshop", document_site_only)
    assert hypotheses["marked_site"].canonical_role_candidates == ("repair_target",)


# ---------------------------------------------------------------------------
# Setting a participant aside is a decision about one operation
# ---------------------------------------------------------------------------


def test_a_bench_one_operation_sets_aside_survives_for_the_tool_return():
    raw = document([
        role("component_part", "physical piece to be attached to the workbench",
             categories=["block"], policy="DISTINCT"),
        role("fastening_tool", "instrument used to secure the component",
             categories=["screwdriver"], policy="SHARED"),
        role("workbench", "surface where the assembly happens and tools are left",
             kind="FIXED_TARGET", policy="SHARED", categories=["workbench"]),
    ], relations=[relation("fit", "the component must fit the marked workbench location",
                           ["component_part", "workbench"])],
        operations=[
            operation("perform_fastening", "attach the component to the workbench using the tool",
                      ["component_part", "fastening_tool", "workbench"]),
            operation("deposit_tool", "place the reusable tool on the workbench surface",
                      ["fastening_tool", "workbench"]),
    ])
    canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    assert "workbench" not in canonical["current_state_operation_context_roles"], (
        canonical["current_state_operation_context_roles"])
    unresolved = trace_of(graph).get("unresolved_required_operations", [])
    assert not any(row.get("id") == "deposit_tool" for row in unresolved), unresolved


def test_a_requirement_about_the_set_aside_role_is_enforced_not_dropped():
    """The relation names the bench; the fastening's site is what it is about."""
    raw = document([
        role("component_part", "physical piece to be attached to the workbench",
             categories=["block"], policy="DISTINCT"),
        role("fastening_tool", "instrument used to secure the component",
             categories=["screwdriver"], policy="SHARED"),
        role("workbench", "surface where the assembly happens and tools are left",
             kind="FIXED_TARGET", policy="SHARED", categories=["workbench"]),
    ], relations=[relation("fit", "the component must fit the marked workbench location",
                           ["component_part", "workbench"])],
        operations=[operation(
            "perform_fastening", "attach the component to the workbench using the tool",
            ["component_part", "fastening_tool", "workbench"])])
    canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    enforced = canonical.get("relations_enforced_by_operations", [])
    assert any(row.get("id") == "fit" for row in enforced), enforced


# ---------------------------------------------------------------------------
# A reading the endpoints refuse is not a reading
# ---------------------------------------------------------------------------


def test_a_coordinated_phrase_is_seated_by_what_its_participants_identify():
    raw = document([
        role("soup_bowl", "container for soup", categories=["bowl"], count=2, policy="DISTINCT"),
        role("soup_utensil", "tool to eat the soup with", categories=["spoon"],
             count=2, policy="DISTINCT"),
        role("soup_liquid", "food item to be served", categories=["broth"], count=2),
    ], operations=[operation("serve_soup", "place soup and add utensil",
                             ["soup_bowl", "soup_liquid", "soup_utensil"], count=2)])
    _, graph = compile_v3(raw, "kitchen", KITCHEN)
    functions = {group.function for group in graph.operation_groups or ()}
    assert "PROVIDE_SOUP_EATING_UTENSIL" in functions, (
        functions, trace_of(graph)["disabled_groups"])


def test_participants_that_identify_several_capabilities_are_still_refused():
    """The adversarial half: inference needs a unique answer, not a plausible one."""
    raw = document([
        role("coffee_cup", "container for the coffee", categories=["mug"],
             count=2, policy="DISTINCT"),
        role("coffee_powder", "ingredient supplying the coffee", categories=["coffee"]),
    ], operations=[operation("do_it", "handle the items", ["coffee_powder", "coffee_cup"])])
    _, graph = compile_v3(raw, "kitchen", KITCHEN)
    assert not graph.operation_groups or all(
        group.function != "PROVIDE_SOUP_EATING_UTENSIL" for group in graph.operation_groups)


# ---------------------------------------------------------------------------
# A context family the domain recognizes is understood, not unresolved
# ---------------------------------------------------------------------------


def test_somewhere_to_search_is_recorded_as_context_not_as_a_missing_participant():
    raw = document([
        role("fastening_tool", "instrument used to drive the fastener",
             categories=["screwdriver"]),
        role("attachment_component", "the part to be secured at the joint",
             categories=["screw"], policy="DISTINCT"),
        role("storage_container", "enclosed spaces to search for parts or tools",
             kind="REGION", policy="SHARED", categories=["cabinet", "drawer"]),
    ], operations=[operation("assemble", "fasten the component into the joint",
                             ["fastening_tool", "attachment_component"])])
    _, graph = compile_v3(raw, "workshop", WORKSHOP)
    trace = trace_of(graph)
    assert not any(
        row.get("raw_role", {}).get("id") == "storage_container"
        for row in trace["unresolved_roles"]
    ), trace["unresolved_roles"]
    assert any(
        row.get("raw_role", {}).get("id") == "storage_container"
        and row.get("status") == "RECOGNIZED_CONTEXT_FAMILY_WITH_NO_FUNCTIONAL_ROLE"
        for row in trace["context_only_roles"]
    ), trace["context_only_roles"]


def test_a_functional_participant_the_runtime_cannot_type_is_still_unresolved():
    """The adversarial half: not understanding a role is not the same as context."""
    raw = document([
        role("soup_bowl", "container for soup", categories=["bowl"], count=2, policy="DISTINCT"),
        role("soup_utensil", "tool to eat the soup with", categories=["spoon"],
             count=2, policy="DISTINCT"),
        role("soup_ingredient", "food ingredient to be served", categories=["broth"], count=2),
    ], operations=[operation("serve", "provide the utensil with the bowl",
                             ["soup_utensil", "soup_bowl"], count=2)])
    _, graph = compile_v3(raw, "kitchen", KITCHEN)
    trace = trace_of(graph)
    assert any(
        row.get("raw_role", {}).get("id") == "soup_ingredient"
        for row in trace["unresolved_roles"]
    ), trace["unresolved_roles"]


# ---------------------------------------------------------------------------
# A stated end state the compiled operation brings about
# ---------------------------------------------------------------------------


def test_an_end_state_the_transfer_achieves_is_recorded_as_its_effect():
    raw = document([
        role("coffee_cup", "container for the coffee", categories=["mug"],
             count=2, policy="DISTINCT"),
        role("coffee_powder", "ingredient supplying the coffee", categories=["coffee"],
             policy="SHARED"),
        role("stirrer", "implement used to stir the drink", categories=["spoon"],
             count=2, policy="DISTINCT"),
    ], relations=[relation("r1", "contained_in", ["coffee_powder", "coffee_cup"])],
        operations=[
            operation("pour", "pour the coffee into the cup",
                      ["coffee_powder", "coffee_cup"], count=2),
            operation("stir", "stir the contents of the cup", ["stirrer", "coffee_cup"], count=2),
    ])
    _, graph = compile_v3(raw, "kitchen", KITCHEN)
    trace = trace_of(graph)
    assert not any(
        row.get("id") == "r1" for row in trace.get("unresolved_required_relations", [])
    ), trace.get("unresolved_required_relations")
    assert any(
        row.get("status") == "TASK_EFFECT_SEMANTICS"
        and row.get("corroborated_by_compiled_operation")
        for row in trace["relations"]
    ), trace["relations"]


def test_an_end_state_no_operation_achieves_is_not_quietly_accepted():
    """The adversarial half: no transfer, so nothing brings the state about."""
    raw = document([
        role("coffee_cup", "container for the coffee", categories=["mug"],
             count=2, policy="DISTINCT"),
        role("coffee_powder", "ingredient supplying the coffee", categories=["coffee"],
             policy="SHARED"),
        role("stirrer", "implement used to stir the drink", categories=["spoon"],
             count=2, policy="DISTINCT"),
    ], relations=[relation("r1", "contained_in", ["coffee_powder", "coffee_cup"])],
        operations=[operation("stir", "stir the contents of the cup",
                              ["stirrer", "coffee_cup"], count=2)])
    _, graph = compile_v3(raw, "kitchen", KITCHEN)
    trace = trace_of(graph)
    assert not any(
        row.get("status") == "TASK_EFFECT_SEMANTICS"
        and row.get("corroborated_by_compiled_operation")
        for row in trace["relations"]
    ), trace["relations"]
