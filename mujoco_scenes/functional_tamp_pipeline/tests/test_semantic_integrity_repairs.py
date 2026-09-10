"""Each repair that could lose a required semantic, paired with what stops it.

Every mechanism here exists because the runtime has to translate expressed
meaning into its own abstractions.  Each one is therefore tested twice: once
where the model gave the evidence and the repair is allowed, and once where it
did not and the repair must refuse.  Nothing here reads a reference graph, an
expected action sequence, or a benchmark variant.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.outcome_classifier import (
    complete_planning_contract,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph


KITCHEN = ("Prepare and serve one coffee and one soup for each of two people. Make each "
           "coffee using coffee and water and stir it before serving. Serve each soup bowl "
           "with its own suitable eating utensil.")
LIVING = ("Prepare the living room for two people to enjoy refreshments while watching "
          "television. Provide each person with their own refreshment setting nearby, and "
          "place the entertainment control where it is accessible to both people.")
WORKSHOP = ("Identify the compatible components required to complete the fastening at the "
            "marked workbench location, complete the fastening, and leave any reusable "
            "equipment used for the task safely on the workbench.")


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


# ---------------------------------------------------------------------------
# Unmapped required semantics: a mapping failure is not evidence of irrelevance
# ---------------------------------------------------------------------------


def test_unrepresentable_material_is_recorded_while_its_container_still_works():
    """A material this domain models no source for does not stop the rest.

    Allowed only because the roles the semantic *does* mention still have an
    operation acting on them.
    """
    raw = document([
        role("soup_bowl", "container for soup", categories=["bowl"], count=2, policy="DISTINCT"),
        role("eating_utensil", "tool to eat soup", categories=["spoon"], count=2, policy="DISTINCT"),
        role("soup_stock", "soup food source", categories=["broth"], count=2),
    ], operations=[
        operation("serve", "provide eating utensil", ["eating_utensil", "soup_bowl"], 2),
        operation("fill", "transfer material into receiving container", ["soup_stock", "soup_bowl"], 2),
    ])
    canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    assert "soup_container" in graph.nodes and "soup_eating_utensil" in graph.nodes
    surplus = graph.metadata["canonicalization_trace"]["surplus_unrepresentable_constraints"]
    assert any(row.get("requirement_provenance") == "RUNTIME_UNREPRESENTABLE_PARTICIPANT"
               for row in surplus), surplus
    assert graph.online_executable_contract_complete


def test_unrepresentable_material_blocks_when_it_leaves_a_role_with_no_operation():
    """The same loss is fatal when it strips a role of everything it was for.

    The bowl is now only ever mentioned by the transfer the runtime cannot
    represent, so accepting the loss would leave a participant of the task with
    nothing to do -- which is how a plan came to report a task done.
    """
    raw = document([
        role("soup_bowl", "container for soup", categories=["bowl"], count=2, policy="DISTINCT"),
        role("soup_stock", "soup food source", categories=["broth"], count=2),
    ], operations=[
        operation("fill", "transfer material into receiving container", ["soup_stock", "soup_bowl"], 2),
    ])
    _, graph = compile_v3(raw, "kitchen", KITCHEN)
    assert not graph.online_executable_contract_complete
    assert graph.metadata["executable_contract_missing_reasons"]


def test_ambiguous_role_is_not_treated_as_an_unrepresentable_one():
    """A mapper that could not decide is a limitation, not a surplus semantic."""
    from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
    raw = document([
        role("implement_a", "long implement for preparation"),
        role("implement_b", "long implement for preparation"),
    ], operations=[operation("use", "insert implement", ["implement_a", "implement_b"])])
    # Nothing here is unrepresentable in the runtime's ontology; the runtime
    # simply cannot tell which role each is.  That fails closed as a task
    # specification problem rather than being written off as surplus.
    with pytest.raises(VLMSpecificationError, match="No executable role could be typed"):
        compile_v3(raw, "kitchen", KITCHEN)


# ---------------------------------------------------------------------------
# Participant dropping: only context may be set aside
# ---------------------------------------------------------------------------


def test_a_current_location_participant_may_be_set_aside():
    raw = document([
        role("payload", "personal refreshment drinkware set", categories=["cup", "saucer"],
             count=2, policy="DISTINCT"),
        role("support", "personal side table beside the seat", kind="REGION",
             count=2, policy="DISTINCT", categories=["side table"]),
        role("tray", "initial staging tray where the items currently sit", kind="REGION",
             policy="SHARED", categories=["tray"]),
    ], operations=[operation("place", "place drinkware", ["payload", "support", "tray"], 2)])
    canonical, graph = compile_v3(raw, "living_room", LIVING)
    group = canonical["interaction_groups"][0]
    assert "tray" in group["v3_current_state_context_roles"]
    assert graph.operation_groups[0].function == "SUPPORT_DRINKWARE"


def test_a_fixed_target_participant_is_not_set_aside_merely_for_being_a_region():
    """A seat, a fastening target and a destination are all non-objects.

    Dropping one to force an operation to fit would silently remove a
    requirement, so an over-specified operation naming a second essential
    reference is recorded rather than trimmed.
    """
    raw = document([
        role("tool", "reusable fastening implement", categories=["screwdriver"]),
        role("fastener", "manipulated joining component", categories=["screw"]),
        role("target", "fixed assembly receiving installed component", kind="FIXED_TARGET",
             policy="SHARED"),
        role("second_target", "another fixed assembly receiving installed component",
             kind="FIXED_TARGET", policy="SHARED"),
    ], operations=[operation("fasten", "fasten component at target",
                             ["tool", "fastener", "target", "second_target"])])
    canonical, graph = compile_v3(raw, "workshop", WORKSHOP)
    assert canonical["interaction_groups"] == []
    assert canonical["unresolved_operation_semantics"]
    assert not graph.online_executable_contract_complete


# ---------------------------------------------------------------------------
# N-ary relations: pairwise decomposition only where the predicate distributes
# ---------------------------------------------------------------------------


def test_containment_over_a_set_distributes_onto_the_carrier_only():
    """"The cup contains coffee and water" says nothing about coffee and water.

    Both materials end up in the cup; neither ends up in the other.  Emitting
    that third pair asserted a claim the model never made and then blocked the
    contract as unrepresentable.
    """
    raw = document([
        role("cup", "container for coffee", categories=["mug"], count=2, policy="DISTINCT"),
        role("grounds", "coffee ingredient source", categories=["coffee jar"]),
        role("water", "water source", categories=["kettle"]),
    ], relations=[relation("mix", "contains", ["cup", "grounds", "water"])],
        operations=[
            operation("pour_a", "transfer material into receiving container", ["grounds", "cup"], 2),
            operation("pour_b", "transfer material into receiving container", ["water", "cup"], 2),
        ])
    canonical, graph = compile_v3(raw, "kitchen", KITCHEN)
    decomposed = {
        tuple(sorted((edge["subject_role"], edge["object_role"])))
        for edge in canonical["functional_relations"]
    }
    assert ("grounds", "water") not in decomposed, decomposed
    assert graph.online_executable_contract_complete


def test_quantified_relation_with_no_legal_reading_is_recorded_not_invented():
    """A recognised predicate the runtime cannot orient here stays unresolved."""
    raw = document([
        role("remote", "entertainment control device", categories=["remote"]),
        role("screen", "television display", kind="FIXED_TARGET", policy="SHARED",
             categories=["screen"]),
    ], relations=[relation("access", "accessible from both seats", ["remote", "screen"])])
    canonical, graph = compile_v3(raw, "living_room", LIVING)
    assert not canonical.get("explicit_context_sets")
    assert "SEATING_PAIR" not in graph.nodes
    assert not graph.online_executable_contract_complete


# ---------------------------------------------------------------------------
# Relations an operation already enforces
# ---------------------------------------------------------------------------


def test_relation_restating_a_capability_precondition_is_enforced_not_lost():
    raw = document([
        role("tool", "reusable fastening implement", categories=["screwdriver"]),
        role("fastener", "manipulated joining component", categories=["screw"]),
        role("target", "fixed assembly receiving installed component", kind="FIXED_TARGET",
             policy="SHARED"),
    ], relations=[relation("secures", "tool secures fastener to target", ["tool", "fastener"])],
        operations=[operation("fasten", "fasten component at target",
                              ["tool", "fastener", "target"])])
    _, graph = compile_v3(raw, "workshop", WORKSHOP)
    enforced = graph.metadata["canonicalization_trace"]["relations_enforced_by_operations"]
    assert enforced, graph.metadata["executable_contract_missing_reasons"]
    assert graph.online_executable_contract_complete


def test_relation_over_roles_no_operation_binds_still_blocks():
    """The same wording, with nothing to enforce it, remains a real gap."""
    raw = document([
        role("tool", "reusable fastening implement", categories=["screwdriver"]),
        role("fastener", "manipulated joining component", categories=["screw"]),
        role("target", "fixed assembly receiving installed component", kind="FIXED_TARGET",
             policy="SHARED"),
    ], relations=[relation("secures", "tool secures fastener to target", ["tool", "fastener"])])
    _, graph = compile_v3(raw, "workshop", WORKSHOP)
    assert not graph.metadata["canonicalization_trace"]["relations_enforced_by_operations"]
    assert not graph.online_executable_contract_complete


# ---------------------------------------------------------------------------
# A plan is only complete for a task the graph actually stated
# ---------------------------------------------------------------------------


class _Grounding:
    def __init__(self, complete=True, bindings=None):
        self.complete = complete
        self.operation_bindings = bindings or {}


def test_a_graph_with_no_operation_cannot_report_a_completed_task():
    from mujoco_scenes.functional_tamp_pipeline.models import (
        FunctionalRequirementGraph, FunctionalRole,
    )
    graph = FunctionalRequirementGraph(
        domain="workshop", task_instruction=WORKSHOP,
        nodes={"driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1)},
        metadata={"online_executable_contract_complete": True},
    )
    assert complete_planning_contract(graph, _Grounding(), {}, {}) is False


def test_an_unrepresented_requirement_cannot_report_a_completed_task():
    from mujoco_scenes.functional_tamp_pipeline.models import (
        FunctionalRequirementGraph, FunctionalRole, OperationGroup,
    )
    group = OperationGroup(id="g", function="F", tool_role="driver", target_role="fastener",
                           required_target_count=1, usage_policy="DEDICATED_PER_TARGET")
    graph = FunctionalRequirementGraph(
        domain="workshop", task_instruction=WORKSHOP,
        nodes={"driver": FunctionalRole(name="driver", entity_kind="OBJECT", count=1),
               "fastener": FunctionalRole(name="fastener", entity_kind="OBJECT", count=1)},
        operation_groups=(group,),
        metadata={"online_executable_contract_complete": False},
    )
    grounding = _Grounding(bindings={"g": [{"tool_id": "d", "target_id": "f"}]})
    assert complete_planning_contract(graph, grounding, {}, {}) is False
    graph.metadata["online_executable_contract_complete"] = True
    assert complete_planning_contract(graph, grounding, {}, {}) is True
