"""One sentence can mean several things at once, and all of it is evidence.

The FM's answer is a noisy structured semantic graph, not a serializer whose
every field must match our private vocabulary.  A phrase such as "the control
is supported by the surface" states an end state *and* says what its two
endpoints are; a phrase such as "the cup contains the coffee" states an end
state *and* names a carrier and a material.  Reading only one of those
meanings, or discarding the sentence because one of them was an end state,
throws away the only evidence there was.

Several readings of one phrase are a disjunction, so the constraint they place
on the endpoints is the union of what each reading admits.  A union can only
under-constrain: it never narrows onto the wrong reading, which is what taking
a single reading would risk.

Nothing here reads a reference graph, an expected plan, or a benchmark variant.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.semantic_typing import (
    build_role_type_hypotheses,
    relation_canonical_role_pairs,
)


def role(rid, function, *, kind="OBJECT", count=1, policy="REUSABLE", categories=(), properties=()):
    return {"id": rid, "entity_kind": kind, "function": function, "description": "",
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": list(properties)}


def relation(rid, text, participants):
    return {"id": rid, "relation": text,
            "subject_role": participants[0], "object_role": participants[1],
            "participant_roles": list(participants), "required": True}


def hypotheses(domain, roles, relations=(), operations=()):
    document = {"functional_roles": list(roles), "functional_relations": list(relations),
                "interaction_groups": list(operations)}
    return build_role_type_hypotheses(domain, document)


def candidates(domain, roles, relations=(), operations=()):
    return {rid: list(h.canonical_role_candidates)
            for rid, h in hypotheses(domain, roles, relations, operations).items()}


# ---------------------------------------------------------------------------
# A stated end state still says what its endpoints are
# ---------------------------------------------------------------------------


def test_an_end_state_relates_kinds_of_participant():
    assert relation_canonical_role_pairs("kitchen", "CONTAINS", "TASK_EFFECT_SEMANTICS")
    assert relation_canonical_role_pairs("living_room", "PLACED_ON", "TASK_EFFECT_SEMANTICS")


def test_an_end_state_over_an_unknown_predicate_says_nothing():
    assert relation_canonical_role_pairs("kitchen", "ZZQ_UNKNOWN", "TASK_EFFECT_SEMANTICS") == set()


def test_a_support_relation_types_the_support_even_though_it_states_an_effect():
    """FIX 1's own example: the only evidence that "surface" is a support.

    The sentence carries a PLACED_ON effect, and previously the whole relation
    was skipped for typing on that account, leaving the role with no evidence
    at all.
    """
    result = candidates("living_room", [
        role("entertainment_control", "Operates the television", categories=["remote control"]),
        role("spot", "the place it should end up", kind="REGION", categories=[]),
    ], [relation("r", "entertainment_control is supported by spot",
                 ("entertainment_control", "spot"))])
    assert result["entertainment_control"] == ["REMOTE"]
    assert set(result["spot"]) <= {"PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION", "staging_tray"}, (
        "the relation is the only evidence that this is a support")
    assert "CUP_SAUCER_SET" not in result["spot"], "a support is not a payload"


def test_both_support_readings_stay_viable_rather_than_being_chosen():
    """Ambiguity is preserved; nothing picks the reading that would succeed."""
    result = candidates("living_room", [
        role("entertainment_control", "Operates the television", categories=["remote control"]),
        role("spot", "the place it should end up", kind="REGION", categories=[]),
    ], [relation("r", "entertainment_control rests on spot", ("entertainment_control", "spot"))])
    assert {"PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION"} <= set(result["spot"])


def test_a_containment_effect_does_not_type_the_material_as_an_implement():
    """The adversarial half: one reading alone would get this wrong.

    "The cup contains the coffee" also reads as an insertion, and an insertion's
    endpoints are an implement and a container.  Narrowing on that reading alone
    would type the coffee as a stirrer.  The union keeps the material reading.
    """
    result = candidates("kitchen", [
        role("coffee_cup", "Hold the prepared coffee", categories=["mug", "cup"]),
        role("coffee_matter", "Provide the coffee flavour", categories=["coffee beans"]),
    ], [relation("r", "coffee_cup contains coffee_matter", ("coffee_cup", "coffee_matter"))])
    assert result["coffee_cup"] == ["coffee_container"]
    assert result["coffee_matter"] == ["coffee_source"], result["coffee_matter"]


def test_an_end_state_alone_invents_no_endpoint_type():
    """With no role evidence and no other reading, nothing is decided.

    The union may only narrow a domain the roles already had; it never creates
    a type for a participant the FM described in no way at all.
    """
    result = candidates("kitchen", [
        role("thing_a", "zzq unspecified", categories=[]),
        role("thing_b", "zzq unspecified", categories=[]),
    ], [relation("r", "thing_a is on thing_b", ("thing_a", "thing_b"))])
    for rid, values in result.items():
        assert len(values) != 1, f"{rid} was decided by a bare end state: {values}"


def test_contradictory_endpoint_evidence_still_fails_closed():
    """A relation no reading can orient over these roles must still block.

    The roles keep the type their own wording gave them -- explicit evidence
    outranks a sentence the runtime cannot orient over it -- and the relation
    is then reported as a requirement the runtime could not represent rather
    than quietly dropped or forced onto a reading.
    """
    from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
    document = {
        "functional_roles": [
            role("stirrer", "Stir the coffee", categories=["spoon"]),
            role("other_stirrer", "Stir the coffee", categories=["spoon"]),
        ],
        "functional_relations": [relation("r", "stirrer contains other_stirrer",
                                          ("stirrer", "other_stirrer"))],
        "interaction_groups": [], "inspectable_regions": [], "inspection_order": [],
    }
    graph = compile_candidate_graph("kitchen", "stir two coffees", document)
    metadata = graph.metadata or {}
    assert not metadata.get("online_executable_contract_complete")
    reasons = " ".join(str(x) for x in metadata.get("executable_contract_missing_reasons") or [])
    assert "contains" in reasons, reasons


def test_the_readings_that_were_unioned_are_recorded():
    """Provenance: which meanings contributed, so a decision can be audited."""
    result = hypotheses("living_room", [
        role("entertainment_control", "Operates the television", categories=["remote control"]),
        role("surface", "holds items for access", kind="REGION", categories=["table"]),
    ], [relation("r", "entertainment_control is supported by surface",
                 ("entertainment_control", "surface"))])
    evidence = [e for e in result["surface"].evidence if e.get("source") == "RELATION_TEXT"]
    assert evidence, "the relation must be recorded as the reason the role was narrowed"
    readings = evidence[0]["readings_unioned"]
    assert any("TASK_EFFECT_SEMANTICS" in r for r in readings)
    assert any("PHYSICAL_VERIFIER" in r for r in readings)
