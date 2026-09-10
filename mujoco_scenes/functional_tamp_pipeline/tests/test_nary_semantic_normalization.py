"""A relation over several participants is decomposed by meaning, not by order.

Three things had to be true at once for this to work, and each was broken.

English separates a phrasal verb from its particle -- "combined to form a drink
IN a cup", "placed neatly ON the table" -- so a contiguous cue list read those
sentences as carrying no meaning at all.  They are matched now as verb-particle
families across a bounded gap inside one clause.

A stated end state relates two *different* kinds of participant: something
carries and something is carried.  Requiring only that one endpoint *could*
carry admitted the pair (cup, bowl) from "the cups and the bowls are placed on
the table", recording a claim that the cup goes on the bowl.

And what survives decomposition is decided by which pairs have a legal reading
under the role domains, never by participant position.

Every sentence in the positive tests below is a paraphrase written for this
test, not text copied from the development set, so passing them is evidence the
mechanism generalizes rather than that it memorized.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    _decompose_nary_relation,
    _relation_options,
)
from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import (
    extract_relation_semantic_candidates,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses


def role(rid, function, *, kind="OBJECT", count=1, policy="REUSABLE", categories=()):
    return {"id": rid, "entity_kind": kind, "function": function, "description": "",
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": []}


def _edges(domain, roles, text, participants):
    document = {"functional_roles": list(roles), "functional_relations": [], "interaction_groups": []}
    hypotheses = build_role_type_hypotheses(domain, document)
    relation = {"id": "r", "relation": text, "participant_roles": list(participants), "required": True}
    out = []
    for edge in _decompose_nary_relation(domain, relation, hypotheses):
        for option in _relation_options(domain, edge, hypotheses) or []:
            out.append((option["predicate"], option["subject_role"], option["object_role"]))
    return set(out), [tuple(e["participant_roles"]) for e in _decompose_nary_relation(
        domain, relation, hypotheses)]


KITCHEN_MATERIALS = [
    role("ground_coffee", "Provide the coffee flavour", categories=["coffee beans"]),
    role("hot_water", "Provide the liquid base", categories=["kettle"]),
    role("drinking_mug", "Hold the prepared coffee", count=2, policy="DISTINCT", categories=["mug"]),
]
KITCHEN_SERVICE = [
    role("drinking_mug", "Hold the prepared coffee", count=2, policy="DISTINCT", categories=["mug"]),
    role("soup_dish", "Hold the prepared soup", count=2, policy="DISTINCT", categories=["bowl"]),
    role("service_top", "surface to place items on", kind="REGION", categories=["dining table"]),
]
WORKSHOP_SITE = [
    role("joining_screw", "Joining component to install", categories=["screw"]),
    role("frame_part", "the part receiving the fastening", categories=["frame"]),
    role("marked_spot", "the marked place where fastening happens",
         kind="FIXED_TARGET", categories=["recess"]),
]


# ---------------------------------------------------------------------------
# Verb and particle need not be adjacent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("pour the hot water carefully into each mug", "PROVIDES_MATERIAL_TO"),
    ("the broth is transferred slowly into the serving bowl", "PROVIDES_MATERIAL_TO"),
    ("blend the grounds and the milk together in the jug", "PROVIDES_MATERIAL_TO"),
    ("set the tray down gently on the side table", "PLACED_ON"),
    ("the mug should be positioned squarely upon the mat", "PLACED_ON"),
    ("drive the screw firmly into the marked recess", "INSTALLED_AT"),
    ("the bracket is fastened securely at the joint", "INSTALLED_AT"),
    ("the two brackets must be joined together", "CONNECTED_TO"),
])
def test_a_separated_verb_and_particle_still_name_the_meaning(text, expected):
    found = {c.predicate_name for c in extract_relation_semantic_candidates("workshop", text)}
    found |= {c.predicate_name for c in extract_relation_semantic_candidates("kitchen", text)}
    assert expected in found, f"{text!r} -> {found}"


@pytest.mark.parametrize("text", [
    "the cup sits beside the bowl",
    "the spoon belongs with the dish",
    "the tool is stored in the drawer until it is needed",
    "the spoon is stirred in the mug",
])
def test_wording_that_states_no_transfer_or_placement_names_none(text):
    """The adversarial half: a gap-tolerant match must not swallow everything.

    Stirring is an implement acting on a vessel's contents, not a material
    being transferred into it, so it must not join the transfer family however
    close the preposition sits.
    """
    found = {c.predicate_name for c in extract_relation_semantic_candidates("kitchen", text)}
    assert "PROVIDES_MATERIAL_TO" not in found, f"{text!r} -> {found}"
    assert "INSTALLED_AT" not in found, f"{text!r} -> {found}"


def test_the_particle_must_share_a_clause_with_its_verb():
    """A sentence boundary stops the match, so two statements do not merge."""
    found = {c.predicate_name for c in extract_relation_semantic_candidates(
        "kitchen", "the jug is filled beforehand. the spoon rests in the drawer")}
    assert "PROVIDES_MATERIAL_TO" in found, "the jug clause is a genuine fill"
    placed = {c.predicate_name for c in extract_relation_semantic_candidates(
        "kitchen", "wipe the counter thoroughly. the mug is nowhere near")}
    assert "PLACED_ON" not in placed


# ---------------------------------------------------------------------------
# Distributive structure: every ingredient to the carrier, and nothing else
# ---------------------------------------------------------------------------


def test_two_materials_and_a_vessel_each_supply_the_vessel():
    edges, pairs = _edges("kitchen", KITCHEN_MATERIALS,
                          "the grounds and the water are combined together in the mug",
                          ["ground_coffee", "hot_water", "drinking_mug"])
    assert ("PROVIDES_MATERIAL_TO", "ground_coffee", "drinking_mug") in edges
    assert ("PROVIDES_MATERIAL_TO", "hot_water", "drinking_mug") in edges
    assert ("ground_coffee", "hot_water") not in pairs, (
        "one ingredient does not supply the other")


def test_two_payloads_and_a_support_each_rest_on_the_support():
    edges, pairs = _edges("kitchen", KITCHEN_SERVICE,
                          "the mugs and the dishes are laid out on the service top",
                          ["drinking_mug", "soup_dish", "service_top"])
    assert ("PLACED_ON", "drinking_mug", "service_top") in edges
    assert ("PLACED_ON", "soup_dish", "service_top") in edges
    assert ("drinking_mug", "soup_dish") not in pairs, (
        "the regression: one carrier does not go on the other")


def test_a_fastening_site_relates_to_the_component_and_not_to_itself():
    edges, pairs = _edges("workshop", WORKSHOP_SITE,
                          "the screw is secured to the frame part at the marked spot",
                          ["joining_screw", "frame_part", "marked_spot"])
    subjects = {s for _p, s, _o in edges}
    assert subjects == {"joining_screw"}, subjects
    assert ("frame_part", "marked_spot") not in pairs


def test_order_of_participants_does_not_change_the_decomposition():
    """Position carries no meaning; the same sentence must read the same."""
    forward, _ = _edges("kitchen", KITCHEN_SERVICE,
                        "the mugs and the dishes are laid out on the service top",
                        ["drinking_mug", "soup_dish", "service_top"])
    shuffled, _ = _edges("kitchen", KITCHEN_SERVICE,
                         "the mugs and the dishes are laid out on the service top",
                         ["service_top", "soup_dish", "drinking_mug"])
    assert forward == shuffled


def test_a_relation_with_no_legal_reading_over_its_roles_emits_nothing():
    """Fail closed: no orientation is guessed when none is admissible."""
    edges, _pairs = _edges("kitchen", [
        role("stirrer_one", "Stir the coffee", categories=["spoon"]),
        role("stirrer_two", "Stir the coffee", categories=["spoon"]),
        role("stirrer_three", "Stir the coffee", categories=["spoon"]),
    ], "the spoons are poured into one another",
        ["stirrer_one", "stirrer_two", "stirrer_three"])
    assert edges == set()


def test_every_emitted_edge_records_the_relation_it_came_from():
    document = {"functional_roles": KITCHEN_SERVICE, "functional_relations": [],
                "interaction_groups": []}
    hypotheses = build_role_type_hypotheses("kitchen", document)
    relation = {"id": "service_layout", "relation": "the mugs and the dishes are laid out on the service top",
                "participant_roles": ["drinking_mug", "soup_dish", "service_top"], "required": True}
    for edge in _decompose_nary_relation("kitchen", relation, hypotheses):
        assert edge["source_relation_ids"] == ["service_layout"]
        assert edge["relation"] == relation["relation"], "raw text is preserved on every edge"
