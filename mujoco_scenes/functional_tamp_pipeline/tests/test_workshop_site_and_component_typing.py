"""Telling the fastening site apart from the thing fastened into it.

Four ordinary sentences were being read as their opposites, and each one cost
a whole workshop contract:

  "a closed structure that may hold the component or tool"  read as the tool
  "the part that needs to be fastened to the site"          read as the site
  "a specific area on the workbench requiring fastening"    read as the bench
  "receives the fastener"                                   read as nothing

None of them needed new vocabulary.  Three needed morphology -- "may hold" is
the same claim as "holds", "requiring" the same as "requires" -- and one needed
a preposition: what separates the thing being fastened from the site receiving
it is not the participle, which both sentences contain, but whether something
else follows it that the fastening happens against.

The sentences below are written for this test rather than copied from the
development set, and each positive has a negative beside it, because a rule
that pulls everything toward one reading is no better than the bug.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.semantic_typing import build_role_type_hypotheses


def role(rid, function, *, kind="OBJECT", categories=(), properties=()):
    return {"id": rid, "entity_kind": kind, "function": function, "description": "",
            "required_count": 1, "binding_policy": "SHARED",
            "candidate_categories": list(categories), "required_properties": list(properties)}


def typed(item):
    document = {"functional_roles": [item], "functional_relations": [], "interaction_groups": []}
    return list(build_role_type_hypotheses("workshop", document)[item["id"]].canonical_role_candidates)


# ---------------------------------------------------------------------------
# A container described by what it might hold is not that thing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function,properties", [
    ("a closed structure that may hold the component or tool", ["enclosed", "openable"]),
    ("a shut compartment that could keep the screws out of sight", ["closed"]),
    ("a closed space where the parts might be hidden", ["searchable"]),
    ("a drawer that requires searching for the equipment", ["has a latch"]),
])
def test_somewhere_to_search_is_not_what_it_holds(function, properties):
    """The regression: a cupboard "that may hold the tool" became the driver."""
    result = typed(role("store", function, categories=["cabinet", "drawer"], properties=properties))
    assert "driver" not in result, result
    assert "fastener" not in result, result


def test_a_tool_is_still_a_tool():
    """The negative: nothing here may push an implement into the storage family."""
    assert typed(role("tool", "drives the fastener into the joint",
                      categories=["screwdriver"])) == ["driver"]


# ---------------------------------------------------------------------------
# Fastened *to* something else, versus receiving the fastening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("function", [
    "the physical part that needs to be fastened to the site",
    "the object that must be secured to the workbench",
    "a piece that is joined onto the frame",
    "the item to be attached at the marked point",
])
def test_something_fastened_onto_something_else_is_the_component(function):
    result = typed(role("part", function, categories=["block", "plate"], properties=["movable"]))
    assert result == ["fastener"], result


@pytest.mark.parametrize("function", [
    "receive fastening at target joint hole",
    "accepts the screw at the recess",
    "the specific location on the bench where the fastening must occur",
    "a specific area on the workbench requiring fastening",
    "the assembly that needs fastening",
    "object to be fastened.",
])
def test_the_site_receiving_the_fastening_is_the_target(function):
    """The adversarial half: the same participles, opposite valency."""
    result = typed(role("site", function, kind="REGION",
                        categories=["joint recess", "workbench surface"]))
    assert result == ["repair_target"], result


def test_a_plain_bench_surface_is_not_the_fastening_site():
    """The other negative: not every workbench surface is a repair target."""
    assert typed(role("bench", "flat surface to rest tools on when finished",
                      kind="REGION", categories=["workbench", "bench top"])) == [
        "MAIN_WORKBENCH_ZONE"]


# ---------------------------------------------------------------------------
# The site and the part can be told apart in the same contract
# ---------------------------------------------------------------------------


def test_a_contract_naming_both_types_each_of_them_differently():
    document = {"functional_roles": [
        role("attachment_part", "the physical part that needs to be fastened to the site",
             categories=["block", "module"], properties=["movable"]),
        role("marked_site", "the specific location on the workbench where the fastening occurs",
             kind="REGION", categories=["marked spot"]),
        role("hand_tool", "drives the fastener", categories=["screwdriver"]),
        role("cupboard", "a closed structure that may hold the part or the tool",
             categories=["cabinet"], properties=["openable"]),
    ], "functional_relations": [], "interaction_groups": []}
    result = {rid: list(h.canonical_role_candidates)
              for rid, h in build_role_type_hypotheses("workshop", document).items()}
    assert result["attachment_part"] == ["fastener"]
    assert result["marked_site"] == ["repair_target"]
    assert result["hand_tool"] == ["driver"]
    assert result["cupboard"] == [], "a place to search is no functional participant"
    assert result["attachment_part"] != result["marked_site"], (
        "the part and the site must not collide on one role")
