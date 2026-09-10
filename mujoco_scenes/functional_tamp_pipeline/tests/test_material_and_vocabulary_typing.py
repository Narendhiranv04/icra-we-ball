"""Typing a role from the vocabulary grounding will actually use.

Two independent mappings decide a role's fate.  Compile-time typing reads the
model's wording; grounding compares an observed label with the role's declared
acceptance categories.  When the model's own ``candidate_categories`` name one
of those very categories the two are talking about the same thing, and typing
that failed to notice cost real trials: a coffee source offered as a "jar" -- a
category coffee_source itself declares -- was read as a cupboard to search, and
the coffee it supplies disappeared from the task.

The other half is materials.  A role described only as a substance had no
family evidence at all, so every canonical role stayed a candidate: neither a
reading nor an honest refusal, and the cross-product of those candidates was
the pipeline's worst runtime.  Naming a material is naming a supply -- and
whether this domain *has* a source for that material is a separate question,
answered separately.

No reference graph, expected plan or benchmark variant appears here.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.role_semantic_ontology import (
    get_all_system_role_semantic_categories,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_typing import (
    acceptance_vocabulary_match,
    build_role_type_hypotheses,
    detect_role_families,
)


def role(rid, function, *, kind="OBJECT", count=1, policy="REUSABLE",
         categories=(), properties=(), description=""):
    return {"id": rid, "entity_kind": kind, "function": function, "description": description,
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": list(properties)}


def candidates(domain, item):
    document = {"functional_roles": [item], "functional_relations": [], "interaction_groups": []}
    return list(build_role_type_hypotheses(domain, document)[item["id"]].canonical_role_candidates)


# ---------------------------------------------------------------------------
# The runtime's own acceptance vocabulary
# ---------------------------------------------------------------------------


def test_a_uniquely_declared_category_names_the_role_that_declares_it():
    assert acceptance_vocabulary_match("kitchen", role("x", "", categories=["jar"])) == ("coffee_source",)
    assert acceptance_vocabulary_match("kitchen", role("x", "", categories=["bowl"])) == ("soup_container",)
    assert acceptance_vocabulary_match("workshop", role("x", "", categories=["screw"])) == ("fastener",)


def test_a_category_two_roles_declare_names_neither_of_them():
    """"spoon" says which family, not which role; the wording has to settle it."""
    assert acceptance_vocabulary_match("kitchen", role("x", "", categories=["spoon"])) == (
        "coffee_stirrer", "soup_eating_utensil")


def test_a_category_the_runtime_never_declared_names_nothing():
    assert acceptance_vocabulary_match("kitchen", role("x", "", categories=["broth", "stew"])) == ()
    assert acceptance_vocabulary_match("living_room", role("x", "", categories=["television"])) == ()


def test_a_role_with_no_categories_matches_nothing():
    assert acceptance_vocabulary_match("kitchen", role("x", "provide coffee")) == ()


def test_a_source_offered_as_a_jar_is_the_source_and_not_a_cupboard():
    """The regression: an openable jar of coffee read as somewhere to search."""
    supply = role("coffee_source", "Provides coffee ingredient",
                  categories=["jar", "canister", "bag", "box"],
                  properties=["contains coffee", "openable"])
    assert candidates("kitchen", supply) == ["coffee_source"]


def test_a_cupboard_is_still_a_cupboard():
    """Nothing in the fix may turn storage into one of the things it holds."""
    storage = role("storage_container", "Enclosed space potentially holding ingredients",
                   kind="REGION", categories=["cabinet", "box", "drawer", "chest"],
                   properties=["enclosed", "openable"])
    assert "coffee_source" not in candidates("kitchen", storage)


@pytest.mark.parametrize("domain", ["kitchen", "living_room", "workshop"])
def test_every_declared_category_still_reaches_its_own_role(domain):
    """The two layers must agree in this direction too."""
    for name, categories in get_all_system_role_semantic_categories(domain).items():
        for category in categories:
            matched = acceptance_vocabulary_match(domain, role("probe", "", categories=[category]))
            assert not matched or name in matched, (domain, name, category, matched)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def test_naming_a_material_nominates_a_supply():
    families, _ = detect_role_families(
        "kitchen", role("m", "Provide the coffee flavor.",
                        categories=["coffee beans", "instant coffee"]))
    assert "SOURCE" in families


def test_a_soup_material_is_refused_rather_than_read_as_something_else():
    """This domain models no soup supply, and that is a representability limit.

    Before, such a role carried no family evidence, so every canonical role
    stayed a candidate and the operation over it was read as placing a spoon in
    a bowl.
    """
    assert candidates("kitchen", role("soup_food", "Provide the soup substance.",
                                      categories=["broth", "curry", "stew"])) == []


def test_a_soup_material_is_not_read_as_the_bowl_it_goes_into():
    """"Hot food item served in the bowl" is the food, not the bowl."""
    portion = role("soup_portion", "Hot food item served in the bowl.",
                   count=2, policy="DISTINCT",
                   categories=["soup", "broth", "stew", "porridge"],
                   properties=["edible", "hot or warm"])
    assert candidates("kitchen", portion) == []


def test_a_coffee_material_still_reaches_the_coffee_supply():
    assert candidates("kitchen", role("Coffee_Powder", "Provide the coffee flavor.",
                                      count=2, policy="REUSABLE",
                                      categories=["coffee beans", "instant coffee"],
                                      properties=["aromatic", "dissolvable"])) == ["coffee_source"]


def test_a_bowl_that_mentions_soup_is_still_a_bowl():
    assert candidates("kitchen", role("soup_bowl", "Container for served soup",
                                      categories=["bowl", "dish"],
                                      properties=["rigid", "holds food"])) == ["soup_container"]


# ---------------------------------------------------------------------------
# An implement's action, and what is done to a material
# ---------------------------------------------------------------------------


def test_a_stirring_implement_is_the_implement_and_not_the_vessel():
    """"Fits in vessel" is a property of the spoon, not a claim to be one."""
    assert candidates("kitchen", role("stirring_tool", "Stir coffee mixture.",
                                      categories=["spoon", "stick"],
                                      properties=["rigid", "fits in vessel"])) == ["coffee_stirrer"]


def test_a_material_described_as_being_mixed_is_not_the_thing_that_mixes():
    """The action words read just as well in the passive; that is not an implement."""
    liquid = role("water", "Liquid ingredient to be mixed with coffee.",
                  count=2, policy="SHARED", categories=["liquid", "water source"])
    assert "coffee_stirrer" not in candidates("kitchen", liquid)


def test_a_cup_holding_a_mixture_is_not_an_implement_either():
    cup = role("coffee_cup", "Receives and holds coffee mixture for one person",
               count=2, policy="DISTINCT", categories=["mug", "cup", "glass"],
               properties=["holds liquid", "drinkable size"])
    assert candidates("kitchen", cup) == ["coffee_container"]
