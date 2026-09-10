"""Reading ordinary English as the semantics it carries, and refusing the rest.

The interpreter used to hold contiguous phrase lists.  "stirs" was in one and
"stir" was not, so an imperative carried no meaning; "suit_for", "must
physically suit" and "is suitable for" all say compatibility and none of them
was listed.  The lists have been replaced by the stem inventory of each
meaning, which is a much wider net, so every family below is tested twice: once
on wordings invented here that a competent reader would accept, and once on
wordings that genuinely say nothing the runtime can verify, which must still
fail closed.

Nothing in this file reads a reference graph, an expected action sequence, a
benchmark variant, or a frozen trial.  Every phrase is written for the test.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.relation_interpreter import (
    extract_relation_semantic_candidates,
    interpret_relation,
    interpret_task_effect_predicate,
    relation_states_a_hedged_possibility,
    relation_states_where_things_currently_are,
)
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    extract_operation_semantic_candidates,
    is_non_physical_operation_phrase,
)


def meanings(domain, phrase):
    return {c.predicate_name for c in extract_relation_semantic_candidates(domain, phrase)}


# ---------------------------------------------------------------------------
# Compatibility and fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "fits",
    "fits_into",
    "must fit into",
    "suit_for",
    "is suitable for",
    "must physically suit the receiving part",
    "has to match the recess",
    "mates with",
    "conforms to",
])
def test_compatibility_wordings_are_read_as_a_mechanical_fit(phrase):
    assert "COMPATIBLE_WITH_TARGET" in meanings("workshop", phrase), phrase


@pytest.mark.parametrize("phrase", [
    "secures",
    "used_to_fasten",
    "attaches_to",
    "must be secured to the receiving part",
    "is used to install",
    "to join",
    "screws in",
])
def test_fastening_wordings_are_read_as_fastening(phrase):
    assert meanings("workshop", phrase), phrase


@pytest.mark.parametrize("phrase", [
    # A gerund modifying a noun names a kind, not an action.
    "tool acts on fastening component",
    "the fastening tool",
    "the fastening component",
])
def test_a_gerund_naming_a_kind_is_not_a_fastening_claim(phrase):
    """The adversarial half: the same stem, used attributively."""
    assert "COMPATIBLE_WITH" not in meanings("workshop", phrase), phrase


@pytest.mark.parametrize("phrase", [
    "must maintain 45 degree tilt during operation",
    "functional",
    "spatial",
    "must satisfy the workshop convention",
    "has the right vibe",
])
def test_wording_with_no_runtime_meaning_nominates_nothing(phrase):
    assert not meanings("workshop", phrase), phrase


# ---------------------------------------------------------------------------
# Causal wordings, and the separation the domain keeps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase,expected", [
    ("stir", "ACTS_ON"),
    ("agitates the contents", "ACTS_ON"),
    ("uses_for_action", "ACTS_ON"),
    ("combine", "PROVIDES_MATERIAL_TO"),
    ("is decanted into", "PROVIDES_MATERIAL_TO"),
    ("pair", "PAIRED_WITH"),
    ("equipped_with", "PAIRED_WITH"),
    ("has_part", "PAIRED_WITH"),
])
def test_causal_wordings_are_read_by_stem(phrase, expected):
    assert expected in meanings("kitchen", phrase), phrase


def test_a_passive_causal_wording_is_read_in_the_other_direction():
    candidates = extract_relation_semantic_candidates("kitchen", "acted_upon_by")
    assert ("ACTS_ON", "REVERSE") in {(c.predicate_name, c.direction) for c in candidates}


def test_tool_action_on_the_fastener_stays_causal_not_a_mechanical_fit():
    """The domain tells "the tool acts on the component" from "it engages it"."""
    result = interpret_relation(
        domain="workshop", raw_phrase="the tool acts on the component",
        subject_role="driver", object_role="fastener",
        subject_kind="OBJECT", object_kind="OBJECT", required=True)
    assert result.succeeded
    assert result.category == "TASK_CAUSAL_SEMANTICS"
    assert result.interpreted_predicates[0].predicate_name == "ACTS_ON"


def test_tool_action_at_the_site_is_read_as_needing_to_reach_it():
    """The other side of the same coin: no causal predicate relates these two."""
    result = interpret_relation(
        domain="workshop", raw_phrase="the tool acts upon the marked location",
        subject_role="driver", object_role="repair_target",
        subject_kind="OBJECT", object_kind="FIXED_TARGET", required=True)
    assert result.succeeded
    assert result.category == "PHYSICAL_VERIFIER"
    assert "REACHES_TARGET" in {p.predicate_name for p in result.interpreted_predicates}


def test_several_causal_readings_the_endpoints_cannot_separate_fail_closed():
    """No lexical tie-break: an undetermined relation stays undetermined."""
    result = interpret_relation(
        domain="kitchen", raw_phrase="is provided with and mixed into",
        subject_role="soup_eating_utensil", object_role="coffee_source",
        subject_kind="OBJECT", object_kind="OBJECT", required=True)
    assert not result.succeeded


def test_vague_association_alone_is_still_not_a_pairing_claim():
    result = interpret_relation(
        domain="kitchen", raw_phrase="associated with",
        subject_role="soup_eating_utensil", object_role="soup_container",
        subject_kind="OBJECT", object_kind="OBJECT", required=True)
    assert not result.succeeded


# ---------------------------------------------------------------------------
# End states, and statements about the scene as it already is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", [
    "contained_in", "is contained within", "holds the hot contents",
    "filled_with", "Coffee is in Mug",
])
def test_containment_end_states_are_read_by_stem(phrase):
    assert interpret_task_effect_predicate(phrase) == "CONTAINS", phrase


@pytest.mark.parametrize("phrase", [
    "can hold drinkware set",
    "holds items for the viewer",
    "supports the remote",
])
def test_holding_an_object_up_is_support_not_containment(phrase):
    """The adversarial half: a surface holds things without containing them."""
    assert interpret_task_effect_predicate(phrase) != "CONTAINS", phrase


@pytest.mark.parametrize("phrase", [
    "potentially_contained_in",
    "may be inside the cupboard",
    "might be in one of the drawers",
    "is possibly located in storage",
])
def test_a_hedged_statement_is_a_guess_about_the_scene(phrase):
    assert relation_states_a_hedged_possibility(phrase), phrase
    assert relation_states_where_things_currently_are(phrase), phrase


@pytest.mark.parametrize("phrase", [
    "The fastener and tool are found inside the storage container.",
    "the parts are kept in the cabinet",
    "found in the left drawer",
    "inside the storage box",
])
def test_where_things_currently_sit_is_not_a_requirement(phrase):
    assert relation_states_where_things_currently_are(phrase), phrase


@pytest.mark.parametrize("phrase", [
    "the component must fit the marked location",
    "the tool is placed on the workbench",
    "the coffee is poured into the cup",
])
def test_a_requirement_is_not_mistaken_for_a_statement_about_the_scene(phrase):
    """The adversarial half: things the task must bring about."""
    assert not relation_states_where_things_currently_are(phrase), phrase


# ---------------------------------------------------------------------------
# Operation wording: the action is not always the first word
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phrase,participants", [
    ("fastening_tool is placed on workbench", ("fastening_tool", "workbench")),
    ("the tool must be returned to the bench", ("tool", "bench")),
    ("the robot arm places the tool on the surface", ("tool", "surface")),
])
def test_a_passive_or_agent_led_phrase_still_names_its_action(phrase, participants):
    assert extract_operation_semantic_candidates("workshop", phrase, participants), phrase


@pytest.mark.parametrize("phrase", [
    "retrieve_from_storage",
    "Open and inspect the storage unit to find components.",
    "The robot arm opens and inspects storage containers to find the part.",
    "verify that the component is compatible",
])
def test_getting_hold_of_something_stowed_away_is_the_search_phase(phrase):
    assert is_non_physical_operation_phrase(phrase, ("storage", "component")), phrase


@pytest.mark.parametrize("phrase", [
    "Place the reusable tool on the workbench surface.",
    "fasten the component at the marked location",
    "Locate the parts and perform the fastening",
])
def test_a_phrase_that_names_a_change_is_not_dismissed_as_perception(phrase):
    """The adversarial half: perception wording in front of a real action."""
    assert not is_non_physical_operation_phrase(phrase, ("tool", "component")), phrase


def test_mixing_is_ambiguous_between_a_transfer_and_a_stirring():
    """Ingredients are mixed into a cup; a drink is mixed with an implement."""
    found = {
        capability.capability_id
        for capability in extract_operation_semantic_candidates(
            "kitchen", "Use stirrer to mix coffee in mug.", ("stirrer", "mug"))
    }
    assert {"STIR_COFFEE", "TRANSFER_CONTENT_TO_CONTAINER"} <= found, found
