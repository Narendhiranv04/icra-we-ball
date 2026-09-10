"""Every role reference must be normalized the same way, at every depth.

The sanitizer rewrites role identifiers to one canonical spelling so the rest of
the compiler can index them.  It normalized a role's own id and an interaction
group's endpoints but not the capability reading recorded alongside the group,
which names roles too.  A contract whose role ids were not already lowercase
slugs -- ``Coffee_Cup``, ``CoffeeContainer`` -- therefore had every one of its
operations rejected as naming a participant the runtime holds in no form, while
that participant sat in the compiled graph under its normalized name.

The FM is asked for the task's meaning, not for our identifier convention, so
how it spells a role id must not decide whether its operations survive.

No reference graph, expected plan or benchmark variant appears here.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    convert_v3_to_canonical_document,
    normalize_and_validate_v3_contract,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph
from mujoco_scenes.functional_tamp_pipeline.structural_sanitizer import (
    normalize_id,
    sanitize_functional_graph,
)


KITCHEN = ("Prepare and serve one coffee and one soup for each of two people. Make each "
           "coffee using coffee and water and stir it before serving. Serve each soup bowl "
           "with its own suitable eating utensil.")


def _contract(cup_id: str, water_id: str, spoon_id: str, bowl_id: str):
    return {
        "schema_version": 3, "status": "SUPPORTED", "task_summary": "two coffees and two soups",
        "task_contract": {
            "functional_roles": [
                {"id": cup_id, "entity_kind": "OBJECT", "function": "Hold the prepared coffee.",
                 "description": "", "required_count": 2, "binding_policy": "DISTINCT",
                 "candidate_categories": ["mug", "cup"], "required_properties": []},
                {"id": water_id, "entity_kind": "OBJECT", "function": "Provide the liquid base.",
                 "description": "", "required_count": 2, "binding_policy": "REUSABLE",
                 "candidate_categories": ["kettle", "jug"], "required_properties": []},
                {"id": spoon_id, "entity_kind": "OBJECT", "function": "Stir the coffee.",
                 "description": "", "required_count": 2, "binding_policy": "DISTINCT",
                 "candidate_categories": ["spoon"], "required_properties": []},
                {"id": bowl_id, "entity_kind": "OBJECT", "function": "Hold the prepared soup.",
                 "description": "", "required_count": 2, "binding_policy": "DISTINCT",
                 "candidate_categories": ["bowl"], "required_properties": []},
            ],
            "functional_relations": [],
            "operation_pairings": [
                {"id": "add_water", "operation": "Pour",
                 "participant_roles": [cup_id, water_id], "operation_count": 2},
                {"id": "stir", "operation": "Stir",
                 "participant_roles": [cup_id, spoon_id], "operation_count": 2},
            ],
        },
        "observation_guidance": {"visible_candidates_per_role": {},
                                 "inspectable_regions": [], "inspection_order": []},
        "unsupported_reason": "",
    }


SPELLINGS = [
    ("coffee_cup", "water", "stirring_spoon", "soup_bowl"),
    ("Coffee_Cup", "Water", "Stirring_Spoon", "Soup_Bowl"),
    ("CoffeeCup", "Water", "StirringSpoon", "SoupBowl"),
    ("coffee cup", "water", "stirring spoon", "soup bowl"),
    ("Coffee-Cup", "Water", "Stirring-Spoon", "Soup-Bowl"),
]


def _compile(raw):
    normalized, _ = normalize_and_validate_v3_contract(raw, domain="kitchen", task_instruction=KITCHEN)
    canonical = convert_v3_to_canonical_document(normalized, domain="kitchen", task_instruction=KITCHEN)
    return compile_candidate_graph("kitchen", KITCHEN, canonical)


@pytest.mark.parametrize("ids", SPELLINGS)
def test_operations_survive_however_the_model_spells_its_role_ids(ids):
    graph = _compile(_contract(*ids))
    functions = sorted(group.function for group in graph.operation_groups)
    assert functions == ["POUR", "STIR_COFFEE"], (
        f"spelling {ids[0]!r} lost operations: {functions}")


@pytest.mark.parametrize("ids", SPELLINGS)
def test_no_operation_is_rejected_for_naming_a_represented_participant(ids):
    graph = _compile(_contract(*ids))
    trace = (graph.metadata or {}).get("canonicalization_trace", {}) or {}
    represented = {role.raw_role_id for role in graph.nodes.values()}
    for row in trace.get("unresolved_required_operations", ()):
        for name in row.get("unrepresented_participants", ()):
            assert normalize_id(name) not in represented, (
                f"{name!r} was reported unrepresented but is in the graph as "
                f"{normalize_id(name)!r}")


def test_the_capability_reading_beside_a_group_is_normalized_with_it():
    """The recorded slot assignment must name roles the same way the group does."""
    raw = _contract("Coffee_Cup", "Water", "Stirring_Spoon", "Soup_Bowl")
    normalized, _ = normalize_and_validate_v3_contract(raw, domain="kitchen", task_instruction=KITCHEN)
    canonical = convert_v3_to_canonical_document(normalized, domain="kitchen", task_instruction=KITCHEN)
    sanitized = sanitize_functional_graph(canonical, domain=None)
    document = sanitized.document if sanitized.succeeded else canonical
    ids = {role["id"] for role in document["functional_roles"]}
    for group in document["interaction_groups"]:
        for row in group.get("v3_slot_assignments", ()):
            for key in ("source_role", "target_role", "anchor_role"):
                value = row.get(key)
                if value:
                    assert value in ids or value.startswith("op_slot__"), (
                        f"{key}={value!r} is not one of the document's role ids")
            for key in ("v3_participant_roles", "v3_explicit_participant_roles"):
                for name in group.get(key, ()):
                    assert name in ids or name.startswith("op_slot__"), name


def test_normalization_is_still_a_normalization_and_not_a_rename():
    assert normalize_id("Coffee_Cup") == "coffee_cup"
    assert normalize_id("  Stirring Spoon ") == "stirring_spoon"
    assert normalize_id("Soup-Bowl") == "soup_bowl"
    assert normalize_id("coffee_cup") == "coffee_cup"
