"""Structural wire noise is repaired; missing semantics are not invented.

The line between the two is the whole point. A participant reference that
denotes nothing the runtime models -- the robot, a person, the room the task
happens in -- carries no semantics, and rejecting an otherwise coherent
contract over it threw away everything the model did express. A reference that
names something the robot could actually act on is missing semantics: its kind,
its cardinality and what would count as an instance of it are all absent, and
every one of those would have to be chosen rather than read.

So each repair here is paired with the case that must still fail closed.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
    normalize_and_validate_v3_contract,
)


KITCHEN = ("Prepare and serve one coffee and one soup for each of two people. Make each "
           "coffee using coffee and water and stir it before serving. Serve each soup bowl "
           "with its own suitable eating utensil.")


def role(rid, function, *, kind="OBJECT", count=1, policy="REUSABLE", categories=()):
    return {"id": rid, "entity_kind": kind, "function": function, "description": "",
            "required_count": count, "binding_policy": policy,
            "candidate_categories": list(categories), "required_properties": []}


def _region(rid):
    return {"id": rid, "label": rid.lower(),
            "visual_description": f"a closed {rid.lower()}",
            "reason": "it is closed, so its contents cannot be seen"}


def document(roles, operations=(), relations=(), regions=(), order=None):
    return {"schema_version": 3, "status": "SUPPORTED", "task_summary": "task",
            "task_contract": {"functional_roles": list(roles),
                              "functional_relations": list(relations),
                              "operation_pairings": list(operations)},
            "observation_guidance": {
                "visible_candidates_per_role": {},
                "inspectable_regions": list(regions),
                "inspection_order": list(order if order is not None
                                         else [r["id"] for r in regions])},
            "unsupported_reason": ""}


def _kitchen_roles():
    return [
        role("mug", "container for the coffee", count=2, policy="DISTINCT",
             categories=["mug"]),
        role("coffee", "ingredient supplying the coffee", count=2, categories=["coffee"]),
        role("stirrer", "implement used to stir the drink", count=2, policy="DISTINCT",
             categories=["spoon"]),
    ]


def _normalize(raw):
    return normalize_and_validate_v3_contract(raw, domain="kitchen", task_instruction=KITCHEN)


@pytest.mark.parametrize("undeclared", ["Person", "person", "people", "viewer_2",
                                        "living_room", "kitchen", "room", "scene"])
def test_a_reference_to_a_non_participant_is_removed_and_recorded(undeclared):
    raw = document(_kitchen_roles(), operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
        {"id": "serve", "operation": "stir the coffee in the mug",
         "participant_roles": ["stirrer", "mug", undeclared], "operation_count": 2},
    ])
    normalized, trace = _normalize(raw)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    assert any(note.get("code") == "UNDECLARED_NON_PARTICIPANT_REMOVED"
               and undeclared in (note.get("removed_participants") or [])
               for note in notes), notes
    served = next(o for o in normalized["task_contract"]["operation_pairings"]
                  if o["id"] == "serve")
    assert undeclared not in served["participant_roles"]


@pytest.mark.parametrize("undeclared", ["target_assembly", "cup", "plate",
                                        "fastening_component", "side_table"])
def test_a_reference_to_something_the_robot_could_act_on_still_fails_closed(undeclared):
    """The adversarial half: this one is missing semantics, not noise."""
    raw = document(_kitchen_roles(), operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
        {"id": "serve", "operation": "stir the coffee in the mug",
         "participant_roles": ["stirrer", "mug", undeclared], "operation_count": 2},
    ])
    with pytest.raises(MalformedVLMSpecificationError, match="UNDECLARED_PARTICIPANT"):
        _normalize(raw)


def test_the_same_region_declared_twice_is_one_region():
    regions = [
        _region("Cabinet"), _region("Cabinet"), _region("Drawer"),
    ]
    raw = document(_kitchen_roles(), operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
    ], regions=regions, order=["Cabinet", "Drawer"])
    normalized, trace = _normalize(raw)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    assert any(note.get("code") == "DUPLICATE_INSPECTION_REGION_DECLARATION_COLLAPSED"
               and note.get("region_id") == "Cabinet" for note in notes), notes
    ids = [r["id"] for r in normalized["observation_guidance"]["inspectable_regions"]]
    assert ids == ["Cabinet", "Drawer"]


def test_two_different_regions_are_not_collapsed():
    """The adversarial half: distinct ids are distinct places."""
    regions = [
        _region("Cabinet"), _region("Drawer"),
    ]
    raw = document(_kitchen_roles(), operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
    ], regions=regions)
    normalized, _ = _normalize(raw)
    ids = [r["id"] for r in normalized["observation_guidance"]["inspectable_regions"]]
    assert ids == ["Cabinet", "Drawer"]


def test_an_order_that_omits_a_declared_region_is_repaired_not_refused():
    """Pre-existing behaviour, pinned here: the forgotten region is searched last."""
    regions = [_region("Cabinet"), _region("Drawer")]
    raw = document(_kitchen_roles(), operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
    ], regions=regions, order=["Cabinet"])
    normalized, trace = _normalize(raw)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    assert any(note.get("code") == "INSPECTION_ORDER_NORMALIZED" for note in notes), notes
    assert normalized["observation_guidance"]["inspection_order"] == ["Cabinet", "Drawer"]


def test_a_repeated_region_id_keeps_the_first_declaration_and_records_the_other():
    """No merging: two declarations under one id are not reconciled into a third."""
    first = {**_region("Cabinet"), "label": "wall cabinet left",
             "visual_description": "closed grey box on the wall"}
    second = {**_region("Cabinet"), "label": "wall cabinet right",
              "visual_description": "a different closed box"}
    raw = document(_kitchen_roles(), operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
    ], regions=[first, second], order=["Cabinet"])
    normalized, trace = _normalize(raw)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    collapsed = [n for n in notes
                 if n.get("code") == "DUPLICATE_INSPECTION_REGION_DECLARATION_COLLAPSED"]
    assert collapsed, notes
    assert collapsed[0]["discarded_declaration"]["label"] == "wall cabinet right"
    kept = normalized["observation_guidance"]["inspectable_regions"]
    assert [r["id"] for r in kept] == ["Cabinet"]
    assert kept[0]["label"] == "wall cabinet left"


# ---------------------------------------------------------------------------
# An undeclared reference to a place the runtime already owns
# ---------------------------------------------------------------------------


LIVING = ("Prepare the living room for two people to enjoy refreshments while watching "
          "television. Provide each person with their own refreshment setting nearby, and "
          "place the entertainment control where it is accessible to both people.")


def _living_document(undeclared):
    roles = [
        role("refreshment_setting", "personal drinkware set for one person",
             count=2, policy="DISTINCT", categories=["cup", "saucer"]),
        role("side_table", "personal side table beside the seat", kind="REGION",
             count=2, policy="DISTINCT", categories=["side table"]),
    ]
    return {"schema_version": 3, "status": "SUPPORTED", "task_summary": "task",
            "task_contract": {
                "functional_roles": roles,
                "functional_relations": [
                    {"id": "near", "relation": "the side table is near the seat",
                     "participant_roles": ["side_table", undeclared], "required": True}],
                "operation_pairings": [
                    {"id": "serve", "operation": "place the drinkware set on the side table",
                     "participant_roles": ["refreshment_setting", "side_table"],
                     "operation_count": 2}]},
            "observation_guidance": {"visible_candidates_per_role": {},
                                     "inspectable_regions": [], "inspection_order": []},
            "unsupported_reason": ""}


@pytest.mark.parametrize("undeclared", ["seat", "seats", "seating_area", "chair"])
def test_an_undeclared_reference_to_the_seating_anchor_is_declared(undeclared):
    """Its kind and its fixedness come from the registry, not from a guess."""
    normalized, trace = normalize_and_validate_v3_contract(
        _living_document(undeclared), domain="living_room", task_instruction=LIVING)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    recovered = [n for n in notes
                 if n.get("code") == "UNDECLARED_FIXED_ANCHOR_REFERENCE_DECLARED"]
    assert recovered, notes
    assert recovered[0]["anchor_family"] == "SEATING"
    declared = {r["id"] for r in normalized["task_contract"]["functional_roles"]}
    assert undeclared in declared


@pytest.mark.parametrize("undeclared", ["cup", "plate", "drink", "surface",
                                        "side_board", "television"])
def test_an_undeclared_reference_to_something_findable_is_not_declared(undeclared):
    """The adversarial half, and the reason the line is where it is.

    A seat is a calibrated reference the runtime owns; a cup is something the
    robot would have had to go and find, so its cardinality, its binding policy
    and what would count as an instance of it would all have to be chosen here.
    """
    with pytest.raises(MalformedVLMSpecificationError, match="UNDECLARED_PARTICIPANT"):
        normalize_and_validate_v3_contract(
            _living_document(undeclared), domain="living_room", task_instruction=LIVING)


def test_a_domain_whose_families_all_have_selectable_roles_recovers_nothing():
    """Kitchen realizes every family it recognizes with a role grounding chooses."""
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import _anchor_only_families

    assert _anchor_only_families("kitchen") == set()
    assert _anchor_only_families("living_room") == {"SEATING"}
    assert _anchor_only_families("workshop") == {"FIXED_TARGET"}


def test_two_elements_sharing_an_id_are_both_kept():
    """An id is a label the contract uses to refer to itself, not a semantic."""
    raw = document(_kitchen_roles(), relations=[
        {"id": "same", "relation": "the stirrer reaches the bottom of the mug",
         "participant_roles": ["stirrer", "mug"], "required": True},
        {"id": "same", "relation": "the coffee is poured into the mug",
         "participant_roles": ["coffee", "mug"], "required": True},
    ], operations=[
        {"id": "pour", "operation": "pour the coffee into the mug",
         "participant_roles": ["coffee", "mug"], "operation_count": 2},
    ])
    normalized, trace = _normalize(raw)
    notes = trace if isinstance(trace, list) else (trace or {}).get("repairs", []) or []
    renamed = [n for n in notes if n.get("code") == "DUPLICATE_ELEMENT_ID_RENAMED"]
    assert renamed, notes
    assert renamed[0]["raw_id"] == "same" and renamed[0]["normalized_id"] == "same_2"
    ids = [r["id"] for r in normalized["task_contract"]["functional_relations"]]
    assert ids == ["same", "same_2"], ids
    # Both relations survive with their own wording.
    texts = {r["id"]: r["relation"] for r in normalized["task_contract"]["functional_relations"]}
    assert "reaches the bottom" in texts["same"]
    assert "poured into" in texts["same_2"]
