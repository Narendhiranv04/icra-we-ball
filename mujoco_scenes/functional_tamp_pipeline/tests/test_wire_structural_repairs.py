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
