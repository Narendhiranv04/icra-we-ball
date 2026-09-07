"""Unit tests for Phase 5: V2 Compiler Integration and Completeness Checking (Gate 5).

Verifies:
1. Complete V2 fixtures across Kitchen, Living Room, and Workshop compile with required_contract_complete == True.
2. Missing a required relation makes required_contract_complete == False.
3. Missing an operation makes required_contract_complete == False.
4. Compiler never synthesizes missing roles, relations, operations, or bindings from reference assumptions.
5. V1 raw replay backward compatibility is fully preserved.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import compile_candidate_graph

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "ideal_raw_vlm"


@pytest.fixture
def v2_kitchen_fixture() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two coffees and two soups for two people.",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_stirrer",
                    "entity_kind": "OBJECT",
                    "function": "stir beverage in cups",
                    "required_count": 1,
                    "binding_policy": "REUSABLE",
                    "candidate_categories": ["spoon", "stirrer"],
                    "required_properties": [],
                },
                {
                    "id": "role_soup_utensil",
                    "entity_kind": "OBJECT",
                    "function": "eating utensil for soup bowl",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["soup spoon", "spoon"],
                    "required_properties": [],
                },
                {
                    "id": "role_coffee_cup",
                    "entity_kind": "OBJECT",
                    "function": "prepared coffee container",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["cup", "mug"],
                    "required_properties": [],
                },
                {
                    "id": "role_soup_bowl",
                    "entity_kind": "OBJECT",
                    "function": "soup bowl container",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["soup bowl", "bowl"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "id": "rel_stir",
                    "subject_role": "role_stirrer",
                    "relation": "inserted into and reaches bottom of cup",
                    "object_role": "role_coffee_cup",
                    "required": True,
                },
                {
                    "id": "rel_soup",
                    "subject_role": "role_soup_utensil",
                    "relation": "inserted into and reaches bottom of soup bowl",
                    "object_role": "role_soup_bowl",
                    "required": True,
                },
            ],
            "operation_pairings": [
                {
                    "id": "op_stir",
                    "operation": "stir beverage in cups",
                    "source_role": "role_stirrer",
                    "target_role": "role_coffee_cup",
                    "operation_count": 2,
                    "reuse_policy": "REUSABLE_ACROSS_TARGETS",
                },
                {
                    "id": "op_soup",
                    "operation": "provide eating utensil for each soup bowl",
                    "source_role": "role_soup_utensil",
                    "target_role": "role_soup_bowl",
                    "operation_count": 2,
                    "reuse_policy": "DEDICATED_PER_TARGET",
                },
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {
                "role_stirrer": [],
                "role_soup_utensil": [],
                "role_coffee_cup": [],
                "role_soup_bowl": [],
            },
            "inspectable_regions": [
                {"id": "drawer_1", "description": "cutlery drawer"},
            ],
            "inspection_order": ["drawer_1"],
        },
        "unsupported_reason": "",
    }


@pytest.fixture
def v2_living_room_fixture() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare living room refreshments and TV remote.",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_side_table",
                    "entity_kind": "REGION",
                    "function": "hold items for viewer",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["side table", "end table"],
                    "required_properties": ["planar horizontal support"],
                },
                {
                    "id": "role_coffee_table",
                    "entity_kind": "REGION",
                    "function": "hold items for viewers",
                    "required_count": 1,
                    "binding_policy": "SHARED",
                    "candidate_categories": ["coffee table", "center table"],
                    "required_properties": ["planar horizontal support"],
                },
                {
                    "id": "role_drinkware",
                    "entity_kind": "OBJECT",
                    "function": "contain hot beverage and saucer",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["cup and saucer set", "cup"],
                    "required_properties": [],
                },
                {
                    "id": "role_remote",
                    "entity_kind": "OBJECT",
                    "function": "control television",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["tv remote", "remote control"],
                    "required_properties": [],
                },
                {
                    "id": "role_seat",
                    "entity_kind": "FIXED_TARGET",
                    "function": "viewer seating position",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["armchair", "chair"],
                    "required_properties": [],
                },
                {
                    "id": "role_seat_pair",
                    "entity_kind": "FIXED_TARGET",
                    "function": "paired viewer seating area",
                    "required_count": 1,
                    "binding_policy": "SHARED",
                    "candidate_categories": ["armchair", "sofa"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "id": "rel_drink",
                    "subject_role": "role_side_table",
                    "relation": "can hold drinkware set",
                    "object_role": "role_drinkware",
                    "required": True,
                },
                {
                    "id": "rel_seat",
                    "subject_role": "role_side_table",
                    "relation": "near seat",
                    "object_role": "role_seat",
                    "required": True,
                },
                {
                    "id": "rel_remote",
                    "subject_role": "role_coffee_table",
                    "relation": "can hold remote",
                    "object_role": "role_remote",
                    "required": True,
                },
                {
                    "id": "rel_pair",
                    "subject_role": "role_coffee_table",
                    "relation": "accessible from both seats",
                    "object_role": "role_seat_pair",
                    "required": True,
                },
            ],
            "operation_pairings": [
                {
                    "id": "op_drink",
                    "operation": "support drinkware set beside seat",
                    "source_role": "role_side_table",
                    "target_role": "role_drinkware",
                    "operation_count": 2,
                    "anchor_role": "role_seat",
                },
                {
                    "id": "op_remote",
                    "operation": "support television remote control",
                    "source_role": "role_coffee_table",
                    "target_role": "role_remote",
                    "operation_count": 1,
                    "anchor_role": "role_seat_pair",
                },
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {
                "role_side_table": [],
                "role_coffee_table": [],
                "role_drinkware": [],
                "role_remote": [],
                "role_seat": [],
                "role_seat_pair": [],
            },
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }


@pytest.fixture
def v2_workshop_fixture() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Repair loose frame joint using compatible screw and driver.",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_driver",
                    "entity_kind": "OBJECT",
                    "function": "drive screw",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["screwdriver", "power driver"],
                    "required_properties": [],
                },
                {
                    "id": "role_fastener",
                    "entity_kind": "OBJECT",
                    "function": "fasten joint",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["screw", "threaded fastener"],
                    "required_properties": [],
                },
                {
                    "id": "role_target",
                    "entity_kind": "FIXED_TARGET",
                    "function": "frame joint repair target",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["frame joint", "screw hole"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "id": "rel_comp",
                    "subject_role": "role_driver",
                    "relation": "compatible with",
                    "object_role": "role_fastener",
                    "required": True,
                },
                {
                    "id": "rel_reach",
                    "subject_role": "role_driver",
                    "relation": "reaches target",
                    "object_role": "role_target",
                    "required": True,
                },
                {
                    "id": "rel_tgt",
                    "subject_role": "role_fastener",
                    "relation": "compatible with target",
                    "object_role": "role_target",
                    "required": True,
                },
            ],
            "operation_pairings": [
                {
                    "id": "op_fasten",
                    "operation": "drive screw into joint target",
                    "source_role": "role_driver",
                    "target_role": "role_fastener",
                    "anchor_role": "role_target",
                    "operation_count": 1,
                    "reuse_policy": "DEDICATED_PER_TARGET",
                }
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {
                "role_driver": [],
                "role_fastener": [],
                "role_target": [],
            },
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }


def test_v2_kitchen_complete_fixture(v2_kitchen_fixture):
    """Complete V2 Kitchen fixture compiles to a complete executable G_F contract."""
    graph = compile_candidate_graph("kitchen", "Kitchen task", v2_kitchen_fixture)
    assert graph.required_contract_complete is True
    assert graph.metadata["is_v2_specification"] is True
    assert len(graph.operation_groups) == 2


def test_v2_kitchen_missing_relation_is_incomplete(v2_kitchen_fixture):
    """Kitchen fixture missing a required relation remains contract-incomplete."""
    # Remove soup relation
    doc = copy.deepcopy(v2_kitchen_fixture)
    doc["task_contract"]["functional_relations"] = [
        r for r in doc["task_contract"]["functional_relations"] if r["id"] != "rel_soup"
    ]
    graph = compile_candidate_graph("kitchen", "Kitchen task", doc)
    assert graph.required_contract_complete is False
    assert len(graph.metadata["contract_missing_reasons"]) > 0


def test_v2_kitchen_missing_operation_is_incomplete(v2_kitchen_fixture):
    """Kitchen fixture missing a required operation remains contract-incomplete."""
    # Remove coffee stirring operation
    doc = copy.deepcopy(v2_kitchen_fixture)
    doc["task_contract"]["operation_pairings"] = [
        op for op in doc["task_contract"]["operation_pairings"] if op["id"] != "op_stir"
    ]
    graph = compile_candidate_graph("kitchen", "Kitchen task", doc)
    assert graph.required_contract_complete is False
    assert any("stir" in r.lower() for r in graph.metadata["contract_missing_reasons"])


def test_v2_living_room_complete_fixture(v2_living_room_fixture):
    """Complete V2 Living Room fixture compiles to a complete executable G_F contract."""
    graph = compile_candidate_graph("living_room", "Living Room task", v2_living_room_fixture)
    assert graph.required_contract_complete is True
    assert graph.metadata["is_v2_specification"] is True
    assert len(graph.roles) >= 4


def test_v2_living_room_missing_seating_relation_is_incomplete(v2_living_room_fixture):
    """Living room fixture missing near-seat relation fails closed and is contract-incomplete."""
    doc = copy.deepcopy(v2_living_room_fixture)
    doc["task_contract"]["functional_relations"] = [
        r for r in doc["task_contract"]["functional_relations"] if r["id"] != "rel_seat"
    ]
    doc["task_contract"]["operation_pairings"][0]["anchor_role"] = None
    graph = compile_candidate_graph("living_room", "Living Room task", doc)
    assert graph.required_contract_complete is False
    assert any("near seat" in r.lower() for r in graph.metadata["contract_missing_reasons"])


def test_v2_living_room_missing_operation_is_incomplete(v2_living_room_fixture):
    """Living room fixture missing drinkware support operation remains contract-incomplete."""
    doc = copy.deepcopy(v2_living_room_fixture)
    doc["task_contract"]["operation_pairings"] = [
        op for op in doc["task_contract"]["operation_pairings"] if op["id"] != "op_drink"
    ]
    graph = compile_candidate_graph("living_room", "Living Room task", doc)
    assert graph.required_contract_complete is False
    assert any("drinkware" in r.lower() for r in graph.metadata["contract_missing_reasons"])


def test_v2_workshop_complete_fixture(v2_workshop_fixture):
    """Complete V2 Workshop fixture compiles to a complete executable G_F contract."""
    graph = compile_candidate_graph("workshop", "Workshop task", v2_workshop_fixture)
    assert graph.required_contract_complete is True
    assert graph.metadata["is_v2_specification"] is True


def test_v2_workshop_missing_relation_is_incomplete(v2_workshop_fixture):
    """Workshop fixture missing a required fastening relation remains contract-incomplete."""
    doc = copy.deepcopy(v2_workshop_fixture)
    doc["task_contract"]["functional_relations"] = [
        r for r in doc["task_contract"]["functional_relations"] if r["id"] != "rel_reach"
    ]
    graph = compile_candidate_graph("workshop", "Workshop task", doc)
    assert graph.required_contract_complete is False
    assert any("reaches_target" in r.lower() or "fastening" in r.lower() for r in graph.metadata["contract_missing_reasons"])


def test_v2_workshop_missing_operation_is_incomplete(v2_workshop_fixture):
    """Workshop fixture missing the fastening operation remains contract-incomplete."""
    doc = copy.deepcopy(v2_workshop_fixture)
    doc["task_contract"]["operation_pairings"] = []
    graph = compile_candidate_graph("workshop", "Workshop task", doc)
    assert graph.required_contract_complete is False
    assert any("fastening" in r.lower() for r in graph.metadata["contract_missing_reasons"])


def test_no_compiler_invention_of_missing_roles():
    """Compiler never synthesizes an omitted role (e.g. omitted remote control)."""
    raw_living_no_remote = {
        "status": "SUPPORTED",
        "task_summary": "Prepare living room seating and drinks only.",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_side_table",
                    "entity_kind": "REGION",
                    "function": "hold items for viewer",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["side table"],
                    "required_properties": ["planar horizontal support"],
                },
                {
                    "id": "role_drinkware",
                    "entity_kind": "OBJECT",
                    "function": "contain hot beverage and saucer",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["cup"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "id": "rel_0",
                    "subject_role": "role_side_table",
                    "relation": "can hold drinkware set",
                    "object_role": "role_drinkware",
                    "required": True,
                }
            ],
            "operation_pairings": [
                {
                    "id": "op_0",
                    "operation": "support drinkware set beside seat",
                    "source_role": "role_side_table",
                    "target_role": "role_drinkware",
                    "operation_count": 2,
                }
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {"role_side_table": [], "role_drinkware": []},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("living_room", "Living room task", raw_living_no_remote)
    assert "ENTERTAINMENT_CONTROL" not in graph.nodes
    assert "role_remote" not in graph.metadata["raw_role_to_canonical"]
    assert graph.required_contract_complete is False


def test_v1_raw_replay_backward_compatibility():
    """V1 raw replay fixtures compile cleanly through the production compiler pipeline."""
    for dom, fixture_file in [
        ("kitchen", "kitchen_K1.json"),
        ("living_room", "living_room_L1.json"),
        ("workshop", "workshop_W1.json"),
    ]:
        p = FIXTURES_DIR / fixture_file
        if not p.exists():
            continue
        raw_v1 = json.loads(p.read_text(encoding="utf-8"))
        graph = compile_candidate_graph(dom, "Run task", raw_v1)
        assert graph.metadata["is_v2_specification"] is False
        assert len(graph.roles) >= 2
        assert graph.required_contract_complete is True
