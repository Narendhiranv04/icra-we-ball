"""Stage 3: Robust Role Canonicalization Using Real FM Language and Paraphrases.

Validates that:
1. Real-language natural FM paraphrases across Kitchen, Living Room, and Workshop
   canonicalize to their correct domain roles without benchmark variant ID leakage.
2. Causal head prioritization prevents collisions (e.g. coffee container vs coffee stirrer).
3. Unreferenced duplicate roles merge cleanly without crashing on ambiguous role mapping.
4. Fixed targets and contextual regions remain distinct from movable payload/tools.
5. Role mapping never synthesizes endpoint-only operations or relations.
"""

from __future__ import annotations

import pytest

from mujoco_scenes.kitchen_vlm_functional_graph import map_kitchen_role_function
from mujoco_scenes.environment_vlm_requirements import (
    map_living_room_object_payload_role,
    map_living_room_role_function,
    map_living_room_fixed_target_role,
)
from mujoco_scenes.workshop_phase1.requirements import (
    map_workshop_role_function,
    map_workshop_fixed_target_role,
    map_workshop_context_region_role,
)
from mujoco_scenes.functional_tamp_pipeline.semantic_compiler import (
    compile_candidate_graph,
    can_merge_roles,
)


# ---------------------------------------------------------------------------
# Section 10.1: Real-language regression fixtures (minimum 6 specified in plan)
# ---------------------------------------------------------------------------

def test_paraphrase_soup_serving_container():
    """Paraphrase 1: 'Holds soup ready for consumption.' -> soup_container."""
    raw = {
        "id": "soup_serving_vessel",
        "entity_kind": "OBJECT",
        "function": "Holds soup ready for consumption.",
        "candidate_categories": ["bowl", "dish", "plate with soup"],
    }
    assert map_kitchen_role_function(raw) == "soup_container"


def test_paraphrase_coffee_stirring_implement():
    """Paraphrase 2: 'Tool used to mix ingredients inside the coffee serving vessel.' -> coffee_stirrer."""
    raw = {
        "id": "stirring_utensil",
        "entity_kind": "OBJECT",
        "function": "Tool used to mix ingredients inside the coffee serving vessel.",
        "candidate_categories": ["spoon", "stirrer", "whisk"],
    }
    assert map_kitchen_role_function(raw) == "coffee_stirrer"


def test_paraphrase_entertainment_operating_device():
    """Paraphrase 3: 'Device used to operate the television or media system.' -> REMOTE."""
    raw = {
        "id": "entertainment_device",
        "entity_kind": "OBJECT",
        "function": "Device used to operate the television or media system.",
        "candidate_categories": ["remote control", "media controller"],
    }
    assert map_living_room_object_payload_role(raw) == "REMOTE"


def test_paraphrase_personal_consumption_item_collection():
    """Paraphrase 4: 'A collection of items designated for consumption by one person.' -> CUP_SAUCER_SET."""
    raw = {
        "id": "refreshment_setting",
        "entity_kind": "OBJECT",
        "function": "A collection of items designated for consumption by one person.",
        "candidate_categories": ["cup and saucer", "mug and plate"],
    }
    assert map_living_room_object_payload_role(raw) == "CUP_SAUCER_SET"


def test_paraphrase_fastening_manipulation_implement():
    """Paraphrase 5: 'An implement used to manipulate or install the fastening component.' -> CAN_DRIVE_SCREW."""
    raw = {
        "id": "driving_implement",
        "entity_kind": "OBJECT",
        "function": "An implement used to manipulate or install the fastening component.",
        "candidate_categories": ["driver", "screwdriver", "tool"],
    }
    assert map_workshop_role_function(raw) == "CAN_DRIVE_SCREW"


def test_paraphrase_fastening_connecting_physical_item():
    """Paraphrase 6: 'A physical item capable of connecting or securing elements at the marked location.' -> CAN_FASTEN."""
    raw = {
        "id": "connecting_fastener",
        "entity_kind": "OBJECT",
        "function": "A physical item capable of connecting or securing elements at the marked location.",
        "candidate_categories": ["bolt", "screw", "fastener"],
    }
    assert map_workshop_role_function(raw) == "CAN_FASTEN"


# ---------------------------------------------------------------------------
# Section 10.3: Collision logic & causal head prioritization
# ---------------------------------------------------------------------------

def test_kitchen_causal_head_prevents_container_stirrer_collision():
    """Container with stirring mention must NOT collide with the stirring tool."""
    container_role = {
        "id": "coffee_serving_vessel",
        "entity_kind": "OBJECT",
        "function": "Receives coffee and water ingredients and is the target of stirring action.",
        "candidate_categories": ["mug", "cup", "glass"],
    }
    tool_role = {
        "id": "stirring_utensil",
        "entity_kind": "OBJECT",
        "function": "Tool used to mix ingredients inside the coffee serving vessel.",
        "candidate_categories": ["spoon", "stirrer", "whisk"],
    }
    assert map_kitchen_role_function(container_role) == "coffee_container"
    assert map_kitchen_role_function(tool_role) == "coffee_stirrer"


def test_kitchen_causal_head_full_compilation_coexistence():
    """Verify both coffee_container and coffee_stirrer compile without AMBIGUOUS_ROLE_MAPPING."""
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "Prepare coffee and soup",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "vessel",
                    "entity_kind": "OBJECT",
                    "function": "Receives coffee and water ingredients and is the target of stirring action.",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["cup", "mug"],
                    "required_properties": [],
                },
                {
                    "id": "tool",
                    "entity_kind": "OBJECT",
                    "function": "Tool used to mix ingredients inside the coffee serving vessel.",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["spoon", "stirrer"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "subject_role": "tool",
                    "relation": "fits inside",
                    "object_role": "vessel",
                    "required": True,
                },
            ],
            "operation_pairings": [
                {
                    "id": "op_stir",
                    "operation": "stir beverage",
                    "source_role": "tool",
                    "target_role": "vessel",
                    "operation_count": 2,
                    "reuse_policy": "DEDICATED_PER_TARGET",
                },
            ],
        },
        "observation_guidance": {
            "inspectable_regions": [],
            "inspection_order": [],
            "visible_candidates_per_role": {},
        },
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("kitchen", "K2", raw_doc)
    assert "coffee_container" in graph.nodes
    assert "coffee_stirrer" in graph.nodes
    assert not any(u.get("code") == "AMBIGUOUS_ROLE_MAPPING" for u in graph.metadata.get("unresolved_roles", []))


# ---------------------------------------------------------------------------
# Section 10.5: Living room role canonicalization and duplicate role merge
# ---------------------------------------------------------------------------

def test_living_room_payload_and_region_roles():
    """Verify living room payload, region, and seating anchors map robustly."""
    remote = {
        "id": "entertainment_controller",
        "entity_kind": "OBJECT",
        "function": "Device used to operate the television.",
        "candidate_categories": ["remote control"],
    }
    refreshment = {
        "id": "refreshment_pair",
        "entity_kind": "OBJECT",
        "function": "A set comprising a drink vessel and a serving dish intended for consumption.",
        "candidate_categories": ["cup and saucer"],
    }
    shared_region = {
        "id": "center_table",
        "entity_kind": "REGION",
        "function": "Central destination surface for the entertainment control device.",
        "required_count": 1,
        "binding_policy": "SHARED",
    }
    personal_region = {
        "id": "side_table",
        "entity_kind": "REGION",
        "function": "refreshment support near seat",
        "required_count": 2,
        "binding_policy": "DISTINCT",
    }
    seating = {
        "id": "seat_ref",
        "entity_kind": "OBJECT",
        "function": "Fixed reference point representing the seating positions for the users.",
        "candidate_categories": ["armchair"],
    }

    assert map_living_room_object_payload_role(remote) == "REMOTE"
    assert map_living_room_object_payload_role(refreshment) == "CUP_SAUCER_SET"
    assert map_living_room_role_function(shared_region) == "SHARED_REMOTE_REGION"
    assert map_living_room_role_function(personal_region) == "PERSONAL_CUP_SAUCER_REGION"
    assert map_living_room_fixed_target_role(seating) == "SEATING_POSITION"


def test_unreferenced_duplicate_role_merges_cleanly():
    """In L1 forensic trace, role_2 and role_3 have identical function, but role_3 is unreferenced.

    Verify can_merge_roles allows merging rather than crashing with AMBIGUOUS_ROLE_MAPPING.
    """
    role_2 = {
        "id": "role_2",
        "entity_kind": "REGION",
        "function": "support entertainment control",
        "required_count": 1,
        "binding_policy": "SHARED",
    }
    role_3 = {
        "id": "role_3",
        "entity_kind": "REGION",
        "function": "support entertainment control",
        "required_count": 1,
        "binding_policy": "SHARED",
    }
    doc = {
        "interaction_groups": [
            {"target_role": "role_2", "tool_role": "role_1"},
        ],
        "functional_relations": [],
    }
    assert can_merge_roles(role_2, role_3, doc) is True


# ---------------------------------------------------------------------------
# Section 10.6: Workshop distinct roles
# ---------------------------------------------------------------------------

def test_workshop_distinct_roles():
    """Verify tool, fastener, fixed target, and support workbench remain distinct."""
    driver = {
        "id": "tool",
        "entity_kind": "OBJECT",
        "function": "An implement used to manipulate or install the fastening component.",
        "candidate_categories": ["driver", "screwdriver"],
    }
    fastener = {
        "id": "fastener",
        "entity_kind": "OBJECT",
        "function": "A physical item capable of connecting or securing elements at the marked location.",
        "candidate_categories": ["screw", "bolt"],
    }
    target = {
        "id": "hole",
        "entity_kind": "REGION",
        "function": "receive fastening component",
    }
    workbench = {
        "id": "bench",
        "entity_kind": "REGION",
        "function": "stable support surface for workbench operations",
    }

    assert map_workshop_role_function(driver) == "CAN_DRIVE_SCREW"
    assert map_workshop_role_function(fastener) == "CAN_FASTEN"
    assert map_workshop_fixed_target_role(target) == "repair_target"
    assert map_workshop_context_region_role(workbench) == "MAIN_WORKBENCH_ZONE"


# ---------------------------------------------------------------------------
# Invariant: No operation or relation synthesis from roles alone
# ---------------------------------------------------------------------------

def test_roles_only_do_not_synthesize_operations_or_relations():
    """Verify contract with only roles and no operations has zero operations."""
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "Roles only contract",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "soup_bowl",
                    "entity_kind": "OBJECT",
                    "function": "Holds soup ready for consumption.",
                    "required_count": 2,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["bowl"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [],
            "operation_pairings": [],
        },
        "observation_guidance": {
            "inspectable_regions": [],
            "inspection_order": [],
            "visible_candidates_per_role": {},
        },
        "unsupported_reason": "",
    }
    graph = compile_candidate_graph("kitchen", "K1", raw_doc)
    assert len(graph.operation_groups) == 0
    assert len(graph.relations) == 0
    assert graph.metadata["online_executable_contract_complete"] is True
