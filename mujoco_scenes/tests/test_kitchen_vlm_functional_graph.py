"""Regression tests for the natural-language Kitchen VLM functional specification path."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest
from mujoco_scenes.kitchen_vlm_functional_graph import (
    compile_vlm_functional_graph,
    resolve_kitchen_region_proposal,
    map_unary_property,
    map_binary_relation,
)
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
REGIONS = ("D1", "D2", "C2", "B1", "C1")


def natural_kitchen_spec() -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two coffees and two soups using available kitchenware.",
        "functional_roles": [
            {
                "id": "drink_receptacle",
                "entity_kind": "OBJECT",
                "function": "contain one coffee serving",
                "required_count": 2,
                "reuse_policy": "DISTINCT",
                "candidate_categories": ["cup", "coffee mug"],
                "required_properties": ["open cavity", "capable of containing liquid"],
            },
            {
                "id": "soup_receptacle",
                "entity_kind": "OBJECT",
                "function": "contain one soup serving",
                "required_count": 2,
                "reuse_policy": "DISTINCT",
                "candidate_categories": ["bowl", "soup bowl"],
                "required_properties": ["open cavity", "holds liquid"],
            },
            {
                "id": "mixing_implement",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "required_count": 1,
                "reuse_policy": "REUSABLE",
                "candidate_categories": ["spoon", "metal spoon"],
                "required_properties": ["elongated", "long utensil"],
            },
            {
                "id": "soup_implement",
                "entity_kind": "OBJECT",
                "function": "serve with soup",
                "required_count": 2,
                "reuse_policy": "DISTINCT",
                "candidate_categories": ["soup_spoon", "soup spoon"],
                "required_properties": ["elongated utensil"],
            },
            {
                "id": "water_source",
                "entity_kind": "OBJECT",
                "function": "provide water for coffee",
                "required_count": 1,
                "reuse_policy": "REUSABLE",
                "candidate_categories": ["kettle", "water jug"],
                "required_properties": [],
            },
            {
                "id": "coffee_source",
                "entity_kind": "OBJECT",
                "function": "provide coffee material",
                "required_count": 1,
                "reuse_policy": "REUSABLE",
                "candidate_categories": ["coffee_jar", "coffee jar"],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "mixing_implement",
                "relation": "must fit inside and reach bottom",
                "object_role": "drink_receptacle",
            },
            {
                "subject_role": "soup_implement",
                "relation": "fits into opening",
                "object_role": "soup_receptacle",
            },
        ],
        "interaction_groups": [
            {
                "id": "mix_drinks",
                "function": "stir",
                "tool_role": "mixing_implement",
                "target_role": "drink_receptacle",
                "required_target_count": 2,
                "reuse_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["fits inside", "reaches bottom"],
            },
            {
                "id": "equip_soups",
                "function": "provide utensil",
                "tool_role": "soup_implement",
                "target_role": "soup_receptacle",
                "required_target_count": 2,
                "reuse_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["fits inside"],
            },
        ],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [
            {
                "id": "reg_1",
                "label": "upper wall cupboard",
                "visual_description": "cupboard above counter",
                "reason": "may hold vessels",
            },
            {
                "id": "reg_2",
                "label": "upper drawer",
                "visual_description": "top drawer below counter",
                "reason": "may hold utensils",
            },
        ],
        "inspection_order": ["reg_1", "reg_2"],
        "initial_satisfaction_assessment": False,
        "initial_satisfaction_reason": "Not all roles are visibly satisfied.",
        "unsupported_reason": "",
    }


qwen_graph = natural_kitchen_spec


class FakeTransport:
    def __init__(self, document: dict):
        self.document = document
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": json.dumps(self.document)}}]}


def test_natural_kitchen_spec_canonicalizes_properties_and_regions():
    contract, vocabularies, trace = compile_vlm_functional_graph(
        natural_kitchen_spec(),
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )

    assert set(contract["roles"]) == {
        "drink_receptacle", "soup_receptacle", "mixing_implement", "soup_implement",
        "water_source", "coffee_source",
    }
    assert contract["specification_source"] == "qwen_vlm_natural_language_specification"
    assert contract["symbolic_task"]["target_requirements"]["coffee"]["witness_role"] == "drink_receptacle"
    assert contract["symbolic_task"]["source_roles"]["water_source"]["witness_role"] == "water_source"
    assert contract["roles"]["drink_receptacle"]["unary_geometry"][0]["predicate"] == "OPEN_CAVITY"
    assert contract["roles"]["mixing_implement"]["unary_geometry"][0]["predicate"] == "ELONGATED_OBJECT"
    
    # Check relations are canonicalized
    rel_preds = [r["predicate"] for r in contract["relations"]]
    assert "INSERTABLE_IN" in rel_preds
    assert "REACHES_BOTTOM" in rel_preds

    # Check candidate regions contain ONLY resolved regions
    assert trace["candidate_regions"] == ["C2", "D1"]
    assert trace["inspection_order"] == ["C2", "D1"]


def test_adapter_outgoing_payload_has_zero_checker_and_region_leaks(tmp_path):
    image = tmp_path / "initial.png"
    image.write_bytes(PNG_1X1)
    transport = FakeTransport(natural_kitchen_spec())
    adapter = FMAdapter(model="qwen", transport=transport)

    result = adapter.generate_kitchen_functional_graph(
        "Prepare two coffees and two soups.", observation_images=[image]
    )

    assert result == natural_kitchen_spec()
    assert adapter.metrics.total_calls == 1
    assert len(transport.payloads) == 1

    payload_json = json.dumps(transport.payloads[0])
    
    # Assert NO checker names in payload
    forbidden_checkers = [
        "OPEN_CAVITY", "ELONGATED_OBJECT", "INSERTABLE_IN", "REACHES_BOTTOM",
        "PLANAR_SUPPORT", "CAN_DRIVE_SCREW", "CAN_FASTEN", "total_length_m", "cavity_depth_m"
    ]
    for checker in forbidden_checkers:
        assert checker not in payload_json, f"Information leak detected: {checker} in payload"

    # Assert NO canonical region IDs in payload
    forbidden_regions = ["D1", "D2", "C2", "B1", "C1", "LEFT_DRAWER", "RIGHT_DRAWER", "TOOL_CABINET"]
    for reg in forbidden_regions:
        assert f'"{reg}"' not in payload_json, f"Information leak detected: region {reg} in payload"


def test_unsupported_task_fails_closed():
    spec = natural_kitchen_spec()
    spec["status"] = "UNSUPPORTED"
    spec["unsupported_reason"] = "Cannot serve food without ingredients"

    with pytest.raises(ValueError, match="VLM marked task unsupported"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_inconsistent_role_count_fails_closed():
    spec = natural_kitchen_spec()
    # Mismatch: role count is 1, but operation requires 2
    spec["functional_roles"][0]["required_count"] = 1
    spec["interaction_groups"][0]["required_target_count"] = 2

    with pytest.raises(ValueError, match="VLM specification inconsistency"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=REGIONS,
        )


def test_unresolved_region_proposal_excluded_from_candidate_regions():
    spec = natural_kitchen_spec()
    spec["inspectable_regions"] = [
        {"id": "reg_1", "label": "upper wall cupboard", "visual_description": "cupboard", "reason": "storage"},
        {"id": "reg_2", "label": "bookshelf in bedroom", "visual_description": "bookshelf", "reason": "storage"},
    ]
    spec["inspection_order"] = ["reg_1", "reg_2"]

    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )

    # Only C2 should be in candidate_regions
    assert trace["candidate_regions"] == ["C2"]
    assert trace["inspection_order"] == ["C2"]
    assert len(trace["unresolved_proposals"]) == 1
    assert trace["unresolved_proposals"][0]["label"] == "bookshelf in bedroom"


def test_no_full_catalog_fallback():
    spec = natural_kitchen_spec()
    # VLM proposes only 1 region
    spec["inspectable_regions"] = [
        {"id": "reg_1", "label": "upper wall cupboard", "visual_description": "cupboard", "reason": "storage"},
    ]
    spec["inspection_order"] = ["reg_1"]

    contract, vocabularies, trace = compile_vlm_functional_graph(
        spec,
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=REGIONS,
    )

    # Must NOT fall back to all 5 regions
    assert trace["candidate_regions"] == ["C2"]
    assert len(trace["candidate_regions"]) == 1

