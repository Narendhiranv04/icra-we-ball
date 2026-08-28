"""Regression tests for the one-call, no-reviewed-contract Kitchen VLM path."""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest
from mujoco_scenes.kitchen_vlm_functional_graph import (
    compile_vlm_functional_graph,
)
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
REGIONS = {"D1": "upper drawer", "C2": "closed cupboard"}


def qwen_graph() -> dict:
    category = lambda label, *phrases: {
        "canonical_label": label,
        "detector_phrases": list(phrases) or [label.replace("_", " ")],
    }
    return {
        "status": "SUPPORTED",
        "task_summary": "Prepare two coffees and two soups.",
        "roles": [
            {
                "id": "drink_receptacle",
                "entity_kind": "OBJECT",
                "function": "contain one coffee serving",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "verification_mode": "SEMANTIC_AND_GEOMETRIC",
                "candidate_categories": [category("cup", "cup", "coffee mug")],
                "unary_properties": [{"predicate": "OPEN_CAVITY", "expected": True}],
                "numeric_properties": [],
            },
            {
                "id": "soup_receptacle",
                "entity_kind": "OBJECT",
                "function": "contain one soup serving",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "verification_mode": "SEMANTIC_AND_GEOMETRIC",
                "candidate_categories": [category("bowl", "soup bowl")],
                "unary_properties": [{"predicate": "OPEN_CAVITY", "expected": True}],
                "numeric_properties": [],
            },
            {
                "id": "mixing_implement",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "verification_mode": "SEMANTIC_AND_GEOMETRIC",
                "candidate_categories": [category("spoon", "metal spoon")],
                "unary_properties": [{"predicate": "ELONGATED_OBJECT", "expected": True}],
                "numeric_properties": [
                    {
                        "property": "usable_length_m",
                        "operator": ">=",
                        "value": 0.08,
                        "unit": "m",
                    }
                ],
            },
            {
                "id": "soup_implement",
                "entity_kind": "OBJECT",
                "function": "serve with soup",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "verification_mode": "SEMANTIC_AND_GEOMETRIC",
                "candidate_categories": [category("soup_spoon", "soup spoon")],
                "unary_properties": [{"predicate": "ELONGATED_OBJECT", "expected": True}],
                "numeric_properties": [],
            },
            {
                "id": "water_source",
                "entity_kind": "OBJECT",
                "function": "provide water for coffee",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "verification_mode": "SEMANTIC_ONLY",
                "candidate_categories": [category("kettle", "kettle", "water jug")],
                "unary_properties": [],
                "numeric_properties": [],
            },
            {
                "id": "coffee_source",
                "entity_kind": "OBJECT",
                "function": "provide coffee material",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "verification_mode": "SEMANTIC_ONLY",
                "candidate_categories": [category("coffee_jar", "coffee jar")],
                "unary_properties": [],
                "numeric_properties": [],
            },
        ],
        "relations": [
            {
                "predicate": "INSERTABLE_IN",
                "subject_role": "mixing_implement",
                "object_role": "drink_receptacle",
                "expected": True,
            },
            {
                "predicate": "REACHES_BOTTOM",
                "subject_role": "mixing_implement",
                "object_role": "drink_receptacle",
                "expected": True,
            },
            {
                "predicate": "INSERTABLE_IN",
                "subject_role": "soup_implement",
                "object_role": "soup_receptacle",
                "expected": True,
            },
        ],
        "operation_groups": [
            {
                "id": "mix_drinks",
                "function": "stir",
                "tool_role": "mixing_implement",
                "target_role": "drink_receptacle",
                "required_target_count": 2,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["INSERTABLE_IN", "REACHES_BOTTOM"],
            },
            {
                "id": "equip_soups",
                "function": "provide utensil",
                "tool_role": "soup_implement",
                "target_role": "soup_receptacle",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["INSERTABLE_IN"],
            },
        ],
        "cross_group_reuse_allowed": False,
        "planning": {
            "contents": ["coffee", "water", "soup"],
            "source_roles": [
                {
                    "id": "water_provider",
                    "provides": "water",
                    "witness_role": "water_source",
                },
                {
                    "id": "coffee_provider",
                    "provides": "coffee",
                    "witness_role": "coffee_source",
                },
            ],
            "target_requirements": [
                {
                    "content": "coffee",
                    "witness_role": "drink_receptacle",
                    "required_contents": ["water", "coffee"],
                    "initial_contents": [],
                    "operation_group": "mix_drinks",
                    "final_goal": "filled and stirred",
                },
                {
                    "content": "soup",
                    "witness_role": "soup_receptacle",
                    "required_contents": ["soup"],
                    "initial_contents": ["soup"],
                    "operation_group": "equip_soups",
                    "final_goal": "served with utensil",
                },
            ],
        },
        "candidate_regions": [
            {"region_id": "C2", "reason": "may hold vessels"},
            {"region_id": "D1", "reason": "may hold utensils"},
        ],
        "inspection_order": ["C2", "D1"],
        "initial_satisfaction_assessment": False,
        "initial_satisfaction_reason": "Not all roles are visibly satisfied.",
        "unsupported_reason": "",
    }


class FakeTransport:
    def __init__(self, document: dict):
        self.document = document
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": json.dumps(self.document)}}]}


def test_exact_qwen_graph_is_preserved_without_reviewed_role_or_predicate_mapping():
    contract, vocabularies, trace = compile_vlm_functional_graph(
        qwen_graph(),
        task_instruction="Prepare two coffees and two soups.",
        observable_regions=tuple(REGIONS),
    )

    assert set(contract["roles"]) == {
        "drink_receptacle", "soup_receptacle", "mixing_implement", "soup_implement",
        "water_source", "coffee_source",
    }
    assert contract["specification_source"] == "qwen_vlm_single_call_exact_graph"
    assert contract["symbolic_task"]["target_requirements"]["coffee"][
        "witness_role"
    ] == "drink_receptacle"
    assert contract["relations"][0]["predicate"] == "INSERTABLE_IN"
    assert contract["roles"]["mixing_implement"]["unary_geometry"][1] == {
        "property": "usable_length_m",
        "operator": ">=",
        "value": 0.08,
        "unit": "m",
    }
    assert trace["transformation"] == "STRUCTURAL_ONLY_NO_ROLE_OR_PROPERTY_ALIAS_MAPPING"
    assert trace["added_task_requirements"] == []
    assert contract["symbolic_task"]["source_roles"]["water_provider"][
        "witness_role"
    ] == "water_source"
    assert "water jug" in vocabularies["object"]["canonical_labels"]["kettle"]


def test_adapter_makes_one_call_with_goal_images_regions_and_checker_api(tmp_path):
    image = tmp_path / "initial.png"
    image.write_bytes(PNG_1X1)
    transport = FakeTransport(qwen_graph())
    adapter = FMAdapter(model="qwen", transport=transport)

    result = adapter.generate_kitchen_functional_graph(
        "Prepare two coffees and two soups.", REGIONS, observation_images=[image]
    )

    assert result == qwen_graph()
    assert adapter.metrics.total_calls == 1
    assert len(transport.payloads) == 1
    payload_text = transport.payloads[0]["messages"][1]["content"][0]["text"]
    prompt = json.loads(payload_text)
    assert prompt["task_instruction"] == "Prepare two coffees and two soups."
    assert prompt["verifier_interface"]["binary_predicates"] == [
        "INSERTABLE_IN", "REACHES_BOTTOM"
    ]
    assert {row["region_id"] for row in prompt["observable_closed_storage_regions"]} == {
        "D1", "C2"
    }
    assert "hidden" not in prompt
    assert adapter.last_raw_kitchen_graph_response == qwen_graph()


def test_unknown_vlm_predicate_is_rejected_instead_of_alias_mapped():
    graph = deepcopy(qwen_graph())
    graph["relations"][0]["predicate"] = "FITS_INSIDE"

    with pytest.raises(ValueError, match="no exact checker exists"):
        compile_vlm_functional_graph(
            graph,
            task_instruction="Prepare two coffees and two soups.",
            observable_regions=tuple(REGIONS),
        )


def test_integrated_runner_has_no_reviewed_provider_or_normalization_dependency():
    source = (
        Path(__file__).resolve().parents[1] / "run_kitchen_vlm_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "EnvironmentVLMRequirementProvider" not in source
    assert "run_environment_vlm_requirements" not in source
    assert "kitchen_living_room_vlm_normalization" not in source
    assert "generate_task_requirements(" not in source
    assert "generate_inspection_priors(" not in source
