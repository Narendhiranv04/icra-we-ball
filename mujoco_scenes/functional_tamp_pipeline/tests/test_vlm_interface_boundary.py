"""Comprehensive tests for the Pass 3.6A VLM interface boundary realignment."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.kitchen_vlm_functional_graph import (
    compile_vlm_functional_graph,
    resolve_kitchen_region_proposal,
    map_unary_property,
    map_binary_relation,
)
from mujoco_scenes.workshop_phase1.requirements import (
    FMRequirementProvider,
    resolve_workshop_region_proposal,
)
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter
from mujoco_scenes.environment_vlm_requirements import EnvironmentVLMRequirementProvider


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class MockTransport:
    def __init__(self, response_map: dict[str, dict]):
        self.response_map = response_map
        self.payloads: list[dict] = []

    def complete(self, payload: dict) -> dict:
        self.payloads.append(payload)
        call_name = payload.get("response_format", {}).get("json_schema", {}).get("name", "default")
        doc = self.response_map.get(call_name, next(iter(self.response_map.values())))
        return {"choices": [{"message": {"content": json.dumps(doc)}}]}


def test_zero_leakage_in_kitchen_and_workshop_payloads(tmp_path):
    """Verify that no internal checker names or canonical region IDs leak in VLM prompts."""
    img = tmp_path / "obs.png"
    img.write_bytes(PNG_1X1)

    kitchen_doc = {
        "status": "SUPPORTED",
        "task_summary": "Make coffee and soup",
        "functional_roles": [
            {
                "id": "cup",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup"],
                "visible_candidates": [],
                "required_properties": ["open cavity container"],
            },
            {
                "id": "stirrer",
                "entity_kind": "OBJECT",
                "function": "stir coffee",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon"],
                "visible_candidates": [],
                "required_properties": ["elongated utensil"],
            },
            {
                "id": "kettle",
                "entity_kind": "OBJECT",
                "function": "water source",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["kettle"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "coffee_jar",
                "entity_kind": "OBJECT",
                "function": "coffee source",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["coffee_jar"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "stirrer",
                "relation": "fits inside",
                "object_role": "cup",
            }
        ],
        "interaction_groups": [
            {
                "id": "stir_group",
                "function": "stir",
                "tool_role": "stirrer",
                "target_role": "cup",
                "required_target_count": 2,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["fits inside"],
            }
        ],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [
            {"id": "c1", "label": "upper wall cupboard", "visual_description": "cupboard", "reason": "storage"}
        ],
        "inspection_order": ["c1"],
        "initial_satisfaction_assessment": False,
        "initial_satisfaction_reason": "Need cups",
        "unsupported_reason": "",
    }

    workshop_req_doc = {
        "status": "SUPPORTED",
        "task_summary": "Drive screw",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive a screw",
                "description": "screwdriver or power driver",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver", "power driver"],
                "visible_candidates": [
                    {"label": "cordless power driver", "visual_description": "driver", "suitability_reason": "fits screw"}
                ],
                "required_properties": [],
            },
            {
                "id": "fastener_obj",
                "entity_kind": "OBJECT",
                "function": "fasten joint",
                "description": "screw fastener",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [
                    {"label": "phillips screw", "visual_description": "screw", "suitability_reason": "fits hole"}
                ],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "driver_tool",
                "relation": "compatible with screw head",
                "object_role": "fastener_obj",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    workshop_insp_doc = {
        "initial_requirements_satisfied": False,
        "decision_reason": "Need tools",
        "inspectable_regions": [
            {"id": "r1", "label": "left workbench drawer", "visual_description": "drawer under workbench", "reason": "storage"}
        ],
        "inspection_order": ["r1"],
    }

    transport = MockTransport({
        "kitchen_functional_requirement_graph": kitchen_doc,
        "functional_specification": workshop_req_doc,
        "inspection_policy": workshop_insp_doc,
    })
    adapter = FMAdapter(model="test_model", transport=transport)

    # 1. Kitchen Call
    adapter.generate_kitchen_functional_graph("Make coffee and soup", observation_images=[img])
    # 2. Workshop Call 1 (Requirements)
    adapter.generate_task_requirements("Drive screw", observation_images=[img])
    # 3. Workshop Call 2 (Inspection)
    adapter.generate_inspection_priors("Drive screw", observation_images=[img])

    assert len(transport.payloads) == 3

    forbidden_checker_strings = [
        "OPEN_CAVITY", "ELONGATED_OBJECT", "INSERTABLE_IN", "REACHES_BOTTOM",
        "CAN_DRIVE_SCREW", "CAN_FASTEN", "PLANAR_SUPPORT", "total_length_m"
    ]
    forbidden_region_strings = [
        '"D1"', '"D2"', '"C2"', '"B1"', '"C1"',
        '"LEFT_DRAWER"', '"RIGHT_DRAWER"', '"TOOL_CABINET"'
    ]

    for idx, payload in enumerate(transport.payloads):
        payload_str = json.dumps(payload)
        for checker in forbidden_checker_strings:
            assert checker not in payload_str, f"Payload {idx} leaked checker {checker}"
        for reg in forbidden_region_strings:
            assert reg not in payload_str, f"Payload {idx} leaked canonical region {reg}"


def test_deterministic_kitchen_region_resolution():
    assert resolve_kitchen_region_proposal({"label": "upper wall cupboard", "visual_description": "above counter"}) == "C2"
    assert resolve_kitchen_region_proposal({"label": "top drawer", "visual_description": "first drawer"}) == "D1"
    assert resolve_kitchen_region_proposal({"label": "lower cupboard", "visual_description": "under counter"}) == "C1"
    assert resolve_kitchen_region_proposal({"label": "countertop storage box", "visual_description": "wooden box"}) == "B1"
    assert resolve_kitchen_region_proposal({"label": "refrigerator door", "visual_description": "fridge"}) is None


def test_deterministic_workshop_region_resolution():
    assert resolve_workshop_region_proposal({"label": "left workbench drawer", "visual_description": "drawer on left"}) == "LEFT_DRAWER"
    assert resolve_workshop_region_proposal({"label": "right storage drawer", "visual_description": "drawer on right"}) == "RIGHT_DRAWER"
    assert resolve_workshop_region_proposal({"label": "wall tool cabinet", "visual_description": "upper tool storage"}) == "TOOL_CABINET"
    assert resolve_workshop_region_proposal({"label": "overhead lamp", "visual_description": "light fixture"}) is None


def test_fail_closed_on_unmapped_kitchen_property():
    spec = {
        "status": "SUPPORTED",
        "task_summary": "Test",
        "functional_roles": [
            {
                "id": "tool",
                "entity_kind": "OBJECT",
                "function": "stir",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["spoon"],
                "visible_candidates": [],
                "required_properties": ["quantum magnetic superconductor"],  # Unmappable!
            }
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    with pytest.raises(VLMSpecificationError, match="no exact or alias checker mapping exists"):
        compile_vlm_functional_graph(spec, task_instruction="Test", observable_regions=("D1", "C2"))


def test_fail_closed_on_unsupported_workshop_region_role(tmp_path):
    doc = {
        "status": "SUPPORTED",
        "task_summary": "Drive screw",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "unsupported_table_region",
                "entity_kind": "REGION",
                "function": "rest workpiece",
                "description": "tabletop surface",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["table surface"],
                "visible_candidates": [{"label": "table surface", "visual_description": "table", "suitability_reason": "flat"}],
                "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
    }
    transport = MockTransport({"functional_specification": doc})
    adapter = FMAdapter(model="test_model", transport=transport)
    provider = FMRequirementProvider(fm_adapter=adapter)

    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    with pytest.raises(VLMSpecificationError, match="Unsupported REGION role"):
        provider.get_requirements("Drive screw", observation_images=[img])


def test_living_room_detector_vocabulary_uses_only_vlm_categories(tmp_path):
    """Ensure living room detector vocabulary does not load full reviewed ontology in VLM mode."""
    provider = EnvironmentVLMRequirementProvider("living_room")
    spec_prov = VLMSpecProvider()

    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)

    doc = {
        "status": "SUPPORTED",
        "task_summary": "Serve drinks and remote",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "personal_support",
                "entity_kind": "REGION",
                "function": "support cup and saucer",
                "description": "side table near armchair",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [
                    {"label": "small round side table", "visual_description": "wood table", "suitability_reason": "near chair"}
                ],
                "required_properties": ["planar support"],
            },
            {
                "id": "shared_support",
                "entity_kind": "REGION",
                "function": "support remote control",
                "description": "coffee table between seats",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee table"],
                "visible_candidates": [
                    {"label": "low coffee table", "visual_description": "central table", "suitability_reason": "central"}
                ],
                "required_properties": ["planar support"],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "personal_support",
                "relation": "near seat",
                "object_role": "personal_support",
            },
            {
                "subject_role": "shared_support",
                "relation": "between seats",
                "object_role": "shared_support",
            },
        ],
        "inspectable_regions": [],
        "inspection_order": [],
    }
    transport = MockTransport({"functional_specification": doc})
    provider.fm_adapter = FMAdapter(model="test_model", transport=transport)

    graph = spec_prov._living_room("Prepare living room", [img], provider=provider)
    
    # Check that vocabulary contains only terms relevant to G_F roles (side table, coffee table, remote, cup, saucer, seating)
    # and does NOT contain unrelated categories like bookshelf, tv_console, floor, rug, dining_table, etc.
    assert "bookshelf" not in graph.detector_vocabulary
    assert "media console" not in graph.detector_vocabulary
    assert "floor" not in graph.detector_vocabulary
    assert any("side table" in term for term in graph.detector_vocabulary)
    assert any("coffee table" in term for term in graph.detector_vocabulary)
