"""Tests for Pass 3.6A.4: Lossless Canonical G_F Construction and Contract Boundary."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLM_CANONICALIZATION_VERSION,
    VLMSpecProvider,
)
from mujoco_scenes.kitchen_vlm_functional_graph import (
    VLM_CANONICALIZATION_VERSION as KITCHEN_VERSION,
    compile_vlm_functional_graph,
)
from mujoco_scenes.environment_vlm_requirements import (
    VLM_CANONICALIZATION_VERSION as ENV_VERSION,
    EnvironmentVLMRequirementProvider,
    map_living_room_fixed_target_role,
    map_living_room_relation,
    map_living_room_role_function,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
    SYSTEM_PROMPT,
    validate_kitchen_functional_specification,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.requirements import (
    FMRequirementProvider,
    map_workshop_fixed_target_role,
    map_workshop_relation,
    map_workshop_role_function,
    map_workshop_unary_property,
    resolve_workshop_region_proposal,
)

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02"
    b"\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
)


# Section 50: Version constant check
def test_phase3_6a4_version_constant():
    assert VLM_CANONICALIZATION_VERSION == "phase3_6a7_2_v1"
    assert KITCHEN_VERSION == "phase3_6a7_2_v1"
    assert ENV_VERSION == "phase3_6a7_2_v1"


# Section 51: Neutral prompt de-biasing
def test_neutral_prompt_contains_no_benchmark_entities():
    forbidden_terms = (
        "coffee", "soup", "cup", "saucer", "remote", "armchair",
        "seating position", "screw", "screwdriver", "fastener", "driver",
        "repair hole", "workpiece", "drawer", "cupboard", "cabinet",
    )
    prompt_lower = SYSTEM_PROMPT.lower()
    for term in forbidden_terms:
        assert term not in prompt_lower, f"Forbidden benchmark term {term!r} found in static SYSTEM_PROMPT"

    assert "functional-requirement specification generator" in SYSTEM_PROMPT
    assert "functional-requirement planner" not in SYSTEM_PROMPT
    assert "candidate_objects" not in SYSTEM_PROMPT


# Section 52: Kitchen schema removes feasibility assessment
def test_kitchen_schema_no_feasibility_assessment():
    props = KITCHEN_FUNCTIONAL_GRAPH_SCHEMA["properties"]
    reqs = KITCHEN_FUNCTIONAL_GRAPH_SCHEMA["required"]
    assert "initial_satisfaction_assessment" not in props
    assert "initial_satisfaction_reason" not in props
    assert "initial_satisfaction_assessment" not in reqs
    assert "initial_satisfaction_reason" not in reqs


# Section 53: Strict kitchen top-level validation
def test_kitchen_validator_strict_top_level():
    valid_supported = {
        "status": "SUPPORTED",
        "task_summary": "Prepare two coffees",
        "functional_roles": [
            {
                "id": "cup_role",
                "entity_kind": "OBJECT",
                "function": "hold coffee",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["coffee cup"],
                "visible_candidates": [],
                "required_properties": ["open cavity"],
            }
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    # Must succeed on exact valid payload
    result = validate_kitchen_functional_specification(valid_supported)
    assert result["status"] == "SUPPORTED"

    # Missing a required field must fail
    for field in ("cross_group_reuse_allowed", "task_summary", "inspection_order", "unsupported_reason"):
        incomplete = {k: v for k, v in valid_supported.items() if k != field}
        with pytest.raises(FMResponseValidationError):
            validate_kitchen_functional_specification(incomplete)

    # Extra unexpected fields must fail
    extra = dict(valid_supported)
    extra["unexpected_field"] = 123
    with pytest.raises(FMResponseValidationError):
        validate_kitchen_functional_specification(extra)

    # UNSUPPORTED with non-empty roles must fail
    bad_unsupported = dict(valid_supported)
    bad_unsupported["status"] = "UNSUPPORTED"
    bad_unsupported["unsupported_reason"] = "Cannot fulfill"
    with pytest.raises(FMResponseValidationError):
        validate_kitchen_functional_specification(bad_unsupported)

    # UNSUPPORTED with empty unsupported_reason must fail
    clean_unsupported = {
        "status": "UNSUPPORTED",
        "task_summary": "Unsupported task",
        "functional_roles": [],
        "functional_relations": [],
        "interaction_groups": [],
        "cross_group_reuse_allowed": False,
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    with pytest.raises(FMResponseValidationError):
        validate_kitchen_functional_specification(clean_unsupported)


# Section 54: Workshop lossless canonical construction
def test_workshop_lossless_canonical_construction(tmp_path):
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Turn screws",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive threaded screw",
                "description": "power driver",
                "required_count": 2,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["compact powered screw tool"],
                "visible_candidates": [{"label": "compact powered screw tool", "visual_description": "driver", "suitability_reason": "drives"}],
                "required_properties": ["planar support"],
            },
            {
                "id": "screw_fastener",
                "entity_kind": "OBJECT",
                "function": "thread into workpiece to secure parts",
                "description": "fastener",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["threaded steel screw"],
                "visible_candidates": [{"label": "threaded steel screw", "visual_description": "screw", "suitability_reason": "threads"}],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "driver_tool",
                "relation": "engages screw head to turn",
                "object_role": "screw_fastener",
            }
        ],
        "interaction_groups": [],
        "inspectable_regions": [
            {"id": "reg_l", "label": "left storage drawer", "visual_description": "drawer on left", "reason": "tools"}
        ],
        "inspection_order": ["reg_l"],
    }

    class FakeTransport:
        def __init__(self):
            self.calls = 0
        def complete(self, payload):
            self.calls += 1
            return {"choices": [{"message": {"content": json.dumps(mock_doc)}}]}

    adapter = FMAdapter(base_url="http://fake", model="fake", transport=FakeTransport())
    provider = FMRequirementProvider(fm_adapter=adapter)

    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    graph = VLMSpecProvider._workshop("repair", [img], provider=provider)

    assert "driver" in graph.nodes
    driver_node = graph.nodes["driver"]
    assert driver_node.count == 2
    assert driver_node.binding_policy == "REUSABLE"
    assert driver_node.entity_kind == "OBJECT"
    assert driver_node.unary_predicates == ("PLANAR_SUPPORT",)
    assert "CAN_DRIVE_SCREW" not in driver_node.unary_predicates
    assert "compact_powered_screw_tool" in driver_node.semantic_categories

    assert "fastener" in graph.nodes
    fastener_node = graph.nodes["fastener"]
    assert fastener_node.count == 2
    assert fastener_node.binding_policy == "DISTINCT"

    assert len(graph.relations) == 1
    rel = graph.relations[0]
    assert rel.subject_role == "driver"
    assert rel.predicate == "COMPATIBLE_WITH"
    assert rel.object_role == "fastener"


# Section 55: Workshop single VLM call
def test_workshop_single_vLM_call_in_pipeline(tmp_path):
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Drive screw",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "tool",
                "entity_kind": "OBJECT",
                "function": "turn screw",
                "description": "driver",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            }
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [
            {"id": "r1", "label": "left drawer", "visual_description": "drawer", "reason": "storage"}
        ],
        "inspection_order": ["r1"],
    }

    class SingleCallTransport:
        def __init__(self):
            self.total_calls = 0
        def complete(self, payload):
            self.total_calls += 1
            return {"choices": [{"message": {"content": json.dumps(mock_doc)}}]}

    transport = SingleCallTransport()
    adapter = FMAdapter(base_url="http://fake", model="fake", transport=transport)
    provider = FMRequirementProvider(fm_adapter=adapter)

    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    graph = VLMSpecProvider._workshop("repair joint", [img], provider=provider)

    assert transport.total_calls == 1
    assert graph.candidate_regions == ("LEFT_DRAWER",)
    assert graph.region_ranking == ("LEFT_DRAWER",)


# Section 56: Explicit FIXED_TARGET and no implicit injection
def test_workshop_explicit_fixed_target_and_no_injection(tmp_path):
    mock_doc_no_target = {
        "status": "SUPPORTED",
        "task_summary": "Fasten joint",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive screw",
                "description": "tool",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc_no_target

    provider = FMRequirementProvider(fm_adapter=FakeAdapter())
    graph = VLMSpecProvider._workshop("drive screw", [tmp_path / "fake.png"], provider=provider)
    assert "repair_target" not in graph.nodes

    # Now with explicit fixed target
    mock_doc_with_target = {
        "status": "SUPPORTED",
        "task_summary": "Fasten joint",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "workpiece_joint",
                "entity_kind": "FIXED_TARGET",
                "function": "workbench hole insertion point to accept screw",
                "description": "pre-drilled hole",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["frame_joint"],
                "visible_candidates": [],
                "required_properties": [],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter2:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc_with_target

    provider2 = FMRequirementProvider(fm_adapter=FakeAdapter2())
    graph2 = VLMSpecProvider._workshop("drive screw", [tmp_path / "fake.png"], provider=provider2)
    assert "repair_target" in graph2.nodes
    assert graph2.nodes["repair_target"].entity_kind == "FIXED_TARGET"


# Section 57: Bare hole fails closed
def test_workshop_bare_hole_fails_closed():
    assert map_workshop_role_function("hole") is None

    provider = FMRequirementProvider()
    with pytest.raises(VLMSpecificationError):
        provider._map_function({"function": "hole", "description": ""})

    raw_ft_bare = {
        "id": "r1",
        "entity_kind": "FIXED_TARGET",
        "function": "hole",
        "description": "",
    }
    assert map_workshop_fixed_target_role(raw_ft_bare) is None


# Section 58: Living Room lossless canonical relations
def test_living_room_lossless_relations(tmp_path):
    mock_lr_doc = {
        "status": "SUPPORTED",
        "task_summary": "Prepare room",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "cup_table",
                "entity_kind": "REGION",
                "function": "support individual drinkware set for seated viewer",
                "description": "side table",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            },
            {
                "id": "remote_table",
                "entity_kind": "REGION",
                "function": "support shared tv remote between both viewers",
                "description": "coffee table",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee table"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "cup_table",
                "relation": "within reach of one seated viewer armchair",
                "object_role": "remote_table",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_lr_doc

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    img = tmp_path / "fake.png"
    img.write_bytes(PNG_1X1)
    graph = VLMSpecProvider._living_room("prepare room", [img], provider=provider)

    assert "PERSONAL_CUP_SAUCER_REGION" in graph.nodes
    assert "SHARED_REMOTE_REGION" in graph.nodes
    assert len(graph.relations) == 1
    rel = graph.relations[0]
    assert rel.subject_role == "PERSONAL_CUP_SAUCER_REGION"
    assert rel.predicate == "NEAR_SEAT"
    assert rel.object_role == "SHARED_REMOTE_REGION"


# Section 59: Living Room unmapped relation fails closed
def test_living_room_unmapped_relation_fails_closed(tmp_path):
    mock_bad_rel = {
        "status": "SUPPORTED",
        "task_summary": "Prepare room",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "cup_table",
                "entity_kind": "REGION",
                "function": "support personal cup and saucer",
                "description": "table",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [
            {
                "subject_role": "cup_table",
                "relation": "teleport above ceiling",
                "object_role": "cup_table",
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_bad_rel

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    img = tmp_path / "fake.png"
    img.write_bytes(PNG_1X1)
    with pytest.raises(VLMSpecificationError):
        VLMSpecProvider._living_room("prepare room", [img], provider=provider)


# Section 60: Living Room no fallback counts or bindings
def test_living_room_no_fallback_counts_or_bindings(tmp_path):
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Prepare room",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "table_role",
                "entity_kind": "REGION",
                "function": "support personal drinkware for person",
                "description": "table",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    img = tmp_path / "fake.png"
    img.write_bytes(PNG_1X1)
    graph = VLMSpecProvider._living_room("prepare room", [img], provider=provider)

    node = graph.nodes["PERSONAL_CUP_SAUCER_REGION"]
    assert node.count == 1
    assert node.binding_policy == "DISTINCT"


# Section 61: Living Room open-vocabulary categories
def test_living_room_open_vocab_categories(tmp_path):
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Prepare room",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "t1",
                "entity_kind": "REGION",
                "function": "support personal drinkware",
                "description": "small table",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["teacup support disc"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    img = tmp_path / "fake.png"
    img.write_bytes(PNG_1X1)
    graph = VLMSpecProvider._living_room("prepare room", [img], provider=provider)

    node = graph.nodes["PERSONAL_CUP_SAUCER_REGION"]
    assert "teacup_support_disc" in node.semantic_categories
    assert "teacup support disc" in graph.detector_vocabulary


# Section 62: Detector vocabulary provenance separation
def test_detector_vocabulary_provenance_separation(tmp_path):
    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Prepare room",
        "unsupported_reason": "",
        "functional_roles": [
            {
                "id": "t1",
                "entity_kind": "REGION",
                "function": "support personal drinkware",
                "description": "table",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            }
        ],
        "functional_relations": [],
        "inspectable_regions": [],
        "inspection_order": [],
    }

    class FakeAdapter:
        last_observation_images = []
        last_raw_requirement_response = {}
        last_raw_inspection_response = {}
        metrics = MagicMock(total_calls=1)

        def generate_task_requirements(self, *args, **kwargs):
            return mock_doc

    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=FakeAdapter())
    img = tmp_path / "fake.png"
    img.write_bytes(PNG_1X1)
    graph = VLMSpecProvider._living_room("prepare room", [img], provider=provider)

    assert "vlm_derived_role_vocabulary" in graph.metadata
    assert "task_explicit_context_vocabulary" in graph.metadata
    assert "side table" in graph.metadata["vlm_derived_role_vocabulary"]
    assert "armchair" in graph.metadata["task_explicit_context_vocabulary"]
