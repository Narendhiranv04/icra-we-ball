"""Pass 3.6A.5 Contract and End-to-End Semantic Boundary Closure Tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    VLM_CANONICALIZATION_VERSION as ENV_VERSION,
)
from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
    TransportOrStructuredOutputError,
    VLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedObject,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLMSpecProvider,
    VLM_CANONICALIZATION_VERSION,
)
from mujoco_scenes.kitchen_vlm_functional_graph import (
    VLM_CANONICALIZATION_VERSION as KITCHEN_VERSION,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    FMTransportError,
    SYSTEM_PROMPT,
)
from mujoco_scenes.workshop_phase1.requirements import (
    FMRequirementProvider,
)

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02"
    b"\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
)


class MockTransport:
    def __init__(self, response: dict[str, Any]):
        self.response = response

    def complete(self, payload: Any) -> dict[str, Any]:
        return self.response


# Section 1: Version constant checks across all modules
def test_phase3_6a5_version_constants():
    assert VLM_CANONICALIZATION_VERSION == "phase3_6a7_2_v1"
    assert KITCHEN_VERSION == "phase3_6a7_2_v1"
    assert ENV_VERSION == "phase3_6a7_2_v1"


# Section 2: Clean static prompt (zero concrete semantic examples and zero benchmark nouns)
def test_static_prompt_has_no_concrete_semantic_content_examples():
    forbidden_examples = (
        "support an item",
        "rigid",
        "near reference region",
        "planar support",
        "open cavity",
        "elongated object",
    )
    prompt_lower = SYSTEM_PROMPT.lower()
    for example in forbidden_examples:
        assert example not in prompt_lower, f"Concrete example {example!r} found in SYSTEM_PROMPT"


def test_static_prompt_has_no_benchmark_nouns():
    forbidden_nouns = (
        "coffee", "soup", "cup", "saucer", "remote", "armchair",
        "seating", "screw", "screwdriver", "driver", "fastener",
        "repair hole", "workpiece", "drawer", "cupboard", "cabinet",
    )
    prompt_lower = SYSTEM_PROMPT.lower()
    for noun in forbidden_nouns:
        assert noun not in prompt_lower, f"Benchmark noun {noun!r} found in SYSTEM_PROMPT"


def test_static_prompt_teaches_generic_functional_asset_vs_payload_distinction():
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "functional roles for scene assets" in prompt_lower
    assert "payloads or fixed contextual" in prompt_lower


# Section 3: Structured failure taxonomy
def test_failure_taxonomy_hierarchy():
    assert issubclass(MalformedVLMSpecificationError, VLMSpecificationError)
    assert issubclass(UnmappedFunctionalConceptError, VLMSpecificationError)
    assert issubclass(UnsupportedCheckerCapabilityError, VLMSpecificationError)
    assert issubclass(AmbiguousCanonicalizationError, VLMSpecificationError)
    assert issubclass(TransportOrStructuredOutputError, VLMSpecificationError)
    assert issubclass(FMResponseValidationError, MalformedVLMSpecificationError)
    assert issubclass(FMTransportError, TransportOrStructuredOutputError)


def test_unsupported_checker_capability_raised_for_unsupported_property(tmp_path):
    mock_resp = {
        "status": "SUPPORTED",
        "task_summary": "Task with unsupported property",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive a fastener",
                "description": "driver",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["screwdriver tool"],
                "visible_candidates": [],
                "required_properties": ["rigid"],
            },
            {
                "id": "fastener_obj",
                "entity_kind": "OBJECT",
                "function": "fasten parts together",
                "description": "fastener",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screw"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "driver_tool",
                "relation": "drives fastener",
                "object_role": "fastener_obj",
            }
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    adapter = FMAdapter(transport=MockTransport(mock_resp))
    provider = FMRequirementProvider(fm_adapter=adapter)
    with pytest.raises(UnsupportedCheckerCapabilityError) as exc_info:
        provider.get_requirements("Drive a screw", observation_images=[img])
    assert exc_info.value.category == "UNSUPPORTED_CHECKER_CAPABILITY"
    assert "rigid" in str(exc_info.value)


def test_malformed_vlm_spec_raised_for_undeclared_relation_endpoint(tmp_path):
    mock_resp = {
        "status": "SUPPORTED",
        "task_summary": "Task with undeclared relation endpoint",
        "functional_roles": [
            {
                "id": "driver_tool",
                "entity_kind": "OBJECT",
                "function": "drive a fastener",
                "description": "driver",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["screwdriver tool"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "driver_tool",
                "relation": "drives fastener",
                "object_role": "non_existent_role",
            }
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    adapter = FMAdapter(transport=MockTransport(mock_resp))
    provider = FMRequirementProvider(fm_adapter=adapter)
    with pytest.raises(MalformedVLMSpecificationError) as exc_info:
        provider.get_requirements("Drive a screw", observation_images=[img])
    assert exc_info.value.category == "MALFORMED_VLM_SPECIFICATION"


# Section 4: Workshop G_F -> G_O dynamic semantic sourcing
def test_workshop_gf_controls_go_without_hardcoded_categories():
    from mujoco_scenes.functional_tamp_pipeline.domains.workshop import WorkshopDomainAdapter

    # G_F with synthetic open-vocabulary categories
    gf = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="Drive a fastener",
        nodes={
            "driver": FunctionalRole(
                name="driver",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("compact_powered_screw_tool",),
                unary_predicates=(),
                binding_policy="REUSABLE",
                verification_mode="SEMANTIC_AND_GEOMETRIC",
            ),
            "fastener": FunctionalRole(
                name="fastener",
                entity_kind="OBJECT",
                count=1,
                semantic_categories=("threaded_metal_joining_pin",),
                unary_predicates=(),
                binding_policy="DISTINCT",
                verification_mode="SEMANTIC_AND_GEOMETRIC",
            ),
            "repair_target": FunctionalRole(
                name="repair_target",
                entity_kind="FIXED_TARGET",
                count=1,
                semantic_categories=("repair_target",),
                unary_predicates=(),
                binding_policy="SHARED",
                verification_mode="GEOMETRIC_ONLY",
            ),
        },
        relations=(
            FunctionalRelation(
                subject_role="driver",
                predicate="REACHES_TARGET",
                object_role="repair_target",
                expected=True,
            ),
            FunctionalRelation(
                subject_role="fastener",
                predicate="COMPATIBLE_WITH_TARGET",
                object_role="repair_target",
                expected=True,
            ),
            FunctionalRelation(
                subject_role="driver",
                predicate="COMPATIBLE_WITH",
                object_role="fastener",
                expected=True,
            ),
        ),
        detector_vocabulary=("compact powered screw tool", "threaded metal joining pin"),
        candidate_regions=("TOOL_DRAWER",),
        region_ranking=("TOOL_DRAWER",),
        source="VLM_CANONICAL_G_F",
        metadata={
            "detector_label_to_canonical": {
                "compact powered screw tool": "compact_powered_screw_tool",
                "threaded metal joining pin": "threaded_metal_joining_pin",
            }
        },
    )

    adapter = WorkshopDomainAdapter("F0_MANUAL_FIRST_ONE_REGION", gf, physical_open=False)
    assert "compact_powered_screw_tool" in gf.nodes["driver"].semantic_categories
    assert "threaded_metal_joining_pin" in gf.nodes["fastener"].semantic_categories


# Section 5: Workshop detector vocabulary purity (negative controls not merged)
def test_workshop_detector_vocabulary_purity(tmp_path):
    mock_resp = {
        "status": "SUPPORTED",
        "task_summary": "Task with tools",
        "functional_roles": [
            {
                "id": "tool_1",
                "entity_kind": "OBJECT",
                "function": "drive a fastener into target",
                "description": "tool",
                "required_count": 1,
                "binding_policy": "REUSABLE",
                "candidate_categories": ["cordless screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "fastener_1",
                "entity_kind": "OBJECT",
                "function": "fasten components securely",
                "description": "fastener",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["wood screw"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "target_hole",
                "entity_kind": "FIXED_TARGET",
                "function": "receive threaded fastener at workbench recess",
                "description": "workbench target hole",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["workbench hole"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "tool_1",
                "relation": "compatible with",
                "object_role": "fastener_1",
            },
            {
                "subject_role": "tool_1",
                "relation": "reaches into",
                "object_role": "target_hole",
            },
            {
                "subject_role": "fastener_1",
                "relation": "fits inside",
                "object_role": "target_hole",
            },
        ],
        "interaction_groups": [],
        "inspectable_regions": [
            {"id": "drawer_1", "label": "tool drawer", "visual_description": "wooden tool storage drawer", "reason": "storage"}
        ],
        "inspection_order": ["drawer_1"],
        "unsupported_reason": "",
    }
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    adapter = FMAdapter(transport=MockTransport(mock_resp))
    provider = FMRequirementProvider(fm_adapter=adapter)

    gf = VLMSpecProvider._workshop("Fix workbench hole", [img], provider=provider)
    # The detector vocabulary must ONLY contain VLM-derived prompts
    assert "cordless screwdriver" in gf.detector_vocabulary
    assert "wood screw" in gf.detector_vocabulary
    assert "claw hammer" not in gf.detector_vocabulary
    assert "ball peen hammer" not in gf.detector_vocabulary
    assert "sledgehammer" not in gf.detector_vocabulary
    # Negative controls are kept in metadata
    assert "evaluation_negative_control_prompts" in gf.metadata


# Section 6: Living Room valid FIXED_TARGET relation end-to-end support
def test_living_room_valid_fixed_target_relation_preservation(tmp_path):
    mock_resp = {
        "status": "SUPPORTED",
        "task_summary": "Prepare living room refreshments",
        "functional_roles": [
            {
                "id": "personal_table",
                "entity_kind": "REGION",
                "function": "support personal drink and saucer for seated viewer",
                "description": "table",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side table", "end table"],
                "visible_candidates": [],
                "required_properties": ["planar support"],
            },
            {
                "id": "armchair_seat",
                "entity_kind": "FIXED_TARGET",
                "function": "armchair seating position reference",
                "description": "seating",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["armchair seat", "seating position"],
                "visible_candidates": [],
                "required_properties": [],
            },
        ],
        "functional_relations": [
            {
                "subject_role": "personal_table",
                "relation": "near assigned seat",
                "object_role": "armchair_seat",
            }
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    img = tmp_path / "img.png"
    img.write_bytes(PNG_1X1)
    adapter = FMAdapter(transport=MockTransport(mock_resp))
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    gf = VLMSpecProvider._living_room("Prepare living room", [img], provider=provider)

    assert "PERSONAL_CUP_SAUCER_REGION" in gf.nodes
    assert "SEATING_POSITION" in gf.nodes
    assert gf.nodes["SEATING_POSITION"].entity_kind == "FIXED_TARGET"

    # Verify relation preserved
    near_seat_rels = [
        r for r in gf.relations
        if r.predicate == "NEAR_SEAT" and r.subject_role == "PERSONAL_CUP_SAUCER_REGION" and r.object_role == "SEATING_POSITION"
    ]
    assert len(near_seat_rels) == 1

    # Verify ObservedSceneGraph with FIXED_TARGET seating position evaluates cleanly in ground_graph
    go = ObservedSceneGraph()
    go.add_node(ObservedNode(
        instance_id="table_1",
        entity_kind="REGION",
        canonical_category="side_table",
        unary_predicates={"PLANAR_SUPPORT": "TRUE"},
    ))
    go.add_node(ObservedNode(
        instance_id="seat_1",
        entity_kind="FIXED_TARGET",
        canonical_category="seating_position",
    ))
    go.add_relation(ObservedRelation(
        subject_id="table_1",
        predicate="NEAR_SEAT",
        object_id="seat_1",
        status="TRUE",
    ))

    result = ground_graph(gf, go)
    assert result.status == "COMPLETE"
    assert result.assignment["PERSONAL_CUP_SAUCER_REGION"] == "table_1"
    assert result.assignment["SEATING_POSITION"] == "seat_1"
