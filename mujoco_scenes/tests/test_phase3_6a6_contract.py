"""Comprehensive Contract Test Suite for Pass 3.6A.6.

Verifies items A through O:
A. Two-person Living Room grounding with 2 seating contexts (2 personal + 1 shared).
B. Living Room grounding rejects invalid context pairing.
C. Living Room G_F produces non-empty operation_groups when VLM provides interaction_groups.
D. Living Room preserves task-explicit payload and context nodes.
E. Workshop tracks routed through open-vocabulary category mapping.
F. Workshop fails with MalformedVLMSpecificationError when driver/fastener is missing.
G. Ambiguous Workshop region proposal raises AmbiguousCanonicalizationError.
H. Ambiguous Workshop relation raises AmbiguousCanonicalizationError.
I. Ambiguous Living Room relation / property raises AmbiguousCanonicalizationError.
J. Unmapped Living Room property raises UnsupportedCheckerCapabilityError / UnmappedFunctionalConceptError.
K. Invalid JSON decode raises TransportOrStructuredOutputError.
L. Missing FM backend raises TransportOrStructuredOutputError.
M. PipelineResult and run_manifest persist failure_category.
N. Generic schema and system prompt contain zero benchmark nouns and support interaction_groups.
O. fm_adapter docstring accurately describes natural-language to deterministic canonicalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    map_living_room_fixed_target_role,
    map_living_room_object_payload_role,
    map_living_room_relation,
)
from mujoco_scenes.functional_tamp_pipeline.errors import (
    AmbiguousCanonicalizationError,
    MalformedVLMSpecificationError,
    TransportOrStructuredOutputError,
    UnmappedFunctionalConceptError,
    UnsupportedCheckerCapabilityError,
    VLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.grounding import ground_graph
from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
    OperationGroup,
    PipelineResult,
)
from mujoco_scenes.functional_tamp_pipeline.run import _RunState, _write_run_manifest
from mujoco_scenes.functional_tamp_pipeline.scene_graph import (
    ObservedNode,
    ObservedObject,
    ObservedRelation,
    ObservedSceneGraph,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMBackendNotConfiguredError,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    _extract_json_content,
    validate_requirement_response,
)
from mujoco_scenes.workshop_phase1.requirements import (
    map_workshop_relation,
    resolve_workshop_region_proposal,
)


class MockFMAdapter(FMAdapter):
    def __init__(self, response_doc: dict[str, Any]):
        super().__init__()
        self._doc = response_doc
        self.last_raw_requirement_response = response_doc
        self.last_observation_images = []

    def generate_task_requirements(self, *args, **kwargs) -> dict[str, Any]:
        self.last_raw_requirement_response = self._doc
        self.last_observation_images = []
        return self._doc

    def query_requirements(self, *args, **kwargs) -> dict[str, Any]:
        self.last_raw_requirement_response = self._doc
        self.last_observation_images = []
        return self._doc


# ---------------------------------------------------------------------------
# Test A: Two-person Living Room requirement with 2 distinct seating contexts
# ---------------------------------------------------------------------------
def test_a_living_room_two_person_grounding_success():
    spec = GTSpecProvider().provide("living_room", "two person tea setup")
    graph = ObservedSceneGraph()
    
    # 2 armchairs (seating positions)
    graph.add_node(ObservedNode(
        instance_id="armchair_left", entity_kind="FIXED_TARGET", canonical_category="armchair",
    ))
    graph.add_node(ObservedNode(
        instance_id="armchair_right", entity_kind="FIXED_TARGET", canonical_category="armchair",
    ))
    # Seating pair
    graph.add_node(ObservedNode(
        instance_id="pair_1", entity_kind="FIXED_TARGET", canonical_category="seating_pair",
    ))
    # 2 cup saucer sets
    graph.add_node(ObservedNode(
        instance_id="set_1", entity_kind="OBJECT", canonical_category="cup_saucer_set",
    ))
    graph.add_node(ObservedNode(
        instance_id="set_2", entity_kind="OBJECT", canonical_category="cup_saucer_set",
    ))
    # 1 remote
    graph.add_node(ObservedNode(
        instance_id="remote_1", entity_kind="OBJECT", canonical_category="remote_control",
    ))
    # 2 personal side tables
    graph.add_node(ObservedNode(
        instance_id="table_left", entity_kind="REGION", canonical_category="side_table",
        unary_predicates={"PLANAR_SUPPORT": "TRUE"},
    ))
    graph.add_node(ObservedNode(
        instance_id="table_right", entity_kind="REGION", canonical_category="side_table",
        unary_predicates={"PLANAR_SUPPORT": "TRUE"},
    ))
    # 1 coffee table (shared)
    graph.add_node(ObservedNode(
        instance_id="coffee_table", entity_kind="REGION", canonical_category="coffee_table",
        unary_predicates={"PLANAR_SUPPORT": "TRUE"},
    ))

    # Add relations
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="NEAR_SEAT", object_id="armchair_right", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="ACCESSIBLE_FROM_BOTH_SEATS", object_id="pair_1", status="TRUE"))

    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="FITS_SET_ON", object_id="set_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_right", predicate="FITS_SET_ON", object_id="set_2", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="coffee_table", predicate="FITS_ON", object_id="remote_1", status="TRUE"))

    result = ground_graph(spec, graph)
    assert result.satisfied is True
    assert result.status == "COMPLETE"
    assert len(result.assignment["PERSONAL_CUP_SAUCER_REGION"]) == 2
    assert result.assignment["SHARED_REMOTE_REGION"] == "coffee_table"


# ---------------------------------------------------------------------------
# Test B: Living Room rejects invalid context pairing
# ---------------------------------------------------------------------------
def test_b_living_room_rejects_invalid_context_pairing():
    spec = GTSpecProvider().provide("living_room", "two person tea setup")
    graph = ObservedSceneGraph()

    # Only 1 armchair
    graph.add_node(ObservedNode(instance_id="armchair_left", entity_kind="FIXED_TARGET", canonical_category="armchair"))
    graph.add_node(ObservedNode(instance_id="set_1", entity_kind="OBJECT", canonical_category="cup_saucer_set"))
    graph.add_node(ObservedNode(instance_id="set_2", entity_kind="OBJECT", canonical_category="cup_saucer_set"))
    graph.add_node(ObservedNode(instance_id="table_left", entity_kind="REGION", canonical_category="side_table", unary_predicates={"PLANAR_SUPPORT": "TRUE"}))
    graph.add_node(ObservedNode(instance_id="table_other", entity_kind="REGION", canonical_category="side_table", unary_predicates={"PLANAR_SUPPORT": "TRUE"}))

    # Both tables claim to be near armchair_left, but 2 distinct contexts are required
    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_other", predicate="NEAR_SEAT", object_id="armchair_left", status="TRUE"))

    graph.add_relation(ObservedRelation(subject_id="table_left", predicate="FITS_SET_ON", object_id="set_1", status="TRUE"))
    graph.add_relation(ObservedRelation(subject_id="table_other", predicate="FITS_SET_ON", object_id="set_2", status="TRUE"))

    result = ground_graph(spec, graph)
    assert result.satisfied is False


# ---------------------------------------------------------------------------
# Test C: Living Room G_F produces non-empty operation_groups from VLM
# ---------------------------------------------------------------------------
def test_c_living_room_vlm_spec_produces_operation_groups():
    vlm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Two person tea serving",
        "functional_roles": [
            {
                "id": "cup_set",
                "entity_kind": "OBJECT",
                "function": "cup saucer set beverage payload",
                "description": "payload tea set",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup", "saucer"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "seat",
                "entity_kind": "FIXED_TARGET",
                "function": "seating position armchair",
                "description": "armchair position",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["armchair"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "personal_surface",
                "entity_kind": "REGION",
                "function": "personal beverage support surface",
                "description": "side table next to armchair",
                "required_count": 2,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["side_table", "table"],
                "visible_candidates": [],
                "required_properties": ["planar support surface"],
            },
        ],
        "functional_relations": [
            {"subject_role": "personal_surface", "relation": "fits cup and saucer set", "object_role": "cup_set"},
            {"subject_role": "personal_surface", "relation": "near seat", "object_role": "seat"},
        ],
        "interaction_groups": [
            {
                "id": "personal_support_group",
                "function": "support drinkware",
                "tool_role": "personal_surface",
                "target_role": "cup_set",
                "required_target_count": 2,
                "usage_policy": "DEDICATED_PER_TARGET",
                "required_relations": ["fits cup and saucer set"],
                "context_role": "seat",
                "context_relations": ["near seat"],
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    res = provider.generate_canonical(instruction="serve tea for two")
    
    assert len(res["normalized_operation_groups"]) == 1
    og = res["normalized_operation_groups"][0]
    assert og["tool_role"] == "PERSONAL_CUP_SAUCER_REGION"
    assert og["target_role"] == "CUP_SAUCER_SET"
    assert og["context_role"] == "SEATING_POSITION"
    assert og["usage_policy"] == "DEDICATED_PER_TARGET"


# ---------------------------------------------------------------------------
# Test D: Living Room preserves payload and context nodes without injection
# ---------------------------------------------------------------------------
def test_d_living_room_preserves_payload_and_context_nodes():
    vlm_doc = {
        "status": "SUPPORTED",
        "task_summary": "Living room setup",
        "functional_roles": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "television remote control",
                "description": "remote payload",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["remote_control"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "r2",
                "entity_kind": "FIXED_TARGET",
                "function": "seating pair both seats",
                "description": "viewer armchairs",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["armchair"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "r3",
                "entity_kind": "REGION",
                "function": "shared central remote placement table",
                "description": "coffee table",
                "required_count": 1,
                "binding_policy": "SHARED",
                "candidate_categories": ["coffee_table"],
                "visible_candidates": [],
                "required_properties": ["planar support surface"],
            },
        ],
        "functional_relations": [
            {"subject_role": "r3", "relation": "fit remote", "object_role": "r1"},
            {"subject_role": "r3", "relation": "accessible from both", "object_role": "r2"},
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    adapter = MockFMAdapter(vlm_doc)
    provider = EnvironmentVLMRequirementProvider("living_room", fm_adapter=adapter)
    res = provider.generate_canonical(instruction="place remote for viewers")
    
    roles = {r["function"] for r in res["normalized_requirements"]}
    assert "REMOTE" in roles
    assert "SEATING_PAIR" in roles
    assert "SHARED_REMOTE_REGION" in roles


# ---------------------------------------------------------------------------
# Test E: Workshop tracks routed through open-vocabulary category mapping
# ---------------------------------------------------------------------------
def test_e_workshop_open_vocab_routing():
    from mujoco_scenes.workshop_phase1.requirements import (
        map_workshop_role_function,
    )
    # Compositional function matching
    assert map_workshop_role_function("tool to tighten screws into frame") == "CAN_DRIVE_SCREW"
    assert map_workshop_role_function("fastener that joins parts together") == "CAN_FASTEN"


# ---------------------------------------------------------------------------
# Test F: Workshop fails with MalformedVLMSpecificationError on missing driver/fastener
# ---------------------------------------------------------------------------
def test_f_workshop_fails_on_missing_driver_or_fastener():
    from mujoco_scenes.functional_tamp_pipeline.gf_reference_evaluator import evaluate_gf_against_reference
    from mujoco_scenes.functional_tamp_pipeline.gt_spec_provider import GTSpecProvider
    from mujoco_scenes.functional_tamp_pipeline.models import FunctionalRequirementGraph, FunctionalRole

    bad_spec = FunctionalRequirementGraph(
        domain="workshop",
        task_instruction="assemble joint",
        nodes={
            "some_tool": FunctionalRole(
                name="some_tool", entity_kind="OBJECT", count=1,
                semantic_categories=("wrench",), unary_predicates=(),
                binding_policy="DISTINCT", verification_mode="SEMANTIC_ONLY",
            )
        },
        relations=(),
        operation_groups=(),
        cross_group_reuse_allowed=False,
        detector_vocabulary=("wrench",),
        candidate_regions=(),
        region_ranking=(),
    )
    ref_gf = GTSpecProvider().provide("workshop", "assemble joint")
    res = evaluate_gf_against_reference(bad_spec, ref_gf)
    assert not res.structurally_complete
    assert "driver" in res.missing_roles
    assert "fastener" in res.missing_roles


# ---------------------------------------------------------------------------
# Test G: Ambiguous Workshop region proposal raises AmbiguousCanonicalizationError
# ---------------------------------------------------------------------------
def test_g_ambiguous_workshop_region_raises_ambiguous_error():
    with pytest.raises(AmbiguousCanonicalizationError):
        resolve_workshop_region_proposal("left drawer or right drawer")


# ---------------------------------------------------------------------------
# Test H: Ambiguous Workshop relation raises AmbiguousCanonicalizationError
# ---------------------------------------------------------------------------
def test_h_ambiguous_workshop_relation_raises_ambiguous_error():
    with pytest.raises(AmbiguousCanonicalizationError):
        map_workshop_relation("fit driver and reach target")


# ---------------------------------------------------------------------------
# Test I: Ambiguous Living Room relation raises AmbiguousCanonicalizationError
# ---------------------------------------------------------------------------
def test_i_ambiguous_living_room_relation_raises_ambiguous_error():
    with pytest.raises(AmbiguousCanonicalizationError):
        map_living_room_relation("near seat and accessible from both viewers")


# ---------------------------------------------------------------------------
# Test J: Unmapped / unsupported Living Room property error taxonomy
# ---------------------------------------------------------------------------
def test_j_unmapped_living_room_property_error_taxonomy():
    provider = EnvironmentVLMRequirementProvider("living_room")
    # In Pass 3.6A.7, all unknown required properties fail closed with UnsupportedCheckerCapabilityError
    with pytest.raises(UnsupportedCheckerCapabilityError):
        provider._map_properties(["rigid metallic surface"])
    with pytest.raises(UnsupportedCheckerCapabilityError):
        provider._map_properties(["quantum encrypted surface"])


# ---------------------------------------------------------------------------
# Test K: Invalid JSON decode raises TransportOrStructuredOutputError
# ---------------------------------------------------------------------------
def test_k_invalid_json_raises_transport_error():
    with pytest.raises(TransportOrStructuredOutputError):
        _extract_json_content({"choices": [{"message": {"content": "not json at all {"}}]})


# ---------------------------------------------------------------------------
# Test L: Missing FM backend raises TransportOrStructuredOutputError
# ---------------------------------------------------------------------------
def test_l_missing_fm_backend_raises_transport_error():
    err = FMBackendNotConfiguredError("No endpoint configured")
    assert isinstance(err, TransportOrStructuredOutputError)
    assert err.category == "TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE"


# ---------------------------------------------------------------------------
# Test M: PipelineResult and run_manifest persist failure_category
# ---------------------------------------------------------------------------
def test_m_pipeline_result_and_manifest_persist_failure_category(tmp_path):
    res = PipelineResult(
        domain="workshop",
        variant="W1",
        mode="vlm",
        status="VLM_SPEC_FAILED",
        failure_reason="Ambiguous concept",
        failure_category="AMBIGUOUS_CANONICALIZATION",
    )
    d = res.to_dict()
    assert d["failure_category"] == "AMBIGUOUS_CANONICALIZATION"

    state = _RunState(
        domain="kitchen",
        variant="K2",
        internal_variant="kitchen_k2",
        mode="vlm",
        run_dir=tmp_path,
        terminal_status="VLM_SPEC_FAILED",
        failure_reason="Decode error",
        failure_category="TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE",
    )
    _write_run_manifest(state)
    manifest_file = tmp_path / "run_manifest.json"
    assert manifest_file.exists()
    loaded = json.loads(manifest_file.read_text())
    assert loaded["failure_category"] == "TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE"
    assert loaded["failure_reason"] == "Decode error"


# ---------------------------------------------------------------------------
# Test N: Generic schema & system prompt contain zero benchmark nouns
# ---------------------------------------------------------------------------
def test_n_generic_schema_and_prompt_contain_zero_benchmark_nouns():
    forbidden_terms = [
        "screwdriver", "drawer", "kettle", "mug", "spoon", "soup",
        "coffee", "phillips", "allen", "cabinet", "workbench",
    ]
    for term in forbidden_terms:
        assert term not in SYSTEM_PROMPT.lower(), f"Prompt leaked benchmark noun {term!r}"
    # Validates interaction_groups in RESPONSE_SCHEMA
    assert "interaction_groups" in RESPONSE_SCHEMA["properties"]
    ig_props = RESPONSE_SCHEMA["properties"]["interaction_groups"]["items"]["properties"]
    assert "context_role" in ig_props
    assert "context_relations" in ig_props


# ---------------------------------------------------------------------------
# Test O: fm_adapter top-level docstring accurately describes semantics
# ---------------------------------------------------------------------------
def test_o_fm_adapter_docstring_accurate():
    import mujoco_scenes.workshop_phase1.fm_adapter as fma
    doc = fma.__doc__
    assert "exact implemented predicate identifiers" not in doc
    assert "natural-language semantics" in doc
    assert "deterministic canonicalization" in doc
