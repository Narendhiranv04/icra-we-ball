"""Pass 3.6A.3 Comprehensive Unit Tests.

Verifies:
1. VLM canonicalization version is phase3_6a3_v1.
2. Single live schema (functional_roles only) and rejection of missing semantic fields.
3. Strict unary/binary separation (required_properties is unary only; binary constraints only from functional_relations).
4. Raw semantics preservation (entity_kind, required_count, binding_policy, open-vocabulary categories).
5. Kitchen OPEN_CAVITY physical cleaning and fail-closed operation group matching.
6. Runner catches only VLMSpecificationError for VLM_SPEC_FAILED.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import VLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.models import (
    FunctionalRelation,
    FunctionalRequirementGraph,
    FunctionalRole,
)
from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import (
    VLMSpecProvider,
    VLM_CANONICALIZATION_VERSION,
)
from mujoco_scenes.kitchen_vlm_functional_graph import (
    compile_vlm_functional_graph,
    map_kitchen_role_function,
    map_unary_property,
    map_binary_relation,
    resolve_kitchen_region_proposal,
)
from mujoco_scenes.environment_vlm_requirements import (
    EnvironmentVLMRequirementProvider,
    map_living_room_role_function,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    FMResponseValidationError,
    validate_requirement_response,
    validate_kitchen_functional_specification,
    SYSTEM_PROMPT,
)
from mujoco_scenes.workshop_phase1.requirements import (
    FMRequirementProvider,
    map_workshop_role_function,
    map_workshop_unary_property,
    map_workshop_relation,
    resolve_workshop_region_proposal,
)


def test_vlm_canonicalization_version_constant():
    assert VLM_CANONICALIZATION_VERSION == "phase3_6a7_2_v1"


def test_single_role_field_enforced():
    # Legacy payload with functional_requirements instead of functional_roles must fail
    legacy_payload = {
        "status": "SUPPORTED",
        "task_summary": "Task with legacy field",
        "functional_requirements": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "turn screw",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            }
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    with pytest.raises(FMResponseValidationError, match="Unexpected top-level fields|missing required top-level field"):
        validate_requirement_response(legacy_payload)


def test_mandatory_role_fields_no_defaults():
    base_role = {
        "id": "r1",
        "entity_kind": "OBJECT",
        "function": "turn screw",
        "required_count": 1,
        "binding_policy": "DISTINCT",
        "candidate_categories": ["screwdriver"],
        "visible_candidates": [],
        "required_properties": [],
    }

    # Missing binding_policy
    no_binding = dict(base_role)
    del no_binding["binding_policy"]
    payload = {
        "status": "SUPPORTED",
        "task_summary": "Test",
        "functional_roles": [no_binding],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    with pytest.raises(FMResponseValidationError, match="missing required fields"):
        validate_requirement_response(payload)

    # Missing required_count
    no_count = dict(base_role)
    del no_count["required_count"]
    payload["functional_roles"] = [no_count]
    with pytest.raises(FMResponseValidationError, match="missing required fields"):
        validate_requirement_response(payload)

    # Missing candidate_categories
    no_cats = dict(base_role)
    del no_cats["candidate_categories"]
    payload["functional_roles"] = [no_cats]
    with pytest.raises(FMResponseValidationError, match="missing required fields"):
        validate_requirement_response(payload)

    # Missing visible_candidates
    no_cands = dict(base_role)
    del no_cands["visible_candidates"]
    payload["functional_roles"] = [no_cands]
    with pytest.raises(FMResponseValidationError, match="missing required fields"):
        validate_requirement_response(payload)


def test_workshop_unary_properties_never_routed_to_relations():
    # If VLM puts a relation in required_properties, workshop provider must fail closed
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "Drive screw",
        "functional_roles": [
            {
                "id": "d1",
                "entity_kind": "OBJECT",
                "function": "tighten threaded screw into workbench",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [{"label": "screwdriver", "visual_description": "black handle"}],
                "required_properties": ["long enough to reach workpiece hole recess"],  # Relation in unary field!
            }
        ],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }

    class FakeAdapter:
        def generate_task_requirements(self, task, observation_images=None):
            return raw_doc

    provider = FMRequirementProvider(fm_adapter=FakeAdapter())
    with pytest.raises(VLMSpecificationError, match="VLM unary property .* is not supported in Workshop"):
        provider.get_requirements()


def test_workshop_explicit_relations_fail_closed():
    # Unknown relation endpoint
    raw_doc = {
        "status": "SUPPORTED",
        "task_summary": "Drive screw",
        "functional_roles": [
            {
                "id": "d1",
                "entity_kind": "OBJECT",
                "function": "tighten threaded screw into workbench",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["screwdriver"],
                "visible_candidates": [],
                "required_properties": [],
            }
        ],
        "functional_relations": [
            {
                "subject_role": "d1",
                "relation": "fits driver bit to screw head",
                "object_role": "unknown_screw_role",
            }
        ],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }

    class FakeAdapter:
        def generate_task_requirements(self, task, observation_images=None):
            return raw_doc

    provider = FMRequirementProvider(fm_adapter=FakeAdapter())
    with pytest.raises(VLMSpecificationError, match="Relation object role 'unknown_screw_role' not declared"):
        provider.get_requirements()


def test_kitchen_open_cavity_semantic_cleaning():
    # 'contain coffee' or 'contain soup' alone must NOT map to OPEN_CAVITY
    assert map_unary_property("contain coffee") is None
    assert map_unary_property("contain soup") is None

    # Genuine physical language must map
    assert map_unary_property("open cavity") == "OPEN_CAVITY"
    assert map_unary_property("hollow receptacle") == "OPEN_CAVITY"
    assert map_unary_property("capable of holding liquid") == "OPEN_CAVITY"
    assert map_unary_property("elongated utensil") == "ELONGATED_OBJECT"


def test_kitchen_operation_group_fail_closed_no_rescue():
    # Operation group with mismatched tool/target roles must fail closed
    spec = {
        "status": "SUPPORTED",
        "task_summary": "Serve drinks",
        "functional_roles": [
            {
                "id": "r1",
                "entity_kind": "OBJECT",
                "function": "provide hot water",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["kettle"],
                "visible_candidates": [],
                "required_properties": [],
            },
            {
                "id": "r2",
                "entity_kind": "OBJECT",
                "function": "contain one serving of coffee",
                "required_count": 1,
                "binding_policy": "DISTINCT",
                "candidate_categories": ["cup"],
                "visible_candidates": [],
                "required_properties": ["open cavity"],
            },
        ],
        "functional_relations": [],
        "interaction_groups": [
            {
                "id": "g1",
                "function": "stir coffee in mug",
                "tool_role": "r1",  # water_source is not a coffee_stirrer!
                "target_role": "r2",
                "required_target_count": 1,
                "usage_policy": "SEQUENTIAL_REUSE_ALLOWED",
                "required_relations": ["insertable in"],
            }
        ],
        "inspectable_regions": [],
        "inspection_order": [],
        "cross_group_reuse_allowed": False,
        "unsupported_reason": "",
    }
    with pytest.raises(VLMSpecificationError, match="does not match any canonical kitchen operation group"):
        compile_vlm_functional_graph(
            spec,
            task_instruction="Serve drinks",
            observable_regions=("D1", "D2", "C2", "B1", "C1"),
        )


def test_unsupported_status_strict_validation():
    valid_unsupported = {
        "status": "UNSUPPORTED",
        "task_summary": "Do impossible task",
        "functional_roles": [],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "Task violates physical feasibility.",
    }
    validated = validate_requirement_response(valid_unsupported)
    assert validated["status"] == "UNSUPPORTED"
    assert validated["unsupported_reason"] != ""

    # Non-empty roles in UNSUPPORTED
    bad = dict(valid_unsupported)
    bad["functional_roles"] = [
        {
            "id": "r1",
            "entity_kind": "OBJECT",
            "function": "fn",
            "required_count": 1,
            "binding_policy": "DISTINCT",
            "candidate_categories": ["c"],
            "visible_candidates": [],
            "required_properties": [],
        }
    ]
    with pytest.raises(FMResponseValidationError, match="must have empty functional_roles"):
        validate_requirement_response(bad)

