"""Unit tests for Phase 2: V2 Schema and Prompt (Gate 2).

Tests:
1. V2 schema unit fixtures validate (positive and negative cases).
2. Prompt anti-leakage scan passes (zero benchmark nouns, zero verifier lists, zero closed relation enums).
3. V1 archived JSON still routes through V1 validator in validate_requirement_response.
4. V2 document routes through V2 validator and converts cleanly to canonical intermediate format.
5. Deterministic hash computation for V2 prompt and schema.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
import pytest

from mujoco_scenes.functional_tamp_pipeline.audit import (
    FORBIDDEN_CANONICAL_REGION_TOKENS,
    FORBIDDEN_CHECKER_STRINGS,
    FORBIDDEN_ORACLE_STRINGS,
)
from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    RESPONSE_SCHEMA_V2,
    SYSTEM_PROMPT_V2,
    compute_v2_prompt_and_schema_hash,
    convert_v2_to_canonical_document,
    is_v2_document,
    validate_v2_functional_specification,
)
from mujoco_scenes.functional_tamp_pipeline.structural_sanitizer import (
    sanitize_functional_graph,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    validate_requirement_response,
)


@pytest.fixture
def valid_v2_document():
    return {
        "status": "SUPPORTED",
        "task_summary": "Assemble the component at the target feature using a tool",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_tool",
                    "entity_kind": "OBJECT",
                    "function": "apply torque to secure component",
                    "required_count": 1,
                    "binding_policy": "REUSABLE",
                    "candidate_categories": ["fastening tool", "driver implement"],
                    "required_properties": ["elongated shaft"],
                },
                {
                    "id": "role_component",
                    "entity_kind": "OBJECT",
                    "function": "component to be secured",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["threaded fastener"],
                    "required_properties": [],
                },
                {
                    "id": "role_target",
                    "entity_kind": "FIXED_TARGET",
                    "function": "receiving hole feature",
                    "required_count": 1,
                    "binding_policy": "DISTINCT",
                    "candidate_categories": ["receiving receptacle"],
                    "required_properties": [],
                },
            ],
            "functional_relations": [
                {
                    "id": "rel_0",
                    "subject_role": "role_component",
                    "relation": "inserted into and secured at target feature",
                    "object_role": "role_target",
                    "required": True,
                },
                {
                    "id": "rel_1",
                    "subject_role": "role_tool",
                    "relation": "mechanically engages head of component",
                    "object_role": "role_component",
                    "required": True,
                },
            ],
            "operation_pairings": [
                {
                    "id": "op_0",
                    "operation": "drive and secure fastener",
                    "source_role": "role_tool",
                    "target_role": "role_component",
                    "operation_count": 1,
                    "reuse_policy": "REUSABLE_ACROSS_TARGETS",
                    "anchor_role": "role_target",
                }
            ],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {
                "role_tool": [
                    {"label": "driver_1", "visual_description": "blue handle driver on rack"}
                ],
                "role_component": [],
                "role_target": [
                    {"label": "fixture_hole", "visual_description": "threaded mounting hole on table"}
                ],
            },
            "inspectable_regions": [
                {
                    "id": "region_storage_1",
                    "description": "storage compartment that may contain small components",
                }
            ],
            "inspection_order": ["region_storage_1"],
        },
        "unsupported_reason": "",
    }


def test_valid_v2_document_validates(valid_v2_document):
    validated = validate_v2_functional_specification(valid_v2_document)
    assert validated["status"] == "SUPPORTED"
    assert len(validated["task_contract"]["functional_roles"]) == 3
    assert len(validated["task_contract"]["functional_relations"]) == 2
    assert len(validated["task_contract"]["operation_pairings"]) == 1


def test_is_v2_document_detection(valid_v2_document):
    assert is_v2_document(valid_v2_document) is True
    v1_doc = {
        "status": "SUPPORTED",
        "task_summary": "v1 summary",
        "functional_roles": [],
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": [],
        "inspection_order": [],
        "unsupported_reason": "",
    }
    assert is_v2_document(v1_doc) is False
    assert is_v2_document("not a doc") is False


def test_v2_schema_rejects_missing_contract_fields(valid_v2_document):
    bad = dict(valid_v2_document)
    del bad["task_contract"]
    with pytest.raises(MalformedVLMSpecificationError):
        validate_v2_functional_specification(bad)


def test_v2_schema_rejects_invalid_entity_kind(valid_v2_document):
    bad = json.loads(json.dumps(valid_v2_document))
    bad["task_contract"]["functional_roles"][0]["entity_kind"] = "INVALID_KIND"
    with pytest.raises(MalformedVLMSpecificationError):
        validate_v2_functional_specification(bad)


def test_v2_schema_rejects_invalid_binding_policy(valid_v2_document):
    bad = json.loads(json.dumps(valid_v2_document))
    bad["task_contract"]["functional_roles"][0]["binding_policy"] = "INVALID_POLICY"
    with pytest.raises(MalformedVLMSpecificationError):
        validate_v2_functional_specification(bad)


def test_v2_schema_rejects_undeclared_relation_endpoint(valid_v2_document):
    bad = json.loads(json.dumps(valid_v2_document))
    bad["task_contract"]["functional_relations"][0]["subject_role"] = "nonexistent_role"
    with pytest.raises(MalformedVLMSpecificationError, match="not in declared roles"):
        validate_v2_functional_specification(bad)


def test_v2_schema_rejects_undeclared_operation_target(valid_v2_document):
    bad = json.loads(json.dumps(valid_v2_document))
    bad["task_contract"]["operation_pairings"][0]["target_role"] = "nonexistent_role"
    with pytest.raises(MalformedVLMSpecificationError, match="not in declared roles"):
        validate_v2_functional_specification(bad)


def test_v2_schema_rejects_undeclared_guidance_role(valid_v2_document):
    bad = json.loads(json.dumps(valid_v2_document))
    bad["observation_guidance"]["visible_candidates_per_role"]["ghost_role"] = [
        {"label": "item", "visual_description": "something"}
    ]
    with pytest.raises(MalformedVLMSpecificationError, match="references undeclared role"):
        validate_v2_functional_specification(bad)


def test_v2_unsupported_specification():
    doc = {
        "status": "UNSUPPORTED",
        "task_summary": "Unachievable task",
        "task_contract": {
            "functional_roles": [],
            "functional_relations": [],
            "operation_pairings": [],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "Task violates safety policies",
    }
    validated = validate_v2_functional_specification(doc)
    assert validated["status"] == "UNSUPPORTED"
    assert validated["unsupported_reason"] == "Task violates safety policies"


def test_v2_convert_to_canonical_document(valid_v2_document):
    canonical = convert_v2_to_canonical_document(valid_v2_document)
    assert canonical["schema_version"] == 2
    assert canonical["status"] == "SUPPORTED"
    assert len(canonical["functional_roles"]) == 3
    assert len(canonical["functional_relations"]) == 2
    assert len(canonical["interaction_groups"]) == 1

    # Check that visible candidates were attached to roles
    tool_role = next(r for r in canonical["functional_roles"] if r["id"] == "role_tool")
    assert len(tool_role["visible_candidates"]) == 1
    assert tool_role["visible_candidates"][0]["label"] == "driver_1"

    # Check interaction group conversion
    group = canonical["interaction_groups"][0]
    assert group["tool_role"] == "role_tool"
    assert group["target_role"] == "role_component"
    assert group["context_role"] == "role_target"
    assert group["function"] == "drive and secure fastener"


def test_validate_requirement_response_routes_v1_and_v2(valid_v2_document):
    # V2 routing
    v2_res = validate_requirement_response(valid_v2_document)
    assert v2_res["status"] == "SUPPORTED"
    assert "task_contract" in v2_res

    # V1 routing
    w1_path = Path("mujoco_scenes/functional_tamp_pipeline/tests/fixtures/ideal_raw_vlm/workshop_W1.json")
    if w1_path.exists():
        v1_doc = json.loads(w1_path.read_text(encoding="utf-8"))
        v1_res = validate_requirement_response(v1_doc)
        assert v1_res["status"] == "SUPPORTED"
        assert "functional_roles" in v1_res
        assert "interaction_groups" in v1_res


def test_sanitize_functional_graph_accepts_v2(valid_v2_document):
    san = sanitize_functional_graph(valid_v2_document)
    assert san.succeeded is True
    assert len(san.document["functional_roles"]) == 3
    assert len(san.document["interaction_groups"]) == 1


def test_prompt_v2_anti_leakage_scan():
    """Verify that SYSTEM_PROMPT_V2 has zero forbidden checkers, region tokens, or oracle strings."""
    prompt = SYSTEM_PROMPT_V2

    for checker in FORBIDDEN_CHECKER_STRINGS:
        assert checker not in prompt, f"Forbidden checker string {checker!r} found in SYSTEM_PROMPT_V2"

    for region in FORBIDDEN_CANONICAL_REGION_TOKENS:
        # Check as whole word
        assert not re.search(r"\b" + re.escape(region) + r"\b", prompt), (
            f"Forbidden region token {region!r} found in SYSTEM_PROMPT_V2"
        )

    for oracle in FORBIDDEN_ORACLE_STRINGS:
        assert oracle not in prompt, f"Forbidden oracle string {oracle!r} found in SYSTEM_PROMPT_V2"


def test_prompt_v2_zero_benchmark_task_leakage():
    """Verify that SYSTEM_PROMPT_V2 has zero benchmark domain-specific nouns and examples."""
    benchmark_nouns = [
        "coffee", "soup", "toast", "salad", "saucer", "beverage",
        "mug", "pot", "carafe", "spoon", "stirrer",
        "screw", "bolt", "fastener", "screwdriver", "wrench", "clamp",
        "workbench", "dining table", "coffee table", "drawer", "cabinet",
        "D1", "D2", "C1", "C2", "B1",
    ]
    prompt_lower = SYSTEM_PROMPT_V2.lower()
    for noun in benchmark_nouns:
        assert not re.search(r"\b" + re.escape(noun.lower()) + r"\b", prompt_lower), (
            f"Benchmark noun/token {noun!r} leaked into SYSTEM_PROMPT_V2"
        )


def test_prompt_v2_distinguishes_physical_counts_operation_counts_and_reuse():
    prompt = SYSTEM_PROMPT_V2.lower()

    assert "minimum number of distinct physical instances" in prompt
    assert "do not set `required_count` equal to an operation count" in prompt
    assert "number of required applications" in prompt
    assert "does not assert that only one source exists" in prompt
    assert "visibility is evidence only" in prompt
    assert "multiple uses of one reusable source" in prompt


def test_v2_validates_reusable_source_with_multiple_operation_applications(valid_v2_document):
    doc = deepcopy(valid_v2_document)
    source, target = doc["task_contract"]["functional_roles"][:2]
    source.update(required_count=1, binding_policy="REUSABLE")
    target.update(required_count=2, binding_policy="DISTINCT")
    operation = doc["task_contract"]["operation_pairings"][0]
    operation.update(operation_count=2, reuse_policy="REUSABLE_ACROSS_TARGETS")

    validated = validate_v2_functional_specification(doc)
    assert validated["task_contract"]["functional_roles"][0]["required_count"] == 1
    assert validated["task_contract"]["operation_pairings"][0]["operation_count"] == 2


def test_v2_validates_distinct_source_structure(valid_v2_document):
    doc = deepcopy(valid_v2_document)
    source, target = doc["task_contract"]["functional_roles"][:2]
    source.update(required_count=2, binding_policy="DISTINCT")
    target.update(required_count=2, binding_policy="DISTINCT")
    doc["task_contract"]["operation_pairings"][0].update(
        operation_count=2, reuse_policy="DEDICATED_PER_TARGET"
    )

    validated = validate_v2_functional_specification(doc)
    assert validated["task_contract"]["functional_roles"][0]["binding_policy"] == "DISTINCT"


@pytest.mark.parametrize("required_count,visible_count", [(2, 1), (1, 2)])
def test_v2_visibility_never_rewrites_role_count(
    valid_v2_document, required_count, visible_count
):
    doc = deepcopy(valid_v2_document)
    role = doc["task_contract"]["functional_roles"][0]
    role["required_count"] = required_count
    doc["observation_guidance"]["visible_candidates_per_role"][role["id"]] = [
        {"label": f"candidate_{index}", "visual_description": "visible item"}
        for index in range(visible_count)
    ]

    validated = validate_v2_functional_specification(doc)
    assert validated["task_contract"]["functional_roles"][0]["required_count"] == required_count


def test_live_v2_provider_validates_cross_references_before_compilation(
    valid_v2_document, monkeypatch
):
    from mujoco_scenes.functional_tamp_pipeline import semantic_compiler
    from mujoco_scenes.functional_tamp_pipeline.vlm_spec_provider import VLMSpecProvider
    from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter

    malformed = deepcopy(valid_v2_document)
    malformed["task_contract"]["operation_pairings"][0]["source_role"] = "undeclared"
    monkeypatch.setattr(
        FMAdapter, "generate_kitchen_functional_graph",
        lambda *args, **kwargs: malformed,
    )
    monkeypatch.setattr(
        semantic_compiler, "compile_candidate_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("compiler must not receive invalid live V2")
        ),
    )

    with pytest.raises(MalformedVLMSpecificationError, match="not in declared roles"):
        VLMSpecProvider().provide("kitchen", "generic task", observation_images=[])


def test_prompt_v2_no_verifier_capability_block():
    """Verify that SYSTEM_PROMPT_V2 does not describe robot verifier capabilities."""
    assert "verifier capabilities" not in SYSTEM_PROMPT_V2.lower()
    assert "robot is equipped" not in SYSTEM_PROMPT_V2.lower()
    assert "open_cavity" not in SYSTEM_PROMPT_V2.lower()
    assert "insertable_in" not in SYSTEM_PROMPT_V2.lower()


def test_v2_prompt_and_schema_hash_deterministic():
    h1 = compute_v2_prompt_and_schema_hash()
    h2 = compute_v2_prompt_and_schema_hash()
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # must be valid hex
