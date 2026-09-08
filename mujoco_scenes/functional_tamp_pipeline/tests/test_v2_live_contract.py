"""Domain-neutral tests for the strict live V2 FM output contract."""

from __future__ import annotations

import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    SYSTEM_PROMPT_V2,
    convert_v2_to_canonical_document,
    validate_v2_functional_specification,
    validate_v2_live_contract,
)


def _role(
    role_id: str,
    function: str,
    *,
    count: int = 1,
    binding: str = "DISTINCT",
    kind: str = "OBJECT",
    properties: list[str] | None = None,
) -> dict:
    return {
        "id": role_id,
        "entity_kind": kind,
        "function": function,
        "required_count": count,
        "binding_policy": binding,
        "candidate_categories": [],
        "required_properties": properties or [],
    }


def _document(
    roles: list[dict],
    *,
    relations: list[dict] | None = None,
    operations: list[dict] | None = None,
    visible: dict | None = None,
    regions: list[dict] | None = None,
) -> dict:
    return {
        "status": "SUPPORTED",
        "task_summary": "Abstract physical transformation contract",
        "task_contract": {
            "functional_roles": roles,
            "functional_relations": relations or [],
            "operation_pairings": operations or [],
        },
        "observation_guidance": {
            "visible_candidates_per_role": visible or {},
            "inspectable_regions": regions or [],
            "inspection_order": [region["id"] for region in (regions or [])],
        },
        "unsupported_reason": "",
    }


def _relation(relation_id: str, subject: str, phrase: str, object_role: str) -> dict:
    return {
        "id": relation_id,
        "subject_role": subject,
        "relation": phrase,
        "object_role": object_role,
        "required": True,
    }


def _operation(
    operation_id: str,
    phrase: str,
    source: str,
    target: str,
    *,
    count: int = 1,
    reuse: str = "REUSABLE_ACROSS_TARGETS",
    anchor: str | None = None,
) -> dict:
    operation = {
        "id": operation_id,
        "operation": phrase,
        "source_role": source,
        "target_role": target,
        "operation_count": count,
        "reuse_policy": reuse,
    }
    if anchor is not None:
        operation["anchor_role"] = anchor
    return operation


def _expressiveness_documents() -> dict[str, dict]:
    return {
        "reusable_source_multiple_targets": _document(
            [
                _role("source", "material source", binding="REUSABLE"),
                _role("target", "receiving targets", count=2),
            ],
            operations=[_operation("transfer", "transfer material", "source", "target", count=2)],
        ),
        "dedicated_companion_per_target": _document(
            [
                _role("companion", "dedicated companion items", count=2),
                _role("target", "associated targets", count=2),
            ],
            relations=[_relation("pairing", "companion", "paired with", "target")],
            operations=[
                _operation(
                    "place_companion", "place companion", "companion", "target",
                    count=2, reuse="DEDICATED_PER_TARGET",
                )
            ],
        ),
        "tool_target_physical_relations": _document(
            [
                _role("tool", "reusable intervention tool", binding="REUSABLE", properties=["elongated"]),
                _role("target", "physical operation targets", count=2),
            ],
            relations=[
                _relation("fit", "tool", "fits inside", "target"),
                _relation("reach", "tool", "reaches bottom", "target"),
            ],
            operations=[_operation("apply", "apply tool", "tool", "target", count=2)],
        ),
        "payload_support_context_anchor": _document(
            [
                _role("payload", "movable payload"),
                _role("support", "placement support", kind="REGION", binding="SHARED"),
                _role("anchor", "contextual placement anchor", kind="FIXED_TARGET", binding="SHARED"),
            ],
            relations=[_relation("context", "support", "near anchor", "anchor")],
            operations=[_operation("place", "place payload", "payload", "support", anchor="anchor")],
        ),
        "tool_component_fixed_anchor": _document(
            [
                _role("tool", "component installation tool", binding="REUSABLE"),
                _role("component", "manipulated component"),
                _role("anchor", "fixed installation anchor", kind="FIXED_TARGET", binding="SHARED"),
            ],
            relations=[_relation("compatible", "tool", "compatible with", "component")],
            operations=[_operation("install", "install component", "tool", "component", anchor="anchor")],
        ),
        "desired_task_effect": _document(
            [_role("payload", "movable payload"), _role("support", "destination support", kind="REGION")],
            relations=[_relation("effect", "payload", "placed on support", "support")],
            operations=[_operation("place", "place payload", "payload", "support")],
        ),
        "partially_observed_role": _document(
            [_role("tool", "required manipulation tool", binding="REUSABLE")],
            visible={"tool": []},
            regions=[{"id": "storage", "description": "possible enclosed storage"}],
        ),
    }


@pytest.mark.parametrize("pattern", sorted(_expressiveness_documents()))
def test_generic_live_contract_expressiveness(pattern):
    document = _expressiveness_documents()[pattern]
    assert validate_v2_live_contract(document) == document


def test_task_effect_relation_is_preserved_as_semantics_not_observation():
    document = _expressiveness_documents()["desired_task_effect"]
    canonical = convert_v2_to_canonical_document(validate_v2_live_contract(document))
    assert canonical["functional_relations"][0]["relation"] == "placed on support"
    assert "observed" not in canonical["functional_relations"][0]


@pytest.mark.parametrize(
    ("section", "field", "code"),
    [
        ("functional_roles", "candidate_categories", "MISSING_LIVE_ROLE_FIELDS"),
        ("functional_relations", "required", "MISSING_LIVE_RELATION_FIELDS"),
        ("operation_pairings", "operation_count", "MISSING_LIVE_OPERATION_FIELDS"),
        ("operation_pairings", "reuse_policy", "MISSING_LIVE_OPERATION_FIELDS"),
        ("operation_pairings", "source_role", "MISSING_LIVE_OPERATION_FIELDS"),
    ],
)
def test_strict_live_requires_explicit_fields_but_archived_parser_remains_compatible(
    section, field, code
):
    document = _expressiveness_documents()["desired_task_effect"]
    item = document["task_contract"][section][0]
    del item[field]
    validate_v2_functional_specification(document)
    with pytest.raises(MalformedVLMSpecificationError, match=code):
        validate_v2_live_contract(document)


@pytest.mark.parametrize(
    ("section", "field", "phrase", "code"),
    [
        ("functional_relations", "relation", "fits inside; reaches bottom", "NON_ATOMIC_RELATION_PHRASE"),
        ("operation_pairings", "operation", "pick payload\nmove payload", "NON_ATOMIC_OPERATION_PHRASE"),
        ("functional_roles", "function", "one two three four five six seven eight nine ten eleven", "NON_ATOMIC_ROLE_FUNCTION"),
    ],
)
def test_live_contract_rejects_obvious_non_atomic_prose(section, field, phrase, code):
    document = _expressiveness_documents()["desired_task_effect"]
    document["task_contract"][section][0][field] = phrase
    with pytest.raises(MalformedVLMSpecificationError, match=code):
        validate_v2_live_contract(document)


def test_live_contract_allows_short_relation_with_conjunction_words():
    document = _expressiveness_documents()["tool_target_physical_relations"]
    document["task_contract"]["functional_relations"][0]["relation"] = (
        "accessible from both target positions"
    )
    validate_v2_live_contract(document)


def test_live_contract_rejects_obvious_binary_required_property():
    document = _expressiveness_documents()["tool_target_physical_relations"]
    document["task_contract"]["functional_roles"][0]["required_properties"] = [
        "fits inside target"
    ]
    with pytest.raises(MalformedVLMSpecificationError, match="BINARY_REQUIRED_PROPERTY"):
        validate_v2_live_contract(document)


def test_live_contract_rejects_self_paired_physical_operation():
    document = _expressiveness_documents()["desired_task_effect"]
    document["task_contract"]["operation_pairings"][0]["target_role"] = "payload"
    with pytest.raises(MalformedVLMSpecificationError, match="INVALID_OPERATION_SELF_PAIRING"):
        validate_v2_live_contract(document)


@pytest.mark.parametrize(
    ("section", "field", "code"),
    [
        ("functional_relations", "id", "INVALID_LIVE_RELATION_ID"),
        ("operation_pairings", "id", "INVALID_LIVE_OPERATION_ID"),
        ("operation_pairings", "source_role", "INVALID_OPERATION_REFERENCE"),
    ],
)
def test_live_contract_rejects_empty_identifiers_and_operation_endpoints(section, field, code):
    document = _expressiveness_documents()["desired_task_effect"]
    document["task_contract"][section][0][field] = ""
    with pytest.raises(MalformedVLMSpecificationError, match=code):
        validate_v2_live_contract(document)


def test_production_prompt_has_no_benchmark_answer_or_variant_leakage():
    prompt = SYSTEM_PROMPT_V2.casefold()
    forbidden = (
        "k1", "k2", "k3", "l1", "w1", "expected assignment", "expected plan",
        "ground truth", "variant-specific counts",
    )
    assert not [term for term in forbidden if term in prompt]
    assert "canonical predicate" not in prompt
