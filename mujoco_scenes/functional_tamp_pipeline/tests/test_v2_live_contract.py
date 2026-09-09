"""Domain-neutral tests for the strict live V2 FM output contract."""

from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    LIVE_RESPONSE_SCHEMA_V2,
    RESPONSE_SCHEMA_V2,
    SYSTEM_PROMPT_V2,
    compute_v2_prompt_and_schema_hash,
    convert_v2_to_canonical_document,
    validate_v2_functional_specification,
    validate_v2_live_contract,
)
from mujoco_scenes.workshop_phase1.fm_adapter import FMAdapter


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


def _region(region_id: str) -> dict:
    return {
        "id": region_id,
        "label": "visible storage region",
        "visual_description": "closed region visible in the observation",
        "reason": "could be inspected for missing requirements",
    }


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
            regions=[_region("storage")],
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


def _three_endpoint_operation_document(phrase: str = "install component") -> dict:
    return _document(
        [
            _role("tool", "physical intervention tool", binding="REUSABLE"),
            _role("component", "manipulated component"),
            _role("fixed_location", "fixed operation location", kind="FIXED_TARGET"),
        ],
        operations=[
            _operation(
                "operation", phrase, "tool", "component", anchor="fixed_location"
            )
        ],
    )


def test_live_contract_accepts_distinct_three_endpoint_operation():
    document = _three_endpoint_operation_document()
    assert validate_v2_live_contract(document) == document


@pytest.mark.parametrize(
    ("source", "target", "anchor", "code"),
    [
        ("tool", "tool", "fixed_location", "INVALID_OPERATION_SELF_PAIRING"),
        ("tool", "component", "tool", "DUPLICATE_OPERATION_ENDPOINT"),
        ("tool", "component", "component", "DUPLICATE_OPERATION_ENDPOINT"),
    ],
)
def test_live_contract_rejects_duplicate_operation_endpoints(source, target, anchor, code):
    document = _three_endpoint_operation_document()
    operation = document["task_contract"]["operation_pairings"][0]
    operation.update(source_role=source, target_role=target, anchor_role=anchor)
    with pytest.raises(MalformedVLMSpecificationError, match=code):
        validate_v2_live_contract(document)


@pytest.mark.parametrize(
    "phrase",
    [
        "select compatible component",
        "identify suitable tool",
        "search for component",
        "choose appropriate item",
        "find usable object",
        "inspect storage region",
    ],
)
def test_live_contract_rejects_leading_non_physical_operation(phrase):
    document = _three_endpoint_operation_document(phrase)
    with pytest.raises(MalformedVLMSpecificationError, match="NON_PHYSICAL_OPERATION"):
        validate_v2_live_contract(document)


def test_live_contract_allows_selected_as_adjective_in_physical_operation():
    document = _three_endpoint_operation_document("place selected component")
    assert validate_v2_live_contract(document) == document


def test_live_contract_rejects_explicit_unbound_role_mention():
    document = _document(
        [
            _role("tool", "physical intervention tool"),
            _role("manipulated_component", "manipulated component"),
            _role("fixed_target", "fixed target", kind="FIXED_TARGET"),
        ],
        operations=[
            _operation(
                "install", "install manipulated component", "tool", "fixed_target"
            )
        ],
    )
    with pytest.raises(
        MalformedVLMSpecificationError, match="OPERATION_MENTIONS_UNBOUND_ROLE"
    ):
        validate_v2_live_contract(document)


def test_live_contract_accepts_explicit_bound_role_mention():
    document = _document(
        [
            _role("tool", "physical intervention tool"),
            _role("manipulated_component", "manipulated component"),
            _role("fixed_target", "fixed target", kind="FIXED_TARGET"),
        ],
        operations=[
            _operation(
                "install",
                "install manipulated component",
                "tool",
                "manipulated_component",
                anchor="fixed_target",
            )
        ],
    )
    assert validate_v2_live_contract(document) == document


def test_live_contract_does_not_bind_unmentioned_declared_role():
    document = _three_endpoint_operation_document("apply tool")
    document["task_contract"]["functional_roles"].append(
        _role("unmentioned_participant", "unmentioned physical participant")
    )
    assert validate_v2_live_contract(document) == document


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


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("functional_roles", "candidate_categories"),
        ("functional_roles", "required_properties"),
        ("functional_relations", "id"),
        ("functional_relations", "required"),
        ("operation_pairings", "id"),
        ("operation_pairings", "source_role"),
        ("operation_pairings", "operation_count"),
        ("operation_pairings", "reuse_policy"),
    ],
)
def test_live_schema_and_validator_agree_on_every_required_field(section, field):
    document = _expressiveness_documents()["desired_task_effect"]
    jsonschema.validate(document, LIVE_RESPONSE_SCHEMA_V2)
    validate_v2_live_contract(document)

    del document["task_contract"][section][0][field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, LIVE_RESPONSE_SCHEMA_V2)
    with pytest.raises(MalformedVLMSpecificationError):
        validate_v2_live_contract(document)


def test_legacy_schema_and_parser_retain_archived_defaults():
    document = _expressiveness_documents()["desired_task_effect"]
    relation = document["task_contract"]["functional_relations"][0]
    operation = document["task_contract"]["operation_pairings"][0]
    del relation["id"]
    del relation["required"]
    for field in ("id", "source_role", "operation_count", "reuse_policy"):
        del operation[field]

    jsonschema.validate(document, RESPONSE_SCHEMA_V2)
    validate_v2_functional_specification(document)
    canonical = convert_v2_to_canonical_document(document)
    assert canonical["functional_relations"][0]["required"] is True
    assert canonical["interaction_groups"][0]["required_target_count"] == 1
    assert canonical["interaction_groups"][0]["usage_policy"] == "DEDICATED_PER_TARGET"
    with pytest.raises(MalformedVLMSpecificationError):
        validate_v2_live_contract(document)


def test_live_schema_allows_semantically_valid_unsupported_response():
    document = _document([])
    document.update(
        status="UNSUPPORTED",
        task_summary="Unsupported abstract request",
        unsupported_reason="Cannot represent requested transformation",
    )
    jsonschema.validate(document, LIVE_RESPONSE_SCHEMA_V2)
    assert validate_v2_live_contract(document)["status"] == "UNSUPPORTED"


def test_live_observation_guidance_accepts_empty_regions_and_order():
    document = _expressiveness_documents()["desired_task_effect"]
    assert validate_v2_live_contract(document) == document


def test_live_observation_guidance_accepts_complete_ranked_regions():
    document = _expressiveness_documents()["desired_task_effect"]
    guidance = document["observation_guidance"]
    guidance["inspectable_regions"] = [_region("region_1"), _region("region_2")]
    guidance["inspection_order"] = ["region_2", "region_1"]
    assert validate_v2_live_contract(document) == document


@pytest.mark.parametrize("field", ["id", "label", "visual_description", "reason"])
def test_live_region_schema_requires_each_observation_field_but_legacy_allows_omission(field):
    document = _expressiveness_documents()["desired_task_effect"]
    guidance = document["observation_guidance"]
    guidance["inspectable_regions"] = [_region("region_1")]
    guidance["inspection_order"] = ["region_1"]
    del guidance["inspectable_regions"][0][field]

    jsonschema.validate(document, RESPONSE_SCHEMA_V2)
    validate_v2_functional_specification(document)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, LIVE_RESPONSE_SCHEMA_V2)
    with pytest.raises(MalformedVLMSpecificationError, match="Live V2 schema validation failed"):
        validate_v2_live_contract(document)


def test_live_region_schema_rejects_additional_properties():
    document = _expressiveness_documents()["desired_task_effect"]
    region = _region("region_1")
    region["unexpected"] = "metadata"
    document["observation_guidance"]["inspectable_regions"] = [region]
    document["observation_guidance"]["inspection_order"] = ["region_1"]
    with pytest.raises(MalformedVLMSpecificationError, match="Live V2 schema validation failed"):
        validate_v2_live_contract(document)


@pytest.mark.parametrize("region_id", ["", "region one", "region-1", "region/1"])
def test_live_region_schema_requires_identifier_safe_nonempty_id(region_id):
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["inspectable_regions"] = [_region(region_id)]
    document["observation_guidance"]["inspection_order"] = [region_id]
    with pytest.raises(MalformedVLMSpecificationError, match="Live V2 schema validation failed"):
        validate_v2_live_contract(document)


@pytest.mark.parametrize("field", ["label", "visual_description", "reason"])
def test_live_region_schema_requires_nonempty_descriptive_fields(field):
    document = _expressiveness_documents()["desired_task_effect"]
    region = _region("region_1")
    region[field] = ""
    document["observation_guidance"]["inspectable_regions"] = [region]
    document["observation_guidance"]["inspection_order"] = ["region_1"]
    with pytest.raises(MalformedVLMSpecificationError, match="Live V2 schema validation failed"):
        validate_v2_live_contract(document)


def test_live_observation_guidance_rejects_duplicate_region_id():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["inspectable_regions"] = [
        _region("region_1"), _region("region_1")
    ]
    document["observation_guidance"]["inspection_order"] = ["region_1"]
    with pytest.raises(MalformedVLMSpecificationError, match="DUPLICATE_INSPECTABLE_REGION_ID"):
        validate_v2_live_contract(document)


def test_live_observation_guidance_rejects_unknown_order_region():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["inspectable_regions"] = [_region("region_1")]
    document["observation_guidance"]["inspection_order"] = ["region_2"]
    with pytest.raises(MalformedVLMSpecificationError, match="UNKNOWN_INSPECTION_ORDER_REGION"):
        validate_v2_live_contract(document)


def test_live_observation_guidance_rejects_duplicate_order_id():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["inspectable_regions"] = [_region("region_1")]
    document["observation_guidance"]["inspection_order"] = ["region_1", "region_1"]
    with pytest.raises(MalformedVLMSpecificationError, match="DUPLICATE_INSPECTION_ORDER_ID"):
        validate_v2_live_contract(document)


def test_live_observation_guidance_rejects_incomplete_order():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["inspectable_regions"] = [
        _region("region_1"), _region("region_2")
    ]
    document["observation_guidance"]["inspection_order"] = ["region_1"]
    with pytest.raises(MalformedVLMSpecificationError, match="INCOMPLETE_INSPECTION_ORDER"):
        validate_v2_live_contract(document)


def test_live_observation_guidance_rejects_order_without_regions():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["inspection_order"] = ["region_1"]
    with pytest.raises(MalformedVLMSpecificationError, match="UNKNOWN_INSPECTION_ORDER_REGION"):
        validate_v2_live_contract(document)


def test_omitted_visible_role_key_means_empty_evidence_without_role_deletion():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["visible_candidates_per_role"] = {"payload": []}
    validated = validate_v2_live_contract(document)
    canonical = convert_v2_to_canonical_document(validated)
    support = next(role for role in canonical["functional_roles"] if role["id"] == "support")
    assert support["visible_candidates"] == []
    assert {role["id"] for role in canonical["functional_roles"]} == {"payload", "support"}


def test_live_observation_guidance_rejects_visible_candidate_for_undeclared_role():
    document = _expressiveness_documents()["desired_task_effect"]
    document["observation_guidance"]["visible_candidates_per_role"]["undeclared"] = []
    with pytest.raises(MalformedVLMSpecificationError, match="references undeclared role"):
        validate_v2_live_contract(document)


class _RecordingTransport:
    def __init__(self, document):
        self.document = document
        self.calls = []

    def complete(self, payload):
        self.calls.append(payload)
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(self.document)},
                }
            ]
        }


def _capture_domain_payload(domain: str, document: dict, observation_image) -> dict:
    transport = _RecordingTransport(document)
    adapter = FMAdapter(
        base_url="http://127.0.0.1:18000/v1",
        model="test-model",
        transport=transport,
    )
    if domain == "kitchen":
        adapter.generate_kitchen_functional_graph(
            "abstract task", observation_images=[observation_image]
        )
    else:
        adapter.generate_task_requirements(
            "abstract task", observation_images=[observation_image]
        )
    return transport.calls[0]


def test_all_domain_requests_send_the_same_strict_live_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("TAMP_FM_SCHEMA_VERSION", "2")
    document = _expressiveness_documents()["desired_task_effect"]
    observation_image = tmp_path / "observation.png"
    observation_image.write_bytes(b"synthetic image")
    payloads = [_capture_domain_payload(domain, document, observation_image) for domain in (
        "kitchen", "living_room", "workshop"
    )]
    schemas = [payload["response_format"]["json_schema"]["schema"] for payload in payloads]

    assert all(payload["response_format"]["type"] == "json_schema" for payload in payloads)
    assert all(payload["response_format"]["json_schema"]["strict"] is True for payload in payloads)
    assert all(schema is LIVE_RESPONSE_SCHEMA_V2 for schema in schemas)
    assert len({hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                for schema in schemas}) == 1

    contract_schema = schemas[0]["properties"]["task_contract"]["properties"]
    assert {"candidate_categories", "required_properties"} <= set(
        contract_schema["functional_roles"]["items"]["required"]
    )
    assert {"id", "required"} <= set(
        contract_schema["functional_relations"]["items"]["required"]
    )
    assert {"id", "source_role", "operation_count", "reuse_policy"} <= set(
        contract_schema["operation_pairings"]["items"]["required"]
    )
    region_schema = schemas[0]["properties"]["observation_guidance"]["properties"][
        "inspectable_regions"
    ]["items"]
    assert set(region_schema["required"]) == {
        "id", "label", "visual_description", "reason"
    }
    assert region_schema["additionalProperties"] is False


def test_fake_transport_cannot_bypass_post_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("TAMP_FM_SCHEMA_VERSION", "2")
    document = _expressiveness_documents()["desired_task_effect"]
    del document["task_contract"]["functional_relations"][0]["id"]
    del document["task_contract"]["functional_relations"][0]["required"]
    observation_image = tmp_path / "observation.png"
    observation_image.write_bytes(b"synthetic image")

    with pytest.raises(MalformedVLMSpecificationError, match="MISSING_LIVE_RELATION_FIELDS"):
        _capture_domain_payload("workshop", document, observation_image)


def test_live_prompt_schema_hash_matches_actual_wire_contract():
    expected = hashlib.sha256(json.dumps(
        {
            "system_prompt_v2": SYSTEM_PROMPT_V2,
            "response_schema_v2": LIVE_RESPONSE_SCHEMA_V2,
        },
        sort_keys=True,
    ).encode()).hexdigest()
    assert compute_v2_prompt_and_schema_hash() == expected
