"""Unit tests for Stage 8: Structured Output and Qwen Inference Reliability.

Verifies:
1. Request path inspection and frozen structured baseline configuration (Section 15.1, 15.2).
2. Detection and explicit attribution of token-limit truncations (finish_reason='length').
3. V2 response schema validity and strict draft-07 compatibility.
4. Exactly 1 semantic VLM call per run with zero duplicated calls (Section 15.3).
5. Diagnostic telemetry preserves finish_reason, token usage, and sanitized request metadata.
6. Thinking disabled by default (enable_thinking=false) for maximum schema stability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import pytest

from mujoco_scenes.functional_tamp_pipeline.errors import (
    TransportOrStructuredOutputError,
    MalformedVLMSpecificationError,
)
from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
    LIVE_RESPONSE_SCHEMA_V2,
    RESPONSE_SCHEMA_V2,
    SYSTEM_PROMPT_V2,
    is_v2_document,
    validate_v2_functional_specification,
)
from mujoco_scenes.workshop_phase1.fm_adapter import (
    FMAdapter,
    _extract_json_content,
    _save_fm_diagnostic,
)


class MockTransport:
    """Mock completion transport for deterministic unit testing."""

    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls: list[Mapping[str, Any]] = []

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append(payload)
        return self.response


def test_extract_json_content_finish_reason_length_empty_content():
    """Section 15.1: finish_reason='length' with empty content must attribute to token limit truncation."""
    raw_response = {
        "id": "cmpl-test-01",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "A" * 500,
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 8192, "total_tokens": 8292},
    }

    with pytest.raises(TransportOrStructuredOutputError) as exc_info:
        _extract_json_content(raw_response, call_kind="test_truncation")

    err_msg = str(exc_info.value)
    assert "truncated due to token limit" in err_msg
    assert "finish_reason='length'" in err_msg
    assert "after reasoning" in err_msg


def test_extract_json_content_finish_reason_length_truncated_json():
    """Section 15.1: finish_reason='length' with cut-off JSON must attribute to token limit truncation."""
    raw_response = {
        "id": "cmpl-test-02",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": '{"status": "SUPPORTED", "task_summary": "Prepare dinner',
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 8192, "total_tokens": 8292},
    }

    with pytest.raises(TransportOrStructuredOutputError) as exc_info:
        _extract_json_content(raw_response, call_kind="test_truncated_json")

    err_msg = str(exc_info.value)
    assert "truncated due to token limit" in err_msg
    assert "finish_reason='length'" in err_msg


def test_extract_json_content_valid_v2():
    """Section 15.3: Clean JSON payload parses correctly and strips markdown wrappers if present."""
    valid_doc = {
        "status": "SUPPORTED",
        "task_summary": "Fasten joint",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_driver",
                    "entity_kind": "OBJECT",
                    "function": "drive fastener",
                    "description": "tool",
                    "required_count": 1,
                    "candidate_categories": ["screwdriver"],
                    "required_properties": [],
                    "binding_policy": "REUSABLE",
                }
            ],
            "functional_relations": [],
            "operation_pairings": [],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }

    wrapped = f"```json\n{json.dumps(valid_doc)}\n```"
    raw_response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": wrapped,
                },
            }
        ]
    }

    extracted = _extract_json_content(raw_response)
    assert extracted == valid_doc
    assert is_v2_document(extracted)
    validated = validate_v2_functional_specification(extracted)
    assert validated["status"] == "SUPPORTED"


def test_thinking_disabled_by_default_in_adapter_payload(tmp_path):
    """Section 15.2 / Config A: Frozen baseline must disable thinking and enforce strict JSON schema."""
    dummy_img = tmp_path / "obs.png"
    dummy_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    mock_doc = {
        "status": "SUPPORTED",
        "task_summary": "Sample task",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "role_tool",
                    "entity_kind": "OBJECT",
                    "function": "manipulate object",
                    "description": "tool",
                    "required_count": 1,
                    "candidate_categories": ["tool"],
                    "required_properties": [],
                    "binding_policy": "REUSABLE",
                }
            ],
            "functional_relations": [],
            "operation_pairings": [],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }

    mock_transport = MockTransport({
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(mock_doc)},
            }
        ]
    })

    adapter = FMAdapter(
        base_url="http://127.0.0.1:18000/v1",
        model="qwen35-9b",
        transport=mock_transport,
    )

    doc = adapter.generate_task_requirements(
        "Find tool and perform operation",
        observation_images=[dummy_img],
    )

    assert len(mock_transport.calls) == 1
    call_payload = mock_transport.calls[0]

    # Verify Config A parameters
    assert call_payload["temperature"] == 0.0
    assert call_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert call_payload["response_format"]["type"] == "json_schema"
    assert call_payload["response_format"]["json_schema"]["strict"] is True
    assert call_payload["response_format"]["json_schema"]["schema"] == LIVE_RESPONSE_SCHEMA_V2
    # The frozen budget: 39 of the 91 archived responses that finished are
    # longer than the old 8192 default, so it would have truncated 43% of them.
    assert call_payload["max_tokens"] == 24000
    assert adapter.metrics.requirement_calls == 1
    assert adapter.metrics.total_calls == 1


def test_diagnostic_preserves_finish_reason_and_usage(tmp_path, monkeypatch):
    """Section 15.1: Diagnostic telemetry must save finish_reason, usage, and sanitized request."""
    diag_dir = tmp_path / "diagnostics"
    monkeypatch.setenv("TAMP_FM_DIAGNOSTIC_DIR", str(diag_dir))

    mock_response = {
        "model": "qwen35-9b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"status": "SUPPORTED"}'},
            }
        ],
        "usage": {
            "prompt_tokens": 512,
            "completion_tokens": 128,
            "total_tokens": 640,
        },
    }

    _save_fm_diagnostic(
        response=mock_response,
        content={"status": "SUPPORTED"},
        call_kind="task_requirements",
        json_parse_success=True,
        parse_error=None,
        sanitized_request={"schema_name": "functional_specification"},
    )

    saved_files = list(diag_dir.glob("fm_call_*.json"))
    assert len(saved_files) == 1

    saved = json.loads(saved_files[0].read_text(encoding="utf-8"))
    assert saved["model"] == "qwen35-9b"
    assert saved["finish_reason"] == "stop"
    assert saved["usage"]["completion_tokens"] == 128
    assert saved["json_parse_success"] is True
    assert saved["parse_error"] is None
    assert saved["sanitized_request"]["schema_name"] == "functional_specification"


def test_vlm_request_uses_only_first_three_ordered_views(tmp_path, monkeypatch):
    monkeypatch.setenv("TAMP_FM_SCHEMA_VERSION", "2")
    images = []
    for index in range(5):
        image = tmp_path / f"view_{index}.png"
        image.write_bytes(f"image-{index}".encode())
        images.append(image)

    mock_doc = {
        "status": "UNSUPPORTED",
        "task_summary": "Unsupported mock request",
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
        "unsupported_reason": "Mock transport response",
    }
    transport = MockTransport({
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(mock_doc)},
        }]
    })
    adapter = FMAdapter(
        base_url="http://127.0.0.1:18000/v1",
        model="qwen35-9b",
        transport=transport,
    )

    adapter.generate_task_requirements("Mock task", observation_images=images)

    assert images == [tmp_path / f"view_{index}.png" for index in range(5)]
    assert [Path(item["path"]) for item in adapter.last_observation_images] == images[:3]
    user_content = transport.calls[0]["messages"][1]["content"]
    assert len([item for item in user_content if item["type"] == "image_url"]) == 3


def test_live_request_reiterates_operation_endpoint_coherence(tmp_path, monkeypatch):
    monkeypatch.setenv("TAMP_FM_SCHEMA_VERSION", "2")
    image = tmp_path / "view.png"
    image.write_bytes(b"image")
    mock_doc = {
        "status": "UNSUPPORTED",
        "task_summary": "Unsupported mock request",
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
        "unsupported_reason": "Mock transport response",
    }
    transport = MockTransport({
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(mock_doc)},
        }]
    })
    adapter = FMAdapter(
        base_url="http://127.0.0.1:18000/v1",
        model="qwen35-9b",
        transport=transport,
    )

    adapter.generate_task_requirements("Mock task", observation_images=[image])

    request = json.loads(transport.calls[0]["messages"][1]["content"][0]["text"])[
        "request"
    ]
    assert "pairwise distinct" in request
    assert "omit anchor when the target itself" in request
    assert "never emit identification, selection, search, or inspection" in request


def test_response_schema_v2_enforces_valid_structure():
    """Gate 8: V2 schema strictly validates valid structures and rejects invalid constructs."""
    import jsonschema

    valid = {
        "status": "SUPPORTED",
        "task_summary": "Test task",
        "task_contract": {
            "functional_roles": [
                {
                    "id": "r1",
                    "entity_kind": "OBJECT",
                    "function": "contain items",
                    "description": "box",
                    "required_count": 1,
                    "candidate_categories": ["box"],
                    "required_properties": [],
                    "binding_policy": "DISTINCT",
                }
            ],
            "functional_relations": [],
            "operation_pairings": [],
        },
        "observation_guidance": {
            "visible_candidates_per_role": {},
            "inspectable_regions": [],
            "inspection_order": [],
        },
        "unsupported_reason": "",
    }

    # Should validate without error
    jsonschema.validate(instance=valid, schema=RESPONSE_SCHEMA_V2)

    # Invalid entity_kind must fail validation
    invalid = json.loads(json.dumps(valid))
    invalid["task_contract"]["functional_roles"][0]["entity_kind"] = "INVALID_KIND"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=RESPONSE_SCHEMA_V2)

    # Missing status must fail validation
    invalid_no_status = json.loads(json.dumps(valid))
    del invalid_no_status["status"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_no_status, schema=RESPONSE_SCHEMA_V2)
