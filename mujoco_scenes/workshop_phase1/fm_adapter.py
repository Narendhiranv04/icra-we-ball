"""One-shot Qwen requirement decomposition over an OpenAI-compatible server.

This module intentionally stops before observation search, grounding, planning,
or execution. All domains ask the model for natural-language semantics; downstream
code performs strict deterministic canonicalization into canonical functional requirement graphs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from typing import Any, Mapping, Protocol

try:
    from mujoco_scenes.functional_tamp_pipeline.errors import (
        MalformedVLMSpecificationError,
        TransportOrStructuredOutputError,
        VLMSpecificationError,
    )
except ImportError:
    class VLMSpecificationError(Exception):
        """Fallback base error for VLM specification failures."""
        category = "UNMAPPED_FUNCTIONAL_CONCEPT"

    class MalformedVLMSpecificationError(VLMSpecificationError):
        category = "MALFORMED_VLM_SPECIFICATION"

    class TransportOrStructuredOutputError(VLMSpecificationError):
        category = "TRANSPORT_OR_STRUCTURED_OUTPUT_FAILURE"


class FMBackendNotConfiguredError(TransportOrStructuredOutputError):
    """Raised when live requirement generation has no configured endpoint."""


class FMTransportError(TransportOrStructuredOutputError):
    """Raised when the inference server cannot return a usable completion."""


class FMResponseValidationError(MalformedVLMSpecificationError):
    """Raised when the model response violates the transport-level schema."""


@dataclass
class FMCallMetrics:
    requirement_calls: int = 0
    search_prior_calls: int = 0
    total_calls: int = 0


class CompletionTransport(Protocol):
    """Injectable transport used by tests and the live HTTP client."""

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        pass


class OpenAICompletionTransport:
    """Small stdlib client for vLLM/SGLang's OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def complete(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise FMTransportError(
                f"Inference server returned HTTP {error.code}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise FMTransportError(
                f"Cannot reach inference server at {self.url}: {error.reason}"
            ) from error
        except (TimeoutError, json.JSONDecodeError) as error:
            raise FMTransportError(f"Invalid inference-server response: {error}") from error
        if not isinstance(decoded, dict):
            raise FMTransportError("Inference server returned non-object JSON")
        return decoded


SYSTEM_PROMPT = """You are a vision-language functional-requirement specification generator.

Return only the requested JSON object. Do not produce an action sequence.

Rules:
- Infer the complete set of physical or spatial functional roles from the task
  instruction and initial multi-view RGB images yourself. The user will not supply
  expected roles, functions, object categories, or properties.
- Create functional roles for scene assets whose identity, suitability, or
  functional capability must be discovered or selected to accomplish the task.
  Objects explicitly specified by the task as payloads or fixed contextual
  entities need not be reintroduced as selectable functional roles unless their
  functional suitability itself must be discovered.
- Use SHORT ATOMIC PHRASES for all functions, properties, and relations:
  - Role function: describe what the physical candidate must be capable of doing (e.g. "contain hot liquid", "stir drink", "drive fastener", "hold items for viewer"), rather than abstract workflow stages (e.g. "coffee preparation", "serving process").
  - Required properties: list only task-critical physical or geometric characteristics of this single role used to decide candidate suitability (e.g. open cavity, elongated shape, planar horizontal support). Do NOT include non-physical adjectives (e.g. good, useful, safe, edible, hot), task state descriptions, or semantic class labels already covered in candidate_categories.
  - Functional relations: describe physical spatial or interface compatibility relations between roles (e.g. "fits into", "reaches into", "compatible with", "placed on", "near seat").
  Do not write long narrative sentences. Do not use complex compound clauses.
- Robot Verifier Capabilities:
  The robot is equipped with physical and geometric verifiers that can check concepts such as:
  * Unary physical shapes: whether an object has an open/deep cavity or container volume; whether an object is elongated enough to serve as an implement; whether a surface is a flat/planar support.
  * Spatial & container relations: whether one object/implement can fit into or enter another object's opening; whether an implement reaches sufficiently deep into a container; whether a region can support a payload.
  * Seating & proximity relations: relative proximity or accessibility of support surfaces to seating/viewers; whether a support is accessible to multiple seating positions.
  * Tool & fastener interfaces: interface compatibility between a tool/driver and a fastener; whether a tool reaches a target workpiece/hole; whether a fastener is compatible with a target opening.
- When a role must be paired independently with multiple task targets or
  contextual references, represent that dependency using an interaction group
  rather than relying on an unconstrained many-to-many relation.
- Set `entity_kind` to:
  - OBJECT: a selectable/manipulable physical item.
  - REGION: a selectable support surface, placement area, or spatial destination.
  - FIXED_TARGET: a non-selectable contextual reference or fixed target feature that participates in relations.
- Set `binding_policy` to:
  - DISTINCT: separate simultaneous physical items are required.
  - REUSABLE: one physical item may be reused sequentially across targets.
  - SHARED: one physical region/entity intentionally serves multiple items/users.
- `candidate_categories`: list open-vocabulary semantic search phrases that could satisfy the role, even if nothing is currently visible.
- `visible_candidates`: list visually apparent items/regions in the initial RGB views.
  This array may be empty ([]).
- `required_properties`: list UNARY-ONLY physical properties of this single role.
  Never place binary relations or compatibility statements here.
- `functional_relations`: list explicit role-to-role relations using `subject_role`, `relation`, and `object_role`.
  Both subject_role and object_role must reference declared role IDs.
- `interaction_groups`: list structured interaction groups with tool_role, target_role, required_target_count, usage_policy, required_relations, and optional context_role/context_relations.
- `inspectable_regions`: propose visible closed/storage regions in the initial images that could be inspected if required items are missing.
- `inspection_order`: rank the proposed inspectable region IDs.
- Status semantics:
  - `SUPPORTED`: task can be represented with functional roles and relations. `functional_roles` must be non-empty, `unsupported_reason` must be empty ("").
  - `UNSUPPORTED`: use only when the task itself cannot be represented by this abstraction. `functional_roles`, `functional_relations`, `interaction_groups`, `inspectable_regions`, `inspection_order` must be empty ([]), and `unsupported_reason` must be a non-empty explanation.
  - Partial observability, missing visible candidates, unmeasured continuous geometry, or needing inspection/search are NOT reasons for UNSUPPORTED.
"""


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
        "task_summary": {"type": "string"},
        "functional_roles": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "entity_kind": {
                        "type": "string",
                        "enum": ["OBJECT", "REGION", "FIXED_TARGET"],
                    },
                    "function": {"type": "string"},
                    "description": {"type": "string"},
                    "required_count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "binding_policy": {
                        "type": "string",
                        "enum": ["DISTINCT", "REUSABLE", "SHARED"],
                    },
                    "candidate_categories": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 12,
                        "items": {"type": "string"},
                    },
                    "visible_candidates": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "visual_description": {"type": "string"},
                                "suitability_reason": {"type": "string"},
                            },
                            "required": ["label", "visual_description"],
                            "additionalProperties": False,
                        },
                    },
                    "required_properties": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 16,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "entity_kind",
                    "function",
                    "required_count",
                    "binding_policy",
                    "candidate_categories",
                    "visible_candidates",
                    "required_properties",
                ],
                "additionalProperties": False,
            },
        },
        "functional_relations": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "subject_role": {"type": "string"},
                    "relation": {"type": "string"},
                    "object_role": {"type": "string"},
                },
                "required": ["subject_role", "relation", "object_role"],
                "additionalProperties": False,
            },
        },
        "interaction_groups": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "function": {"type": "string"},
                    "tool_role": {"type": "string"},
                    "target_role": {"type": "string"},
                    "required_target_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "usage_policy": {
                        "type": "string",
                        "enum": ["SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"],
                    },
                    "required_relations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "context_role": {"type": "string"},
                    "context_relations": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 12,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id",
                    "function",
                    "tool_role",
                    "target_role",
                    "required_target_count",
                    "usage_policy",
                    "required_relations",
                ],
                "additionalProperties": False,
            },
        },
        "inspectable_regions": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "label", "visual_description", "reason"],
                "additionalProperties": False,
            },
        },
        "inspection_order": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unsupported_reason": {"type": "string"},
    },
    "required": [
        "status",
        "task_summary",
        "functional_roles",
        "functional_relations",
        "interaction_groups",
        "inspectable_regions",
        "inspection_order",
        "unsupported_reason",
    ],
    "additionalProperties": False,
}

INSPECTION_POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "initial_requirements_satisfied": {"type": "boolean"},
        "decision_reason": {"type": "string"},
        "inspectable_regions": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "label", "visual_description", "reason"],
                "additionalProperties": False,
            },
        },
        "inspection_order": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "initial_requirements_satisfied", "decision_reason",
        "inspectable_regions", "inspection_order",
    ],
    "additionalProperties": False,
}

KITCHEN_FUNCTIONAL_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
        "task_summary": {"type": "string"},
        "functional_roles": {
            "type": "array", "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "entity_kind": {"type": "string", "enum": ["OBJECT", "REGION", "FIXED_TARGET"]},
                    "function": {"type": "string"},
                    "required_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "binding_policy": {"type": "string", "enum": ["DISTINCT", "REUSABLE", "SHARED"]},
                    "candidate_categories": {
                        "type": "array", "minItems": 1, "maxItems": 12,
                        "items": {"type": "string"},
                    },
                    "visible_candidates": {
                        "type": "array", "minItems": 0, "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "visual_description": {"type": "string"},
                            },
                            "required": ["label", "visual_description"],
                            "additionalProperties": False,
                        },
                    },
                    "required_properties": {
                        "type": "array", "minItems": 0, "maxItems": 12,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id", "entity_kind", "function", "required_count",
                    "binding_policy", "candidate_categories", "visible_candidates", "required_properties",
                ],
                "additionalProperties": False,
            },
        },
        "functional_relations": {
            "type": "array", "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "subject_role": {"type": "string"},
                    "relation": {"type": "string"},
                    "object_role": {"type": "string"},
                },
                "required": ["subject_role", "relation", "object_role"],
                "additionalProperties": False,
            },
        },
        "interaction_groups": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "function": {"type": "string"},
                    "tool_role": {"type": "string"},
                    "target_role": {"type": "string"},
                    "required_target_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "usage_policy": {
                        "type": "string",
                        "enum": ["SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"],
                    },
                    "required_relations": {
                        "type": "array", "minItems": 1, "maxItems": 12,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": [
                    "id", "function", "tool_role", "target_role",
                    "required_target_count", "usage_policy", "required_relations",
                ],
                "additionalProperties": False,
            },
        },
        "cross_group_reuse_allowed": {"type": "boolean"},
        "inspectable_regions": {
            "type": "array", "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "visual_description": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "label", "visual_description", "reason"],
                "additionalProperties": False,
            },
        },
        "inspection_order": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unsupported_reason": {"type": "string"},
    },
    "required": [
        "status", "task_summary", "functional_roles", "functional_relations",
        "interaction_groups", "cross_group_reuse_allowed",
        "inspectable_regions", "inspection_order",
        "unsupported_reason",
    ],
    "additionalProperties": False,
}


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _short_string(value: object, maximum: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value.strip()) <= maximum


def _encode_observation_images(
    paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise ValueError("At least one initial-observation image is required")
    if len(paths) > 8:
        raise ValueError("At most eight initial-observation images are supported")
    blocks: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    total_bytes = 0
    for source in paths:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Initial-observation image is unavailable: {path}")
        data = path.read_bytes()
        total_bytes += len(data)
        if len(data) > 20 * 1024 * 1024 or total_bytes > 64 * 1024 * 1024:
            raise ValueError("Initial-observation image payload is too large")
        mime_type = mimetypes.guess_type(path.name)[0]
        if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"Unsupported observation image type: {path}")
        encoded = base64.b64encode(data).decode("ascii")
        sha256 = hashlib.sha256(data).hexdigest()
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{encoded}",
                    "detail": "high",
                },
            }
        )
        metadata.append(
            {
                "path": str(path),
                "mime_type": mime_type,
                "bytes": len(data),
                "sha256": sha256,
            }
        )
    return blocks, metadata


def _save_fm_diagnostic(
    response: Any,
    content: Any,
    call_kind: str,
    json_parse_success: bool,
    parse_error: str | None = None,
    *,
    sanitized_request: dict[str, Any] | None = None,
) -> None:
    diag_dir_env = os.environ.get("TAMP_FM_DIAGNOSTIC_DIR") or os.environ.get("TAMP_FM_DIAGNOSTICS_DIR")
    if not diag_dir_env:
        return
    try:
        diag_dir = Path(diag_dir_env)
        diag_dir.mkdir(parents=True, exist_ok=True)
        existing = list(diag_dir.glob("fm_call_*.json"))
        call_idx = len(existing) + 1
        diag_path = diag_dir / f"fm_call_{call_idx:03d}.json"

        content_str = str(content) if content is not None else ""
        content_sha = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        finish_reason = None
        usage = None
        model = None
        if isinstance(response, dict):
            model = response.get("model")
            usage = response.get("usage")
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                finish_reason = choices[0].get("finish_reason")

        diag_data = {
            "model": model,
            "call_kind": call_kind,
            "sanitized_request": sanitized_request,
            "finish_reason": finish_reason,
            "usage": usage,
            "content_length_chars": len(content_str),
            "content_sha256": content_sha,
            "content": content,
            "json_parse_success": json_parse_success,
            "parse_error": parse_error,
        }
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_data, f, indent=2)
    except Exception:
        pass


def _extract_json_content(
    response: Mapping[str, Any],
    call_kind: str = "completion",
    *,
    sanitized_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(response, dict) and "choices" not in response:
        _save_fm_diagnostic(None, response, call_kind, True, None, sanitized_request=sanitized_request)
        return dict(response)
    try:
        choice = response["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        _save_fm_diagnostic(response, None, call_kind, False, parse_error=str(error), sanitized_request=sanitized_request)
        raise TransportOrStructuredOutputError(
            "Completion response has no choices[0].message.content"
        ) from error
    if isinstance(content, dict):
        _save_fm_diagnostic(response, content, call_kind, True, None, sanitized_request=sanitized_request)
        return content
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        suffix = " after reasoning" if reasoning else ""
        msg = f"Completion has no final JSON content{suffix}"
        _save_fm_diagnostic(response, content, call_kind, False, parse_error=msg, sanitized_request=sanitized_request)
        raise TransportOrStructuredOutputError(msg)
    text = content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fenced:
        text = fenced.group(1)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        _save_fm_diagnostic(response, content, call_kind, False, parse_error=str(error), sanitized_request=sanitized_request)
        raise TransportOrStructuredOutputError(f"Completion content is not valid JSON: {error}") from error
    if not isinstance(decoded, dict):
        msg = "Completion JSON must be an object"
        _save_fm_diagnostic(response, content, call_kind, False, parse_error=msg, sanitized_request=sanitized_request)
        raise TransportOrStructuredOutputError(msg)
    _save_fm_diagnostic(response, content, call_kind, True, None, sanitized_request=sanitized_request)
    return decoded


def validate_requirement_response(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the generic Living Room and Workshop specification schema."""
    if not isinstance(document, Mapping):
        raise FMResponseValidationError("Requirement response must be a JSON object")

    allowed_top = {
        "status", "task_summary", "functional_roles",
        "functional_relations", "interaction_groups", "inspectable_regions", "inspection_order",
        "unsupported_reason",
    }
    if not set(document).issubset(allowed_top):
        unexpected = set(document) - allowed_top
        raise FMResponseValidationError(f"Unexpected top-level fields in requirement response: {sorted(unexpected)}")

    for req_field in {
        "status", "task_summary", "functional_roles",
        "functional_relations", "interaction_groups", "inspectable_regions", "inspection_order",
        "unsupported_reason",
    }:
        if req_field not in document:
            raise FMResponseValidationError(f"Requirement response missing required top-level field {req_field!r}")

    status = document.get("status")
    if status not in {"SUPPORTED", "UNSUPPORTED"}:
        raise FMResponseValidationError("status must be 'SUPPORTED' or 'UNSUPPORTED'")

    summary = document.get("task_summary", "")
    if not isinstance(summary, str) or not summary.strip():
        raise FMResponseValidationError("task_summary must be a non-empty string")

    unsupported_reason = document.get("unsupported_reason", "")
    if not isinstance(unsupported_reason, str):
        raise FMResponseValidationError("unsupported_reason must be a string")

    roles = document.get("functional_roles")

    if status == "UNSUPPORTED":
        if not unsupported_reason.strip():
            raise FMResponseValidationError("UNSUPPORTED status requires a non-empty unsupported_reason")
        if roles != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty functional_roles")
        if document.get("functional_relations") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty functional_relations")
        if document.get("interaction_groups") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty interaction_groups")
        if document.get("inspectable_regions") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty inspectable_regions")
        if document.get("inspection_order") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty inspection_order")
        return {
            "status": "UNSUPPORTED",
            "task_summary": summary.strip(),
            "functional_roles": [],
            "functional_relations": [],
            "interaction_groups": [],
            "inspectable_regions": [],
            "inspection_order": [],
            "unsupported_reason": unsupported_reason.strip(),
        }

    if unsupported_reason.strip():
        raise FMResponseValidationError("SUPPORTED status requires an empty unsupported_reason")

    if not isinstance(roles, list) or not roles:
        raise FMResponseValidationError("SUPPORTED status requires a non-empty functional_roles array")
    if len(roles) > 12:
        raise FMResponseValidationError("functional_roles must contain at most 12 items")

    normalized_roles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, role in enumerate(roles):
        if not isinstance(role, dict):
            raise FMResponseValidationError(f"functional_roles[{index}] must be a dict")
        role_allowed = {
            "id", "entity_kind", "function", "description", "required_count",
            "binding_policy", "candidate_categories",
            "visible_candidates", "required_properties",
        }
        role_required = {
            "id", "entity_kind", "function", "required_count",
            "binding_policy", "candidate_categories",
            "visible_candidates", "required_properties",
        }
        if not role_required.issubset(set(role)):
            raise FMResponseValidationError(f"functional_roles[{index}] missing required fields: {sorted(role_required - set(role))}")
        if not set(role).issubset(role_allowed):
            raise FMResponseValidationError(f"functional_roles[{index}] has invalid fields: {sorted(set(role) - role_allowed)}")
        identifier = role.get("id")
        if not _short_string(identifier, 80) or not re.fullmatch(r"[a-zA-Z0-9_]+", str(identifier)):
            raise FMResponseValidationError(f"functional_roles[{index}].id must be a valid identifier")
        if identifier in seen_ids:
            raise FMResponseValidationError(f"Duplicate role ID {identifier!r} in functional_roles")
        seen_ids.add(identifier)

        entity_kind = role.get("entity_kind")
        if entity_kind not in {"OBJECT", "REGION", "FIXED_TARGET"}:
            raise FMResponseValidationError(f"functional_roles[{index}].entity_kind must be OBJECT, REGION, or FIXED_TARGET")

        function_text = role.get("function")
        if not _short_string(function_text, 600) or not str(function_text).strip():
            raise FMResponseValidationError(f"functional_roles[{index}].function must be a non-empty string")

        desc_text = role.get("description", "")
        if not isinstance(desc_text, str):
            raise FMResponseValidationError(f"functional_roles[{index}].description must be a string")

        required_count = role.get("required_count")
        if isinstance(required_count, bool) or not isinstance(required_count, int) or required_count < 1 or required_count > 20:
            raise FMResponseValidationError(f"functional_roles[{index}].required_count must be an integer from 1 to 20")

        binding_policy = role.get("binding_policy")
        if binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
            raise FMResponseValidationError(f"functional_roles[{index}].binding_policy must be DISTINCT, REUSABLE, or SHARED")

        cand_cats = role.get("candidate_categories")
        if not isinstance(cand_cats, list):
            raise FMResponseValidationError(f"functional_roles[{index}].candidate_categories must be a list")
        for cat in cand_cats:
            if not isinstance(cat, str) or not cat.strip():
                raise FMResponseValidationError(f"functional_roles[{index}].candidate_categories items must be non-empty strings")

        candidates = role.get("visible_candidates")
        if not isinstance(candidates, list) or len(candidates) > 16:
            raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates must be a list of at most 16 items")

        cleaned_candidates: list[dict[str, str]] = []
        seen_cand: set[tuple[str, str]] = set()
        for c_idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}] must be a dict")
            cand_allowed = {"label", "visual_description", "suitability_reason"}
            cand_required = {"label", "visual_description"}
            if not cand_required.issubset(set(candidate)):
                missing = sorted(cand_required - set(candidate))
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}] missing required fields: {missing}")
            if not set(candidate).issubset(cand_allowed):
                unexpected = sorted(set(candidate) - cand_allowed)
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}] has invalid fields: {unexpected}")
            label = candidate.get("label", "")
            v_desc = candidate.get("visual_description", "")
            s_reason = candidate.get("suitability_reason", "")
            if not _short_string(label, 400) or not str(label).strip():
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}].label is invalid")
            if not _short_string(v_desc, 400):
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}].visual_description is invalid")
            if s_reason and not isinstance(s_reason, str):
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}].suitability_reason must be a string")
            key = (str(label).strip().casefold(), str(v_desc).strip().casefold())
            if key in seen_cand:
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates contains duplicates")
            seen_cand.add(key)
            cleaned_candidates.append({
                "label": str(label).strip(),
                "visual_description": str(v_desc).strip(),
                "suitability_reason": str(s_reason).strip(),
            })

        properties = role.get("required_properties", [])
        if not isinstance(properties, list) or len(properties) > 16:
            raise FMResponseValidationError(f"functional_roles[{index}].required_properties must be a list of at most 16 items")
        cleaned_props = []
        for prop in properties:
            if not isinstance(prop, str) or not prop.strip() or len(prop) > 160:
                raise FMResponseValidationError(f"functional_roles[{index}].required_properties item is invalid")
            cleaned_props.append(prop.strip())
        lowered_props = [p.casefold() for p in cleaned_props]
        if len(lowered_props) != len(set(lowered_props)):
            raise FMResponseValidationError(f"functional_roles[{index}].required_properties contains duplicates")

        normalized_roles.append({
            "id": identifier.strip(),
            "entity_kind": entity_kind,
            "function": str(function_text).strip(),
            "description": desc_text.strip(),
            "required_count": required_count,
            "binding_policy": binding_policy,
            "candidate_categories": [str(c).strip() for c in cand_cats],
            "visible_candidates": cleaned_candidates,
            "required_properties": cleaned_props,
        })

    declared_region_ids: set[str] = set()
    raw_regions = document.get("inspectable_regions", [])
    if not isinstance(raw_regions, list) or len(raw_regions) > 12:
        raise FMResponseValidationError("inspectable_regions must be a list of at most 12 items")
    cleaned_regions = []
    for r_idx, reg in enumerate(raw_regions):
        if not isinstance(reg, dict) or set(reg) != {"id", "label", "visual_description", "reason"}:
            raise FMResponseValidationError(f"inspectable_regions[{r_idx}] must contain exact fields id, label, visual_description, reason")
        reg_id = reg.get("id")
        if not _short_string(reg_id, 80) or not str(reg_id).strip():
            raise FMResponseValidationError(f"inspectable_regions[{r_idx}].id must be a non-empty string")
        if reg_id in declared_region_ids:
            raise FMResponseValidationError(f"Duplicate inspectable_region id {reg_id!r}")
        declared_region_ids.add(reg_id)
        if not isinstance(reg.get("label"), str) or not isinstance(reg.get("visual_description"), str) or not isinstance(reg.get("reason"), str):
            raise FMResponseValidationError(f"inspectable_regions[{r_idx}] fields must be strings")
        cleaned_regions.append({
            "id": str(reg_id).strip(),
            "label": str(reg.get("label", "")).strip(),
            "visual_description": str(reg.get("visual_description", "")).strip(),
            "reason": str(reg.get("reason", "")).strip(),
        })

    raw_order = document.get("inspection_order", [])
    if not isinstance(raw_order, list):
        raise FMResponseValidationError("inspection_order must be a list")
    seen_order: set[str] = set()
    cleaned_order = []
    for o_idx, item in enumerate(raw_order):
        if not isinstance(item, str):
            raise FMResponseValidationError(f"inspection_order[{o_idx}] must be a string ID")
        if item not in declared_region_ids:
            raise FMResponseValidationError(f"inspection_order[{o_idx}] references undeclared region ID {item!r}")
        if item in seen_order:
            raise FMResponseValidationError(f"Duplicate region ID {item!r} in inspection_order")
        seen_order.add(item)
        cleaned_order.append(item.strip())
    if cleaned_order and declared_region_ids and set(cleaned_order) != declared_region_ids:
        raise FMResponseValidationError("inspection_order must be a complete permutation of declared inspectable_regions")

    raw_relations = document.get("functional_relations", [])
    if not isinstance(raw_relations, list) or len(raw_relations) > 24:
        raise FMResponseValidationError("functional_relations must be a list of at most 24 items")
    cleaned_relations = []
    for rel_idx, rel in enumerate(raw_relations):
        if not isinstance(rel, dict) or set(rel) != {"subject_role", "relation", "object_role"}:
            raise FMResponseValidationError(f"functional_relations[{rel_idx}] has invalid fields")
        s = rel.get("subject_role")
        r = rel.get("relation")
        o = rel.get("object_role")
        if s not in seen_ids or o not in seen_ids:
            raise FMResponseValidationError(f"functional_relations[{rel_idx}] references undeclared role ({s!r}, {o!r})")
        if not _short_string(r, 400) or not str(r).strip():
            raise FMResponseValidationError(f"functional_relations[{rel_idx}].relation must be a non-empty string")
        cleaned_relations.append({
            "subject_role": str(s).strip(),
            "relation": str(r).strip(),
            "object_role": str(o).strip(),
        })

    raw_groups = document.get("interaction_groups", [])
    if not isinstance(raw_groups, list) or len(raw_groups) > 8:
        raise FMResponseValidationError("interaction_groups must be a list of at most 8 items")
    cleaned_groups = []
    seen_group_ids: set[str] = set()
    for g_idx, grp in enumerate(raw_groups):
        if not isinstance(grp, dict):
            raise FMResponseValidationError(f"interaction_groups[{g_idx}] must be a dict")
        grp_allowed = {
            "id", "function", "tool_role", "target_role",
            "required_target_count", "usage_policy", "required_relations",
            "context_role", "context_relations",
        }
        grp_required = {
            "id", "function", "tool_role", "target_role",
            "required_target_count", "usage_policy", "required_relations",
        }
        if not grp_required.issubset(set(grp)):
            raise FMResponseValidationError(f"interaction_groups[{g_idx}] missing required fields: {sorted(grp_required - set(grp))}")
        if not set(grp).issubset(grp_allowed):
            raise FMResponseValidationError(f"interaction_groups[{g_idx}] has invalid fields: {sorted(set(grp) - grp_allowed)}")
        gid = grp.get("id")
        if not _short_string(gid, 80) or gid in seen_group_ids:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].id must be unique non-empty string")
        seen_group_ids.add(gid)
        t_role = grp.get("tool_role")
        tgt_role = grp.get("target_role")
        if t_role not in seen_ids or tgt_role not in seen_ids:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}] references undeclared role ({t_role!r}, {tgt_role!r})")
        if grp.get("usage_policy") not in {"SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"}:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].usage_policy must be SEQUENTIAL_REUSE_ALLOWED or DEDICATED_PER_TARGET")
        req_tgt_c = grp.get("required_target_count")
        if isinstance(req_tgt_c, bool) or not isinstance(req_tgt_c, int) or req_tgt_c < 1 or req_tgt_c > 20:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_target_count must be an integer from 1 to 20")

        raw_req_rels = grp.get("required_relations")
        if not isinstance(raw_req_rels, list) or len(raw_req_rels) < 1:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_relations must be a non-empty list")
        cleaned_req_rels = []
        for r_idx, r in enumerate(raw_req_rels):
            if not isinstance(r, str) or not r.strip():
                raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_relations[{r_idx}] must be a non-empty string")
            cleaned_req_rels.append(r.strip())
        if len(cleaned_req_rels) != len(set(cleaned_req_rels)):
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_relations contains duplicate relations")

        ctx_role = grp.get("context_role")
        raw_ctx_rels = grp.get("context_relations")
        if ctx_role is not None:
            if not isinstance(ctx_role, str) or not ctx_role.strip() or ctx_role not in seen_ids:
                raise FMResponseValidationError(f"interaction_groups[{g_idx}].context_role references undeclared role {ctx_role!r}")
            if not isinstance(raw_ctx_rels, list) or len(raw_ctx_rels) < 1:
                raise FMResponseValidationError(f"interaction_groups[{g_idx}].context_relations must be a non-empty list when context_role is provided")
            cleaned_ctx_rels = []
            for r_idx, r in enumerate(raw_ctx_rels):
                if not isinstance(r, str) or not r.strip():
                    raise FMResponseValidationError(f"interaction_groups[{g_idx}].context_relations[{r_idx}] must be a non-empty string")
                cleaned_ctx_rels.append(r.strip())
            if len(cleaned_ctx_rels) != len(set(cleaned_ctx_rels)):
                raise FMResponseValidationError(f"interaction_groups[{g_idx}].context_relations contains duplicate relations")
        else:
            if raw_ctx_rels is not None:
                if not isinstance(raw_ctx_rels, list) or len(raw_ctx_rels) > 0:
                    raise FMResponseValidationError(f"interaction_groups[{g_idx}].context_relations provided without context_role")
            cleaned_ctx_rels = []

        cleaned_groups.append({
            "id": str(gid).strip(),
            "function": str(grp.get("function", "")).strip(),
            "tool_role": str(t_role).strip(),
            "target_role": str(tgt_role).strip(),
            "required_target_count": req_tgt_c,
            "usage_policy": str(grp.get("usage_policy")),
            "required_relations": cleaned_req_rels,
            "context_role": str(ctx_role).strip() if ctx_role else None,
            "context_relations": cleaned_ctx_rels,
        })

    return {
        "status": status,
        "task_summary": summary.strip(),
        "functional_roles": normalized_roles,
        "functional_relations": cleaned_relations,
        "interaction_groups": cleaned_groups,
        "inspectable_regions": cleaned_regions,
        "inspection_order": cleaned_order,
        "unsupported_reason": "",
    }


def validate_kitchen_functional_specification(document: dict[str, Any]) -> dict[str, Any]:
    """Deterministically validate raw Kitchen functional specification against strict schema."""
    if not isinstance(document, dict):
        raise FMResponseValidationError("Kitchen functional graph response must be a JSON object")

    required_top = {
        "status", "task_summary", "functional_roles", "functional_relations",
        "interaction_groups", "cross_group_reuse_allowed", "inspectable_regions",
        "inspection_order", "unsupported_reason",
    }
    if not required_top.issubset(set(document)):
        missing = required_top - set(document)
        raise FMResponseValidationError(f"Missing required top-level fields in kitchen spec: {sorted(missing)}")
    if not set(document).issubset(required_top):
        unexpected = set(document) - required_top
        raise FMResponseValidationError(f"Unexpected top-level fields in kitchen spec: {sorted(unexpected)}")

    status = document.get("status")
    if status not in {"SUPPORTED", "UNSUPPORTED"}:
        raise FMResponseValidationError("status must be 'SUPPORTED' or 'UNSUPPORTED'")

    summary = document.get("task_summary", "")
    if not isinstance(summary, str) or not summary.strip():
        raise FMResponseValidationError("task_summary must be a non-empty string")

    unsupported_reason = document.get("unsupported_reason", "")
    if not isinstance(unsupported_reason, str):
        raise FMResponseValidationError("unsupported_reason must be a string")

    cross_group_reuse = document.get("cross_group_reuse_allowed")
    if not isinstance(cross_group_reuse, bool):
        raise FMResponseValidationError("cross_group_reuse_allowed must be a boolean")

    roles = document.get("functional_roles")

    if status == "UNSUPPORTED":
        if not unsupported_reason.strip():
            raise FMResponseValidationError("UNSUPPORTED status requires a non-empty unsupported_reason")
        if roles != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty functional_roles")
        if document.get("functional_relations") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty functional_relations")
        if document.get("interaction_groups") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty interaction_groups")
        if document.get("inspectable_regions") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty inspectable_regions")
        if document.get("inspection_order") != []:
            raise FMResponseValidationError("UNSUPPORTED status must have empty inspection_order")
        return deepcopy(document)

    if unsupported_reason.strip():
        raise FMResponseValidationError("SUPPORTED status requires an empty unsupported_reason")

    if not isinstance(roles, list) or not roles:
        raise FMResponseValidationError("SUPPORTED status requires a non-empty functional_roles array")

    role_ids: set[str] = set()
    role_counts: dict[str, int] = {}
    for index, role in enumerate(roles):
        if not isinstance(role, dict):
            raise FMResponseValidationError(f"functional_roles[{index}] must be a dict")
        role_req_fields = {
            "id", "entity_kind", "function", "required_count",
            "binding_policy", "candidate_categories", "visible_candidates", "required_properties",
        }
        role_allowed_fields = {
            "id", "entity_kind", "function", "description", "required_count",
            "binding_policy", "candidate_categories", "visible_candidates", "required_properties",
        }
        if not role_req_fields.issubset(set(role)):
            missing = role_req_fields - set(role)
            raise FMResponseValidationError(
                f"functional_roles[{index}] missing required fields: {sorted(missing)}"
            )
        if not set(role).issubset(role_allowed_fields):
            unexpected = set(role) - role_allowed_fields
            raise FMResponseValidationError(
                f"functional_roles[{index}] has invalid fields: {sorted(unexpected)}"
            )
        r_id = role.get("id")
        if not isinstance(r_id, str) or not re.fullmatch(r"[a-zA-Z0-9_]+", r_id):
            raise FMResponseValidationError(f"functional_roles[{index}].id must be a valid identifier")
        if r_id in role_ids:
            raise FMResponseValidationError(f"Duplicate role ID {r_id!r} in functional_roles")
        role_ids.add(r_id)

        if role.get("entity_kind") not in {"OBJECT", "REGION", "FIXED_TARGET"}:
            raise FMResponseValidationError(
                f"functional_roles[{index}].entity_kind must be OBJECT, REGION, or FIXED_TARGET"
            )
        if not isinstance(role.get("function"), str) or not role.get("function").strip():
            raise FMResponseValidationError(f"functional_roles[{index}].function must be a non-empty string")
        req_count = role.get("required_count")
        if isinstance(req_count, bool) or not isinstance(req_count, int) or req_count < 1:
            raise FMResponseValidationError(f"functional_roles[{index}].required_count must be an integer >= 1")
        role_counts[r_id] = req_count

        if role.get("binding_policy") not in {"DISTINCT", "REUSABLE", "SHARED"}:
            raise FMResponseValidationError(
                f"functional_roles[{index}].binding_policy must be DISTINCT, REUSABLE, or SHARED"
            )
        cand_cats = role.get("candidate_categories")
        if not isinstance(cand_cats, list) or len(cand_cats) < 1 or len(cand_cats) > 12:
            raise FMResponseValidationError(f"functional_roles[{index}].candidate_categories must contain 1 to 12 items")
        for c in cand_cats:
            if not isinstance(c, str) or not c.strip():
                raise FMResponseValidationError(f"functional_roles[{index}].candidate_categories items must be non-empty strings")
        cand_objs = role.get("visible_candidates")
        if not isinstance(cand_objs, list) or len(cand_objs) > 16:
            raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates must be a list of at most 16 items")
        for c_idx, candidate in enumerate(cand_objs):
            if not isinstance(candidate, dict) or set(candidate) != {"label", "visual_description"}:
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}] has invalid fields")
            label = candidate.get("label", "")
            v_desc = candidate.get("visual_description", "")
            if not _short_string(label, 400) or not str(label).strip():
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}].label is invalid")
            if not _short_string(v_desc, 400):
                raise FMResponseValidationError(f"functional_roles[{index}].visible_candidates[{c_idx}].visual_description is invalid")
        req_props = role.get("required_properties")
        if not isinstance(req_props, list) or len(req_props) > 12:
            raise FMResponseValidationError(f"functional_roles[{index}].required_properties must be a list of at most 12 items")
        for p in req_props:
            if not isinstance(p, str) or not p.strip():
                raise FMResponseValidationError(f"functional_roles[{index}].required_properties items must be non-empty strings")

    declared_region_ids: set[str] = set()
    raw_regions = document.get("inspectable_regions", [])
    if not isinstance(raw_regions, list) or len(raw_regions) > 12:
        raise FMResponseValidationError("inspectable_regions must be a list of at most 12 items")
    for r_idx, reg in enumerate(raw_regions):
        if not isinstance(reg, dict) or set(reg) != {"id", "label", "visual_description", "reason"}:
            raise FMResponseValidationError(f"inspectable_regions[{r_idx}] must have exact fields id, label, visual_description, reason")
        reg_id = reg.get("id")
        if not isinstance(reg_id, str) or not reg_id.strip():
            raise FMResponseValidationError(f"inspectable_regions[{r_idx}].id must be a non-empty string")
        if reg_id in declared_region_ids:
            raise FMResponseValidationError(f"Duplicate inspectable_region id {reg_id!r}")
        declared_region_ids.add(reg_id)

    raw_order = document.get("inspection_order", [])
    if not isinstance(raw_order, list):
        raise FMResponseValidationError("inspection_order must be a list")
    seen_order: set[str] = set()
    cleaned_order: list[str] = []
    for o_idx, item in enumerate(raw_order):
        if not isinstance(item, str):
            raise FMResponseValidationError(f"inspection_order[{o_idx}] must be a string ID")
        if item not in declared_region_ids:
            raise FMResponseValidationError(f"inspection_order[{o_idx}] references undeclared region ID {item!r}")
        if item in seen_order:
            raise FMResponseValidationError(f"Duplicate region ID {item!r} in inspection_order")
        seen_order.add(item)
        cleaned_order.append(item.strip())
    if raw_order and declared_region_ids and set(raw_order) != declared_region_ids:
        raise FMResponseValidationError("inspection_order must be a complete permutation of declared inspectable_regions")

    raw_relations = document.get("functional_relations", [])
    if not isinstance(raw_relations, list) or len(raw_relations) > 24:
        raise FMResponseValidationError("functional_relations must be a list of at most 24 items")
    for rel_idx, rel in enumerate(raw_relations):
        if not isinstance(rel, dict) or set(rel) != {"subject_role", "relation", "object_role"}:
            raise FMResponseValidationError(f"functional_relations[{rel_idx}] has invalid fields")
        s = rel.get("subject_role")
        o = rel.get("object_role")
        if s not in role_ids or o not in role_ids:
            raise FMResponseValidationError(f"functional_relations[{rel_idx}] references undeclared role ({s!r}, {o!r})")

    raw_groups = document.get("interaction_groups", [])
    if not isinstance(raw_groups, list) or len(raw_groups) > 8:
        raise FMResponseValidationError("interaction_groups must be a list of at most 8 items")
    group_ids: set[str] = set()
    for g_idx, grp in enumerate(raw_groups):
        grp_fields = {
            "id", "function", "tool_role", "target_role",
            "required_target_count", "usage_policy", "required_relations",
        }
        if not isinstance(grp, dict) or set(grp) != grp_fields:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}] has invalid fields")
        gid = grp.get("id")
        if not isinstance(gid, str) or not gid.strip() or gid in group_ids:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].id must be unique non-empty string")
        group_ids.add(gid)
        t_role = grp.get("tool_role")
        tgt_role = grp.get("target_role")
        if t_role not in role_ids or tgt_role not in role_ids:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}] references undeclared role ({t_role!r}, {tgt_role!r})")
        if grp.get("usage_policy") not in {"SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"}:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].usage_policy must be SEQUENTIAL_REUSE_ALLOWED or DEDICATED_PER_TARGET")
        raw_req_rels = grp.get("required_relations")
        if not isinstance(raw_req_rels, list) or len(raw_req_rels) < 1 or len(raw_req_rels) > 12:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_relations must contain 1 to 12 items")
        for r_i, r in enumerate(raw_req_rels):
            if not isinstance(r, str) or not r.strip():
                raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_relations[{r_i}] must be a non-empty string")
        req_tgt_c = grp.get("required_target_count")
        if isinstance(req_tgt_c, bool) or not isinstance(req_tgt_c, int) or req_tgt_c < 1 or req_tgt_c > 20:
            raise FMResponseValidationError(f"interaction_groups[{g_idx}].required_target_count must be an integer from 1 to 20")
        if role_counts.get(tgt_role) != req_tgt_c:
            raise FMResponseValidationError(
                f"interaction_groups[{g_idx}] target role {tgt_role} has required_count {role_counts.get(tgt_role)}, but group requires {req_tgt_c}"
            )

    return deepcopy(document)


class FMAdapter:
    """Generate one structured, planning-free requirement decomposition."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        transport: CompletionTransport | None = None,
    ) -> None:
        self.base_url = base_url or _env_first("TAMP_FM_BASE_URL", "FM_BASE_URL")
        self.model = model or _env_first("TAMP_FM_MODEL", "FM_MODEL")
        self.api_key = api_key if api_key is not None else _env_first(
            "TAMP_FM_API_KEY", "FM_API_KEY"
        )
        self.timeout_seconds = timeout_seconds or float(
            _env_first("TAMP_FM_TIMEOUT_SECONDS", "FM_TIMEOUT_SECONDS", default="600")
        )
        self.max_tokens = max_tokens or int(
            _env_first("TAMP_FM_MAX_TOKENS", "FM_MAX_TOKENS", default="8192")
        )
        self.metrics = FMCallMetrics()
        self._transport = transport
        self.last_observation_images: list[dict[str, Any]] = []
        self.last_raw_requirement_response: dict[str, Any] | None = None
        self.last_raw_inspection_response: dict[str, Any] | None = None
        self.last_raw_kitchen_graph_response: dict[str, Any] | None = None

    def generate_kitchen_functional_graph(
        self,
        task_instruction: str,
        search_region_descriptors: dict[str, str] | None = None,
        *,
        observation_images: Sequence[str | Path],
    ) -> dict[str, Any]:
        """Produce the complete Kitchen natural functional requirement specification."""
        del search_region_descriptors
        image_blocks, self.last_observation_images = _encode_observation_images(
            observation_images
        )
        system_prompt = SYSTEM_PROMPT
        prompt = {
            "task_instruction": task_instruction.strip(),
            "request": (
                "In one response, infer the functional roles, qualitative properties, "
                "qualitative relations between roles, interaction groups with reuse policies, "
                "candidate semantic categories, inspectable closed storage regions visible in the initial images, "
                "and a complete inspection ranking over those proposed regions if evidence is incomplete. "
                "Decide them yourself from the goal and initial RGB views. "
                "Do not output physical instance assignments or an action sequence."
            ),
        }
        sanitized_req = {
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "schema_name": "kitchen_functional_requirement_graph",
            "num_images": len(self.last_observation_images),
            "image_metadata": self.last_observation_images,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": [
                    {"type": "text", "text": json.dumps(prompt, separators=(",", ":"))},
                    *image_blocks,
                ]},
            ],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "kitchen_functional_requirement_graph",
                    "strict": True,
                    "schema": KITCHEN_FUNCTIONAL_GRAPH_SCHEMA,
                },
            },
        }
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        document = _extract_json_content(
            self._completion_transport().complete(payload),
            call_kind="kitchen_functional_graph",
            sanitized_request=sanitized_req,
        )
        self.last_raw_kitchen_graph_response = deepcopy(document)
        validated = validate_kitchen_functional_specification(document)
        self.last_validated_kitchen_graph_response = deepcopy(validated)
        return validated

    def _completion_transport(self) -> CompletionTransport:
        if self._transport is not None:
            return self._transport
        if not self.base_url or not self.model:
            raise FMBackendNotConfiguredError(
                "Live FM requirements need TAMP_FM_BASE_URL and TAMP_FM_MODEL. "
                "Use an SSH tunnel to the Qwen OpenAI-compatible endpoint or "
                "run with requirements_source: static."
            )
        return OpenAICompletionTransport(
            self.base_url, self.api_key, self.timeout_seconds
        )

    def generate_task_requirements(
        self,
        task_instruction: str,
        *,
        observation_images: Sequence[str | Path],
    ) -> dict[str, Any]:
        """Infer roles, properties, and visible candidates in one Qwen call."""
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError("task_instruction must be a non-empty string")
        if len(task_instruction) > 4000:
            raise ValueError("task_instruction exceeds 4000 characters")
        image_blocks, self.last_observation_images = _encode_observation_images(
            observation_images
        )
        transport = self._completion_transport()
        user_prompt_data = {
            "task_instruction": task_instruction.strip(),
            "request": (
                "Using the task goal and initial-observation images, infer the "
                "functional roles, describe the qualitative properties each role "
                "requires, and rank any visually plausible candidate objects or "
                "regions for each role. Decide all role and property content yourself."
            ),
        }
        user_text = json.dumps(
            user_prompt_data,
            separators=(",", ":"),
        )
        sanitized_req = {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt_data,
            "schema_name": "functional_specification",
            "num_images": len(self.last_observation_images),
            "image_metadata": self.last_observation_images,
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *image_blocks,
                    ],
                },
            ],
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "functional_specification",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        }
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        response = transport.complete(payload)
        raw_document = _extract_json_content(
            response, call_kind="task_requirements", sanitized_request=sanitized_req
        )
        self.last_raw_requirement_response = deepcopy(raw_document)
        return validate_requirement_response(raw_document)

    def generate_inspection_priors(
        self,
        task_instruction: str,
        search_region_descriptors: dict[str, str] | None = None,
        *,
        observation_images: Sequence[str | Path],
    ) -> dict[str, Any]:
        """Ask Qwen for visually proposed inspectable regions and search order."""
        del search_region_descriptors
        image_blocks, self.last_observation_images = _encode_observation_images(observation_images)
        system_prompt = (
            "You choose an evidence-gathering order for the task. Return only the requested JSON. "
            "Visually identify any closed storage regions (such as drawers or cabinets) visible in the "
            "initial scene images, and rank them in the order they should be inspected if evidence is incomplete. "
            "Never infer hidden contents, ground-truth assignments, feasibility labels, or actions beyond "
            "opening/inspecting the regions you identified."
        )
        prompt = {
            "task_instruction": task_instruction.strip(),
            "request": (
                "Decide from the initial images whether all functional requirements "
                "are already visibly satisfiable. Visually identify and propose any closed "
                "storage regions (such as drawers or cabinets) visible in the scene, and rank them "
                "in the order they should be inspected if evidence is incomplete. Do not predict or invent stored contents."
            ),
        }
        sanitized_req = {
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "schema_name": "inspection_policy",
            "num_images": len(self.last_observation_images),
            "image_metadata": self.last_observation_images,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": [
                    {"type": "text", "text": json.dumps(prompt, separators=(",", ":"))},
                    *image_blocks,
                ]},
            ],
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": min(self.max_tokens, 2048),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "inspection_policy",
                    "strict": True,
                    "schema": INSPECTION_POLICY_SCHEMA,
                },
            },
        }
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        document = _extract_json_content(
            self._completion_transport().complete(payload),
            call_kind="inspection_priors",
            sanitized_request=sanitized_req,
        )
        self.last_raw_inspection_response = deepcopy(document)
        expected = {
            "initial_requirements_satisfied", "decision_reason", "inspection_order",
        }
        if not isinstance(document, dict) or not expected.issubset(set(document)):
            raise FMResponseValidationError("Inspection policy has invalid fields")
        if not isinstance(document["initial_requirements_satisfied"], bool):
            raise FMResponseValidationError("initial_requirements_satisfied must be boolean")
        if not _short_string(document["decision_reason"], 1000):
            raise FMResponseValidationError("decision_reason must be non-empty")
        inspectable = list(document.get("inspectable_regions", []))
        raw_order = list(document.get("inspection_order", []))
        if not inspectable and raw_order and isinstance(raw_order[0], dict):
            inspectable = [{"id": item.get("region_id", ""), "label": item.get("region_id", ""), "visual_description": item.get("reason", "")} for item in raw_order]
        return {
            "initial_requirements_satisfied": document["initial_requirements_satisfied"],
            "decision_reason": str(document["decision_reason"]).strip(),
            "inspectable_regions": inspectable,
            "inspection_order": raw_order,
        }
