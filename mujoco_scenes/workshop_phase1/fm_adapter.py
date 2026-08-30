"""One-shot Qwen requirement decomposition over an OpenAI-compatible server.

This module intentionally stops before observation search, grounding, planning,
or execution. The integrated Kitchen entry point asks the model to emit exact
implemented predicate identifiers; downstream code validates those identifiers
without translating natural-language aliases into task requirements.
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


class FMBackendNotConfiguredError(RuntimeError):
    """Raised when live requirement generation has no configured endpoint."""


class FMTransportError(RuntimeError):
    """Raised when the inference server cannot return a usable completion."""


class FMResponseValidationError(RuntimeError):
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


SYSTEM_PROMPT = """You are a vision-language functional-requirement planner.

Return only the requested JSON object. Do not produce an action sequence.

Rules:
- Infer the complete set of physical or spatial functional roles from the task
  instruction yourself. The user will not supply expected roles, functions,
  object categories, or properties.
- Use the initial-observation images as visual evidence. Do not assume that an
  object is present merely because it would normally be useful for the task.
- Create one requirement for each independently satisfiable functional role
  needed by the task. Give it your own short, stable snake_case id.
- Set `entity_kind` to OBJECT for a manipulable physical item and REGION for a
  usable support surface, placement area, or spatial destination.
- `functional_requirements` describes role types, never individual assignment
  slots. Functionally identical needs MUST be one record with `required_count`
  equal to the minimum number of distinct simultaneous candidates. A response
  containing duplicate function records distinguished only by left/right,
  person identity, target identity, or location is invalid. List all visible
  candidates for those slots in the one consolidated record.
- Count simultaneous distinct assignments, not repeated operations. A tool
  that can be reused sequentially can have required_count 1 even when it acts
  on several targets.
- Locating, grasping, aligning, inserting, rotating, and placing are actions,
  not replaceable functional roles. Never create requirements for them.
- Include a support or destination role when its functional suitability must be
  evaluated among observed objects or regions. Do not create a role for a truly
  immutable target that requires no selection or suitability decision.
- Express `function` and `required_properties` in concise natural language. A
  downstream ontology mapper will normalize them; do not invent metric values.
- For each role, independently rank zero or more `candidate_objects` from the
  initial-observation images. Each candidate label must be a concrete,
  detector-searchable noun phrase. Its visual description must distinguish the
  observed item or region without inventing a simulator identifier.
- An empty candidate list means that no plausible candidate is visible. It does
  not mean the functional role is unnecessary. Use an empty list only after
  examining every supplied view; include fixed objects named by the goal when
  they are visibly plausible satisfiers of an inferred role.
- A required property states a qualitative capability, fit, reach, interface,
  or compatibility condition. Never emit centimetres, thresholds, coordinates,
  measurements, numerical values, simulator identifiers, object IDs, or
  variant names.
- Include every qualitative property whose failure would disqualify a
  candidate. Consider capacity, containment, shape, reach, fit, interface,
  stability, support, proximity, and shared accessibility when relevant, but
  decide which of them actually apply to each inferred role.
- A candidate is only visually plausible. Do not claim that it is selected,
  graspable, geometrically valid, or physically compatible; those are separate
  downstream checks.
- The task is supported when its physical roles can be described, even though
  no suitable candidate is visible or later geometry may find the scene
  infeasible.
- `SUPPORTED` requires at least one requirement and an empty unsupported_reason.
  `UNSUPPORTED` requires no requirements and a concise non-empty reason.
- Do not include chain-of-thought or robot execution instructions.
- Before returning, check that every explicit desired outcome in the task is
  covered by at least one functional role. Do not turn a search instruction or
  storage location into its own role unless that storage entity itself must be
  selected by functional suitability.
"""


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
        "task_summary": {"type": "string"},
        "functional_requirements": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "entity_kind": {"type": "string", "enum": ["OBJECT", "REGION"]},
                    "function": {"type": "string"},
                    "description": {"type": "string"},
                    "required_count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "candidate_objects": {
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
                            "required": [
                                "label", "visual_description", "suitability_reason"
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "required_properties": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id", "entity_kind", "function", "description", "required_count",
                    "candidate_objects",
                    "required_properties",
                ],
                "additionalProperties": False,
            },
        },
        "unsupported_reason": {"type": "string"},
    },
    "required": [
        "status", "task_summary", "functional_requirements", "unsupported_reason",
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
                    "required_properties": {
                        "type": "array", "minItems": 0, "maxItems": 12,
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "id", "entity_kind", "function", "required_count",
                    "candidate_categories", "required_properties",
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
                        "items": {"type": "string"},
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
        "initial_satisfaction_assessment": {"type": "boolean"},
        "initial_satisfaction_reason": {"type": "string"},
        "unsupported_reason": {"type": "string"},
    },
    "required": [
        "status", "task_summary", "functional_roles", "functional_relations",
        "interaction_groups", "cross_group_reuse_allowed",
        "inspectable_regions", "inspection_order",
        "initial_satisfaction_assessment", "initial_satisfaction_reason",
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
        raise FMResponseValidationError(
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
        raise FMResponseValidationError(msg)
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
        raise FMResponseValidationError(f"Completion content is not valid JSON: {error}") from error
    if not isinstance(decoded, dict):
        msg = "Completion JSON must be an object"
        _save_fm_diagnostic(response, content, call_kind, False, parse_error=msg, sanitized_request=sanitized_request)
        raise FMResponseValidationError(msg)
    _save_fm_diagnostic(response, content, call_kind, True, None, sanitized_request=sanitized_request)
    return decoded


def validate_requirement_response(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the model-facing schema before ontology normalization."""
    allowed_top = {
        "status", "task_summary", "functional_requirements", "unsupported_reason"
    }
    if set(document) != allowed_top:
        raise FMResponseValidationError(
            f"Requirement response fields must be exactly {sorted(allowed_top)}"
        )
    status = document.get("status")
    summary = document.get("task_summary")
    requirements = document.get("functional_requirements")
    reason = document.get("unsupported_reason")
    if status not in {"SUPPORTED", "UNSUPPORTED"}:
        raise FMResponseValidationError("status must be SUPPORTED or UNSUPPORTED")
    if not _short_string(summary, 1000):
        raise FMResponseValidationError("task_summary must be a short non-empty string")
    if not isinstance(requirements, list) or len(requirements) > 12:
        raise FMResponseValidationError(
            "functional_requirements must be an array of at most 12 items"
        )
    if not isinstance(reason, str) or len(reason) > 1000:
        raise FMResponseValidationError("unsupported_reason must be a string")
    if status == "SUPPORTED" and (not requirements or reason.strip()):
        raise FMResponseValidationError(
            "SUPPORTED requires requirements and an empty unsupported_reason"
        )
    if status == "UNSUPPORTED" and (requirements or not reason.strip()):
        raise FMResponseValidationError(
            "UNSUPPORTED requires no requirements and a non-empty unsupported_reason"
        )

    normalized_requirements: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    required_fields = {
        "id", "entity_kind", "function", "description", "required_count",
        "candidate_objects", "required_properties"
    }
    for index, requirement in enumerate(requirements):
        if not isinstance(requirement, dict) or set(requirement) != required_fields:
            raise FMResponseValidationError(
                f"functional_requirements[{index}] has invalid fields"
            )
        identifier = requirement.get("id")
        if not _short_string(identifier, 80) or identifier in seen_ids:
            raise FMResponseValidationError(
                f"functional_requirements[{index}].id must be short and unique"
            )
        seen_ids.add(identifier)
        if requirement.get("entity_kind") not in {"OBJECT", "REGION"}:
            raise FMResponseValidationError(
                f"functional_requirements[{index}].entity_kind must be OBJECT or REGION"
            )
        for field in ("function", "description"):
            if not _short_string(requirement.get(field), 600):
                raise FMResponseValidationError(
                    f"functional_requirements[{index}].{field} is invalid"
                )
        required_count = requirement.get("required_count")
        if (
            isinstance(required_count, bool)
            or not isinstance(required_count, int)
            or not 1 <= required_count <= 20
        ):
            raise FMResponseValidationError(
                f"functional_requirements[{index}].required_count must be an integer from 1 to 20"
            )
        properties = requirement.get("required_properties")
        if not isinstance(properties, list) or not 1 <= len(properties) <= 16:
            raise FMResponseValidationError(
                f"functional_requirements[{index}].required_properties must contain 1-16 items"
            )
        if not all(_short_string(value, 160) for value in properties):
            raise FMResponseValidationError(
                f"functional_requirements[{index}].required_properties contains invalid text"
            )
        lowered = [value.strip().casefold() for value in properties]
        if len(lowered) != len(set(lowered)):
            raise FMResponseValidationError(
                f"functional_requirements[{index}].required_properties contains duplicates"
            )
        candidates = requirement.get("candidate_objects")
        if not isinstance(candidates, list) or len(candidates) > 16:
            raise FMResponseValidationError(
                f"functional_requirements[{index}].candidate_objects must contain 0-16 items"
            )
        cleaned_candidates: list[dict[str, str]] = []
        seen_candidates: set[tuple[str, str]] = set()
        candidate_fields = {"label", "visual_description", "suitability_reason"}
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
                raise FMResponseValidationError(
                    f"functional_requirements[{index}].candidate_objects[{candidate_index}] has invalid fields"
                )
            if not all(_short_string(candidate.get(field), 400) for field in candidate_fields):
                raise FMResponseValidationError(
                    f"functional_requirements[{index}].candidate_objects[{candidate_index}] contains invalid text"
                )
            key = (
                candidate["label"].strip().casefold(),
                candidate["visual_description"].strip().casefold(),
            )
            if key in seen_candidates:
                raise FMResponseValidationError(
                    f"functional_requirements[{index}].candidate_objects contains duplicates"
                )
            seen_candidates.add(key)
            cleaned_candidates.append(
                {
                    "label": candidate["label"].strip(),
                    "visual_description": candidate["visual_description"].strip(),
                    "suitability_reason": candidate["suitability_reason"].strip(),
                }
            )
        normalized_requirements.append(
            {
                "id": identifier.strip(),
                "entity_kind": requirement["entity_kind"],
                "function": requirement["function"].strip(),
                "description": requirement["description"].strip(),
                "required_count": required_count,
                "candidate_objects": cleaned_candidates,
                "required_properties": [value.strip() for value in properties],
            }
        )
    return {
        "status": status,
        "task_summary": summary.strip(),
        "functional_requirements": normalized_requirements,
        "unsupported_reason": reason.strip(),
    }


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
        system_prompt = (
            "You are a vision-language functional-requirement graph generator. "
            "Return only the requested JSON. Infer the functional roles, qualitative "
            "properties, qualitative relations between roles, interaction groups with "
            "reuse policies, candidate categories, and visually proposed inspectable "
            "closed storage regions from the task instruction and initial multi-view RGB images. "
            "Never assume hidden contents, ground-truth identities, poses, measurements, "
            "assignments, feasibility labels, or plans."
        )
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
        return document

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
            "schema_name": "functional_requirements",
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
                    "name": "functional_requirements",
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
