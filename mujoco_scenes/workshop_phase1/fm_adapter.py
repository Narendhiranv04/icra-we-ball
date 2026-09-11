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
import time
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
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v2 import (
        SYSTEM_PROMPT_V2,
        USER_REQUEST_V2,
        LIVE_RESPONSE_SCHEMA_V2,
        is_v2_document,
        validate_v2_functional_specification,
        normalize_and_validate_v2_contract,
        convert_v2_to_canonical_document,
        compute_v2_prompt_and_schema_hash,
    )
    from mujoco_scenes.functional_tamp_pipeline.fm_schema_v3 import (
        SYSTEM_PROMPT_V3,
        USER_REQUEST_V3,
        LIVE_RESPONSE_SCHEMA_V3,
        is_v3_document,
        normalize_and_validate_v3_contract,
        compute_v3_prompt_and_schema_hash,
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


MAX_VLM_OBSERVATION_IMAGES = 3


def _select_vlm_observation_images(
    observation_images: Sequence[str | Path],
) -> list[str | Path]:
    """Select the fixed ordered view budget used only at the VLM boundary."""
    return list(observation_images[:MAX_VLM_OBSERVATION_IMAGES])


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
        last_error = None
        for attempt in range(1):
            from mujoco_scenes.functional_tamp_pipeline.telemetry import current_run
            if current_run.get() is not None:
                current_run.get().transport_attempts += 1
                current_run.get().write()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = json.load(response)
                    if not isinstance(decoded, dict):
                        raise FMTransportError("Inference server returned non-object JSON")
                    return decoded
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                last_error = FMTransportError(
                    f"Inference server returned HTTP {error.code}: {detail}"
                )
                break
            except urllib.error.URLError as error:
                last_error = FMTransportError(
                    f"Cannot reach inference server at {self.url}: {error.reason}"
                )
                time.sleep(1.0)
            except (TimeoutError, json.JSONDecodeError, ConnectionError, OSError) as error:
                last_error = FMTransportError(f"Invalid inference-server response or connection error: {error}")
                time.sleep(1.0)
        raise last_error or FMTransportError("Transport failed without explicit error")


LEGACY_SYSTEM_PROMPT_REFERENCE = """You are a vision-language functional-requirement specification generator.

Return only the requested JSON object. Do not produce an action sequence.

Structure your reasoning into two complementary aspects:

A. FUNCTIONAL REASONING
- Infer the complete set of physical participants and spatial functional roles required
  to achieve the task from task semantics, including participants that may be absent,
  occluded, or located inside closed storage.
- Visibility is evidence about current availability, not about whether a functional role is required.
- Represent physically distinct participants separately whenever they have different causal
  functions, including a source, payload, manipulated component, tool, receiving target,
  support, or contextual anchor.
- Before returning the functional graph, check that every required task transformation has all
  of its necessary physical participants and relations represented.
- Functional roles must describe capabilities rather than physical assignments.
  Do not assign physical scene instance IDs to roles; describe required functional capabilities.
- Use SHORT ATOMIC PHRASES for all functions, properties, and relations:
  - Role function: describe what the physical candidate must be capable of doing (e.g. "support payload", "contain item", "manipulate object"), rather than abstract workflow stages.
  - Required properties: list only task-critical UNARY physical or geometric characteristics of this single role used to decide candidate suitability (e.g. "planar support", "open cavity", "elongated shape"). Leave empty ([]) if no special intrinsic physical property is required beyond semantic category. Never place binary relations, part names, or non-physical adjectives here.
  - Functional relations: describe physical spatial or interface compatibility relations between roles (e.g. "compatible with", "fits inside", "placed on", "near anchor"). Both subject_role and object_role must reference declared role IDs.
  Do not write long narrative sentences. Do not use complex compound clauses.
- Robot Verifier Capabilities:
  The robot is equipped with physical and geometric verifiers that can check concepts such as:
  * Unary physical shapes: whether an object has an open/deep cavity or container volume; whether an object is elongated enough to serve as an implement; whether a surface provides a planar support.
  * Spatial & container relations: whether one object/implement can fit into or enter another object's opening; whether an implement reaches sufficiently deep into a container; whether a region can support a payload.
  * Proximity & accessibility relations: relative proximity or accessibility of support surfaces to observer or reference positions; whether a support is accessible to multiple positions.
  * Mechanical & interface compatibility: interface compatibility between tools and components; whether an implement reaches a target feature; whether components are compatible with target openings.
- When a role must be paired independently with multiple task targets or
  contextual references, represent that dependency using an interaction group
  rather than relying on an unconstrained many-to-many relation.
- Set `entity_kind` to:
  - OBJECT: a selectable/manipulable physical item.
  - REGION: a selectable support surface, placement area, or spatial destination.
  - FIXED_TARGET: a non-selectable contextual reference or fixed target feature that participates in relations.
- Set `binding_policy` to:
  - DISTINCT: separate simultaneous physical items or individual regions are required.
  - REUSABLE: one physical item may be reused sequentially across multiple targets.
  - SHARED: one physical region/entity intentionally serves multiple items/users.
- `candidate_categories`: list open-vocabulary semantic search phrases that could actually satisfy the role's stated capability (candidate realizations of the role). Visibility alone does not make a category a valid candidate; do not list broad scene distractors merely because they are visible.

B. OBSERVATION-BASED SEARCH GUIDANCE
- Use the initial multi-view RGB images to determine visible candidates, inspectable regions, and inspection ranking.
- `visible_candidates`: list visually apparent items/regions in the initial RGB views.
  This array may be empty ([]).
- `required_properties`: list UNARY-ONLY physical properties of this single role.
  Never place binary relations or compatibility statements here.
- `functional_relations`: list explicit role-to-role relations using `subject_role`, `relation`, and `object_role`.
  Both subject_role and object_role must reference declared role IDs. Use simple atomic phrases for relation.
- `interaction_groups`: list structured interaction groups with tool_role, target_role, required_target_count, usage_policy, required_relations, and optional context_role/context_relations.
- `inspectable_regions`: propose visible closed/storage regions in the initial images that could be inspected if required items are missing. Each physical storage unit must be proposed at most once; never propose duplicate regions. If all required items are visible or no closed storage search is required, leave inspectable_regions and inspection_order empty ([]).
- `inspection_order`: rank the proposed inspectable region IDs. If inspectable_regions is empty, leave inspection_order empty ([]).
- Status semantics:
  - `SUPPORTED`: task can be represented with functional roles and relations. `functional_roles` must be non-empty, `unsupported_reason` must be empty ("").
  - `UNSUPPORTED`: use only when the task itself cannot be represented by this abstraction. `functional_roles`, `functional_relations`, `interaction_groups`, `inspectable_regions`, `inspection_order` must be empty ([]), and `unsupported_reason` must be a non-empty explanation.
  - Partial observability, missing visible candidates, unmeasured continuous geometry, or needing inspection/search are NOT reasons for UNSUPPORTED.

C. DOMAIN-AGNOSTIC CAUSAL COVERAGE AUDIT
For a physical connection or assembly operation, distinguish a component that
remains in the resulting assembly from any reusable implement used to establish
that connection.
Before returning the JSON, silently decompose the user instruction into its atomic physical task requirements and verify that every requirement is represented by the functional graph:
1. For every required transformation, distinguish physically separate causal participants when applicable, including:
   - material or object being transferred/manipulated;
   - source/provider if something must come from somewhere;
   - receiving container/target/destination (use a single role with count/cardinality >= 1 rather than duplicating roles for identical destinations);
   - reusable implement/tool if an external implement causes the transformation;
   - component that remains in the final assembly;
   - contextual support or fixed target.
2. For a physical connection/assembly operation, distinguish the component that remains in the final assembly from any reusable implement used to establish that connection.
3. For material-transfer/preparation operations, represent the required source or provider for each explicitly required material unless the material is explicitly stated to already be present in its target.
4. For placement, staging, or arrangement tasks, represent BOTH the movable payload items to be placed (entity_kind: OBJECT, e.g. drinkware/refreshment items, handheld devices) AND the supporting surfaces or destinations (entity_kind: REGION, e.g. tables/stands).
5. When tasks reference contextual anchors, target persons, or reference locations:
   - For individual references (e.g. each person/seat, each user, joint frame), represent as a distinct contextual reference (entity_kind: FIXED_TARGET, e.g. viewer seating position, mounting frame, binding_policy: DISTINCT) participating in spatial relations (e.g. 'near seat', 'fastened to').
   - For collective/shared references (e.g. both people/seats, shared seating area), represent as a shared contextual reference (entity_kind: FIXED_TARGET, e.g. paired viewer seating area, binding_policy: SHARED) participating in accessibility relations (e.g. 'accessible from both seats').
6. Propagate explicit quantifiers such as 'each', 'both', and numerical counts into role cardinalities, binding policies, or interaction groups. An individual/dedicated requirement uses binding_policy: DISTINCT and interaction group usage_policy: DEDICATED_PER_TARGET.
7. Do not reuse one functional role across different causal functions unless the task semantics actually permit the same physical object to satisfy both. When a task requires distinct operations (such as preparing a beverage and serving soup), represent each operation's distinct utensils or implements (e.g., a stirring implement for the beverage vs an eating utensil for the soup) as separate functional roles. Never merge different task-specific implements into a single role.
8. Before emitting the JSON, verify:
   - every declared functional role participates in at least one functional relation or interaction group;
   - relations connect each movable item to its declared supporting destination (e.g. refreshments to individual tables, entertainment control to shared table);
   - every task clause is covered by at least one role/relation/group;
   - every transformation has its necessary participants;
   - quantities are represented;
   - any reuse/shared/distinct requirement is represented.
"""


SYSTEM_PROMPT = """You are a vision-language functional-requirement specification generator.

Return only the requested JSON object. Do not produce an action sequence.

Derive the complete task contract from the instruction before using the images:
1. Split every clause into atomic required physical transformations.
2. Declare every independently groundable physical participant, including participants not visible initially.
3. Express each required physical or causal relation as a short free-form phrase.
4. Express every required operation separately from its relations.
5. Preserve explicit counts and distinct, shared, or sequentially reusable bindings.
6. Audit that every instruction clause is represented by roles, relations, operations, counts, and bindings.

Use OBJECT for selectable physical items, REGION for selectable spatial destinations, and FIXED_TARGET for non-selectable physical anchors. Do not create roles for users, actions, states, events, quantities, abstract outcomes, or unmanipulated contents. Use one counted role for equivalent instances. Do not expose or guess physical instance identifiers.

Use `required_properties` only for task-critical unary characteristics of one role. Use `functional_relations` for required binary dependencies between declared roles. Use `interaction_groups` for required transformations, with explicit source, target, count, reuse policy, and optional anchor context. Keep all semantic phrases open-ended and domain-neutral.

Only after the task contract is complete, use the initial RGB images to populate visible candidates, inspectable regions, and inspection order. Visibility determines observation guidance, never whether a task participant exists. Missing or occluded candidates are not grounds for declaring the task unsupported.
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
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "entity_kind": {"type": "string", "enum": ["OBJECT"]},
                    "function": {"type": "string"},
                    "description": {"type": "string"},
                    "required_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "binding_policy": {"type": "string", "enum": ["DISTINCT", "REUSABLE", "SHARED"]},
                    "binding_cardinality": {
                        "type": "object",
                        "properties": {
                            "minimum_distinct_physical_objects": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "maximum_distinct_physical_objects": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "preferred": {
                                "type": "string",
                                "enum": ["minimize_distinct", "maximize_distinct", "deterministic_rank"],
                            },
                            "preference": {
                                "type": "string",
                                "enum": ["minimize_distinct", "maximize_distinct", "deterministic_rank"],
                            },
                        },
                        "required": [
                            "minimum_distinct_physical_objects",
                            "maximum_distinct_physical_objects",
                        ],
                        "additionalProperties": False,
                    },
                    "min_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "max_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    "preference": {
                        "type": "string",
                        "enum": ["minimize_distinct", "maximize_distinct", "deterministic_rank"],
                    },
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
            "type": "array", "maxItems": 2,
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
            "type": "array", "maxItems": 5,
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


def _sampling_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)).strip())


def _sampling_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)).strip())


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
    from mujoco_scenes.functional_tamp_pipeline.telemetry import current_run
    run = current_run.get()
    diag_dir_env = str(run.directory / "fm_diagnostics") if run else (os.environ.get("TAMP_FM_DIAGNOSTIC_DIR") or os.environ.get("TAMP_FM_DIAGNOSTICS_DIR"))
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
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if not isinstance(content, str) or not content.strip():
        reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
        suffix = " after reasoning" if reasoning else ""
        if finish_reason == "length":
            msg = f"Completion truncated due to token limit (finish_reason='length'){suffix}"
        else:
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
        if finish_reason == "length":
            msg = f"Completion JSON truncated due to token limit (finish_reason='length'): {error}"
        else:
            msg = f"Completion content is not valid JSON: {error}"
        _save_fm_diagnostic(response, content, call_kind, False, parse_error=msg, sanitized_request=sanitized_request)
        raise TransportOrStructuredOutputError(msg) from error
    if not isinstance(decoded, dict):
        msg = "Completion JSON must be an object"
        _save_fm_diagnostic(response, content, call_kind, False, parse_error=msg, sanitized_request=sanitized_request)
        raise TransportOrStructuredOutputError(msg)
    _save_fm_diagnostic(response, content, call_kind, True, None, sanitized_request=sanitized_request)
    return decoded


def validate_requirement_response(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the generic Living Room and Workshop specification schema (supporting V1 and V2)."""
    if not isinstance(document, Mapping):
        raise FMResponseValidationError("Requirement response must be a JSON object")

    if is_v2_document(document):
        return validate_v2_functional_specification(document)

    allowed_top = {
        "status", "task_summary", "functional_roles",
        "functional_relations", "interaction_groups", "cross_group_reuse_allowed",
        "inspectable_regions", "inspection_order",
        "unsupported_reason",
    }
    if not set(document).issubset(allowed_top):
        unexpected = set(document) - allowed_top
        raise FMResponseValidationError(f"Unexpected top-level fields in requirement response: {sorted(unexpected)}")

    for req_field in {
        "status", "task_summary", "functional_roles",
        "functional_relations", "interaction_groups", "inspectable_regions",
        "inspection_order",
        "unsupported_reason",
    }:
        if req_field not in document:
            raise FMResponseValidationError(f"Requirement response missing required top-level field {req_field!r}")

    status = document.get("status")
    if status not in {"SUPPORTED", "UNSUPPORTED"}:
        raise FMResponseValidationError("status must be 'SUPPORTED' or 'UNSUPPORTED'")
    if ("cross_group_reuse_allowed" in document
            and not isinstance(document["cross_group_reuse_allowed"], bool)):
        raise FMResponseValidationError("cross_group_reuse_allowed must be a boolean")

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
        normalized = {
            "status": "UNSUPPORTED",
            "task_summary": summary.strip(),
            "functional_roles": [],
            "functional_relations": [],
            "interaction_groups": [],
            "inspectable_regions": [],
            "inspection_order": [],
            "unsupported_reason": unsupported_reason.strip(),
        }
        if "cross_group_reuse_allowed" in document:
            normalized["cross_group_reuse_allowed"] = document["cross_group_reuse_allowed"]
        return normalized

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

    def _resolve_id(rid: Any) -> str | None:
        if not rid or not isinstance(rid, str):
            return None
        if rid in seen_ids:
            return rid
        base = re.sub(r'[-_]\d+$', '', rid)
        matches = [i for i in seen_ids if re.sub(r'[-_]\d+$', '', i) == base]
        if len(matches) == 1:
            return matches[0]
        return None

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
        s_res = _resolve_id(s)
        o_res = _resolve_id(o)
        if not s_res or not o_res:
            raise FMResponseValidationError(f"functional_relations[{rel_idx}] references undeclared role ({s!r}, {o!r})")
        if not _short_string(r, 400) or not str(r).strip():
            raise FMResponseValidationError(f"functional_relations[{rel_idx}].relation must be a non-empty string")
        cleaned_relations.append({
            "subject_role": str(s_res).strip(),
            "relation": str(r).strip(),
            "object_role": str(o_res).strip(),
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
        t_res = _resolve_id(t_role)
        tgt_res = _resolve_id(tgt_role)
        if not t_res or not tgt_res:
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
        ctx_res = None
        if ctx_role is not None:
            ctx_res = _resolve_id(ctx_role)
            if not isinstance(ctx_role, str) or not ctx_role.strip() or not ctx_res:
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
                raise FMResponseValidationError(f"interaction_groups[{g_idx}].context_relations must be null/omitted when context_role is null")

        grp_record = {
            "id": str(gid).strip(),
            "function": str(grp["function"]).strip(),
            "tool_role": str(t_res).strip(),
            "target_role": str(tgt_res).strip(),
            "required_target_count": int(req_tgt_c),
            "usage_policy": str(grp["usage_policy"]).strip(),
            "required_relations": cleaned_req_rels,
        }
        if ctx_role is not None:
            grp_record["context_role"] = str(ctx_res).strip()
            grp_record["context_relations"] = cleaned_ctx_rels
        cleaned_groups.append(grp_record)

    normalized = {
        "status": status,
        "task_summary": summary.strip(),
        "functional_roles": normalized_roles,
        "functional_relations": cleaned_relations,
        "interaction_groups": cleaned_groups,
        "inspectable_regions": cleaned_regions,
        "inspection_order": cleaned_order,
        "unsupported_reason": "",
    }
    if "cross_group_reuse_allowed" in document:
        normalized["cross_group_reuse_allowed"] = document["cross_group_reuse_allowed"]
    return normalized


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
            "binding_policy", "binding_cardinality", "min_count", "max_count", "preference",
            "candidate_categories", "visible_candidates", "required_properties",
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
        if "binding_cardinality" in role and not isinstance(role["binding_cardinality"], dict):
            raise FMResponseValidationError(
                f"functional_roles[{index}].binding_cardinality must be a dict"
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

        # Strict cardinality validation
        card_data = role.get("binding_cardinality")
        direct_min = role.get("min_count")
        direct_max = role.get("max_count")
        direct_pref = role.get("preference")

        if card_data is not None:
            if not isinstance(card_data, dict):
                raise FMResponseValidationError(f"functional_roles[{index}].binding_cardinality must be a dict")
            allowed_card_keys = {
                "minimum_distinct_physical_objects",
                "maximum_distinct_physical_objects",
                "preferred",
                "preference",
            }
            unexpected_keys = set(card_data) - allowed_card_keys
            if unexpected_keys:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality has unexpected keys: {sorted(unexpected_keys)}"
                )
            if "minimum_distinct_physical_objects" not in card_data or "maximum_distinct_physical_objects" not in card_data:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality missing required minimum/maximum fields"
                )
            c_min = card_data["minimum_distinct_physical_objects"]
            c_max = card_data["maximum_distinct_physical_objects"]
            if isinstance(c_min, bool) or not isinstance(c_min, int) or c_min < 1:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality minimum_distinct_physical_objects must be integer >= 1"
                )
            if isinstance(c_max, bool) or not isinstance(c_max, int) or c_max < 1:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality maximum_distinct_physical_objects must be integer >= 1"
                )
            if c_min > c_max:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality minimum ({c_min}) > maximum ({c_max})"
                )
            if c_max > req_count:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality maximum ({c_max}) > required_count ({req_count})"
                )
            if "preferred" in card_data and "preference" in card_data:
                if card_data["preferred"] != card_data["preference"]:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].binding_cardinality conflicting preferred ({card_data['preferred']!r}) and preference ({card_data['preference']!r})"
                    )
            c_pref = card_data.get("preferred") if "preferred" in card_data else card_data.get("preference")
            if c_pref is not None and c_pref not in {"minimize_distinct", "maximize_distinct", "deterministic_rank"}:
                raise FMResponseValidationError(
                    f"functional_roles[{index}].binding_cardinality invalid preference: {c_pref!r}"
                )
            if role.get("binding_policy") == "DISTINCT":
                if c_min != req_count or c_max != req_count:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}] has DISTINCT binding_policy but cardinality range [{c_min}, {c_max}] != required_count {req_count}"
                    )
            if direct_min is not None:
                if isinstance(direct_min, bool) or not isinstance(direct_min, int) or direct_min != c_min:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].min_count ({direct_min}) contradicts binding_cardinality minimum ({c_min})"
                    )
            if direct_max is not None:
                if isinstance(direct_max, bool) or not isinstance(direct_max, int) or direct_max != c_max:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].max_count ({direct_max}) contradicts binding_cardinality maximum ({c_max})"
                    )
            if direct_pref is not None:
                if direct_pref not in {"minimize_distinct", "maximize_distinct", "deterministic_rank"}:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].preference invalid: {direct_pref!r}"
                    )
                if c_pref is not None and direct_pref != c_pref:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].preference ({direct_pref!r}) contradicts binding_cardinality preference ({c_pref!r})"
                    )
        else:
            if direct_min is not None or direct_max is not None:
                if direct_min is None or direct_max is None:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}] must provide both min_count and max_count"
                    )
                if isinstance(direct_min, bool) or not isinstance(direct_min, int) or direct_min < 1:
                    raise FMResponseValidationError(f"functional_roles[{index}].min_count must be integer >= 1")
                if isinstance(direct_max, bool) or not isinstance(direct_max, int) or direct_max < 1:
                    raise FMResponseValidationError(f"functional_roles[{index}].max_count must be integer >= 1")
                if direct_min > direct_max:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].min_count ({direct_min}) > max_count ({direct_max})"
                    )
                if direct_max > req_count:
                    raise FMResponseValidationError(
                        f"functional_roles[{index}].max_count ({direct_max}) > required_count ({req_count})"
                    )
                if role.get("binding_policy") == "DISTINCT":
                    if direct_min != req_count or direct_max != req_count:
                        raise FMResponseValidationError(
                            f"functional_roles[{index}] has DISTINCT binding_policy but cardinality range [{direct_min}, {direct_max}] != required_count {req_count}"
                        )
            if direct_pref is not None and direct_pref not in {"minimize_distinct", "maximize_distinct", "deterministic_rank"}:
                raise FMResponseValidationError(f"functional_roles[{index}].preference invalid: {direct_pref!r}")

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
        if role_counts.get(tgt_role) < req_tgt_c:
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
        # Measured, not guessed.  Across the 96 archived semantic responses the
        # median completion is 7.7k tokens and 39 of the 91 that finished are
        # longer than 8192 -- so the old default would have truncated 43% of
        # them, and the distribution that produced those archives only worked
        # because an environment variable happened to be set.  The longest
        # response that finished is 12222 tokens; the five that did not finish
        # ran to 24000 and would not have been saved by any budget.  24000 is
        # therefore the frozen default: it reproduces the archived runs, leaves
        # roughly twice the headroom over the longest complete response, and
        # does not depend on anyone remembering to export anything.
        #
        # Truncation stays a reported failure of the response rather than
        # something to retry.  Exactly one semantic request per trial is the
        # scientific rule, and a second request would change what is being
        # measured.
        self.max_tokens = max_tokens or int(
            _env_first("TAMP_FM_MAX_TOKENS", "FM_MAX_TOKENS", default="24000")
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
        schema_version = int(os.environ.get("TAMP_FM_SCHEMA_VERSION", "2"))
        if schema_version in (2, 3):
            return self.generate_task_requirements(
                task_instruction, observation_images=observation_images
            )
        image_blocks, self.last_observation_images = _encode_observation_images(
            _select_vlm_observation_images(observation_images)
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
            "temperature": 0.0,
            "top_p": 1.0,
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
        from mujoco_scenes.functional_tamp_pipeline.telemetry import record_semantic_request
        record_semantic_request(payload)
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        document = _extract_json_content(
            self._completion_transport().complete(payload),
            call_kind="kitchen_functional_graph",
            sanitized_request=sanitized_req,
        )
        self.last_raw_kitchen_graph_response = deepcopy(document)
        if getattr(self, "return_raw_graph", False):
            return document
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
            _select_vlm_observation_images(observation_images)
        )
        transport = self._completion_transport()
        schema_version = int(os.environ.get("TAMP_FM_SCHEMA_VERSION", "2"))
        if schema_version == 3:
            system_prompt = SYSTEM_PROMPT_V3
            user_request = USER_REQUEST_V3
            response_schema = LIVE_RESPONSE_SCHEMA_V3
            schema_name = "functional_specification_v3"
        elif schema_version == 2:
            system_prompt = SYSTEM_PROMPT_V2
            user_request = USER_REQUEST_V2
            response_schema = LIVE_RESPONSE_SCHEMA_V2
            schema_name = "functional_specification"
        else:
            system_prompt = SYSTEM_PROMPT
            user_request = USER_REQUEST_V2
            response_schema = RESPONSE_SCHEMA
            schema_name = "functional_specification"
        user_prompt_data = {
            "task_instruction": task_instruction.strip(),
            "request": user_request,
        }
        user_text = json.dumps(user_prompt_data, separators=(",", ":"))

        sanitized_req = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt_data,
            "schema_name": schema_name,
            "num_images": len(self.last_observation_images),
            "image_metadata": self.last_observation_images,
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *image_blocks,
                    ],
                },
            ],
            "temperature": _sampling_float("TAMP_FM_TEMPERATURE", 0.0),
            "top_p": _sampling_float("TAMP_FM_TOP_P", 1.0),
            "top_k": _sampling_int("TAMP_FM_TOP_K", 20),
            "min_p": 0.0,
            "presence_penalty": _sampling_float("TAMP_FM_PRESENCE_PENALTY", 0.0),
            "repetition_penalty": _sampling_float("TAMP_FM_REPETITION_PENALTY", 1.0),
            "max_tokens": self.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": os.environ.get("TAMP_FM_ENABLE_THINKING", "false").strip().lower() in ("true", "1", "yes")},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        from mujoco_scenes.functional_tamp_pipeline.telemetry import record_semantic_request
        record_semantic_request(payload)
        self.metrics.requirement_calls += 1
        self.metrics.total_calls += 1
        response = transport.complete(payload)
        raw_document = _extract_json_content(
            response, call_kind="task_requirements", sanitized_request=sanitized_req
        )
        self.last_raw_requirement_response = deepcopy(raw_document)
        live_wire_document = None
        if getattr(self, "return_raw_graph", False):
            # The caller asked for the document as it arrived because it knows
            # which domain this is and normalizes it accordingly.  This adapter
            # does not know the domain, and normalizing without one silently
            # skips every domain-conditional repair -- the Living Room
            # entity-kind normalization, and the recovery of a reference to a
            # place the runtime owns -- and then rejects the contract at the
            # very gate those repairs exist to get past.  Returning here leaves
            # the caller's own domain-aware call to do the work.
            return raw_document
        if schema_version == 3 and is_v3_document(raw_document):
            live_wire_document, self.last_normalization_trace = normalize_and_validate_v3_contract(
                raw_document, task_instruction=task_instruction
            )
        elif schema_version == 2 and is_v2_document(raw_document):
            live_wire_document, self.last_normalization_trace = normalize_and_validate_v2_contract(
                raw_document
            )
        if getattr(self, "return_raw_graph", False):
            return live_wire_document if live_wire_document is not None else raw_document
        if live_wire_document is not None:
            return live_wire_document
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
        image_blocks, self.last_observation_images = _encode_observation_images(
            _select_vlm_observation_images(observation_images)
        )
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
            "temperature": 0.0,
            "top_p": 1.0,
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
        from mujoco_scenes.functional_tamp_pipeline.telemetry import record_semantic_request
        record_semantic_request(payload)
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
