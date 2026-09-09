"""V2 Foundation Model Schema and Generic System Prompt.

Adheres strictly to Phase 2 contract:
- Generic structural and completeness guidance only.
- ZERO benchmark-specific nouns, task examples, relation examples, or verifier capability lists.
- Structured into task_contract (roles, free-form relations, free-form operation pairings)
  and observation_guidance (visible candidates, inspectable regions, ranking).
- Preserves full compatibility with V1 documents for archived replay.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping
import jsonschema

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError
from mujoco_scenes.functional_tamp_pipeline.robot_capability_registry import (
    is_non_physical_operation_phrase,
)


SYSTEM_PROMPT_V2 = """You are a vision-language functional-requirement specification generator.

Return only the requested JSON object. Do not produce an action sequence.

Structure your response into two complementary sections:
1. task_contract: Complete physical and functional requirements derived from the task instruction.
2. observation_guidance: Visually apparent candidates and inspectable storage regions derived from the initial multi-view RGB images.

Before writing JSON, perform this generic completeness audit internally:
1. Split every task clause into atomic required physical transformations.
2. Identify every distinct physical participant needed by each transformation.
3. Declare those task participants independently of whether they are visible.
4. Express every required functional or causal relation between participants.
5. Express every required physical operation; an operation is not interchangeable with a relation.
6. Separate physical role instance counts from operation application counts; never copy a target, user, or output quantity onto every participating source or tool.
7. For every operation, audit the source physical count, target physical count, operation count, source binding policy, and operation reuse policy. Ask whether the task requires multiple source objects or merely multiple uses of one reusable source.
8. Preserve distinct, shared, and sequentially reusable binding meaning. If a role is DISTINCT with count N, verify that the instruction actually requires N separate physical objects.
9. If one source or tool can be reused across several targets, do not multiply its physical role count by the number of targets.
10. Check that every task clause is represented by roles, relations, operations, counts, and bindings.
11. Only after the task_contract is complete, use RGB evidence to populate observation_guidance.
12. Never omit a participant merely because it is absent or occluded in the initial images.

Contract granularity rules:
- A role must denote an independently groundable physical object, support region, or fixed physical anchor that participates in a required transformation or physical constraint.
- Write each role `function` as one short functional noun phrase, preferably no more than 8 words, with no task narrative.
- Write each `required_properties` entry as one short unary physical or functional property of that role, preferably no more than 8 words. Do not put a binary relation, endpoint reference, explanation, or action sequence in a required property.
- Write each `relation` as one atomic binary relation, preferably 2-8 words. Do not join constraints, explain them, or encode an action sequence.
- Write each `operation` as one atomic physical transformation, preferably 1-8 words. Do not encode a multi-step sequence or explanation.
- Do not create roles for people/users, actions, goal states, events, quantities, abstract outcomes, or material contents that are not independently manipulated. Represent those meanings through counts, relations, and operations on their physical carriers.
- Do not create a separate role for each numbered instance when one functional role plus `required_count` represents equivalent participants.
- Relations must state physical functional, geometric, or causal dependencies needed to execute a transformation. Do not use mere purpose, ownership, narrative, visibility, or current-location facts as substitutes.
- Consolidate repeated equivalent transformations into one operation pairing with `operation_count`. Source and target must be different roles with meaningful physical interaction; avoid self-pairings.

A. FUNCTIONAL TASK CONTRACT (Derive from task semantics before considering visibility)
- Infer the complete set of physical participants and spatial functional roles required to achieve the task from task semantics, including participants that may currently be absent, occluded, or located inside closed storage.
- Visibility is evidence about current availability, not about whether a functional role is required. A role remains required even if no candidate is currently visible.
- Represent physically distinct participants separately whenever they perform different causal functions (such as a source, payload, manipulated component, tool, receiving target, support, or contextual anchor).
- Never assign physical object or region instance identifiers to roles; describe required functional capabilities.
- Every role must explicitly provide `id`, `entity_kind`, `function`, `required_count`, `binding_policy`, `candidate_categories`, and `required_properties`; do not rely on omitted-field defaults.
- Set `entity_kind` to:
  - OBJECT: a selectable or manipulable physical item.
  - REGION: a selectable support surface, placement area, or spatial destination.
  - FIXED_TARGET: a non-selectable contextual reference or fixed target feature that participates in relations.
- Set `binding_policy` to:
  - DISTINCT: the task requires `required_count` separate physical instances; they may not collapse to one object.
  - REUSABLE: one physical instance may be used sequentially in multiple operation applications. Usually use `required_count = 1` unless the task explicitly requires multiple reusable copies.
  - SHARED: one physical region, context, or entity intentionally serves several task participants simultaneously or as common context.
- Role counts:
  - `required_count` is the minimum number of distinct physical instances of this role that must exist simultaneously or independently for task completion.
  - Do not set `required_count` equal to an operation count merely because the role participates in that operation.
  - Do not copy the number of target items, users, or outputs onto a reusable source or tool. Task semantics alone determine physical role count.
  - Use `required_count > 1` only when multiple independent physical instances are actually required by the instruction.
- `candidate_categories`: list open-vocabulary semantic search phrases describing valid physical realizations capable of satisfying the role's stated function.
- `required_properties`: list only task-critical unary physical or geometric characteristics of this single role (leave empty [] if no special unary property is needed). Never place binary relations or part names here.
- `functional_relations`: express task-critical binary dependencies between declared roles using `subject_role`, `relation`, and `object_role`.
  - Use short atomic free-form phrases for `relation` in your own words.
  - Each relation entry must explicitly provide `id`, `subject_role`, `relation`, `object_role`, and `required`. Do not rely on defaults.
  - Set `required: true` for relations necessary for task completion.
  - Both `subject_role` and `object_role` must reference declared role IDs in `functional_roles`.
- `operation_pairings`: express each task-required physical transformation or intervention between declared roles.
  - Specify `id`, `operation` (short free-form atomic phrase describing the transformation), `source_role`, `target_role`, `operation_count`, and `reuse_policy` ('DEDICATED_PER_TARGET' or 'REUSABLE_ACROSS_TARGETS').
  - Include optional `anchor_role` if the operation is anchored to a specific reference or fixed target.
  - Do not omit any required operation field or rely on a default count, reuse policy, or source.
  - `source_role` is the physical participant that performs, carries, or provides the intervention; `target_role` is the distinct physical participant directly acted on or supported; `anchor_role` is the contextual destination or fixed reference when needed. Do not substitute the anchor for the acted-on target.
  - When an implement acts on a manipulated item at a fixed location, the implement is the source, the manipulated item is the target, and the fixed location is the anchor. Identification, search, and selection are not physical operations.
  - `operation_count` is the number of required applications of the physical transformation. It is not the source or tool object count, although it may equal target count when one application is required per target.
  - `REUSABLE_ACROSS_TARGETS` means the same physical source or tool may be used across multiple applications. It does not assert that only one source exists and never overrides the role's physical cardinality.
  - `DEDICATED_PER_TARGET` means distinct source instances are required for the target applications represented by the operation.
  - Abstract example: one reusable source acting on two distinct targets uses source `required_count = 1` with REUSABLE, target `required_count = 2` with DISTINCT, and `operation_count = 2` with REUSABLE_ACROSS_TARGETS.
  - For each operation, express the physical compatibility, fit, reach, support, or access dependencies that determine whether the declared source can perform it on the target and at any anchor. Use only dependencies implied by the task semantics.
  - For an anchored operation, separately consider the required source-to-target, source-to-anchor, and target-to-anchor dependencies; do not collapse all three participants into one vague relation.
  - Do not create operations for passive storage, visibility, or descriptive scene facts. Include the transformations the task actually requires, including final placement or restoration transformations stated by the user.

B. OBSERVATION GUIDANCE (Derive from multi-view RGB images after the task contract is complete)
- `visible_candidates_per_role`: map declared role IDs to arrays of visually apparent candidate items or regions in the initial images, with `label` and `visual_description`. May be empty for roles not currently visible.
- Candidate visibility is evidence only and never determines `required_count`: seeing one or several candidates must not rewrite the task-derived physical role count.
- `inspectable_regions`: propose visible closed or storage regions that could be inspected if required participants are missing. For each region explicitly provide a unique identifier-safe `id`, `label`, `visual_description`, and `reason`. Each physical unit must be proposed at most once. Do not predict or invent what is inside a closed region.
- `inspection_order`: provide every proposed inspectable-region ID exactly once, ranked in inspection order. If no closed storage search is required, leave inspectable_regions and inspection_order empty ([]).

C. STATUS SEMANTICS
- `SUPPORTED`: task can be represented with functional roles and relations. `functional_roles` in `task_contract` must be non-empty, `unsupported_reason` must be empty ("").
- `UNSUPPORTED`: use only when the task itself cannot be represented by this abstraction. All contract and guidance lists must be empty ([]), and `unsupported_reason` must be a non-empty explanation.
- Partial observability, missing visible candidates, or needing search are NOT reasons for UNSUPPORTED.
"""

RESPONSE_SCHEMA_V2: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["SUPPORTED", "UNSUPPORTED"],
        },
        "task_summary": {
            "type": "string",
        },
        "task_contract": {
            "type": "object",
            "properties": {
                "functional_roles": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "entity_kind": {
                                "type": "string",
                                "enum": ["OBJECT", "REGION", "FIXED_TARGET"],
                            },
                            "function": {
                                "type": "string",
                                "description": "One short functional noun phrase with no task narrative.",
                            },
                            "description": {"type": "string"},
                            "required_count": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Minimum number of distinct physical instances of this role required simultaneously or independently for task completion; not the number of operation applications.",
                            },
                            "binding_policy": {
                                "type": "string",
                                "enum": ["DISTINCT", "REUSABLE", "SHARED"],
                                "description": "Physical-role binding: DISTINCT requires separate instances, REUSABLE permits sequential use of an instance, and SHARED denotes common context or an entity intentionally shared by participants.",
                            },
                            "candidate_categories": {
                                "type": "array",
                                "minItems": 0,
                                "maxItems": 16,
                                "items": {"type": "string"},
                            },
                            "required_properties": {
                                "type": "array",
                                "minItems": 0,
                                "maxItems": 16,
                                "items": {
                                    "type": "string",
                                    "description": "One short unary physical or functional property of this role; not a relation or operation.",
                                },
                            },
                        },
                        "required": [
                            "id",
                            "entity_kind",
                            "function",
                            "required_count",
                            "binding_policy",
                        ],
                        "additionalProperties": False,
                    },
                },
                "functional_relations": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "subject_role": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "description": "One short atomic free-form binary relation, without explanation or action sequence.",
                            },
                            "object_role": {"type": "string"},
                            "required": {"type": "boolean"},
                        },
                        "required": ["subject_role", "relation", "object_role"],
                        "additionalProperties": False,
                    },
                },
                "operation_pairings": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "operation": {
                                "type": "string",
                                "description": "One short atomic free-form physical transformation, without explanation or multi-step sequence.",
                            },
                            "source_role": {"type": "string"},
                            "target_role": {"type": "string"},
                            "operation_count": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                                "description": "Number of required applications of this transformation; it does not determine source/tool physical instance count.",
                            },
                            "reuse_policy": {
                                "type": "string",
                                "enum": [
                                    "DEDICATED_PER_TARGET",
                                    "REUSABLE_ACROSS_TARGETS",
                                ],
                                "description": "Operation-level source participation: reuse may use the same source across applications, while dedicated requires distinct sources for represented targets; this never overrides role cardinality.",
                            },
                            "anchor_role": {"type": "string"},
                        },
                        "required": ["operation", "target_role"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "functional_roles",
                "functional_relations",
                "operation_pairings",
            ],
            "additionalProperties": False,
        },
        "observation_guidance": {
            "type": "object",
            "properties": {
                "visible_candidates_per_role": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
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
                },
                "inspectable_regions": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "label": {"type": "string"},
                            "visual_description": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                },
                "inspection_order": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "visible_candidates_per_role",
                "inspectable_regions",
                "inspection_order",
            ],
            "additionalProperties": False,
        },
        "unsupported_reason": {"type": "string"},
    },
    "required": [
        "status",
        "task_summary",
        "task_contract",
        "observation_guidance",
        "unsupported_reason",
    ],
    "additionalProperties": False,
}

# Archived V2 documents were generated against RESPONSE_SCHEMA_V2 and may omit
# fields that the legacy canonical conversion defaults. New one-call generation
# uses this structurally stricter copy so constrained decoding and post-validation
# agree without changing archived replay behavior.
LIVE_RESPONSE_SCHEMA_V2: dict[str, Any] = deepcopy(RESPONSE_SCHEMA_V2)
_live_contract_schema = LIVE_RESPONSE_SCHEMA_V2["properties"]["task_contract"]
_live_contract_schema["properties"]["operation_pairings"]["items"]["properties"][
    "anchor_role"
] = {"type": "string", "minLength": 1}
_live_contract_schema["properties"]["functional_roles"]["items"]["required"] = [
    "id",
    "entity_kind",
    "function",
    "required_count",
    "binding_policy",
    "candidate_categories",
    "required_properties",
]
_live_contract_schema["properties"]["functional_relations"]["items"]["required"] = [
    "id",
    "subject_role",
    "relation",
    "object_role",
    "required",
]
_live_contract_schema["properties"]["operation_pairings"]["items"]["required"] = [
    "id",
    "operation",
    "source_role",
    "target_role",
    "operation_count",
    "reuse_policy",
]
_live_guidance_schema = LIVE_RESPONSE_SCHEMA_V2["properties"]["observation_guidance"]
_live_region_schema = _live_guidance_schema["properties"]["inspectable_regions"]["items"]
_live_region_schema["properties"] = {
    "id": {
        "type": "string",
        "minLength": 1,
        "pattern": "^[a-zA-Z0-9_]+$",
    },
    "label": {"type": "string", "minLength": 1},
    "visual_description": {"type": "string", "minLength": 1},
    "reason": {"type": "string", "minLength": 1},
    "description": {"type": "string"},
}
_live_region_schema["required"] = ["id", "label", "visual_description", "reason"]
_live_region_schema["additionalProperties"] = False


USER_REQUEST_V2 = (
    "First derive a complete task contract from the instruction alone: atomic "
    "transformations, every required physical participant, all functional relations, "
    "all operations, explicit counts, and distinct/shared/reusable bindings. Audit "
    "every instruction clause. For every operation, keep source, target, and optional "
    "anchor pairwise distinct; omit anchor when the target itself is the operation "
    "location. Bind every declared role explicitly named by the operation, and never "
    "emit identification, selection, search, or inspection as an operation. Use one "
    "counted role for equivalent physical instances; exclude users, actions, states, "
    "and unmanipulated contents as standalone roles. Use physical dependencies rather "
    "than purpose or narrative relations. Only then use the initial images for visible "
    "candidates and search guidance. Inspectable regions must be visible closed, "
    "enclosed, or storage-access structures that could be inspected for missing "
    "candidates; never assert what a region contains and do not nominate open tabletops "
    "or empty staging areas. Do not omit a participant because it is not visible."
)


def normalize_v2_live_document(
    doc: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply only semantics-preserving cleanup before strict live validation."""
    if not isinstance(doc, Mapping):
        raise MalformedVLMSpecificationError("Live V2 specification must be a JSON object")
    normalized = deepcopy(dict(doc))
    trace: list[dict[str, Any]] = []
    operations = normalized.get("task_contract", {}).get("operation_pairings", [])
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or "anchor_role" not in operation:
            continue
        anchor = operation.get("anchor_role")
        location = f"task_contract.operation_pairings[{index}].anchor_role"
        if isinstance(anchor, str) and not anchor.strip():
            del operation["anchor_role"]
            trace.append({
                "code": "EMPTY_OPTIONAL_ANCHOR_REMOVED",
                "location": location,
                "operation_id": operation.get("id"),
            })
        elif anchor in (operation.get("source_role"), operation.get("target_role")):
            duplicate_of = (
                "source_role" if anchor == operation.get("source_role") else "target_role"
            )
            del operation["anchor_role"]
            trace.append({
                "code": "REDUNDANT_ANCHOR_REMOVED",
                "location": location,
                "operation_id": operation.get("id"),
                "duplicate_of": duplicate_of,
            })
    return normalized, trace


def is_v2_document(doc: Mapping[str, Any]) -> bool:
    """Return True if the document uses the V2 schema structure (has 'task_contract')."""
    return isinstance(doc, Mapping) and "task_contract" in doc


def validate_v2_functional_specification(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a V2 specification document against RESPONSE_SCHEMA_V2."""
    if not isinstance(doc, Mapping):
        raise MalformedVLMSpecificationError("V2 specification must be a JSON object")

    try:
        jsonschema.validate(instance=dict(doc), schema=RESPONSE_SCHEMA_V2)
    except jsonschema.ValidationError as err:
        raise MalformedVLMSpecificationError(f"V2 schema validation failed: {err.message}") from err

    # Semantic cross-field validation
    status = doc.get("status")
    contract = doc.get("task_contract", {})
    guidance = doc.get("observation_guidance", {})
    roles = contract.get("functional_roles", [])
    relations = contract.get("functional_relations", [])
    operations = contract.get("operation_pairings", [])

    if status == "SUPPORTED":
        if not roles:
            raise MalformedVLMSpecificationError("SUPPORTED specification must declare at least one functional role")
        if doc.get("unsupported_reason", "") != "":
            raise MalformedVLMSpecificationError("SUPPORTED specification must have empty unsupported_reason")
    elif status == "UNSUPPORTED":
        if roles or relations or operations:
            raise MalformedVLMSpecificationError("UNSUPPORTED specification must not declare roles, relations, or operations")
        if not doc.get("unsupported_reason"):
            raise MalformedVLMSpecificationError("UNSUPPORTED specification must provide non-empty unsupported_reason")

    # Validate role IDs uniqueness
    role_ids = set()
    for r in roles:
        rid = r.get("id")
        if not rid:
            raise MalformedVLMSpecificationError("Functional role must have non-empty id")
        if rid in role_ids:
            raise MalformedVLMSpecificationError(f"Duplicate functional role id: {rid}")
        role_ids.add(rid)

    # Validate relation endpoints point to declared roles
    for rel in relations:
        s = rel.get("subject_role")
        o = rel.get("object_role")
        if s not in role_ids:
            raise MalformedVLMSpecificationError(f"Relation subject_role {s!r} not in declared roles")
        if o not in role_ids:
            raise MalformedVLMSpecificationError(f"Relation object_role {o!r} not in declared roles")

    # Validate operation targets point to declared roles
    for op in operations:
        tgt = op.get("target_role")
        src = op.get("source_role")
        anch = op.get("anchor_role")
        if tgt and tgt not in role_ids:
            raise MalformedVLMSpecificationError(f"Operation target_role {tgt!r} not in declared roles")
        if src and src not in role_ids:
            raise MalformedVLMSpecificationError(f"Operation source_role {src!r} not in declared roles")
        if anch and anch not in role_ids:
            raise MalformedVLMSpecificationError(f"Operation anchor_role {anch!r} not in declared roles")

    # Validate observation guidance role references
    vis_map = guidance.get("visible_candidates_per_role", {})
    for rid in vis_map:
        if rid not in role_ids:
            raise MalformedVLMSpecificationError(
                f"visible_candidates_per_role references undeclared role {rid!r}"
            )

    return dict(doc)


_LIVE_ROLE_FIELDS = frozenset({
    "id",
    "entity_kind",
    "function",
    "required_count",
    "binding_policy",
    "candidate_categories",
    "required_properties",
})
_LIVE_RELATION_FIELDS = frozenset({
    "id",
    "subject_role",
    "relation",
    "object_role",
    "required",
})
_LIVE_OPERATION_FIELDS = frozenset({
    "id",
    "operation",
    "source_role",
    "target_role",
    "operation_count",
    "reuse_policy",
})
_BINARY_PROPERTY_MARKERS = frozenset({
    "at", "from", "inside", "into", "near", "on", "onto", "to", "under", "with",
})


def _require_live_fields(
    item: Mapping[str, Any], required_fields: frozenset[str], code: str, location: str
) -> None:
    missing = sorted(required_fields - set(item))
    if missing:
        raise MalformedVLMSpecificationError(
            f"{code}: {location} missing explicit fields {missing}"
        )


def _validate_atomic_phrase(value: Any, code: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedVLMSpecificationError(f"{code}: {location} must be non-empty")
    phrase = value.strip()
    words = re.findall(r"\b[\w'-]+\b", phrase)
    has_multiple_sentences = bool(re.search(r"[.!?]\s+\S", phrase))
    if "\n" in phrase or "\r" in phrase or ";" in phrase or has_multiple_sentences or len(words) > 10:
        raise MalformedVLMSpecificationError(
            f"{code}: {location} must be one phrase of at most 10 words"
        )
    return phrase


def _normalized_role_phrase(role_id: str) -> str:
    """Normalize an explicit FM role ID for conservative phrase matching."""
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", role_id).casefold()).strip()


def _phrase_mentions_role_id(operation_phrase: str, role_id: str) -> bool:
    normalized_operation = re.sub(
        r"\s+", " ", re.sub(r"[_-]+", " ", operation_phrase).casefold()
    ).strip()
    normalized_role = _normalized_role_phrase(role_id)
    return bool(
        normalized_role
        and re.search(rf"(?<!\w){re.escape(normalized_role)}(?!\w)", normalized_operation)
    )


def _validate_live_response_schema(doc: Mapping[str, Any]) -> None:
    try:
        jsonschema.validate(instance=dict(doc), schema=LIVE_RESPONSE_SCHEMA_V2)
    except jsonschema.ValidationError as err:
        raise MalformedVLMSpecificationError(
            f"Live V2 schema validation failed: {err.message}"
        ) from err


def validate_v2_live_contract(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce invariants required of newly generated live V2 documents.

    The base validator intentionally remains backward compatible for archived
    V2 replay. This stricter layer is called only at a live generation boundary.
    It rejects omissions and non-atomic prose without rewriting model output.
    """
    if not isinstance(doc, Mapping):
        raise MalformedVLMSpecificationError("Live V2 specification must be a JSON object")
    validated = validate_v2_functional_specification(doc)
    if validated.get("status") != "SUPPORTED":
        _validate_live_response_schema(validated)
        return validated

    contract = validated["task_contract"]
    roles = contract["functional_roles"]
    relations = contract["functional_relations"]
    operations = contract["operation_pairings"]
    role_ids = {role["id"] for role in roles}

    # Preserve stable, field-specific diagnostics while enforcing the same
    # required-field sets carried by the live constrained-decoding schema.
    for index, role in enumerate(roles):
        _require_live_fields(
            role, _LIVE_ROLE_FIELDS, "MISSING_LIVE_ROLE_FIELDS", f"functional_roles[{index}]"
        )
    for index, relation in enumerate(relations):
        _require_live_fields(
            relation,
            _LIVE_RELATION_FIELDS,
            "MISSING_LIVE_RELATION_FIELDS",
            f"functional_relations[{index}]",
        )
    for index, operation in enumerate(operations):
        _require_live_fields(
            operation,
            _LIVE_OPERATION_FIELDS,
            "MISSING_LIVE_OPERATION_FIELDS",
            f"operation_pairings[{index}]",
        )
    _validate_live_response_schema(validated)

    guidance = validated["observation_guidance"]
    regions = guidance["inspectable_regions"]
    inspection_order = guidance["inspection_order"]
    region_ids = [region["id"] for region in regions]
    if len(region_ids) != len(set(region_ids)):
        raise MalformedVLMSpecificationError(
            "DUPLICATE_INSPECTABLE_REGION_ID: inspectable region IDs must be unique"
        )
    if len(inspection_order) != len(set(inspection_order)):
        raise MalformedVLMSpecificationError(
            "DUPLICATE_INSPECTION_ORDER_ID: inspection_order IDs must be unique"
        )
    unknown_order_ids = sorted(set(inspection_order) - set(region_ids))
    if unknown_order_ids:
        raise MalformedVLMSpecificationError(
            f"UNKNOWN_INSPECTION_ORDER_REGION: undeclared IDs {unknown_order_ids}"
        )
    if set(inspection_order) != set(region_ids):
        missing_order_ids = sorted(set(region_ids) - set(inspection_order))
        raise MalformedVLMSpecificationError(
            f"INCOMPLETE_INSPECTION_ORDER: missing declared IDs {missing_order_ids}"
        )

    for index, role in enumerate(roles):
        location = f"functional_roles[{index}]"
        _validate_atomic_phrase(role["function"], "NON_ATOMIC_ROLE_FUNCTION", f"{location}.function")
        for prop_index, prop in enumerate(role["required_properties"]):
            property_location = f"{location}.required_properties[{prop_index}]"
            phrase = _validate_atomic_phrase(prop, "NON_ATOMIC_REQUIRED_PROPERTY", property_location)
            phrase_words = {word.casefold() for word in re.findall(r"\b[\w'-]+\b", phrase)}
            references_endpoint = any(
                referenced_id != role["id"]
                and re.search(rf"(?<!\w){re.escape(referenced_id)}(?!\w)", phrase, re.IGNORECASE)
                for referenced_id in role_ids
            )
            if references_endpoint and phrase_words.intersection(_BINARY_PROPERTY_MARKERS):
                raise MalformedVLMSpecificationError(
                    f"BINARY_REQUIRED_PROPERTY: {property_location} references another role endpoint"
                )

    for index, relation in enumerate(relations):
        location = f"functional_relations[{index}]"
        if not isinstance(relation["id"], str) or not relation["id"].strip():
            raise MalformedVLMSpecificationError(
                f"INVALID_LIVE_RELATION_ID: {location}.id must be non-empty"
            )
        _validate_atomic_phrase(
            relation["relation"], "NON_ATOMIC_RELATION_PHRASE", f"{location}.relation"
        )

    operation_semantic_errors: list[str] = []
    roles_by_id = {role["id"]: role for role in roles}
    for index, operation in enumerate(operations):
        location = f"operation_pairings[{index}]"
        if not isinstance(operation["id"], str) or not operation["id"].strip():
            raise MalformedVLMSpecificationError(
                f"INVALID_LIVE_OPERATION_ID: {location}.id must be non-empty"
            )
        operation_phrase = _validate_atomic_phrase(
            operation["operation"], "NON_ATOMIC_OPERATION_PHRASE", f"{location}.operation"
        )
        for endpoint in ("source_role", "target_role"):
            if operation[endpoint] not in role_ids:
                raise MalformedVLMSpecificationError(
                    f"INVALID_OPERATION_REFERENCE: {location}.{endpoint} must reference a declared role"
                )
        if "anchor_role" in operation and operation["anchor_role"] not in role_ids:
            raise MalformedVLMSpecificationError(
                f"INVALID_OPERATION_REFERENCE: {location}.anchor_role must reference a declared role"
            )
        if operation["source_role"] == operation["target_role"]:
            operation_semantic_errors.append(
                f"INVALID_OPERATION_SELF_PAIRING: {location} source_role equals target_role"
            )
        source_contract = roles_by_id.get(operation["source_role"], {})
        if (
            operation["reuse_policy"] == "DEDICATED_PER_TARGET"
            and operation["operation_count"] > source_contract.get("required_count", 0)
        ):
            operation_semantic_errors.append(
                "INCONSISTENT_OPERATION_REUSE_CARDINALITY: "
                f"{location} requires {operation['operation_count']} distinct source instances "
                f"but role {operation['source_role']!r} declares "
                f"required_count={source_contract.get('required_count')} and "
                f"binding_policy={source_contract.get('binding_policy')!r}"
            )
        anchor_role = operation.get("anchor_role")
        if anchor_role is not None and anchor_role in (
            operation["source_role"], operation["target_role"]
        ):
            duplicate = (
                "source_role" if anchor_role == operation["source_role"] else "target_role"
            )
            operation_semantic_errors.append(
                f"DUPLICATE_OPERATION_ENDPOINT: {location}.anchor_role equals {duplicate}"
            )
        if is_non_physical_operation_phrase(operation_phrase):
            operation_semantic_errors.append(
                f"NON_PHYSICAL_OPERATION: {location}.operation leads with a non-physical action"
            )

        bound_roles = {
            operation["source_role"], operation["target_role"], anchor_role
        }
        for role_id in sorted(role_ids):
            if role_id not in bound_roles and _phrase_mentions_role_id(operation_phrase, role_id):
                operation_semantic_errors.append(
                    f"OPERATION_MENTIONS_UNBOUND_ROLE: {location}.operation explicitly "
                    f"mentions declared role {role_id!r} absent from its endpoints"
                )

    if operation_semantic_errors:
        raise MalformedVLMSpecificationError("; ".join(operation_semantic_errors))

    return validated


def convert_v2_to_canonical_document(v2_doc: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a validated V2 specification to the canonical intermediate document format.

    Extracts roles, relations, operations, and observation guidance into the standard
    representation expected by downstream compiler and canonicalizers.
    """
    contract = v2_doc.get("task_contract", {})
    guidance = v2_doc.get("observation_guidance", {})
    vis_map = guidance.get("visible_candidates_per_role", {})

    canonical_roles = []
    for r in contract.get("functional_roles", []):
        role_copy = dict(r)
        rid = role_copy.get("id")
        role_copy["visible_candidates"] = vis_map.get(rid, [])
        canonical_roles.append(role_copy)

    canonical_relations = []
    for rel in contract.get("functional_relations", []):
        r_copy = dict(rel)
        if "required" not in r_copy:
            r_copy["required"] = True
        canonical_relations.append(r_copy)

    canonical_groups = []
    ops = contract.get("operation_pairings") or contract.get("interaction_groups") or []
    for op in ops:
        group_item = {
            "id": op.get("id", f"group_{len(canonical_groups)}"),
            "function": op.get("operation", op.get("function", "")),
            "tool_role": op.get("source_role", op.get("tool_role", "")),
            "target_role": op.get("target_role", ""),
            "required_target_count": op.get("operation_count", op.get("required_target_count", 1)),
            "usage_policy": op.get("reuse_policy", op.get("usage_policy", "DEDICATED_PER_TARGET")),
            "required_relations": op.get("required_relations", []),
            "context_role": op.get("anchor_role", op.get("context_role")),
            "context_relations": op.get("context_relations", []),
        }
        canonical_groups.append(group_item)

    return {
        "schema_version": 2,
        "status": v2_doc.get("status", "SUPPORTED"),
        "task_summary": v2_doc.get("task_summary", ""),
        "functional_roles": canonical_roles,
        "functional_relations": canonical_relations,
        "interaction_groups": canonical_groups,
        "inspectable_regions": guidance.get("inspectable_regions", []),
        "inspection_order": guidance.get("inspection_order", []),
        "unsupported_reason": v2_doc.get("unsupported_reason", ""),
        "raw_v2_contract": contract,
        "raw_v2_guidance": guidance,
    }


def compute_v2_prompt_and_schema_hash() -> str:
    """Hash the V2 prompt and strict schema actually sent for live generation."""
    blob = json.dumps(
        {
            "system_prompt_v2": SYSTEM_PROMPT_V2,
            "user_request_v2": USER_REQUEST_V2,
            "response_schema_v2": LIVE_RESPONSE_SCHEMA_V2,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
