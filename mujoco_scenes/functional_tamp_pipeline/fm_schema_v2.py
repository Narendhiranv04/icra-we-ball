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
from typing import Any, Mapping
import jsonschema

from mujoco_scenes.functional_tamp_pipeline.errors import MalformedVLMSpecificationError


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
6. Propagate all explicit quantities to role and operation counts.
7. Preserve distinct, shared, and sequentially reusable binding meaning.
8. Check that every task clause is represented by roles, relations, operations, counts, and bindings.
9. Only after the task_contract is complete, use RGB evidence to populate observation_guidance.
10. Never omit a participant merely because it is absent or occluded in the initial images.

Contract granularity rules:
- A role must denote an independently groundable physical object, support region, or fixed physical anchor that participates in a required transformation or physical constraint.
- Do not create roles for people/users, actions, goal states, events, quantities, abstract outcomes, or material contents that are not independently manipulated. Represent those meanings through counts, relations, and operations on their physical carriers.
- Do not create a separate role for each numbered instance when one functional role plus `required_count` represents equivalent participants.
- Relations must state physical functional, geometric, or causal dependencies needed to execute a transformation. Do not use mere purpose, ownership, narrative, visibility, or current-location facts as substitutes.
- Consolidate repeated equivalent transformations into one operation pairing with `operation_count`. Source and target must be different roles with meaningful physical interaction; avoid self-pairings.

A. FUNCTIONAL TASK CONTRACT (Derive from task semantics before considering visibility)
- Infer the complete set of physical participants and spatial functional roles required to achieve the task from task semantics, including participants that may currently be absent, occluded, or located inside closed storage.
- Visibility is evidence about current availability, not about whether a functional role is required. A role remains required even if no candidate is currently visible.
- Represent physically distinct participants separately whenever they perform different causal functions (such as a source, payload, manipulated component, tool, receiving target, support, or contextual anchor).
- Never assign physical object or region instance identifiers to roles; describe required functional capabilities.
- Set `entity_kind` to:
  - OBJECT: a selectable or manipulable physical item.
  - REGION: a selectable support surface, placement area, or spatial destination.
  - FIXED_TARGET: a non-selectable contextual reference or fixed target feature that participates in relations.
- Set `binding_policy` to:
  - DISTINCT: separate simultaneous physical items or individual entities are required.
  - REUSABLE: one physical item may be reused sequentially across multiple targets.
  - SHARED: one physical region or entity intentionally serves multiple items or users.
- Role counts:
  - Explicitly specify `required_count` (positive integer) for each role. Propagate explicit quantifiers ('each', 'both', numerical counts) from the user instruction.
- `candidate_categories`: list open-vocabulary semantic search phrases describing valid physical realizations capable of satisfying the role's stated function.
- `required_properties`: list only task-critical unary physical or geometric characteristics of this single role (leave empty [] if no special unary property is needed). Never place binary relations or part names here.
- `functional_relations`: express task-critical binary dependencies between declared roles using `subject_role`, `relation`, and `object_role`.
  - Use short atomic free-form phrases for `relation` in your own words.
  - Set `required: true` for relations necessary for task completion.
  - Both `subject_role` and `object_role` must reference declared role IDs in `functional_roles`.
- `operation_pairings`: express each task-required physical transformation or intervention between declared roles.
  - Specify `id`, `operation` (short free-form atomic phrase describing the transformation), `source_role`, `target_role`, `operation_count`, and `reuse_policy` ('DEDICATED_PER_TARGET' or 'REUSABLE_ACROSS_TARGETS').
  - Include optional `anchor_role` if the operation is anchored to a specific reference or fixed target.
  - `source_role` is the physical participant that performs, carries, or provides the intervention; `target_role` is the distinct physical participant directly acted on or supported; `anchor_role` is the contextual destination or fixed reference when needed. Do not substitute the anchor for the acted-on target.
  - When an implement acts on a manipulated item at a fixed location, the implement is the source, the manipulated item is the target, and the fixed location is the anchor. Identification, search, and selection are not physical operations.
  - For each operation, express the physical compatibility, fit, reach, support, or access dependencies that determine whether the declared source can perform it on the target and at any anchor. Use only dependencies implied by the task semantics.
  - For an anchored operation, separately consider the required source-to-target, source-to-anchor, and target-to-anchor dependencies; do not collapse all three participants into one vague relation.
  - Do not create operations for passive storage, visibility, or descriptive scene facts. Include the transformations the task actually requires, including final placement or restoration transformations stated by the user.

B. OBSERVATION GUIDANCE (Derive from multi-view RGB images after the task contract is complete)
- `visible_candidates_per_role`: map declared role IDs to arrays of visually apparent candidate items or regions in the initial images, with `label` and `visual_description`. May be empty for roles not currently visible.
- `inspectable_regions`: propose visible closed or storage regions that could be inspected if required participants are missing. Each physical unit must be proposed at most once.
- `inspection_order`: rank the proposed inspectable region IDs. If no closed storage search is required, leave inspectable_regions and inspection_order empty ([]).

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
                                "maxItems": 16,
                                "items": {"type": "string"},
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
                            "relation": {"type": "string"},
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
                            "operation": {"type": "string"},
                            "source_role": {"type": "string"},
                            "target_role": {"type": "string"},
                            "operation_count": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "reuse_policy": {
                                "type": "string",
                                "enum": [
                                    "DEDICATED_PER_TARGET",
                                    "REUSABLE_ACROSS_TARGETS",
                                ],
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
    """Compute deterministic SHA-256 hash of the V2 system prompt and V2 response schema."""
    blob = json.dumps(
        {
            "system_prompt_v2": SYSTEM_PROMPT_V2,
            "response_schema_v2": RESPONSE_SCHEMA_V2,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
