"""V3 one-call FM wire contract with unordered semantic participants.

V3 removes backend slot and operation-reuse syntax from the model-facing
contract.  Conversion resolves those details from the runtime ontology and
capability signatures before handing the existing canonical IR downstream.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from itertools import permutations, product
from typing import Any, Mapping

import jsonschema

from .errors import MalformedVLMSpecificationError, TaskSpecificationValidationError
from .fm_schema_v2 import RESPONSE_SCHEMA_V2
from .relation_interpreter import extract_relation_semantic_candidates, interpret_task_effect_predicate
from .robot_capability_registry import extract_operation_semantic_candidates
from .semantic_typing import build_role_type_hypotheses, relation_canonical_role_pairs


SYSTEM_PROMPT_V3 = """You generate one open-ended functional task contract from an instruction and initial RGB views. Return only JSON matching the schema. Do not output an action sequence or backend predicate names.

Reason silently in this order:
1. Clause audit: for every task clause identify its physical participants, required binary relations, and required physical operations.
2. Participant ledger: declare every independently groundable object, support region, or fixed anchor required by the task before considering visibility. Hidden participants remain required.
3. Function audit: distinguish material sources, receiving containers, reusable instruments, manipulated joining components, fixed receiving targets, personal supports, shared supports, and seating/context anchors. Roles with different causal functions remain distinct even if they share a broad object category.
4. Operation audit: list every directly participating role in participant_roles. Array order has no source/target/anchor meaning. A material transfer needs its material source and receiver. Placement relative to seating needs payload, support, and seating/context reference. Fastening needs an implement, manipulated joining component, and fixed receiving target; a generic workbench is not automatically that target. A display is not automatically a seating/accessibility anchor.
5. Count audit: required_count is the minimum number of distinct physical instances; operation_count is the number of operation applications. REUSABLE or SHARED roles may participate repeatedly.
6. Consistency audit: every participant is declared, every clause is represented, personal and shared supports remain distinct when their functions differ, and no robot arm, body, gripper, or end effector is proposed as a task object unless the instruction explicitly asks to manipulate that robot component.

Use short atomic free-form role functions, relations, operations, and unary properties. Relation participant order is not directional. Do not estimate numeric geometry. Observation candidates are visible evidence only. Inspectable regions must be visible closed/storage structures; their reasons may say they could be inspected for task-relevant candidates but must not claim hidden contents."""


USER_REQUEST_V3 = (
    "Derive the complete functional task contract from the instruction, then use the "
    "three initial views only for observation guidance. Declare all physical participants "
    "before visibility analysis. For each relation provide exactly two unordered declared "
    "participant roles. For each physical operation provide every directly involved declared "
    "role as an unordered participant set. Keep physical-instance counts separate from "
    "operation-application counts. Output only the final JSON."
)


RESPONSE_SCHEMA_V3: dict[str, Any] = deepcopy(RESPONSE_SCHEMA_V2)
RESPONSE_SCHEMA_V3["properties"] = dict(RESPONSE_SCHEMA_V3["properties"])
RESPONSE_SCHEMA_V3["properties"]["schema_version"] = {"type": "integer", "enum": [3]}
RESPONSE_SCHEMA_V3["required"] = ["schema_version", *RESPONSE_SCHEMA_V3["required"]]
_contract = RESPONSE_SCHEMA_V3["properties"]["task_contract"]
_contract["properties"]["functional_roles"]["items"]["required"] = [
    "id", "entity_kind", "function", "required_count", "binding_policy",
    "candidate_categories", "required_properties",
]
_contract["properties"]["functional_relations"]["items"] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "pattern": "^[a-zA-Z0-9_]+$"},
        "relation": {"type": "string", "minLength": 1},
        "participant_roles": {
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": {"type": "string", "minLength": 1},
        },
        "required": {"type": "boolean"},
    },
    "required": ["id", "relation", "participant_roles", "required"],
    "additionalProperties": False,
}
_contract["properties"]["operation_pairings"]["items"] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "pattern": "^[a-zA-Z0-9_]+$"},
        "operation": {"type": "string", "minLength": 1},
        "participant_roles": {
            "type": "array", "minItems": 2, "maxItems": 4,
            "items": {"type": "string", "minLength": 1},
        },
        "operation_count": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    "required": ["id", "operation", "participant_roles", "operation_count"],
    "additionalProperties": False,
}

LIVE_RESPONSE_SCHEMA_V3: dict[str, Any] = deepcopy(RESPONSE_SCHEMA_V3)
_guidance = LIVE_RESPONSE_SCHEMA_V3["properties"]["observation_guidance"]
_region = _guidance["properties"]["inspectable_regions"]["items"]
_region["properties"] = {
    "id": {"type": "string", "minLength": 1, "pattern": "^[a-zA-Z0-9_]+$"},
    "label": {"type": "string", "minLength": 1},
    "visual_description": {"type": "string", "minLength": 1},
    "reason": {"type": "string", "minLength": 1},
}
_region["required"] = ["id", "label", "visual_description", "reason"]
_region["additionalProperties"] = False


def is_v3_document(doc: Mapping[str, Any]) -> bool:
    return isinstance(doc, Mapping) and doc.get("schema_version") == 3 and "task_contract" in doc


_ROBOT_SELF = re.compile(
    r"\b(robot(?:ic)?\s+(?:arm|body|platform)|gripper|end[ -]?effector(?:\s+attachment)?)\b",
    re.IGNORECASE,
)


def normalize_v3_live_document(doc: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(doc, Mapping):
        raise MalformedVLMSpecificationError("Live V3 specification must be a JSON object")
    normalized = deepcopy(dict(doc))
    trace: list[dict[str, Any]] = []
    contract = normalized.get("task_contract", {})
    roles = {role.get("id"): role for role in contract.get("functional_roles", []) if isinstance(role, dict)}
    visible = normalized.get("observation_guidance", {}).get("visible_candidates_per_role", {})
    for role_id, candidates in list(visible.items()):
        function = str(roles.get(role_id, {}).get("function", ""))
        if _ROBOT_SELF.search(function):
            continue
        kept = []
        for candidate in candidates if isinstance(candidates, list) else []:
            text = f"{candidate.get('label', '')} {candidate.get('visual_description', '')}" if isinstance(candidate, dict) else ""
            if _ROBOT_SELF.search(text):
                trace.append({"code": "ROBOT_SELF_CANDIDATE_REMOVED", "role_id": role_id, "candidate": candidate})
            else:
                kept.append(candidate)
        visible[role_id] = kept
    return normalized, trace


def _validate_declared_references(doc: Mapping[str, Any]) -> None:
    contract = doc["task_contract"]
    roles = contract["functional_roles"]
    role_ids = [role["id"] for role in roles]
    if len(role_ids) != len(set(role_ids)):
        raise MalformedVLMSpecificationError("DUPLICATE_ROLE_ID: V3 role IDs must be unique")
    declared = set(role_ids)
    for collection in ("functional_relations", "operation_pairings"):
        entries = contract[collection]
        entry_ids = [entry["id"] for entry in entries]
        if len(entry_ids) != len(set(entry_ids)):
            raise MalformedVLMSpecificationError(f"DUPLICATE_ENTRY_ID: {collection} IDs must be unique")
        for index, entry in enumerate(entries):
            missing = sorted(set(entry["participant_roles"]) - declared)
            if missing:
                raise MalformedVLMSpecificationError(
                    f"UNDECLARED_PARTICIPANT: {collection}[{index}] references {missing}"
                )
    visible = doc["observation_guidance"]["visible_candidates_per_role"]
    unknown_visible = sorted(set(visible) - declared)
    if unknown_visible:
        raise MalformedVLMSpecificationError(f"UNKNOWN_VISIBLE_ROLE: {unknown_visible}")
    regions = doc["observation_guidance"]["inspectable_regions"]
    order = doc["observation_guidance"]["inspection_order"]
    region_ids = [region["id"] for region in regions]
    if len(region_ids) != len(set(region_ids)) or len(order) != len(set(order)) or set(order) != set(region_ids):
        raise MalformedVLMSpecificationError("INVALID_INSPECTION_ORDER: must list every unique region ID exactly once")


def _base_canonical_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    contract = doc["task_contract"]
    guidance = doc["observation_guidance"]
    visible = guidance["visible_candidates_per_role"]
    roles = []
    for role in contract["functional_roles"]:
        item = dict(role)
        item["visible_candidates"] = visible.get(role["id"], [])
        roles.append(item)
    return {
        "schema_version": 3,
        "status": doc["status"],
        "task_summary": doc["task_summary"],
        "functional_roles": roles,
        "functional_relations": [],
        "interaction_groups": [],
        "inspectable_regions": guidance["inspectable_regions"],
        "inspection_order": guidance["inspection_order"],
        "unsupported_reason": doc["unsupported_reason"],
        "raw_v3_contract": contract,
        "raw_v3_guidance": guidance,
    }


def _relation_options(domain: str, relation: Mapping[str, Any], hypotheses: Mapping[str, Any]) -> list[dict[str, Any]]:
    left, right = relation["participant_roles"]
    options: list[dict[str, Any]] = []
    for semantic in extract_relation_semantic_candidates(domain, relation["relation"]):
        pairs = relation_canonical_role_pairs(domain, semantic.predicate_name, semantic.category)
        if semantic.category == "TASK_EFFECT_SEMANTICS":
            # Effect carrier orientation is linguistic; both raw arrangements are
            # retained until an explicit operation can corroborate the state edge.
            pairs = {(s, o) for s in hypotheses[left].canonical_role_candidates
                     for o in hypotheses[right].canonical_role_candidates}
        for subject_raw, object_raw in ((left, right), (right, left)):
            subject_types = set(hypotheses[subject_raw].canonical_role_candidates)
            object_types = set(hypotheses[object_raw].canonical_role_candidates)
            if any(s in subject_types and o in object_types for s, o in pairs):
                options.append({
                    "subject_role": subject_raw, "object_role": object_raw,
                    "predicate": semantic.predicate_name, "category": semantic.category,
                })
    return [dict(row) for row in sorted({tuple(sorted(item.items())) for item in options})]


def resolve_v3_operation_slots(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
) -> list[dict[str, Any]]:
    capabilities = extract_operation_semantic_candidates(domain, str(operation["operation"]))
    participants = tuple(operation["participant_roles"])
    options: list[dict[str, Any]] = []
    for capability in capabilities:
        slot_names = ("source", "target", "anchor") if capability.allowed_anchor_roles else ("source", "target")
        if len(participants) != len(slot_names):
            continue
        allowed = {
            "source": set(capability.allowed_source_roles),
            "target": set(capability.allowed_target_roles),
            "anchor": set(capability.allowed_anchor_roles),
        }
        for assigned in permutations(participants):
            raw_slots = dict(zip(slot_names, assigned))
            type_domains = [
                sorted(set(hypotheses[raw_slots[slot]].canonical_role_candidates) & allowed[slot])
                for slot in slot_names
            ]
            if any(not values for values in type_domains):
                continue
            for types in product(*type_domains):
                typed = dict(zip(slot_names, types))
                source_role = roles_by_id[raw_slots["source"]]
                binding = source_role["binding_policy"]
                count = int(operation["operation_count"])
                if binding == "DISTINCT" and int(source_role["required_count"]) < count:
                    continue
                options.append({
                    "capability_id": capability.capability_id,
                    "planner_operation": capability.planner_operation,
                    "source_role": raw_slots["source"],
                    "target_role": raw_slots["target"],
                    "anchor_role": raw_slots.get("anchor"),
                    "source_type": typed["source"],
                    "target_type": typed["target"],
                    "anchor_type": typed.get("anchor"),
                    "usage_policy": "SEQUENTIAL_REUSE_ALLOWED" if binding in {"REUSABLE", "SHARED"} else "DEDICATED_PER_TARGET",
                })
    unique = {json.dumps(item, sort_keys=True): item for item in options}
    return [unique[key] for key in sorted(unique)]


def convert_v3_to_canonical_document(v3_doc: Mapping[str, Any], *, domain: str) -> dict[str, Any]:
    canonical = _base_canonical_document(v3_doc)
    contract = v3_doc["task_contract"]
    roles_by_id = {role["id"]: role for role in contract["functional_roles"]}
    hypotheses = build_role_type_hypotheses(domain, canonical)

    relations = []
    for relation in contract["functional_relations"]:
        options = _relation_options(domain, relation, hypotheses)
        semantic_candidates = extract_relation_semantic_candidates(domain, relation["relation"])
        if semantic_candidates and not options:
            raise TaskSpecificationValidationError(
                "FM_INTERNAL_RELATION_PARTICIPANT_CONTRADICTION: "
                f"relation {relation['id']!r} ({relation['relation']!r}) has no valid "
                f"orientation for participant_roles={relation['participant_roles']}"
            )
        left, right = relation["participant_roles"]
        endpoint_orientations = {(row["subject_role"], row["object_role"]) for row in options}
        if len(endpoint_orientations) == 1:
            subject, object_ = next(iter(endpoint_orientations))
        else:
            subject, object_ = left, right
        relations.append({
            "id": relation["id"], "subject_role": subject,
            "relation": relation["relation"], "object_role": object_,
            "required": relation["required"], "unordered_participants": True,
            "v3_orientation_options": options,
        })
    canonical["functional_relations"] = relations
    hypotheses = build_role_type_hypotheses(domain, canonical)

    groups = []
    for operation in contract["operation_pairings"]:
        capabilities = extract_operation_semantic_candidates(domain, operation["operation"])
        options = resolve_v3_operation_slots(domain, operation, roles_by_id, hypotheses)
        if capabilities and not options:
            raise TaskSpecificationValidationError(
                "MISSING_OR_CONTRADICTORY_OPERATION_PARTICIPANTS: "
                f"operation {operation['id']!r} ({operation['operation']!r}) cannot instantiate "
                f"any supported capability from participant_roles={operation['participant_roles']}"
            )
        if options:
            first = options[0]
            source, target, anchor = first["source_role"], first["target_role"], first["anchor_role"]
            usage = first["usage_policy"]
        else:
            source, target = operation["participant_roles"][:2]
            anchor = operation["participant_roles"][2] if len(operation["participant_roles"]) > 2 else None
            usage = "SEQUENTIAL_REUSE_ALLOWED" if roles_by_id[source]["binding_policy"] in {"REUSABLE", "SHARED"} else "DEDICATED_PER_TARGET"
        groups.append({
            "id": operation["id"], "function": operation["operation"],
            "tool_role": source, "target_role": target, "context_role": anchor,
            "required_target_count": operation["operation_count"], "usage_policy": usage,
            "required_relations": [], "context_relations": [],
            "v3_slot_assignments": options,
        })
    canonical["interaction_groups"] = groups
    for edge in canonical["functional_relations"]:
        effect = interpret_task_effect_predicate(edge["relation"])
        participants = {edge["subject_role"], edge["object_role"]}
        for group in groups:
            if participants != {group["tool_role"], group["target_role"]}:
                continue
            if effect == "CONTAINS":
                edge["subject_role"], edge["object_role"] = group["target_role"], group["tool_role"]
            elif effect == "PLACED_ON":
                edge["subject_role"], edge["object_role"] = group["tool_role"], group["target_role"]
            break
    return canonical


def validate_v3_live_contract(doc: Mapping[str, Any], *, domain: str | None = None) -> dict[str, Any]:
    try:
        jsonschema.validate(instance=dict(doc), schema=LIVE_RESPONSE_SCHEMA_V3)
    except jsonschema.ValidationError as exc:
        raise MalformedVLMSpecificationError(f"V3 schema validation failed: {exc.message}") from exc
    _validate_declared_references(doc)
    if doc["status"] == "SUPPORTED" and not doc["task_contract"]["functional_roles"]:
        raise MalformedVLMSpecificationError("SUPPORTED V3 specification must declare roles")
    if doc["status"] == "UNSUPPORTED":
        if any(doc["task_contract"][key] for key in ("functional_roles", "functional_relations", "operation_pairings")):
            raise MalformedVLMSpecificationError("UNSUPPORTED V3 specification must have an empty contract")
        return deepcopy(dict(doc))
    if domain is not None:
        convert_v3_to_canonical_document(doc, domain=domain)
    return deepcopy(dict(doc))


def normalize_and_validate_v3_contract(
    doc: Mapping[str, Any], *, domain: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized, trace = normalize_v3_live_document(doc)
    return validate_v3_live_contract(normalized, domain=domain), trace


def compute_v3_prompt_and_schema_hash() -> str:
    blob = json.dumps({
        "system_prompt_v3": SYSTEM_PROMPT_V3,
        "user_request_v3": USER_REQUEST_V3,
        "response_schema_v3": LIVE_RESPONSE_SCHEMA_V3,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
