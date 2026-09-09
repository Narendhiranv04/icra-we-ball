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

SUPPORTED means the task can be represented as roles, relations, and operations. Hidden objects, missing visibility, unknown inventory, closed storage, or required inspection/search are not reasons for UNSUPPORTED. Use UNSUPPORTED only when the task itself cannot be represented by this abstraction; then emit an empty contract and explain why. A supported contract is non-empty and unsupported_reason is empty.

Entity kinds: OBJECT is an independently selectable/manipulable item; REGION is a spatial area, support surface, or selectable destination; FIXED_TARGET is a non-manipulated contextual reference or fixed interaction target. Cups, plates, remotes, tools, and fasteners are normally OBJECT. Support areas are normally REGION. Marked workpiece features and seating references may be FIXED_TARGET.

Binding policies: DISTINCT requires different physical instances; REUSABLE permits an instance to participate in multiple operation applications; SHARED denotes one intentionally common object, context, or region. Multiple independent payloads with required_count greater than one are normally DISTINCT, not SHARED. required_count never means operation count.

Reason silently in this order:
1. Clause audit: for every task clause identify its physical participants, required binary relations, and required physical operations.
2. Participant ledger: declare every independently groundable object, support region, or fixed anchor required by the task before considering visibility. Hidden participants remain required.
3. Function audit: distinguish material sources, receiving containers, reusable instruments, manipulated joining components, fixed receiving targets, personal supports, shared supports, and seating/context anchors. Roles with different causal functions remain distinct even if they share a broad object category.
4. Operation audit: use one atomic operation per transformation and list only its directly participating roles in participant_roles. Array order has no source/target/anchor meaning. Do not combine multiple material sources into one transfer. Placement relative to seating needs payload, support, and a seating/context role that actually represents the referenced seat set. Fastening needs an implement, manipulated joining component, and fixed receiving target; never substitute the robot or an undeclared abstract collection. A generic workbench is not automatically that target. A display is not automatically a seating/accessibility anchor.
5. Count audit: required_count is the minimum number of distinct physical instances; operation_count is the number of operation applications. REUSABLE or SHARED roles may participate repeatedly.
6. Consistency audit: every exact participant ID is declared; no relation or operation uses an abstract undeclared plural; every clause is represented; equivalent interchangeable instances use one counted role rather than numbered duplicate roles; personal and shared supports remain distinct when their functions differ; a relation saying "both" uses a declared context role representing both; and no robot arm, body, gripper, or end effector is declared or proposed as a task object unless the instruction explicitly asks to manipulate that robot component.

Represent only participants required by the actual instruction. Do not invent ingredients, material sources, or preparation mechanisms merely to explain a broad end-state verb.

Use short atomic natural-language role functions, relations, operations, and unary properties, not uppercase backend-style predicate names. Relation participant order is not directional. Initial-location statements such as currently on, located initially, or stored on are observation context, not automatically required final relations. Required relations express compatibility, functional dependency, final state, or a physical relation needed by an operation. Do not add operations merely to describe an already satisfied state unless the instruction requires the transformation. Do not estimate numeric geometry. Observation candidates are visible evidence only. Inspectable regions must be visible closed/storage structures; inspection_order must contain every exact declared region id once and nothing else. Region reasons may say they could be inspected for task-relevant candidates but must not claim hidden contents."""


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
    r"\b(robot(?:ic)?(?:\s+(?:arm|body|platform|system))?|manipulator|gripper|end[ -]?effector(?:\s+attachment)?)\b",
    re.IGNORECASE,
)


def normalize_v3_live_document(
    doc: Mapping[str, Any], *, task_instruction: str = "", domain: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(doc, Mapping):
        raise MalformedVLMSpecificationError("Live V3 specification must be a JSON object")
    normalized = deepcopy(dict(doc))
    trace: list[dict[str, Any]] = []
    contract = normalized.get("task_contract", {})
    reason = str(normalized.get("unsupported_reason", ""))
    if (
        normalized.get("status") == "UNSUPPORTED"
        and any(contract.get(key) for key in ("functional_roles", "functional_relations", "operation_pairings"))
        and re.search(r"\b(hidden|visibility|visible|inventory|closed|inspect|search|initial (?:image|view)|rgb)\b", reason, re.I)
        and not re.search(r"\b(cannot be represented|unrepresentable|unsupported (?:process|abstraction)|outside (?:the )?abstraction)\b", reason, re.I)
    ):
        normalized["status"] = "SUPPORTED"
        normalized["unsupported_reason"] = ""
        trace.append({"code": "OBSERVABILITY_UNSUPPORTED_NORMALIZED_TO_SUPPORTED", "raw_reason": reason})

    roles_list = contract.get("functional_roles", [])
    instruction_targets_robot = bool(
        re.search(r"\b(robot|robotic arm|gripper|end[ -]?effector|manipulator)\b", task_instruction, re.I)
        and re.search(r"\b(inspect|repair|replace|remove|install|manipulate|service|calibrate)\b", task_instruction, re.I)
    )
    executor_ids: set[str] = set()
    if not instruction_targets_robot:
        for role in roles_list:
            text = " ".join(str(role.get(key, "")) for key in ("id", "function"))
            if _ROBOT_SELF.search(text) or re.search(r"\b(active agent|manipulation agent|executor)\b", text, re.I):
                executor_ids.add(str(role.get("id")))
        if executor_ids:
            contract["functional_roles"] = [role for role in roles_list if role.get("id") not in executor_ids]
            operations = []
            from .robot_capability_registry import is_non_physical_operation_phrase
            for operation in contract.get("operation_pairings", []):
                participants = [p for p in operation.get("participant_roles", []) if p not in executor_ids]
                if len(participants) < 2 and is_non_physical_operation_phrase(str(operation.get("operation", ""))):
                    trace.append({"code": "IMPLICIT_ROBOT_EXECUTOR_REMOVED", "removed_role_ids": sorted(executor_ids), "removed_operation": operation.get("id")})
                    continue
                item = deepcopy(operation)
                item["participant_roles"] = participants
                operations.append(item)
            contract["operation_pairings"] = operations
            kept_relations = []
            for relation in contract.get("functional_relations", []):
                if executor_ids.intersection(relation.get("participant_roles", [])):
                    trace.append({"code": "IMPLICIT_ROBOT_EXECUTOR_REMOVED", "removed_role_ids": sorted(executor_ids), "removed_relation": relation.get("id")})
                else:
                    kept_relations.append(relation)
            contract["functional_relations"] = kept_relations
            trace.append({"code": "IMPLICIT_ROBOT_EXECUTOR_REMOVED", "removed_role_ids": sorted(executor_ids)})

    if domain == "living_room":
        op_participants = {
            participant
            for operation in contract.get("operation_pairings", [])
            if re.search(r"\b(move|transfer|relocate|place|position|transport)\b", str(operation.get("operation", "")), re.I)
            for participant in operation.get("participant_roles", [])
        }
        for role in contract.get("functional_roles", []):
            categories = re.sub(r"[_-]+", " ", " ".join(role.get("candidate_categories", [])).lower())
            text = re.sub(r"[_-]+", " ", f"{role.get('function', '')} {role.get('description', '')}".lower())
            old_kind = role.get("entity_kind")
            if role.get("id") in op_participants and re.search(r"\b(cup|plate|saucer|drinkware|remote|controller|electronic device)\b", categories + " " + text):
                role["entity_kind"] = "OBJECT"
            elif role.get("id") in op_participants and re.search(r"\b(personal|shared|central)\s+support\b", text) and re.search(r"\b(table|surface|support)\b", categories + " " + text):
                role["entity_kind"] = "REGION"
            if role.get("entity_kind") != old_kind:
                trace.append({"code": "ENTITY_KIND_SEMANTIC_NORMALIZATION", "role_id": role.get("id"), "old": old_kind, "new": role.get("entity_kind")})

    roles = {role.get("id"): role for role in contract.get("functional_roles", []) if isinstance(role, dict)}
    visible = normalized.get("observation_guidance", {}).get("visible_candidates_per_role", {})
    for executor_id in executor_ids:
        visible.pop(executor_id, None)
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
    guidance = normalized.get("observation_guidance", {})
    regions = guidance.get("inspectable_regions", [])
    kept_regions = []
    for region in regions:
        region_text = " ".join(str(region.get(key, "")) for key in ("id", "label", "visual_description"))
        if re.search(r"\b(open|visible)\b.*\b(tabletop|desk top|surface)\b|\b(tabletop|desk top)\b.*\b(open|visible)\b", region_text, re.I):
            trace.append({"code": "NON_STORAGE_INSPECTION_REGION_REMOVED", "region_id": region.get("id")})
            continue
        item = deepcopy(region)
        if re.search(r"\b(contains?|holds?|likely|exact|missing)\b", str(item.get("reason", "")), re.I):
            item["reason"] = "Could be inspected for additional task-relevant candidates."
            trace.append({"code": "HIDDEN_CONTENT_REASON_SANITIZED", "region_id": item.get("id")})
        kept_regions.append(item)
    guidance["inspectable_regions"] = kept_regions
    region_ids = [str(region.get("id")) for region in kept_regions]
    raw_order = guidance.get("inspection_order", [])
    repaired_order = [item for item in raw_order if item in region_ids]
    repaired_order.extend(item for item in region_ids if item not in repaired_order)
    if repaired_order != raw_order:
        guidance["inspection_order"] = repaired_order
        trace.append({"code": "INSPECTION_ORDER_NORMALIZED", "raw": raw_order, "normalized": repaired_order})
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
            participants = entry["participant_roles"]
            if len(set(participants)) != len(participants):
                code = "DUPLICATE_RELATION_PARTICIPANT" if collection == "functional_relations" else "DUPLICATE_OPERATION_PARTICIPANT"
                raise MalformedVLMSpecificationError(f"{code}: {collection}[{index}]")
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
        slot_names = (
            ("source", "target")
            if not capability.allowed_anchor_roles or (domain == "living_room" and len(participants) == 2)
            else ("source", "target", "anchor")
        )
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
    current_state_relations = []
    operation_pairs = [set(item.get("participant_roles", [])) for item in contract["operation_pairings"]]
    for relation in contract["functional_relations"]:
        phrase_norm = re.sub(r"[_\-/]+", " ", str(relation["relation"])).lower()
        pair = set(relation["participant_roles"])
        role_texts = [" ".join(str(roles_by_id[p].get(key, "")) for key in ("function", "description")) for p in pair]
        is_current = bool(re.search(r"\b(currently|initially|initial|starts?|stored|located at)\b", phrase_norm))
        if not is_current and re.search(r"\bsupported by\b", phrase_norm) and pair not in operation_pairs:
            is_current = any(re.search(r"\b(support|surface|workbench|context)\b", text, re.I) for text in role_texts)
        if is_current:
            current_state_relations.append({**dict(relation), "category": "CURRENT_STATE_CONTEXT", "provenance": "FM_EXPLICIT_SEMANTIC"})
            continue
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
    canonical["current_state_relations"] = current_state_relations
    hypotheses = build_role_type_hypotheses(domain, canonical)

    groups = []
    for operation in contract["operation_pairings"]:
        from .robot_capability_registry import is_non_physical_operation_phrase
        operation_norm = re.sub(r"[_-]+", " ", str(operation["operation"])).lower().strip()
        if is_non_physical_operation_phrase(str(operation["operation"])) or operation_norm.split(maxsplit=1)[0] in {
            "associate", "associates", "associating", "associated",
        }:
            canonical.setdefault("non_physical_operations", []).append({
                **dict(operation), "category": "NON_PHYSICAL_TASK_DIRECTIVE",
            })
            continue
        capabilities = extract_operation_semantic_candidates(domain, operation["operation"])
        options = resolve_v3_operation_slots(domain, operation, roles_by_id, hypotheses)
        if capabilities and not options:
            raise TaskSpecificationValidationError(
                "MISSING_OR_CONTRADICTORY_OPERATION_PARTICIPANTS: "
                f"operation {operation['id']!r} ({operation['operation']!r}) cannot instantiate "
                f"any supported capability from participant_roles={operation['participant_roles']}"
            )
        if len(options) == 1:
            first = options[0]
            source, target, anchor = first["source_role"], first["target_role"], first["anchor_role"]
            usage = first["usage_policy"]
        else:
            source = target = anchor = None
            usage = None
        groups.append({
            "id": operation["id"], "function": operation["operation"],
            "tool_role": source, "target_role": target, "context_role": anchor,
            "required_target_count": operation["operation_count"], "usage_policy": usage,
            "required_relations": [], "context_relations": [],
            "v3_slot_assignments": options,
            "v3_participant_roles": list(operation["participant_roles"]),
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
    doc: Mapping[str, Any], *, domain: str | None = None, task_instruction: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized, trace = normalize_v3_live_document(doc, task_instruction=task_instruction, domain=domain)
    return validate_v3_live_contract(normalized, domain=domain), trace


def compute_v3_prompt_and_schema_hash() -> str:
    blob = json.dumps({
        "system_prompt_v3": SYSTEM_PROMPT_V3,
        "user_request_v3": USER_REQUEST_V3,
        "response_schema_v3": LIVE_RESPONSE_SCHEMA_V3,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
