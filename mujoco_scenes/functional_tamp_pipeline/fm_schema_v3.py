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
from itertools import combinations, permutations, product
from typing import Any, Mapping, Sequence

import jsonschema

from .errors import MalformedVLMSpecificationError, TaskSpecificationValidationError
from .fm_schema_v2 import RESPONSE_SCHEMA_V2
from .relation_interpreter import (
    extract_relation_semantic_candidates,
    interpret_task_effect_predicate,
    relation_states_where_things_currently_are,
)
from .robot_capability_registry import (
    ABSTRACT_ROLE_PROVENANCE,
    leads_with_abstract_task_directive,
    extract_operation_semantic_candidates,
    get_robot_capabilities,
    is_non_physical_operation_phrase,
)
from .operation_slot_completion import (
    ExpressedSemantics,
    _slot_cardinality,
    _slot_role_kind,
    capability_anchor_roles,
    collect_expressed_semantics,
    complete_operation_slots,
    probe_named_participant_slots,
)
from .semantic_typing import (
    RUNTIME_AUTHORED_ROLE_PROVENANCE,
    build_role_type_hypotheses,
    canonical_role_family,
    detect_role_families,
    relation_canonical_role_pairs,
)


SYSTEM_PROMPT_V3 = """You turn one instruction and three photographs of the starting scene into a single functional task contract. Return only JSON matching the schema. Never output an action sequence, a plan, or backend predicate names.

Almost every task is SUPPORTED. A robot will later open, search and inspect the scene, so nothing about what you cannot currently see bears on this decision. Hidden objects, closed drawers or cabinets, unknown contents, absent people, missing equipment, and anything needing a search are never grounds for UNSUPPORTED. Use UNSUPPORTED only when the task itself cannot be described as roles, relations and operations at all; then leave the contract empty and say why. A supported contract has roles and an empty unsupported_reason.

Do the work in two passes and keep them apart.

PASS 1 - THE TASK, WITHOUT LOOKING. Read the instruction alone. List every physical participant the task needs to be finished, and describe each by the job it does, in plain words. Two participants whose jobs differ are separate roles even when they could be the same kind of object: something that supplies material is not the thing that receives it, something used as an implement is not the thing being worked on, and a support belonging to one individual is not a support deliberately shared. Declare a role once with a count rather than repeating it. A participant the instruction needs stays required even when nothing in the photographs could serve it.

Only the instruction creates requirements. Do not invent a material, a supply, or a preparation step because a finished result implies one might exist.

PASS 2 - THE SCENE. Now use the three photographs, and only for guidance: what appears to be present, where things currently sit, and which closed or storage structures could be opened and searched. Being visible never makes something required. Where something currently sits is present state, not a goal, unless the instruction asks for it to go there or stay there.

ROLES. entity_kind is OBJECT for an independently movable item, REGION for an area or surface that can be a destination, FIXED_TARGET for a fixed reference or interaction point that is not carried around. required_count is how many times the task needs this participant, not how many of them must exist: two cups for two people is a count of two, and one jar poured from twice is a count of two as well. binding_policy is what says how those come down to physical things: DISTINCT when each time needs its own, REUSABLE when one can serve every time, SHARED for a single thing deliberately common to several. candidate_categories are ordinary names the thing might go by; required_properties are qualities it must have.

RELATIONS say what must hold for the task to count as done: one thing must suit or fit another, one must depend on another, or things must end up in a particular arrangement. Do not state a relation that merely reports how the scene already looks. When the instruction says both, all, each, between, or a pair, name every declared role the phrase covers instead of inventing one combined role.

OPERATIONS are the physical changes the instruction demands. Name each one as a single concrete physical action the robot performs on the participants you list: the specific motion, in a verb. Not an abstract noun for the activity, not a stage of the task, and not two actions bundled into one phrase. If a step needs a thing moved and then worked on, that is two operations. Give one operation for each distinct change and list only the roles directly taking part in it. Participant order carries no meaning: the runtime decides which one is acted on, which receives, and which is a reference. Do not add an operation to describe something already true. operation_count is how many times the change happens, and is independent of required_count.

People who only determine how many portions are needed change counts. They are not roles and not things to be handled, unless the task genuinely requires placing something relative to where someone is.

Do not guess measurements, distances, or coordinates. Write roles, relations and operations as short everyday phrases, not capitalised code-like names.

Observation guidance: list visible candidates under the role ids you declared; list only genuinely closed or storage structures as inspectable regions, saying why each could be worth searching without claiming to know its contents; let inspection_order name every region you declared exactly once and nothing else."""""


USER_REQUEST_V3 = (
    "Build the task contract from the instruction first, then use the three views only "
    "to describe what is present and what could be searched. Declare every required "
    "participant, including any that is not currently visible. Give each relation two to "
    "four declared roles, naming every member the instruction groups together. Give each "
    "operation the declared roles that directly take part. Keep instance counts and "
    "operation counts separate. Output only the final JSON."
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
            "type": "array", "minItems": 2, "maxItems": 4,
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


# Keys the normalizer adds for the runtime's own accounting.  They travel with
# the document but are never validated as wire content.
_RUNTIME_ONLY_DOCUMENT_KEYS = frozenset({"structurally_unusable_elements"})


def is_v3_document(doc: Mapping[str, Any]) -> bool:
    return isinstance(doc, Mapping) and doc.get("schema_version") == 3 and "task_contract" in doc


_ROBOT_SELF = re.compile(
    r"\b(robot(?:ic)?(?:\s+(?:arm|body|platform|system))?|manipulator|gripper|end[ -]?effector(?:\s+attachment)?)\b",
    re.IGNORECASE,
)


# A reference to a person, or to the room the task happens in, names something
# the runtime holds in no form and the prompt already says is not a role: "people
# who only determine how many portions are needed change counts.  They are not
# roles and not things to be handled."  Written into a participant list without a
# declaration, it is the same non-participant written more loosely.
_UNDECLARED_NON_PARTICIPANT = re.compile(
    r"^(?:the[_ ])?(?:person|people|persons|human|humans|user|users|occupant|occupants|"
    r"viewer|viewers|guest|guests|individual|individuals|recipient|recipients|diner|diners)"
    r"(?:[_ ]\d+)?$"
    r"|^(?:the[_ ])?(?:living[_ ]?room|kitchen|workshop|room|scene|environment|world|"
    r"workspace|work[_ ]?area)$",
    re.IGNORECASE,
)

_UNDECLARED_ROBOT_SELF = re.compile(
    r"^(robot|robotic|manipulator|gripper|end[ _-]?effector|arm)(_|$)|"
    r"(_|^)(robot|agent|executor)(_agent|_arm|_self)?$",
    re.IGNORECASE,
)


def _wire_schema_field_words() -> frozenset[str]:
    """Every word the wire schema's own field names are made of."""
    names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key == "properties" and isinstance(value, Mapping):
                    names.update(str(name) for name in value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(LIVE_RESPONSE_SCHEMA_V3)
    words: set[str] = set()
    for name in names:
        words.update(part for part in re.split(r"[^a-z]+", name.lower()) if part)
    return frozenset(words)


_WIRE_SCHEMA_FIELD_WORDS = _wire_schema_field_words()
_SCHEMA_ECHO_JOINERS = frozenset({"in", "of", "the", "a", "an", "for", "per", "and", "to"})


def _names_only_wire_schema_fields(participant: str) -> bool:
    """Whether this participant name is built entirely out of our own field names.

    The model occasionally answers the schema instead of the task, naming a
    participant "required_count_in_operation_pairing".  That is a description of
    a field of this very contract, so it denotes nothing in the world and
    removing it takes no semantics with it -- unlike an undeclared participant
    that names a real thing, a person or a part, which stays a wire failure
    because declaring one would assert a role the model never gave.

    The vocabulary is read off the schema itself rather than listed by hand, so
    it cannot drift from the contract the model was actually shown.
    """
    words = [word for word in re.split(r"[^a-zA-Z]+", str(participant).lower()) if word]
    content = [word for word in words if word not in _SCHEMA_ECHO_JOINERS]
    if not content:
        return False
    return all(
        word in _WIRE_SCHEMA_FIELD_WORDS
        or word.rstrip("s") in _WIRE_SCHEMA_FIELD_WORDS
        or f"{word}s" in _WIRE_SCHEMA_FIELD_WORDS
        for word in content
    )


def _anchor_only_families(domain: str) -> set[str]:
    """Role families this domain realizes with fixed anchors and nothing else.

    Read off the registry rather than listed here, so it cannot drift from what
    the runtime actually owns: Living Room realizes SEATING only as
    SEATING_POSITION and SEATING_PAIR, and the workshop realizes FIXED_TARGET
    only as repair_target, while every kitchen family has a selectable role.
    """
    from .semantic_typing import canonical_role_family
    from .system_context_registry import (
        get_domain_selectable_roles, get_domain_system_fixed_anchors)

    selectable = {canonical_role_family(domain, role)
                  for role in get_domain_selectable_roles(domain)}
    fixed = {canonical_role_family(domain, role)
             for role in get_domain_system_fixed_anchors(domain)}
    return {family for family in fixed - selectable if family != "OTHER"}


def _undeclared_reference_names_a_fixed_anchor(domain: str, name: str) -> str | None:
    """The fixed-anchor family an undeclared participant reference names.

    A reference to "the seat" or "the seating area" names a place the runtime
    already owns as a calibrated reference.  Declaring it takes nothing from
    guesswork: its kind, and the fact that it is one fixed place rather than
    something to go and find, come from the registry, and which quantified form
    of it the task needs is still decided later from the model's own words.

    A reference to something the robot would have had to *find* -- a cup, a
    plate, a part -- is refused here, because its acceptance vocabulary and its
    cardinality would both have to be chosen rather than read.  The two cases
    are separated by the registry itself: only a family this domain realizes
    with no selectable role at all is recoverable.
    """
    from .semantic_typing import detect_role_families

    families = _anchor_only_families(domain)
    if not families:
        return None
    probe = {
        "id": name,
        "entity_kind": "FIXED_TARGET",
        "function": re.sub(r"[_-]+", " ", str(name)),
        "description": "",
        "required_count": 1,
        "binding_policy": "SHARED",
        "candidate_categories": [],
        "required_properties": [],
    }
    detected, _ = detect_role_families(domain, probe)
    matched = detected & families
    if len(matched) != 1:
        return None
    # The wording must name *only* an anchor family.  "personal side table" is
    # a support described next to a seat, not a seat.
    if detected - families:
        return None
    return next(iter(matched))


def _repair_structural_wire_noise(
    normalized: dict[str, Any], trace: list[dict[str, Any]], domain: str | None = None
) -> None:
    """Repair the wire noise that carries no semantics, and nothing else.

    Three repairs, all deterministic and all recorded:

    An exact duplicate participant says nothing the single mention did not --
    "accessible to person, person" is one accessibility claim -- so the list is
    deduplicated.  Rejecting the whole contract over it discarded every other
    coherent semantic the model expressed.

    A reference to the robot itself is a reference to the executor, which is
    definitionally not one of the task's participants; the same removal already
    happens for a robot role the model declared, and an undeclared reference to
    it is the same thing written more loosely.

    A participant named entirely out of this contract's own field names is the
    model answering the schema instead of the task, so it denotes nothing in the
    world and removing it takes no semantics with it.

    A reference to a person, or to the room the task happens in, is removed on
    the same footing as the robot: the prompt tells the model that people are
    not roles and not things to be handled, so an undeclared reference to one
    denotes no participant, and rejecting an otherwise coherent contract over it
    threw away every semantic the model did express.

    Nothing else is repaired.  An undeclared participant that names a real thing
    the runtime could act on -- a part, a payload -- is missing semantics and
    stays a wire failure, because inventing a declaration for it would assert a
    role the model never gave: its kind, its cardinality and what would count as
    an instance of it are all absent, and every one of those would have to be
    chosen here rather than read.  An element left with too few participants to state a relation is not
    quietly dropped either: it is carried out separately so that the stage which
    accounts for required semantics still sees it.
    """
    contract = normalized.get("task_contract", {})
    declared = {
        str(role.get("id")) for role in contract.get("functional_roles", [])
        if isinstance(role, dict)
    }
    # An element id is a label the contract uses to refer to itself, not a
    # semantic.  Two relations that happen to share one are still two relations,
    # and rejecting the contract over the collision discarded both of them along
    # with everything else the model said.  The first keeps its id and the
    # others are given a suffixed one, recorded, so every element stays
    # addressable and nothing is merged.
    for collection in ("functional_relations", "operation_pairings"):
        seen_ids: set[str] = set()
        for entry in contract.get(collection, []) or []:
            if not isinstance(entry, dict):
                continue
            element_id = str(entry.get("id", ""))
            if element_id not in seen_ids:
                seen_ids.add(element_id)
                continue
            suffix = 2
            while f"{element_id}_{suffix}" in seen_ids:
                suffix += 1
            renamed = f"{element_id}_{suffix}"
            entry["id"] = renamed
            seen_ids.add(renamed)
            trace.append({
                "code": "DUPLICATE_ELEMENT_ID_RENAMED",
                "collection": collection,
                "raw_id": element_id,
                "normalized_id": renamed,
            })

    unusable: list[dict[str, Any]] = []
    for collection, minimum in (("functional_relations", 2), ("operation_pairings", 2)):
        kept = []
        for entry in contract.get(collection, []) or []:
            if not isinstance(entry, dict):
                kept.append(entry)
                continue
            participants = list(entry.get("participant_roles", ()) or ())
            deduplicated = list(dict.fromkeys(participants))
            if deduplicated != participants:
                trace.append({
                    "code": "DUPLICATE_PARTICIPANT_DEDUPLICATED",
                    "collection": collection, "element_id": entry.get("id"),
                    "raw_participant_roles": participants,
                    "normalized_participant_roles": deduplicated,
                })
            robots = [p for p in deduplicated
                      if p not in declared and _UNDECLARED_ROBOT_SELF.search(str(p))]
            if robots:
                deduplicated = [p for p in deduplicated if p not in robots]
                trace.append({
                    "code": "UNDECLARED_ROBOT_SELF_PARTICIPANT_REMOVED",
                    "collection": collection, "element_id": entry.get("id"),
                    "removed_participants": robots,
                })
            if domain:
                for participant in list(deduplicated):
                    if participant in declared:
                        continue
                    family = _undeclared_reference_names_a_fixed_anchor(domain, participant)
                    if family is None:
                        continue
                    recovered = {
                        "id": participant,
                        "entity_kind": "FIXED_TARGET",
                        "function": (
                            f"{re.sub(r'[_-]+', ' ', str(participant))} referenced by the "
                            f"model as a participant without being declared"
                        ),
                        "description": (
                            "Declaration recovered from an explicit reference to a place "
                            "the runtime holds as a calibrated fixed anchor; no kind, "
                            "count or acceptance vocabulary was chosen here."
                        ),
                        "required_count": 1,
                        "binding_policy": "SHARED",
                        "candidate_categories": [],
                        "required_properties": [],
                    }
                    # Only the fields the wire contract declares; where this
                    # declaration came from is recorded in the trace below,
                    # because the contract still has to validate as something
                    # the model could have written.
                    contract.setdefault("functional_roles", []).append(recovered)
                    declared.add(participant)
                    trace.append({
                        "code": "UNDECLARED_FIXED_ANCHOR_REFERENCE_DECLARED",
                        "collection": collection, "element_id": entry.get("id"),
                        "participant": participant, "anchor_family": family,
                        "provenance": "FM_EXPLICIT_REFERENCE",
                    })
            people = [p for p in deduplicated
                       if p not in declared and _UNDECLARED_NON_PARTICIPANT.match(str(p).strip())]
            if people:
                deduplicated = [p for p in deduplicated if p not in people]
                trace.append({
                    "code": "UNDECLARED_NON_PARTICIPANT_REMOVED",
                    "collection": collection, "element_id": entry.get("id"),
                    "removed_participants": people,
                    "provenance": "FM_REFERENCE_TO_A_NON_PARTICIPANT",
                })
            echoes = [p for p in deduplicated
                      if p not in declared and _names_only_wire_schema_fields(p)]
            if echoes:
                deduplicated = [p for p in deduplicated if p not in echoes]
                trace.append({
                    "code": "WIRE_SCHEMA_FIELD_ECHO_PARTICIPANT_REMOVED",
                    "collection": collection, "element_id": entry.get("id"),
                    "removed_participants": echoes,
                })
            item = deepcopy(entry)
            item["participant_roles"] = deduplicated
            if len(deduplicated) < minimum:
                unusable.append({
                    "element_kind": "relation" if collection == "functional_relations" else "operation",
                    "id": entry.get("id"),
                    "relation": entry.get("relation"),
                    "operation": entry.get("operation"),
                    "participant_roles": deduplicated,
                    "raw_participant_roles": participants,
                    "reason": "TOO_FEW_DISTINCT_PARTICIPANTS_TO_STATE_A_RELATION",
                })
                trace.append({
                    "code": "STRUCTURALLY_UNUSABLE_ELEMENT_CARRIED_OUT",
                    "collection": collection, "element_id": entry.get("id"),
                    "participant_roles": deduplicated,
                })
                continue
            kept.append(item)
        contract[collection] = kept
    if unusable:
        normalized["structurally_unusable_elements"] = unusable

    # The same region declared twice under the same id is one region.  The
    # model's own inspection order lists it once, which is the model saying so,
    # and rejecting the contract over the repeated declaration discarded every
    # semantic in it.  Only an exact repeat of an id is collapsed; the first
    # declaration is kept, and anything it said differently the second time is
    # recorded rather than merged.
    guidance = normalized.get("observation_guidance", {})
    regions = guidance.get("inspectable_regions", []) or []
    seen: dict[str, Any] = {}
    deduplicated_regions = []
    for region in regions:
        if not isinstance(region, dict):
            deduplicated_regions.append(region)
            continue
        region_id = str(region.get("id"))
        if region_id in seen:
            trace.append({
                "code": "DUPLICATE_INSPECTION_REGION_DECLARATION_COLLAPSED",
                "region_id": region_id,
                "kept_declaration": seen[region_id],
                "discarded_declaration": region,
            })
            continue
        seen[region_id] = region
        deduplicated_regions.append(region)
    if len(deduplicated_regions) != len(regions):
        guidance["inspectable_regions"] = deduplicated_regions


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
                if len(participants) < 2 and is_non_physical_operation_phrase(
                    str(operation.get("operation", "")), participants):
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
    for unknown_role in sorted(set(visible) - set(roles)):
        trace.append({
            "code": "UNDECLARED_OBSERVATION_GUIDANCE_REMOVED",
            "role_id": unknown_role,
            "provenance": "FM_OBSERVATION_GUIDANCE_ONLY",
        })
        visible.pop(unknown_role, None)
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
    _repair_structural_wire_noise(normalized, trace, domain)
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


# Wording that says where things *currently* are, as opposed to where the task
# must put them.  A bare "storage" or "support surface" is not enough: a
# workbench a tool must be left on is a destination, and setting it aside as
# present state would drop a requirement.
_CURRENT_LOCATION_WORDING = re.compile(
    r"\b(initial|initially|currently|starting position|staging|"
    r"current (?:location|position|place|resting)|present location|source location|"
    r"(?:in|from) storage|storage (?:location|position|area|region)|"
    r"stored (?:on|in|at)|where .{0,20}(?:currently|already) (?:sits?|is|are))\b",
    re.IGNORECASE,
)

_QUANTIFIED_CONTEXT_TEXT = re.compile(
    r"\b(both|all|pair|each|between(?:\s+(?:the\s+)?two)?|accessible\s+from\s+both)\b",
    re.IGNORECASE,
)


def _role_has_candidate(hypotheses: Mapping[str, Any], role_id: str, candidate: str) -> bool:
    hypothesis = hypotheses.get(role_id)
    return bool(hypothesis and candidate in hypothesis.canonical_role_candidates)


def _aggregate_explicit_binary_seat_access(
    *,
    domain: str,
    relations: list[Mapping[str, Any]],
    operations: list[Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Conjoin two explicit per-seat accessibility edges into one pair edge.

    This is intentionally stricter than merely noticing two seat roles.  The FM
    must assert the same payload accessible to each of exactly two explicit
    seating roles and must explicitly move that payload to one uniquely typed
    shared support.  Thus the support and both context members all come from the
    FM graph; no scene or implicit seating context is introduced.
    """
    copied = [deepcopy(dict(item)) for item in relations]
    if domain != "living_room":
        return copied, []

    by_payload: dict[str, list[tuple[int, str]]] = {}
    for index, relation in enumerate(copied):
        participants = list(relation.get("participant_roles", ()))
        meanings = extract_relation_semantic_candidates(domain, str(relation.get("relation", "")))
        if len(participants) != 2 or not any(
            item.predicate_name == "ACCESSIBLE_FROM_BOTH_SEATS" for item in meanings
        ):
            continue
        payloads = [p for p in participants if _role_has_candidate(hypotheses, p, "REMOTE")]
        seats = [p for p in participants if _role_has_candidate(hypotheses, p, "SEATING_POSITION")]
        if len(payloads) == 1 and len(seats) == 1:
            by_payload.setdefault(payloads[0], []).append((index, seats[0]))

    consumed: set[int] = set()
    additions: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for payload, members in sorted(by_payload.items()):
        unique_seats = sorted({seat for _, seat in members})
        if len(members) != 2 or len(unique_seats) != 2:
            continue
        supports = {
            participant
            for operation in operations
            if payload in operation.get("participant_roles", ())
            and re.search(r"\b(move|transfer|relocate|place|position|transport)\b",
                          re.sub(r"[_-]+", " ", str(operation.get("operation", ""))), re.I)
            for participant in operation.get("participant_roles", ())
            if _role_has_candidate(hypotheses, participant, "SHARED_REMOTE_REGION")
        }
        if len(supports) != 1:
            continue
        source_ids = [str(copied[index].get("id")) for index, _ in members]
        aggregate_id = "__".join(source_ids)
        additions.append({
            "id": aggregate_id,
            "relation": "accessible from both",
            "participant_roles": [next(iter(supports)), *unique_seats],
            "required": all(bool(copied[index].get("required")) for index, _ in members),
            "source_relation_ids": source_ids,
        })
        consumed.update(index for index, _ in members)
        trace.append({
            "code": "EXPLICIT_BINARY_RELATION_SET_CONJUNCTION",
            "source_relation_ids": source_ids,
            "payload_role": payload,
            "support_role": next(iter(supports)),
            "member_raw_roles": unique_seats,
            "provenance": "FM_EXPLICIT_RELATIONS_AND_OPERATION",
        })
    return [item for index, item in enumerate(copied) if index not in consumed] + additions, trace


def _explicit_seating_context(
    *,
    domain: str,
    entry: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
    canonical: dict[str, Any],
    roles_by_id: dict[str, Mapping[str, Any]],
    bundles: dict[tuple[str, str], str],
    source_kind: str,
) -> tuple[dict[str, Any], bool]:
    """Replace an explicitly enumerated two-seat set with a runtime context role.

    This adapter is intentionally limited to Living Room quantified accessibility
    semantics.  It never searches the scene for omitted seats: both member role
    IDs must occur in the FM participant set and must type as seating positions.
    """
    item = deepcopy(dict(entry))
    participants = list(item.get("participant_roles", ()))
    phrase = str(item.get("relation") or item.get("operation") or "")
    quantified = bool(_QUANTIFIED_CONTEXT_TEXT.search(re.sub(r"[_-]+", " ", phrase)))
    if domain != "living_room" or not quantified:
        return item, False

    seats = [p for p in participants if _role_has_candidate(hypotheses, p, "SEATING_POSITION")]
    if len(seats) != 2:
        # Not an enumerated two-seat set.  That is a statement this adapter has
        # nothing to say about, not a contradictory contract: aborting the whole
        # conversion discarded every other coherent relation and operation the
        # model had expressed, over one phrase.  Decline and let the general
        # n-ary handling and completeness checking decide.
        return item, False
    members = tuple(sorted(seats))
    bundle_id = bundles.get(members)
    if bundle_id is None:
        suffix = "__".join(re.sub(r"[^a-zA-Z0-9_]+", "_", member).strip("_") for member in members)
        bundle_id = f"fm_context_set__{suffix}"
        bundles[members] = bundle_id
        role = {
            "id": bundle_id,
            "entity_kind": "FIXED_TARGET",
            "function": "explicit two-seat seating pair context",
            "description": "Runtime context bundle derived only from explicitly enumerated FM seating roles.",
            "required_count": 1,
            "binding_policy": "SHARED",
            "candidate_categories": ["SEATING_CONTEXT"],
            "required_properties": [],
            "visible_candidates": [],
            "context_set_members": list(members),
            "provenance": "EXPLICIT_CONTEXT_SET_CANONICALIZATION",
        }
        canonical["functional_roles"].append(role)
        roles_by_id[bundle_id] = role

    remaining = [p for p in participants if p not in seats]
    support = [p for p in remaining if any(
        _role_has_candidate(hypotheses, p, candidate)
        for candidate in ("SHARED_REMOTE_REGION", "PERSONAL_CUP_SAUCER_REGION")
    )]
    payload = [p for p in remaining if _role_has_candidate(hypotheses, p, "REMOTE")]
    if source_kind == "relation":
        meanings = extract_relation_semantic_candidates(domain, phrase)
        relation_family = next(
            (m.predicate_name for m in meanings if m.predicate_name in {
                "ACCESSIBLE_FROM_BOTH_SEATS", "SITUATED_BETWEEN",
            }),
            "SITUATED_BETWEEN" if re.fullmatch(r"\s*between\s*", re.sub(r"[_-]+", " ", phrase), re.I) else None,
        )
        if relation_family is None:
            # A quantified phrase whose relation family this adapter does not
            # recognise is left to the general handling below.
            return item, False
        if (
            len(support) != 1
            or len(remaining) not in {1, 2}
            or (len(remaining) == 2 and (relation_family != "ACCESSIBLE_FROM_BOTH_SEATS" or len(payload) != 1))
        ):
            return item, False
        item["participant_roles"] = [support[0], bundle_id]
        item["relation"] = (
            "situated between" if relation_family == "SITUATED_BETWEEN" else item["relation"]
        )
    else:
        if len(remaining) != 2 or len(support) != 1 or len(payload) != 1:
            return item, False
        item["participant_roles"] = [payload[0], support[0], bundle_id]

    item["explicit_participant_roles"] = participants
    item["explicit_context_set_id"] = bundle_id
    canonical.setdefault("explicit_context_sets", []).append({
        "code": "EXPLICIT_CONTEXT_SET_CANONICALIZATION",
        "source_kind": source_kind,
        "source_id": item.get("id"),
        "runtime_role": "SEATING_PAIR",
        "bundle_role_id": bundle_id,
        "member_raw_roles": list(members),
        "primary_raw_roles": remaining,
        "provenance": "FM_EXPLICIT_SEMANTIC" if source_kind == "relation" else "FM_EXPLICIT_OPERATION",
    })
    return item, True


def _relation_options(domain: str, relation: Mapping[str, Any], hypotheses: Mapping[str, Any]) -> list[dict[str, Any]]:
    left, right = relation["participant_roles"]
    options: list[dict[str, Any]] = []
    for semantic in extract_relation_semantic_candidates(domain, relation["relation"]):
        pairs = relation_canonical_role_pairs(domain, semantic.predicate_name, semantic.category)
        if semantic.category == "TASK_EFFECT_SEMANTICS" and not pairs:
            # A stated end state relates two *different* kinds of participant:
            # something carries, and something is carried.  The declared family
            # signature says which, and only that keeps a distributive sentence
            # honest.  Requiring merely that one endpoint *could* carry was not
            # enough -- with "the cups and the soup bowls are placed on the
            # table" both cups and bowls can carry, so the pair (cup, bowl) was
            # admitted and the runtime recorded a claim that the cup goes on
            # the bowl, which the model never made.
            pairs = set()
        for subject_raw, object_raw in ((left, right), (right, left)):
            subject_types = set(hypotheses[subject_raw].canonical_role_candidates)
            object_types = set(hypotheses[object_raw].canonical_role_candidates)
            if any(s in subject_types and o in object_types for s, o in pairs):
                options.append({
                    "subject_role": subject_raw, "object_role": object_raw,
                    "predicate": semantic.predicate_name, "category": semantic.category,
                })
    return [dict(row) for row in sorted({tuple(sorted(item.items())) for item in options})]


def _contextual_participants(
    domain: str,
    operation: Mapping[str, Any],
    participants: Sequence[str],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
) -> set[str]:
    """Participants that may be set aside to make an over-specified operation fit.

    Being a REGION or a FIXED_TARGET was the old test, and it is not enough: a
    seating anchor, a fastening target and a serving destination are all
    non-objects and all essential, and dropping one silently removes a
    requirement.  A participant may only be set aside when something says it is
    context rather than a participant of the change: the interpreter already
    elided it as present state, its own wording describes where things
    currently sit, it types into a family this domain recognizes but realizes
    with no role, or the runtime could type it as nothing at all.
    """
    from .semantic_typing import context_only_families, role_text_scopes, canonical_role_family
    from .system_context_registry import get_domain_planner_context_constants
    planner_constants = set(get_domain_planner_context_constants(domain))
    elided = set(operation.get("current_state_context_roles", ()) or ())
    contextual: set[str] = set()
    context_families = context_only_families(domain)
    for rid in participants:
        role = roles_by_id.get(rid) or {}
        candidates = tuple(getattr(hypotheses.get(rid), "canonical_role_candidates", ()) or ())
        families = {canonical_role_family(domain, candidate) for candidate in candidates}
        text = " ".join(value for value in role_text_scopes(dict(role)).values() if value)
        # A participant the runtime holds only as a fixed place the planner owns
        # -- a work surface, a serving area -- is not a functional participant
        # that grounding selects.  The model names it as the coarse whereabouts
        # of a change ("attach the component to the workbench"), and refusing
        # the operation on that account lost a fastening the model plainly
        # expressed.  Set aside only as a fallback, so an operation that really
        # does act on that place, such as leaving the tool on the bench, keeps
        # it as its own target.
        planner_context_only = bool(candidates) and all(
            candidate in planner_constants for candidate in candidates
        )
        if (
            rid in elided
            or not candidates
            or planner_context_only
            or (context_families and families and families <= set(context_families))
            or _CURRENT_LOCATION_WORDING.search(text)
        ):
            contextual.add(rid)
    return contextual


def _resolve_operation_slots_with_subsets(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Resolve capability slots, falling back to subsets of the named participants.

    The model routinely names a participant the capability has no slot for, as in
    placing a bowl on a table *with* a utensil: the placement itself takes a
    payload and a destination, and the utensil is context rather than a third
    slot.  Refusing the whole operation because one extra participant was named
    discards an operation the runtime can perfectly well execute.

    Subsets are tried largest first.  If several subsets of the same size resolve,
    the operation has more than one legal reading -- typically because two named
    participants are each plausible in the same slot.  Rather than choosing one,
    which would invent a reading the model did not give, or discarding them all,
    which throws away an operation the runtime can execute, every legal slot
    assignment is carried forward.  Grounding enumerates them against the
    observed scene, and the objects that are actually present decide.

    Returns (options, participants_used, participants_left_as_context).
    """
    participants = list(dict.fromkeys(operation["participant_roles"]))
    options = resolve_v3_operation_slots(domain, operation, roles_by_id, hypotheses)
    if options:
        return options, participants, []
    droppable = _contextual_participants(domain, operation, participants, roles_by_id, hypotheses)
    if not droppable:
        return [], participants, []
    for size in range(len(participants) - 1, 1, -1):
        merged: list[dict[str, Any]] = []
        used: list[str] = []
        for subset in combinations(participants, size):
            if any(p not in subset and p not in droppable for p in participants):
                continue
            probe = {**dict(operation), "participant_roles": list(subset)}
            subset_options = resolve_v3_operation_slots(domain, probe, roles_by_id, hypotheses)
            if not subset_options:
                continue
            for option in subset_options:
                if option not in merged:
                    merged.append(option)
            for participant in subset:
                if participant not in used:
                    used.append(participant)
        if merged:
            ordered_used = [p for p in participants if p in used]
            return merged, ordered_used, [p for p in participants if p not in used]
    return [], participants, []


_SLOT_FIELD_TYPES: tuple[tuple[str, str, str], ...] = (
    ("tool_role", "source_role", "source_type"),
    ("target_role", "target_role", "target_type"),
    ("context_role", "anchor_role", "anchor_type"),
)
_SLOT_FIELD_TO_SLOT = {"tool_role": "source", "target_role": "target", "context_role": "anchor"}


def _is_abstract_directive_over_unrepresented_participant(
    operation: Mapping[str, Any], hypotheses: Mapping[str, Any]
) -> bool:
    """Whether this is a desired end state rather than an operation to execute.

    "Serve the soup to the person", "hand the bowl to the recipient": the phrase
    states what must be true when the task is done, and one of the participants
    is something the runtime represents in no form at all -- a person.  Reading
    that as a motion means inventing a physical endpoint for it, which is how a
    bowl came to be moved into a person.

    Both halves are required.  A phrase built from a directive whose
    participants the runtime *can* represent is left alone, because the model
    routinely words a real capability that way -- "provide the eating utensil"
    is how it asks for the utensil placement -- and such an operation has
    already been matched to its capability before this is consulted.
    """
    phrase = str(operation.get("operation") or operation.get("function") or "")
    if not leads_with_abstract_task_directive(phrase):
        return False
    return any(
        _runtime_says_nothing_about(hypotheses.get(participant))
        for participant in operation.get("participant_roles", ()) or ()
    )


def _runtime_says_nothing_about(hypothesis: Any) -> bool:
    """Whether the runtime holds this participant in no form it can act on.

    Two ways that happens: nothing typed it at all, or nothing narrowed it and
    every canonical role is still nominally possible -- which is not a reading
    of the participant but the absence of one.  A person is the usual case: the
    runtime models where a person sits, never the person.
    """
    if hypothesis is None:
        return True
    candidates = getattr(hypothesis, "canonical_role_candidates", ()) or ()
    if not candidates:
        return True
    return getattr(hypothesis, "status", "") == "UNCONSTRAINED_ROLE_TYPE" and len(candidates) > 3


def _split_generic_participant_per_operation_form(
    domain: str,
    canonical: dict[str, Any],
    roles_by_id: dict[str, Any],
    hypotheses: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Give one generically-worded participant a role per form its operations need.

    The prompt asks the model to keep apart participants whose jobs differ -- a
    support belonging to one person is not a support deliberately shared, an
    implement used on a drink is not the one a person eats with -- and it often
    does not.  It writes one "surface" role and uses it in both placements, or
    one "spoon" role counted four times for stirring and for eating.

    One raw role became one canonical role, so the first reading won and the
    second operation was rejected as incompatible with its own endpoints -- a
    shared-control placement told that its support was the personal one.

    Each operation says which form it needs, and when the role's own wording
    singled out none of them, honouring both is reading the model rather than
    correcting it.  The raw role keeps the first form and is recorded as the
    witness for the others.  Nothing is invented: every form supplied is one an
    expressed operation demanded, all the forms belong to the one family the
    role's own wording put it in, and a role that named its own form
    specifically is left alone.
    """
    demands: dict[str, list[tuple[dict[str, Any], str, str, str]]] = {}
    for group in canonical.get("interaction_groups", ()) or ():
        for field, option_key, type_key in _SLOT_FIELD_TYPES:
            rid = group.get(field)
            if not rid or rid not in roles_by_id:
                continue
            types = {
                row.get(type_key) for row in group.get("v3_slot_assignments", ()) or ()
                if row.get(option_key) == rid and row.get(type_key)
            }
            if len(types) != 1:
                continue
            demands.setdefault(rid, []).append((group, field, option_key, types.pop()))

    synthesized: list[dict[str, Any]] = []
    for rid, rows in sorted(demands.items()):
        wanted = {want for _group, _field, _key, want in rows}
        if len(wanted) < 2:
            continue
        hypothesis = hypotheses.get(rid)
        candidates = set(getattr(hypothesis, "canonical_role_candidates", ()) or ())
        role_families, _ = detect_role_families(
            domain, dict(roles_by_id.get(rid) or {}))
        # Every form demanded has to be a form this participant could actually
        # take: either its resolved type admits it, or its own functional
        # families do.  The families matter because a participant may be settled
        # globally and still be demanded in another form by an operation that
        # seated it there -- one "workbench" role is the site fastened into for
        # the fastening and the surface for the tool return, and its families
        # say both.  Those demands come from capability signatures the
        # operations themselves resolved, which is evidence about this
        # participant and not a guess.
        admissible = {
            want for want in wanted
            if want in candidates
            or (canonical_role_family(domain, want) in role_families
                and canonical_role_family(domain, want) != "OTHER")
        }
        if admissible != wanted:
            continue
        # Two forms of one kind, or two operations that each seated it
        # differently.  A single operation demanding two forms of one
        # participant is a contradiction rather than a projection.
        families = {canonical_role_family(domain, want) for want in wanted}
        distinct_operations = {str(group.get("id")) for group, _f, _k, _w in rows}
        if (len(families) != 1 or families == {"OTHER"}) and len(distinct_operations) < 2:
            continue
        # Which form stays on the raw role matters, because everything else the
        # model wrote about that participant -- every relation naming it -- is
        # still read against it.  Keep the form the participant already resolved
        # to globally, so those statements go on meaning what they meant; the
        # other forms are the ones that move.  Choosing alphabetically moved a
        # workbench off the site it is fastened into, and the relation saying the
        # component must fit that site lost its endpoint.
        resolved_globally = sorted(candidates & wanted)
        keep = resolved_globally[0] if resolved_globally else sorted(wanted)[0]
        # The raw role's count was the aggregate over every functional use the
        # model bundled into it -- one "spoon" role counted four times, for two
        # stirrings and two servings.  Each projected form gets the count of the
        # operations that induced it, and the form left on the raw role has to be
        # partitioned the same way.  It used to keep the aggregate, so the form
        # that stayed behind demanded four of itself and was reported short of
        # objects while the other form asked for two.
        keep_rows = [row for row in rows if row[3] == keep]
        if keep_rows:
            keep_slot = _SLOT_FIELD_TO_SLOT[keep_rows[0][1]]
            keep_applications = 0
            for group, _field, _key, _want in keep_rows:
                try:
                    keep_applications += int(group.get("required_target_count", 1) or 1)
                except (TypeError, ValueError):
                    keep_applications += 1
            keep_count, keep_policy = _slot_cardinality(
                domain, keep_slot, keep, max(1, keep_applications))
            raw_role = roles_by_id.get(rid)
            if raw_role is not None and (
                raw_role.get("required_count") != keep_count
                or raw_role.get("binding_policy") != keep_policy
            ):
                canonical.setdefault("functional_constraint_interpretation", []).append({
                    "code": "AGGREGATE_PARTICIPANT_COUNT_PARTITIONED_BY_FUNCTIONAL_FORM",
                    "fm_role": rid,
                    "forms_demanded": sorted(wanted),
                    "form_kept_on_the_fm_role": keep,
                    "declared_required_count": raw_role.get("required_count"),
                    "declared_binding_policy": raw_role.get("binding_policy"),
                    "applications_of_the_kept_form": keep_applications,
                    "resolved_required_count": keep_count,
                    "resolved_binding_policy": keep_policy,
                })
                raw_role["required_count"] = keep_count
                raw_role["binding_policy"] = keep_policy
        for group, field, option_key, want in rows:
            if want == keep:
                continue
            slot = _SLOT_FIELD_TO_SLOT[field]
            try:
                count = int(group.get("required_target_count", 1) or 1)
            except (TypeError, ValueError):
                count = 1
            role_id = f"fm_form__{rid}__{want}"
            if role_id in roles_by_id:
                continue
            required_count, binding_policy = _slot_cardinality(domain, slot, want, count)
            role = {
                "id": role_id,
                "entity_kind": _slot_role_kind(domain, want),
                "function": (
                    f"{canonical_role_family(domain, want).lower().replace('_', ' ')} participant "
                    f"the expressed {group.get('function')!r} operation requires in this form"
                ),
                "description": (
                    "The FM named one participant for several operations that need different "
                    "forms of it; this holds the form this operation requires, and the FM's own "
                    "role is recorded as the witness."
                ),
                "required_count": required_count,
                "binding_policy": binding_policy,
                "candidate_categories": [],
                "required_properties": [],
                "visible_candidates": [],
                "canonical_role": want,
                "provenance": ABSTRACT_ROLE_PROVENANCE,
                "induced_by_operation": group.get("id"),
                "induced_slot": slot,
                "fm_witness_role": rid,
            }
            roles_by_id[role["id"]] = role
            canonical["functional_roles"].append(role)
            synthesized.append(role)
            group[field] = role["id"]
            for row in group.get("v3_slot_assignments", ()) or ():
                if row.get(option_key) == rid:
                    row[option_key] = role["id"]
            for key in ("v3_participant_roles", "v3_explicit_participant_roles"):
                values = group.get(key) or []
                group[key] = [role["id"] if value == rid else value for value in values]
            canonical.setdefault("generic_participant_form_realizations", []).append({
                "code": "FM_GENERIC_PARTICIPANT_REALIZED_PER_OPERATION_FORM",
                "fm_role": rid,
                "form_role_id": role["id"],
                "operation_id": group.get("id"),
                "slot": slot,
                "canonical_role": want,
                "forms_demanded": sorted(wanted),
                "form_kept_on_the_fm_role": keep,
                "provenance": ABSTRACT_ROLE_PROVENANCE,
            })
    if synthesized:
        canonical["functional_constraint_interpretation"].extend(
            canonical.get("generic_participant_form_realizations", ()))
    return synthesized


def _route_relations_to_operation_specific_forms(
    domain: str, canonical: dict[str, Any]
) -> list[dict[str, Any]]:
    """Point a relation at the form of the participant its own operation uses.

    When one generically-worded participant is realized per operation -- one
    "surface" role standing for the support belonging to a person and the
    support deliberately shared -- the relations the model wrote about it still
    name the raw role, and the raw role keeps only one of the forms.  So "the
    control is supported by the surface" was checked against the *personal*
    support and reported as a relation the runtime cannot represent, while the
    control's own operation had already been compiled against the shared one.

    A relation is re-pointed only when exactly one form's operation also
    involves the relation's other participants, which is the relation and the
    operation being about the same thing.  Where several forms would fit, or
    none, the relation is left as the model wrote it and fails closed.
    """
    realizations = canonical.get("generic_participant_form_realizations") or ()
    if not realizations:
        return []
    forms: dict[str, list[tuple[str, str]]] = {}
    for row in realizations:
        form_role = row.get("form_role_id")
        if form_role:
            forms.setdefault(str(row["fm_role"]), []).append(
                (str(row.get("operation_id")), str(form_role)))
    if not forms:
        return []
    operation_participants = {
        str(group.get("id")): set(group.get("v3_participant_roles", ()) or ())
        | set(group.get("v3_witness_roles", ()) or ())
        for group in canonical.get("interaction_groups", ()) or ()
    }
    rerouted: list[dict[str, Any]] = []
    for relation in canonical.get("functional_relations", ()) or ():
        endpoints = [relation.get("subject_role"), relation.get("object_role")]
        for index, endpoint in enumerate(endpoints):
            if endpoint not in forms:
                continue
            others = {value for position, value in enumerate(endpoints) if position != index}
            matching = {
                form_role for operation_id, form_role in forms[endpoint]
                if others and others <= operation_participants.get(operation_id, set())
            }
            if len(matching) != 1:
                continue
            form_role = matching.pop()
            field = "subject_role" if index == 0 else "object_role"
            relation[field] = form_role
            for option in relation.get("v3_orientation_options", ()) or ():
                for key in ("subject_role", "object_role"):
                    if option.get(key) == endpoint:
                        option[key] = form_role
            rerouted.append({
                "code": "RELATION_READ_AGAINST_THE_FORM_ITS_OWN_OPERATION_USES",
                "relation_id": relation.get("id"),
                "relation": relation.get("relation"),
                "fm_role": endpoint,
                "form_role": form_role,
                "other_participants": sorted(others),
            })
    if rerouted:
        canonical["functional_constraint_interpretation"].extend(rerouted)
    return rerouted


def _decompose_nary_relation(
    domain: str, relation: Mapping[str, Any], hypotheses: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Split a relation over more than two participants into legal binary edges.

    The wire contract admits two to four participants and the prompt asks the
    model to name every member a quantified phrase covers, so n-ary relations are
    expected rather than exceptional.  Those that match a declared context-set
    family are handled by that family.  For the rest, a relation holding over a
    set is represented as the conjunction of the pairwise edges that have a legal
    reading under the same predicate, which preserves the expressed meaning
    without inventing an orientation.  Pairs with no legal reading are simply not
    emitted; if no pair has one the caller still fails.
    """
    participants = list(dict.fromkeys(relation["participant_roles"]))
    # A binary relation whose predicate the runtime does not recognise is carried
    # through and handled downstream as context.  An n-ary one is decomposed the
    # same way rather than aborting the contract, so arity alone never decides
    # whether an uninterpretable relation is fatal.
    interpretable = bool(extract_relation_semantic_candidates(domain, relation["relation"]))
    decomposed: list[dict[str, Any]] = []
    for index, left in enumerate(participants):
        for right in participants[index + 1:]:
            if left not in hypotheses or right not in hypotheses:
                continue
            probe = {**dict(relation), "participant_roles": [left, right]}
            if interpretable and not _relation_options(domain, probe, hypotheses):
                continue
            decomposed.append({
                **dict(relation),
                "id": f"{relation['id']}__{left}__{right}",
                "participant_roles": [left, right],
                "source_relation_ids": list(
                    relation.get("source_relation_ids", ()) or [relation["id"]]
                ),
            })
    return decomposed


def _reconciled_operation_count(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
    all_operations: Sequence[Mapping[str, Any]],
) -> int:
    """How many times the operation runs, given what its participant declares.

    ``operation_count`` says how many times the change happens and
    ``required_count`` says how many times the task needs a participant, so the
    two are about the same number whenever one application consumes one instance
    of that participant.  Where the model's two answers disagree, the larger is
    the one that keeps every requirement it stated: lowering the participant's
    count would discard part of what it asked for.

    Deliberately narrow.  A participant raises the count only when all of the
    following hold, and otherwise the model's own number stands:

      * it is a role the model itself declared, not a slot the runtime
        synthesized -- a synthesized slot takes its own count from this
        operation, so reading it back would be circular;
      * its binding policy is DISTINCT, which is the model saying each occasion
        needs its own instance, so occasions and instances are the same number;
      * no other operation names it, so its instances cannot be accounted for by
        the applications of a different operation.  This is what keeps a seating
        reference shared by two placements from multiplying either of them.

      * every capability the phrase names can only seat it as the thing acted
        on, never as an implement, a support or a reference.  This is the
        condition that makes occasions and instances the same number: one
        application consumes one of what it acts on, while a tool or a fixed
        reference serves any number of them.  Without it, a contract declaring
        two workpieces "being joined" -- the thing a single fastener is driven
        through -- asked for two fastenings, and one declaring two seats asked
        for the remote control to be placed twice.
    """
    try:
        declared = int(operation.get("operation_count", 1) or 1)
    except (TypeError, ValueError):
        declared = 1
    participants = list(dict.fromkeys(operation.get("participant_roles", ()) or ()))
    capabilities = extract_operation_semantic_candidates(
        domain, str(operation.get("operation", "")), participants)
    if not capabilities:
        return declared
    acted_on: set[str] = set()
    other_slots: set[str] = set()
    for capability in capabilities:
        acted_on |= set(capability.allowed_target_roles)
        other_slots |= set(capability.allowed_source_roles)
        other_slots |= set(capability.allowed_anchor_roles)
    needed = declared
    for participant in participants:
        role = roles_by_id.get(participant)
        if not role or role.get("provenance") in RUNTIME_AUTHORED_ROLE_PROVENANCE:
            continue
        if str(role.get("binding_policy")) != "DISTINCT":
            continue
        candidates = set(
            getattr(hypotheses.get(participant), "canonical_role_candidates", ()) or ())
        if not candidates or not candidates <= acted_on or candidates & other_slots:
            continue
        try:
            count = int(role.get("required_count", 1) or 1)
        except (TypeError, ValueError):
            continue
        if count <= needed:
            continue
        if any(
            other.get("id") != operation.get("id")
            and participant in (other.get("participant_roles") or ())
            for other in all_operations
        ):
            continue
        needed = count
    return needed


def resolve_v3_operation_slots(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
) -> list[dict[str, Any]]:
    # An operation naming a participant that is not a declared role cannot be
    # seated.  Strict wire validation rejects such a contract before this point;
    # reaching here anyway used to raise a bare KeyError instead of reporting an
    # operation the runtime could not seat.
    participants_named = list(dict.fromkeys(operation.get("participant_roles", ()) or ()))
    if any(p not in hypotheses or p not in roles_by_id for p in participants_named):
        return []
    phrase = str(operation["operation"])
    capabilities = extract_operation_semantic_candidates(domain, phrase, participants_named)
    # The model often names an operation with a bare noun or a generic verb --
    # "manipulation", "placement", "combine" -- which identifies no capability by
    # wording alone.  The participants still do: a driver, a fastener and a fixed
    # target admit exactly one capability signature.  So when the phrase says
    # nothing, every capability is considered and the result is accepted only if
    # exactly one of them can seat these participants.  A phrase that is not a
    # physical operation at all is left alone, and an ambiguous fit is refused
    # rather than guessed.
    inferred_from_participants = False
    if not capabilities and not is_non_physical_operation_phrase(phrase, participants_named):
        capabilities = tuple(get_robot_capabilities(domain))
        inferred_from_participants = True
    participants = tuple(operation["participant_roles"])
    resolved = _slot_options_for_capabilities(
        domain, operation, roles_by_id, hypotheses, capabilities, participants)
    if not resolved and not inferred_from_participants and not is_non_physical_operation_phrase(
        phrase, participants_named
    ):
        # A reading the participants cannot support is not a reading.  The model
        # coordinates two acts in one phrase -- "place soup and add utensil" --
        # and the word the wording offers is the one belonging to the other act,
        # so honouring it alone seats nothing.  Falling back to what the
        # participants themselves identify is the same admission as for a phrase
        # that said nothing: accepted only when exactly one capability can seat
        # them, so nothing is guessed.
        resolved = _slot_options_for_capabilities(
            domain, operation, roles_by_id, hypotheses,
            tuple(get_robot_capabilities(domain)), participants)
        inferred_from_participants = bool(resolved)
    if inferred_from_participants:
        # Inference is only admissible when the participants pick out a single
        # capability.  If several could seat them, the operation is genuinely
        # undetermined and choosing one would invent semantics the model never
        # expressed.
        identified = {row.get("capability_id") for row in resolved}
        if len(identified) != 1:
            return []
        for row in resolved:
            row["capability_provenance"] = "INFERRED_FROM_PARTICIPANT_SIGNATURE"
    return resolved


def _slot_options_for_capabilities(
    domain: str,
    operation: Mapping[str, Any],
    roles_by_id: Mapping[str, Mapping[str, Any]],
    hypotheses: Mapping[str, Any],
    capabilities: Sequence[Any],
    participants: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Every legal seating of these participants in any of these capabilities."""
    options: list[dict[str, Any]] = []
    for capability in capabilities:
        # A capability whose physical preconditions check an anchor is not
        # executable without one.  Living Room used to be allowed to drop the
        # anchor whenever the model named only two participants, which produced
        # a support group with no seating context and then discarded it a stage
        # later for exactly that reason.  The anchor is supplied by slot
        # completion instead, on the model's own evidence, or not at all.
        anchor_roles = capability_anchor_roles(domain, capability)
        slot_names = ("source", "target", "anchor") if anchor_roles else ("source", "target")
        if len(participants) != len(slot_names):
            continue
        allowed = {
            "source": set(capability.allowed_source_roles),
            "target": set(capability.allowed_target_roles),
            "anchor": set(anchor_roles),
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


# ---------------------------------------------------------------------------
# Requirement provenance: why is this element required?
# ---------------------------------------------------------------------------
# The single FM call sees the instruction and the images together, so nothing in
# the returned contract says which of the two produced any given element.  That
# distinction matters: an instruction clause may create a task requirement, but a
# visual observation may only propose a candidate, a current location, or a place
# worth searching.  Recovering the distinction after the fact is necessarily
# evidential rather than certain, so it is recorded as evidence and not used to
# silently delete anything.

_PROVENANCE_STOPWORDS = frozenset({
    "each", "every", "with", "their", "them", "they", "this", "that", "these",
    "those", "then", "than", "from", "into", "onto", "over", "under", "near",
    "have", "make", "made", "used", "using", "use", "when", "where", "which",
    "while", "also", "both", "same", "other", "another", "must", "should",
    "will", "would", "there", "here", "some", "any", "one", "two", "item",
    "items", "thing", "things", "object", "objects", "area", "areas", "place",
    "places", "part", "parts", "side", "left", "right", "front", "back",
    "required", "requires", "require", "needed", "needs", "need", "task",
    "instruction", "scene", "visible", "available", "suitable", "appropriate",
    "for", "the", "and", "its", "it",
})


def _provenance_terms(text: str) -> set[str]:
    """Content words usable as evidence of shared reference between two texts."""
    words = re.findall(r"[a-zA-Z]+", str(text).replace("_", " ").lower())
    return {w for w in words if len(w) > 3 and w not in _PROVENANCE_STOPWORDS}


# Words that put what follows in an input position: the instruction is naming
# what a process is carried out *with*, not what it produces.
_INSTRUCTION_INPUT_MARKER = re.compile(
    r"\b(using|use|with|from|out of|made (?:of|from)|consisting of|composed of)\b", re.I
)


def instruction_clauses(instruction: str) -> list[str]:
    """The instruction's clauses, so a term can be read in the company it keeps."""
    parts: list[str] = []
    for sentence in re.split(r"[.;]", str(instruction or "")):
        for clause in re.split(r",\s*(?:and|then|or)\s+|,\s+", sentence):
            clause = clause.strip()
            if clause:
                parts.append(clause)
    return parts


def instruction_names_as_task_input(element_terms: set[str], instruction: str) -> bool:
    """Whether the instruction names this element among some process's inputs.

    "Make each coffee using coffee and water" names two inputs.  "Serve one
    soup" and "serve each soup bowl with its own eating utensil" name a product
    and an accompaniment; neither says the soup is poured from anything.

    That difference is what separates a requirement the instruction created
    from a mechanism the model inferred from a finished result -- a soup
    supply, a cooking vessel -- and it decides whether losing an
    unrepresentable participant loses a requirement or only a proposal.  It is
    read off the instruction alone; no reference graph is consulted.
    """
    if not element_terms:
        return False
    for clause in instruction_clauses(instruction):
        match = _INSTRUCTION_INPUT_MARKER.search(clause)
        if not match:
            continue
        if element_terms & _provenance_terms(clause[match.end():]):
            return True
    return False


def classify_requirement_provenance(
    element_terms: set[str],
    *,
    instruction_terms: set[str],
    participates_in_expressed_operation: bool,
) -> str:
    """Classify why one contract element is required.

    INSTRUCTION_CLAUSE_SUPPORT   the element's own wording shares content with
                                 the instruction, so a clause can account for it.
    DERIVED_FROM_EXPLICIT_OPERATION
                                 no direct instruction wording, but the element
                                 takes part in an operation the FM expressed, so
                                 it is a semantic consequence of that operation.
    OBSERVATION_ONLY             neither: nothing but the images accounts for it.
    """
    if instruction_terms and (element_terms & instruction_terms):
        return "INSTRUCTION_CLAUSE_SUPPORT"
    if participates_in_expressed_operation:
        return "DERIVED_FROM_EXPLICIT_OPERATION"
    return "OBSERVATION_ONLY"


_PLACEMENT_VERB = re.compile(
    r"\b(place|placing|put|set|serve|serving|deliver|deposit|position|return|leave|"
    r"restore|store|move|transfer|bring|hand)\w*\b", re.I
)


def _planner_context_transition(
    domain: str,
    operation: Mapping[str, Any],
    hypotheses: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Whether this operation just moves something to a planner-owned place.

    The symbolic compilers own the transitions to their fixed contexts, so an
    expressed "serve the coffee on the dining table" needs no functional
    operation of its own.  Recognising that here keeps it from being counted as
    an operation the runtime failed to represent.
    """
    from .system_context_registry import get_domain_planner_context_constants
    constants = set(get_domain_planner_context_constants(domain))
    if not constants or not _PLACEMENT_VERB.search(str(operation.get("operation", ""))):
        return None
    destinations, movable = [], []
    for participant in dict.fromkeys(operation.get("participant_roles", ()) or ()):
        candidates = set(getattr(hypotheses.get(participant), "canonical_role_candidates", ()) or ())
        if candidates and candidates <= constants:
            destinations.append(participant)
        elif candidates:
            movable.append(participant)
        else:
            return None
    if len(destinations) != 1 or not movable:
        return None
    return {
        "code": "ABSORBED_INTO_PLANNER_CONTEXT",
        "operation_id": operation.get("id"),
        "raw_operation": operation.get("operation"),
        "planner_context_role": destinations[0],
        "moved_roles": movable,
        "provenance": "FM_EXPLICIT_OPERATION",
    }


# Wording that marks a role as one more of the same kind rather than another
# aspect of one thing.  "Another fixed assembly receiving the component" is the
# model enumerating a second assembly, and merging it into the first would
# delete a requirement it stated.
_ENUMERATES_A_FURTHER_INSTANCE = re.compile(
    r"\b(?:another|other|second|third|fourth|additional|further|extra|"
    r"next|remaining|each|every|both|either)\b", re.I)


def _role_enumerates_a_further_instance(role: Mapping[str, Any]) -> bool:
    text = " ".join(str(role.get(key, "")) for key in ("id", "function", "description"))
    return bool(_ENUMERATES_A_FURTHER_INSTANCE.search(re.sub(r"[_\-]+", " ", text)))


def _anchor_is_a_singular_place(domain: str, anchor: str) -> bool:
    """Whether two roles naming this anchor cannot be two members of a set.

    Living Room declares SEATING_POSITION for one seat and SEATING_PAIR for
    both, so two roles that each read as a seating position may perfectly well
    be the two seats the model enumerated, and merging them would erase the
    enumeration -- which is how this lowering first destroyed an explicit
    seating pair.  Where the domain offers no set form of an anchor's family,
    the place it names is singular and two roles naming it are two descriptions
    of one thing.
    """
    from .operation_slot_completion import FIXED_ANCHOR_SEMANTICS, anchor_semantics
    from .semantic_typing import canonical_role_family

    semantics = anchor_semantics(domain, anchor)
    family = semantics.family if semantics else canonical_role_family(domain, anchor)
    return not any(
        item.domain == domain and item.family == family and item.quantification == "SET"
        for item in FIXED_ANCHOR_SEMANTICS
    )


def _lower_composite_receiving_context(
    domain: str,
    contract: dict[str, Any],
    canonical: dict[str, Any],
    roles_by_id: dict[str, Any],
    hypotheses: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collapse two FM roles that name one fixed place the task acts on.

    The model routinely describes the receiving context of a fastening twice --
    "the object to be fastened" and "the specific spot on the workbench for the
    fastening" -- and the runtime holds that place as one non-manipulable
    reference.  Left as two participants, the operation had four where its
    capability has three slots and was discarded whole, taking the fastening the
    model plainly expressed with it.

    Narrow on purpose.  Only a system fixed anchor, never a selectable asset, so
    this can never merge two things the robot would have had to find separately.
    Only when each role's own reading is that one anchor and nothing else, at
    cardinality one, because the anchor is one place.  And only when the model
    itself named both in one statement, which is what says they are two aspects
    of one thing rather than two participants it enumerated.  Two roles that
    land on the same anchor and are never named together are left colliding.
    """
    from .semantic_typing import canonical_role_family
    from .system_context_registry import get_domain_system_fixed_anchors

    anchors = set(get_domain_system_fixed_anchors(domain))
    by_anchor: dict[str, list[str]] = {}
    for role in contract["functional_roles"]:
        rid = str(role["id"])
        candidates = tuple(getattr(hypotheses.get(rid), "canonical_role_candidates", ()) or ())
        if len(candidates) != 1 or candidates[0] not in anchors:
            continue
        if not _anchor_is_a_singular_place(domain, candidates[0]):
            continue
        try:
            count = int(role.get("required_count", 1))
        except (TypeError, ValueError):
            count = 1
        if count != 1:
            continue
        by_anchor.setdefault(candidates[0], []).append(rid)

    statements = [
        *({"kind": "operation", "id": item.get("id"),
           "text": str(item.get("operation") or ""),
           "participants": set(item.get("participant_roles", ()) or ())}
          for item in contract["operation_pairings"]),
        *({"kind": "relation", "id": item.get("id"),
           "text": str(item.get("relation") or ""),
           "participants": set(item.get("participant_roles", ()) or ())}
          for item in contract["functional_relations"]),
    ]

    lowerings: list[dict[str, Any]] = []
    for anchor, members in sorted(by_anchor.items()):
        if len(members) < 2:
            continue
        keeper, aspects = members[0], members[1:]
        for aspect in aspects:
            together = [
                {"kind": row["kind"], "id": row["id"], "text": row["text"],
                 "participant_roles": sorted(row["participants"])}
                for row in statements
                if {keeper, aspect} <= row["participants"]
            ]
            if not together:
                continue
            if any(_role_enumerates_a_further_instance(roles_by_id.get(rid) or {})
                   for rid in (keeper, aspect)):
                continue
            for item in contract["operation_pairings"]:
                item["participant_roles"] = list(dict.fromkeys(
                    keeper if p == aspect else p for p in item.get("participant_roles", ()) or ()))
            for item in contract["functional_relations"]:
                item["participant_roles"] = list(dict.fromkeys(
                    keeper if p == aspect else p for p in item.get("participant_roles", ()) or ()))
            contract["functional_roles"] = [
                role for role in contract["functional_roles"] if str(role["id"]) != aspect]
            canonical["functional_roles"] = [
                role for role in canonical.get("functional_roles", ())
                if str(role.get("id")) != aspect]
            aspect_role = dict(roles_by_id.pop(aspect, {}))
            keeper_role = dict(roles_by_id.get(keeper) or {})
            keeper_role["required_properties"] = list(dict.fromkeys([
                *(keeper_role.get("required_properties") or ()),
                *(aspect_role.get("required_properties") or ()),
            ]))
            keeper_role["fm_composite_aspect_roles"] = list(dict.fromkeys([
                *(keeper_role.get("fm_composite_aspect_roles") or ()), aspect]))
            roles_by_id[keeper] = keeper_role
            for collection in (contract["functional_roles"], canonical.get("functional_roles", [])):
                for index, role in enumerate(collection):
                    if str(role.get("id")) == keeper:
                        collection[index] = keeper_role
            # A statement whose only two participants were the two aspects was
            # relating one place to itself, and once they are one role it says
            # nothing further about the scene.  It is recorded here rather than
            # dropped quietly, because a relation that disappears without a
            # reason is indistinguishable from one that was not understood.
            absorbed = [
                {"kind": "relation", "id": item.get("id"),
                 "text": str(item.get("relation") or ""),
                 "reason": "BOTH_PARTICIPANTS_ARE_ASPECTS_OF_THE_SAME_ANCHOR"}
                for item in contract["functional_relations"]
                if len(set(item.get("participant_roles", ()) or ())) < 2
            ]
            contract["functional_relations"] = [
                item for item in contract["functional_relations"]
                if len(set(item.get("participant_roles", ()) or ())) >= 2
            ]
            absorbed += [
                {"kind": "operation", "id": item.get("id"),
                 "text": str(item.get("operation") or ""),
                 "reason": "BOTH_PARTICIPANTS_ARE_ASPECTS_OF_THE_SAME_ANCHOR"}
                for item in contract["operation_pairings"]
                if len(set(item.get("participant_roles", ()) or ())) < 2
            ]
            contract["operation_pairings"] = [
                item for item in contract["operation_pairings"]
                if len(set(item.get("participant_roles", ()) or ())) >= 2
            ]
            lowerings.append({
                "code": "COMPOSITE_RECEIVING_CONTEXT_LOWERED_TO_ONE_ANCHOR",
                "canonical_role": anchor,
                "kept_raw_role": keeper,
                "aspect_raw_role": aspect,
                "aspect_role_function": str(aspect_role.get("function", "")),
                "fm_statements_naming_both": together,
                "statements_absorbed_by_the_lowering": absorbed,
                "provenance": "FM_EXPLICIT_SEMANTIC",
            })
    return lowerings


def convert_v3_to_canonical_document(
    v3_doc: Mapping[str, Any], *, domain: str, task_instruction: str = ""
) -> dict[str, Any]:
    canonical = _base_canonical_document(v3_doc)
    contract = v3_doc["task_contract"]
    contract = json.loads(json.dumps(contract))
    roles_by_id: dict[str, Mapping[str, Any]] = {role["id"]: role for role in contract["functional_roles"]}
    hypotheses = build_role_type_hypotheses(domain, canonical)
    composite_lowerings = _lower_composite_receiving_context(
        domain, contract, canonical, roles_by_id, hypotheses)
    if composite_lowerings:
        hypotheses = build_role_type_hypotheses(domain, canonical)

    bundles: dict[tuple[str, str], str] = {}

    relation_inputs, relation_conjunction_trace = _aggregate_explicit_binary_seat_access(
        domain=domain,
        relations=contract["functional_relations"],
        operations=contract["operation_pairings"],
        hypotheses=hypotheses,
    )
    normalized_relations = []
    for relation in relation_inputs:
        if len(relation["participant_roles"]) > 2:
            normalized, handled = _explicit_seating_context(
                domain=domain, entry=relation, hypotheses=hypotheses, canonical=canonical,
                roles_by_id=roles_by_id, bundles=bundles, source_kind="relation",
            )
            if not handled:
                decomposed = _decompose_nary_relation(domain, relation, hypotheses)
                if not decomposed:
                    # A recognised predicate the runtime can give no legal
                    # reading over these participants is one relation it cannot
                    # represent, recorded below with the binary cases.  Aborting
                    # the conversion threw away every other coherent semantic in
                    # the contract over a single phrase.
                    normalized_relations.append({
                        **dict(relation),
                        "participant_roles": list(relation["participant_roles"])[:2],
                        "nary_reading_unavailable": True,
                        "raw_fm_participant_roles": list(relation["participant_roles"]),
                    })
                    continue
                normalized_relations.extend(decomposed)
                continue
            normalized_relations.append(normalized)
        else:
            normalized_relations.append(relation)

    normalized_operations = []
    for operation in contract["operation_pairings"]:
        if len(operation["participant_roles"]) == 4 and domain == "living_room":
            normalized, handled = _explicit_seating_context(
                domain=domain, entry=operation, hypotheses=hypotheses, canonical=canonical,
                roles_by_id=roles_by_id, bundles=bundles, source_kind="operation",
            )
            normalized_operations.append(normalized if handled else operation)
        else:
            normalized_operations.append(operation)
    hypotheses = build_role_type_hypotheses(domain, canonical)

    expressed = collect_expressed_semantics(domain, {
        "task_summary": v3_doc.get("task_summary", ""),
        "functional_roles": contract["functional_roles"],
        "functional_relations": normalized_relations,
        "operation_pairings": normalized_operations,
        "explicit_context_sets": canonical.get("explicit_context_sets", []),
    }, hypotheses)

    def _resolve_or_probe(dom, operation, roles, hyps):
        """Seat an operation, or report which capability its named roles could be.

        The probe introduces nothing.  It exists so the graph join below can look
        for a missing anchor among the FM's own relations, which names the actual
        seating role the model wrote, before slot completion falls back to the
        registered anchor.
        """
        direct = resolve_v3_operation_slots(dom, operation, roles, hyps)
        return direct or probe_named_participant_slots(dom, operation, roles, hyps)

    from .functional_constraint_interpreter import FunctionalConstraintInterpreter
    constraint_interpreter = FunctionalConstraintInterpreter(
        domain=domain,
        roles_by_id=roles_by_id,
        hypotheses=hypotheses,
        relations=normalized_relations,
        slot_resolver=_resolve_or_probe,
    )
    normalized_operations = constraint_interpreter.interpret(normalized_operations)
    canonical["functional_constraint_interpretation"] = [
        *composite_lowerings, *relation_conjunction_trace, *constraint_interpreter.trace,
    ]
    if composite_lowerings:
        canonical["composite_receiving_context_lowerings"] = composite_lowerings
    canonical["fm_semantic_accounting"] = list(constraint_interpreter.accounting)
    canonical["current_state_operation_context_roles"] = sorted({
        str(row["raw_role"])
        for row in constraint_interpreter.trace
        if row.get("code") == "CURRENT_STATE_OPERATION_CONTEXT_ELIDED"
    })

    relations = []
    current_state_relations = []
    unresolved_relation_semantics = []
    unresolved_operation_semantics = []
    operation_pairs = [set(item.get("participant_roles", [])) for item in normalized_operations]
    for relation in normalized_relations:
        phrase_norm = re.sub(r"[_\-/]+", " ", str(relation["relation"])).lower()
        pair = set(relation["participant_roles"])
        role_texts = [
            " ".join(str((roles_by_id.get(p) or {}).get(key, "")) for key in ("function", "description"))
            for p in pair
        ]
        is_current = bool(
            re.search(r"\b(currently|initially|initial|starts?|stored|located at)\b", phrase_norm)
            or relation_states_where_things_currently_are(str(relation["relation"]))
        )
        if not is_current and re.fullmatch(r"\s*(?:is\s+)?(?:on|at)\s*", phrase_norm):
            pair_has_physical_placement = any(
                pair <= set(operation.get("participant_roles", ()))
                and re.search(r"\b(place|move|transfer|relocate|position|deposit|return)\w*\b",
                              re.sub(r"[_-]+", " ", str(operation.get("operation", ""))), re.I)
                for operation in normalized_operations
            )
            context_endpoint = any(
                re.search(r"\b(initial|current|storage|source container|tray|workbench|table|support surface)\b",
                          text.replace("_", " "), re.I)
                for text in role_texts
            )
            is_current = context_endpoint and not pair_has_physical_placement
        if not is_current and re.search(r"\bsupported by\b", phrase_norm) and pair not in operation_pairs:
            is_current = any(re.search(r"\b(support|surface|workbench|context)\b", text, re.I) for text in role_texts)
        if is_current:
            current_state_relations.append({**dict(relation), "category": "CURRENT_STATE_CONTEXT", "provenance": "FM_EXPLICIT_SEMANTIC"})
            continue
        options = _relation_options(domain, relation, hypotheses)
        semantic_candidates = extract_relation_semantic_candidates(domain, relation["relation"])
        if semantic_candidates and not options:
            # The runtime recognises the predicate but can give it no legal
            # orientation over these participants -- typically because the model
            # named a material or content as a participant where the runtime
            # models only the vessel.  That is one relation the runtime cannot
            # represent, not a contradictory task.  Record it as an unresolved
            # required semantic and carry on; completeness checking downstream
            # decides whether the task still stands without it.  Aborting the
            # whole contract here discarded every other coherent relation and
            # operation the model had expressed.
            unresolved_relation_semantics.append({
                **dict(relation),
                "category": "UNRESOLVED_REQUIRED_SEMANTIC",
                "provenance": "FM_EXPLICIT_SEMANTIC",
                "reason": "NO_LEGAL_ORIENTATION_FOR_PARTICIPANTS",
            })
            continue
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
    induced_roles: list[dict[str, Any]] = []
    from .system_context_registry import get_domain_selectable_roles
    selectable_roles = set(get_domain_selectable_roles(domain))
    for operation in normalized_operations:
        # Settle how many times this operation runs before anything is seated,
        # so slot completion sizes the slots it supplies to the same number the
        # compiled group will use.  Doing it only at group emission left a
        # synthesized support counted once while its group ran twice, and the
        # second placement had nowhere to go.
        reconciled_count = _reconciled_operation_count(
            domain, operation, roles_by_id, hypotheses, normalized_operations)
        if reconciled_count != int(operation.get("operation_count", 1) or 1):
            canonical["functional_constraint_interpretation"].append({
                "code": "OPERATION_COUNT_RAISED_TO_A_DISTINCT_PARTICIPANTS_DECLARED_COUNT",
                "operation_id": operation["id"],
                "operation": operation["operation"],
                "declared_operation_count": operation.get("operation_count"),
                "reconciled_operation_count": reconciled_count,
                "participant_roles": list(operation.get("participant_roles", ()) or ()),
            })
            operation = {**dict(operation), "operation_count": reconciled_count}
        operation_norm = re.sub(r"[_-]+", " ", str(operation["operation"])).lower().strip()
        if is_non_physical_operation_phrase(
            str(operation["operation"]), operation.get("participant_roles", ())
        ) or operation_norm.split(maxsplit=1)[0] in {
            "associate", "associates", "associating", "associated",
        }:
            canonical.setdefault("non_physical_operations", []).append({
                **dict(operation), "category": "NON_PHYSICAL_TASK_DIRECTIVE",
            })
            continue
        if _is_abstract_directive_over_unrepresented_participant(operation, hypotheses):
            # A desired end state, not a motion: the phrase states what must be
            # true when the task is done and one participant is something the
            # runtime holds in no form -- a person.  Asked before seating,
            # because a participant nothing typed is nominally admissible
            # everywhere, so "serve the soup to the diner" could otherwise be
            # seated as putting the diner in the bowl.
            canonical.setdefault("non_physical_operations", []).append({
                **dict(operation),
                "category": "ABSTRACT_TASK_DIRECTIVE_WITH_NO_PHYSICAL_ENDPOINT",
            })
            canonical["functional_constraint_interpretation"].append({
                "code": "ABSTRACT_TASK_DIRECTIVE_WITH_NO_PHYSICAL_ENDPOINT",
                "operation_id": operation["id"],
                "operation": operation["operation"],
                "participant_roles": list(operation["participant_roles"]),
                "provenance": "NON_PHYSICAL_DIRECTIVE",
            })
            continue
        capabilities = extract_operation_semantic_candidates(
            domain, operation["operation"], operation["participant_roles"])
        options, slot_participants, context_participants = _resolve_operation_slots_with_subsets(
            domain, operation, roles_by_id, hypotheses
        )
        if not options:
            # The model expressed the operation but named only some of the
            # participants its capability structurally involves.  Supply the
            # rest as typed existential requirements, which search and grounding
            # resolve, rather than discarding an operation the task needs.
            #
            # Tried twice: once as written, and once with the participants that
            # only say where things currently sit set aside, because a placement
            # naming its staging tray is under-specified about its destination
            # and over-specified about its origin at the same time.
            completed = complete_operation_slots(
                domain, operation, roles_by_id, hypotheses, expressed)
            elided = _contextual_participants(
                domain, operation, list(operation["participant_roles"]), roles_by_id, hypotheses)
            if completed is None and elided:
                reduced = [p for p in operation["participant_roles"] if p not in elided]
                if reduced:
                    completed = complete_operation_slots(
                        domain, {**dict(operation), "participant_roles": reduced},
                        roles_by_id, hypotheses, expressed)
                    # Setting a participant aside and inventing a selectable one
                    # in the same step compiles an operation that shares almost
                    # nothing with what the model wrote.  A fastening whose bench
                    # is set aside may still have its receiving site entailed by
                    # the capability -- that site is not something to go and find
                    # -- but if the *fastener* is what is missing, the model
                    # omitted a participant and the operation must say so.
                    if completed is not None and any(
                        role.get("canonical_role") in selectable_roles
                        for role in completed.synthesized_roles
                    ):
                        canonical["functional_constraint_interpretation"].append({
                            "code": "REDUCED_OPERATION_WOULD_ALSO_NEED_AN_INVENTED_PARTICIPANT",
                            "operation_id": operation["id"],
                            "set_aside": sorted(elided),
                            "would_have_synthesized": [
                                role.get("canonical_role") for role in completed.synthesized_roles
                            ],
                        })
                        completed = None
                    if completed is not None:
                        operation = {
                            **dict(operation), "participant_roles": reduced,
                            "current_state_context_roles": list(dict.fromkeys([
                                *operation.get("current_state_context_roles", ()), *sorted(elided),
                            ])),
                        }
            if completed is not None:
                for role in completed.synthesized_roles:
                    canonical["functional_roles"].append(role)
                    roles_by_id[role["id"]] = role
                    induced_roles.append(role)
                options = list(completed.options)
                slot_participants = list(dict.fromkeys([
                    *operation["participant_roles"],
                    *(role["id"] for role in completed.synthesized_roles),
                ]))
                context_participants = []
                operation = {**dict(operation), "participant_roles": slot_participants}
                if operation.get("current_state_context_roles"):
                    canonical["current_state_operation_context_roles"] = sorted(
                        set(canonical.get("current_state_operation_context_roles", ()))
                        | set(operation["current_state_context_roles"]))
                canonical.setdefault("operation_induced_slot_completions", []).append(completed.trace)
                canonical["functional_constraint_interpretation"].append(completed.trace)
                hypotheses = build_role_type_hypotheses(domain, canonical)
        if context_participants:
            operation = {
                **dict(operation),
                "participant_roles": slot_participants,
                "current_state_context_roles": list(dict.fromkeys(
                    [*operation.get("current_state_context_roles", ()), *context_participants]
                )),
            }
        canonical.setdefault("role_operation_consistency_audit", []).append({
            "operation_id": operation["id"],
            "raw_operation": operation["operation"],
            "explicit_participant_roles": list(operation.get("explicit_participant_roles", operation["participant_roles"])),
            "normalized_participant_roles": list(operation["participant_roles"]),
            "semantic_capability_candidates": [cap.capability_id for cap in capabilities],
            "viable_slot_assignments": len(options),
            "status": "PASS" if options or not capabilities else "FAIL",
            "provenance": "FM_EXPLICIT_OPERATION",
        })
        if capabilities and not options:
            # The runtime recognises the operation but can seat none of its
            # participants in a capability signature, even over subsets.  That is
            # one operation the runtime cannot execute, not a contradictory task.
            # Record it and continue; completeness checking decides whether the
            # task still stands, which surfaces as a graph compilation failure
            # rather than discarding the whole contract as malformed.
            if _is_abstract_directive_over_unrepresented_participant(operation, hypotheses):
                canonical.setdefault("non_physical_operations", []).append({
                    **dict(operation),
                    "category": "ABSTRACT_TASK_DIRECTIVE_WITH_NO_PHYSICAL_ENDPOINT",
                })
                canonical["functional_constraint_interpretation"].append({
                    "code": "ABSTRACT_TASK_DIRECTIVE_WITH_NO_PHYSICAL_ENDPOINT",
                    "operation_id": operation["id"],
                    "operation": operation["operation"],
                    "participant_roles": list(operation["participant_roles"]),
                    "provenance": "NON_PHYSICAL_DIRECTIVE",
                })
                continue
            unresolved_operation_semantics.append({
                "id": operation["id"],
                "operation": operation["operation"],
                "participant_roles": list(operation["participant_roles"]),
                "semantic_capability_candidates": [cap.capability_id for cap in capabilities],
                "reason": "NO_CAPABILITY_SIGNATURE_ACCEPTS_THESE_PARTICIPANTS",
            })
            continue
        # Take what every legal reading agrees on.  Several readings often differ
        # in only one slot -- two named participants each plausible as the anchor,
        # say -- and blanking all three because of that discarded an operation
        # whose source and target were never in doubt.  A slot the readings
        # disagree about stays open, and the full set of assignments travels with
        # the group so grounding can settle it against the observed scene.
        def _agreed(field: str):
            values = {row.get(field) for row in options}
            return values.pop() if len(values) == 1 else None

        if options:
            source, target = _agreed("source_role"), _agreed("target_role")
            anchor, usage = _agreed("anchor_role"), _agreed("usage_policy")
        else:
            source = target = anchor = None
            usage = None
        if not options:
            absorbed = _planner_context_transition(domain, operation, hypotheses)
            if absorbed is not None:
                # Moving something to a fixed place the planner already owns --
                # a serving area, a work surface -- is a domain transition, not a
                # selectable functional operation.  It is audited rather than
                # compiled, and it is not a semantic the runtime lost.
                canonical.setdefault("planner_context_transitions", []).append(absorbed)
                canonical["functional_constraint_interpretation"].append(absorbed)
                continue
            if _is_abstract_directive_over_unrepresented_participant(operation, hypotheses):
                canonical.setdefault("non_physical_operations", []).append({
                    **dict(operation),
                    "category": "ABSTRACT_TASK_DIRECTIVE_WITH_NO_PHYSICAL_ENDPOINT",
                })
                canonical["functional_constraint_interpretation"].append({
                    "code": "ABSTRACT_TASK_DIRECTIVE_WITH_NO_PHYSICAL_ENDPOINT",
                    "operation_id": operation["id"],
                    "operation": operation["operation"],
                    "participant_roles": list(operation["participant_roles"]),
                    "provenance": "NON_PHYSICAL_DIRECTIVE",
                })
                continue
            unresolved_operation_semantics.append({
                "id": operation["id"],
                "operation": operation["operation"],
                "participant_roles": list(operation["participant_roles"]),
                "semantic_capability_candidates": [],
                "reason": "NO_RUNTIME_CAPABILITY_MATCHES_THIS_OPERATION_PHRASE",
            })
            continue
        groups.append({
            "id": operation["id"], "function": operation["operation"],
            "tool_role": source, "target_role": target, "context_role": anchor,
            "required_target_count": operation["operation_count"], "usage_policy": usage,
            "required_relations": [], "context_relations": [],
            "v3_slot_assignments": options,
            "v3_participant_roles": list(operation["participant_roles"]),
            "v3_explicit_participant_roles": list(operation.get("explicit_participant_roles", operation["participant_roles"])),
            "v3_raw_fm_participant_roles": list(operation.get("raw_fm_participant_roles", operation.get("explicit_participant_roles", operation["participant_roles"]))),
            "v3_witness_roles": sorted({
                str(role["fm_witness_role"])
                for role in induced_roles
                if role.get("induced_by_operation") == operation["id"]
                and role.get("fm_witness_role")
            }),
            "v3_current_state_context_roles": list(operation.get("current_state_context_roles", [])),
            "v3_graph_join_relation_id": operation.get("graph_join_relation_id"),
            "v3_explicit_context_set_id": operation.get("explicit_context_set_id"),
        })
    canonical["interaction_groups"] = groups
    # Setting a participant aside is a decision about one operation, and it used
    # to be recorded against the whole document.  A fastening that sets the bench
    # aside so its receiving site can be entailed then silently removed the bench
    # from the tool return the model wrote next, and an operation that resolved
    # perfectly well was reported as naming a participant with no runtime role.
    # A role some compiled operation seats in a slot of its own is not present
    # state, whatever another operation made of it.
    seated_in_a_slot = {
        group.get(field)
        for group in groups
        for field in ("tool_role", "target_role", "context_role")
        if group.get(field)
    }
    if seated_in_a_slot & set(canonical.get("current_state_operation_context_roles", ()) or ()):
        kept = [
            rid for rid in canonical.get("current_state_operation_context_roles", ())
            if rid not in seated_in_a_slot
        ]
        canonical["functional_constraint_interpretation"].append({
            "code": "CURRENT_STATE_ELISION_WITHDRAWN_FOR_ROLE_AN_OPERATION_SEATS",
            "withdrawn": sorted(
                set(canonical.get("current_state_operation_context_roles", ())) & seated_in_a_slot),
            "still_elided": kept,
            "provenance": "FM_EXPLICIT_OPERATION",
        })
        canonical["current_state_operation_context_roles"] = kept
    form_roles = _split_generic_participant_per_operation_form(
        domain, canonical, roles_by_id, hypotheses)
    if form_roles:
        induced_roles.extend(form_roles)
        hypotheses = build_role_type_hypotheses(domain, canonical)
        # The relations were read before the operations were seated, so they
        # still name the raw role the forms came from.  Re-point the ones whose
        # own operation is unambiguous; see the routing function.
        _route_relations_to_operation_specific_forms(domain, canonical)
    # A relation the model wrote to state the very requirement a compiled
    # operation already carries is not a second constraint the runtime failed to
    # represent.  Once SUPPORT_DRINKWARE is identified the runtime owns
    # FITS_SET_ON and NEAR_SEAT and verifies them physically; the model saying
    # "the setting is near the seat" in its own words is the evidence that
    # justified the capability, not extra vocabulary it must also get right.
    # Only a predicate the compiled capability itself enforces, over that same
    # operation's participants, is treated this way.
    subsumed_relations = []
    if unresolved_relation_semantics:
        capability_by_id = {
            capability.capability_id: capability
            for capability in get_robot_capabilities(domain)
        }
        surviving = []
        for item in unresolved_relation_semantics:
            meanings = {
                candidate.predicate_name
                for candidate in extract_relation_semantic_candidates(domain, item["relation"])
            }
            participants = set(item.get("participant_roles", ()))
            enforced_by = None
            for group in groups:
                witnesses = set(group.get("v3_witness_roles", ()))
                named = set(group.get("v3_participant_roles", ())) | witnesses
                slot_families = {
                    canonical_role_family(domain, row[key])
                    for row in group.get("v3_slot_assignments", ())
                    for key in ("source_type", "target_type", "anchor_type")
                    if row.get(key)
                }
                slot_families.discard("OTHER")

                stood_aside = set(group.get("v3_current_state_context_roles", ()) or ())

                def _talks_about_this_operation(participant: str) -> bool:
                    """Whether the relation's participant is one this operation involves.

                    Named outright, set aside by this very operation so the slot
                    it stood for could be supplied, or of the same functional
                    kind as one of the operation's slots: the model may state the
                    requirement about its own seating role while the operation's
                    anchor is the registered seating reference standing for it,
                    and it may write one role for both a bench and the site on it.
                    """
                    if participant in named or participant in stood_aside:
                        return True
                    families = {
                        canonical_role_family(domain, candidate)
                        for candidate in getattr(
                            hypotheses.get(participant), "canonical_role_candidates", ()) or ()
                    }
                    return bool(families & slot_families)

                if not all(_talks_about_this_operation(p) for p in participants):
                    continue
                for row in group.get("v3_slot_assignments", ()):
                    capability = capability_by_id.get(row.get("capability_id"))
                    if capability is None:
                        continue
                    owned = {predicate for _, predicate, _ in capability.required_relation_templates}
                    if meanings & owned:
                        enforced_by = (group["id"], capability.capability_id,
                                       sorted(meanings & owned))
                        break
                if enforced_by:
                    break
            if enforced_by:
                subsumed_relations.append({
                    **dict(item),
                    "code": "RELATION_ENFORCED_BY_COMPILED_OPERATION",
                    "operation_id": enforced_by[0],
                    "capability_id": enforced_by[1],
                    "enforced_predicates": enforced_by[2],
                    "provenance": "ROBOT_CAPABILITY_PRECONDITION",
                })
            else:
                surviving.append(item)
        unresolved_relation_semantics = surviving
        canonical["relations_enforced_by_operations"] = subsumed_relations
        canonical["functional_constraint_interpretation"].extend(subsumed_relations)
    primitive_participants = {
        participant
        for operation in normalized_operations
        if not is_non_physical_operation_phrase(
            str(operation.get("operation", "")), operation.get("participant_roles", ()))
        for participant in operation.get("participant_roles", ())
    }
    relation_only_context = {
        participant
        for relation in current_state_relations
        for participant in relation.get("participant_roles", ())
        if participant not in primitive_participants
        and re.search(
            r"\b(storage|source container|tray|support|surface|table|workbench|location|region)\b",
            " ".join((
                str(roles_by_id.get(participant, {}).get("function", "")),
                str(roles_by_id.get(participant, {}).get("description", "")),
                " ".join(roles_by_id.get(participant, {}).get("candidate_categories", ())),
            )).replace("_", " "),
            re.I,
        )
    }
    canonical["current_state_operation_context_roles"] = sorted(
        set(canonical.get("current_state_operation_context_roles", ())) | relation_only_context
    )
    for item in v3_doc.get("structurally_unusable_elements", ()) or ():
        # Repaired down to too little to state anything, and therefore still a
        # semantic the runtime could not represent rather than one it repaired
        # away.
        record = {
            "id": item.get("id"),
            "participant_roles": list(item.get("participant_roles", ())),
            "reason": item.get("reason"),
            "requirement_provenance": "INSTRUCTION_CLAUSE_SUPPORT",
        }
        if item.get("element_kind") == "operation":
            unresolved_operation_semantics.append({
                **record, "operation": item.get("operation"),
                "semantic_capability_candidates": [],
            })
        else:
            unresolved_relation_semantics.append({
                **record, "relation": item.get("relation"), "category": "UNRESOLVED_REQUIRED_SEMANTIC",
                "provenance": "FM_EXPLICIT_SEMANTIC",
            })
    canonical["unresolved_operation_semantics"] = list(unresolved_operation_semantics)
    for item in unresolved_operation_semantics:
        for row in canonical["fm_semantic_accounting"]:
            if row.get("element_kind") == "operation" and row.get("raw_id") == item["id"]:
                row["disposition"] = "UNRESOLVED_REQUIRED_SEMANTIC"
                row["reason"] = item["reason"]
                break
    canonical["unresolved_relation_semantics"] = [
        {"id": item["id"], "relation": item["relation"],
         "participant_roles": list(item.get("participant_roles", ())),
         "reason": item.get("reason", ""),
         "requirement_provenance": classify_requirement_provenance(
             _provenance_terms(item.get("relation", "")),
             instruction_terms=_provenance_terms(task_instruction),
             participates_in_expressed_operation=bool(
                 set(item.get("participant_roles", ())) & {
                     participant
                     for operation in contract["operation_pairings"]
                     for participant in operation.get("participant_roles", ())
                 }),
         )}
        for item in unresolved_relation_semantics
    ]
    instruction_terms = _provenance_terms(task_instruction)
    operation_participants = {
        participant
        for operation in contract["operation_pairings"]
        for participant in operation.get("participant_roles", ())
    }
    # Operations need the same provenance record as roles and relations.  Without
    # it, an operation the runtime could not represent had nothing to say about
    # why it was required, and downstream completeness checking had no basis for
    # deciding whether losing it mattered.
    operation_provenance = {
        operation["id"]: classify_requirement_provenance(
            _provenance_terms(operation.get("operation", "")),
            instruction_terms=instruction_terms,
            participates_in_expressed_operation=True,
        )
        for operation in contract["operation_pairings"]
    }
    for row in canonical["fm_semantic_accounting"]:
        if row.get("element_kind") == "operation" and "requirement_provenance" not in row:
            row["requirement_provenance"] = operation_provenance.get(
                row.get("raw_id"), "DERIVED_FROM_EXPLICIT_OPERATION")
    for item in unresolved_operation_semantics:
        item["requirement_provenance"] = operation_provenance.get(
            item["id"], "DERIVED_FROM_EXPLICIT_OPERATION")
    for role in contract["functional_roles"]:
        is_current_context = role["id"] in set(canonical.get("current_state_operation_context_roles", ()))
        role_terms = _provenance_terms(" ".join((
            str(role.get("function", "")),
            str(role.get("description", "")),
            " ".join(role.get("candidate_categories", ()) or ()),
        )))
        canonical["fm_semantic_accounting"].append({
            "element_kind": "role", "raw_id": role["id"],
            "disposition": "CURRENT_STATE_CONTEXT" if is_current_context else "GROUNDED_TASK_PARTICIPANT",
            "canonical_representation": None if is_current_context else list(hypotheses[role["id"]].canonical_role_candidates),
            "provenance": "FM_EXPLICIT_ROLE",
            "requirement_provenance": classify_requirement_provenance(
                role_terms,
                instruction_terms=instruction_terms,
                participates_in_expressed_operation=role["id"] in operation_participants,
            ),
        })
    relation_dispositions = {
        item["id"]: "UNRESOLVED_REQUIRED_SEMANTIC" for item in unresolved_relation_semantics
    }
    relation_dispositions.update({item["id"]: "CURRENT_STATE_CONTEXT" for item in current_state_relations})
    relation_dispositions.update({item["id"]: "GROUNDED_TASK_RELATION" for item in relations})
    for item in normalized_relations:
        if item.get("source_relation_ids") and item["id"] in relation_dispositions:
            relation_dispositions.update({source_id: relation_dispositions[item["id"]]
                                          for source_id in item["source_relation_ids"]})
    for relation in contract["functional_relations"]:
        canonical["fm_semantic_accounting"].append({
            "element_kind": "relation", "raw_id": relation["id"],
            "disposition": relation_dispositions.get(relation["id"], "EXPLICIT_RUNTIME_CONTEXT"),
            "canonical_representation": relation.get("relation"),
            "provenance": "FM_EXPLICIT_RELATION",
            "requirement_provenance": classify_requirement_provenance(
                _provenance_terms(relation.get("relation", "")),
                instruction_terms=instruction_terms,
                participates_in_expressed_operation=bool(
                    set(relation.get("participant_roles", ())) & operation_participants
                ),
            ),
        })
    # A stated end state used to be re-oriented here to match the raw group's
    # slot layout, and the stage that corroborated it matched against that same
    # layout, so the two agreed only because both were wrong in the same way.
    # A slot position is a call-signature convention, not a claim about the
    # world: the runtime happens to hold the support as an operation's source,
    # which turned "the setting is on the table" into "the table is on the
    # setting".  Orientation now comes from the reading, which the declared
    # endpoint families settle, and corroboration matches the compiled
    # operation rather than the wire order.

    # Reconcile the accounting against the raw contract.  Every raw role,
    # relation and operation must leave a trace of what became of it, so that a
    # required semantic cannot disappear without the runtime saying so.  Anything
    # not already accounted for is recorded as unresolved rather than dropped.
    accounted = {
        (row.get("element_kind"), row.get("raw_id"))
        for row in canonical["fm_semantic_accounting"]
    }
    for kind, collection in (
        ("role", contract["functional_roles"]),
        ("relation", contract["functional_relations"]),
        ("operation", contract["operation_pairings"]),
    ):
        for element in collection:
            if (kind, element["id"]) in accounted:
                continue
            canonical["fm_semantic_accounting"].append({
                "element_kind": kind, "raw_id": element["id"],
                "disposition": "UNRESOLVED_REQUIRED_SEMANTIC",
                "canonical_representation": None,
                "provenance": "FM_EXPLICIT_SEMANTIC",
                "requirement_provenance": "UNDETERMINED",
                "reason": "no stage claimed this element",
            })
    return canonical


def validate_v3_live_contract(
    doc: Mapping[str, Any], *, domain: str | None = None, task_instruction: str = ""
) -> dict[str, Any]:
    # Runtime bookkeeping the normalizer attaches is not model output and is not
    # part of the wire contract, so the schema never sees it.
    wire = {key: value for key, value in dict(doc).items()
            if key not in _RUNTIME_ONLY_DOCUMENT_KEYS}
    try:
        jsonschema.validate(instance=wire, schema=LIVE_RESPONSE_SCHEMA_V3)
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
        convert_v3_to_canonical_document(doc, domain=domain, task_instruction=task_instruction)
    return deepcopy(dict(doc))


def normalize_and_validate_v3_contract(
    doc: Mapping[str, Any], *, domain: str | None = None, task_instruction: str = ""
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized, trace = normalize_v3_live_document(doc, task_instruction=task_instruction, domain=domain)
    return validate_v3_live_contract(
        normalized, domain=domain, task_instruction=task_instruction), trace


def compute_v3_prompt_and_schema_hash() -> str:
    blob = json.dumps({
        "system_prompt_v3": SYSTEM_PROMPT_V3,
        "user_request_v3": USER_REQUEST_V3,
        "response_schema_v3": LIVE_RESPONSE_SCHEMA_V3,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
