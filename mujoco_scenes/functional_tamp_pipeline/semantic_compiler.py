"""Stage B: typed candidate compilation, independent of benchmark references.

Domain mappers supply vocabulary; causal position resolves participants. Unknown
content remains auditable. This module never reads task/reference fixtures.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from .models import (
    FunctionalRequirementGraph,
    FunctionalRole,
    FunctionalRelation,
    OperationGroup,
    ProvisionalOperationConstraint,
    ProvisionalRelationConstraint,
    RoleTypeHypothesis,
    TaskEffectRelation,
)
from .structural_sanitizer import sanitize_functional_graph
from . import role_semantic_ontology as ontology
from .predicate_registry import validate_predicate_signature
from .fm_schema_v2 import is_v2_document
from .fm_schema_v3 import is_v3_document
from .errors import VLMSpecificationError
from .robot_capability_registry import extract_operation_semantic_candidates


def causal_position(role: dict, document: dict) -> set[str]:
    """Causal types from role language and graph position, without material names."""
    text = re.sub(r"[_-]", " ", f"{role.get('function', '')} {role.get('description', '')}").lower()
    positions = set()
    if re.search(r"\b(supplies|supply|source|provider|provides|providing|ingredient|powder|granules|raw material)\b", text):
        positions.add("source")
    if re.search(r"\b(receives|receiving|prepared|served|serving|destination)\b", text):
        positions.add("destination")
    if re.search(r"\b(tool|implement|instrument|equipment)\b", text):
        positions.add("instrument")
    if re.search(r"\b(installed|component|remains|assembly part)\b", text):
        positions.add("component")
    rid = role['id']
    for group in document.get('interaction_groups', []):
        if group.get('tool_role') == rid:
            positions.add('group_tool')
        if group.get('target_role') == rid:
            positions.add('group_target')
    for relation in document.get('functional_relations', []):
        phrase = re.sub(
            r"\s+", " ", re.sub(r"[_-]+", " ", str(relation.get('relation', relation.get('predicate', ''))).lower())
        ).strip()
        if re.search(r"\b(supplies|pours into|transfers to|provides material to)\b", phrase):
            if relation.get('subject_role') == rid:
                positions.add('source')
            if relation.get('object_role') == rid:
                positions.add('destination')
    return positions


def can_merge_roles(a: dict, b: dict, document: dict, *, allow_counted_context: bool = False) -> bool:
    if any(a.get(k) != b.get(k) for k in ('entity_kind', 'binding_policy', 'binding_cardinality', 'min_count', 'max_count')):
        return False
    if (a.get('entity_kind') == 'FIXED_TARGET' or b.get('entity_kind') == 'FIXED_TARGET') and not allow_counted_context:
        return False
    pos_a = causal_position(a, document)
    pos_b = causal_position(b, document)
    if pos_a != pos_b:
        # If both roles have distinct non-empty causal positions, they serve different causal functions.
        # However, if one role is an unreferenced duplicate (empty causal position) with identical function
        # and binding specification, it can safely merge with the active role.
        if pos_a and pos_b:
            return False
    pair = {a['id'], b['id']}
    if any({r.get('subject_role'), r.get('object_role')} == pair for r in document.get('functional_relations', [])):
        return False
    if any({g.get('tool_role'), g.get('target_role')} == pair for g in document.get('interaction_groups', [])):
        return False
    # Equal canonical names alone do not establish equal causal function.
    def normalize_function(value):
        text = re.sub(r'\b(?:first|second|third|fourth|one|two|three|[0-9]+|[a-z])\b', ' ', (value or '').lower())
        return re.sub(r'\W+', ' ', text).strip()
    if normalize_function(a.get('function', '')) != normalize_function(b.get('function', '')):
        return False
    if set(a.get('candidate_categories', ())) != set(b.get('candidate_categories', ())):
        return False
    if set(a.get('required_properties', ())) != set(b.get('required_properties', ())):
        return False
    return True


def _map_role(domain: str, role: dict, doc: dict) -> tuple[str | None, str]:
    position = causal_position(role, doc)
    if domain == 'kitchen':
        from mujoco_scenes.kitchen_vlm_functional_graph import (
            map_kitchen_planner_context_role,
            map_kitchen_role_function,
        )
        planner_context = map_kitchen_planner_context_role(role, doc)
        if planner_context:
            return planner_context, 'TASK_EXPRESSED_SYSTEM_CONTEXT'
        enriched = dict(role)
        if 'source' in position and 'destination' not in position:
            # Preserve material identity while preventing a receptacle's physical
            # container form from overriding its causal source function.
            enriched['function'] = 'source provider ' + role['function']
            enriched['description'] = role.get('description', '')
            text = (enriched['function'] + ' ' + enriched['description']).lower()
            for material in ('water', 'coffee'):
                if re.search(r'\b' + material + r'\b', text):
                    return material + '_source', 'CAUSAL_SOURCE_PROVIDER'
        return map_kitchen_role_function(enriched), 'DOMAIN_SEMANTICS'
    if domain == 'living_room':
        from mujoco_scenes.environment_vlm_requirements import (map_living_room_role_function,
            map_living_room_object_payload_role, map_living_room_fixed_target_role)
        role_text = re.sub(
            r"\s+", " ", re.sub(r"[_-]+", " ", f"{role.get('function', '')} {role.get('description', '')}".lower())
        ).strip()
        category_text = ' '.join(role.get('candidate_categories', [])).lower()
        if re.search(r'\b(initial|current|original)\b.*\b(source|location|storage)\b', role_text) and re.search(
            r'\b(table|desk|surface|staging|support)\b', role_text + ' ' + category_text
        ):
            return None, 'CURRENT_STATE_CONTEXT_ONLY'
        seating = map_living_room_fixed_target_role(role)
        if seating:
            return seating, 'FIXED_TARGET_SEMANTICS'
        support_semantics = bool(re.search(r"\b(table|surface|support|platform|area|zone)\b", role_text))
        personal_semantics = bool(re.search(r"\b(personal|individual|refreshment|near(?:by)?|beside|adjacent|side table)\b", role_text))
        shared_control_semantics = bool(re.search(r"\b(shared|central|common|accessible|both|control|remote|media)\b", role_text))
        if support_semantics and shared_control_semantics and not personal_semantics:
            return 'SHARED_REMOTE_REGION', 'EXPLICIT_SHARED_CONTROL_SUPPORT_SEMANTICS'
        if support_semantics and personal_semantics and not shared_control_semantics:
            return 'PERSONAL_CUP_SAUCER_REGION', 'EXPLICIT_PERSONAL_REFRESHMENT_SUPPORT_SEMANTICS'
        if re.search(r"\b(set|pair|setting|drinkware)\b", role_text) and re.search(
            r"\b(refreshment|cup|dish|plate|saucer|drinkware|food|drink)\b", role_text
        ):
            return 'CUP_SAUCER_SET', 'EXPLICIT_REFRESHMENT_PAYLOAD_SEMANTICS'
        if re.search(r"\b(carr(?:y|ies)|container|receptacle|vessel|payload)\b", role_text) and re.search(
            r"\b(refreshment|beverage|drink|food|consumable)\b", role_text
        ) and re.search(r"\b(person|occupant|user|individual|one)\b", role_text):
            return 'CUP_SAUCER_SET', 'EXPLICIT_REFRESHMENT_PAYLOAD_SEMANTICS'
        mapper = {'REGION': map_living_room_role_function, 'OBJECT': map_living_room_object_payload_role,
                  'FIXED_TARGET': map_living_room_fixed_target_role}.get(role['entity_kind'], map_living_room_role_function)
        mapped = mapper(role)
        if mapped is None and role['entity_kind'] == 'REGION':
            function = (role.get('function', '') + ' ' + role.get('description', '')).lower()
            if re.search(r'\b(current|currently|original|initial)\b', function) and re.search(
                r'\b(storage|location|surface|position)\b', function
            ):
                return None, 'CURRENT_STATE_CONTEXT_ONLY'
            edges = [r for r in doc.get('functional_relations', []) if role['id'] in (r.get('subject_role'), r.get('object_role'))]
            spatial = ' '.join(str(r.get('relation', r.get('predicate', ''))) for r in edges).lower()
            bp = role.get('binding_policy')
            count = role.get('required_count', 1)
            if re.search(r'\b(television|screen|monitor|display|wall)\b', function):
                # Television mounting/wall is scene context, not a placement region for remotes
                return None, 'SCENE_CONTEXT'
            if re.search(r'\b(support|placement|setting|surface)\b', function):
                if bp == 'SHARED' or re.search(r'\b(accessible|both|shared|remote|entertainment|control)\b', spatial + ' ' + function):
                    mapped = 'SHARED_REMOTE_REGION'
                elif bp == 'DISTINCT' or re.search(r'\b(near|nearby|beside|adjacent|refreshment|cup|saucer|drink)\b', spatial + ' ' + function):
                    mapped = 'PERSONAL_CUP_SAUCER_REGION'
            elif bp == 'SHARED' and count == 1:
                mapped = 'SHARED_REMOTE_REGION'
            elif bp == 'DISTINCT' and count >= 2:
                mapped = 'PERSONAL_CUP_SAUCER_REGION'
        return mapped, 'TYPED_DOMAIN_SEMANTICS'
    from mujoco_scenes.workshop_phase1.requirements import (map_workshop_role_function,
        map_workshop_fixed_target_role, map_workshop_context_region_role)
    target = map_workshop_fixed_target_role(role)
    if target:
        return target, 'FIXED_TARGET_SEMANTICS'
    if role['entity_kind'] == 'REGION':
        return map_workshop_context_region_role(role), 'SUPPORT_CONTEXT'
    role_text = re.sub(
        r"\s+", " ", re.sub(r"[_-]+", " ", f"{role.get('function', '')} {role.get('description', '')}".lower())
    ).strip()
    if re.search(
        r"\b(fastening target|repair target|fixture assembly|target workpiece|workpiece target|part receiving fastener|object receiving fastening)\b",
        role_text,
    ):
        return 'repair_target', 'EXPLICIT_FIXED_FASTENING_TARGET_SEMANTICS'
    mapped = map_workshop_role_function(role)
    if mapped is None and re.search(
        r"\b(fastener|fastening component|joining part|connector|component to be (?:installed|tightened)|part to be (?:attached|assembled))\b",
        role_text,
    ) and not re.search(r"\b(tool|driver|wrench|drill|instrument|equipment)\b", role_text):
        mapped = 'CAN_FASTEN'
    if mapped == 'CAN_FASTEN' and re.search(
        r"\b(receiv(?:e|es|ing)|fixture|assembly|workpiece)\b", role_text
    ) and not re.search(r"\b(fastener|connector|joining part|screw|bolt)\b", role_text):
        mapped = 'repair_target'
    if mapped is None and 'instrument' in position and 'group_tool' in position and 'component' not in position:
        # Operation-instrument position plus explicit implement semantics.
        from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract
        aliases = ManualWorkshopFMContract().get_alias_to_canonical_map()
        categories = {aliases.get(c.lower(), c.lower().replace(' ', '_')) for c in role.get('candidate_categories', [])}
        if categories.intersection(ontology.get_system_role_semantic_categories('workshop', 'driver')):
            mapped = 'CAN_DRIVE_SCREW'
    return {'CAN_DRIVE_SCREW': 'driver', 'CAN_FASTEN': 'fastener'}.get(mapped, mapped), 'CAUSAL_OPERATION_PARTICIPANT'


def _relation(domain: str, subject: str, phrase: str, target: str) -> tuple[str, str, str]:
    if domain == 'kitchen':
        from mujoco_scenes.kitchen_vlm_functional_graph import map_binary_relation
        return subject, map_binary_relation(phrase), target
    if domain == 'living_room':
        from mujoco_scenes.environment_vlm_requirements import canonicalize_living_room_relation
        return canonicalize_living_room_relation(phrase, subject, target)[:3]
    from mujoco_scenes.workshop_phase1.requirements import canonicalize_workshop_relation
    return canonicalize_workshop_relation(subject, subject, phrase, target, target)[:3]


def _canonical_role_kind(domain: str, role_name: str) -> str:
    from .system_context_registry import get_domain_system_fixed_anchors
    if role_name in set(get_domain_system_fixed_anchors(domain)):
        return "FIXED_TARGET"
    if role_name in {"PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION", "MAIN_WORKBENCH_ZONE"}:
        return "REGION"
    return "OBJECT"


def _causal_role_pairs(domain: str, predicate: str) -> tuple[tuple[str, str], ...]:
    """Generic domains induced by a causal predicate, never a physical check."""
    if domain == "kitchen":
        if predicate == "PROVIDES_MATERIAL_TO":
            return tuple((s, t) for s in ("coffee_source", "water_source") for t in ("coffee_container", "soup_container"))
        if predicate == "ACTS_ON":
            return (("coffee_stirrer", "coffee_container"), ("soup_eating_utensil", "soup_container"))
        if predicate == "PAIRED_WITH":
            return (("soup_eating_utensil", "soup_container"),)
    if domain == "workshop":
        if predicate == "ACTS_ON":
            return (("driver", "fastener"),)
        if predicate in {"INSTALLED_AT", "CONNECTED_TO"}:
            return (("fastener", "repair_target"),)
    if domain == "living_room" and predicate == "SITUATED_BETWEEN":
        return tuple((support, seating)
                     for support in ("PERSONAL_CUP_SAUCER_REGION", "SHARED_REMOTE_REGION")
                     for seating in ("SEATING_POSITION", "SEATING_PAIR"))
    return ()


def _explicit_incompatible_role_claim(domain: str, role: dict[str, Any], candidates: set[str]) -> bool:
    """Detect a clear function claim outside the candidate role family."""
    text = re.sub(r"[_-]+", " ", f"{role.get('function', '')} {role.get('description', '')}").lower()
    if domain == "living_room" and re.search(r"\b(television|tv screen|display screen|monitor)\b", text):
        return bool(candidates)
    return False


def resolve_role_type_hypotheses(domain: str, document: dict[str, Any]) -> dict[str, RoleTypeHypothesis]:
    """Resolve FM roles jointly from function, relation, and operation evidence.

    Direct function mapping is evidence, not a prerequisite.  Relation and
    operation text first nominate semantic candidates independently, then their
    registered signatures constrain both endpoints to a fixed point.
    """
    from .predicate_registry import get_predicate_signature
    from .relation_interpreter import extract_relation_semantic_candidates
    from .robot_capability_registry import extract_operation_semantic_candidates
    from .system_context_registry import get_domain_selectable_roles, get_domain_system_fixed_anchors

    allowed = set(get_domain_selectable_roles(domain)) | set(get_domain_system_fixed_anchors(domain))
    roles_by_id = {str(role["id"]): role for role in document.get("functional_roles", ())}
    candidates: dict[str, set[str]] = {}
    direct: dict[str, str | None] = {}
    evidence: dict[str, list[dict[str, Any]]] = {rid: [] for rid in roles_by_id}
    constrained_by: dict[str, set[str]] = {rid: set() for rid in roles_by_id}
    contradictions: set[str] = set()

    for rid, role in roles_by_id.items():
        try:
            mapped, rule = _map_role(domain, role, document)
        except (VLMSpecificationError, KeyError, ValueError) as exc:
            mapped, rule = None, str(exc)
        mapped = mapped if mapped in allowed else None
        direct[rid] = mapped
        kind = role.get("entity_kind", "OBJECT")
        candidates[rid] = ({mapped} if mapped else {
            name for name in allowed if _canonical_role_kind(domain, name) == kind
        })
        evidence[rid].append({
            "source": "FUNCTION_TEXT", "status": "MATCH" if mapped else "UNKNOWN",
            "canonical_role": mapped, "rule": rule,
        })

    constraints: list[tuple[str, str, set[tuple[str, str]], dict[str, Any]]] = []
    for relation in document.get("functional_relations", ()):
        raw_s = relation.get("subject_role")
        raw_o = relation.get("object_role")
        if raw_s not in roles_by_id or raw_o not in roles_by_id:
            continue
        phrase = str(relation.get("relation", relation.get("predicate", "")))
        pairs: set[tuple[str, str]] = set()
        semantic = extract_relation_semantic_candidates(domain, phrase)
        for item in semantic:
            if item.category == "PHYSICAL_VERIFIER":
                sig = get_predicate_signature(domain, item.predicate_name)
                if not sig or sig.arity != 2 or not sig.active_in_functional_graph:
                    continue
                canonical_pairs = {
                    (s, o) for s in sig.allowed_subject_roles for o in sig.allowed_object_roles
                }
                if item.predicate_name in {"NEAR_SEAT", "ACCESSIBLE_FROM_BOTH_SEATS", "COMPATIBLE_WITH"}:
                    canonical_pairs |= {(o, s) for s, o in canonical_pairs}
            elif item.category == "TASK_CAUSAL_SEMANTICS":
                canonical_pairs = set(_causal_role_pairs(domain, item.predicate_name))
            else:
                continue
            if item.direction == "REVERSE":
                canonical_pairs = {(o, s) for s, o in canonical_pairs}
            pairs.update(canonical_pairs)
        if pairs and any(s in candidates[raw_s] and o in candidates[raw_o] for s, o in pairs):
            detail = {"source": "RELATION_TEXT", "raw_phrase": phrase,
                      "semantic_candidates": [item.__dict__ for item in semantic]}
            constraints.append((raw_s, raw_o, pairs, detail))

    for operation in document.get("interaction_groups", ()) or document.get("operations", ()):
        raw_s = operation.get("tool_role") or operation.get("source_role")
        raw_o = operation.get("target_role")
        raw_a = operation.get("context_role") or operation.get("anchor_role")
        if raw_s not in roles_by_id or raw_o not in roles_by_id:
            continue
        phrase = str(operation.get("function") or operation.get("operation") or "")
        capabilities = extract_operation_semantic_candidates(domain, phrase)
        pairs: set[tuple[str, str]] = set()
        for capability in capabilities:
            direct_pairs = {
                (source, target)
                for source in capability.allowed_source_roles
                for target in capability.allowed_target_roles
            }
            reverse_pairs = (
                {(target, source) for source, target in direct_pairs}
                if domain == "living_room" else set()
            )
            # Payload-to-support language in Living Room is represented by a
            # support capability whose executable source is the support region.
            pairs.update(direct_pairs | reverse_pairs)
        if pairs and any(s in candidates[raw_s] and o in candidates[raw_o] for s, o in pairs):
            detail = {"source": "OPERATION_TEXT", "raw_phrase": phrase,
                      "capability_candidates": [cap.capability_id for cap in capabilities]}
            constraints.append((raw_s, raw_o, pairs, detail))
        if raw_a in roles_by_id and capabilities:
            anchor_allowed = {role for cap in capabilities for role in cap.allowed_anchor_roles}
            if anchor_allowed:
                before = set(candidates[raw_a])
                candidates[raw_a].intersection_update(anchor_allowed)
                constrained_by[raw_a].add("OPERATION_TEXT")
                evidence[raw_a].append({"source": "OPERATION_TEXT", "raw_phrase": phrase,
                                        "allowed_anchor_roles": sorted(anchor_allowed)})
                if before and not candidates[raw_a]:
                    contradictions.add(raw_a)

    changed = True
    while changed:
        changed = False
        for raw_s, raw_o, pairs, detail in constraints:
            viable = {(s, o) for s, o in pairs if s in candidates[raw_s] and o in candidates[raw_o]}
            s_allowed = {s for s, _ in viable}
            o_allowed = {o for _, o in viable}
            before_s, before_o = set(candidates[raw_s]), set(candidates[raw_o])
            candidates[raw_s].intersection_update(s_allowed)
            candidates[raw_o].intersection_update(o_allowed)
            source = detail["source"]
            constrained_by[raw_s].add(source)
            constrained_by[raw_o].add(source)
            if detail not in evidence[raw_s]: evidence[raw_s].append(detail)
            if detail not in evidence[raw_o]: evidence[raw_o].append(detail)
            if (before_s and not candidates[raw_s]) or (before_o and not candidates[raw_o]):
                contradictions.update((raw_s, raw_o))
            changed |= before_s != candidates[raw_s] or before_o != candidates[raw_o]

    for rid, role in roles_by_id.items():
        if direct[rid] is None and _explicit_incompatible_role_claim(domain, role, candidates[rid]):
            contradictions.add(rid)

    result: dict[str, RoleTypeHypothesis] = {}
    for rid, role in roles_by_id.items():
        values = tuple(sorted(candidates[rid]))
        sources = constrained_by[rid]
        if rid in contradictions or not values:
            status = "CONTRADICTORY_ROLE_TYPE"
        elif len(values) > 1:
            status = "AMBIGUOUS_ROLE_TYPE" if sources else "UNCONSTRAINED_ROLE_TYPE"
        elif direct[rid] is not None:
            status = "DIRECT_FUNCTION_MATCH"
        elif sources == {"RELATION_TEXT"}:
            status = "RELATION_ASSISTED"
        elif sources == {"OPERATION_TEXT"}:
            status = "OPERATION_ASSISTED"
        else:
            status = "JOINT_SEMANTIC_RESOLUTION"
        result[rid] = RoleTypeHypothesis(
            raw_role_id=rid,
            raw_function=str(role.get("function", "")),
            raw_description=str(role.get("description", "")),
            entity_kind=str(role.get("entity_kind", "OBJECT")),
            canonical_role_candidates=values,
            status=status,
            evidence=tuple(evidence[rid]),
        )
    return result


# Keep the public compiler entry point backed by the pure hypothesis layer.
# The legacy implementation above remains temporarily local so downstream
# imports retain compatibility while the shared layer owns all new decisions.
_legacy_resolve_role_type_hypotheses = resolve_role_type_hypotheses


def resolve_role_type_hypotheses(
    domain: str, document: dict[str, Any]
) -> dict[str, RoleTypeHypothesis]:
    from .semantic_typing import build_role_type_hypotheses
    return build_role_type_hypotheses(domain, document, weak_mapper=_map_role)


def check_required_contract_complete(
    domain: str,
    nodes: dict[str, Any],
    relations: Sequence[Any],
    groups: Sequence[Any],
    trace: dict[str, Any],
    sanitized: Any,
    unresolved: list[Any],
) -> tuple[bool, list[str]]:
    """Determine whether the compiled G_F represents a generic complete executable task contract.

    This function validates that everything explicitly expressed by the FM is
    well-formed, internally coherent, mapped to valid canonical roles, relations,
    and robot capabilities, and structurally executable. It does NOT check against
    hidden offline reference task expectations.
    """
    missing: list[str] = []

    if sanitized.semantically_incomplete:
        missing.append("Sanitizer reported semantic incompleteness")

    if not nodes:
        missing.append("No valid functional roles compiled")

    if trace.get("unresolved_roles"):
        missing.append(f"Unresolved roles: {trace['unresolved_roles']}")

    if trace.get("disabled_groups"):
        missing.append(f"Disabled/unsupported operations: {trace['disabled_groups']}")

    if trace.get("unresolved_required_operations"):
        missing.append(f"Unresolved required operations: {trace['unresolved_required_operations']}")

    if trace.get("unresolved_required_relations") or unresolved:
        missing.append(f"Unresolved required relations: {trace.get('unresolved_required_relations') or unresolved}")

    for name, node in nodes.items():
        if node.minimum_count < 1:
            missing.append(f"Role {name!r} minimum count must be >= 1, got {node.minimum_count}")
        if node.maximum_count < node.minimum_count:
            missing.append(f"Role {name!r} maximum count ({node.maximum_count}) < minimum count ({node.minimum_count})")
        if node.binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
            missing.append(f"Role {name!r} has invalid binding policy {node.binding_policy!r}")

    for g in groups:
        if getattr(g, "tool_role", None) and g.tool_role not in nodes:
            missing.append(f"Operation group {getattr(g, 'id', 'unknown')} tool role {g.tool_role!r} not in compiled nodes")
        if getattr(g, "target_role", None) and g.target_role not in nodes:
            missing.append(f"Operation group {getattr(g, 'id', 'unknown')} target role {g.target_role!r} not in compiled nodes")
        if not getattr(g, "capability_id", None) and not getattr(g, "function", None):
            missing.append(f"Operation group {getattr(g, 'id', 'unknown')} lacks capability or function")

    for r in relations:
        if r.subject_role not in nodes:
            missing.append(f"Relation {r.predicate!r} subject {r.subject_role!r} not in compiled nodes")
        if r.object_role not in nodes:
            missing.append(f"Relation {r.predicate!r} object {r.object_role!r} not in compiled nodes")

    complete = len(missing) == 0
    return complete, missing


def compile_candidate_graph(domain: str, task: str, raw: dict) -> FunctionalRequirementGraph:
    sanitized = sanitize_functional_graph(raw, domain=domain)
    if not sanitized.succeeded:
        raise VLMSpecificationError('No meaningful functional roles recovered', category='SANITIZER_UNRECOVERABLE')
    doc = sanitized.document
    trace: dict[str, Any] = dict(
        roles=[], properties=[], relations=[], groups=[], context_only_roles=[],
        unresolved_roles=[], merged_roles=[], disambiguated_roles=[], disabled_groups=[],
        unresolved_required_relations=[], unresolved_required_operations=[],
        task_causal_relations=[], role_operation_reconciliations=[],
        provisional_roles=[], provisional_relation_constraints=[],
        provisional_operation_constraints=[], relation_orientation_resolutions=[],
        operation_participant_slot_resolutions=[],
        explicit_context_set_canonicalizations=list(doc.get('explicit_context_sets', [])),
        explicit_role_pairings=[],
    )
    nodes = {}
    id_map = {}
    planner_context_id_map = {}
    planner_context_provenance = {}
    owners = {}
    soft = []
    unresolved = []
    unverified_required = []
    from .system_context_registry import (
        get_domain_planner_context_constants,
        get_domain_selectable_roles,
        get_domain_system_fixed_anchors,
    )
    allowed = set(get_domain_selectable_roles(domain)) | set(get_domain_system_fixed_anchors(domain))
    planner_constants = set(get_domain_planner_context_constants(domain))
    role_hypotheses = resolve_role_type_hypotheses(domain, doc)
    for role in doc['functional_roles']:
        rid = role['id']
        if rid in set(doc.get('current_state_operation_context_roles', [])):
            trace['context_only_roles'].append({
                'raw_role': role,
                'status': 'CURRENT_STATE_OPERATION_CONTEXT_ELIDED',
                'provenance': 'FM_ROLE_FUNCTION_AND_GRAPH_STRUCTURE',
            })
            continue
        hypothesis = role_hypotheses[rid]
        try:
            direct_name, direct_rule = _map_role(domain, role, doc)
        except VLMSpecificationError as exc:
            direct_name, direct_rule = None, str(exc)
        name = hypothesis.resolved_role
        rule = hypothesis.status
        if direct_name == name and direct_name is not None:
            rule = direct_rule
        if role.get('provenance') == 'EXPLICIT_CONTEXT_SET_CANONICALIZATION':
            rule = 'EXPLICIT_CONTEXT_SET_CANONICALIZATION'
        if direct_name in planner_constants:
            name, rule = direct_name, direct_rule
        elif hypothesis.status == 'AMBIGUOUS_ROLE_TYPE' and hypothesis.canonical_role_candidates:
            name = 'fm_role__' + re.sub(r'[^a-zA-Z0-9_]+', '_', rid).strip('_')
            rule = hypothesis.status
        is_provisional = hypothesis.status == 'AMBIGUOUS_ROLE_TYPE' and bool(hypothesis.canonical_role_candidates)
        if name not in allowed and not is_provisional:
            if name in planner_constants:
                planner_context_id_map[rid] = name
                context_provenance = (
                    'EXPLICIT_CONTEXT_SET_CANONICALIZATION'
                    if role.get('provenance') == 'EXPLICIT_CONTEXT_SET_CANONICALIZATION'
                    else 'TASK_EXPRESSED_SYSTEM_CONTEXT'
                    if rule == 'TASK_EXPRESSED_SYSTEM_CONTEXT'
                    else 'PLANNER_CONTEXT_CONSTANT'
                )
                planner_context_provenance[rid] = context_provenance
                trace['context_only_roles'].append({
                    'raw_role': role,
                    'canonical_role': name,
                    'status': context_provenance,
                    'rule': rule,
                })
                continue
            # Only non-manipulated anchors/support context can be context-only.
            is_operation_participant = any(rid in (g.get('tool_role'), g.get('target_role')) for g in doc['interaction_groups'])
            context = role['entity_kind'] in {'FIXED_TARGET', 'REGION'} and not is_operation_participant
            field = 'context_only_roles' if context else 'unresolved_roles'
            trace[field].append({'raw_role': role, 'status': 'CONTEXT_ONLY_ROLE' if context else 'UNRESOLVED_SEMANTIC'})
            continue
        if name in nodes:
            if can_merge_roles(
                owners[name], role, doc,
                allow_counted_context=(domain == 'living_room' and name == 'SEATING_POSITION'),
            ):
                id_map[rid] = name
                trace['merged_roles'].append({'code': 'CONSOLIDATED_EQUIVALENT_FM_INSTANCES', 'raw_ids': [owners[name]['id'], rid], 'canonical_role': name})
                node = nodes[name]
                total = node.count + int(role.get('required_count', 1))
                nodes[name] = replace(
                    node, count=total, min_count=total, max_count=total,
                    binding_policy='DISTINCT',
                    semantic_hints=tuple(dict.fromkeys(node.semantic_hints + tuple(role['required_properties']))),
                )
                # Process additional property evidence below.
            else:
                trace['unresolved_roles'].append({'code': 'AMBIGUOUS_ROLE_MAPPING', 'raw_role': role, 'collision_with': owners[name]['id']})
                continue
        else:
            if name in set(get_domain_system_fixed_anchors(domain)):
                canonical_kind = 'FIXED_TARGET'
            elif name in {'PERSONAL_CUP_SAUCER_REGION', 'SHARED_REMOTE_REGION'}:
                canonical_kind = 'REGION'
            else:
                canonical_kind = role['entity_kind']
            id_map[rid] = name
            owners[name] = role
            role_candidates = hypothesis.canonical_role_candidates or (name,)
            # A candidate may be a planner context constant (a fixed region the
            # planner references, such as a work surface or a serving area) rather
            # than a perception-grounded role.  No domain declares acceptance
            # categories for those, by design, so they contribute none here.
            # Demanding categories for every candidate turned an ordinary role
            # typing into a hard crash that aborted the whole run.
            semantic_categories = tuple(dict.fromkeys(
                category for candidate in role_candidates
                for category in ontology.get_role_semantic_categories_or_empty(domain, candidate)
            ))
            nodes[name] = FunctionalRole(name=name, entity_kind=canonical_kind, count=role['required_count'],
                binding_policy=role['binding_policy'], semantic_categories=semantic_categories,
                description=role.get('description', ''), semantic_hints=tuple(role['required_properties']),
                min_count=role.get('binding_cardinality', {}).get('minimum_distinct_physical_objects', role.get('min_count')),
                max_count=role.get('binding_cardinality', {}).get('maximum_distinct_physical_objects', role.get('max_count')),
                preference=role.get('binding_cardinality', {}).get('preferred', role.get('preference')),
                verification_mode=('GEOMETRIC_ONLY' if domain == 'workshop' and canonical_kind == 'FIXED_TARGET'
                    else 'SEMANTIC_ONLY' if domain != 'workshop' and not role['required_properties'] else 'SEMANTIC_AND_GEOMETRIC'),
                raw_role_id=rid,
                canonical_role_candidates=role_candidates,
                role_resolution_status=hypothesis.status,
                role_resolution_provenance=hypothesis.evidence)
            if role.get('provenance') == 'EXPLICIT_CONTEXT_SET_CANONICALIZATION':
                nodes[name] = replace(
                    nodes[name],
                    role_resolution_status='EXPLICIT_CONTEXT_SET_CANONICALIZATION',
                    role_resolution_provenance=({
                        'source': 'EXPLICIT_CONTEXT_SET_CANONICALIZATION',
                        'member_raw_roles': list(role.get('context_set_members', [])),
                    },),
                )
        role_status = 'PROVISIONAL_ROLE_HYPOTHESIS' if is_provisional else 'CANONICAL_EXECUTABLE_SEMANTIC'
        trace['roles'].append({'raw_id': rid, 'canonical_role': name, 'canonical_role_candidates': list(hypothesis.canonical_role_candidates),
                               'rule': rule, 'status': role_status, 'resolution_status': hypothesis.status})
        if is_provisional:
            trace['provisional_roles'].append(hypothesis.to_dict())
        if rule == 'CAUSAL_SOURCE_PROVIDER':
            trace['disambiguated_roles'].append({'code': 'RAW_ROLE_DISAMBIGUATED', 'raw_id': rid, 'canonical_role': name, 'evidence': sorted(causal_position(role, doc))})
        for prop in role['required_properties']:
            mapped = None
            try:
                if domain == 'kitchen':
                    from mujoco_scenes.kitchen_vlm_functional_graph import map_unary_property
                    mapped = map_unary_property(prop)
                elif domain == 'workshop':
                    from mujoco_scenes.workshop_phase1.requirements import map_workshop_unary_property
                    mapped = map_workshop_unary_property(prop)
                elif re.search(r'\b(planar|flat|horizontal)\b', prop.lower()) and nodes[name].entity_kind == 'REGION':
                    mapped = 'PLANAR_SUPPORT'
                if mapped:
                    validate_predicate_signature(domain=domain, predicate=mapped, subject_kind=nodes[name].entity_kind, subject_role=name)
            except VLMSpecificationError:
                mapped = None
            evidence = {'raw_role_id': rid, 'raw_phrase': prop, 'canonical_predicate': mapped,
                        'status': 'CANONICAL_EXECUTABLE_SEMANTIC' if mapped else 'SOFT_SEMANTIC_EVIDENCE'}
            trace['properties'].append(evidence)
            if mapped:
                nodes[name] = replace(nodes[name], unary_predicates=tuple(dict.fromkeys(nodes[name].unary_predicates + (mapped,))))
            else:
                if re.search(r'\b(must|safety|safe|sterile|insulated|rated|load bearing|heat resistant)\b', prop.lower()):
                    evidence['status'] = 'UNRESOLVED_SEMANTIC'
                    unverified_required.append({'role': name, 'property': prop, 'status': 'UNVERIFIABLE_REQUIRED_PROPERTY'})
                else:
                    soft.append(evidence)
    if not nodes:
        raise VLMSpecificationError('No executable role could be typed', category='CANONICALIZATION_AMBIGUITY')
    relations = []
    task_causal_relations = []
    task_effect_relations = []
    provisional_relations: list[ProvisionalRelationConstraint] = []
    provisional_operations: list[ProvisionalOperationConstraint] = []

    def add_relation(raw_subject: str, phrase: str, raw_target: str, *, grouped=False,
                     expected=True, unordered=False) -> str | None:
        # Extract explicit role IDs if embedded in phrase (e.g. "role_3 manipulates role_2")
        role_match = re.match(r'^(role_\w+)\s+(.+?)\s+(role_\w+)$', phrase.strip())
        if role_match:
            cand_s, cand_phrase, cand_o = role_match.groups()
            if cand_s in id_map and cand_o in id_map:
                raw_subject = cand_s
                raw_target = cand_o
                phrase = cand_phrase.strip()

        evidence = {'raw_subject': raw_subject, 'raw_phrase': phrase, 'raw_object': raw_target}
        try:
            if raw_subject not in id_map or raw_target not in id_map:
                context_endpoint = planner_context_id_map.get(raw_subject) or planner_context_id_map.get(raw_target)
                non_context_endpoint = raw_target if raw_subject in planner_context_id_map else raw_subject
                if context_endpoint and non_context_endpoint in id_map:
                    context_raw = raw_subject if raw_subject in planner_context_id_map else raw_target
                    evidence.update(
                        status='ABSORBED_INTO_PLANNER_CONTEXT',
                        planner_context=context_endpoint,
                        provenance=planner_context_provenance[context_raw],
                    )
                    trace['relations'].append(evidence)
                    return None
                context_ids = {r['raw_role']['id'] for r in trace['context_only_roles']}
                if (raw_subject in id_map or raw_subject in context_ids) and (raw_target in id_map or raw_target in context_ids):
                    evidence.update(status='SOFT_SEMANTIC_EVIDENCE', reason='Context-only support endpoint')
                    soft.append(evidence)
                    trace['relations'].append(evidence)
                    return None
                raise ValueError('Unresolved or context-only endpoint')

            canon_s = id_map[raw_subject]
            canon_o = id_map[raw_target]
            if len(nodes[canon_s].canonical_role_candidates) > 1 or len(nodes[canon_o].canonical_role_candidates) > 1:
                from .relation_interpreter import extract_relation_semantic_candidates
                from .predicate_registry import get_predicate_signature
                semantic_candidates = extract_relation_semantic_candidates(domain, phrase)
                allowed_pairs = []
                for item in semantic_candidates:
                    if item.category == 'PHYSICAL_VERIFIER':
                        signature = get_predicate_signature(domain, item.predicate_name)
                        if signature:
                            pairs = [(s, o) for s in signature.allowed_subject_roles for o in signature.allowed_object_roles]
                            if item.direction == 'REVERSE':
                                pairs = [(o, s) for s, o in pairs]
                            if unordered:
                                pairs += [(o, s) for s, o in pairs]
                            allowed_pairs.extend((s, o, item.predicate_name, item.category) for s, o in pairs)
                    elif item.category == 'TASK_CAUSAL_SEMANTICS':
                        pairs = list(_causal_role_pairs(domain, item.predicate_name))
                        if item.direction == 'REVERSE':
                            pairs = [(o, s) for s, o in pairs]
                        if unordered:
                            pairs += [(o, s) for s, o in pairs]
                        allowed_pairs.extend((s, o, item.predicate_name, item.category) for s, o in pairs)
                evidence.update(
                    status='PROVISIONAL_RELATION_CONSTRAINT',
                    semantic_candidates=[item.__dict__ for item in semantic_candidates],
                )
                constraint = ProvisionalRelationConstraint(
                    raw_subject_role=raw_subject, raw_object_role=raw_target,
                    subject_node=canon_s, object_node=canon_o,
                    semantic_candidates=tuple(item.__dict__ for item in semantic_candidates),
                    allowed_canonical_role_pairs=tuple(sorted(set(allowed_pairs))),
                    expected=expected,
                )
                provisional_relations.append(constraint)
                trace['relations'].append(evidence)
                trace['provisional_relation_constraints'].append(constraint.to_dict())
                return None
            s_kind = nodes[canon_s].entity_kind
            o_kind = nodes[canon_o].entity_kind

            from .relation_interpreter import interpret_relation
            interp = interpret_relation(
                domain=domain,
                raw_phrase=phrase,
                subject_role=canon_s,
                object_role=canon_o,
                subject_kind=s_kind,
                object_kind=o_kind,
                required=expected,
            )

            if interp.succeeded:
                first_pred = None
                for ip in interp.interpreted_predicates:
                    s, p, o = ip.subject_role, ip.predicate_name, ip.object_role
                    if interp.category == "TASK_CAUSAL_SEMANTICS":
                        # Preserved semantic edge for task narrative/causal dependency
                        rel = FunctionalRelation(
                            subject_role=s,
                            predicate=p,
                            object_role=o,
                            expected=expected,
                            provenance="TASK_CAUSAL_SEMANTICS",
                            category="TASK_CAUSAL_SEMANTICS",
                        )
                        if rel not in task_causal_relations:
                            task_causal_relations.append(rel)
                        if first_pred is None:
                            first_pred = p
                    else:
                        fixed_anchors = set(get_domain_system_fixed_anchors(domain))
                        if s not in nodes and s in fixed_anchors:
                            nodes[s] = FunctionalRole(name=s, entity_kind='FIXED_TARGET', count=1, binding_policy='SHARED',
                                                      semantic_categories=ontology.get_role_semantic_categories_or_empty(domain, s),
                                                      verification_mode='GEOMETRIC_ONLY')
                        if o not in nodes and o in fixed_anchors:
                            nodes[o] = FunctionalRole(name=o, entity_kind='FIXED_TARGET', count=1, binding_policy='SHARED',
                                                      semantic_categories=ontology.get_role_semantic_categories_or_empty(domain, o),
                                                      verification_mode='GEOMETRIC_ONLY')
                        validate_predicate_signature(domain=domain, predicate=p, subject_kind=nodes[s].entity_kind,
                            subject_role=s, object_kind=nodes[o].entity_kind, object_role=o)
                        rel = FunctionalRelation(
                            subject_role=s,
                            predicate=p,
                            object_role=o,
                            expected=expected,
                            provenance="EXPLICIT_REQUIREMENT",
                            category="PHYSICAL_VERIFIER",
                        )
                        if not grouped and rel not in relations:
                            relations.append(rel)
                        if first_pred is None:
                            first_pred = p
                evidence.update(
                    status='CANONICAL_EXECUTABLE_SEMANTIC' if interp.category == "PHYSICAL_VERIFIER" else 'TASK_CAUSAL_SEMANTICS',
                    category=interp.category,
                    interp_status=interp.status,
                    canonical=[[ip.subject_role, ip.predicate_name, ip.object_role] for ip in interp.interpreted_predicates],
                    direction_normalized=interp.direction_normalized,
                )
                if interp.category == "TASK_CAUSAL_SEMANTICS":
                    trace['task_causal_relations'].append(evidence)
                if unordered:
                    trace['relation_orientation_resolutions'].append({
                        'raw_participants': [raw_subject, raw_target],
                        'canonical': evidence.get('canonical', []),
                        'direction_normalized': interp.direction_normalized,
                    })
                trace['relations'].append(evidence)
                return first_pred if interp.category == "PHYSICAL_VERIFIER" else None

            # Fail closed: uninterpretable required relation
            raise ValueError(interp.reason or 'No explicit semantic evidence for a required relation')
        except (VLMSpecificationError, ValueError) as exc:
            evidence.update(
                status='UNINTERPRETABLE_REQUIRED_RELATION' if expected else 'UNRESOLVED_SEMANTIC',
                reason=str(exc),
            )
            unresolved.append(evidence)
            if expected and not grouped:
                trace['unresolved_required_relations'].append(evidence)
            trace['relations'].append(evidence)
            return None

    for rel in doc['functional_relations']:
        from .relation_interpreter import (
            has_compatible_explicit_effect_operation,
            has_compatible_explicit_pairing_operation,
            interpret_task_effect_predicate,
        )
        raw_subject = rel['subject_role']
        raw_object = rel['object_role']
        raw_phrase = rel.get('relation', rel.get('predicate'))
        effect_predicate = rel.get('effect_predicate') or interpret_task_effect_predicate(raw_phrase)
        object_is_literal = rel.get('object_is_literal') is True
        pairing_matches, pairing_operation_id = has_compatible_explicit_pairing_operation(
            raw_phrase, raw_subject, raw_object, doc.get('interaction_groups', [])
        )
        if pairing_matches and raw_subject in id_map and raw_object in id_map:
            paired = FunctionalRelation(
                subject_role=id_map[raw_subject],
                predicate='PAIRED_WITH',
                object_role=id_map[raw_object],
                expected=rel.get('expected', True),
                provenance='TASK_CAUSAL_SEMANTICS',
                source_operation_id=pairing_operation_id,
                category='TASK_CAUSAL_SEMANTICS',
            )
            task_causal_relations.append(paired)
            pairing_evidence = {
                'raw_subject': raw_subject,
                'raw_phrase': raw_phrase,
                'raw_object': raw_object,
                'status': 'TASK_CAUSAL_SEMANTICS',
                'category': 'TASK_CAUSAL_SEMANTICS',
                'canonical': paired.to_dict(),
                'provenance': 'FM_EXPLICIT_SEMANTIC',
            }
            trace['relations'].append(pairing_evidence)
            trace['task_causal_relations'].append(pairing_evidence)
            continue
        operation_matches, source_operation_id = has_compatible_explicit_effect_operation(
            effect_predicate or '', raw_subject, raw_object, doc.get('interaction_groups', [])
        )
        if (
            effect_predicate
            and raw_subject in id_map
            and ((raw_object in id_map and operation_matches) or object_is_literal)
        ):
            effect = TaskEffectRelation(
                subject_role=id_map[raw_subject],
                predicate=effect_predicate,
                object_value=(raw_object if object_is_literal else id_map[raw_object]),
                object_is_literal=object_is_literal,
                source_operation_id=source_operation_id,
                raw_subject=raw_subject,
                raw_phrase=raw_phrase,
                raw_object=raw_object,
            )
            task_effect_relations.append(effect)
            trace['relations'].append({
                'raw_subject': raw_subject,
                'raw_phrase': raw_phrase,
                'raw_object': raw_object,
                'status': 'TASK_EFFECT_SEMANTICS',
                'category': 'TASK_EFFECT_SEMANTICS',
                'canonical': effect.to_dict(),
                'provenance': 'FM_EXPLICIT_SEMANTIC',
            })
            continue
        add_relation(
            rel['subject_role'], rel.get('relation', rel.get('predicate')), rel['object_role'],
            expected=rel.get('expected', True), unordered=rel.get('unordered_participants', False),
        )
    from .robot_capability_registry import interpret_operation
    raw_groups = doc.get('interaction_groups') or doc.get('operations') or []
    groups = []
    for group in raw_groups:
        tool_raw = group.get('tool_role') or group.get('source_role')
        target_raw = group.get('target_role')
        ctx_raw = group.get('context_role') or group.get('anchor_role')
        raw_op = group.get('function') or group.get('operation') or ''
        v3_slot_assignments = group.get('v3_slot_assignments', [])
        v3_participants = group.get('v3_participant_roles', [])

        if len(v3_slot_assignments) > 1:
            capabilities = extract_operation_semantic_candidates(domain, raw_op)
            capability_records = [{
                'capability_id': capability.capability_id,
                'planner_operation': capability.planner_operation,
                'allowed_source_roles': list(capability.allowed_source_roles),
                'allowed_target_roles': list(capability.allowed_target_roles),
                'allowed_anchor_roles': list(capability.allowed_anchor_roles),
                'required_relation_templates': [list(row) for row in capability.required_relation_templates],
            } for capability in capabilities]
            first = v3_slot_assignments[0]
            count = group.get('required_target_count', group.get('operation_count', 1))
            constraint = ProvisionalOperationConstraint(
                raw_operation_id=group.get('id', raw_op), raw_source_role='', raw_target_role='',
                raw_anchor_role=None, source_node='', target_node='', anchor_node=None,
                capability_candidates=tuple(capability_records), required_count=count,
                reuse_policy=first['usage_policy'],
                slot_assignments=tuple({
                    **row,
                    'source_node': id_map[row['source_role']],
                    'target_node': id_map[row['target_role']],
                    'anchor_node': (
                        id_map.get(row.get('anchor_role'))
                        or planner_context_id_map.get(row.get('anchor_role'))
                    ) if row.get('anchor_role') else None,
                } for row in v3_slot_assignments),
            )
            provisional_operations.append(constraint)
            trace['groups'].append({'raw_group': group, 'status': 'PROVISIONAL_OPERATION_CONSTRAINT',
                                    'capability_candidates': [c.capability_id for c in capabilities]})
            trace['provisional_operation_constraints'].append(constraint.to_dict())
            continue

        # Explicit final placement/restoration to a registered planner constant
        # is audited but is not a selectable G_F operation.  The domain planner
        # owns that context transition.  Other operations with unresolved
        # context endpoints still fail closed below.
        participant_contexts = [p for p in v3_participants if p in planner_context_id_map]
        participant_objects = [p for p in v3_participants if p in id_map]
        if (
            len(participant_contexts) == 1 and participant_objects
            and re.search(r'\b(place|return|leave|restore|set down|put|deposit|store|release|transport|deliver)\b', re.sub(r'[_-]+', ' ', raw_op.lower()))
        ):
            trace['groups'].append({
                'raw_group': group, 'status': 'ABSORBED_INTO_PLANNER_CONTEXT',
                'planner_context': planner_context_id_map[participant_contexts[0]],
                'provenance': planner_context_provenance[participant_contexts[0]],
            })
            continue
        context_endpoint = planner_context_id_map.get(tool_raw) or planner_context_id_map.get(target_raw)
        non_context_endpoint = target_raw if tool_raw in planner_context_id_map else tool_raw
        if (
            context_endpoint
            and non_context_endpoint in id_map
            and re.search(r'\b(place|return|leave|restore|set down|put|deposit|store|release)\b', raw_op.lower())
        ):
            context_raw = tool_raw if tool_raw in planner_context_id_map else target_raw
            trace['groups'].append({
                'raw_group': group,
                'status': 'ABSORBED_INTO_PLANNER_CONTEXT',
                'planner_context': context_endpoint,
                'provenance': planner_context_provenance[context_raw],
            })
            continue

        if not tool_raw or not target_raw or tool_raw not in id_map or target_raw not in id_map:
            if v3_participants:
                trace['unresolved_required_operations'].append({
                    'raw_group': group, 'status': 'UNSUPPORTED_RUNTIME_OPERATION_SEMANTIC',
                    'reason': 'V3 participant order cannot supply source/target slots',
                })
            trace['disabled_groups'].append({'raw_group': group, 'status': 'UNINSTANTIABLE_MISSING_ROLE'})
            continue

        if (
            len(nodes[id_map[tool_raw]].canonical_role_candidates) > 1
            or len(nodes[id_map[target_raw]].canonical_role_candidates) > 1
        ):
            capabilities = extract_operation_semantic_candidates(domain, raw_op)
            capability_records = []
            for capability in capabilities:
                capability_records.append({
                    'capability_id': capability.capability_id,
                    'planner_operation': capability.planner_operation,
                    'allowed_source_roles': list(capability.allowed_source_roles),
                    'allowed_target_roles': list(capability.allowed_target_roles),
                    'allowed_anchor_roles': list(capability.allowed_anchor_roles),
                    'required_relation_templates': [list(row) for row in capability.required_relation_templates],
                })
            count = group.get('required_target_count', group.get('operation_count', 1))
            policy = group.get('usage_policy') or group.get('reuse_policy') or 'DEDICATED_PER_TARGET'
            if policy == 'REUSABLE_ACROSS_TARGETS':
                policy = 'SEQUENTIAL_REUSE_ALLOWED'
            constraint = ProvisionalOperationConstraint(
                raw_operation_id=group.get('id', raw_op),
                raw_source_role=tool_raw, raw_target_role=target_raw,
                raw_anchor_role=ctx_raw, source_node=id_map[tool_raw],
                target_node=id_map[target_raw], anchor_node=id_map.get(ctx_raw) if ctx_raw else None,
                capability_candidates=tuple(capability_records), required_count=count,
                reuse_policy=policy,
                slot_assignments=tuple({
                    **row,
                    'source_node': id_map[row['source_role']],
                    'target_node': id_map[row['target_role']],
                    'anchor_node': id_map.get(row.get('anchor_role')) if row.get('anchor_role') else None,
                } for row in v3_slot_assignments),
            )
            provisional_operations.append(constraint)
            trace['groups'].append({
                'raw_group': group,
                'status': 'PROVISIONAL_OPERATION_CONSTRAINT',
                'capability_candidates': [c.capability_id for c in capabilities],
            })
            trace['provisional_operation_constraints'].append(constraint.to_dict())
            continue

        if v3_slot_assignments:
            trace['operation_participant_slot_resolutions'].append({
                'operation_id': group.get('id'),
                'participant_roles': sorted({
                    v3_slot_assignments[0]['source_role'], v3_slot_assignments[0]['target_role'],
                    *([v3_slot_assignments[0]['anchor_role']] if v3_slot_assignments[0].get('anchor_role') else []),
                }),
                'source_role': tool_raw, 'target_role': target_raw, 'anchor_role': ctx_raw,
                'capability_id': v3_slot_assignments[0]['capability_id'],
                'usage_policy': v3_slot_assignments[0]['usage_policy'],
            })

        tool_role_id = id_map[tool_raw]
        target_role_id = id_map[target_raw]
        ctx_role_id = (
            id_map.get(ctx_raw) or planner_context_id_map.get(ctx_raw)
        ) if ctx_raw else None

        if ctx_raw in planner_context_id_map and ctx_role_id not in nodes:
            nodes[ctx_role_id] = FunctionalRole(
                name=ctx_role_id,
                entity_kind='FIXED_TARGET',
                count=1,
                binding_policy='SHARED',
                semantic_categories=ontology.get_role_semantic_categories_or_empty(domain, ctx_role_id),
                verification_mode='GEOMETRIC_ONLY',
                role_resolution_status='EXPLICIT_CONTEXT_SET_CANONICALIZATION',
                canonical_role_candidates=(ctx_role_id,),
                role_resolution_provenance=({
                    'source': 'EXPLICIT_CONTEXT_SET_CANONICALIZATION',
                    'raw_context_role': ctx_raw,
                },),
            )

        if domain == 'living_room':
            if nodes[tool_role_id].entity_kind == 'OBJECT' and nodes[target_role_id].entity_kind == 'REGION':
                tool_role_id, target_role_id = target_role_id, tool_role_id
                tool_raw, target_raw = target_raw, tool_raw

        is_v2 = is_v2_document(raw)
        if not ctx_role_id and not is_v3_document(raw):
            semantic_capabilities = extract_operation_semantic_candidates(domain, raw_op)
            capability_anchors = {
                anchor for capability in semantic_capabilities
                for anchor in capability.allowed_anchor_roles
            }
            from .predicate_registry import get_predicate_signature
            capability_anchors = {
                anchor for anchor in capability_anchors
                if all(
                    not (subj_key == 'anchor' or obj_key == 'anchor')
                    or (
                        (signature := get_predicate_signature(domain, predicate)) is not None
                        and (subj_key != 'anchor' or not signature.allowed_subject_roles or anchor in signature.allowed_subject_roles)
                        and (obj_key != 'anchor' or not signature.allowed_object_roles or anchor in signature.allowed_object_roles)
                    )
                    for capability in semantic_capabilities
                    for subj_key, predicate, obj_key in capability.required_relation_templates
                )
            }
            fixed_anchors = set(get_domain_system_fixed_anchors(domain))
            if len(capability_anchors) == 1 and capability_anchors <= fixed_anchors:
                ctx_role_id = next(iter(capability_anchors))
                if ctx_role_id not in nodes:
                    nodes[ctx_role_id] = FunctionalRole(
                        name=ctx_role_id, entity_kind='FIXED_TARGET', count=1,
                        binding_policy='SHARED',
                        semantic_categories=ontology.get_role_semantic_categories_or_empty(domain, ctx_role_id),
                        verification_mode='GEOMETRIC_ONLY',
                        role_resolution_status='OPERATION_ASSISTED',
                        canonical_role_candidates=(ctx_role_id,),
                        role_resolution_provenance=({
                            'source': 'ROBOT_CAPABILITY_SIGNATURE',
                            'capability_ids': [cap.capability_id for cap in semantic_capabilities],
                        },),
                    )

        if is_v3_document(raw) and not ctx_role_id:
            explicit_capability_ids = {
                row.get('capability_id') for row in v3_slot_assignments if row.get('capability_id')
            }
            requires_explicit_context = any(
                capability.capability_id in explicit_capability_ids and capability.allowed_anchor_roles
                for capability in extract_operation_semantic_candidates(domain, raw_op)
            )
            if requires_explicit_context:
                trace['disabled_groups'].append({
                    'raw_group': group,
                    'status': 'MISSING_EXPLICIT_OPERATION_CONTEXT',
                    'reason': 'V3 operation capability requires FM-expressed anchor/context evidence',
                })
                continue

        # Slot resolution may already have identified the capability from the
        # participant signature when the phrase carried no text evidence.  Pass
        # it through rather than re-deriving from the same uninformative phrase.
        hinted = {
            row.get("capability_id") for row in (v3_slot_assignments or [])
            if row.get("capability_id")
        }
        op_interp = interpret_operation(
            domain=domain,
            raw_phrase=raw_op,
            source_role=tool_role_id,
            target_role=target_role_id,
            anchor_role=ctx_role_id,
            capability_hint=next(iter(hinted)) if len(hinted) == 1 else None,
        )

        if not op_interp.succeeded:
            trace['disabled_groups'].append({
                'raw_group': group,
                'status': 'UNSUPPORTED_OPERATOR',
                'reason': op_interp.reason,
                'interp_status': op_interp.status,
            })
            continue

        required = [
            mapped for phrase in group.get('required_relations', [])
            if (mapped := add_relation(
                tool_raw, phrase, target_raw, grouped=True
            )) is not None
        ]
        context = []
        if ctx_raw:
            for phrase in group.get('context_relations', []):
                mapped = add_relation(tool_raw, phrase, ctx_raw, grouped=True)
                if mapped is None and target_raw:
                    mapped = add_relation(target_raw, phrase, ctx_raw, grouped=True)
                if mapped is not None:
                    context.append(mapped)

        # Union explicit relations or fallback to physical preconditions from mapped capability
        all_required = required if required else list(op_interp.required_relations)
        all_context = context if context else list(op_interp.context_relations)

        count = group.get('required_target_count') if group.get('required_target_count') is not None else group.get('operation_count', 1)
        policy = group.get('usage_policy') or group.get('reuse_policy')
        if policy == 'REUSABLE_ACROSS_TARGETS':
            policy = 'SEQUENTIAL_REUSE_ALLOWED'
        elif policy is None:
            policy = 'DEDICATED_PER_TARGET'

        if (
            policy == 'DEDICATED_PER_TARGET'
            and isinstance(count, int)
            and count > nodes[tool_role_id].maximum_count
        ):
            trace['disabled_groups'].append({
                'raw_group': group,
                'status': 'INCONSISTENT_OPERATION_REUSE_CARDINALITY',
                'reason': (
                    f'Dedicated operation count {count} exceeds source role '
                    f'{tool_role_id!r} maximum distinct count '
                    f'{nodes[tool_role_id].maximum_count}'
                ),
            })
            continue

        if not count or type(count) is not int or count < 1 or count > nodes[target_role_id].maximum_count or policy not in {'DEDICATED_PER_TARGET', 'SEQUENTIAL_REUSE_ALLOWED'}:
            trace['disabled_groups'].append({'raw_group': group, 'status': 'UNSUPPORTED_OPERATOR', 'reason': f'Invalid count or policy: count={count}, policy={policy}'})
            continue

        # Singleton interaction requirements in Workshop:
        if domain == 'workshop' and count == 1:
            for phrase in group.get('required_relations', []):
                add_relation(tool_raw, phrase, target_raw)
            for phrase in group.get('context_relations', []):
                add_relation(tool_raw, phrase, ctx_raw)
            preconditions_provenance = []
            for s_r, p, o_r in op_interp.physical_preconditions:
                fixed_anchors = set(get_domain_system_fixed_anchors(domain))
                if s_r not in nodes and s_r in fixed_anchors:
                    nodes[s_r] = FunctionalRole(name=s_r, entity_kind='FIXED_TARGET', count=1, binding_policy='SHARED',
                                              semantic_categories=ontology.get_role_semantic_categories_or_empty(domain, s_r),
                                              verification_mode='GEOMETRIC_ONLY')
                if o_r not in nodes and o_r in fixed_anchors:
                    nodes[o_r] = FunctionalRole(name=o_r, entity_kind='FIXED_TARGET', count=1, binding_policy='SHARED',
                                              semantic_categories=ontology.get_role_semantic_categories_or_empty(domain, o_r),
                                              verification_mode='GEOMETRIC_ONLY')
                validate_predicate_signature(domain=domain, predicate=p, subject_kind=nodes[s_r].entity_kind,
                    subject_role=s_r, object_kind=nodes[o_r].entity_kind, object_role=o_r)
                source_op_id = group.get('id', op_interp.planner_operation or "FASTEN_JOINT")
                cap_id = op_interp.capability.capability_id if op_interp.capability else None
                rel = FunctionalRelation(
                    subject_role=s_r,
                    predicate=p,
                    object_role=o_r,
                    expected=True,
                    provenance="ROBOT_CAPABILITY_PRECONDITION",
                    source_operation_id=source_op_id,
                    capability_id=cap_id,
                    category="PHYSICAL_VERIFIER",
                )
                if not any(r.subject_role == s_r and r.predicate == p and r.object_role == o_r for r in relations):
                    relations.append(rel)
                preconditions_provenance.append({
                    "predicate": p,
                    "subject_role": s_r,
                    "object_role": o_r,
                    "provenance": "ROBOT_CAPABILITY_PRECONDITION",
                    "source_operation_id": source_op_id,
                    "capability_id": cap_id,
                })

        runtime_function = op_interp.planner_operation
        canonical_group_id = {
            'STIR_COFFEE': 'coffee_stirring',
            'PROVIDE_SOUP_EATING_UTENSIL': 'soup_serving',
            'SUPPORT_DRINKWARE': 'personal_support_group',
            'SUPPORT_ENTERTAINMENT_CONTROL': 'shared_entertainment_group',
            'DRIVE_FASTENER_INTO_TARGET': 'drive_fastener_group',
            'POUR': 'material_transfer',
            'PLACE': 'equipment_return',
        }.get(runtime_function, group.get('id', runtime_function))
        g_id = canonical_group_id if not any(g.id == canonical_group_id for g in groups) else group.get('id', canonical_group_id)

        executable_context_role = ctx_role_id if all_context else None
        usage_policy = policy
        if domain == 'living_room':
            if executable_context_role == 'SEATING_POSITION' and 'ACCESSIBLE_FROM_BOTH_SEATS' in all_context:
                executable_context_role = 'SEATING_PAIR'
                if executable_context_role not in nodes:
                    nodes[executable_context_role] = FunctionalRole(
                        name=executable_context_role,
                        entity_kind='FIXED_TARGET',
                        count=1,
                        binding_policy='SHARED',
                        semantic_categories=ontology.get_role_semantic_categories_or_empty(
                            domain, executable_context_role
                        ),
                        verification_mode='GEOMETRIC_ONLY',
                    )
            if runtime_function == 'SUPPORT_DRINKWARE' or (op_interp.capability and op_interp.capability.capability_id == 'SUPPORT_DRINKWARE'):
                usage_policy = 'DEDICATED_PER_TARGET'
        source_node = nodes[tool_role_id]
        if (
            usage_policy == 'SEQUENTIAL_REUSE_ALLOWED'
            and source_node.binding_policy == 'DISTINCT'
            and source_node.minimum_count > 1
        ):
            trace['role_operation_reconciliations'].append({
                'code': 'ROLE_OPERATION_REUSE_RECONCILED',
                'raw_role_id': tool_raw,
                'canonical_role': tool_role_id,
                'role_count': source_node.count,
                'role_minimum_count': source_node.minimum_count,
                'role_maximum_count': source_node.maximum_count,
                'role_binding_policy': source_node.binding_policy,
                'operation_id': group.get('id', runtime_function),
                'operation_target_count': count,
                'operation_usage_policy': usage_policy,
                'resolution': 'PRESERVE_ROLE_DISTINCTNESS_REUSE_REMAINS_OPTIONAL',
            })
        group_op_id = group.get('id', runtime_function)
        cap_id = op_interp.capability.capability_id if op_interp.capability else None
        grp_precond_provenance = [
            {
                "predicate": p,
                "provenance": "ROBOT_CAPABILITY_PRECONDITION",
                "source_operation_id": group_op_id,
                "capability_id": cap_id,
            }
            for p in all_required + all_context
        ]
        groups.append(OperationGroup(
            id=g_id,
            function=runtime_function,
            tool_role=tool_role_id,
            target_role=target_role_id,
            required_target_count=count,
            usage_policy=usage_policy,
            required_relations=tuple(dict.fromkeys(all_required)),
            context_role=executable_context_role,
            context_relations=tuple(dict.fromkeys(all_context)),
            distinct_within_group=group.get('distinct_within_group', usage_policy == 'DEDICATED_PER_TARGET'),
            same_tool_must_cover_all_targets=group.get('same_tool_must_cover_all_targets', False),
            selection_preference=group.get('selection_preference', ('minimize_distinct_tools' if usage_policy == 'SEQUENTIAL_REUSE_ALLOWED' else 'deterministic_rank') if domain == 'kitchen' else None),
            capability_id=cap_id,
            preconditions_provenance=tuple(grp_precond_provenance),
            physical_preconditions=op_interp.physical_preconditions,
        ))
        if domain == 'living_room' and group.get('v3_explicit_participant_roles'):
            trace['explicit_role_pairings'].append({
                'operation_id': group.get('id'),
                'raw_participant_roles': list(group['v3_explicit_participant_roles']),
                'raw_fm_participant_roles': list(group.get('v3_raw_fm_participant_roles', group['v3_explicit_participant_roles'])),
                'current_state_context_roles': list(group.get('v3_current_state_context_roles', [])),
                'graph_join_relation_id': group.get('v3_graph_join_relation_id'),
                'canonical_source_role': tool_role_id,
                'canonical_target_role': target_role_id,
                'canonical_context_role': executable_context_role,
                'context_set_id': group.get('v3_explicit_context_set_id'),
                'provenance': 'FM_EXPLICIT_OPERATION',
            })
        trace['groups'].append({
            'raw_group': group,
            'status': 'CANONICAL_OPERATION_GROUP',
            'capability_id': cap_id,
            'planner_operation': runtime_function,
            'required_relations': all_required,
            'context_relations': all_context,
            'interp_status': op_interp.status,
            'preconditions_provenance': grp_precond_provenance,
        })
    if domain == 'living_room':
        # Separate FM operations may preserve explicit one-to-one pairings while
        # canonicalization consolidates their equivalent role types.  Execute
        # those as one counted capability group; keeping duplicate groups would
        # falsely demand disjoint copies of the same consolidated source set.
        merged_groups: list[OperationGroup] = []
        for group in groups:
            key = (
                group.capability_id, group.function, group.tool_role,
                group.target_role, group.context_role, group.required_relations,
                group.context_relations, group.usage_policy,
            )
            prior_index = next((i for i, existing in enumerate(merged_groups) if (
                existing.capability_id, existing.function, existing.tool_role,
                existing.target_role, existing.context_role, existing.required_relations,
                existing.context_relations, existing.usage_policy,
            ) == key), None)
            if prior_index is None:
                merged_groups.append(group)
                continue
            prior = merged_groups[prior_index]
            combined_count = prior.required_target_count + group.required_target_count
            merged_groups[prior_index] = replace(
                prior,
                required_target_count=combined_count,
                preconditions_provenance=tuple(prior.preconditions_provenance + group.preconditions_provenance),
            )
            trace['groups'].append({
                'status': 'CONSOLIDATED_EQUIVALENT_OPERATION_INSTANCES',
                'canonical_group_id': prior.id,
                'merged_group_id': group.id,
                'required_target_count': combined_count,
                'provenance': 'FM_EXPLICIT_OPERATION_PAIRINGS',
            })
        groups = merged_groups
        # Group pairing governs these edges, not unconstrained all-to-all checks.
        grouped_triples = {(g.tool_role, p, g.target_role) for g in groups for p in g.required_relations}
        grouped_triples |= {(g.tool_role, p, g.context_role) for g in groups for p in g.context_relations}
        relations = [r for r in relations if (r.subject_role, r.predicate, r.object_role) not in grouped_triples]
    # Search region discovery/order is handled separately from functional typing.
    proposed, ranking = [], []
    if domain != 'living_room':
        if domain == 'kitchen':
            from mujoco_scenes.kitchen_vlm_functional_graph import resolve_kitchen_region_proposal as resolve
        else:
            from mujoco_scenes.workshop_phase1.requirements import resolve_workshop_region_proposal as resolve
        region_ids = {}
        for proposal in doc['inspectable_regions']:
            try:
                mapped = resolve(proposal)
            except (VLMSpecificationError, TypeError, AttributeError):
                mapped = None
            if mapped:
                if mapped not in proposed:
                    proposed.append(mapped)
                if isinstance(proposal, dict):
                    region_ids[proposal.get('id')] = mapped
        ranking = list(dict.fromkeys(region_ids[r] for r in doc['inspection_order'] if isinstance(r, str) and r in region_ids))
        ranking.extend(r for r in proposed if r not in ranking)
    partial = sanitized.semantically_incomplete or bool(unverified_required or unresolved or trace['unresolved_roles'] or trace['disabled_groups'])
    # Semantics the converter could not represent must block completeness.  An
    # operation the runtime cannot seat, or a relation it cannot orient, is
    # recorded rather than fatal so the rest of the contract survives -- but the
    # task is not complete without it, and reporting otherwise would be a false
    # completion.
    # A constraint only blocks the task when it constrains something the task
    # actually contains.  A relation or operation over a raw role that resolved
    # to no canonical runtime role is constraining a participant the runtime
    # never took on -- typically a material the model inferred from an end
    # product -- so it is surplus rather than an unmet requirement.  When every
    # participant did resolve, the constraint is over real task roles and its
    # failure is a genuine gap that must block.
    def _constrains_resolved_roles(item):
        participants = item.get("participant_roles") or ()
        return bool(participants) and all(p in id_map for p in participants)

    blocking_ops = [
        item for item in doc.get("unresolved_operation_semantics", ())
        if _constrains_resolved_roles(item)
    ]
    blocking_rels = [
        item for item in doc.get("unresolved_relation_semantics", ())
        if _constrains_resolved_roles(item)
    ]
    surplus_constraints = [
        item for item in
        (*doc.get("unresolved_operation_semantics", ()), *doc.get("unresolved_relation_semantics", ()))
        if not _constrains_resolved_roles(item)
    ]
    if blocking_ops:
        trace.setdefault("unresolved_required_operations", []).extend(blocking_ops)
    if blocking_rels:
        trace.setdefault("unresolved_required_relations", []).extend(blocking_rels)
    trace["surplus_unrepresentable_constraints"] = surplus_constraints
    # An uninterpretable relation contributes no constraint the runtime can act
    # on.  Whether that is fatal depends on whether the task already determines
    # the outcome for its participants: the operations are the authoritative
    # statement of what must happen, and relations are supporting constraints.
    # When every participant is already handled by a compiled operation, an
    # uninterpretable relation is redundant description -- typically a container
    # "containing" the material an expressed transfer already puts there.  When
    # no operation touches the participants, nothing in the task addresses the
    # relation and it remains a genuine unmet requirement.
    def _constrains_only_real_roles(evidence) -> bool:
        endpoints = [evidence.get("raw_subject"), evidence.get("raw_object")]
        named = [e for e in endpoints if e]
        return bool(named) and all(id_map.get(e) is not None for e in named)

    # An uninterpretable relation between two roles the runtime actually took on
    # is a real gap and must block: the runtime was asked for a constraint it
    # cannot express over participants it does own.  One with an endpoint that
    # never became a runtime role is constraining something outside the task --
    # typically a material inferred from an end product -- and cannot be an unmet
    # requirement of a task that does not contain it.
    blocking_unresolved = [e for e in unresolved if _constrains_only_real_roles(e)]
    trace["constraints_outside_the_runtime_task"] = [
        e for e in unresolved if not _constrains_only_real_roles(e)
    ]
    contract_complete, contract_missing_reasons = check_required_contract_complete(
        domain, nodes, relations, groups, trace, sanitized, blocking_unresolved
    )
    all_precond_prov = [
        p for g_trace in trace['groups']
        for p in g_trace.get('preconditions_provenance', [])
    ]
    trace['precondition_provenance'] = all_precond_prov
    graph = FunctionalRequirementGraph(domain=domain, task_instruction=task, nodes=nodes, relations=tuple(relations),
        task_causal_relations=tuple(task_causal_relations),
        task_effect_relations=tuple(task_effect_relations),
        operation_groups=tuple(groups), source='VLM_CANONICAL_G_F', candidate_regions=tuple(proposed), region_ranking=tuple(ranking),
        provisional_relation_constraints=tuple(provisional_relations),
        provisional_operation_constraints=tuple(provisional_operations),
        detector_vocabulary=tuple(dict.fromkeys([c for r in doc['functional_roles'] for c in r['candidate_categories']] + [c.replace('_', ' ') for n in nodes.values() if n.entity_kind == 'OBJECT' for c in n.semantic_categories])),
        cross_group_reuse_allowed=doc.get('cross_group_reuse_allowed', domain == 'workshop' and not groups) is True,
        metadata={'role_semantic_ontology_version': ontology.PHASE3_ROLE_SEMANTIC_ONTOLOGY_VERSION,
            'semantic_acceptance_source': 'SYSTEM_ROLE_SEMANTIC_ONTOLOGY',
            'detector_vocabulary_source': 'VLM_CANDIDATES_PLUS_RELEVANT_SYSTEM_ALIASES',
            'candidate_categories_used_for_role_identity': True,
            'candidate_categories_used_for_grounding_acceptance': False,
            'candidate_categories_used_for_detector_vocabulary': True,
            'raw_vlm_response': raw, 'raw_decomposition': raw, 'structural_sanitizer': sanitized.to_dict(),
            'canonicalization_trace': trace, 'canonicalization_status': 'PARTIAL' if partial else 'FULL',
            'required_contract_complete': contract_complete,
            'online_executable_contract_complete': contract_complete,
            'contract_missing_reasons': contract_missing_reasons,
            'is_v2_specification': is_v2_document(raw),
            'is_v3_specification': is_v3_document(raw),
            'precondition_provenance': all_precond_prov,
            'task_causal_relations': [r.to_dict() for r in task_causal_relations],
            'explicit_context_sets': list(doc.get('explicit_context_sets', [])),
            'explicit_role_pairings': trace['explicit_role_pairings'],
            'role_operation_consistency_audit': list(doc.get('role_operation_consistency_audit', [])),
            'functional_constraint_interpretation': list(doc.get('functional_constraint_interpretation', [])),
            'fm_semantic_accounting': list(doc.get('fm_semantic_accounting', [])),
            'current_state_operation_context_roles': list(doc.get('current_state_operation_context_roles', [])),
            'task_effect_relations': [r.to_dict() for r in task_effect_relations],
            'role_type_hypotheses': {rid: hypothesis.to_dict() for rid, hypothesis in role_hypotheses.items()},
            'provisional_relation_constraints': trace['provisional_relation_constraints'],
            'soft_semantic_evidence': soft, 'unresolved_semantics': unresolved, 'unverified_required_properties': unverified_required, 'raw_role_to_canonical': id_map})
    from pathlib import Path
    if domain == 'living_room':
        graph.metadata['semantic_vocabulary_path'] = str(Path(__file__).resolve().parents[1] / 'configs/l2_integrated_region_function_semantic_vocabulary.yaml')
    elif domain == 'workshop':
        from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract
        vocabulary = ManualWorkshopFMContract()
        graph.metadata['alias_to_canonical'] = vocabulary.get_alias_to_canonical_map()
        graph.metadata['detector_label_to_canonical'] = vocabulary.get_detector_label_to_canonical_map()
    graph.validate()
    from .task_interface_validator import validate_runtime_gf
    validate_runtime_gf(graph)
    return graph
