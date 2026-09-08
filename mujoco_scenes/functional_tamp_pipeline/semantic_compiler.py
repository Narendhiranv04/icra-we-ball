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
    TaskEffectRelation,
)
from .structural_sanitizer import sanitize_functional_graph
from . import role_semantic_ontology as ontology
from .predicate_registry import validate_predicate_signature
from .fm_schema_v2 import is_v2_document
from .errors import VLMSpecificationError


def causal_position(role: dict, document: dict) -> set[str]:
    """Causal types from role language and graph position, without material names."""
    text = re.sub(r"[_-]", " ", f"{role.get('function', '')} {role.get('description', '')}").lower()
    positions = set()
    if re.search(r"\b(supplies|supply|source|provider|ingredient|powder|granules|raw material)\b", text):
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
        phrase = str(relation.get('relation', relation.get('predicate', ''))).lower()
        if re.search(r"\b(supplies|pours into|transfers to|provides material to)\b", phrase):
            if relation.get('subject_role') == rid:
                positions.add('source')
            if relation.get('object_role') == rid:
                positions.add('destination')
    return positions


def can_merge_roles(a: dict, b: dict, document: dict) -> bool:
    if any(a.get(k) != b.get(k) for k in ('entity_kind', 'binding_policy', 'required_count', 'binding_cardinality', 'min_count', 'max_count')):
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
    normalize = lambda x: re.sub(r'\W+', ' ', (x or '').lower()).strip()
    return normalize(a.get('function', '')) == normalize(b.get('function', '')) and normalize(a.get('description', '')) == normalize(b.get('description', ''))


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
            for material in ('coffee', 'water'):
                if re.search(r'\b' + material + r'\b', text):
                    return material + '_source', 'CAUSAL_SOURCE_PROVIDER'
        return map_kitchen_role_function(enriched), 'DOMAIN_SEMANTICS'
    if domain == 'living_room':
        from mujoco_scenes.environment_vlm_requirements import (map_living_room_role_function,
            map_living_room_object_payload_role, map_living_room_fixed_target_role)
        seating = map_living_room_fixed_target_role(role)
        if seating:
            return seating, 'FIXED_TARGET_SEMANTICS'
        mapper = {'REGION': map_living_room_role_function, 'OBJECT': map_living_room_object_payload_role,
                  'FIXED_TARGET': map_living_room_fixed_target_role}.get(role['entity_kind'], map_living_room_role_function)
        mapped = mapper(role)
        if mapped is None and role['entity_kind'] == 'REGION':
            function = (role.get('function', '') + ' ' + role.get('description', '')).lower()
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
    mapped = map_workshop_role_function(role)
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
    sanitized = sanitize_functional_graph(raw)
    if not sanitized.succeeded:
        raise VLMSpecificationError('No meaningful functional roles recovered', category='SANITIZER_UNRECOVERABLE')
    doc = sanitized.document
    trace: dict[str, Any] = dict(roles=[], properties=[], relations=[], groups=[], context_only_roles=[],
                                unresolved_roles=[], merged_roles=[], disambiguated_roles=[], disabled_groups=[],
                                unresolved_required_relations=[], unresolved_required_operations=[],
                                task_causal_relations=[], role_operation_reconciliations=[])
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
    for role in doc['functional_roles']:
        rid = role['id']
        try:
            name, rule = _map_role(domain, role, doc)
        except VLMSpecificationError as exc:
            name, rule = None, str(exc)
        if name not in allowed:
            if name in planner_constants:
                planner_context_id_map[rid] = name
                context_provenance = (
                    'TASK_EXPRESSED_SYSTEM_CONTEXT'
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
            if can_merge_roles(owners[name], role, doc):
                id_map[rid] = name
                trace['merged_roles'].append({'code': 'RAW_ROLE_MERGED', 'raw_ids': [owners[name]['id'], rid], 'canonical_role': name})
                node = nodes[name]
                nodes[name] = replace(node, semantic_hints=tuple(dict.fromkeys(node.semantic_hints + tuple(role['required_properties']))))
                # Process additional property evidence below.
            else:
                trace['unresolved_roles'].append({'code': 'AMBIGUOUS_ROLE_MAPPING', 'raw_role': role, 'collision_with': owners[name]['id']})
                continue
        else:
            canonical_kind = 'FIXED_TARGET' if name in set(get_domain_system_fixed_anchors(domain)) else role['entity_kind']
            id_map[rid] = name
            owners[name] = role
            nodes[name] = FunctionalRole(name=name, entity_kind=canonical_kind, count=role['required_count'],
                binding_policy=role['binding_policy'], semantic_categories=ontology.get_system_role_semantic_categories(domain, name),
                description=role.get('description', ''), semantic_hints=tuple(role['required_properties']),
                min_count=role.get('binding_cardinality', {}).get('minimum_distinct_physical_objects', role.get('min_count')),
                max_count=role.get('binding_cardinality', {}).get('maximum_distinct_physical_objects', role.get('max_count')),
                preference=role.get('binding_cardinality', {}).get('preferred', role.get('preference')),
                verification_mode=('GEOMETRIC_ONLY' if domain == 'workshop' and canonical_kind == 'FIXED_TARGET'
                    else 'SEMANTIC_ONLY' if domain != 'workshop' and not role['required_properties'] else 'SEMANTIC_AND_GEOMETRIC'))
        trace['roles'].append({'raw_id': rid, 'canonical_role': name, 'rule': rule, 'status': 'CANONICAL_EXECUTABLE_SEMANTIC'})
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

    def add_relation(raw_subject: str, phrase: str, raw_target: str, *, grouped=False, expected=True) -> str | None:
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
                                                      semantic_categories=ontology.get_system_role_semantic_categories(domain, s),
                                                      verification_mode='GEOMETRIC_ONLY')
                        if o not in nodes and o in fixed_anchors:
                            nodes[o] = FunctionalRole(name=o, entity_kind='FIXED_TARGET', count=1, binding_policy='SHARED',
                                                      semantic_categories=ontology.get_system_role_semantic_categories(domain, o),
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
            interpret_task_effect_predicate,
        )
        raw_subject = rel['subject_role']
        raw_object = rel['object_role']
        raw_phrase = rel.get('relation', rel.get('predicate'))
        effect_predicate = rel.get('effect_predicate') or interpret_task_effect_predicate(raw_phrase)
        object_is_literal = rel.get('object_is_literal') is True
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
        add_relation(rel['subject_role'], rel.get('relation', rel.get('predicate')), rel['object_role'], expected=rel.get('expected', True))
    from .robot_capability_registry import interpret_operation
    raw_groups = doc.get('interaction_groups') or doc.get('operations') or []
    groups = []
    for group in raw_groups:
        tool_raw = group.get('tool_role') or group.get('source_role')
        target_raw = group.get('target_role')
        ctx_raw = group.get('context_role') or group.get('anchor_role')
        raw_op = group.get('function') or group.get('operation') or ''

        # Explicit final placement/restoration to a registered planner constant
        # is audited but is not a selectable G_F operation.  The domain planner
        # owns that context transition.  Other operations with unresolved
        # context endpoints still fail closed below.
        context_endpoint = planner_context_id_map.get(tool_raw) or planner_context_id_map.get(target_raw)
        non_context_endpoint = target_raw if tool_raw in planner_context_id_map else tool_raw
        if (
            context_endpoint
            and non_context_endpoint in id_map
            and re.search(r'\b(place|return|leave|restore|set down|put)\b', raw_op.lower())
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
            trace['disabled_groups'].append({'raw_group': group, 'status': 'UNINSTANTIABLE_MISSING_ROLE'})
            continue

        tool_role_id = id_map[tool_raw]
        target_role_id = id_map[target_raw]
        ctx_role_id = id_map.get(ctx_raw) if ctx_raw else None

        if domain == 'living_room':
            if nodes[tool_role_id].entity_kind == 'OBJECT' and nodes[target_role_id].entity_kind == 'REGION':
                tool_role_id, target_role_id = target_role_id, tool_role_id
                tool_raw, target_raw = target_raw, tool_raw

        is_v2 = is_v2_document(raw)
        if not ctx_role_id:
            if domain == 'workshop' and 'repair_target' in nodes:
                ctx_role_id = 'repair_target'

        op_interp = interpret_operation(
            domain=domain,
            raw_phrase=raw_op,
            source_role=tool_role_id,
            target_role=target_role_id,
            anchor_role=ctx_role_id,
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
                                              semantic_categories=ontology.get_system_role_semantic_categories(domain, s_r),
                                              verification_mode='GEOMETRIC_ONLY')
                if o_r not in nodes and o_r in fixed_anchors:
                    nodes[o_r] = FunctionalRole(name=o_r, entity_kind='FIXED_TARGET', count=1, binding_policy='SHARED',
                                              semantic_categories=ontology.get_system_role_semantic_categories(domain, o_r),
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
            trace['groups'].append({
                'raw_group': group,
                'status': 'STATIC_ALREADY_SATISFIED',
                'representation': 'SINGLETON_RELATIONS',
                'capability_id': op_interp.capability.capability_id if op_interp.capability else None,
                'planner_operation': op_interp.planner_operation,
                'physical_preconditions': [list(t) for t in op_interp.physical_preconditions],
                'preconditions_provenance': preconditions_provenance,
            })
            continue

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
        ))
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
    contract_complete, contract_missing_reasons = check_required_contract_complete(
        domain, nodes, relations, groups, trace, sanitized, unresolved
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
            'precondition_provenance': all_precond_prov,
            'task_causal_relations': [r.to_dict() for r in task_causal_relations],
            'task_effect_relations': [r.to_dict() for r in task_effect_relations],
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
