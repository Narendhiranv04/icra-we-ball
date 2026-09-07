"""Stage B: typed candidate compilation, independent of benchmark references.

Domain mappers supply vocabulary; causal position resolves participants. Unknown
content remains auditable. This module never reads task/reference fixtures.
"""
from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from .models import FunctionalRequirementGraph, FunctionalRole, FunctionalRelation, OperationGroup
from .structural_sanitizer import sanitize_functional_graph
from . import role_semantic_ontology as ontology
from .predicate_registry import validate_predicate_signature
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
    if causal_position(a, document) != causal_position(b, document):
        return False
    pair = {a['id'], b['id']}
    if any({r.get('subject_role'), r.get('object_role')} == pair for r in document.get('functional_relations', [])):
        return False
    if any({g.get('tool_role'), g.get('target_role')} == pair for g in document.get('interaction_groups', [])):
        return False
    # Equal canonical names alone do not establish equal causal function.
    normalize = lambda x: re.sub(r'\W+', ' ', x.lower()).strip()
    return normalize(a['function']) == normalize(b['function']) and normalize(a.get('description', '')) == normalize(b.get('description', ''))


def _map_role(domain: str, role: dict, doc: dict) -> tuple[str | None, str]:
    position = causal_position(role, doc)
    if domain == 'kitchen':
        from mujoco_scenes.kitchen_vlm_functional_graph import map_kitchen_role_function
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
    if role['entity_kind'] in ('FIXED_TARGET', 'REGION'):
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


def compile_candidate_graph(domain: str, task: str, raw: dict) -> FunctionalRequirementGraph:
    sanitized = sanitize_functional_graph(raw)
    if not sanitized.succeeded:
        raise VLMSpecificationError('No meaningful functional roles recovered', category='SANITIZER_UNRECOVERABLE')
    doc = sanitized.document
    trace: dict[str, Any] = dict(roles=[], properties=[], relations=[], groups=[], context_only_roles=[],
                                unresolved_roles=[], merged_roles=[], disambiguated_roles=[], disabled_groups=[])
    nodes = {}
    id_map = {}
    owners = {}
    soft = []
    unresolved = []
    unverified_required = []
    from .system_context_registry import get_domain_selectable_roles, get_domain_system_fixed_anchors
    allowed = set(get_domain_selectable_roles(domain)) | set(get_domain_system_fixed_anchors(domain))
    for role in doc['functional_roles']:
        rid = role['id']
        try:
            name, rule = _map_role(domain, role, doc)
        except VLMSpecificationError as exc:
            name, rule = None, str(exc)
        if name not in allowed:
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
                    rel = FunctionalRelation(s, p, o, expected=expected)
                    if not grouped and rel not in relations:
                        relations.append(rel)
                    if first_pred is None:
                        first_pred = p
                evidence.update(
                    status='CANONICAL_EXECUTABLE_SEMANTIC',
                    interp_status=interp.status,
                    canonical=[[ip.subject_role, ip.predicate_name, ip.object_role] for ip in interp.interpreted_predicates],
                    direction_normalized=interp.direction_normalized,
                )
                trace['relations'].append(evidence)
                return first_pred

            # Secondary fallback via domain-specific canonicalizer
            s, p, o = _relation(domain, canon_s, phrase, canon_o)
            if not p:
                raise ValueError(interp.reason or 'No executable relation checker')
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
        except (VLMSpecificationError, ValueError) as exc:
            evidence.update(status='UNRESOLVED_SEMANTIC', reason=str(exc))
            unresolved.append(evidence)
            trace['relations'].append(evidence)
            return None
        relation = FunctionalRelation(s, p, o, expected=expected)
        if not grouped and relation not in relations:
            relations.append(relation)
        evidence.update(status='CANONICAL_EXECUTABLE_SEMANTIC', canonical=[s, p, o])
        trace['relations'].append(evidence)
        return p

    for rel in doc['functional_relations']:
        add_relation(rel['subject_role'], rel.get('relation', rel.get('predicate')), rel['object_role'], expected=rel.get('expected', True))
    groups = []
    for group in doc['interaction_groups']:
        required = [
            mapped for phrase in group.get('required_relations', [])
            if (mapped := add_relation(
                group['tool_role'], phrase, group['target_role'], grouped=True
            )) is not None
        ]
        context = []
        if group.get('context_role'):
            for phrase in group.get('context_relations', []):
                mapped = add_relation(group['tool_role'], phrase, group['context_role'], grouped=True)
                if mapped is None and group.get('target_role'):
                    mapped = add_relation(group['target_role'], phrase, group['context_role'], grouped=True)
                if mapped is not None:
                    context.append(mapped)
        if not required or any(group[k] not in id_map for k in ('tool_role', 'target_role')):
            trace['disabled_groups'].append({'raw_group': group, 'status': 'UNINSTANTIABLE_MISSING_RELATION'})
            continue
        tool_role_id = id_map[group['tool_role']]
        target_role_id = id_map[group['target_role']]
        if domain == 'kitchen':
            from mujoco_scenes.kitchen_vlm_functional_graph import map_kitchen_interaction_group_function
            function = map_kitchen_interaction_group_function(group.get('function', ''))
            if not function:
                if (tool_role_id, target_role_id) == ('coffee_stirrer', 'coffee_container'):
                    function = 'coffee_stirring'
                elif (tool_role_id, target_role_id) == ('soup_eating_utensil', 'soup_container'):
                    function = 'soup_serving'
        elif domain == 'living_room':
            from mujoco_scenes.environment_vlm_requirements import map_living_room_operation_group_function
            function = map_living_room_operation_group_function(group.get('function', ''))
            if not function:
                pair = {tool_role_id, target_role_id}
                if pair == {'CUP_SAUCER_SET', 'PERSONAL_CUP_SAUCER_REGION'}:
                    function = 'personal_support_group'
                elif pair == {'REMOTE', 'SHARED_REMOTE_REGION'}:
                    function = 'shared_entertainment_group'
            if nodes[tool_role_id].entity_kind == 'OBJECT' and nodes[target_role_id].entity_kind == 'REGION':
                tool_role_id, target_role_id = target_role_id, tool_role_id
        else:
            function = 'DRIVE_FASTENER_INTO_TARGET' if (tool_role_id, target_role_id) == ('driver', 'fastener') else None
        count = group.get('required_target_count')
        if not function or type(count) is not int or count < 1 or count > nodes[target_role_id].maximum_count or group.get('usage_policy') not in {'DEDICATED_PER_TARGET', 'SEQUENTIAL_REUSE_ALLOWED'}:
            trace['disabled_groups'].append({'raw_group': group, 'status': 'UNSUPPORTED_OPERATOR'})
            continue
        # Singleton interaction requirements are equivalent to ordinary graph
        # edges. Multi-target bindings must retain their group quantification.
        if domain == 'workshop' and count == 1:
            for phrase in group.get('required_relations', []):
                add_relation(group['tool_role'], phrase, group['target_role'])
            for phrase in group.get('context_relations', []):
                add_relation(group['tool_role'], phrase, group['context_role'])
            ctx_id = group.get('context_role')
            if ctx_id and id_map.get(ctx_id) == 'repair_target':
                add_relation(group['target_role'], 'compatible with target', ctx_id)
            elif 'repair_target' in nodes and any(r.object_role == 'repair_target' for r in relations):
                target_ids = [k for k, v in id_map.items() if v == 'repair_target']
                if target_ids:
                    add_relation(group['target_role'], 'compatible with target', target_ids[0])
            trace['groups'].append({'raw_group': group, 'status': 'STATIC_ALREADY_SATISFIED',
                                    'representation': 'SINGLETON_RELATIONS'})
            continue
        runtime_function = {'coffee_stirring': 'STIR_COFFEE', 'soup_serving': 'PROVIDE_SOUP_EATING_UTENSIL',
                            'personal_support_group': 'SUPPORT_DRINKWARE',
                            'shared_entertainment_group': 'SUPPORT_ENTERTAINMENT_CONTROL'}.get(function, function)
        executable_context_role = id_map.get(group.get('context_role')) if context else None
        usage_policy = group['usage_policy']
        if domain == 'living_room':
            if executable_context_role == 'SEATING_POSITION' and 'ACCESSIBLE_FROM_BOTH_SEATS' in context:
                executable_context_role = 'SEATING_PAIR'
            if function == 'personal_support_group':
                usage_policy = 'DEDICATED_PER_TARGET'
                if not executable_context_role and 'SEATING_POSITION' in nodes:
                    executable_context_role = 'SEATING_POSITION'
                    if 'NEAR_SEAT' not in context:
                        context.append('NEAR_SEAT')
        if (
            usage_policy == 'SEQUENTIAL_REUSE_ALLOWED'
            and tool_role_id in nodes
            and nodes[tool_role_id].entity_kind == 'OBJECT'
            and not nodes[tool_role_id].shared
        ):
            t_node = nodes[tool_role_id]
            nodes[tool_role_id] = replace(
                t_node,
                binding_policy='REUSABLE',
                min_count=t_node.min_count or 1,
                preference=t_node.preference or 'minimize_distinct',
            )
        groups.append(OperationGroup(id=function if not any(g.id == function for g in groups) else group['id'], function=runtime_function, tool_role=tool_role_id,
            target_role=target_role_id, required_target_count=count, usage_policy=usage_policy,
            required_relations=tuple(dict.fromkeys(required)), context_role=executable_context_role,
            context_relations=tuple(dict.fromkeys(context)),
            distinct_within_group=group.get('distinct_within_group', usage_policy == 'DEDICATED_PER_TARGET'),
            same_tool_must_cover_all_targets=group.get('same_tool_must_cover_all_targets', False),
            selection_preference=group.get('selection_preference', ('minimize_distinct_tools' if usage_policy == 'SEQUENTIAL_REUSE_ALLOWED' else 'deterministic_rank') if domain == 'kitchen' else None)))
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
    graph = FunctionalRequirementGraph(domain=domain, task_instruction=task, nodes=nodes, relations=tuple(relations),
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
