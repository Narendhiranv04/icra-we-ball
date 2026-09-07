"""Offline raw-output scoring, callable before sanitizer/compiler acceptance.

This is deterministic ontology-based semantic matching, not human annotation.
Invalid endpoints and unrecognized content remain unmatched predictions. No
result of this module is consumed by the online pipeline.
"""
from collections import Counter
import re
from typing import Any


def prf(predicted, reference):
    predicted, reference = Counter(predicted), Counter(reference)
    matches = sum((predicted & reference).values())
    precision = matches / sum(predicted.values()) if predicted else 0.0
    recall = matches / sum(reference.values()) if reference else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else (0.0 if recall is not None else None)
    return dict(precision=precision, recall=recall, f1=f1)


def _causal_position(role: dict, document: dict) -> set[str]:
    positions = set()
    rid = role.get('id')
    for group in document.get('interaction_groups', []):
        if group.get('tool_role') == rid:
            positions.add('instrument')
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


def _frozen_map_role(domain: str, role: dict, doc: dict) -> tuple[str | None, str]:
    position = _causal_position(role, doc)
    if domain == 'kitchen':
        from mujoco_scenes.kitchen_vlm_functional_graph import map_kitchen_role_function
        enriched = dict(role)
        if 'source' in position and 'destination' not in position:
            enriched['function'] = 'source provider ' + role['function']
            enriched['description'] = role.get('description', '')
            text = (enriched['function'] + ' ' + enriched['description']).lower()
            for material in ('coffee', 'water'):
                if re.search(r'\b' + material + r'\b', text):
                    return material + '_source', 'CAUSAL_SOURCE_PROVIDER'
        return map_kitchen_role_function(enriched), 'DOMAIN_SEMANTICS'
    if domain == 'living_room':
        from mujoco_scenes.environment_vlm_requirements import (
            map_living_room_role_function,
            map_living_room_object_payload_role,
            map_living_room_fixed_target_role,
        )
        seating = map_living_room_fixed_target_role(role)
        if seating:
            return seating, 'FIXED_TARGET_SEMANTICS'
        mapper = {
            'REGION': map_living_room_role_function,
            'OBJECT': map_living_room_object_payload_role,
            'FIXED_TARGET': map_living_room_fixed_target_role,
        }.get(role.get('entity_kind', 'OBJECT'), map_living_room_role_function)
        mapped = mapper(role)
        if mapped is None and role.get('entity_kind') == 'REGION':
            function = (role.get('function', '') + ' ' + role.get('description', '')).lower()
            edges = [r for r in doc.get('functional_relations', []) if role['id'] in (r.get('subject_role'), r.get('object_role'))]
            spatial = ' '.join(str(r.get('relation', r.get('predicate', ''))) for r in edges).lower()
            bp = role.get('binding_policy')
            count = role.get('required_count', 1)
            if re.search(r'\b(television|screen|monitor|display|wall)\b', function):
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
    from mujoco_scenes.workshop_phase1.requirements import (
        map_workshop_role_function,
        map_workshop_fixed_target_role,
        map_workshop_context_region_role,
    )
    if role.get('entity_kind') in ('FIXED_TARGET', 'REGION'):
        target = map_workshop_fixed_target_role(role)
        if target:
            return target, 'FIXED_TARGET_SEMANTICS'
        if role.get('entity_kind') == 'REGION':
            return map_workshop_context_region_role(role), 'SUPPORT_CONTEXT'
    mapped = map_workshop_role_function(role)
    if mapped is None and 'instrument' in position and 'group_tool' in position and 'component' not in position:
        from mujoco_scenes.workshop_phase1.requirements import ManualWorkshopFMContract
        from mujoco_scenes.functional_tamp_pipeline import role_semantic_ontology as ontology
        aliases = ManualWorkshopFMContract().get_alias_to_canonical_map()
        categories = {aliases.get(c.lower(), c.lower().replace(' ', '_')) for c in role.get('candidate_categories', [])}
        if categories.intersection(ontology.get_system_role_semantic_categories('workshop', 'driver')):
            mapped = 'CAN_DRIVE_SCREW'
    return {'CAN_DRIVE_SCREW': 'driver', 'CAN_FASTEN': 'fastener'}.get(mapped, mapped), 'CAUSAL_OPERATION_PARTICIPANT'


def _frozen_relation(domain: str, subject: str, phrase: str, target: str) -> tuple[str, str, str]:
    if domain == 'kitchen':
        from mujoco_scenes.kitchen_vlm_functional_graph import map_binary_relation
        return subject, map_binary_relation(phrase), target
    if domain == 'living_room':
        from mujoco_scenes.environment_vlm_requirements import canonicalize_living_room_relation
        return canonicalize_living_room_relation(phrase, subject, target)[:3]
    from mujoco_scenes.workshop_phase1.requirements import canonicalize_workshop_relation
    return canonicalize_workshop_relation(subject, subject, phrase, target, target)[:3]


def evaluate_raw_semantics(domain: str, task: str, raw: Any, reference=None) -> dict:
    from .gt_spec_provider import GTSpecProvider
    if reference is None:
        reference = GTSpecProvider().provide(domain, task)
    raw = raw if isinstance(raw, dict) else {}
    roles = raw.get('functional_roles', [])
    roles = roles if isinstance(roles, list) else []
    mapped = {}
    predicted_roles = []
    for index, role in enumerate(roles):
        name = None
        if isinstance(role, dict):
            try:
                name, _ = _frozen_map_role(domain, role, raw)
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
            mapped[role.get('id', f'invalid_{index}')] = name
        predicted_roles.append(name or f'__unmatched_role_{index}')
    predicted_relations = []
    relations = raw.get('functional_relations', [])
    relations = relations if isinstance(relations, list) else []
    for index, rel in enumerate(relations):
        triple = None
        if isinstance(rel, dict):
            s, o = mapped.get(rel.get('subject_role')), mapped.get(rel.get('object_role'))
            if s and o:
                try:
                    triple = _frozen_relation(domain, s, rel.get('relation', rel.get('predicate', '')), o)
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
        predicted_relations.append(triple or ('__unmatched', str(index), ''))
    predicted_groups = []
    groups = raw.get('interaction_groups', [])
    groups = groups if isinstance(groups, list) else []
    for index, group in enumerate(groups):
        if isinstance(group, dict):
            s, o = mapped.get(group.get('tool_role')), mapped.get(group.get('target_role'))
            predicted_groups.append((s, o, group.get('required_target_count'), group.get('usage_policy')))
        else:
            predicted_groups.append(('__invalid', index))
    return {
        'role': prf(predicted_roles, list(reference.nodes)),
        'relation': prf(predicted_relations, [(r.subject_role, r.predicate, r.object_role) for r in reference.relations]),
        'group': prf(predicted_groups, [(g.tool_role, g.target_role, g.required_target_count, g.usage_policy) for g in reference.operation_groups]),
        'matching_method': 'frozen_offline_semantic_matching',
        'role_count': len(roles),
        'raw_id_to_semantic_role': mapped,
    }
