"""Offline raw-output scoring, callable before sanitizer/compiler acceptance.

This is deterministic ontology-based semantic matching, not human annotation.
Invalid endpoints and unrecognized content remain unmatched predictions. No
result of this module is consumed by the online pipeline.
"""
from collections import Counter
from typing import Any


def prf(predicted, reference):
    predicted, reference = Counter(predicted), Counter(reference)
    matches = sum((predicted & reference).values())
    precision = matches / sum(predicted.values()) if predicted else 0.0
    recall = matches / sum(reference.values()) if reference else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else (0.0 if recall is not None else None)
    return dict(precision=precision, recall=recall, f1=f1)


def evaluate_raw_semantics(domain: str, task: str, raw: Any, reference=None) -> dict:
    from .gt_spec_provider import GTSpecProvider
    from .semantic_compiler import _map_role, _relation
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
                name, _ = _map_role(domain, role, raw)
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
                    triple = _relation(domain, s, rel.get('relation', rel.get('predicate', '')), o)
                except (ValueError, KeyError, TypeError, AttributeError):
                    pass
        predicted_relations.append(triple or ('__unmatched',str(index),''))
    predicted_groups = []
    groups = raw.get('interaction_groups', [])
    groups = groups if isinstance(groups, list) else []
    for index, group in enumerate(groups):
        if isinstance(group, dict):
            s, o = mapped.get(group.get('tool_role')), mapped.get(group.get('target_role'))
            predicted_groups.append((s, o, group.get('required_target_count'), group.get('usage_policy')))
        else:
            predicted_groups.append(('__invalid', index))
    return {'role': prf(predicted_roles, list(reference.nodes)),
            'relation': prf(predicted_relations, [(r.subject_role,r.predicate,r.object_role) for r in reference.relations]),
            'group': prf(predicted_groups, [(g.tool_role,g.target_role,g.required_target_count,g.usage_policy) for g in reference.operation_groups]),
            'matching_method': 'deterministic_ontology_semantic_matching',
            'role_count': len(roles), 'raw_id_to_semantic_role': mapped}
