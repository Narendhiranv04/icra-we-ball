"""Stage C: expose unresolved requirements independently of graph acceptance.

Pre-grounding analysis does not claim execution readiness. Grounding-sensitive
statuses are recomputed against verified bindings, never against reference GT.
"""
from __future__ import annotations

from typing import Any, Mapping

from .models import FunctionalRequirementGraph
from .robot_capability_registry import get_robot_capabilities


def analyze_executability(graph: FunctionalRequirementGraph,
                         assignment: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    result = [dict(item) for item in graph.metadata.get("unverified_required_properties", [])]
    trace = graph.metadata.get('canonicalization_trace', {})
    for item in trace.get('context_only_roles', []):
        result.append({'id': item['raw_role']['id'], 'status': 'CONTEXT_ONLY'})
    for item in trace.get('unresolved_roles', []):
        result.append({'id': item['raw_role']['id'], 'status': 'UNINSTANTIABLE_MISSING_ROLE'})
    for item in graph.metadata.get('soft_semantic_evidence', []):
        r_id = item.get('raw_role_id') or item.get('raw_subject') or item.get('id', 'unknown')
        result.append({'id': r_id, 'requirement': item.get('raw_phrase', ''), 'status': 'SOFT_SEMANTIC_ONLY'})
    for item in trace.get('disabled_groups', []):
        result.append({'id': item['raw_group']['id'], 'status': item['status'], 'raw_group': item['raw_group']})
    for group in graph.operation_groups:
        roles = [group.tool_role, group.target_role] + ([group.context_role] if group.context_role else [])
        missing = [r for r in roles if r not in graph.nodes]
        capabilities = {
            capability.capability_id: capability
            for capability in get_robot_capabilities(graph.domain)
        }
        relationless_capability = bool(
            group.capability_id
            and group.capability_id in capabilities
            and not capabilities[group.capability_id].required_relation_templates
        )
        if missing:
            status = 'UNINSTANTIABLE_MISSING_ROLE'
        elif not group.required_relations and not relationless_capability:
            status = 'UNINSTANTIABLE_MISSING_RELATION'
        elif assignment is None:
            status = 'UNINSTANTIABLE_UNGROUNDED_ROLE'
        else:
            def grounded(role: str) -> bool:
                return bool(assignment.get(role)) or any(k.startswith(role + '_') and v for k, v in assignment.items())
            status = 'EXECUTABLE' if all(grounded(r) for r in roles) else 'UNINSTANTIABLE_UNGROUNDED_ROLE'
        result.append({'id': group.id, 'status': status, 'roles': roles, 'missing_roles': missing})
    return result
