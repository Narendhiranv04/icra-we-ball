"""Offline Reference G_F Evaluator for Pass 3.6B VLM specification evaluation.

Compares a candidate FunctionalRequirementGraph against the ground-truth / reference
FunctionalRequirementGraph produced by GTSpecProvider().

IMPORTANT SCIENTIFIC BOUNDARY:
- This module is for OFFLINE evaluation and diagnostic metrics ONLY.
- It MUST NOT participate in runtime specification acceptance, search, observation,
  grounding, planning, or execution.
- It NEVER mutates candidate G_F.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .gt_spec_provider import GTSpecProvider
from .models import FunctionalRequirementGraph


@dataclass(frozen=True)
class GFReferenceEvaluationResult:
    """Comprehensive offline structural comparison of candidate G_F against reference G_F."""

    domain: str
    task_instruction: str

    # Roles comparison
    reference_roles: tuple[str, ...]
    candidate_roles: tuple[str, ...]
    matched_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    extra_roles: tuple[str, ...]
    role_attribute_mismatches: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Relations comparison
    reference_relations: tuple[tuple[str, str, str], ...] = ()
    candidate_relations: tuple[tuple[str, str, str], ...] = ()
    missing_relations: tuple[tuple[str, str, str], ...] = ()
    extra_relations: tuple[tuple[str, str, str], ...] = ()

    # Operation groups comparison
    reference_operation_groups: tuple[dict[str, Any], ...] = ()
    candidate_operation_groups: tuple[dict[str, Any], ...] = ()
    missing_operation_groups: tuple[str, ...] = ()
    extra_operation_groups: tuple[str, ...] = ()
    operation_group_mismatches: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metrics
    structurally_complete: bool = False
    role_recall: float = 0.0
    relation_recall: float = 0.0
    operation_group_recall: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task_instruction": self.task_instruction,
            "roles": {
                "reference": list(self.reference_roles),
                "candidate": list(self.candidate_roles),
                "matched": list(self.matched_roles),
                "missing": list(self.missing_roles),
                "extra": list(self.extra_roles),
                "attribute_mismatches": self.role_attribute_mismatches,
            },
            "relations": {
                "reference": [list(r) for r in self.reference_relations],
                "candidate": [list(r) for r in self.candidate_relations],
                "missing": [list(r) for r in self.missing_relations],
                "extra": [list(r) for r in self.extra_relations],
            },
            "operation_groups": {
                "reference": list(self.reference_operation_groups),
                "candidate": list(self.candidate_operation_groups),
                "missing": list(self.missing_operation_groups),
                "extra": list(self.extra_operation_groups),
                "mismatches": self.operation_group_mismatches,
            },
            "metrics": {
                "structurally_complete": self.structurally_complete,
                "role_recall": self.role_recall,
                "relation_recall": self.relation_recall,
                "operation_group_recall": self.operation_group_recall,
            },
        }


def evaluate_gf_against_reference(
    candidate_graph: FunctionalRequirementGraph,
    reference_graph: FunctionalRequirementGraph | None = None,
    *,
    domain: str | None = None,
    task_instruction: str | None = None,
) -> GFReferenceEvaluationResult:
    """Deterministically compare candidate G_F against reference G_F without mutating candidate."""
    target_domain = domain or candidate_graph.domain
    target_instruction = task_instruction or candidate_graph.task_instruction

    if reference_graph is None:
        reference_graph = GTSpecProvider().provide(
            domain=target_domain,
            task_instruction=target_instruction,
        )

    ref_roles = set(reference_graph.nodes.keys())
    cand_roles = set(candidate_graph.nodes.keys())
    matched_roles = ref_roles & cand_roles
    missing_roles = ref_roles - cand_roles
    extra_roles = cand_roles - ref_roles

    role_attribute_mismatches: dict[str, dict[str, Any]] = {}
    for r_name in matched_roles:
        ref_node = reference_graph.nodes[r_name]
        cand_node = candidate_graph.nodes[r_name]
        diffs = {}
        if ref_node.entity_kind != cand_node.entity_kind:
            diffs["entity_kind"] = {"reference": ref_node.entity_kind, "candidate": cand_node.entity_kind}
        if ref_node.minimum_count != cand_node.minimum_count:
            diffs["minimum_count"] = {"reference": ref_node.minimum_count, "candidate": cand_node.minimum_count}
        if ref_node.maximum_count != cand_node.maximum_count:
            diffs["maximum_count"] = {"reference": ref_node.maximum_count, "candidate": cand_node.maximum_count}
        if ref_node.binding_policy != cand_node.binding_policy:
            diffs["binding_policy"] = {"reference": ref_node.binding_policy, "candidate": cand_node.binding_policy}
        ref_unary = set(ref_node.unary_predicates)
        cand_unary = set(cand_node.unary_predicates)
        if ref_unary != cand_unary:
            diffs["unary_predicates"] = {"reference": sorted(ref_unary), "candidate": sorted(cand_unary)}
        if diffs:
            role_attribute_mismatches[r_name] = diffs

    # Relations comparison (subject_role, predicate, object_role)
    ref_rel_triples = set((r.subject_role, r.predicate, r.object_role) for r in reference_graph.relations)
    cand_rel_triples = set((r.subject_role, r.predicate, r.object_role) for r in candidate_graph.relations)
    matched_rels = ref_rel_triples & cand_rel_triples
    missing_rels = ref_rel_triples - cand_rel_triples
    extra_rels = cand_rel_triples - ref_rel_triples

    # Operation groups comparison
    ref_ops_by_id = {g.id: g for g in reference_graph.operation_groups}
    cand_ops_by_id = {g.id: g for g in candidate_graph.operation_groups}

    matched_ops = set()
    missing_ops = set()
    op_mismatches = {}

    for r_id, r_op in ref_ops_by_id.items():
        cand_op = cand_ops_by_id.get(r_id)
        if cand_op is None:
            for c_op in cand_ops_by_id.values():
                if c_op.tool_role == r_op.tool_role and c_op.target_role == r_op.target_role:
                    cand_op = c_op
                    break
        if cand_op is None:
            missing_ops.add(r_id)
        else:
            matched_ops.add(r_id)
            diffs = {}
            if r_op.tool_role != cand_op.tool_role:
                diffs["tool_role"] = {"reference": r_op.tool_role, "candidate": cand_op.tool_role}
            if r_op.target_role != cand_op.target_role:
                diffs["target_role"] = {"reference": r_op.target_role, "candidate": cand_op.target_role}
            if r_op.required_target_count != cand_op.required_target_count:
                diffs["required_target_count"] = {"reference": r_op.required_target_count, "candidate": cand_op.required_target_count}
            if r_op.usage_policy != cand_op.usage_policy:
                diffs["usage_policy"] = {"reference": r_op.usage_policy, "candidate": cand_op.usage_policy}
            if set(r_op.required_relations) != set(cand_op.required_relations):
                diffs["required_relations"] = {"reference": sorted(r_op.required_relations), "candidate": sorted(cand_op.required_relations)}
            if r_op.context_role != cand_op.context_role:
                diffs["context_role"] = {"reference": r_op.context_role, "candidate": cand_op.context_role}
            if set(r_op.context_relations) != set(cand_op.context_relations):
                diffs["context_relations"] = {"reference": sorted(r_op.context_relations), "candidate": sorted(cand_op.context_relations)}
            if diffs:
                op_mismatches[r_id] = diffs

    extra_ops = set(cand_ops_by_id.keys()) - {cand_ops_by_id[g].id for g in matched_ops if g in cand_ops_by_id}

    role_recall = len(matched_roles) / len(ref_roles) if ref_roles else 1.0
    rel_recall = len(matched_rels) / len(ref_rel_triples) if ref_rel_triples else 1.0
    op_recall = len(matched_ops) / len(ref_ops_by_id) if ref_ops_by_id else 1.0

    structurally_complete = (
        len(missing_roles) == 0
        and len(role_attribute_mismatches) == 0
        and len(missing_rels) == 0
        and len(missing_ops) == 0
        and len(op_mismatches) == 0
    )

    return GFReferenceEvaluationResult(
        domain=target_domain,
        task_instruction=target_instruction,
        reference_roles=tuple(sorted(ref_roles)),
        candidate_roles=tuple(sorted(cand_roles)),
        matched_roles=tuple(sorted(matched_roles)),
        missing_roles=tuple(sorted(missing_roles)),
        extra_roles=tuple(sorted(extra_roles)),
        role_attribute_mismatches=role_attribute_mismatches,
        reference_relations=tuple(sorted(ref_rel_triples)),
        candidate_relations=tuple(sorted(cand_rel_triples)),
        missing_relations=tuple(sorted(missing_rels)),
        extra_relations=tuple(sorted(extra_rels)),
        reference_operation_groups=tuple(g.to_dict() for g in reference_graph.operation_groups),
        candidate_operation_groups=tuple(g.to_dict() for g in candidate_graph.operation_groups),
        missing_operation_groups=tuple(sorted(missing_ops)),
        extra_operation_groups=tuple(sorted(extra_ops)),
        operation_group_mismatches=op_mismatches,
        structurally_complete=structurally_complete,
        role_recall=role_recall,
        relation_recall=rel_recall,
        operation_group_recall=op_recall,
    )
