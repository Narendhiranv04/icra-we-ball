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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .gt_spec_provider import GTSpecProvider
from .models import FunctionalRequirementGraph

# Evaluation-only legacy functional-role markers that are encoded as canonical role identities
# in the final VLM interface (e.g. 'driver' role identity implies capability to drive screws).
LEGACY_ROLE_FUNCTION_MARKERS: dict[tuple[str, str], set[str]] = {
    ("workshop", "driver"): {"CAN_DRIVE_SCREW"},
    ("workshop", "fastener"): {"CAN_FASTEN"},
}


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
    role_cardinality_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    role_normalization_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    category_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Relations comparison (subject_role, predicate, object_role, expected)
    reference_relations: tuple[tuple[str, str, str, bool], ...] = ()
    candidate_relations: tuple[tuple[str, str, str, bool], ...] = ()
    missing_relations: tuple[tuple[str, str, str, bool], ...] = ()
    extra_relations: tuple[tuple[str, str, str, bool], ...] = ()

    # Operation groups comparison
    reference_operation_groups: tuple[dict[str, Any], ...] = ()
    candidate_operation_groups: tuple[dict[str, Any], ...] = ()
    matched_operation_groups: tuple[tuple[str, str], ...] = ()  # (ref_id, cand_id)
    missing_operation_groups: tuple[str, ...] = ()
    extra_operation_groups: tuple[str, ...] = ()
    ambiguous_operation_groups: dict[str, list[str]] = field(default_factory=dict)
    operation_group_mismatches: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Graph-level semantics
    cross_group_reuse_mismatch: bool = False
    graph_attribute_mismatches: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Primary Evaluation Metrics
    reference_complete: bool = False
    exact_structural_match: bool = False
    structurally_complete: bool = False  # Backward-compatible alias for reference_complete

    # Detailed Recall & Precision
    role_identity_recall: float = 0.0
    role_identity_precision: float = 0.0
    role_exact_recall: float = 0.0
    role_exact_precision: float = 0.0

    relation_recall: float = 0.0
    relation_precision: float = 0.0

    operation_group_identity_recall: float = 0.0
    operation_group_identity_precision: float = 0.0
    operation_group_exact_recall: float = 0.0
    operation_group_exact_precision: float = 0.0

    # Backward compatibility metric aliases
    role_recall: float = 0.0
    role_precision: float = 0.0
    operation_group_recall: float = 0.0
    operation_group_precision: float = 0.0

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
                "cardinality_diagnostics": self.role_cardinality_diagnostics,
                "normalization_diagnostics": self.role_normalization_diagnostics,
                "category_diagnostics": self.category_diagnostics,
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
                "matched_pairs": [list(p) for p in self.matched_operation_groups],
                "missing": list(self.missing_operation_groups),
                "extra": list(self.extra_operation_groups),
                "ambiguous": self.ambiguous_operation_groups,
                "mismatches": self.operation_group_mismatches,
            },
            "graph_level": {
                "cross_group_reuse_mismatch": self.cross_group_reuse_mismatch,
                "attribute_mismatches": self.graph_attribute_mismatches,
            },
            "metrics": {
                "reference_complete": self.reference_complete,
                "exact_structural_match": self.exact_structural_match,
                "structurally_complete": self.structurally_complete,
                "role_identity_recall": self.role_identity_recall,
                "role_identity_precision": self.role_identity_precision,
                "role_exact_recall": self.role_exact_recall,
                "role_exact_precision": self.role_exact_precision,
                "relation_recall": self.relation_recall,
                "relation_precision": self.relation_precision,
                "operation_group_identity_recall": self.operation_group_identity_recall,
                "operation_group_identity_precision": self.operation_group_identity_precision,
                "operation_group_exact_recall": self.operation_group_exact_recall,
                "operation_group_exact_precision": self.operation_group_exact_precision,
                # Backward-compatible metric keys
                "role_recall": self.role_recall,
                "role_precision": self.role_precision,
                "operation_group_recall": self.operation_group_recall,
                "operation_group_precision": self.operation_group_precision,
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
    # Pre-evaluation state snapshots to verify zero mutation invariance
    cand_nodes_snapshot = deepcopy(candidate_graph.nodes)
    cand_rels_snapshot = deepcopy(candidate_graph.relations)
    cand_ops_snapshot = deepcopy(candidate_graph.operation_groups)

    target_domain = domain or candidate_graph.domain
    target_instruction = task_instruction or candidate_graph.task_instruction

    if reference_graph is None:
        reference_graph = GTSpecProvider().provide(
            domain=target_domain,
            task_instruction=target_instruction,
        )

    ref_nodes_snapshot = deepcopy(reference_graph.nodes)
    ref_rels_snapshot = deepcopy(reference_graph.relations)
    ref_ops_snapshot = deepcopy(reference_graph.operation_groups)

    ref_roles = set(reference_graph.nodes.keys())
    cand_roles = set(candidate_graph.nodes.keys())
    matched_roles = ref_roles & cand_roles
    missing_roles = ref_roles - cand_roles
    extra_roles = cand_roles - ref_roles

    role_attribute_mismatches: dict[str, dict[str, Any]] = {}
    role_cardinality_diagnostics: dict[str, dict[str, Any]] = {}
    role_normalization_diagnostics: dict[str, dict[str, Any]] = {}
    category_diagnostics: dict[str, dict[str, Any]] = {}

    for r_name in matched_roles:
        ref_node = reference_graph.nodes[r_name]
        cand_node = candidate_graph.nodes[r_name]
        diffs: dict[str, Any] = {}

        # 1. Entity Kind
        if ref_node.entity_kind != cand_node.entity_kind:
            diffs["entity_kind"] = {"reference": ref_node.entity_kind, "candidate": cand_node.entity_kind}

        # 2. Cardinality and Binding Policy Compatibility
        ref_min = ref_node.minimum_count
        ref_max = ref_node.maximum_count
        cand_min = cand_node.minimum_count
        cand_max = cand_node.maximum_count
        ref_binding = ref_node.binding_policy
        cand_binding = cand_node.binding_policy

        binding_compatible = (ref_binding == cand_binding)
        if not binding_compatible:
            is_card_compat = False
            is_card_exact = False
            reason = f"Binding policy mismatch: reference requires {ref_binding}, candidate provides {cand_binding}"
        else:
            if ref_binding == "REUSABLE":
                # Candidate interval [cand_min, cand_max] is contained in allowed reference range [ref_min, ref_max]
                is_card_compat = (ref_min <= cand_min) and (cand_max <= ref_max)
                is_card_exact = (cand_min == ref_min) and (cand_max == ref_max)
                if is_card_compat:
                    if is_card_exact:
                        reason = f"Exact reusable count [{cand_min}, {cand_max}] matches reference interval [{ref_min}, {ref_max}]"
                    else:
                        reason = f"Candidate reusable count [{cand_min}, {cand_max}] is semantically compatible with reference allowed interval [{ref_min}, {ref_max}]"
                else:
                    reason = f"Candidate reusable count [{cand_min}, {cand_max}] is outside reference allowed interval [{ref_min}, {ref_max}]"
            else:
                # DISTINCT / SHARED: exact capacity or within intentional reference interval
                is_card_compat = (ref_min <= cand_min) and (cand_max <= ref_max)
                is_card_exact = (cand_min == ref_min) and (cand_max == ref_max)
                if is_card_compat:
                    if is_card_exact:
                        reason = f"Exact count [{cand_min}, {cand_max}] matches reference [{ref_min}, {ref_max}]"
                    else:
                        reason = f"Candidate count [{cand_min}, {cand_max}] is compatible with reference interval [{ref_min}, {ref_max}]"
                else:
                    reason = f"Candidate count [{cand_min}, {cand_max}] does not match reference requirement [{ref_min}, {ref_max}]"

        role_cardinality_diagnostics[r_name] = {
            "reference_binding": ref_binding,
            "candidate_binding": cand_binding,
            "reference_range": [ref_min, ref_max],
            "candidate_range": [cand_min, cand_max],
            "cardinality_compatible": is_card_compat,
            "cardinality_exact": is_card_exact,
            "reason": reason,
        }

        if not is_card_compat:
            diffs["cardinality"] = role_cardinality_diagnostics[r_name]
        if not binding_compatible:
            diffs["binding_policy"] = {"reference": ref_binding, "candidate": cand_binding}

        # 3. Unary Predicates with Evaluation-Only Legacy Marker Normalization
        ref_unary = set(ref_node.unary_predicates)
        cand_unary = set(cand_node.unary_predicates)

        ignored_markers = LEGACY_ROLE_FUNCTION_MARKERS.get((target_domain, r_name), set())
        if ignored_markers & ref_unary:
            role_normalization_diagnostics[r_name] = {
                "ignored_legacy_role_function_markers": sorted(ignored_markers & ref_unary),
                "description": "Legacy GT functional-role marker, represented by canonical role identity in the VLM interface.",
            }

        ref_effective_unary = ref_unary - ignored_markers
        cand_effective_unary = cand_unary - ignored_markers

        if ref_effective_unary != cand_effective_unary:
            diffs["unary_predicates"] = {
                "reference": sorted(ref_effective_unary),
                "candidate": sorted(cand_effective_unary),
            }

        if diffs:
            role_attribute_mismatches[r_name] = diffs

        # 4. Open-Vocabulary Semantic Category Diagnostics (Informative only, not failing completeness)
        ref_cats = list(ref_node.semantic_categories)
        cand_cats = list(cand_node.semantic_categories)
        ref_norm = sorted(set(c.casefold().strip() for c in ref_cats if str(c).strip()))
        cand_norm = sorted(set(c.casefold().strip() for c in cand_cats if str(c).strip()))
        exact_overlap = sorted(set(ref_norm) & set(cand_norm))
        overlap_ratio = len(exact_overlap) / len(ref_norm) if ref_norm else 1.0
        category_diagnostics[r_name] = {
            "reference_categories": ref_cats,
            "candidate_categories": cand_cats,
            "normalized_reference_categories": ref_norm,
            "normalized_candidate_categories": cand_norm,
            "exact_overlap": exact_overlap,
            "normalized_overlap": exact_overlap,
            "overlap_count": len(exact_overlap),
            "overlap_ratio": overlap_ratio,
        }

    # Relations comparison (subject_role, predicate, object_role, expected)
    ref_rel_quads = set((r.subject_role, r.predicate, r.object_role, bool(r.expected)) for r in reference_graph.relations)
    cand_rel_quads = set((r.subject_role, r.predicate, r.object_role, bool(r.expected)) for r in candidate_graph.relations)
    matched_rels = ref_rel_quads & cand_rel_quads
    missing_rels = ref_rel_quads - cand_rel_quads
    extra_rels = cand_rel_quads - ref_rel_quads

    # Operation groups deterministic 1-to-1 matching
    ref_ops = list(reference_graph.operation_groups)
    cand_ops = list(candidate_graph.operation_groups)
    cand_ops_by_id = {g.id: g for g in cand_ops}

    matched_pairs: list[tuple[str, str]] = []  # (ref_id, cand_id)
    matched_ref_ids: set[str] = set()
    matched_cand_ids: set[str] = set()
    op_mismatches: dict[str, dict[str, Any]] = {}
    missing_ops: set[str] = set()
    ambiguous_ops: dict[str, list[str]] = {}

    # Step 1: Match exact group IDs first
    for r_op in ref_ops:
        if r_op.id in cand_ops_by_id and r_op.id not in matched_cand_ids:
            c_op = cand_ops_by_id[r_op.id]
            matched_pairs.append((r_op.id, c_op.id))
            matched_ref_ids.add(r_op.id)
            matched_cand_ids.add(c_op.id)

    # Step 2: For unmatched reference groups, search unmatched candidates by (tool_role, target_role)
    for r_op in ref_ops:
        if r_op.id in matched_ref_ids:
            continue
        candidates_matching_roles = [
            c for c in cand_ops
            if c.id not in matched_cand_ids and c.tool_role == r_op.tool_role and c.target_role == r_op.target_role
        ]
        if len(candidates_matching_roles) == 1:
            c_op = candidates_matching_roles[0]
            matched_pairs.append((r_op.id, c_op.id))
            matched_ref_ids.add(r_op.id)
            matched_cand_ids.add(c_op.id)
        elif len(candidates_matching_roles) == 0:
            missing_ops.add(r_op.id)
        else:
            missing_ops.add(r_op.id)
            ambiguous_cands = [c.id for c in candidates_matching_roles]
            ambiguous_ops[r_op.id] = ambiguous_cands
            op_mismatches[r_op.id] = {
                "ambiguous_candidate_matches": ambiguous_cands
            }

    # Step 3: Compare matched operation-group fields
    for r_id, c_id in matched_pairs:
        r_op = next(g for g in ref_ops if g.id == r_id)
        c_op = next(g for g in cand_ops if g.id == c_id)
        diffs = {}
        if r_op.tool_role != c_op.tool_role:
            diffs["tool_role"] = {"reference": r_op.tool_role, "candidate": c_op.tool_role}
        if r_op.target_role != c_op.target_role:
            diffs["target_role"] = {"reference": r_op.target_role, "candidate": c_op.target_role}
        if r_op.required_target_count != c_op.required_target_count:
            diffs["required_target_count"] = {"reference": r_op.required_target_count, "candidate": c_op.required_target_count}
        if r_op.usage_policy != c_op.usage_policy:
            diffs["usage_policy"] = {"reference": r_op.usage_policy, "candidate": c_op.usage_policy}
        if set(r_op.required_relations) != set(c_op.required_relations):
            diffs["required_relations"] = {"reference": sorted(r_op.required_relations), "candidate": sorted(c_op.required_relations)}
        if r_op.context_role != c_op.context_role:
            diffs["context_role"] = {"reference": r_op.context_role, "candidate": c_op.context_role}
        if set(r_op.context_relations) != set(c_op.context_relations):
            diffs["context_relations"] = {"reference": sorted(r_op.context_relations), "candidate": sorted(c_op.context_relations)}
        if r_op.same_tool_must_cover_all_targets != c_op.same_tool_must_cover_all_targets:
            diffs["same_tool_must_cover_all_targets"] = {"reference": r_op.same_tool_must_cover_all_targets, "candidate": c_op.same_tool_must_cover_all_targets}
        if r_op.distinct_within_group != c_op.distinct_within_group:
            diffs["distinct_within_group"] = {"reference": r_op.distinct_within_group, "candidate": c_op.distinct_within_group}
        # Note: selection_preference is intentionally excluded from structural mismatch because it does not affect current grounding feasibility.
        if diffs:
            op_mismatches[r_id] = diffs

    extra_ops = set(c.id for c in cand_ops if c.id not in matched_cand_ids)

    cross_group_reuse_mismatch = (
        candidate_graph.cross_group_reuse_allowed != reference_graph.cross_group_reuse_allowed
    )
    graph_attribute_mismatches: dict[str, dict[str, Any]] = {}
    if cross_group_reuse_mismatch:
        graph_attribute_mismatches["cross_group_reuse_allowed"] = {
            "reference": reference_graph.cross_group_reuse_allowed,
            "candidate": candidate_graph.cross_group_reuse_allowed,
        }

    # Exact vs Identity metrics computation
    role_identity_recall = len(matched_roles) / len(ref_roles) if ref_roles else 1.0
    role_identity_precision = len(matched_roles) / len(cand_roles) if cand_roles else (1.0 if not ref_roles else 0.0)

    exact_roles = {
        r for r in matched_roles
        if r not in role_attribute_mismatches and role_cardinality_diagnostics.get(r, {}).get("cardinality_exact", False)
    }
    role_exact_recall = len(exact_roles) / len(ref_roles) if ref_roles else 1.0
    role_exact_precision = len(exact_roles) / len(cand_roles) if cand_roles else (1.0 if not ref_roles else 0.0)

    relation_recall = len(matched_rels) / len(ref_rel_quads) if ref_rel_quads else 1.0
    relation_precision = len(matched_rels) / len(cand_rel_quads) if cand_rel_quads else (1.0 if not ref_rel_quads else 0.0)

    operation_group_identity_recall = len(matched_ref_ids) / len(ref_ops) if ref_ops else 1.0
    operation_group_identity_precision = len(matched_cand_ids) / len(cand_ops) if cand_ops else (1.0 if not ref_ops else 0.0)

    exact_group_ref_ids = {r_id for r_id, _ in matched_pairs if r_id not in op_mismatches}
    exact_group_cand_ids = {c_id for _, c_id in matched_pairs if _ not in op_mismatches}
    operation_group_exact_recall = len(exact_group_ref_ids) / len(ref_ops) if ref_ops else 1.0
    operation_group_exact_precision = len(exact_group_cand_ids) / len(cand_ops) if cand_ops else (1.0 if not ref_ops else 0.0)

    all_cardinalities_exact = all(
        role_cardinality_diagnostics.get(r, {}).get("cardinality_exact", False)
        for r in matched_roles
    )

    reference_complete = (
        len(missing_roles) == 0
        and len(role_attribute_mismatches) == 0
        and len(missing_rels) == 0
        and len(missing_ops) == 0
        and len(op_mismatches) == 0
        and not cross_group_reuse_mismatch
    )

    exact_structural_match = (
        reference_complete
        and len(extra_roles) == 0
        and len(extra_rels) == 0
        and len(extra_ops) == 0
        and all_cardinalities_exact
    )

    # Invariance check: verify neither candidate nor reference was mutated during evaluation
    assert candidate_graph.nodes == cand_nodes_snapshot, "Evaluator violated candidate_graph.nodes non-mutation invariant"
    assert candidate_graph.relations == cand_rels_snapshot, "Evaluator violated candidate_graph.relations non-mutation invariant"
    assert candidate_graph.operation_groups == cand_ops_snapshot, "Evaluator violated candidate_graph.operation_groups non-mutation invariant"
    assert reference_graph.nodes == ref_nodes_snapshot, "Evaluator violated reference_graph.nodes non-mutation invariant"
    assert reference_graph.relations == ref_rels_snapshot, "Evaluator violated reference_graph.relations non-mutation invariant"
    assert reference_graph.operation_groups == ref_ops_snapshot, "Evaluator violated reference_graph.operation_groups non-mutation invariant"

    return GFReferenceEvaluationResult(
        domain=target_domain,
        task_instruction=target_instruction,
        reference_roles=tuple(sorted(ref_roles)),
        candidate_roles=tuple(sorted(cand_roles)),
        matched_roles=tuple(sorted(matched_roles)),
        missing_roles=tuple(sorted(missing_roles)),
        extra_roles=tuple(sorted(extra_roles)),
        role_attribute_mismatches=role_attribute_mismatches,
        role_cardinality_diagnostics=role_cardinality_diagnostics,
        role_normalization_diagnostics=role_normalization_diagnostics,
        category_diagnostics=category_diagnostics,
        reference_relations=tuple(sorted(ref_rel_quads, key=lambda x: (x[0], x[1], x[2], x[3]))),
        candidate_relations=tuple(sorted(cand_rel_quads, key=lambda x: (x[0], x[1], x[2], x[3]))),
        missing_relations=tuple(sorted(missing_rels, key=lambda x: (x[0], x[1], x[2], x[3]))),
        extra_relations=tuple(sorted(extra_rels, key=lambda x: (x[0], x[1], x[2], x[3]))),
        reference_operation_groups=tuple(g.to_dict() for g in reference_graph.operation_groups),
        candidate_operation_groups=tuple(g.to_dict() for g in candidate_graph.operation_groups),
        matched_operation_groups=tuple(sorted(matched_pairs)),
        missing_operation_groups=tuple(sorted(missing_ops)),
        extra_operation_groups=tuple(sorted(extra_ops)),
        ambiguous_operation_groups=ambiguous_ops,
        operation_group_mismatches=op_mismatches,
        cross_group_reuse_mismatch=cross_group_reuse_mismatch,
        graph_attribute_mismatches=graph_attribute_mismatches,
        reference_complete=reference_complete,
        exact_structural_match=exact_structural_match,
        structurally_complete=reference_complete,
        role_identity_recall=role_identity_recall,
        role_identity_precision=role_identity_precision,
        role_exact_recall=role_exact_recall,
        role_exact_precision=role_exact_precision,
        relation_recall=relation_recall,
        relation_precision=relation_precision,
        operation_group_identity_recall=operation_group_identity_recall,
        operation_group_identity_precision=operation_group_identity_precision,
        operation_group_exact_recall=operation_group_exact_recall,
        operation_group_exact_precision=operation_group_exact_precision,
        role_recall=role_identity_recall,
        role_precision=role_identity_precision,
        operation_group_recall=operation_group_identity_recall,
        operation_group_precision=operation_group_identity_precision,
    )

