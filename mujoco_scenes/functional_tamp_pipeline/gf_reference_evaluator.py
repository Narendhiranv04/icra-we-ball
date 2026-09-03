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
    operation_group_representation_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)

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
                "representation_diagnostics": self.operation_group_representation_diagnostics,
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
    cand_dict_snapshot = deepcopy(candidate_graph.to_dict())

    # Domain consistency guards
    if domain is not None and domain != candidate_graph.domain:
        raise ValueError(
            f"Reference evaluation domain mismatch: explicit domain='{domain}' differs from candidate domain='{candidate_graph.domain}'"
        )

    target_domain = candidate_graph.domain
    target_instruction = task_instruction or candidate_graph.task_instruction

    if reference_graph is None:
        reference_graph = GTSpecProvider().provide(
            domain=target_domain,
            task_instruction=target_instruction,
        )

    if reference_graph.domain != candidate_graph.domain:
        raise ValueError(
            f"Reference evaluation domain mismatch: candidate='{candidate_graph.domain}', reference='{reference_graph.domain}'"
        )

    ref_nodes_snapshot = deepcopy(reference_graph.nodes)
    ref_rels_snapshot = deepcopy(reference_graph.relations)
    ref_ops_snapshot = deepcopy(reference_graph.operation_groups)
    ref_dict_snapshot = deepcopy(reference_graph.to_dict())

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

        if ref_binding != cand_binding:
            diffs["binding_policy"] = {"reference": ref_binding, "candidate": cand_binding}

        if ref_binding == "REUSABLE" and cand_binding == "REUSABLE":
            # Semantic compatibility: candidate point count [c, c] is compatible if ref_min <= c <= ref_max
            is_compatible = (cand_min >= ref_min) and (cand_max <= ref_max) and (cand_min == cand_max or (cand_min == ref_min and cand_max == ref_max))
            is_exact = (cand_min == ref_min) and (cand_max == ref_max)
            role_cardinality_diagnostics[r_name] = {
                "reference_binding": ref_binding,
                "candidate_binding": cand_binding,
                "reference_range": [ref_min, ref_max],
                "candidate_range": [cand_min, cand_max],
                "cardinality_compatible": is_compatible,
                "cardinality_exact": is_exact,
                "reason": (
                    f"Candidate reusable count [{cand_min}, {cand_max}] is semantically compatible with reference allowed interval [{ref_min}, {ref_max}]"
                    if is_compatible
                    else f"Candidate reusable count [{cand_min}, {cand_max}] outside reference allowed interval [{ref_min}, {ref_max}]"
                ),
            }
            if not is_compatible:
                diffs["cardinality"] = {
                    "reference": [ref_min, ref_max],
                    "candidate": [cand_min, cand_max],
                }
        else:
            is_exact = (cand_min == ref_min) and (cand_max == ref_max)
            role_cardinality_diagnostics[r_name] = {
                "reference_binding": ref_binding,
                "candidate_binding": cand_binding,
                "reference_range": [ref_min, ref_max],
                "candidate_range": [cand_min, cand_max],
                "cardinality_compatible": is_exact,
                "cardinality_exact": is_exact,
                "reason": "Exact cardinality match required for non-reusable roles",
            }
            if not is_exact:
                diffs["cardinality"] = {
                    "reference": [ref_min, ref_max],
                    "candidate": [cand_min, cand_max],
                }

        # 3. Unary Predicates (with legacy role-function marker normalization for evaluation)
        ref_unary = set(ref_node.unary_predicates)
        cand_unary = set(cand_node.unary_predicates)

        ignored_markers = LEGACY_ROLE_FUNCTION_MARKERS.get((target_domain, r_name), set())
        effective_ref_unary = ref_unary - ignored_markers
        effective_cand_unary = cand_unary - ignored_markers

        if ignored_markers and (ref_unary & ignored_markers):
            role_normalization_diagnostics[r_name] = {
                "ignored_legacy_role_function_markers": sorted(ref_unary & ignored_markers),
                "description": "Legacy GT functional-role marker, represented by canonical role identity in the VLM interface.",
            }

        if effective_ref_unary != effective_cand_unary:
            diffs["unary_predicates"] = {
                "reference": sorted(effective_ref_unary),
                "candidate": sorted(effective_cand_unary),
            }

        if diffs:
            role_attribute_mismatches[r_name] = diffs

        # 5. Open-Vocabulary Semantic Categories (Diagnostic Only, not affecting reference_complete)
        ref_cats = list(ref_node.semantic_categories)
        cand_cats = list(cand_node.semantic_categories)
        ref_norm = [c.casefold().strip() for c in ref_cats]
        cand_norm = [c.casefold().strip() for c in cand_cats]
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

    # Relations comparison (4-tuples: subject_role, predicate, object_role, expected)
    ref_rel_quads = set((r.subject_role, r.predicate, r.object_role, bool(r.expected)) for r in reference_graph.relations)
    cand_rel_quads = set((r.subject_role, r.predicate, r.object_role, bool(r.expected)) for r in candidate_graph.relations)
    matched_rels = ref_rel_quads & cand_rel_quads
    missing_rels = ref_rel_quads - cand_rel_quads
    extra_rels = cand_rel_quads - ref_rel_quads

    # Operation groups deterministic 1-to-1 matching (Semantic-First, ID tie-break only)
    ref_ops = list(reference_graph.operation_groups)
    cand_ops = list(candidate_graph.operation_groups)

    matched_pairs: list[tuple[str, str]] = []  # (ref_id, cand_id)
    matched_ref_ids: set[str] = set()
    matched_cand_ids: set[str] = set()
    op_mismatches: dict[str, dict[str, Any]] = {}
    op_representation_diagnostics: dict[str, dict[str, Any]] = {}
    missing_ops: set[str] = set()
    ambiguous_ops: dict[str, list[str]] = {}

    for r_op in ref_ops:
        # Match by semantic grounding key: (tool_role, target_role, context_role)
        semantically_eligible = [
            c for c in cand_ops
            if c.id not in matched_cand_ids
            and c.tool_role == r_op.tool_role
            and c.target_role == r_op.target_role
            and c.context_role == r_op.context_role
        ]

        if len(semantically_eligible) == 1:
            c_op = semantically_eligible[0]
            matched_pairs.append((r_op.id, c_op.id))
            matched_ref_ids.add(r_op.id)
            matched_cand_ids.add(c_op.id)
        elif len(semantically_eligible) > 1:
            # Tie-break: if exactly one eligible candidate shares reference ID, select it
            same_id_candidates = [c for c in semantically_eligible if c.id == r_op.id]
            if len(same_id_candidates) == 1:
                c_op = same_id_candidates[0]
                matched_pairs.append((r_op.id, c_op.id))
                matched_ref_ids.add(r_op.id)
                matched_cand_ids.add(c_op.id)
            else:
                missing_ops.add(r_op.id)
                ambig_ids = [c.id for c in semantically_eligible]
                ambiguous_ops[r_op.id] = ambig_ids
                op_mismatches[r_op.id] = {
                    "ambiguous_candidate_matches": ambig_ids
                }
        else:
            missing_ops.add(r_op.id)

    # Compare matched operation-group fields
    for r_id, c_id in matched_pairs:
        r_op = next(g for g in ref_ops if g.id == r_id)
        c_op = next(g for g in cand_ops if g.id == c_id)
        diffs = {}
        rep_diffs = {}

        # Grounding-relevant fields
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

        # Non-grounding-relevant representation diagnostics (distinct_within_group & selection_preference)
        if r_op.distinct_within_group != c_op.distinct_within_group:
            rep_diffs["distinct_within_group"] = {
                "reference": r_op.distinct_within_group,
                "candidate": c_op.distinct_within_group,
                "grounding_relevant": False,
            }
        r_sel_pref = getattr(r_op, "selection_preference", None)
        c_sel_pref = getattr(c_op, "selection_preference", None)
        if r_sel_pref != c_sel_pref:
            rep_diffs["selection_preference"] = {
                "reference": r_sel_pref,
                "candidate": c_sel_pref,
                "grounding_relevant": False,
            }

        if diffs:
            op_mismatches[r_id] = diffs
        if rep_diffs:
            op_representation_diagnostics[r_id] = rep_diffs

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
    assert candidate_graph.to_dict() == cand_dict_snapshot, "Evaluator violated candidate_graph.to_dict() non-mutation invariant"
    assert reference_graph.nodes == ref_nodes_snapshot, "Evaluator violated reference_graph.nodes non-mutation invariant"
    assert reference_graph.relations == ref_rels_snapshot, "Evaluator violated reference_graph.relations non-mutation invariant"
    assert reference_graph.operation_groups == ref_ops_snapshot, "Evaluator violated reference_graph.operation_groups non-mutation invariant"
    assert reference_graph.to_dict() == ref_dict_snapshot, "Evaluator violated reference_graph.to_dict() non-mutation invariant"

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
        operation_group_representation_diagnostics=op_representation_diagnostics,
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
