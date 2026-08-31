"""Generic runtime G_F structural consistency validation.

Enforces domain-independent structural consistency and frozen predicate signature
compliance on canonical FunctionalRequirementGraph without any expected-task oracle knowledge.

IMPORTANT SCIENTIFIC BOUNDARY:
- Runtime validation checks representation consistency against the frozen Predicate Registry.
- Runtime NEVER checks or rejects against ground-truth/reference task specifications.
- Expected task roles and relations are evaluated offline in Pass 3.6B.
"""

from __future__ import annotations

from typing import Any

from .errors import MalformedVLMSpecificationError
from .models import FunctionalRequirementGraph
from .predicate_registry import validate_predicate_signature


def validate_runtime_gf(graph: FunctionalRequirementGraph) -> None:
    """Validate domain-independent structural consistency and predicate signatures of canonical FunctionalRequirementGraph."""
    if not isinstance(graph, FunctionalRequirementGraph):
        raise MalformedVLMSpecificationError(
            f"Expected FunctionalRequirementGraph instance, got {type(graph).__name__}"
        )

    if not isinstance(graph.domain, str) or not graph.domain.strip():
        raise MalformedVLMSpecificationError("FunctionalRequirementGraph domain must be a non-empty string")

    domain_norm = graph.domain.strip().lower()
    if domain_norm not in {"kitchen", "living_room", "workshop"}:
        raise MalformedVLMSpecificationError(f"Unsupported graph domain {graph.domain!r}")

    if not graph.nodes:
        raise MalformedVLMSpecificationError("FunctionalRequirementGraph must have at least one node")

    for name, node in graph.nodes.items():
        if not isinstance(name, str) or not name.strip():
            raise MalformedVLMSpecificationError("Role node name must be a non-empty string")
        if node.entity_kind not in {"OBJECT", "REGION", "FIXED_TARGET"}:
            raise MalformedVLMSpecificationError(
                f"Role {name!r} has invalid entity_kind {node.entity_kind!r}"
            )
        if node.minimum_count < 1:
            raise MalformedVLMSpecificationError(
                f"Role {name!r} minimum count must be >= 1, got {node.minimum_count}"
            )
        if node.maximum_count < node.minimum_count:
            raise MalformedVLMSpecificationError(
                f"Role {name!r} maximum count ({node.maximum_count}) < minimum count ({node.minimum_count})"
            )
        if node.binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
            raise MalformedVLMSpecificationError(
                f"Role {name!r} has invalid binding_policy {node.binding_policy!r}"
            )

        # Validate unary predicates against frozen predicate registry
        for unary_pred in node.unary_predicates:
            validate_predicate_signature(
                domain=domain_norm,
                predicate=unary_pred,
                subject_kind=node.entity_kind,
                subject_role=name,
            )

    for rel in graph.relations:
        if rel.subject_role not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Relation subject {rel.subject_role!r} not in graph nodes"
            )
        if rel.object_role not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Relation object {rel.object_role!r} not in graph nodes"
            )
        if not isinstance(rel.predicate, str) or not rel.predicate.strip():
            raise MalformedVLMSpecificationError(
                f"Relation between {rel.subject_role!r} and {rel.object_role!r} has empty predicate"
            )

        s_node = graph.nodes[rel.subject_role]
        o_node = graph.nodes[rel.object_role]
        # Validate binary predicate signature and direction against frozen registry
        validate_predicate_signature(
            domain=domain_norm,
            predicate=rel.predicate,
            subject_kind=s_node.entity_kind,
            object_kind=o_node.entity_kind,
            subject_role=rel.subject_role,
            object_role=rel.object_role,
        )

    seen_op_ids: set[str] = set()
    for grp in graph.operation_groups:
        if not isinstance(grp.id, str) or not grp.id.strip():
            raise MalformedVLMSpecificationError("Operation group id must be a non-empty string")
        if grp.id in seen_op_ids:
            raise MalformedVLMSpecificationError(f"Duplicate operation group id {grp.id!r}")
        seen_op_ids.add(grp.id)

        if grp.tool_role not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} tool_role {grp.tool_role!r} not in graph nodes"
            )
        if grp.target_role not in graph.nodes:
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} target_role {grp.target_role!r} not in graph nodes"
            )
        if grp.required_target_count < 1:
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} required_target_count must be >= 1, got {grp.required_target_count}"
            )
        target_node = graph.nodes[grp.target_role]
        if grp.required_target_count > target_node.maximum_count:
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} required_target_count {grp.required_target_count} "
                f"exceeds target role {grp.target_role!r} maximum count {target_node.maximum_count}"
            )
        if grp.usage_policy not in {"SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"}:
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} has invalid usage_policy {grp.usage_policy!r}"
            )
        if not grp.required_relations:
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} has empty required_relations"
            )
        cleaned_req_rels = []
        for r in grp.required_relations:
            if not isinstance(r, str) or not r.strip():
                raise MalformedVLMSpecificationError(
                    f"Operation group {grp.id!r} required_relations contains empty item"
                )
            cleaned_req_rels.append(r.strip())
        if len(cleaned_req_rels) != len(set(cleaned_req_rels)):
            raise MalformedVLMSpecificationError(
                f"Operation group {grp.id!r} required_relations contains duplicates"
            )

        tool_node = graph.nodes[grp.tool_role]
        for req_rel in cleaned_req_rels:
            validate_predicate_signature(
                domain=domain_norm,
                predicate=req_rel,
                subject_kind=tool_node.entity_kind,
                object_kind=target_node.entity_kind,
                subject_role=grp.tool_role,
                object_role=grp.target_role,
            )

        if grp.context_role is not None:
            if not isinstance(grp.context_role, str) or not grp.context_role.strip():
                raise MalformedVLMSpecificationError(
                    f"Operation group {grp.id!r} context_role must be a non-empty string"
                )
            if grp.context_role not in graph.nodes:
                raise MalformedVLMSpecificationError(
                    f"Operation group {grp.id!r} context_role {grp.context_role!r} not in graph nodes"
                )
            if not grp.context_relations:
                raise MalformedVLMSpecificationError(
                    f"Operation group {grp.id!r} specifies context_role {grp.context_role!r} "
                    "but has empty context_relations"
                )
            cleaned_ctx_rels = []
            for r in grp.context_relations:
                if not isinstance(r, str) or not r.strip():
                    raise MalformedVLMSpecificationError(
                        f"Operation group {grp.id!r} context_relations contains empty item"
                    )
                cleaned_ctx_rels.append(r.strip())
            if len(cleaned_ctx_rels) != len(set(cleaned_ctx_rels)):
                raise MalformedVLMSpecificationError(
                    f"Operation group {grp.id!r} context_relations contains duplicates"
                )

            context_node = graph.nodes[grp.context_role]
            for ctx_rel in cleaned_ctx_rels:
                validate_predicate_signature(
                    domain=domain_norm,
                    predicate=ctx_rel,
                    subject_kind=tool_node.entity_kind,
                    object_kind=context_node.entity_kind,
                    subject_role=grp.tool_role,
                    object_role=grp.context_role,
                )
        else:
            if grp.context_relations:
                raise MalformedVLMSpecificationError(
                    f"Operation group {grp.id!r} has context_relations without context_role"
                )

    if graph.candidate_regions and graph.region_ranking:
        if len(graph.region_ranking) != len(set(graph.region_ranking)):
            raise MalformedVLMSpecificationError(
                f"Duplicate regions in region_ranking: {graph.region_ranking}"
            )
        for r in graph.region_ranking:
            if r not in graph.candidate_regions:
                raise MalformedVLMSpecificationError(
                    f"Region {r!r} in region_ranking not in candidate_regions"
                )
