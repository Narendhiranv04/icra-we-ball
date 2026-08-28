"""Small shared data contracts for the canonical functional TAMP pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NumericConstraint:
    """Explicit numeric property requirement."""

    property_name: str
    operator: str  # ">=", "<=", "=="
    threshold: float
    unit: str

    def matches(self, value: float | int | None) -> bool:
        if value is None:
            return False
        val = float(value)
        if self.operator == ">=":
            return val >= self.threshold
        if self.operator == "<=":
            return val <= self.threshold
        if self.operator == "==":
            return abs(val - self.threshold) < 1e-6
        return False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NumericConstraint:
        return cls(
            property_name=str(data["property_name"]),
            operator=str(data["operator"]),
            threshold=float(data["threshold"]),
            unit=str(data["unit"]),
        )


@dataclass(frozen=True)
class FunctionalRelation:
    """Explicit required directional binary relation between functional roles."""

    subject_role: str
    predicate: str
    object_role: str
    expected: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FunctionalRelation:
        return cls(
            subject_role=str(data["subject_role"]),
            predicate=str(data["predicate"]),
            object_role=str(data["object_role"]),
            expected=bool(data.get("expected", True)),
        )


@dataclass(frozen=True)
class FunctionalRole:
    """Node in the functional requirement graph G_F representing a required role."""

    name: str
    entity_kind: str = "OBJECT"  # "OBJECT", "REGION", "FIXED_TARGET"
    count: int = 1
    min_count: int | None = None
    max_count: int | None = None
    preference: str | None = None  # e.g. "minimize_distinct"
    semantic_categories: tuple[str, ...] = ()
    unary_predicates: tuple[str, ...] = ()
    numeric_constraints: tuple[NumericConstraint, ...] = ()
    binding_policy: str = "DISTINCT"  # "DISTINCT", "REUSABLE", "SHARED"
    verification_mode: str = "SEMANTIC_AND_GEOMETRIC"  # "SEMANTIC_ONLY", "SEMANTIC_AND_GEOMETRIC"
    description: str = ""
    semantic_hints: tuple[str, ...] = ()

    @property
    def minimum_count(self) -> int:
        return self.min_count if self.min_count is not None else self.count

    @property
    def maximum_count(self) -> int:
        return self.max_count if self.max_count is not None else self.count

    # Backward compatibility properties & aliases
    @property
    def distinct(self) -> bool:
        return self.binding_policy == "DISTINCT" or self.maximum_count > 1

    @property
    def reusable(self) -> bool:
        return self.binding_policy == "REUSABLE"

    @property
    def shared(self) -> bool:
        return self.binding_policy == "SHARED"

    @property
    def unary_properties(self) -> tuple[str, ...]:
        return self.unary_predicates

    @property
    def required_relations(self) -> tuple[str, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_kind": self.entity_kind,
            "count": self.count,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "preference": self.preference,
            "semantic_categories": list(self.semantic_categories),
            "unary_predicates": list(self.unary_predicates),
            "numeric_constraints": [c.to_dict() for c in self.numeric_constraints],
            "binding_policy": self.binding_policy,
            "verification_mode": self.verification_mode,
            "description": self.description,
            "semantic_hints": list(self.semantic_hints),
            "distinct": self.distinct,
            "reusable": self.reusable,
            "shared": self.shared,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FunctionalRole:
        constraints = tuple(
            NumericConstraint.from_dict(c)
            for c in data.get("numeric_constraints", [])
        )
        binding = data.get("binding_policy")
        if not binding:
            if data.get("reusable"):
                binding = "REUSABLE"
            elif data.get("shared"):
                binding = "SHARED"
            else:
                binding = "DISTINCT"
        count = int(data.get("count", 1))
        min_c = data.get("min_count")
        max_c = data.get("max_count")
        return cls(
            name=str(data["name"]),
            entity_kind=str(data.get("entity_kind", "OBJECT")),
            count=count,
            min_count=int(min_c) if min_c is not None else None,
            max_count=int(max_c) if max_c is not None else None,
            preference=str(data["preference"]) if data.get("preference") else None,
            semantic_categories=tuple(map(str, data.get("semantic_categories", ()))),
            unary_predicates=tuple(map(str, data.get("unary_predicates", data.get("unary_properties", ())))),
            numeric_constraints=constraints,
            binding_policy=str(binding),
            verification_mode=str(data.get("verification_mode", "SEMANTIC_AND_GEOMETRIC")),
            description=str(data.get("description", "")),
            semantic_hints=tuple(map(str, data.get("semantic_hints", ()))),
        )


@dataclass(frozen=True)
class OperationGroup:
    """Structure for repeated / multi-target tool operations (e.g. Kitchen, Living Room)."""

    id: str
    function: str
    tool_role: str
    target_role: str
    required_target_count: int
    usage_policy: str  # "SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"
    required_relations: tuple[str, ...] = ()
    context_role: str | None = None
    context_relations: tuple[str, ...] = ()
    distinct_within_group: bool = True
    same_tool_must_cover_all_targets: bool = False
    selection_preference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OperationGroup:
        return cls(
            id=str(data["id"]),
            function=str(data["function"]),
            tool_role=str(data["tool_role"]),
            target_role=str(data["target_role"]),
            required_target_count=int(data["required_target_count"]),
            usage_policy=str(data["usage_policy"]),
            required_relations=tuple(map(str, data.get("required_relations", ()))),
            context_role=str(data["context_role"]) if data.get("context_role") else None,
            context_relations=tuple(map(str, data.get("context_relations", ()))),
            distinct_within_group=bool(data.get("distinct_within_group", True)),
            same_tool_must_cover_all_targets=bool(data.get("same_tool_must_cover_all_targets", False)),
            selection_preference=str(data["selection_preference"]) if data.get("selection_preference") else None,
        )


@dataclass(frozen=True)
class FunctionalRequirementGraph:
    """Canonical Functional Requirement Graph G_F = (V_F, E_F)."""

    domain: str
    task_instruction: str
    nodes: dict[str, FunctionalRole]
    relations: tuple[FunctionalRelation, ...] = ()
    operation_groups: tuple[OperationGroup, ...] = ()
    cross_group_reuse_allowed: bool = True
    detector_vocabulary: tuple[str, ...] = ()
    candidate_regions: tuple[str, ...] = ()
    region_ranking: tuple[str, ...] = ()
    source: str = "UNKNOWN"
    raw_requirements: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def roles(self) -> tuple[FunctionalRole, ...]:
        return tuple(self.nodes.values())

    def get_node(self, name: str) -> FunctionalRole | None:
        return self.nodes.get(name)

    def get_outgoing_relations(self, subject_role: str) -> tuple[FunctionalRelation, ...]:
        return tuple(r for r in self.relations if r.subject_role == subject_role)

    def get_incoming_relations(self, object_role: str) -> tuple[FunctionalRelation, ...]:
        return tuple(r for r in self.relations if r.object_role == object_role)

    def validate(self) -> None:
        """Validate structural integrity of the functional requirement graph."""
        for name, node in self.nodes.items():
            if node.minimum_count < 1:
                raise ValueError(
                    f"Invalid functional graph: role {name!r} minimum count must be >= 1, got {node.minimum_count}"
                )
            if node.maximum_count < node.minimum_count:
                raise ValueError(
                    f"Invalid functional graph: role {name!r} max_count ({node.maximum_count}) "
                    f"< min_count ({node.minimum_count})"
                )
            if node.binding_policy not in {"DISTINCT", "REUSABLE", "SHARED"}:
                raise ValueError(
                    f"Invalid functional graph: role {name!r} has unknown binding_policy {node.binding_policy!r}"
                )

        for rel in self.relations:
            if rel.subject_role not in self.nodes:
                raise ValueError(
                    f"Invalid functional graph: relation subject {rel.subject_role!r} "
                    f"not in nodes ({list(self.nodes.keys())})"
                )
            if rel.object_role not in self.nodes:
                raise ValueError(
                    f"Invalid functional graph: relation object {rel.object_role!r} "
                    f"not in nodes ({list(self.nodes.keys())})"
                )
            if not rel.predicate:
                raise ValueError(f"Invalid functional graph: relation has empty predicate: {rel}")

        seen_op_ids: set[str] = set()
        for grp in self.operation_groups:
            if grp.id in seen_op_ids:
                raise ValueError(f"Invalid functional graph: duplicate operation group id {grp.id!r}")
            seen_op_ids.add(grp.id)

            if grp.tool_role not in self.nodes:
                raise ValueError(
                    f"Invalid functional graph: operation group {grp.id!r} tool_role "
                    f"{grp.tool_role!r} not in nodes ({list(self.nodes.keys())})"
                )
            if grp.target_role not in self.nodes:
                raise ValueError(
                    f"Invalid functional graph: operation group {grp.id!r} target_role "
                    f"{grp.target_role!r} not in nodes ({list(self.nodes.keys())})"
                )
            if grp.context_role and grp.context_role not in self.nodes:
                raise ValueError(
                    f"Invalid functional graph: operation group {grp.id!r} context_role "
                    f"{grp.context_role!r} not in nodes ({list(self.nodes.keys())})"
                )
            if grp.usage_policy not in {"SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"}:
                raise ValueError(
                    f"Invalid functional graph: operation group {grp.id!r} usage_policy "
                    f"{grp.usage_policy!r} not supported"
                )
            target_node = self.nodes[grp.target_role]
            if grp.required_target_count > target_node.maximum_count:
                raise ValueError(
                    f"Invalid functional graph: operation group {grp.id!r} required_target_count "
                    f"{grp.required_target_count} exceeds target role {grp.target_role!r} max_count {target_node.maximum_count}"
                )

        if self.candidate_regions and self.region_ranking:
            if len(self.region_ranking) != len(set(self.region_ranking)):
                raise ValueError(
                    f"Invalid functional graph: duplicate regions in region_ranking: {self.region_ranking}"
                )
            if set(self.region_ranking) != set(self.candidate_regions):
                raise ValueError(
                    f"Invalid functional graph: region_ranking {self.region_ranking} "
                    f"must match candidate_regions {self.candidate_regions}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task_instruction": self.task_instruction,
            "roles": [node.to_dict() for node in self.nodes.values()],
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "relations": [r.to_dict() for r in self.relations],
            "operation_groups": [g.to_dict() for g in self.operation_groups],
            "cross_group_reuse_allowed": self.cross_group_reuse_allowed,
            "detector_vocabulary": list(self.detector_vocabulary),
            "candidate_regions": list(self.candidate_regions),
            "region_ranking": list(self.region_ranking),
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FunctionalRequirementGraph:
        if "nodes" in data and isinstance(data["nodes"], dict):
            nodes = {
                name: FunctionalRole.from_dict(node_data)
                for name, node_data in data["nodes"].items()
            }
        elif "roles" in data and isinstance(data["roles"], (list, tuple)):
            nodes = {
                role_data["name"]: FunctionalRole.from_dict(role_data)
                for role_data in data["roles"]
            }
        else:
            nodes = {}

        relations = tuple(
            FunctionalRelation.from_dict(r)
            for r in data.get("relations", ())
        )
        operation_groups = tuple(
            OperationGroup.from_dict(g)
            for g in data.get("operation_groups", ())
        )
        graph = cls(
            domain=str(data["domain"]),
            task_instruction=str(data["task_instruction"]),
            nodes=nodes,
            relations=relations,
            operation_groups=operation_groups,
            cross_group_reuse_allowed=bool(data.get("cross_group_reuse_allowed", True)),
            detector_vocabulary=tuple(map(str, data.get("detector_vocabulary", ()))),
            candidate_regions=tuple(map(str, data.get("candidate_regions", ()))),
            region_ranking=tuple(map(str, data.get("region_ranking", ()))),
            source=str(data.get("source", "UNKNOWN")),
            metadata=dict(data.get("metadata", {})),
        )
        return graph


# Backward-compatible alias
FunctionalSpecification = FunctionalRequirementGraph


@dataclass(frozen=True)
class GraphGroundingResult:
    """Output of graph grounding phi : G_F -> G_O."""

    status: str  # "COMPLETE", "INCOMPLETE", "INFEASIBLE"
    complete: bool
    assignment: dict[str, Any] | None = None
    operation_bindings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    missing_roles: tuple[str, ...] = ()
    unsatisfied_relations: tuple[dict[str, Any], ...] = ()
    unresolved_constraints: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def satisfied(self) -> bool:
        return self.complete

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        reqs = list(self.missing_roles)
        reqs.extend(self.unresolved_constraints)
        for rel in self.unsatisfied_relations:
            if isinstance(rel, dict):
                reqs.append(f"{rel.get('predicate')}({rel.get('subject_role')}, {rel.get('object_role')})")
            else:
                reqs.append(str(rel))
        return tuple(reqs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "complete": self.complete,
            "satisfied": self.satisfied,
            "assignment": self.assignment,
            "operation_bindings": self.operation_bindings,
            "missing_roles": list(self.missing_roles),
            "missing_requirements": list(self.missing_requirements),
            "unsatisfied_relations": list(self.unsatisfied_relations),
            "unresolved_constraints": list(self.unresolved_constraints),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GraphGroundingResult:
        return cls(
            status=str(data["status"]),
            complete=bool(data.get("complete", data.get("satisfied", False))),
            assignment=dict(data["assignment"]) if data.get("assignment") is not None else None,
            operation_bindings=dict(data.get("operation_bindings", {})),
            missing_roles=tuple(map(str, data.get("missing_roles", ()))),
            unsatisfied_relations=tuple(dict(r) for r in data.get("unsatisfied_relations", ())),
            unresolved_constraints=tuple(map(str, data.get("unresolved_constraints", ()))),
            evidence=dict(data.get("evidence", {})),
        )


# Backward-compatible alias
SatisfactionResult = GraphGroundingResult


@dataclass(frozen=True)
class PipelineResult:
    domain: str
    variant: str
    mode: str
    status: str
    inspected_regions: tuple[str, ...] = ()
    assignment: dict[str, Any] | None = None
    plan: tuple[dict[str, Any], ...] = ()
    search_statistics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

