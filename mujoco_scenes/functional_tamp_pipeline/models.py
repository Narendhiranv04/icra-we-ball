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
    semantic_categories: tuple[str, ...] = ()
    unary_predicates: tuple[str, ...] = ()
    numeric_constraints: tuple[NumericConstraint, ...] = ()
    binding_policy: str = "DISTINCT"  # "DISTINCT", "REUSABLE", "SHARED"
    verification_mode: str = "SEMANTIC_AND_GEOMETRIC"  # "SEMANTIC_ONLY", "SEMANTIC_AND_GEOMETRIC"
    description: str = ""
    semantic_hints: tuple[str, ...] = ()

    # Backward compatibility properties & aliases
    @property
    def distinct(self) -> bool:
        return self.binding_policy == "DISTINCT" or self.count > 1

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
        return cls(
            name=str(data["name"]),
            entity_kind=str(data.get("entity_kind", "OBJECT")),
            count=int(data.get("count", 1)),
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
    """Structure for repeated / multi-target tool operations (e.g. Kitchen)."""

    id: str
    function: str
    tool_role: str
    target_role: str
    required_target_count: int
    usage_policy: str  # "SEQUENTIAL_REUSE_ALLOWED", "DEDICATED_PER_TARGET"
    required_relations: tuple[str, ...] = ()

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
        return cls(
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


# Backward-compatible alias
FunctionalSpecification = FunctionalRequirementGraph


@dataclass(frozen=True)
class GraphGroundingResult:
    """Output of graph grounding phi : G_F -> G_O."""

    status: str  # "COMPLETE", "INCOMPLETE", "INFEASIBLE"
    complete: bool
    assignment: dict[str, Any] | None = None
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
        return self.missing_roles or self.unresolved_constraints

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "complete": self.complete,
            "satisfied": self.satisfied,
            "assignment": self.assignment,
            "missing_roles": list(self.missing_roles),
            "missing_requirements": list(self.missing_requirements),
            "unsatisfied_relations": list(self.unsatisfied_relations),
            "unresolved_constraints": list(self.unresolved_constraints),
            "evidence": self.evidence,
        }


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

