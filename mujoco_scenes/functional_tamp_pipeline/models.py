"""Small shared data contracts for the canonical functional TAMP pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FunctionalRole:
    name: str
    count: int = 1
    semantic_categories: tuple[str, ...] = ()
    unary_properties: tuple[str, ...] = ()
    required_relations: tuple[str, ...] = ()
    distinct: bool = False
    reusable: bool = False
    shared: bool = False


@dataclass(frozen=True)
class FunctionalSpecification:
    domain: str
    task_instruction: str
    roles: tuple[FunctionalRole, ...]
    detector_vocabulary: tuple[str, ...]
    candidate_regions: tuple[str, ...]
    region_ranking: tuple[str, ...]
    source: str
    raw_requirements: tuple[Any, ...] = field(default_factory=tuple, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw_requirements", None)
        return payload


@dataclass(frozen=True)
class SatisfactionResult:
    satisfied: bool
    assignment: dict[str, str] | None = None
    missing_requirements: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "INCOMPLETE"


@dataclass(frozen=True)
class PipelineResult:
    domain: str
    variant: str
    mode: str
    status: str
    inspected_regions: tuple[str, ...] = ()
    assignment: dict[str, str] | None = None
    plan: tuple[dict[str, Any], ...] = ()
    search_statistics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
