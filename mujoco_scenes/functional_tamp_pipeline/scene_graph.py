"""Lightweight observed scene graph populated only by perception outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class ObservedObject:
    instance_id: str
    semantic_labels: dict[str, Any] = field(default_factory=dict)
    region: str | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    unary_properties: dict[str, Any] = field(default_factory=dict)
    last_seen_stage: int = 0


@dataclass
class ObservedRelation:
    subject: str
    predicate: str
    object: str
    status: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservedSceneGraph:
    objects: dict[str, ObservedObject] = field(default_factory=dict)
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], ObservedRelation] = field(default_factory=dict)
    inspected_regions: list[str] = field(default_factory=list)
    stage_index: int = -1

    def update_objects(self, objects: Iterable[ObservedObject], stage_index: int) -> None:
        self.stage_index = stage_index
        for observed in objects:
            observed.last_seen_stage = stage_index
            self.objects[observed.instance_id] = observed

    def mark_region_inspected(self, region: str) -> None:
        if region not in self.inspected_regions:
            self.inspected_regions.append(region)
        self.regions.setdefault(region, {})["inspected"] = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "inspected_regions": list(self.inspected_regions),
            "objects": {key: asdict(value) for key, value in sorted(self.objects.items())},
            "regions": dict(sorted(self.regions.items())),
            "relations": [asdict(value) for _, value in sorted(self.relations.items())],
        }
