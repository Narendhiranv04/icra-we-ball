"""Lightweight observed scene graph G_O populated only by perception outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping


@dataclass
class ObservedNode:
    """Node in the observed scene graph G_O representing an observed entity."""

    instance_id: str
    entity_kind: str = "OBJECT"  # "OBJECT", "REGION", "FIXED_TARGET"
    canonical_category: str | None = None
    semantic_labels: dict[str, Any] = field(default_factory=dict)
    source_region: str | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    unary_properties: dict[str, Any] = field(default_factory=dict)
    unary_predicates: dict[str, Any] = field(default_factory=dict)
    first_seen_stage: int = 0
    last_seen_stage: int = 0

    @property
    def region(self) -> str | None:
        return self.source_region

    @region.setter
    def region(self, value: str | None) -> None:
        self.source_region = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "entity_kind": self.entity_kind,
            "canonical_category": self.canonical_category,
            "semantic_labels": dict(self.semantic_labels),
            "source_region": self.source_region,
            "region": self.source_region,
            "geometry": dict(self.geometry),
            "unary_properties": dict(self.unary_properties),
            "unary_predicates": dict(self.unary_predicates),
            "first_seen_stage": self.first_seen_stage,
            "last_seen_stage": self.last_seen_stage,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservedNode:
        return cls(
            instance_id=str(data["instance_id"]),
            entity_kind=str(data.get("entity_kind", "OBJECT")),
            canonical_category=data.get("canonical_category"),
            semantic_labels=dict(data.get("semantic_labels", {})),
            source_region=data.get("source_region", data.get("region")),
            geometry=dict(data.get("geometry", {})),
            unary_properties=dict(data.get("unary_properties", {})),
            unary_predicates=dict(data.get("unary_predicates", {})),
            first_seen_stage=int(data.get("first_seen_stage", 0)),
            last_seen_stage=int(data.get("last_seen_stage", 0)),
        )


# Backward-compatible alias
ObservedObject = ObservedNode


@dataclass
class ObservedRelation:
    """Explicit verified or evaluated binary relation between observed entities."""

    subject_id: str
    predicate: str
    object_id: str
    status: str = "UNKNOWN"  # "TRUE", "FALSE", "UNKNOWN"
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def subject(self) -> str:
        return self.subject_id

    @subject.setter
    def subject(self, val: str) -> None:
        self.subject_id = val

    @property
    def object(self) -> str:
        return self.object_id

    @object.setter
    def object(self, val: str) -> None:
        self.object_id = val

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "object": self.object_id,
            "status": self.status,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservedRelation:
        return cls(
            subject_id=str(data.get("subject_id", data.get("subject"))),
            predicate=str(data["predicate"]),
            object_id=str(data.get("object_id", data.get("object"))),
            status=str(data.get("status", "UNKNOWN")),
            evidence=dict(data.get("evidence", {})),
        )


@dataclass
class ObservedSceneGraph:
    """Canonical Observed Scene Graph G_O = (V_O, E_O)."""

    nodes: dict[str, ObservedNode] = field(default_factory=dict)
    relations: dict[tuple[str, str, str], ObservedRelation] = field(default_factory=dict)
    regions: dict[str, dict[str, Any]] = field(default_factory=dict)
    inspected_regions: list[str] = field(default_factory=list)
    stage_index: int = -1

    @property
    def objects(self) -> dict[str, ObservedNode]:
        """Backward-compatible access to object nodes."""
        return {k: v for k, v in self.nodes.items() if v.entity_kind == "OBJECT"}

    def add_node(self, node: ObservedNode) -> None:
        self.nodes[node.instance_id] = node

    def get_node(self, instance_id: str) -> ObservedNode | None:
        return self.nodes.get(instance_id)

    def add_relation(self, relation: ObservedRelation) -> None:
        key = (relation.predicate, relation.subject_id, relation.object_id)
        self.relations[key] = relation

    def get_relation(
        self, predicate: str, subject_id: str, object_id: str
    ) -> ObservedRelation | None:
        return self.relations.get((predicate, subject_id, object_id))

    def update_objects(self, objects: Iterable[ObservedNode], stage_index: int) -> None:
        self.stage_index = stage_index
        for observed in objects:
            observed.last_seen_stage = stage_index
            self.nodes[observed.instance_id] = observed

    def mark_region_inspected(self, region: str) -> None:
        if region not in self.inspected_regions:
            self.inspected_regions.append(region)
        self.regions.setdefault(region, {})["inspected"] = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "inspected_regions": list(self.inspected_regions),
            "nodes": {key: value.to_dict() for key, value in sorted(self.nodes.items())},
            "objects": {key: value.to_dict() for key, value in sorted(self.objects.items())},
            "regions": dict(sorted(self.regions.items())),
            "relations": [value.to_dict() for _, value in sorted(self.relations.items())],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservedSceneGraph:
        graph = cls(
            inspected_regions=list(data.get("inspected_regions", ())),
            stage_index=int(data.get("stage_index", -1)),
            regions=dict(data.get("regions", {})),
        )
        nodes_raw = data.get("nodes", data.get("objects", {}))
        if isinstance(nodes_raw, dict):
            for node_data in nodes_raw.values():
                graph.add_node(ObservedNode.from_dict(node_data))
        elif isinstance(nodes_raw, list):
            for node_data in nodes_raw:
                graph.add_node(ObservedNode.from_dict(node_data))

        relations_raw = data.get("relations", [])
        if isinstance(relations_raw, list):
            for r_data in relations_raw:
                graph.add_relation(ObservedRelation.from_dict(r_data))
        elif isinstance(relations_raw, dict):
            for r_data in relations_raw.values():
                graph.add_relation(ObservedRelation.from_dict(r_data))
        return graph

