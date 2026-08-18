"""Monotonic, evidence-centric observed graph for Workshop Phase 1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mujoco_scenes.workshop_phase1.types import (
    ObservedObjectTrack,
    ObservedRegion,
)


@dataclass
class GraphNode:
    """A node in the observed evidence graph."""

    node_id: str
    node_type: str  # "OBJECT", "FUNCTIONAL_REGION", "INSPECTION_REGION", "WORKPIECE"
    first_seen_stage: int
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "first_seen_stage": self.first_seen_stage,
            "attributes": self.attributes,
            "provenance": self.provenance,
        }


@dataclass
class GraphEdge:
    """A directed edge representing a grounded relation with evidence provenance."""

    source_id: str
    target_id: str
    relation: str  # "OBSERVED_IN", "SUPPORTED_BY", "NEAR", "CANDIDATE_FOR", "SEMANTICALLY_SATISFIES", "GEOMETRICALLY_SATISFIES"
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


class GrowingObservedGraph:
    """Monotonic observed evidence graph accumulating across inspection stages."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._stage_history: list[dict[str, Any]] = []

    @property
    def nodes(self) -> dict[str, GraphNode]:
        return self._nodes

    @property
    def edges(self) -> list[GraphEdge]:
        return self._edges

    def register_inspection_region_node(self, region_id: str, descriptor: str) -> None:
        if region_id not in self._nodes:
            self._nodes[region_id] = GraphNode(
                node_id=region_id,
                node_type="INSPECTION_REGION",
                first_seen_stage=0,
                attributes={"description": descriptor},
                provenance={"source": "scene_configuration"},
            )

    def register_workpiece_node(self, workpiece_id: str = "workpiece_joint_0001", target_spec: dict[str, Any] | None = None) -> None:
        if workpiece_id not in self._nodes:
            self._nodes[workpiece_id] = GraphNode(
                node_id=workpiece_id,
                node_type="WORKPIECE",
                first_seen_stage=0,
                attributes=target_spec or {"task": "frame_joint_repair"},
                provenance={"source": "task_specification"},
            )

    def update_from_observed_regions(self, regions: list[ObservedRegion], stage_idx: int) -> None:
        """Add or update candidate functional region nodes."""
        for reg in regions:
            r_id = reg.region_instance_id
            if r_id not in self._nodes:
                self._nodes[r_id] = GraphNode(
                    node_id=r_id,
                    node_type="FUNCTIONAL_REGION",
                    first_seen_stage=stage_idx,
                    attributes={
                        "proposal_bounds_m": reg.proposal_bounds_m,
                        "support_plane": reg.support_plane,
                        "cavity_geometry": reg.cavity_geometry,
                        "obstruction": reg.obstruction_evidence,
                    },
                    provenance={"observation_source": reg.observation_source},
                )
            else:
                # Update attributes
                self._nodes[r_id].attributes.update({
                    "support_plane": reg.support_plane,
                    "cavity_geometry": reg.cavity_geometry,
                    "obstruction": reg.obstruction_evidence,
                })

    def update_from_object_tracks(
        self,
        tracks: list[ObservedObjectTrack],
        stage_idx: int,
        stage_region_id: str,
    ) -> None:
        """Add new object tracks and update existing tracks in the evidence graph."""
        for trk in tracks:
            inst_id = trk.instance_id
            if inst_id not in self._nodes:
                self._nodes[inst_id] = GraphNode(
                    node_id=inst_id,
                    node_type="OBJECT",
                    first_seen_stage=stage_idx,
                    attributes={
                        "semantic_belief": trk.current_semantic_belief,
                        "geometric_properties": trk.current_geometric_properties,
                        "point_count": len(trk.fused_points) if trk.fused_points is not None else 0,
                    },
                    provenance={
                        "source_region": stage_region_id,
                        "contributing_cameras": list(trk.contributing_cameras),
                        "evidence_count": trk.evidence_count,
                    },
                )
                # Edge to source inspection region
                self.add_edge(
                    source_id=inst_id,
                    target_id=stage_region_id,
                    relation="OBSERVED_IN",
                    confidence=trk.overall_confidence,
                    evidence={"stage_index": stage_idx},
                )
            else:
                self._nodes[inst_id].attributes.update({
                    "semantic_belief": trk.current_semantic_belief,
                    "geometric_properties": trk.current_geometric_properties,
                    "point_count": len(trk.fused_points) if trk.fused_points is not None else 0,
                })
                self._nodes[inst_id].provenance["evidence_count"] = trk.evidence_count
                self._nodes[inst_id].provenance["contributing_cameras"] = list(trk.contributing_cameras)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        # Avoid exact duplicates
        for edge in self._edges:
            if edge.source_id == source_id and edge.target_id == target_id and edge.relation == relation:
                edge.confidence = confidence
                if evidence:
                    edge.evidence.update(evidence)
                return
        self._edges.append(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                confidence=confidence,
                evidence=evidence or {},
            )
        )

    def snapshot(self, stage_idx: int, output_dir: Path | None = None) -> dict[str, Any]:
        """Generate and optionally save an immutable JSON snapshot of graph state."""
        snap = {
            "stage_index": stage_idx,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
        }
        self._stage_history.append(snap)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            snap_path = output_dir / f"observed_graph_stage_{stage_idx:03d}.json"
            with open(snap_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2)
        return snap
