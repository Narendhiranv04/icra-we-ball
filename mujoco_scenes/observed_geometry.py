"""Persistent, semantics-light state built only from accepted point clouds."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mujoco_scenes.geometry_checker import PointCloudRun, write_ply
from mujoco_scenes.geometry_properties import (
    extract_object_properties,
    load_geometry_config,
    pairwise_relation_evaluation,
)


@dataclass(frozen=True)
class CompatibilitySelection:
    status: str
    selected_object_id: str | None
    evaluations: tuple[dict[str, Any], ...]


class ObservedGeometryState:
    """Generic object registry with no access to hidden scene contents."""

    def __init__(
        self,
        root: str | Path,
        *,
        scene_name: str,
        geometry_config: dict[str, Any] | None = None,
    ):
        if not isinstance(scene_name, str) or not scene_name.strip():
            raise ValueError("scene_name must be a non-empty string")
        if geometry_config is not None and not isinstance(geometry_config, dict):
            raise TypeError("geometry_config must be a mapping")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.scene_name = scene_name
        self.geometry_config = (
            load_geometry_config() if geometry_config is None else geometry_config
        )
        self.registry_path = self.root / "object_registry.json"
        if self.registry_path.exists():
            self.registry = json.loads(
                self.registry_path.read_text(encoding="utf-8")
            )
            self._validate_registry()
        else:
            self.registry = {
                "schema_version": 1,
                "scene_name": scene_name,
                "current_stage": -1,
                "instance_index": {},
                "objects": {},
            }

    def _validate_registry(self) -> None:
        if not isinstance(self.registry, dict):
            raise ValueError("observed geometry registry must be an object")
        if self.registry.get("schema_version") != 1:
            raise ValueError("unsupported observed geometry registry schema")
        if self.registry.get("scene_name") != self.scene_name:
            raise ValueError(
                "observed geometry registry belongs to a different scene"
            )
        if (
            isinstance(self.registry.get("current_stage"), bool)
            or not isinstance(self.registry.get("current_stage"), int)
            or self.registry["current_stage"] < -1
            or not isinstance(self.registry.get("instance_index"), dict)
            or not isinstance(self.registry.get("objects"), dict)
        ):
            raise ValueError("observed geometry registry is malformed")
        objects = self.registry["objects"]
        index = self.registry["instance_index"]
        if any(
            not isinstance(object_id, str) or not isinstance(record, dict)
            for object_id, record in objects.items()
        ) or any(
            not isinstance(token, str)
            or not isinstance(object_id, str)
            or object_id not in objects
            for token, object_id in index.items()
        ):
            raise ValueError("observed geometry registry references are malformed")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _association_token(self, source_id: str, mode: str) -> str:
        payload = f"{self.scene_name}:{mode}:{source_id}".encode()
        return hashlib.sha256(payload).hexdigest()[:24]

    def _new_object_id(self) -> str:
        index = len(self.registry["objects"]) + 1
        while f"object_{index:04d}" in self.registry["objects"]:
            index += 1
        return f"object_{index:04d}"

    def update(self, run: PointCloudRun, *, stage_label: str) -> dict[str, str]:
        """Measure accepted stage evidence and persist generic object IDs."""
        if run.inspection is None:
            raise ValueError("observed state requires a region inspection run")
        safe_label = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(stage_label)
        ).strip("._")
        if not safe_label:
            raise ValueError(
                "stage_label must contain a safe filename component"
            )
        inspection = run.inspection
        stage = int(self.registry["current_stage"]) + 1
        stage_dir = self.root / "stages" / f"{stage:03d}_{safe_label}"
        stage_dir.mkdir(parents=True, exist_ok=False)
        mode = str(inspection.metadata.get("perception_mode", "unknown"))
        associations: dict[str, str] = {}

        for source_id, evidence in inspection.evidence_clouds.items():
            token = self._association_token(source_id, mode)
            object_id = self.registry["instance_index"].get(token)
            if object_id is None:
                object_id = self._new_object_id()
                self.registry["instance_index"][token] = object_id
            associations[source_id] = object_id

            evidence_dir = stage_dir / object_id
            evidence_dir.mkdir(parents=True, exist_ok=True)
            cloud_path = evidence_dir / "fused.ply"
            write_ply(
                cloud_path,
                evidence.measurement_points,
                evidence.measurement_colors,
            )
            measured = extract_object_properties(
                evidence.with_provenance(
                    source_stage=stage,
                    measurement_cloud_path=str(
                        cloud_path.relative_to(self.root)
                    ),
                ),
                config=self.geometry_config,
            )
            previous = self.registry["objects"].get(object_id, {})
            label = None
            if mode != "mujoco_oracle" and source_id in run.clouds:
                candidate = run.clouds[source_id].object_kind
                label = None if candidate == "unknown" else candidate
            record = {
                "object_id": object_id,
                "semantic_label": label or previous.get("semantic_label"),
                "perception_mode": mode,
                "source_region": evidence.source_region,
                "first_seen_stage": previous.get("first_seen_stage", stage),
                "last_seen_stage": stage,
                "observation_count": int(previous.get("observation_count", 0)) + 1,
                **measured,
            }
            self.registry["objects"][object_id] = record
            self._write_json(evidence_dir / "properties.json", record)

        self.registry["current_stage"] = stage
        self._write_json(self.registry_path, self.registry)
        return associations

    def relation(
        self, relation: str, source_id: str, target_id: str
    ) -> dict[str, Any]:
        """Evaluate one relation from cached point-cloud measurements."""
        objects = self.registry["objects"]
        if source_id not in objects or target_id not in objects:
            raise KeyError("relation references an unobserved object")
        return pairwise_relation_evaluation(
            relation,
            objects[source_id],
            objects[target_id],
            self.geometry_config,
        )

    def select_first_compatible(
        self,
        ranked_candidate_ids: Sequence[str],
        *,
        target_id: str,
        required_relations: Sequence[str],
    ) -> CompatibilitySelection:
        """Return the first ranked candidate proven geometrically feasible."""
        if not ranked_candidate_ids:
            return CompatibilitySelection("INCOMPLETE", None, ())
        if len(set(ranked_candidate_ids)) != len(ranked_candidate_ids):
            raise ValueError("ranked candidate IDs must be unique")
        if not required_relations:
            raise ValueError("at least one geometric relation is required")
        evaluations: list[dict[str, Any]] = []
        saw_unknown = False
        for candidate_id in ranked_candidate_ids:
            checks = [
                self.relation(relation, candidate_id, target_id)
                for relation in required_relations
            ]
            statuses = {check["status"] for check in checks}
            evaluation = {
                "candidate_id": candidate_id,
                "target_id": target_id,
                "checks": checks,
            }
            evaluations.append(evaluation)
            if statuses == {"TRUE"}:
                return CompatibilitySelection(
                    "COMPLETE", candidate_id, tuple(evaluations)
                )
            if "UNKNOWN" in statuses and "FALSE" not in statuses:
                saw_unknown = True
        return CompatibilitySelection(
            "INDETERMINATE" if saw_unknown else "INCOMPLETE",
            None,
            tuple(evaluations),
        )
