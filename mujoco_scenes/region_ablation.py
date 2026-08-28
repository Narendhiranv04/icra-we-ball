"""Persistent L2 region registry and same-evidence functional grounding."""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from mujoco_scenes.region_grounding import (
    L2RegionEvidenceCapture,
    evaluate_fits_on,
    evaluate_near_seating_area,
    extract_payload_properties,
    extract_region_properties,
    load_region_task,
    run_region_semantics,
    semantic_region_role_status,
)
from mujoco_scenes.semantic_grounding import (
    NullSemanticDetector,
    SemanticDetector,
    create_semantic_detector,
    load_semantic_config,
)


REGION_SCHEMA_VERSION = 1
GROUNDING_MODES = ("geometry_only", "semantic_only", "joint")
DEFAULT_RIG_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "l2_region_inspection_rigs.yaml"
)
DEFAULT_TASK_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "l2_region_ablation1_task.yaml"
)
DEFAULT_EVALUATION_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "l2_region_ablation1_evaluation.yaml"
)
DEFAULT_SEMANTIC_VOCABULARY = (
    Path(__file__).resolve().parent
    / "configs"
    / "l2_region_semantic_vocabulary.yaml"
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(record: dict[str, Any], key: str) -> Any:
    return record.get(key, {}).get("value")


def _tri_and(*statuses: str) -> str:
    if "FALSE" in statuses:
        return "FALSE"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    return "TRUE"


class PersistentRegionRegistry:
    """Generic region identities maintained from observed support geometry."""

    def __init__(self):
        self.records: dict[str, dict[str, Any]] = {}

    def _match(
        self,
        properties: dict[str, Any],
        semantic_context: dict[str, Any],
    ) -> str | None:
        centroid = _value(properties, "centroid_world_m")
        length = _value(properties, "support_length_m")
        width = _value(properties, "support_width_m")
        if centroid is None:
            return None
        candidate = np.asarray(centroid, float)
        matches = []
        for region_id, record in self.records.items():
            existing = _value(
                record.get("geometric_properties", {}),
                "centroid_world_m",
            )
            if existing is None:
                continue
            distance = float(
                np.linalg.norm(candidate[:2] - np.asarray(existing)[:2])
            )
            old_length = _value(
                record.get("geometric_properties", {}),
                "support_length_m",
            )
            old_width = _value(
                record.get("geometric_properties", {}),
                "support_width_m",
            )
            relative_size_error = 0.0
            overlap_ratio = 0.0
            if (
                length is not None
                and width is not None
                and old_length is not None
                and old_width is not None
            ):
                relative_size_error = max(
                    abs(float(length) - float(old_length))
                    / max(float(length), float(old_length), 1e-6),
                    abs(float(width) - float(old_width))
                    / max(float(width), float(old_width), 1e-6),
                )
                delta = np.abs(
                    candidate[:2] - np.asarray(existing, float)[:2]
                )
                intersection = np.maximum(
                    0.0,
                    np.minimum(
                        [float(length), float(width)],
                        [float(old_length), float(old_width)],
                    )
                    - delta,
                )
                intersection_area = float(np.prod(intersection))
                smaller_area = min(
                    float(length) * float(width),
                    float(old_length) * float(old_width),
                )
                overlap_ratio = intersection_area / max(smaller_area, 1e-6)
            incoming_label = semantic_context.get(
                "parent_furniture", {}
            ).get("canonical_label")
            existing_label = record.get("semantic_context", {}).get(
                "parent_furniture", {}
            ).get("canonical_label")
            semantics_compatible = (
                incoming_label is None
                or existing_label is None
                or incoming_label == existing_label
            )
            if (
                distance <= 0.12
                and relative_size_error <= 0.35
                and overlap_ratio >= 0.45
                and semantics_compatible
            ):
                matches.append(
                    (distance, relative_size_error, -overlap_ratio, region_id)
                )
        return min(matches)[3] if matches else None

    def update(
        self,
        *,
        stage: int,
        inspection_label: str,
        properties: dict[str, Any],
        semantic_context: dict[str, Any],
        functional_evaluation: dict[str, Any],
        evidence_path: str,
        contributing_cameras: Iterable[str],
        point_count: int,
    ) -> tuple[str, bool]:
        region_id = self._match(properties, semantic_context)
        discovered = region_id is None
        if discovered:
            region_id = f"region_{len(self.records) + 1:04d}"
            first_seen = stage
            observations = 0
        else:
            first_seen = self.records[region_id]["identity"][
                "discovery_stage"
            ]
            observations = self.records[region_id]["identity"][
                "observation_count"
            ]
        self.records[region_id] = {
            "identity": {
                "region_id": region_id,
                "entity_type": "region",
                "discovery_stage": first_seen,
                "latest_validated_evidence_stage": stage,
                "observation_count": observations + 1,
            },
            "semantic_context": deepcopy(semantic_context),
            "geometric_properties": deepcopy(properties),
            "functional_evaluations": deepcopy(functional_evaluation),
            "provenance": {
                "stage": stage,
                "configured_inspection_label": inspection_label,
                "measurement_cloud_path": evidence_path,
                "contributing_camera_ids": list(contributing_cameras),
                "point_count": int(point_count),
                "extractor_version": "region_support_geometry_v1",
                "measurement_purpose": "REGION_MEASUREMENT_EVIDENCE",
            },
        }
        return region_id, discovered

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGION_SCHEMA_VERSION,
            "entity_type": "region_registry",
            "regions": deepcopy(self.records),
        }


class RegionAblationRun:
    """One perception run followed by offline same-evidence ablations."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        scene_name: str,
        task_config: str | Path = DEFAULT_TASK_CONFIG,
        evaluation_config: str | Path = DEFAULT_EVALUATION_CONFIG,
        rig_config: str | Path = DEFAULT_RIG_CONFIG,
        semantic_detector: SemanticDetector | None = None,
        semantic_config: dict[str, Any] | None = None,
        width: int = 1280,
        height: int = 960,
    ):
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.exists():
            raise RuntimeError(f"Run directory already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self.stages_dir = self.run_dir / "stages"
        self.stages_dir.mkdir()
        self.scene_name = scene_name
        self.task = load_region_task(task_config)
        with Path(evaluation_config).open(encoding="utf-8") as source:
            self.evaluation = yaml.safe_load(source)
        if not isinstance(self.evaluation, dict) or not isinstance(
            self.evaluation.get("scenes"), dict
        ):
            raise ValueError("Region evaluation config requires a scenes mapping")
        self.rig_config = Path(rig_config)
        self.semantic_detector = (
            NullSemanticDetector() if semantic_detector is None else semantic_detector
        )
        self.semantic_config = (
            load_semantic_config(vocabulary_path=DEFAULT_SEMANTIC_VOCABULARY)
            if semantic_config is None
            else semantic_config
        )
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("Region run dimensions must be positive integers")
        self.width = width
        self.height = height
        self.registry = PersistentRegionRegistry()
        self.payload_record: dict[str, Any] | None = None
        self.seating_semantic_record: dict[str, Any] | None = None
        self.events_path = self.run_dir / "events.jsonl"
        self.stage_records: list[dict[str, Any]] = []
        self.production_status = "INCOMPLETE"
        self.selected_region_id: str | None = None
        self._write_run_config()

    def _write_run_config(self) -> None:
        _atomic_json(
            self.run_dir / "run_config.json",
            {
                "schema_version": REGION_SCHEMA_VERSION,
                "scene_name": self.scene_name,
                "task_id": self.task["task_id"],
                "natural_language_goal": self.task[
                    "natural_language_goal"
                ],
                "function": self.task["function"],
                "production_mode": "joint",
                "uses_robot": False,
                "uses_foundation_model": False,
                "uses_tamp": False,
                "uses_placement_execution": False,
                "capture_resolution": [self.width, self.height],
                "same_evidence_ablations": list(GROUNDING_MODES),
                "created_at": datetime.now().astimezone().isoformat(),
                "detector": {
                    "name": getattr(
                        self.semantic_detector,
                        "name",
                        self.semantic_detector.__class__.__name__,
                    ),
                    "checkpoint": getattr(
                        self.semantic_detector, "checkpoint", None
                    ),
                    "version": getattr(
                        self.semantic_detector, "version", None
                    ),
                    "device": getattr(
                        self.semantic_detector, "device", None
                    ),
                },
            },
        )
        _atomic_json(self.run_dir / "task_requirements.json", self.task)

    def event(self, event: str, **payload: Any) -> None:
        record = {
            "event": event,
            "stage": None,
            "configured_inspection_label": None,
            "region_id": None,
            "semantic_status": None,
            "geometric_status": None,
            "joint_status": None,
            "fit_margin_m": None,
            "near_margin_m": None,
            "evidence_path": None,
            "rejection_reason": None,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(record, sort_keys=True) + "\n")

    def ranking(self) -> list[str]:
        scene_evaluation = self.evaluation["scenes"].get(
            self.scene_name, {}
        )
        if scene_evaluation.get("ranking"):
            return list(scene_evaluation["ranking"])
        candidates = self.task["candidate_ranking"]["candidates"]
        return [
            item["inspection_label"]
            for item in sorted(candidates, key=lambda item: item["rank"])
        ]

    def _seating_semantic_status(
        self, semantic: dict[str, Any]
    ) -> dict[str, Any]:
        if semantic.get("status") != "SUPPORTED":
            return {
                "status": "UNKNOWN",
                "value": None,
                "reason": "INSUFFICIENT_SEATING_SEMANTICS",
            }
        label = semantic.get("canonical_label")
        accepted = set(
            self.task["semantic_requirements"]["seating_categories"]
        )
        return {
            "status": "TRUE" if label in accepted else "FALSE",
            "value": label in accepted,
            "canonical_label": label,
        }

    def _evaluate(
        self,
        *,
        properties: dict[str, Any],
        semantic_records: dict[str, Any],
        sofa_points: np.ndarray,
    ) -> dict[str, Any]:
        region_semantic = semantic_region_role_status(
            semantic_records.get("region_parent", {}),
            self.task,
        )
        seating_semantic = self._seating_semantic_status(
            semantic_records.get("seating", {})
        )
        fits = evaluate_fits_on(
            self.payload_record["geometric_properties"]
            if self.payload_record
            else {},
            properties,
            task_config=self.task,
        )
        near = evaluate_near_seating_area(
            properties, sofa_points, task_config=self.task
        )
        planar = properties.get("PLANAR_SUPPORT", {})
        planar_status = (
            "UNKNOWN"
            if planar.get("value") is None
            else "TRUE"
            if planar.get("value")
            else "FALSE"
        )
        geometry_status = _tri_and(
            planar_status, fits["status"], near["status"]
        )
        semantic_status = _tri_and(
            region_semantic["status"], seating_semantic["status"]
        )
        joint_status = _tri_and(geometry_status, semantic_status)
        rejection = None
        if region_semantic["status"] == "FALSE":
            rejection = "REGION_REJECTED_SEMANTIC"
        elif planar_status == "FALSE":
            rejection = "REGION_REJECTED_PLANARITY"
        elif fits["status"] == "FALSE":
            rejection = "REGION_REJECTED_FIT"
        elif near["status"] == "FALSE":
            rejection = "REGION_REJECTED_CONTEXT"
        elif joint_status == "UNKNOWN":
            rejection = "INSUFFICIENT_VALIDATED_EVIDENCE"
        return {
            "semantic_role": region_semantic,
            "seating_semantics": seating_semantic,
            "PLANAR_SUPPORT": {
                **planar,
                "tri_state": planar_status,
            },
            "FITS_ON": fits,
            "NEAR_SEATING_AREA": near,
            "geometry_only_status": geometry_status,
            "semantic_only_status": semantic_status,
            "joint_status": joint_status,
            "rejection_reason": rejection,
        }

    @staticmethod
    def _semantic_quality_key(record: dict[str, Any]) -> tuple[int, int, float]:
        return (
            int(record.get("status") == "SUPPORTED"),
            int(record.get("supporting_view_count", 0)),
            float(record.get("confidence") or 0.0),
        )

    def _persist_state(self) -> None:
        _atomic_json(
            self.run_dir / "region_registry.json", self.registry.to_dict()
        )
        if self.payload_record is not None:
            _atomic_json(
                self.run_dir / "payload_registry.json",
                {
                    "schema_version": 1,
                    "objects": {
                        self.payload_record["identity"]["object_id"]: (
                            self.payload_record
                        )
                    },
                },
            )
        evaluations = {
            region_id: record["functional_evaluations"]
            for region_id, record in self.registry.records.items()
        }
        _atomic_json(
            self.run_dir / "region_function_evaluations.json", evaluations
        )
        matrix = self.compatibility_matrix()
        _atomic_json(
            self.run_dir / "region_compatibility_matrix.json", matrix
        )
        self._write_matrix_csv(matrix)
        _atomic_json(
            self.run_dir / "observed_graph.json", self.build_graph()
        )

    def _write_matrix_csv(self, matrix: dict[str, Any]) -> None:
        path = self.run_dir / "region_compatibility_matrix.csv"
        rows = matrix["rows"]
        fieldnames = sorted({key for row in rows for key in row})
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(
                target, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)

    def observe(
        self,
        scene,
        inspection_label: str,
        *,
        stage: int,
    ) -> dict[str, Any]:
        stage_name = (
            f"{stage:03d}_initial_{inspection_label}"
            if stage == 0
            else f"{stage:03d}_after_{inspection_label}"
        )
        stage_dir = self.stages_dir / stage_name
        capture = L2RegionEvidenceCapture(
            scene,
            rig_config=self.rig_config,
            width=self.width,
            height=self.height,
        ).capture(inspection_label, stage=stage, stage_dir=stage_dir)
        properties = extract_region_properties(
            capture.region_evidence, task_config=self.task
        )
        if (
            self.payload_record is None
            and capture.payload_evidence is not None
            and capture.payload_evidence.measurement_quality.get(
                "quality_is_valid", False
            )
        ):
            payload_properties = extract_payload_properties(
                capture.payload_evidence
            )
            self.payload_record = {
                "identity": {
                    "object_id": "object_0001",
                    "entity_type": "payload",
                    "discovery_stage": stage,
                },
                "semantic_context": {},
                "geometric_properties": payload_properties,
                "provenance": {
                    "measurement_cloud_path": (
                        capture.payload_evidence.measurement_cloud_path
                    ),
                    "measurement_purpose": "PAYLOAD_MEASUREMENT_EVIDENCE",
                },
            }
        semantic_records = run_region_semantics(
            capture,
            detector=self.semantic_detector,
            semantic_config=self.semantic_config,
            task_config=self.task,
            stage_dir=stage_dir,
        )
        current_seating = semantic_records.get("seating", {})
        accepted_seating_labels = set(
            self.task["semantic_requirements"]["seating_categories"]
        )
        current_is_accepted = (
            current_seating.get("status") == "SUPPORTED"
            and current_seating.get("canonical_label")
            in accepted_seating_labels
        )
        cached_is_accepted = bool(
            self.seating_semantic_record
            and self.seating_semantic_record.get("status") == "SUPPORTED"
            and self.seating_semantic_record.get("canonical_label")
            in accepted_seating_labels
        )
        if (
            self.seating_semantic_record is None
            or (current_is_accepted and not cached_is_accepted)
            or (
                current_is_accepted
                and cached_is_accepted
                and self._semantic_quality_key(current_seating)
                > self._semantic_quality_key(self.seating_semantic_record)
            )
        ):
            self.seating_semantic_record = {
                **deepcopy(current_seating),
                "source_stage": stage,
                "source_inspection_label": inspection_label,
                "semantic_record_path": (
                    f"stages/{stage_name}/semantics/fused_semantics.json"
                ),
            }
        effective_semantic_records = {
            **semantic_records,
            "seating": deepcopy(self.seating_semantic_record or {}),
        }
        if self.payload_record is not None:
            payload_semantic = semantic_records.get("payload", {})
            current = self.payload_record.get("semantic_context", {})
            if payload_semantic.get("status") == "SUPPORTED" or not current:
                self.payload_record["semantic_context"] = payload_semantic
        functional = self._evaluate(
            properties=properties,
            semantic_records=effective_semantic_records,
            sofa_points=capture.sofa_points,
        )
        semantic_context = {
            "parent_furniture": semantic_records.get("region_parent", {}),
            "seating_context": effective_semantic_records.get("seating", {}),
            "current_stage_seating_context": semantic_records.get(
                "seating", {}
            ),
        }
        region_id, discovered = self.registry.update(
            stage=stage,
            inspection_label=inspection_label,
            properties=properties,
            semantic_context=semantic_context,
            functional_evaluation=functional,
            evidence_path=capture.region_evidence.measurement_cloud_path,
            contributing_cameras=(
                capture.region_evidence.contributing_camera_ids
            ),
            point_count=len(capture.region_evidence.measurement_points),
        )
        event_payload = {
            "stage": stage,
            "configured_inspection_label": inspection_label,
            "region_id": region_id,
            "semantic_status": functional["semantic_only_status"],
            "geometric_status": functional["geometry_only_status"],
            "joint_status": functional["joint_status"],
            "fit_margin_m": functional["FITS_ON"].get(
                "signed_fit_margin_m"
            ),
            "near_margin_m": functional["NEAR_SEATING_AREA"].get(
                "signed_margin_m"
            ),
            "evidence_path": capture.region_evidence.measurement_cloud_path,
            "rejection_reason": functional["rejection_reason"],
        }
        if discovered:
            self.event("REGION_DISCOVERED", **event_payload)
        quality_valid = capture.region_evidence.measurement_quality.get(
            "quality_is_valid", False
        )
        self.event(
            (
                "REGION_MEASUREMENT_ACCEPTED"
                if quality_valid
                else "REGION_MEASUREMENT_REJECTED"
            ),
            **event_payload,
        )
        self.event("REGION_SEMANTICS_FUSED", **event_payload)
        self.event("REGION_FUNCTION_EVALUATED", **event_payload)
        if functional["joint_status"] != "TRUE":
            self.event(
                functional["rejection_reason"] or "NO_VERIFIED_REGION",
                **event_payload,
            )
        else:
            self.event("REGION_COMPATIBILITY_ACCEPTED", **event_payload)
        stage_record = {
            "stage": stage,
            "stage_name": stage_name,
            "configured_inspection_label": inspection_label,
            "region_id": region_id,
            "geometric_properties": properties,
            "semantic_context": semantic_context,
            "functional_evaluations": functional,
            "evidence_path": capture.region_evidence.measurement_cloud_path,
            "semantic_overview_path": (
                f"stages/{stage_name}/semantic_overview.png"
            ),
            "timings_seconds": capture.timings_seconds,
        }
        self.stage_records.append(stage_record)
        _atomic_json(stage_dir / "region_properties.json", properties)
        _atomic_json(stage_dir / "region_evaluation.json", stage_record)
        self._persist_state()
        if functional["joint_status"] == "TRUE":
            self.production_status = "COMPLETE"
            self.selected_region_id = region_id
            # Persist once more so the graph contains ASSIGNED_AS_DESTINATION
            # and the root snapshots reflect the authoritative selection.
            self._persist_state()
            self._write_verified_handoff(stage_record)
            self.event(
                "VERIFIED_REGION_ASSIGNMENT_COMPLETE", **event_payload
            )
        return stage_record

    def _write_verified_handoff(self, selected: dict[str, Any]) -> None:
        region_record = self.registry.records[selected["region_id"]]
        handoff = {
            "task_id": self.task["task_id"],
            "natural_language_goal": self.task["natural_language_goal"],
            "function_id": self.task["function"]["function_id"],
            "payload_object_id": self.payload_record["identity"]["object_id"],
            "selected_region_id": selected["region_id"],
            "selected_region_parent_semantics": region_record[
                "semantic_context"
            ]["parent_furniture"],
            "PLANAR_SUPPORT": region_record["functional_evaluations"][
                "PLANAR_SUPPORT"
            ],
            "FITS_ON": region_record["functional_evaluations"]["FITS_ON"],
            "NEAR_SEATING_AREA": region_record[
                "functional_evaluations"
            ]["NEAR_SEATING_AREA"],
            "candidate_ranking": self.ranking(),
            "rejected_earlier_candidates": [
                {
                    "region_id": record["region_id"],
                    "inspection_label": record[
                        "configured_inspection_label"
                    ],
                    "rejection_reason": record[
                        "functional_evaluations"
                    ]["rejection_reason"],
                }
                for record in self.stage_records
                if record["region_id"] != selected["region_id"]
            ],
            "completion_stage": selected["stage"],
            "completion_inspection_label": selected[
                "configured_inspection_label"
            ],
            "evidence_paths": {
                "region": selected["evidence_path"],
                "payload": self.payload_record["provenance"][
                    "measurement_cloud_path"
                ],
            },
            "verified": True,
            "ready_for_tamp": True,
            "placement_executed": False,
            "tamp_executed": False,
        }
        _atomic_json(self.run_dir / "verified_region_handoff.json", handoff)

    def run(self, scene, *, full_order: bool = False) -> "RegionAblationRun":
        for stage, inspection_label in enumerate(self.ranking()):
            record = self.observe(scene, inspection_label, stage=stage)
            print(
                f"[REGION] stage {stage:03d} {inspection_label}: "
                f"joint={record['functional_evaluations']['joint_status']}"
            )
            if self.production_status == "COMPLETE" and not full_order:
                break
        if self.production_status != "COMPLETE":
            self.production_status = "EXHAUSTED"
            self.event(
                "REGION_SEARCH_EXHAUSTED",
                stage=len(self.stage_records) - 1,
                configured_inspection_label=(
                    self.stage_records[-1]["configured_inspection_label"]
                    if self.stage_records
                    else None
                ),
                region_id=None,
                semantic_status=None,
                geometric_status=None,
                measured_margins={},
                evidence_paths=[],
                rejection_reason="NO_JOINTLY_VALID_REGION",
            )
            self.event(
                "NO_VERIFIED_REGION",
                stage=len(self.stage_records) - 1,
                rejection_reason="NO_JOINTLY_VALID_REGION",
            )
        self.evaluate_same_evidence()
        self.validate_expected_outcomes()
        return self

    def compatibility_matrix(self) -> dict[str, Any]:
        rows = []
        rank_by_label = {
            label: index + 1 for index, label in enumerate(self.ranking())
        }
        for stage_record in self.stage_records:
            properties = stage_record["geometric_properties"]
            semantic = stage_record["semantic_context"]["parent_furniture"]
            functional = stage_record["functional_evaluations"]
            fits = functional["FITS_ON"]
            near = functional["NEAR_SEATING_AREA"]
            rows.append(
                {
                    "region_id": stage_record["region_id"],
                    "discovery_stage": stage_record["stage"],
                    "candidate_rank": rank_by_label.get(
                        stage_record["configured_inspection_label"]
                    ),
                    "parent_semantic_label": semantic.get(
                        "canonical_label"
                    ),
                    "semantic_confidence": semantic.get("confidence"),
                    "semantic_supporting_views": semantic.get(
                        "supporting_view_count"
                    ),
                    "semantic_role_status": functional["semantic_role"][
                        "status"
                    ],
                    "support_length_m": _value(
                        properties, "support_length_m"
                    ),
                    "support_width_m": _value(
                        properties, "support_width_m"
                    ),
                    "support_area_m2": _value(
                        properties, "support_area_m2"
                    ),
                    "support_normal_world": _value(
                        properties, "support_normal_world"
                    ),
                    "planarity_score": _value(
                        properties, "planarity_score"
                    ),
                    "payload_length_m": fits.get("payload_length_m"),
                    "payload_width_m": fits.get("payload_width_m"),
                    "tested_orientations": fits.get(
                        "tested_orientations"
                    ),
                    "selected_orientation_degrees": fits.get(
                        "selected_orientation_degrees"
                    ),
                    "fit_margin_m": fits.get("signed_fit_margin_m"),
                    "PLANAR_SUPPORT": functional["PLANAR_SUPPORT"][
                        "tri_state"
                    ],
                    "FITS_ON": fits["status"],
                    "sofa_distance_m": near.get("measured_distance_m"),
                    "near_sofa_margin_m": near.get("signed_margin_m"),
                    "NEAR_SEATING_AREA": near["status"],
                    "geometry_only_status": functional[
                        "geometry_only_status"
                    ],
                    "semantic_only_status": functional[
                        "semantic_only_status"
                    ],
                    "joint_status": functional["joint_status"],
                    "rejection_reason": functional["rejection_reason"],
                    "region_evidence_path": stage_record["evidence_path"],
                    "semantic_evidence_path": stage_record[
                        "semantic_overview_path"
                    ],
                }
            )
        return {
            "schema_version": 1,
            "task_id": self.task["task_id"],
            "rows": rows,
        }

    def evaluate_same_evidence(self) -> dict[str, Any]:
        selections = {}
        for mode in GROUNDING_MODES:
            status_key = f"{mode}_status"
            selected = next(
                (
                    row
                    for row in self.compatibility_matrix()["rows"]
                    if row[status_key] == "TRUE"
                ),
                None,
            )
            selections[mode] = {
                "status": "COMPLETE" if selected else "EXHAUSTED",
                "selected_region_id": (
                    selected["region_id"] if selected else None
                ),
                "selected_inspection_label": (
                    next(
                        record["configured_inspection_label"]
                        for record in self.stage_records
                        if selected
                        and record["region_id"] == selected["region_id"]
                    )
                    if selected
                    else None
                ),
                "completion_stage": (
                    selected["discovery_stage"] if selected else None
                ),
                "production_authoritative": mode == "joint",
            }
        evidence_files = sorted(
            {
                path
                for pattern in (
                    "stages/*/region_evidence/fused.ply",
                    "stages/*/rejected_region_evidence/fused.ply",
                    "stages/*/payload_evidence/fused.ply",
                    "stages/*/seating_context/observed_points.ply",
                    "stages/*/cameras/*/rgb.png",
                    "stages/*/cameras/*/depth.png",
                    "stages/*/cameras/*/segmentation.png",
                    "stages/*/cameras/*/evidence_masks.png",
                    "stages/*/semantics/detections.json",
                    "stages/*/semantics/associations.json",
                    "stages/*/semantics/fused_semantics.json",
                )
                for path in self.run_dir.glob(pattern)
            }
        )
        evidence_manifest = [
            {
                "path": path.relative_to(self.run_dir).as_posix(),
                "sha256": _file_hash(path),
            }
            for path in evidence_files
        ]
        summary = {
            "schema_version": 1,
            "task_id": self.task["task_id"],
            "scene_name": self.scene_name,
            "same_evidence_manifest": evidence_manifest,
            "rerendered_for_diagnostics": False,
            "modes": selections,
            "production_status": self.production_status,
        }
        _atomic_json(
            self.run_dir / "offline_region_ablation_evaluation.json",
            summary,
        )
        _atomic_json(
            self.run_dir / "region_ablation_summary.json", summary
        )
        return summary

    def validate_expected_outcomes(self) -> dict[str, Any]:
        """Compare observed ablation outcomes with evaluation-only targets."""
        summary_path = (
            self.run_dir / "offline_region_ablation_evaluation.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = self.evaluation["scenes"].get(
            self.scene_name, {}
        ).get("expected", {})
        checks = []
        for mode, expectation in expected.items():
            observed = summary["modes"][mode]
            expected_label = expectation.get("inspection_label")
            expected_status = expectation.get(
                "terminal_status",
                "COMPLETE" if expected_label is not None else "EXHAUSTED",
            )
            checks.append(
                {
                    "mode": mode,
                    "expected_status": expected_status,
                    "observed_status": observed["status"],
                    "expected_inspection_label": expected_label,
                    "observed_inspection_label": observed[
                        "selected_inspection_label"
                    ],
                    "expected_completion_stage": expectation.get(
                        "completion_stage"
                    ),
                    "observed_completion_stage": observed[
                        "completion_stage"
                    ],
                    "passed": (
                        observed["status"] == expected_status
                        and observed["selected_inspection_label"]
                        == expected_label
                        and (
                            expectation.get("completion_stage") is None
                            or observed["completion_stage"]
                            == expectation["completion_stage"]
                        )
                    ),
                }
            )
        validation = {
            "schema_version": 1,
            "scene_name": self.scene_name,
            "checks": checks,
            "passed": bool(checks) and all(
                check["passed"] for check in checks
            ),
        }
        _atomic_json(
            self.run_dir / "region_ablation_validation.json", validation
        )
        return validation

    def build_graph(self) -> dict[str, Any]:
        nodes = [
            {
                "id": "function:PLACE_REFRESHMENT_TRAY",
                "type": "FUNCTION",
                "attributes": {
                    "natural_language_goal": self.task[
                        "natural_language_goal"
                    ]
                },
            },
            {
                "id": "payload:object_0001",
                "type": "PAYLOAD",
                "attributes": (
                    deepcopy(self.payload_record)
                    if self.payload_record
                    else {}
                ),
            },
        ]
        edges = []
        for region_id, record in self.registry.records.items():
            functional = record["functional_evaluations"]
            parent_semantic = record["semantic_context"][
                "parent_furniture"
            ].get("canonical_label")
            context_node = (
                f"furniture_context:{parent_semantic}"
                if parent_semantic
                else f"furniture_context:{region_id}:unknown"
            )
            if not any(node["id"] == context_node for node in nodes):
                nodes.append(
                    {
                        "id": context_node,
                        "type": "FURNITURE_CONTEXT",
                        "attributes": {
                            "canonical_label": parent_semantic,
                            "semantic_source": "RGB_DETECTOR",
                        },
                    }
                )
            nodes.append(
                {
                    "id": f"region:{region_id}",
                    "type": "REGION",
                    "attributes": deepcopy(record),
                }
            )
            edges.extend(
                [
                    {
                        "source": f"region:{region_id}",
                        "target": context_node,
                        "type": "REGION_OF",
                        "status": record["semantic_context"][
                            "parent_furniture"
                        ].get("status", "UNKNOWN"),
                    },
                    {
                        "source": f"region:{region_id}",
                        "target": "function:PLACE_REFRESHMENT_TRAY",
                        "type": "CANDIDATE_FOR_REGION_FUNCTION",
                        "status": functional["semantic_role"]["status"],
                    },
                    {
                        "source": f"region:{region_id}",
                        "target": "function:PLACE_REFRESHMENT_TRAY",
                        "type": "SATISFIES_REGION_SEMANTICS",
                        "status": functional["semantic_role"]["status"],
                    },
                    {
                        "source": f"region:{region_id}",
                        "target": "function:PLACE_REFRESHMENT_TRAY",
                        "type": "SATISFIES_REGION_GEOMETRY",
                        "status": functional["geometry_only_status"],
                    },
                    {
                        "source": "payload:object_0001",
                        "target": f"region:{region_id}",
                        "type": "FITS_ON",
                        "status": functional["FITS_ON"]["status"],
                    },
                    {
                        "source": f"region:{region_id}",
                        "target": "context:seating_area",
                        "type": "NEAR",
                        "status": functional["NEAR_SEATING_AREA"][
                            "status"
                        ],
                    },
                ]
            )
            if functional["FITS_ON"]["status"] == "FALSE":
                edges.append(
                    {
                        "source": "payload:object_0001",
                        "target": f"region:{region_id}",
                        "type": "INCOMPATIBLE_WITH_PAYLOAD",
                        "status": "TRUE",
                    }
                )
            if (
                self.selected_region_id == region_id
                and functional["joint_status"] == "TRUE"
            ):
                edges.append(
                    {
                        "source": f"region:{region_id}",
                        "target": "function:PLACE_REFRESHMENT_TRAY",
                        "type": "ASSIGNED_AS_DESTINATION",
                        "status": "TRUE",
                    }
                )
                edges.append(
                    {
                        "source": f"region:{region_id}",
                        "target": "function:PLACE_REFRESHMENT_TRAY",
                        "type": "SATISFIES_FUNCTION",
                        "status": "TRUE",
                    }
                )
        nodes.append(
            {
                "id": "context:seating_area",
                "type": "FURNITURE_CONTEXT",
                "attributes": {"semantic_source": "RGB_DETECTOR"},
            }
        )
        return {"schema_version": 1, "nodes": nodes, "edges": edges}


def create_region_semantic_detector(
    *,
    checkpoint: str,
    confidence_threshold: float = 0.08,
    vocabulary_path: str | Path = DEFAULT_SEMANTIC_VOCABULARY,
) -> tuple[SemanticDetector, dict[str, Any]]:
    config = load_semantic_config(vocabulary_path=vocabulary_path)
    config["detector"]["confidence_threshold"] = confidence_threshold
    detector = create_semantic_detector(
        config,
        backend="yolo_world",
        checkpoint=checkpoint,
        confidence_threshold=confidence_threshold,
    )
    return detector, config
